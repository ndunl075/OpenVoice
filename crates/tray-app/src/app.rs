//! The floating recording pill: a small, borderless, always-on-top window
//! that shows the engine's current state -- the interaction pattern
//! Wispr Flow and its peers popularized for dictation apps (a minimal
//! "bar" near the cursor that appears while recording, runs through a
//! listening -> cleaning up -> final flow, and fades after inserting
//! text). Redrawn here from scratch with original shapes and an original
//! light cream-and-lavender palette, not any product's actual assets
//! (logo, wordmark, exact colors) -- OpenVoice isn't and
//! doesn't claim to be Wispr Flow; it follows the same well-established
//! *pattern* other tools in this category use, same as most apps in any
//! category share conventions (a phone app having a bottom tab bar
//! doesn't make it a copy of the first app that had one).

use std::sync::mpsc;
use std::time::{Duration, Instant};

use eframe::egui;

use crate::icon;

/// How long a terminal state (just inserted text, or a warning) stays
/// visible before the pill hides itself again.
const FLASH_DURATION: Duration = Duration::from_millis(1800);

/// How often to poll the status/menu/tray channels and repaint, even with
/// no user input -- matches the engine's own `daemon::POLL_INTERVAL` so
/// the pill never visibly lags a real state change.
const REPAINT_INTERVAL: Duration = Duration::from_millis(100);

/// While the level bars are animating (mic actively live -- recording or
/// hands-free listening), repaint faster than [`REPAINT_INTERVAL`] so the
/// motion actually reads as motion instead of a slideshow. Not used for
/// idle/terminal states, which don't animate and don't need it.
const ANIMATION_REPAINT_INTERVAL: Duration = Duration::from_millis(50);

/// The pill window's size. Wider and slightly shorter than the original
/// capsule so [`PILL_CORNER_RADIUS`] reads as a rounded rectangle rather
/// than a lozenge.
pub const PILL_SIZE: [f32; 2] = [300.0, 52.0];

/// Corner radius: a rounded rectangle, not a capsule. A true capsule
/// would be half the height (26); this is deliberately well under that.
const PILL_CORNER_RADIUS: u8 = 14;

/// Gap between the pill and the screen's bottom-left corner.
const SCREEN_MARGIN: f32 = 24.0;

#[derive(Clone)]
enum Display {
    Hidden,
    Recording,
    Listening,
    Transcribing,
    Inserted { text: String, shown_at: Instant },
    Warning { text: String, shown_at: Instant },
}

pub struct PillApp {
    status_rx: mpsc::Receiver<daemon::PipelineStatus>,
    control_tx: mpsc::Sender<daemon::ControlEvent>,
    menu_ids: crate::tray::MenuIds,
    display: Display,
    hands_free_on: bool,
    // Kept alive for the app's lifetime: dropping it removes the tray icon.
    // Also repainted per state change (see `apply_status`) so the icon
    // itself reflects idle/recording/etc. without opening the pill.
    tray_icon: tray_icon::TrayIcon,
    // Static config, known at startup and shown read-only in the settings
    // window -- see draw_settings_window. Not expected to change without
    // a restart (remapping hotkeys live is a bigger feature -- see that
    // function's doc comment).
    mic_name: String,
    hotkey_config: hotkey::MultiHotkeyConfig,
    // Whether a cleanup model actually loaded at startup (from
    // PipelineStatus::Ready) -- the settings checkbox is disabled
    // entirely if not, since there's nothing to toggle.
    cleanup_model_loaded: bool,
    // The live runtime toggle -- starts equal to `cleanup_model_loaded`,
    // flips on the settings checkbox, and is what's actually sent as
    // ControlEvent::SetCleanupEnabled.
    cleanup_enabled: bool,
    show_settings: bool,
    /// Bottom-left placement can only be computed once egui can tell us
    /// the monitor size, which isn't known until the first frame -- so
    /// it's done once, then latched.
    positioned: bool,
}

impl PillApp {
    pub fn new(
        status_rx: mpsc::Receiver<daemon::PipelineStatus>,
        control_tx: mpsc::Sender<daemon::ControlEvent>,
        tray_icon: tray_icon::TrayIcon,
        menu_ids: crate::tray::MenuIds,
        mic_name: String,
        hotkey_config: hotkey::MultiHotkeyConfig,
    ) -> Self {
        Self {
            status_rx,
            control_tx,
            menu_ids,
            display: Display::Hidden,
            hands_free_on: false,
            tray_icon,
            mic_name,
            hotkey_config,
            cleanup_model_loaded: false,
            cleanup_enabled: false,
            show_settings: false,
            positioned: false,
        }
    }

    /// Parks the pill in the bottom-left corner of the primary monitor,
    /// once, on the first frame that reports a monitor size. Done here
    /// rather than in `main.rs`'s `ViewportBuilder` because the monitor
    /// dimensions aren't available before the window exists.
    fn position_bottom_left(&mut self, ctx: &egui::Context) {
        if self.positioned {
            return;
        }
        let Some(monitor) = ctx.input(|i| i.viewport().monitor_size) else {
            return; // not known yet; try again next frame
        };
        let x = SCREEN_MARGIN;
        let y = monitor.y - PILL_SIZE[1] - SCREEN_MARGIN;
        ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(egui::pos2(x, y)));
        self.positioned = true;
    }

    fn apply_status(&mut self, status: daemon::PipelineStatus) {
        use daemon::PipelineStatus as S;
        let _ = self.tray_icon.set_icon(Some(crate::tray::icon_for_status(&status)));
        match status {
            S::Ready { cleanup_enabled, .. } => {
                self.cleanup_model_loaded = cleanup_enabled;
                self.cleanup_enabled = cleanup_enabled;
                self.display = Display::Hidden;
            }
            S::Recording => self.display = Display::Recording,
            S::Listening => self.display = Display::Listening,
            S::Transcribing => self.display = Display::Transcribing,
            S::Inserted(text) => {
                self.display = Display::Inserted {
                    text,
                    shown_at: Instant::now(),
                };
            }
            S::HeardNothing => self.display = Display::Hidden,
            S::HandsFreeOn => self.hands_free_on = true,
            S::HandsFreeOff => {
                self.hands_free_on = false;
                self.display = Display::Hidden;
            }
            S::Warning(text) => {
                self.display = Display::Warning {
                    text,
                    shown_at: Instant::now(),
                };
            }
        }
    }

    /// Drains every pending status/menu event since the last frame. Menu
    /// clicks are translated into the same `ControlEvent`s a physical
    /// hotkey would send -- there's no separate "UI-triggered" behavior to
    /// keep in sync with the keyboard path.
    fn drain_events(&mut self) {
        while let Ok(status) = self.status_rx.try_recv() {
            self.apply_status(status);
        }
        while let Ok(event) = tray_icon::menu::MenuEvent::receiver().try_recv() {
            if event.id() == &self.menu_ids.hands_free {
                let _ = self.control_tx.send(daemon::ControlEvent::HandsFreeTogglePressed);
            } else if event.id() == &self.menu_ids.settings {
                self.show_settings = true;
            } else if event.id() == &self.menu_ids.quit {
                self.quit();
            }
        }
    }

    /// The one way this process actually ends: the tray menu's "Quit"
    /// item and the pill's own close button both call this, so there's
    /// no separate "UI close button" behavior to keep in sync with the
    /// menu path.
    fn quit(&self) {
        let _ = self.control_tx.send(daemon::ControlEvent::Quit);
        std::process::exit(0);
    }

    /// Terminal states (inserted/warning) auto-hide after [`FLASH_DURATION`].
    fn expire_flash(&mut self) {
        let expired = match &self.display {
            Display::Inserted { shown_at, .. } | Display::Warning { shown_at, .. } => {
                shown_at.elapsed() >= FLASH_DURATION
            }
            _ => false,
        };
        if expired {
            self.display = if self.hands_free_on {
                Display::Listening
            } else {
                Display::Hidden
            };
        }
    }

    /// The "desktop app... where u can adjust settings" -- a real
    /// decorated, resizable-in-the-taskbar OS window (unlike the pill,
    /// which is deliberately chrome-less), opened from the tray menu's
    /// "Settings…" item. First cut: read-only config display (mic,
    /// hotkeys) plus the one setting that's actually adjustable so far
    /// -- the cleanup pass. Hotkey remapping would need a "press keys to
    /// set" capture flow, conflict checking, and persisting the result
    /// to disk -- a bigger feature than this pass; keys are shown
    /// read-only for now.
    fn draw_settings_window(&mut self, ctx: &egui::Context) {
        let mut close_requested = false;
        let mut quit_requested = false;
        ctx.show_viewport_immediate(
            egui::ViewportId::from_hash_of("openvoice-settings"),
            egui::ViewportBuilder::default()
                .with_title("OpenVoice Settings")
                .with_inner_size([380.0, 260.0])
                .with_resizable(false),
            |settings_ui, _class| {
                settings_ui.heading("OpenVoice");
                settings_ui.add_space(4.0);
                settings_ui.label(format!("Microphone: {}", self.mic_name));
                let (ptt_a, ptt_b) = self.hotkey_config.push_to_talk_keys;
                settings_ui.label(format!("Push-to-talk: hold {ptt_a:?} + {ptt_b:?}"));
                settings_ui.label(format!("Hands-free: tap {:?} to toggle", self.hotkey_config.hands_free_toggle_key));
                settings_ui.separator();
                settings_ui.add_enabled_ui(self.cleanup_model_loaded, |ui| {
                    let mut enabled = self.cleanup_enabled;
                    if ui
                        .checkbox(&mut enabled, "Clean up disfluencies (\"um\", false starts) before inserting")
                        .changed()
                    {
                        self.cleanup_enabled = enabled;
                        let _ = self.control_tx.send(daemon::ControlEvent::SetCleanupEnabled(enabled));
                    }
                });
                if !self.cleanup_model_loaded {
                    settings_ui.label(egui::RichText::new("(no cleanup model loaded -- nothing to toggle)").weak());
                }
                settings_ui.separator();
                if settings_ui.button("Quit OpenVoice").clicked() {
                    quit_requested = true;
                }
                if settings_ui.ctx().input(|i| i.viewport().close_requested()) {
                    close_requested = true;
                }
            },
        );
        if quit_requested {
            self.quit();
        }
        if close_requested {
            self.show_settings = false;
        }
    }
}

impl eframe::App for PillApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        // Fully transparent clear so the pill's rounded shape (drawn as an
        // egui::Frame below) is all that's visible -- no window chrome.
        [0.0, 0.0, 0.0, 0.0]
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_events();
        self.expire_flash();

        let ctx = ui.ctx().clone();
        self.position_bottom_left(&ctx);
        if self.show_settings {
            self.draw_settings_window(&ctx);
        }
        ctx.send_viewport_cmd(egui::ViewportCommand::Visible(!matches!(self.display, Display::Hidden)));

        // Both "mic is actually live" states read as "Listening…" -- from
        // the user's side, push-to-talk-held and hands-free-armed are the
        // same thing (the mic is capturing right now), so the copy
        // shouldn't imply a distinction they don't experience. The color
        // still tells them apart (near-black while a key is held,
        // lavender for hands-free) and so does the bars/no-bars animation
        // state.
        let (color, label, animate) = match &self.display {
            Display::Hidden => return,
            Display::Recording => (rgba_to_color32(icon::LISTENING_ACTIVE), "Listening…".to_string(), true),
            Display::Listening => (rgba_to_color32(icon::LAVENDER), "Listening…".to_string(), true),
            Display::Transcribing => (rgba_to_color32(icon::THINKING), "Cleaning up…".to_string(), false),
            Display::Inserted { text, .. } => (rgba_to_color32(icon::SUCCESS), truncate(text, 60), false),
            Display::Warning { text, .. } => (rgba_to_color32(icon::WARNING), truncate(text, 60), false),
        };

        ctx.request_repaint_after(if animate { ANIMATION_REPAINT_INTERVAL } else { REPAINT_INTERVAL });

        let mut quit_clicked = false;
        egui::Frame::new()
            .fill(rgba_to_color32(icon::CREAM_BACKGROUND))
            .corner_radius(egui::CornerRadius::same(PILL_CORNER_RADIUS))
            .inner_margin(egui::Margin::symmetric(16, 12))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    draw_level_bars(ui, color, animate);
                    ui.add_space(6.0);
                    ui.colored_label(rgba_to_color32(icon::DARK_TEXT), label);
                    // Right-aligned within whatever space is left in the
                    // row, so it sits at the pill's edge rather than
                    // right after the label -- a panic button needs to be
                    // somewhere predictable, not wherever the text
                    // happened to end.
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if draw_close_button(ui) {
                            quit_clicked = true;
                        }
                    });
                });
            });

        // Quitting from inside the frame closure above would try to exit
        // mid-borrow of `ui`; do it here instead, once the frame's done
        // drawing.
        if quit_clicked {
            self.quit();
        }
    }
}

/// A minimal "×" on the pill itself. Quitting used to only be reachable
/// from the tray icon's right-click menu -- not much use if the pipeline
/// is misbehaving and you just want it to stop *right now* while you're
/// looking at the pill, not hunting for a tray icon. Calls through
/// [`PillApp::quit`] via the caller's return value, so there's still
/// exactly one quit code path.
fn draw_close_button(ui: &mut egui::Ui) -> bool {
    let text = egui::RichText::new("×").size(16.0).color(rgba_to_color32(icon::DARK_TEXT));
    ui.add(egui::Button::new(text).frame(false))
        .on_hover_text("Quit OpenVoice")
        .clicked()
}

/// A small vertical-bar level meter -- the "equalizer bars" visual
/// convention practically every voice-input UI uses to show a live mic
/// (Siri, Google Assistant, Wispr Flow's own bar all draw on the same
/// genre convention; it's not any one product's invention, same as a red
/// dot for "recording" isn't). Replaces the plain colored dot this pill
/// used to show.
///
/// `animate` drives per-bar heights from wall-clock time with a phase
/// offset per bar, so they move independently rather than in lockstep --
/// when `false` (not actively listening), bars sit flat at their resting
/// height instead, same shape either way so the layout never jumps.
///
/// This is a time-based approximation, not a real audio level meter: the
/// pill doesn't currently see actual mic amplitude, only pipeline state
/// over `PipelineStatus`. Wiring real per-frame RMS through would make
/// this genuinely reactive to the user's voice instead of just "looks
/// alive" -- a reasonable next step if it's worth the plumbing.
fn draw_level_bars(ui: &mut egui::Ui, color: egui::Color32, animate: bool) {
    const BAR_COUNT: usize = 4;
    const BAR_WIDTH: f32 = 3.0;
    const BAR_GAP: f32 = 3.0;
    const MAX_HEIGHT: f32 = 16.0;
    const MIN_HEIGHT: f32 = 4.0;
    const CYCLES_PER_SECOND: f64 = 2.2;
    const PHASE_STEP: f64 = 1.7; // desyncs bars so they don't move in lockstep

    let total_width = BAR_COUNT as f32 * BAR_WIDTH + (BAR_COUNT as f32 - 1.0) * BAR_GAP;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(total_width, MAX_HEIGHT), egui::Sense::hover());
    let t = ui.ctx().input(|i| i.time);
    let painter = ui.painter();

    for i in 0..BAR_COUNT {
        let height = if animate {
            let phase = i as f64 * PHASE_STEP;
            let wave = ((t * std::f64::consts::TAU * CYCLES_PER_SECOND + phase).sin() * 0.5 + 0.5) as f32;
            MIN_HEIGHT + wave * (MAX_HEIGHT - MIN_HEIGHT)
        } else {
            MIN_HEIGHT
        };
        let x = rect.left() + i as f32 * (BAR_WIDTH + BAR_GAP);
        let bar_rect = egui::Rect::from_min_max(
            egui::pos2(x, rect.center().y - height / 2.0),
            egui::pos2(x + BAR_WIDTH, rect.center().y + height / 2.0),
        );
        painter.rect_filled(bar_rect, egui::CornerRadius::same(1), color);
    }
}

fn rgba_to_color32(rgba: [u8; 4]) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(rgba[0], rgba[1], rgba[2], rgba[3])
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let mut truncated: String = s.chars().take(max_chars.saturating_sub(1)).collect();
        truncated.push('…');
        truncated
    }
}

/// The pill's icon (window/taskbar), matching [`icon::render_mic_icon`]'s
/// neutral idle color -- the pill itself is normally hidden, so this is
/// mostly what shows up in alt-tab if it's ever visible unexpectedly.
pub fn window_icon() -> egui::IconData {
    let size = 32;
    let rgba = icon::render_mic_icon(size, icon::NEUTRAL, icon::GLYPH_COLOR);
    egui::IconData {
        rgba,
        width: size,
        height: size,
    }
}
