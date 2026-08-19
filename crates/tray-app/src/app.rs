//! The floating recording pill: a small, borderless, always-on-top window
//! that shows the engine's current state -- the visual language Wispr
//! Flow popularized for dictation apps (a minimal pill near the cursor
//! that appears while recording and fades after inserting text), redrawn
//! here from scratch with original colors/shapes, not their assets.
//!
//! Palette follows OpenNote's (opennote.com) warm, approachable direction
//! -- cream background, warm brown text, golds/terracotta/sage for state
//! -- instead of the cooler dark-slate "engineering tool" look most
//! dictation-adjacent utilities default to. Same reasoning as the icon in
//! `icon.rs`: the *shapes and colors* are original work inspired by that
//! product's general aesthetic, not anything copied from it.

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
}

impl PillApp {
    pub fn new(
        status_rx: mpsc::Receiver<daemon::PipelineStatus>,
        control_tx: mpsc::Sender<daemon::ControlEvent>,
        tray_icon: tray_icon::TrayIcon,
        menu_ids: crate::tray::MenuIds,
    ) -> Self {
        Self {
            status_rx,
            control_tx,
            menu_ids,
            display: Display::Hidden,
            hands_free_on: false,
            tray_icon,
        }
    }

    fn apply_status(&mut self, status: daemon::PipelineStatus) {
        use daemon::PipelineStatus as S;
        let _ = self.tray_icon.set_icon(Some(crate::tray::icon_for_status(&status)));
        match status {
            S::Ready { .. } => self.display = Display::Hidden,
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
            } else if event.id() == &self.menu_ids.quit {
                let _ = self.control_tx.send(daemon::ControlEvent::Quit);
                std::process::exit(0);
            }
        }
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
        ctx.request_repaint_after(REPAINT_INTERVAL);
        ctx.send_viewport_cmd(egui::ViewportCommand::Visible(!matches!(self.display, Display::Hidden)));

        let (color, label) = match &self.display {
            Display::Hidden => return,
            Display::Recording => (rgba_to_color32(icon::TERRACOTTA), "Recording…".to_string()),
            Display::Listening => (rgba_to_color32(icon::GOLDEN_AMBER), "Listening…".to_string()),
            Display::Transcribing => (rgba_to_color32(icon::MUSTARD), "Thinking…".to_string()),
            Display::Inserted { text, .. } => (rgba_to_color32(icon::SAGE), truncate(text, 60)),
            Display::Warning { text, .. } => (rgba_to_color32(icon::RUST), truncate(text, 60)),
        };

        // A true capsule, not just a rounded rect: corner radius = half
        // the pill's fixed height (see main.rs's viewport inner_size).
        let corner_radius = 28;

        egui::Frame::new()
            .fill(rgba_to_color32(icon::CREAM_BACKGROUND))
            .corner_radius(egui::CornerRadius::same(corner_radius))
            .inner_margin(egui::Margin::symmetric(16, 12))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                    ui.painter().circle_filled(rect.center(), 5.0, color);
                    ui.add_space(6.0);
                    ui.colored_label(rgba_to_color32(icon::WARM_TEXT), label);
                });
            });
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
    let rgba = icon::render_mic_icon(size, icon::WARM_NEUTRAL, icon::GLYPH_COLOR);
    egui::IconData {
        rgba,
        width: size,
        height: size,
    }
}
