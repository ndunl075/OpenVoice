//! OpenVoice's GUI front end: a system tray icon (with an original mic
//! logo, not any third-party product's branding -- see `icon.rs`) plus a
//! small floating "recording pill" window, both driving the exact same
//! `daemon::Engine` the console binary uses. No console window: this
//! binary is built with the Windows GUI subsystem, so launching it
//! (double-click, Start Menu, etc.) never pops a terminal.
//!
//! Startup happens on the main thread (model loading takes a few seconds;
//! blocking briefly before the window/tray appear is simpler and more
//! honest than showing an empty window first). The engine then moves to a
//! background thread for [`daemon::Engine::run`], which blocks until a
//! `Quit` control event arrives; the main thread runs the GUI event loop
//! and talks to that background thread only over channels.
#![windows_subsystem = "windows"]

mod app;
mod icon;
mod tray;

use std::sync::mpsc;

/// The one place the product name lives -- every user-visible string
/// (window title, tray tooltip, message box, console banner) reads from
/// here instead of repeating the name as a scattered literal.
pub(crate) const APP_NAME: &str = "OpenVoice";

fn main() {
    // No console window in GUI-subsystem builds, so this only shows up
    // when run from a terminal (e.g. `cargo run`) -- still useful there.
    println!("{APP_NAME} -- tray mode");

    let engine = match daemon::Engine::load(|line| println!("{line}")) {
        Ok(engine) => engine,
        Err(e) => {
            fatal_error_dialog(&format!(
                "Couldn't start the dictation engine:\n\n{e}\n\nSee the relevant crate's README.md for how to fetch a model file."
            ));
            std::process::exit(1);
        }
    };

    let mic_name = engine.mic_name().to_string();
    let hotkey_config = engine.hotkey_config();
    let control_tx = engine.control_sender();

    let (tray_icon, menu_ids) = match tray::build(&mic_name) {
        Ok(built) => built,
        Err(e) => {
            fatal_error_dialog(&format!("Couldn't create the system tray icon:\n\n{e}"));
            std::process::exit(1);
        }
    };

    let (status_tx, status_rx) = mpsc::channel::<daemon::PipelineStatus>();
    std::thread::spawn(move || engine.run(status_tx));

    let viewport = eframe::egui::ViewportBuilder::default()
        .with_title(APP_NAME)
        .with_decorations(false)
        .with_transparent(true)
        .with_always_on_top()
        .with_taskbar(false)
        .with_resizable(false)
        .with_inner_size([app::PILL_SIZE[0], app::PILL_SIZE[1]])
        .with_visible(false)
        // The pill must never take keyboard focus. It is shown at exactly
        // the moment the user is dictating into some *other* window, and
        // §2.5's injection sends synthetic keystrokes to whatever has
        // focus -- so a pill that activates when it appears would quietly
        // receive the paste itself, with every layer still reporting
        // success. It's a status indicator; it has nothing to type into.
        .with_active(false)
        .with_icon(app::window_icon());

    let native_options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    let result = eframe::run_native(
        APP_NAME,
        native_options,
        Box::new(move |cc| {
            // The pill draws its own rounded background (app.rs). egui's
            // default panel fill and window stroke/shadow would render a
            // second, larger rectangle *behind* it -- visible as an
            // outline around the pill. Clear all three so the only thing
            // on screen is the shape we paint ourselves.
            // all_styles_mut, not style_mut_of(theme): the pill paints
            // its own fixed cream/dark palette, so it must look the same
            // whether the OS reports light or dark mode.
            cc.egui_ctx.all_styles_mut(|style| {
                style.visuals.panel_fill = eframe::egui::Color32::TRANSPARENT;
                style.visuals.window_fill = eframe::egui::Color32::TRANSPARENT;
                style.visuals.window_stroke = eframe::egui::Stroke::NONE;
                style.visuals.window_shadow = eframe::egui::epaint::Shadow::NONE;
                style.visuals.popup_shadow = eframe::egui::epaint::Shadow::NONE;
            });

            Ok(Box::new(app::PillApp::new(
                status_rx,
                control_tx,
                tray_icon,
                menu_ids,
                mic_name,
                hotkey_config,
            )))
        }),
    );

    if let Err(e) = result {
        eprintln!("error: GUI event loop exited with an error: {e}");
        std::process::exit(1);
    }
}

/// Startup failures happen before any window exists, and this binary has
/// no console in a normal launch -- a message box is the only way the
/// user would ever see them.
fn fatal_error_dialog(message: &str) {
    eprintln!("error: {message}");
    native_message_box(message);
}

#[cfg(windows)]
fn native_message_box(message: &str) {
    use std::ffi::OsStr;
    use std::iter::once;
    use std::os::windows::ffi::OsStrExt;

    use windows::Win32::UI::WindowsAndMessaging::{MessageBoxW, MB_ICONERROR, MB_OK};

    let title: Vec<u16> = OsStr::new(APP_NAME).encode_wide().chain(once(0)).collect();
    let body: Vec<u16> = OsStr::new(message).encode_wide().chain(once(0)).collect();
    unsafe {
        MessageBoxW(
            None,
            windows::core::PCWSTR(body.as_ptr()),
            windows::core::PCWSTR(title.as_ptr()),
            MB_OK | MB_ICONERROR,
        );
    }
}

#[cfg(not(windows))]
fn native_message_box(_message: &str) {}
