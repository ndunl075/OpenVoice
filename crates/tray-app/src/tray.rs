//! System tray icon + menu setup. Menu items are read back by id in
//! [`crate::app::PillApp::drain_events`] -- `MenuIds` is how that matching
//! stays typed instead of comparing against raw strings.

use tray_icon::menu::{Menu, MenuId, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

use crate::icon;

pub struct MenuIds {
    pub hands_free: MenuId,
    pub quit: MenuId,
}

#[derive(Debug, thiserror::Error)]
pub enum TrayError {
    #[error("tray icon error: {0}")]
    TrayIcon(#[from] tray_icon::Error),
    #[error("menu error: {0}")]
    Menu(#[from] tray_icon::menu::Error),
}

/// Builds and registers the tray icon with its right-click menu. The icon
/// itself is the neutral/idle color from [`icon::state_background_rgba`]'s
/// palette -- `app.rs` repaints it as pipeline state changes (see
/// `PillApp::apply_status`).
pub fn build(mic_name: &str) -> Result<(TrayIcon, MenuIds), TrayError> {
    let hands_free_item = MenuItem::new("Toggle Hands-Free", true, None);
    let quit_item = MenuItem::new("Quit", true, None);
    let menu_ids = MenuIds {
        hands_free: hands_free_item.id().clone(),
        quit: quit_item.id().clone(),
    };

    let menu = Menu::new();
    menu.append(&hands_free_item)?;
    menu.append(&PredefinedMenuItem::separator())?;
    menu.append(&quit_item)?;

    let tray_icon = TrayIconBuilder::new()
        .with_icon(idle_icon())
        .with_menu(Box::new(menu))
        .with_tooltip(format!("Local Dictation Engine\nMic: {mic_name}"))
        .build()?;

    Ok((tray_icon, menu_ids))
}

fn idle_icon() -> Icon {
    icon_for_rgba(icon::WARM_NEUTRAL)
}

pub fn icon_for_status(status: &daemon::PipelineStatus) -> Icon {
    icon_for_rgba(icon::state_background_rgba(status))
}

/// We generate the RGBA buffer ourselves at a fixed, correct size, so
/// `from_rgba` failing would mean a bug in [`icon::render_mic_icon`], not
/// a runtime condition callers need to handle.
fn icon_for_rgba(background: [u8; 4]) -> Icon {
    const SIZE: u32 = 32;
    let rgba = icon::render_mic_icon(SIZE, background, icon::GLYPH_COLOR);
    Icon::from_rgba(rgba, SIZE, SIZE).expect("generated icon buffer is always the declared size")
}
