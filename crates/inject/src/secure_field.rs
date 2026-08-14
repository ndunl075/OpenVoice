//! Best-effort detection of whether the currently focused input control is
//! a password/secure field, so the daemon can refuse to inject dictated
//! text into it (§2.5: "Never inject into a field flagged as secure
//! input.").
//!
//! On Windows this recognizes classic Win32 Edit controls with
//! `ES_PASSWORD` set, via the same `EM_GETPASSWORDCHAR` message screen
//! readers and password managers use to make the same call. It does *not*
//! see into password fields rendered by Chromium/Electron/UWP apps, which
//! don't expose a native Win32 edit control for it -- there's no fully
//! general way to detect those short of app-specific accessibility
//! integration. Treat this as a floor, not a guarantee: it catches the
//! classic case for free, it doesn't catch everything.

#[cfg(windows)]
pub fn is_focused_field_secure() -> bool {
    use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
    use windows::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetGUIThreadInfo, GetWindowThreadProcessId, SendMessageW,
        GUITHREADINFO,
    };

    const EM_GETPASSWORDCHAR: u32 = 0x00D2;

    unsafe {
        let fg = GetForegroundWindow();
        if fg.0.is_null() {
            return false;
        }
        let thread_id = GetWindowThreadProcessId(fg, None);

        let mut info = GUITHREADINFO {
            cbSize: std::mem::size_of::<GUITHREADINFO>() as u32,
            ..Default::default()
        };
        if GetGUIThreadInfo(thread_id, &mut info).is_err() {
            return false;
        }

        let focused: HWND = if info.hwndFocus.0.is_null() {
            fg
        } else {
            info.hwndFocus
        };

        let result = SendMessageW(focused, EM_GETPASSWORDCHAR, Some(WPARAM(0)), Some(LPARAM(0)));
        result.0 != 0
    }
}

#[cfg(not(windows))]
pub fn is_focused_field_secure() -> bool {
    // No platform-specific check implemented yet; fail closed would block
    // all injection everywhere, so this stays permissive until a real
    // check is written for the platform. Windows is this project's
    // primary target (see dictation-architecture.md §3).
    false
}
