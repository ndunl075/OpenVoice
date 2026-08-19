//! Original, mic-themed icon generation.
//!
//! Deliberately **not** a copy of any third-party product's logo -- this
//! draws a simple circle-with-a-mic-glyph from scratch, in code, so there
//! are no external brand assets in this repo at all. Pure pixel math over
//! an RGBA buffer, so the shape is unit-testable without a graphics
//! context; only the actual `Icon`/`IconData` construction from that
//! buffer (in `main.rs`) touches a real windowing API.

/// Renders a filled circle in `background` with a simple mic glyph (a
/// rounded "head" + a stand) in `glyph`, on an otherwise-transparent
/// `size` x `size` RGBA buffer. Used for both the tray icon (small,
/// state-colored) and the pill window's icon.
pub fn render_mic_icon(size: u32, background: [u8; 4], glyph: [u8; 4]) -> Vec<u8> {
    let mut buf = vec![0u8; (size as usize) * (size as usize) * 4];
    draw_filled_circle(&mut buf, size, background);
    draw_mic_glyph(&mut buf, size, glyph);
    buf
}

fn set_pixel(buf: &mut [u8], size: u32, x: i32, y: i32, color: [u8; 4]) {
    if x < 0 || y < 0 || x as u32 >= size || y as u32 >= size {
        return;
    }
    let idx = ((y as u32 * size + x as u32) * 4) as usize;
    buf[idx..idx + 4].copy_from_slice(&color);
}

fn draw_filled_circle(buf: &mut [u8], size: u32, color: [u8; 4]) {
    let center = size as f32 / 2.0;
    let radius = size as f32 * 0.46;
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 + 0.5 - center;
            let dy = y as f32 + 0.5 - center;
            if dx * dx + dy * dy <= radius * radius {
                set_pixel(buf, size, x as i32, y as i32, color);
            }
        }
    }
}

/// A capsule "head" (rounded rectangle, approximated as a rect with
/// circular caps) plus a short stand below it -- the universal mic
/// pictogram, simplified to what's legible at 16-32px.
fn draw_mic_glyph(buf: &mut [u8], size: u32, color: [u8; 4]) {
    let size_f = size as f32;
    let head_w = size_f * 0.26;
    let head_h = size_f * 0.38;
    let head_cx = size_f / 2.0;
    let head_top = size_f * 0.20;
    let head_cap_r = head_w / 2.0;

    let head_left = head_cx - head_w / 2.0;
    let head_right = head_cx + head_w / 2.0;
    let head_rect_top = head_top + head_cap_r;
    let head_rect_bottom = head_top + head_h - head_cap_r;

    for y in 0..size {
        for x in 0..size {
            let xf = x as f32 + 0.5;
            let yf = y as f32 + 0.5;
            let in_rect_body =
                xf >= head_left && xf <= head_right && yf >= head_rect_top && yf <= head_rect_bottom;
            let dist_top_cap = ((xf - head_cx).powi(2) + (yf - head_rect_top).powi(2)).sqrt();
            let dist_bottom_cap = ((xf - head_cx).powi(2) + (yf - head_rect_bottom).powi(2)).sqrt();
            let in_cap = (yf < head_rect_top && dist_top_cap <= head_cap_r)
                || (yf > head_rect_bottom && dist_bottom_cap <= head_cap_r);
            if in_rect_body || in_cap {
                set_pixel(buf, size, x as i32, y as i32, color);
            }
        }
    }

    // Stand: a vertical stroke below the head, with a small horizontal base.
    let stand_x = head_cx;
    let stand_top = head_top + head_h;
    let stand_bottom = size_f * 0.80;
    let stroke = (size_f * 0.06).max(1.0);
    for y in 0..size {
        for x in 0..size {
            let xf = x as f32 + 0.5;
            let yf = y as f32 + 0.5;
            if yf >= stand_top && yf <= stand_bottom && (xf - stand_x).abs() <= stroke / 2.0 {
                set_pixel(buf, size, x as i32, y as i32, color);
            }
        }
    }
    let base_half_w = size_f * 0.14;
    let base_y = stand_bottom;
    for y in 0..size {
        for x in 0..size {
            let xf = x as f32 + 0.5;
            let yf = y as f32 + 0.5;
            if (xf - stand_x).abs() <= base_half_w && (yf - base_y).abs() <= stroke / 2.0 {
                set_pixel(buf, size, x as i32, y as i32, color);
            }
        }
    }
}

/// Tray icon background color for each pipeline state -- lets the mode be
/// visible at a glance without opening the pill. Fully opaque; the mic
/// glyph itself is always [`GLYPH_COLOR`] for contrast against any of
/// these. Warm palette (golds, terracotta, sage) rather than the
/// cool slate/blue/red of a typical dark "engineering tool" look --
/// see `crates/tray-app/README.md` for the OpenNote-inspired direction
/// this and the pill (`app.rs`) both follow.
pub fn state_background_rgba(status: &daemon::PipelineStatus) -> [u8; 4] {
    use daemon::PipelineStatus::*;
    match status {
        Ready { .. } => WARM_NEUTRAL,
        Recording => TERRACOTTA,
        Listening => GOLDEN_AMBER,
        Transcribing => MUSTARD,
        Inserted(_) => SAGE,
        HeardNothing => WARM_NEUTRAL,
        HandsFreeOn => GOLDEN_AMBER,
        HandsFreeOff => WARM_NEUTRAL,
        Warning(_) => RUST,
    }
}

// pub(crate): app.rs's pill reuses these directly so the tray icon and
// the pill are never one state-color edit away from disagreeing with
// each other.
pub(crate) const WARM_NEUTRAL: [u8; 4] = [196, 176, 145, 255]; // sand/taupe: idle, resting
pub(crate) const TERRACOTTA: [u8; 4] = [224, 122, 95, 255]; // actively recording
pub(crate) const GOLDEN_AMBER: [u8; 4] = [232, 180, 84, 255]; // hands-free listening
pub(crate) const MUSTARD: [u8; 4] = [212, 160, 60, 255]; // thinking / transcribing
pub(crate) const SAGE: [u8; 4] = [139, 163, 120, 255]; // success / just inserted
pub(crate) const RUST: [u8; 4] = [196, 90, 74, 255]; // warning
/// Warm cream, for the pill's own background (`app.rs`) -- deliberately
/// close to but distinct from [`GLYPH_COLOR`] so a glyph drawn in it
/// would still be very faintly legible rather than truly invisible.
pub(crate) const CREAM_BACKGROUND: [u8; 4] = [250, 242, 227, 255];
/// Warm dark brown, for text against [`CREAM_BACKGROUND`].
pub(crate) const WARM_TEXT: [u8; 4] = [74, 58, 45, 255];

/// Warm cream (not stark cool white) for the mic glyph, against any
/// background color above.
pub const GLYPH_COLOR: [u8; 4] = [250, 244, 230, 255];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_has_correct_length_for_size() {
        let buf = render_mic_icon(32, [10, 20, 30, 255], GLYPH_COLOR);
        assert_eq!(buf.len(), 32 * 32 * 4);
    }

    #[test]
    fn corners_are_transparent_outside_the_circle() {
        let size = 32;
        let buf = render_mic_icon(size, [10, 20, 30, 255], GLYPH_COLOR);
        let idx = 0; // top-left corner pixel
        assert_eq!(&buf[idx..idx + 4], &[0, 0, 0, 0]);
    }

    #[test]
    fn center_of_circle_is_background_or_glyph_but_not_transparent() {
        let size = 32;
        let background = [10, 20, 30, 255];
        let buf = render_mic_icon(size, background, GLYPH_COLOR);
        let cx = size / 2;
        let cy = size / 2;
        let idx = ((cy * size + cx) * 4) as usize;
        assert_ne!(buf[idx + 3], 0, "center pixel should be opaque (circle interior)");
    }

    #[test]
    fn some_pixels_are_drawn_in_the_glyph_color() {
        let size = 32;
        let buf = render_mic_icon(size, [10, 20, 30, 255], GLYPH_COLOR);
        let has_glyph_pixel = buf
            .chunks_exact(4)
            .any(|px| px == GLYPH_COLOR);
        assert!(has_glyph_pixel, "expected at least one glyph-colored pixel");
    }

    #[test]
    fn different_sizes_all_produce_a_valid_buffer() {
        for size in [16, 24, 32, 64] {
            let buf = render_mic_icon(size, [1, 2, 3, 255], GLYPH_COLOR);
            assert_eq!(buf.len(), (size * size * 4) as usize);
        }
    }

    #[test]
    fn state_colors_are_distinct_for_active_vs_idle() {
        let idle = state_background_rgba(&daemon::PipelineStatus::HeardNothing);
        let recording = state_background_rgba(&daemon::PipelineStatus::Recording);
        assert_ne!(idle, recording);
    }

    #[test]
    fn warm_text_has_real_contrast_against_the_cream_pill_background() {
        // Not a full WCAG contrast-ratio check -- just a floor against
        // shipping near-invisible text if someone tweaks either color
        // later. Sum of per-channel difference is a cheap, good-enough
        // proxy for "these are clearly different tones."
        let diff: i32 = CREAM_BACKGROUND
            .iter()
            .zip(WARM_TEXT.iter())
            .take(3) // RGB only, ignore alpha
            .map(|(&a, &b)| (a as i32 - b as i32).abs())
            .sum();
        assert!(diff > 300, "text/background too close in tone: diff={diff}");
    }
}
