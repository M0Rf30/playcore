// SPDX-License-Identifier: MIT OR Apache-2.0

//! ReplayGain scale computation (pure math, no tag parsing).

use crate::convert::db_to_linear;

/// Which ReplayGain tags to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReplayGainMode {
    /// No gain adjustment.
    #[default]
    Off,
    /// Use the track gain/peak.
    Track,
    /// Use the album gain/peak.
    Album,
    /// Track gain while shuffling (`random`), album gain otherwise.
    Auto,
}

/// ReplayGain tag values of one song (gain in dB, peak as linear amplitude).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ReplayGainTags {
    /// Track gain in dB.
    pub track_gain: Option<f32>,
    /// Track peak (linear, 1.0 = full scale).
    pub track_peak: Option<f32>,
    /// Album gain in dB.
    pub album_gain: Option<f32>,
    /// Album peak (linear, 1.0 = full scale).
    pub album_peak: Option<f32>,
}

/// Linear gain factor for a song under `mode`.
///
/// - `preamp` (dB) is added to the selected gain; `missing_preamp` (dB) is used instead when the
///   selected gain tag is absent.
/// - `normalization` clamps the scale so `scale * peak` never exceeds full scale.
/// - `random` only matters for [`ReplayGainMode::Auto`]: shuffled playback breaks album context,
///   so the track gain is used while `random` is set and the album gain otherwise. There is no
///   cross-fallback to the other gain when the selected one is missing (MPD behaves the same
///   way); the `missing_preamp` applies.
#[must_use]
pub fn compute_gain_scale(
    tags: &ReplayGainTags,
    mode: ReplayGainMode,
    preamp: f32,
    missing_preamp: f32,
    normalization: bool,
    random: bool,
) -> f32 {
    let (gain_opt, peak_opt) = match mode {
        ReplayGainMode::Off => return 1.0,
        ReplayGainMode::Track => (tags.track_gain, tags.track_peak),
        ReplayGainMode::Album => (tags.album_gain, tags.album_peak),
        ReplayGainMode::Auto => {
            if random {
                (tags.track_gain, tags.track_peak)
            } else {
                (tags.album_gain, tags.album_peak)
            }
        }
    };
    let (db, peak) = match gain_opt {
        Some(gain) => (gain + preamp, peak_opt),
        None => (missing_preamp, None),
    };
    let mut scale = db_to_linear(db);
    if normalization
        && let Some(pk) = peak
        && pk > 0.0
        && scale * pk > 1.0
    {
        scale = 1.0 / pk;
    }
    scale
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags() -> ReplayGainTags {
        ReplayGainTags {
            track_gain: Some(-3.0),
            track_peak: Some(0.9),
            album_gain: Some(-6.0),
            album_peak: Some(0.8),
        }
    }

    #[test]
    fn off_is_unity() {
        assert!(
            (compute_gain_scale(&tags(), ReplayGainMode::Off, 5.0, 5.0, true, false) - 1.0).abs()
                < 1e-9
        );
    }

    #[test]
    fn auto_random_uses_track_gain() {
        // Random mode: `auto` selects track gain, not album gain, though both are present.
        let scale = compute_gain_scale(&tags(), ReplayGainMode::Auto, 0.0, 0.0, false, true);
        assert!((scale - 10f32.powf(-3.0 / 20.0)).abs() < 1e-6);
    }

    #[test]
    fn auto_not_random_uses_album_gain() {
        let scale = compute_gain_scale(&tags(), ReplayGainMode::Auto, 0.0, 0.0, false, false);
        assert!((scale - 10f32.powf(-6.0 / 20.0)).abs() < 1e-6);
    }

    #[test]
    fn auto_not_random_missing_album_gain_falls_back_to_missing_preamp() {
        // No cross-fallback to track gain: a missing album gain uses `missing_preamp`.
        let t = ReplayGainTags {
            album_gain: None,
            album_peak: None,
            ..tags()
        };
        let scale = compute_gain_scale(&t, ReplayGainMode::Auto, 0.0, -9.0, false, false);
        assert!((scale - 10f32.powf(-9.0 / 20.0)).abs() < 1e-6);
    }

    #[test]
    fn preamp_is_added_to_the_selected_gain() {
        let scale = compute_gain_scale(&tags(), ReplayGainMode::Track, 2.0, 0.0, false, false);
        assert!((scale - 10f32.powf(-1.0 / 20.0)).abs() < 1e-6);
    }

    #[test]
    fn normalization_clamps_to_the_peak() {
        let t = ReplayGainTags {
            track_gain: Some(6.0),
            track_peak: Some(0.9),
            ..Default::default()
        };
        let scale = compute_gain_scale(&t, ReplayGainMode::Track, 0.0, 0.0, true, false);
        assert!((scale - 1.0 / 0.9).abs() < 1e-6);
        let unclamped = compute_gain_scale(&t, ReplayGainMode::Track, 0.0, 0.0, false, false);
        assert!(unclamped > scale);
    }
}
