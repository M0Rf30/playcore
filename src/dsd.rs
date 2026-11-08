// SPDX-License-Identifier: MIT OR Apache-2.0

//! DSD helpers: the DoP (DSD over PCM) encoder and the DSD-to-PCM rate selection.
//!
//! # DoP
//!
//! Packs DSD data into 24-bit PCM frames for transmission over standard PCM audio interfaces.
//! The DAC recognizes the DoP markers and extracts the original DSD stream.
//!
//! DoP Format:
//! - DSD64 (2.8224 MHz) → 176.4 kHz PCM (2.8224 / 16 = 176.4)
//! - DSD128 (5.6448 MHz) → 352.8 kHz PCM (5.6448 / 16 = 352.8)
//! - DSD256 (11.2896 MHz) → 705.6 kHz PCM (11.2896 / 16 = 705.6)
//! - DSD512 (22.5792 MHz) → 1411.2 kHz PCM (22.5792 / 16 = 1411.2)
//!
//! The DoP framing itself is rate-independent (16 DSD bits per 24-bit PCM
//! frame), so any DSD rate that is an integer multiple of the DSD64 base
//! rate is accepted here; mpd's `src/pcm/Dop.cxx` imposes no upper cap
//! either and lets the output device decide what it can open. A device
//! that cannot open the resulting PCM rate fails cleanly when the DoP
//! output stream is started, and playback falls back to DSD->PCM
//! conversion (see the caller's DoP setup).
//!
//! Each PCM sample contains:
//! - Byte 0: Marker (0x05 or 0xFA alternating)
//! - Byte 1: 8 bits of DSD data
//! - Byte 2: 8 more bits of DSD data
//!
//! Reference: <https://dsd-guide.com/sites/default/files/white-papers/DoP_openStandard_1v1.pdf>

use crate::error::{Error, Result};
use symphonia::core::codecs::audio::{BitOrder, ChannelDataLayout};

const DOP_MARKER_1: u8 = 0x05;
const DOP_MARKER_2: u8 = 0xFA;

/// Base DSD sample rate (DSD64 = 44.1 kHz x 64). Every standard DSD rate
/// (DSD128, DSD256, DSD512, ...) is a power-of-two multiple of this.
const DSD64_RATE: u32 = 2_822_400;

/// Lookup table for bit reversal (LSB-first to MSB-first).
/// Generated at compile time. Each index maps to its bit-reversed value.
const BIT_REVERSE_TABLE: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut i = 0u16;
    while i < 256 {
        let mut result = 0u8;
        let mut b = i as u8;
        let mut bit = 0;
        while bit < 8 {
            result = (result << 1) | (b & 1);
            b >>= 1;
            bit += 1;
        }
        table[i as usize] = result;
        i += 1;
    }
    table
};

/// Reverse the bits in a byte using lookup table.
/// Used when source DSD is LSB-first but DAC expects MSB-first in DoP.
#[inline]
fn reverse_bits(byte: u8) -> u8 {
    BIT_REVERSE_TABLE[byte as usize]
}

/// DoP encoder that converts DSD data to DoP-encoded PCM
pub struct DopEncoder {
    dsd_sample_rate: u32,
    channels: usize,
    marker_toggle: bool,
    channel_layout: ChannelDataLayout,
    bit_order: BitOrder,
}

impl DopEncoder {
    pub fn new(
        dsd_sample_rate: u32,
        channels: usize,
        channel_layout: ChannelDataLayout,
        bit_order: BitOrder,
    ) -> Result<Self> {
        // Validate DSD sample rate: must be a power-of-two multiple of
        // DSD64 (2.8224 MHz) — DSD128, DSD256, DSD512, etc. The DoP framing
        // (16 DSD bits per 24-bit PCM frame) is rate-independent, so unlike
        // a hand-picked list this accepts DSD256/DSD512 too; mpd's
        // `src/pcm/Dop.cxx` imposes no such cap either. The output device
        // gets the final say: opening the resulting PCM rate fails cleanly
        // (`DopOutput::new`) if the DAC can't do it, and the caller reverts
        // to DSD->PCM conversion (see the caller's DoP setup).
        let dsd_multiple = dsd_sample_rate / DSD64_RATE;
        if dsd_sample_rate == 0
            || !dsd_sample_rate.is_multiple_of(DSD64_RATE)
            || !dsd_multiple.is_power_of_two()
        {
            return Err(Error::other(format!(
                "Unsupported DSD sample rate: {dsd_sample_rate}"
            )));
        }

        Ok(Self {
            dsd_sample_rate,
            channels,
            marker_toggle: false,
            channel_layout,
            bit_order,
        })
    }

    /// Get the PCM sample rate for DoP encoding
    pub fn pcm_sample_rate(&self) -> u32 {
        // Each PCM frame contains 16 DSD bits (2 bytes)
        self.dsd_sample_rate / 16
    }

    /// Convert DSD samples to DoP-encoded 24-bit PCM samples
    ///
    /// Input: Raw DSD data (1 bit per sample, packed into bytes)
    /// Output: 24-bit PCM samples (as i32) ready for playback
    ///
    /// Handles both planar and interleaved channel layouts
    /// Handles both LSB-first and MSB-first bit orders
    ///
    /// DoP PCM output (always interleaved):
    /// [L_marker_DSD0-15, R_marker_DSD0-15, ...] for each frame
    pub fn encode(&mut self, dsd_data: &[u8], output: &mut Vec<i32>) {
        // Each DoP PCM sample contains 16 DSD bits (2 bytes per channel)
        let dsd_bytes_per_channel_per_frame = 2;

        match self.channel_layout {
            ChannelDataLayout::Planar => {
                // Planar layout: all bytes for channel 0, then all for channel 1
                let bytes_per_channel = dsd_data.len() / self.channels;
                let num_frames = bytes_per_channel / dsd_bytes_per_channel_per_frame;

                // Reserve space for output (one sample per channel per frame, interleaved)
                output.clear();
                output.reserve(num_frames * self.channels);

                for frame_idx in 0..num_frames {
                    // Alternate marker for each frame
                    let marker = if self.marker_toggle {
                        DOP_MARKER_2
                    } else {
                        DOP_MARKER_1
                    };
                    self.marker_toggle = !self.marker_toggle;

                    // Process each channel (convert planar to interleaved output)
                    for ch in 0..self.channels {
                        // Calculate offset in planar layout
                        let channel_offset = ch * bytes_per_channel;
                        let dsd_offset =
                            channel_offset + (frame_idx * dsd_bytes_per_channel_per_frame);

                        // Guard against a truncated trailing frame: stop
                        // encoding rather than panic on short input.
                        if dsd_offset + 1 >= dsd_data.len() {
                            break;
                        }

                        // Get 2 bytes of DSD data for this channel
                        let dsd_byte1 = dsd_data[dsd_offset];
                        let dsd_byte2 = dsd_data[dsd_offset + 1];

                        // Apply bit reversal (LSB-first -> MSB-first)
                        let (byte1, byte2) = if self.bit_order == BitOrder::LsbFirst {
                            (reverse_bits(dsd_byte1), reverse_bits(dsd_byte2))
                        } else {
                            (dsd_byte1, dsd_byte2)
                        };

                        // Pack into 24-bit DoP sample: [marker, byte1, byte2]
                        // For 32-bit output (S32_LE), left-align by shifting left 8 bits
                        // Result: [marker][byte1][byte2][0x00]
                        let dop_sample = ((marker as i32) << 24)
                            | ((byte1 as i32) << 16)
                            | ((byte2 as i32) << 8);

                        output.push(dop_sample);
                    }
                }
            }
            ChannelDataLayout::Interleaved => {
                // Interleaved layout: [L0, R0, L1, R1, L2, R2, ...]
                let num_frames = dsd_data.len() / (self.channels * dsd_bytes_per_channel_per_frame);

                // Reserve space for output
                output.clear();
                output.reserve(num_frames * self.channels);

                for frame_idx in 0..num_frames {
                    // Alternate marker for each frame
                    let marker = if self.marker_toggle {
                        DOP_MARKER_2
                    } else {
                        DOP_MARKER_1
                    };
                    self.marker_toggle = !self.marker_toggle;

                    // Process each channel
                    for ch in 0..self.channels {
                        // Calculate offset in interleaved layout
                        let dsd_offset =
                            (frame_idx * self.channels + ch) * dsd_bytes_per_channel_per_frame;

                        // Guard against a truncated trailing frame: stop
                        // encoding rather than panic on short input.
                        if dsd_offset + 1 >= dsd_data.len() {
                            break;
                        }

                        // Get 2 bytes of DSD data for this channel
                        let dsd_byte1 = dsd_data[dsd_offset];
                        let dsd_byte2 = dsd_data[dsd_offset + 1];

                        // Apply bit reversal (LSB-first -> MSB-first)
                        let (byte1, byte2) = if self.bit_order == BitOrder::LsbFirst {
                            (reverse_bits(dsd_byte1), reverse_bits(dsd_byte2))
                        } else {
                            (dsd_byte1, dsd_byte2)
                        };

                        // Pack into 24-bit DoP sample: [marker, byte1, byte2]
                        // For 32-bit output (S32_LE), left-align by shifting left 8 bits
                        // Result: [marker][byte1][byte2][0x00]
                        let dop_sample = ((marker as i32) << 24)
                            | ((byte1 as i32) << 16)
                            | ((byte2 as i32) << 8);

                        output.push(dop_sample);
                    }
                }
            }
        }
    }

    /// Convert DoP 24-bit samples (i32) to f32 for a float-only output API
    pub fn to_f32_samples(dop_i32: &[i32]) -> Vec<f32> {
        dop_i32
            .iter()
            .map(|&sample| {
                // Normalize 24-bit to f32 range [-1.0, 1.0]
                // 24-bit range: -8388608 to 8388607
                (sample as f32) / 8388608.0
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dop_encoder_dsd64() {
        // Test with MSB-first bit order (no reversal needed)
        let mut encoder =
            DopEncoder::new(2822400, 2, ChannelDataLayout::Planar, BitOrder::MsbFirst).unwrap();
        assert_eq!(encoder.pcm_sample_rate(), 176400);

        // Test data: 4 bytes planar format (all left, then all right)
        // Left channel: [0x12, 0x34]
        // Right channel: [0x56, 0x78]
        let dsd_data = vec![0x12, 0x34, 0x56, 0x78];
        let mut output = Vec::new();

        encoder.encode(&dsd_data, &mut output);

        // Should produce 2 samples: L and R for one frame
        assert_eq!(output.len(), 2);

        // First frame, left channel: [marker1][0x12][0x34][0x00] (left-aligned)
        assert_eq!(
            output[0],
            (DOP_MARKER_1 as i32) << 24 | 0x12 << 16 | 0x34 << 8
        );

        // First frame, right channel: [marker1][0x56][0x78][0x00] (left-aligned)
        assert_eq!(
            output[1],
            (DOP_MARKER_1 as i32) << 24 | 0x56 << 16 | 0x78 << 8
        );
    }

    #[test]
    fn test_marker_alternation() {
        // Test with MSB-first bit order
        let mut encoder =
            DopEncoder::new(2822400, 1, ChannelDataLayout::Planar, BitOrder::MsbFirst).unwrap();
        let dsd_data = vec![0xAA, 0xBB, 0xCC, 0xDD];
        let mut output = Vec::new();

        encoder.encode(&dsd_data, &mut output);

        // Extract markers from samples (now in MSB due to left-alignment)
        let marker1 = (output[0] >> 24) as u8;
        let marker2 = (output[1] >> 24) as u8;

        // Markers should alternate
        assert_eq!(marker1, DOP_MARKER_1);
        assert_eq!(marker2, DOP_MARKER_2);
    }

    #[test]
    fn test_encode_truncated_last_frame_no_panic() {
        // Interleaved, 2 channels: a full frame needs channels * 2 = 4 bytes.
        // 11 bytes is one byte short of 3 full frames (12 bytes).
        let mut encoder = DopEncoder::new(
            2822400,
            2,
            ChannelDataLayout::Interleaved,
            BitOrder::MsbFirst,
        )
        .unwrap();
        let dsd_data = vec![0u8; 11];
        let mut output = Vec::new();

        // Must not panic on the truncated trailing data.
        encoder.encode(&dsd_data, &mut output);

        // A naive `len / bytes_per_channel_per_frame` calculation (ignoring
        // channel count entirely) would predict 11 / 2 = 5 "frames"; only
        // whole multi-channel frames that fully fit are actually produced.
        let naive_frames = dsd_data.len() / 2;
        let produced_frames = output.len() / 2; // 2 channels per frame
        assert!(produced_frames < naive_frames);
        assert_eq!(produced_frames, 2); // floor(11 / (2 channels * 2 bytes)) == 2
    }

    #[test]
    fn dop_encoder_construction_succeeds_for_dsd64_dsd128_dsd256() {
        for (dsd_rate, expected_pcm_rate) in
            [(2822400, 176400), (5644800, 352800), (11289600, 705600)]
        {
            let encoder =
                DopEncoder::new(dsd_rate, 2, ChannelDataLayout::Planar, BitOrder::MsbFirst);
            assert!(
                encoder.is_ok(),
                "DSD rate {dsd_rate} Hz should build a DoP encoder"
            );
            assert_eq!(encoder.unwrap().pcm_sample_rate(), expected_pcm_rate);
        }
    }

    #[test]
    fn dop_encoder_construction_rejects_non_dsd_rate() {
        // 48000 Hz is ordinary PCM, not a power-of-two multiple of DSD64.
        let encoder = DopEncoder::new(48000, 2, ChannelDataLayout::Planar, BitOrder::MsbFirst);
        assert!(encoder.is_err());
    }
}

// ── DSD-to-PCM rate selection ────────────────────────────────────────────────

/// Valid DSD-to-PCM decode rates, ascending. DSD decimates cleanly only by an
/// integer power of two, so every target is 44.1 kHz-family.
pub const DSD_PCM_RATES: [u32; 4] = [44100, 88200, 176400, 352800];

/// Choose the DSD-to-PCM decode rate for a device running at `device_rate`.
///
/// Returns the SMALLEST DSD-family rate that both covers `device_rate` and is
/// reported as supported, falling back to the largest supported family rate and
/// finally to 88.2 kHz.
///
/// Decoding to the highest rate a device merely *advertises* is harmful:
/// systems like PipeWire advertise enormous ranges (up to ~768 kHz) but
/// resample internally, so an over-high PCM rate (a) gives a punishingly short
/// real-time callback period that underruns on scheduling jitter (audible
/// crackle), and (b) leaves DSD's ultrasonic shaped noise in the PCM, muddying
/// the sound. A moderate rate lets the decimation filter remove that noise and
/// keeps the buffer period comfortable.
pub fn select_dsd_pcm_rate(device_rate: u32, supports_rate: impl Fn(u32) -> bool) -> u32 {
    DSD_PCM_RATES
        .iter()
        .copied()
        .find(|&r| r >= device_rate && supports_rate(r))
        .or_else(|| {
            DSD_PCM_RATES
                .iter()
                .rev()
                .copied()
                .find(|&r| supports_rate(r))
        })
        .unwrap_or(88200)
}

/// The device rate the audio output should open at for a DSD-to-PCM stream
/// decoded at `decode_rate` on a device whose native rate is `device_rate`.
///
/// Returns `None` (open the stream at `decode_rate`, no resampling) when the
/// rates already match, or when `native_decode_ok` — the resolved output is an
/// explicitly-configured device that natively supports `decode_rate`, so it is
/// safe to play bit-perfect. Otherwise returns `Some(device_rate)` so the caller
/// resamples to the device's native rate itself, rather than letting a sound
/// server (e.g. PipeWire) resample a rate it merely advertises — which underruns
/// and leaves DSD ultrasonic noise in-band.
pub fn dsd_output_target_rate(
    decode_rate: u32,
    device_rate: u32,
    native_decode_ok: bool,
) -> Option<u32> {
    if native_decode_ok || decode_rate == device_rate {
        None
    } else {
        Some(device_rate)
    }
}

#[cfg(test)]
mod rate_tests {
    use super::*;

    #[test]
    fn pipewire_like_device_picks_moderate_rate_not_advertised_max() {
        // PipeWire advertises everything (huge range) and defaults to 48 kHz.
        // Must NOT pick 352.8 kHz; the smallest family rate covering 48 kHz
        // is 88.2 kHz.
        let rate = select_dsd_pcm_rate(48000, |_| true);
        assert_eq!(rate, 88200);
    }

    #[test]
    fn device_at_44100_picks_44100() {
        let rate = select_dsd_pcm_rate(44100, |_| true);
        assert_eq!(rate, 44100);
    }

    #[test]
    fn device_at_96000_picks_176400() {
        // 88.2 kHz does not cover 96 kHz; the smallest family rate >= 96 kHz is
        // 176.4 kHz.
        let rate = select_dsd_pcm_rate(96000, |_| true);
        assert_eq!(rate, 176400);
    }

    #[test]
    fn device_at_192000_picks_352800() {
        let rate = select_dsd_pcm_rate(192000, |_| true);
        assert_eq!(rate, 352800);
    }

    #[test]
    fn strict_48k_device_falls_back_to_largest_supported_family_rate() {
        // A device that only natively supports 44.1 kHz (e.g. some hw-locked
        // ALSA devices) while running at 48 kHz: no family rate >= 48 kHz is
        // supported, so fall back to the largest supported one (44.1 kHz). The
        // output layer then resamples 44.1 -> 48 kHz.
        let rate = select_dsd_pcm_rate(48000, |r| r == 44100);
        assert_eq!(rate, 44100);
    }

    #[test]
    fn no_support_info_falls_back_to_default() {
        let rate = select_dsd_pcm_rate(48000, |_| false);
        assert_eq!(rate, 88200);
    }

    // ── dsd_output_target_rate tests ─────────────────────────────────────────

    #[test]
    fn dsd_output_target_rate_default_device_resamples() {
        // System default (PipeWire, not bit-perfect): DSD decoded at 88200 Hz,
        // device natively 48000 Hz. the output must open at 48000 and resample
        // internally, not let PipeWire resample the advertised-but-non-native
        // 88200 Hz stream.
        assert_eq!(dsd_output_target_rate(88200, 48000, false), Some(48000));
    }

    #[test]
    fn dsd_output_target_rate_configured_dac_is_bit_perfect() {
        // Explicit DAC that natively supports 88200 Hz: open at the decode rate
        // (no override) so playback is bit-perfect, even though the device's
        // default rate is 48000 Hz.
        assert_eq!(dsd_output_target_rate(88200, 48000, true), None);
    }

    #[test]
    fn dsd_output_target_rate_same_returns_none() {
        // Device natively 44100 Hz, decode rate 44100 Hz: no extra resample needed.
        assert_eq!(dsd_output_target_rate(44100, 44100, false), None);
    }

    #[test]
    fn select_dsd_pcm_rate_unchanged_by_output_fix() {
        // Decode-rate selection is not affected by the output-rate fix.
        // A PipeWire-like device at 48000 Hz that advertises 88200 Hz support
        // should still decode to 88200 Hz; only the output stream opens at 48000.
        let rate = select_dsd_pcm_rate(48000, |r| r == 88200);
        assert_eq!(rate, 88200);
    }
}
