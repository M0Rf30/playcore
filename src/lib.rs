// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared Symphonia-based playback core.
//!
//! Everything here is pure computation over Symphonia and the standard library: no audio
//! device, no async runtime, no network, no configuration format. Applications supply the
//! device layer and the policy; this crate supplies the parts that must behave identically
//! everywhere.
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`error`] | [`Error`] / [`MediaErrorKind`] |
//! | [`mod@format`] | the shared Symphonia probe, supported extension/MIME lists, WavPack `.wvc` options |
//! | [`decoder`] | [`SymphoniaDecoder`]: seek-accurate, gapless, chained Ogg, HE-AAC, DSD |
//! | [`source`] | `Read`/`Seek` → Symphonia `MediaSource` adapters |
//! | [`dsd`] | DoP encoder and DSD-to-PCM rate selection |
//! | [`convert`] | `f32` → integer conversion, the real-time sample queue, gain ramp |
//! | [`control`] | lock-free pause/flush/gain control block shared with audio callbacks |
//! | [`crossfade`] | equal-power / linear crossfade and MixRamp primitives |
//! | [`fade`] | fade-in/out ramp control |
//! | [`filter`] | [`AudioFilter`](filter::AudioFilter), [`FilterChain`](filter::FilterChain), volume |
//! | [`dsp`] | equalizer, normalizer/limiter, channel routing |
//! | [`replaygain`] | ReplayGain scale computation |
//! | `resample` | streaming sample-rate converter (feature `resample`) |

pub mod control;
pub mod convert;
pub mod crossfade;
pub mod decoder;
pub mod dsd;
pub mod dsp;
pub mod error;
pub mod fade;
pub mod filter;
pub mod format;
pub mod replaygain;
#[cfg(feature = "resample")]
pub mod resample;
pub mod source;

pub use decoder::{AudioFormat, Decoder, DecoderOptions, SymphoniaDecoder};
pub use error::{Error, MediaErrorKind, Result};
