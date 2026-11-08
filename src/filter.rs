// SPDX-License-Identifier: MIT OR Apache-2.0

//! In-place DSP stage trait, the ordered [`FilterChain`] and the live [`VolumeFilter`].
//!
//! [`VolumeFilter`] reads a live `Arc<AtomicU8>` (0..=100) so the volume can be changed without
//! touching the chain.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

/// An in-place DSP stage over interleaved f32 samples.
///
/// The slice passed to [`Self::apply`] is the same length in and out; the filter
/// mutates it in place.
pub trait AudioFilter: Send {
    /// Human-readable name used for logging / debug.
    fn name(&self) -> &str;

    /// Apply the filter to `buf` in place.
    fn apply(&mut self, buf: &mut [f32]);

    /// Told the stream format before the first `apply` and again whenever it
    /// changes. Filters whose coefficients depend on the sample rate or that
    /// keep per-channel state recompute here. Default: ignore.
    fn set_format(&mut self, _sample_rate: u32, _channels: u8) {}
}

/// Software volume control (0..=100) read live from a shared atomic.
///
/// At `volume == 100` the filter short-circuits and returns immediately (no
/// multiply).  The atomic is read with `Acquire` ordering so any preceding
/// `store(Release)` from another thread is visible.
pub struct VolumeFilter {
    volume: Arc<AtomicU8>,
}

impl VolumeFilter {
    pub fn new(volume: Arc<AtomicU8>) -> Self {
        Self { volume }
    }
}

impl AudioFilter for VolumeFilter {
    fn name(&self) -> &str {
        "volume"
    }

    fn apply(&mut self, buf: &mut [f32]) {
        let v = self.volume.load(Ordering::Acquire);
        if v == 100 {
            return;
        }
        let scale = v as f32 / 100.0;
        for s in buf.iter_mut() {
            *s *= scale;
        }
    }
}

// ── FilterChain ──────────────────────────────────────────────────────────────

/// Ordered chain of [`AudioFilter`]s applied left-to-right in sequence.
#[derive(Default)]
pub struct FilterChain {
    filters: Vec<Box<dyn AudioFilter>>,
}

impl FilterChain {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a filter to the end of the chain.
    pub fn push(&mut self, f: Box<dyn AudioFilter>) {
        self.filters.push(f);
    }

    /// Apply every filter in order.
    pub fn apply(&mut self, buf: &mut [f32]) {
        for f in self.filters.iter_mut() {
            f.apply(buf);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Number of filters in the chain.
    pub fn len(&self) -> usize {
        self.filters.len()
    }

    /// Names of the filters, in processing order.
    pub fn names(&self) -> Vec<&str> {
        self.filters.iter().map(|f| f.name()).collect()
    }

    /// Propagate the stream format to every filter.
    pub fn configure(&mut self, sample_rate: u32, channels: u8) {
        for f in self.filters.iter_mut() {
            f.set_format(sample_rate, channels);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ones(n: usize) -> Vec<f32> {
        vec![1.0f32; n]
    }

    // VolumeFilter: v=50 halves every sample.
    #[test]
    fn volume_filter_50_halves_buffer() {
        let vol = Arc::new(AtomicU8::new(50));
        let mut f = VolumeFilter::new(Arc::clone(&vol));
        let mut buf = ones(8);
        f.apply(&mut buf);
        for s in &buf {
            assert!((*s - 0.5).abs() < f32::EPSILON, "expected 0.5, got {s}");
        }
    }

    // VolumeFilter: v=100 is a no-op (early return, buffer unchanged).
    #[test]
    fn volume_filter_100_leaves_buffer_unchanged() {
        let vol = Arc::new(AtomicU8::new(100));
        let mut f = VolumeFilter::new(Arc::clone(&vol));
        let mut buf = ones(8);
        f.apply(&mut buf);
        for s in &buf {
            assert!((*s - 1.0).abs() < f32::EPSILON, "expected 1.0, got {s}");
        }
    }

    // VolumeFilter: v=0 zeros the buffer.
    #[test]
    fn volume_filter_0_silences_buffer() {
        let vol = Arc::new(AtomicU8::new(0));
        let mut f = VolumeFilter::new(Arc::clone(&vol));
        let mut buf = ones(4);
        f.apply(&mut buf);
        for s in &buf {
            assert!(s.abs() < f32::EPSILON, "expected 0.0, got {s}");
        }
    }

    // FilterChain: two VolumeFilters at 50 each → 0.25 (0.5 × 0.5 = 0.25).
    #[test]
    fn filter_chain_applies_multiplicatively() {
        let v1 = Arc::new(AtomicU8::new(50));
        let v2 = Arc::new(AtomicU8::new(50));
        let mut chain = FilterChain::new();
        chain.push(Box::new(VolumeFilter::new(Arc::clone(&v1))));
        chain.push(Box::new(VolumeFilter::new(Arc::clone(&v2))));
        let mut buf = ones(4);
        chain.apply(&mut buf);
        for s in &buf {
            assert!(
                (*s - 0.25).abs() < f32::EPSILON,
                "expected 0.25 (0.5×0.5), got {s}"
            );
        }
    }

    // FilterChain is_empty before / after push.
    #[test]
    fn filter_chain_is_empty_reflects_contents() {
        let mut chain = FilterChain::new();
        assert!(chain.is_empty());
        chain.push(Box::new(VolumeFilter::new(Arc::new(AtomicU8::new(100)))));
        assert!(!chain.is_empty());
    }
}
