# playcore

Shared Symphonia-based playback core, extracted from [rmpd](https://github.com/M0Rf30/rmpd) and
[aulos](https://github.com/M0Rf30/aulos). Pure computation over Symphonia and `std`: no audio
device, no async runtime, no network, no configuration format. The applications bring the device
layer and the policy; playcore brings the parts that must behave identically everywhere.

| Module | Contents |
|--------|----------|
| `error` | `Error` / `MediaErrorKind` (the Symphonia error class survives) |
| `format` | the shared Symphonia probe, supported extension / MIME lists, WavPack `.wvc` sibling options |
| `decoder` | `SymphoniaDecoder`: sample-accurate seek, gapless trimming, chained Ogg, HE-AAC output-format priming, raw / converted DSD |
| `source` | `Read` / `Read + Seek` → Symphonia `MediaSource` adapters |
| `dsd` | DoP encoder, DSD-to-PCM rate selection |
| `convert` | `f32` → integer conversion, real-time sample queue, click-free gain ramp |
| `control` | lock-free pause / flush / gain block shared with audio callbacks |
| `crossfade` | equal-power / linear crossfade, MixRamp, crossfade eligibility |
| `fade` | fade-in / fade-out ramp control |
| `filter` | `AudioFilter`, `FilterChain`, `VolumeFilter` |
| `dsp` | equalizer, normalizer / limiter, channel routing |
| `replaygain` | ReplayGain scale computation |
| `resample` | streaming sample-rate converter (feature `resample`, [rubato](https://crates.io/crates/rubato)) |

## Dependencies

The default build depends only on the
[Symphonia fork](https://github.com/M0Rf30/Symphonia/tree/rmpd-next) and `log`. Every consumer must
use the same Symphonia spec so that a single `symphonia-core` is built:

```toml
symphonia = { git = "https://github.com/M0Rf30/Symphonia", branch = "rmpd-next", features = ["all", "dsd"] }
```

The crate builds for `wasm32-unknown-unknown` with and without `resample`.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option.
