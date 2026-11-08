// SPDX-License-Identifier: MIT OR Apache-2.0

//! Symphonia-backed audio decoder: probing, track selection, sample-accurate seeking, chained
//! Ogg handling, gapless trimming, HE-AAC output-format priming and raw/converted DSD.

use crate::error::{Error, Result};
use crate::format::{local_file_format_options, probe};
use crate::source::{ReadSeekSource, StreamSource};
use std::io::{Read, Seek};
use std::path::Path;
use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::CodecParameters;
use symphonia::core::codecs::audio::well_known::CODEC_ID_AAC;
use symphonia::core::codecs::audio::{
    AudioCodecId, AudioDecoder, AudioDecoderOptions, BitOrder, ChannelDataLayout,
};
use symphonia::core::errors::{Error as SymphoniaError, SeekErrorKind};
use symphonia::core::formats::well_known::FORMAT_ID_OGG;
use symphonia::core::formats::{
    FormatId, FormatOptions, FormatReader, MediaInfo, SeekMode, SeekTo, SeekedTo, TrackType,
};
use symphonia::core::io::{MediaSource, MediaSourceStream};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::packet::Packet;
use symphonia::core::units::{Time, TimeBase, Timestamp};
// DSD codec type (from Symphonia with DSD support)
use symphonia::default::formats::CODEC_TYPE_DSD;

pub use symphonia::core::formats::probe::Hint;

/// PCM stream format reported by a decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioFormat {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel count.
    pub channels: u8,
    /// Nominal bit depth. Symphonia decodes to `f32`; 16 is reported for MPD compatibility.
    pub bits_per_sample: u8,
}

/// How a [`SymphoniaDecoder`] is opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderOptions {
    /// Trim the encoder delay / padding frames the demuxer flags on each packet
    /// (`trim_start` / `trim_end`), so the decoded buffers contain exactly the track's frames.
    /// Seek accounting follows the setting. Default: `true`.
    pub gapless: bool,
}

impl Default for DecoderOptions {
    fn default() -> Self {
        Self { gapless: true }
    }
}

/// Symphonia decoder options for the gapless setting: on, the decoder trims the delay/padding
/// the demuxer flags on each packet; off, the decoded audio is left untrimmed.
fn decoder_options(gapless: bool) -> AudioDecoderOptions {
    AudioDecoderOptions::default().gapless(gapless)
}

/// Supplies the current ICY "now playing" title of a remote stream.
type TitleSource = Box<dyn Fn() -> Option<String> + Send>;

/// Symphonia-based audio decoder
pub struct SymphoniaDecoder {
    reader: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    track_id: u32,
    codec_id: AudioCodecId,
    sample_rate: u32,
    channels: Option<u8>,
    total_duration: Option<f64>,
    sample_buf: Vec<f32>,
    sample_pos: usize,
    current_bitrate: Option<u32>,
    time_base: Option<TimeBase>,
    channel_data_layout: Option<ChannelDataLayout>,
    bit_order: Option<BitOrder>,
    uses_pcm_conversion: bool,
    /// ICY "now playing" title source when decoding a remote stream.
    stream_title: Option<TitleSource>,
    /// Whether the decoder trims encoder delay/padding (`packet.trim_start`/`trim_end`).
    gapless: bool,
    /// Set by a `seek` to (or past) the end of the stream: `read` /
    /// `read_dsd_raw` then report end-of-stream, whatever position the
    /// demuxer was left at. Cleared by the next seek.
    ended: bool,
    /// Pending sample-accurate seek: after a seek the demuxer lands at or before the
    /// requested position (e.g. a packet boundary, or a seek pre-roll in Matroska), and
    /// decoded frames before the requested timestamp are discarded.
    seek_skip: Option<SeekSkip>,
}

impl SymphoniaDecoder {
    /// Open a local file. A WavPack `.wv` picks up a sibling `.wvc` correction file (hybrid
    /// lossless).
    pub fn open(path: &Path, options: DecoderOptions) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| Error::other(format!("Failed to open file: {e}")))?;
        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }
        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        Self::from_media_source(mss, hint, local_file_format_options(path), options.gapless)
    }

    /// Open any Symphonia [`MediaSource`] (an HTTP reader, a live stream, an in-memory buffer).
    /// `hint` carries the extension / MIME type of the resource when known. Use
    /// [`ReadSeekSource`] or [`StreamSource`] to adapt plain readers.
    pub fn open_source(
        source: Box<dyn MediaSource>,
        hint: Hint,
        options: DecoderOptions,
    ) -> Result<Self> {
        let mss = MediaSourceStream::new(source, Default::default());
        Self::from_media_source(mss, hint, FormatOptions::default(), options.gapless)
    }

    /// Open a seekable `Read + Seek` reader of known (or unknown) length, with an optional file
    /// extension hint. Shorthand for [`Self::open_source`] with a [`ReadSeekSource`].
    pub fn open_reader<R>(
        reader: R,
        byte_len: Option<u64>,
        hint_extension: Option<&str>,
        options: DecoderOptions,
    ) -> Result<Self>
    where
        R: Read + Seek + Send + Sync + 'static,
    {
        let mut hint = Hint::new();
        if let Some(ext) = hint_extension {
            hint.with_extension(ext);
        }
        Self::open_source(
            Box::new(ReadSeekSource::new(reader, byte_len)),
            hint,
            options,
        )
    }

    /// Open a non-seekable live stream (internet radio). Shorthand for [`Self::open_source`]
    /// with a [`StreamSource`]; seeking such a decoder fails with "Not seekable".
    pub fn open_stream<R>(
        reader: R,
        hint_extension: Option<&str>,
        mime_type: Option<&str>,
        options: DecoderOptions,
    ) -> Result<Self>
    where
        R: Read + Send + Sync + 'static,
    {
        let mut hint = Hint::new();
        if let Some(ext) = hint_extension {
            hint.with_extension(ext);
        }
        if let Some(mime) = mime_type {
            hint.mime_type(mime);
        }
        Self::open_source(Box::new(StreamSource::new(reader)), hint, options)
    }

    /// Install the callback [`Self::stream_title`] reads (e.g. the ICY metadata handle of a
    /// remote stream).
    #[must_use]
    pub fn with_stream_title(
        mut self,
        title: impl Fn() -> Option<String> + Send + 'static,
    ) -> Self {
        self.stream_title = Some(Box::new(title));
        self
    }

    /// Shared probe + track-selection + decoder-construction logic.
    fn from_media_source(
        mss: MediaSourceStream<'static>,
        hint: Hint,
        format_opts: FormatOptions,
        gapless: bool,
    ) -> Result<Self> {
        // Probe the media source
        let reader = probe()
            .probe(&hint, mss, format_opts, MetadataOptions::default())
            .map_err(|e| Error::from_symphonia_context("Failed to probe format", e))?;

        // Find the default audio track
        let track = reader
            .default_track(TrackType::Audio)
            .ok_or_else(|| Error::other("No audio tracks found"))?;

        let track_id = track.id;
        let time_base = track.time_base;

        // Get the audio codec parameters.
        let audio = match track.codec_params.as_ref() {
            Some(CodecParameters::Audio(audio)) => audio,
            _ => return Err(Error::other("No audio codec parameters")),
        };

        // Store codec id for DSD detection.
        let codec_id = audio.codec;

        let sample_rate = audio
            .sample_rate
            .ok_or_else(|| Error::other("Sample rate not available"))?;

        // Channels might not be available until after decoding starts.
        let channels = audio.channels.as_ref().map(|ch| ch.count() as u8);

        // DSD metadata if available.
        let channel_data_layout = audio.channel_data_layout;
        let bit_order = audio.bit_order;

        // Calculate total duration. A chained Ogg stream reports the duration of the whole
        // chain at the media level; the track only describes the first link. Otherwise use the
        // track frame count and timebase.
        let total_duration = chain_duration_secs(reader.format_info().format, reader.media_info())
            .or_else(|| match (track.num_frames, time_base) {
                (Some(n_frames), Some(tb)) => tb
                    .calc_time(Timestamp::new(n_frames as i64))
                    .map(|t| t.as_secs_f64()),
                _ => None,
            });

        // Create decoder in pass-through mode (no PCM conversion).
        // PCM conversion can be enabled later if needed.
        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(audio, &decoder_options(gapless))
            .map_err(|e| Error::from_symphonia_context("Failed to create decoder", e))?;

        // The decoder may report a different output rate than the container (e.g. explicitly
        // signalled HE-AAC decodes at twice the AAC core rate the container declares). Raw DSD
        // keeps the DSD rate (DoP needs it).
        let sample_rate = if codec_id == CODEC_TYPE_DSD {
            sample_rate
        } else {
            decoder.codec_params().sample_rate.unwrap_or(sample_rate)
        };

        let mut decoder = Self {
            reader,
            decoder,
            track_id,
            codec_id,
            sample_rate,
            channels,
            total_duration,
            sample_buf: Vec::new(),
            sample_pos: 0,
            current_bitrate: None,
            time_base,
            channel_data_layout,
            bit_order,
            uses_pcm_conversion: false,
            stream_title: None,
            gapless,
            ended: false,
            seek_skip: None,
        };

        // Some containers don't declare the channel count in the codec
        // header; it is only known once the first packet is decoded. Resolve
        // it eagerly here rather than letting `format()`/`channels()` default
        // to stereo before the first real `read()` call (PLAY-05) — that
        // default would otherwise get latched into the device/output config
        // for the whole track. AAC is always primed: implicitly signalled
        // HE-AAC (SBR/PS found only in the bitstream, common in ADTS radio
        // streams) doubles the output rate and may turn a mono core into
        // stereo, which only the first decoded buffer reveals.
        if decoder.channels.is_none() || codec_id == CODEC_ID_AAC {
            decoder.prime_output_format()?;
        }

        Ok(decoder)
    }

    /// Decode packets until the first non-empty buffer (or the end of the stream), take the
    /// output channel count and sample rate from it, and buffer its audio (rather than
    /// discarding it) so the first real `read()` call still sees it.
    fn prime_output_format(&mut self) -> Result<()> {
        loop {
            let packet = match self.reader.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => return Ok(()), // EOS with no decodable audio.
                Err(SymphoniaError::ResetRequired) => {
                    self.reinit_after_reset().map_err(|e| {
                        Error::from_symphonia_context(
                            "Failed to reinitialise after stream change",
                            e,
                        )
                    })?;
                    continue;
                }
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    return Ok(());
                }
                Err(e) => {
                    return Err(Error::from_symphonia_context("Failed to read packet", e));
                }
            };

            if packet.track_id != self.track_id {
                continue;
            }

            let decoded = match self.decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(e) => {
                    return Err(Error::from_symphonia_context("Failed to decode packet", e));
                }
            };

            if decoded.frames() == 0 {
                continue;
            }

            self.channels = Some(decoded.spec().channels().count() as u8);
            // A pass-through DSD buffer's `AudioSpec` rate is `dsd_rate / 8` (one frame = one
            // byte per channel), but the decoder's rate must stay the DSD rate (DoP needs it).
            if !(self.codec_id == CODEC_TYPE_DSD && !self.uses_pcm_conversion) {
                self.sample_rate = decoded.spec().rate();
            }
            decoded.copy_to_vec_interleaved(&mut self.sample_buf);
            self.sample_pos = 0;
            return Ok(());
        }
    }

    /// The current ICY "now playing" title for a remote stream, if any has been received (see
    /// [`Self::with_stream_title`]). Returns `None` for local files or before the first
    /// metadata block arrives.
    #[must_use]
    pub fn stream_title(&self) -> Option<String> {
        self.stream_title.as_ref().and_then(|title| title())
    }

    /// Check if this is a DSD file
    pub fn is_dsd(&self) -> bool {
        self.codec_id == CODEC_TYPE_DSD
    }

    /// Enable PCM conversion for DSD (can be called multiple times with different rates)
    pub fn enable_pcm_conversion(&mut self, output_rate: u32) -> Result<()> {
        if self.codec_id != CODEC_TYPE_DSD {
            return Ok(()); // Not DSD, nothing to do
        }

        // If already enabled at the same rate, nothing to do
        if self.uses_pcm_conversion && self.sample_rate == output_rate {
            return Ok(());
        }

        // Get the current track's audio codec parameters.
        let track = self
            .reader
            .tracks()
            .iter()
            .find(|t| t.id == self.track_id)
            .ok_or_else(|| Error::other("Track not found"))?;

        let audio = match track.codec_params.as_ref() {
            Some(CodecParameters::Audio(audio)) => audio,
            _ => return Err(Error::other("No audio codec parameters")),
        };
        let input_rate = audio
            .sample_rate
            .ok_or_else(|| Error::other("Sample rate not available"))?;

        // Clone params and add PCM conversion mode via extra_data
        let mut params_with_pcm = audio.clone();
        params_with_pcm.extra_data = Some(output_rate.to_le_bytes().to_vec().into_boxed_slice());

        log::info!(
            "enabling DSD-to-PCM conversion: {} Hz DSD -> {} Hz PCM",
            input_rate,
            output_rate
        );

        // Create new decoder with PCM conversion
        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(&params_with_pcm, &decoder_options(self.gapless))
            .map_err(|e| Error::from_symphonia_context("Failed to create PCM decoder", e))?;

        // Get actual output sample rate from decoder
        let actual_sample_rate = decoder
            .codec_params()
            .sample_rate
            .ok_or_else(|| Error::other("Decoder sample rate not available"))?;

        // Replace decoder
        self.decoder = decoder;
        self.sample_rate = actual_sample_rate;
        self.uses_pcm_conversion = true;

        Ok(())
    }

    pub fn read(&mut self, buffer: &mut [f32]) -> Result<usize> {
        if self.ended {
            return Ok(0);
        }
        let mut samples_written = 0;

        while samples_written < buffer.len() {
            // Drain any buffered interleaved samples first.
            if self.sample_pos < self.sample_buf.len() {
                let available = self.sample_buf.len() - self.sample_pos;
                let to_copy = (buffer.len() - samples_written).min(available);
                buffer[samples_written..samples_written + to_copy]
                    .copy_from_slice(&self.sample_buf[self.sample_pos..self.sample_pos + to_copy]);
                samples_written += to_copy;
                self.sample_pos += to_copy;
                if samples_written >= buffer.len() {
                    break;
                }
            }

            // Read the next packet.
            let packet = match self.reader.next_packet() {
                Ok(Some(packet)) => packet,
                Ok(None) => break, // End of stream.
                Err(SymphoniaError::ResetRequired) => {
                    // A new link of a chained stream (or a changed stream): the track list
                    // must be re-read and the decoder re-created.
                    self.reinit_after_reset().map_err(|e| {
                        Error::from_symphonia_context(
                            "Failed to reinitialise after stream change",
                            e,
                        )
                    })?;
                    continue;
                }
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(e) => {
                    log::error!("failed to read packet: {}", e);
                    return Err(Error::from_symphonia_context("Failed to read packet", e));
                }
            };

            // Skip packets from other tracks.
            if packet.track_id != self.track_id {
                continue;
            }

            // Calculate instantaneous bitrate from the packet. Use the full block
            // duration (not the trimmed `dur`): a packet whose head/tail was trimmed
            // after a seek or at stream edges still carries all its bytes, so dividing
            // by the shortened duration would produce transient spikes.
            if let Some(tb) = self.time_base
                && let Some(time) = tb.calc_time(Timestamp::new(packet.block_dur().get() as i64))
            {
                let duration_secs = time.as_secs_f64();
                if duration_secs > 0.0 {
                    let bitrate_bps = (packet.data.len() as f64 * 8.0) / duration_secs;
                    self.current_bitrate = Some((bitrate_bps / 1000.0) as u32);
                }
            }

            // Decode the packet.
            let decoded = match self.decoder.decode(&packet) {
                Ok(decoded) => decoded,
                // A corrupt packet (`DecodeError`) or a transient I/O hiccup (`IoError`) costs
                // only that packet; keep decoding.
                Err(SymphoniaError::DecodeError(_) | SymphoniaError::IoError(_)) => continue,
                // The stream changed under the decoder: rebuild it, then carry on.
                Err(SymphoniaError::ResetRequired) => {
                    self.reinit_after_reset().map_err(|e| {
                        Error::from_symphonia_context(
                            "Failed to reinitialise after stream change",
                            e,
                        )
                    })?;
                    continue;
                }
                Err(e) => {
                    return Err(Error::from_symphonia_context("Failed to decode packet", e));
                }
            };

            // For DSD with PCM conversion, the decoder must return F32.
            if self.uses_pcm_conversion && !matches!(decoded, GenericAudioBufferRef::F32(_)) {
                log::error!("DSD-to-PCM decoder returned a non-F32 buffer");
                return Err(Error::other(
                    "DSD decoder returned wrong sample format".to_owned(),
                ));
            }

            // Skip empty packets (can happen with metadata or padding).
            if decoded.frames() == 0 {
                continue;
            }

            // Update channels if not yet known.
            if self.channels.is_none() {
                self.channels = Some(decoded.spec().channels().count() as u8);
            }

            // Copy decoded audio as interleaved f32 into the reusable buffer.
            let frames = decoded.frames();
            let rate = decoded.spec().rate();
            let channel_count = decoded.spec().channels().count();
            decoded.copy_to_vec_interleaved(&mut self.sample_buf);
            // After a seek, drop the frames in front of the requested position: the demuxer
            // only lands on a packet at or before it (Matroska MP3/AAC/Vorbis even start a
            // seek pre-roll earlier).
            let skip_frames = self.seek_skip_frames(&packet, frames, rate);
            self.sample_pos = (skip_frames * channel_count).min(self.sample_buf.len());
        }

        Ok(samples_written)
    }

    pub fn seek(&mut self, position: f64) -> Result<()> {
        if position < 0.0 {
            return Err(Error::other("Invalid seek position"));
        }

        let time = Time::try_from_secs_f64(position)
            .ok_or_else(|| Error::other("Invalid seek position"))?;

        // A seek to (or past) the end of a stream of known length just ends
        // the song, like MPD (`Player::SeekDecoder` clamps to the song length
        // and the decoder then hits end-of-file). Demuxers disagree on what a
        // target at exactly the end is: isomp4, ape and dsf/dff refuse it as
        // out-of-range, flac runs into the end of the file while searching —
        // and either way the reader is left wherever it was. So on those
        // errors report end of stream from `read` ourselves instead of
        // failing the seek.
        let at_end = self.total_duration.is_some_and(|d| position >= d);

        // The demuxer may need a reset to complete the seek (a time seek into another link of
        // a chained Ogg stream): re-read the tracks, re-create the decoder, repeat the seek.
        self.seek_skip = None;
        let seeked = match seek_with_resets(self, time, MAX_SEEK_RESETS) {
            Ok(seeked) => Some(seeked),
            Err(SymphoniaError::SeekError(SeekErrorKind::OutOfRange)) if at_end => None,
            Err(SymphoniaError::IoError(e))
                if at_end && e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                None
            }
            // MPD's text for an unseekable source (`DecoderControl::Seek`:
            // `throw std::runtime_error("Not seekable")`), which `seek` /
            // `seekcur` now show the client verbatim.
            Err(SymphoniaError::SeekError(SeekErrorKind::Unseekable)) => {
                return Err(Error::other("Not seekable"));
            }
            Err(e) => return Err(Error::from_symphonia_context("Seek failed", e)),
        };

        self.decoder.reset();
        self.sample_buf.clear();
        self.sample_pos = 0;
        self.ended = at_end;

        // Raw (pass-through) DSD is not decoded here, so there is nothing to discard.
        let raw_dsd = self.codec_id == CODEC_TYPE_DSD && !self.uses_pcm_conversion;
        self.seek_skip = seeked.filter(|_| !raw_dsd && !at_end).and_then(|s| {
            SeekSkip::new(s.required_ts, s.actual_ts, self.time_base, self.sample_rate)
        });

        Ok(())
    }

    /// Number of leading frames of the just-decoded `packet` to discard to honour a pending
    /// sample-accurate seek, updating (and eventually clearing) the seek state.
    fn seek_skip_frames(&mut self, packet: &Packet, frames: usize, rate: u32) -> usize {
        let Some(tb) = self.time_base else {
            self.seek_skip = None;
            return 0;
        };
        let Some(skip) = self.seek_skip.as_mut() else {
            return 0;
        };
        // `pts` is the start of the decoded block. With gapless on, the decoder has already
        // removed the `trim_start` frames (encoder delay / pre-roll flagged by the demuxer)
        // from the buffer, so the first frame left sits `trim_start` after `pts`; with it off
        // nothing is trimmed and the buffer starts at `pts`.
        let valid_start = if self.gapless {
            packet.pts.saturating_add(packet.trim_start)
        } else {
            packet.pts
        };
        let (n, done) = skip.advance(valid_start, frames, tb, rate);
        if done {
            self.seek_skip = None;
        }
        n
    }

    /// Re-read the track list after the demuxer returned `ResetRequired` (a new link of a
    /// chained Ogg stream, ...) and re-create the decoder for the default audio track.
    fn reinit_after_reset(&mut self) -> std::result::Result<(), SymphoniaError> {
        let (track_id, time_base, audio) =
            {
                let track = self.reader.default_track(TrackType::Audio).ok_or(
                    SymphoniaError::Unsupported("no audio track after stream reset"),
                )?;
                let audio = match track.codec_params.as_ref() {
                    Some(CodecParameters::Audio(audio)) => audio.clone(),
                    _ => {
                        return Err(SymphoniaError::Unsupported(
                            "no audio codec parameters after stream reset",
                        ));
                    }
                };
                (track.id, track.time_base, audio)
            };

        let is_dsd = audio.codec == CODEC_TYPE_DSD;
        let keep_pcm = self.uses_pcm_conversion && is_dsd;
        let mut params = audio.clone();
        if keep_pcm {
            // `sample_rate` is the PCM output rate while DSD-to-PCM conversion is active.
            params.extra_data = Some(self.sample_rate.to_le_bytes().to_vec().into_boxed_slice());
        }
        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(&params, &decoder_options(self.gapless))?;

        // Pass-through DSD keeps the container (DSD) rate, which DoP needs.
        let new_rate = if keep_pcm {
            self.sample_rate
        } else if is_dsd {
            audio.sample_rate.unwrap_or(self.sample_rate)
        } else {
            decoder
                .codec_params()
                .sample_rate
                .or(audio.sample_rate)
                .unwrap_or(self.sample_rate)
        };
        let new_channels = audio.channels.as_ref().map(|c| c.count() as u8);
        if new_rate != self.sample_rate || (new_channels.is_some() && new_channels != self.channels)
        {
            // The output was opened for the first link's format and cannot follow.
            log::warn!(
                "stream format changed mid-stream: {} Hz/{:?} ch -> {} Hz/{:?} ch",
                self.sample_rate,
                self.channels,
                new_rate,
                new_channels
            );
        }

        self.decoder = decoder;
        self.track_id = track_id;
        self.time_base = time_base;
        self.codec_id = audio.codec;
        self.sample_rate = new_rate;
        self.channels = new_channels;
        self.channel_data_layout = audio.channel_data_layout;
        self.bit_order = audio.bit_order;
        self.uses_pcm_conversion = keep_pcm;
        self.sample_buf.clear();
        self.sample_pos = 0;
        self.seek_skip = None;
        self.current_bitrate = None;
        Ok(())
    }

    pub fn format(&self) -> AudioFormat {
        AudioFormat {
            sample_rate: self.sample_rate,
            channels: self.channels.unwrap_or(2), // Default to stereo if not yet known
            bits_per_sample: 16,
        }
    }

    pub fn duration(&self) -> Option<f64> {
        self.total_duration
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u8 {
        self.channels.unwrap_or(2) // Default to stereo if not yet known
    }

    /// Get the current instantaneous bitrate in kbps (for VBR files this changes during playback)
    pub fn current_bitrate(&self) -> Option<u32> {
        self.current_bitrate
    }

    /// Get channel data layout (planar vs interleaved) for DSD files
    pub fn channel_data_layout(&self) -> Option<ChannelDataLayout> {
        self.channel_data_layout
    }

    /// Get bit order (LSB-first vs MSB-first) for DSD files
    pub fn bit_order(&self) -> Option<BitOrder> {
        self.bit_order
    }

    /// Read raw DSD data (for DoP encoding).
    /// Returns raw DSD bytes without conversion.
    pub fn read_dsd_raw(&mut self, buffer: &mut Vec<u8>) -> Result<usize> {
        buffer.clear();
        if self.ended {
            return Ok(0);
        }

        let packet = next_dsd_packet(
            self,
            |d| d.track_id,
            |d| match d.reader.next_packet() {
                Ok(Some(packet)) => Ok(DsdPacketEvent::Packet {
                    track_id: packet.track_id,
                    packet,
                }),
                Ok(None) => Ok(DsdPacketEvent::End),
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    Ok(DsdPacketEvent::End)
                }
                Err(SymphoniaError::ResetRequired) => Ok(DsdPacketEvent::Reset),
                Err(e) => Err(Error::from_symphonia_context(
                    "Failed to read DSD packet",
                    e,
                )),
            },
            |d| {
                d.reinit_after_reset().map_err(|e| {
                    Error::from_symphonia_context("Failed to reinitialise after stream change", e)
                })
            },
        )?;

        let Some(packet) = packet else {
            return Ok(0);
        };

        // For DSD, the packet buffer contains raw DSD data.
        // Copy it directly without decoding.
        buffer.extend_from_slice(&packet.data);

        Ok(buffer.len())
    }
}

/// Upper bound on back-to-back demuxer resets while looking for the next DSD packet; a reader
/// that keeps asking for resets is broken.
const MAX_CONSECUTIVE_DSD_RESETS: usize = 1024;

enum DsdPacketEvent<T> {
    Packet { track_id: u32, packet: T },
    Reset,
    End,
}

/// Pull the next packet of the current DSD track, handling resets via `reset` (which may
/// change the track id, hence it is re-read from `state` for every packet). Iterative, so a
/// long run of foreign-track packets cannot overflow the stack.
fn next_dsd_packet<S, T>(
    state: &mut S,
    track_id: impl Fn(&S) -> u32,
    mut next_event: impl FnMut(&mut S) -> Result<DsdPacketEvent<T>>,
    mut reset: impl FnMut(&mut S) -> Result<()>,
) -> Result<Option<T>> {
    let mut consecutive_resets = 0;

    loop {
        match next_event(state)? {
            DsdPacketEvent::Packet {
                track_id: packet_track_id,
                packet,
            } => {
                consecutive_resets = 0;
                if packet_track_id == track_id(state) {
                    return Ok(Some(packet));
                }
            }
            DsdPacketEvent::Reset => {
                consecutive_resets += 1;
                if consecutive_resets > MAX_CONSECUTIVE_DSD_RESETS {
                    return Err(Error::other("Too many consecutive DSD decoder resets"));
                }
                reset(state)?;
            }
            DsdPacketEvent::End => return Ok(None),
        }
    }
}

/// How often a seek is repeated after the demuxer returned `ResetRequired` (a time seek that
/// crosses into another link of a chained Ogg stream switches links and asks for a reset; the
/// repeated seek then completes inside the new link). One reset is the normal case; the
/// second retry is slack. A reader still asking for resets after that is broken.
pub(crate) const MAX_SEEK_RESETS: u32 = 2;

/// Something that can seek and be rebuilt after a reset (the decoder; a mock in tests).
pub(crate) trait ResettableSeek {
    /// Seek to `time` on the current default audio track.
    fn try_seek(&mut self, time: Time) -> std::result::Result<SeekedTo, SymphoniaError>;
    /// Re-read the tracks and re-create the decoder after `ResetRequired`.
    fn rebuild_after_reset(&mut self) -> std::result::Result<(), SymphoniaError>;
}

impl ResettableSeek for SymphoniaDecoder {
    fn try_seek(&mut self, time: Time) -> std::result::Result<SeekedTo, SymphoniaError> {
        // The track id is read on every attempt: it changes when the reset switched links.
        self.reader.seek(
            SeekMode::Accurate,
            SeekTo::Time {
                time,
                track_id: Some(self.track_id),
            },
        )
    }

    fn rebuild_after_reset(&mut self) -> std::result::Result<(), SymphoniaError> {
        self.reinit_after_reset()
    }
}

/// Seek, and on `ResetRequired` rebuild and repeat the same seek, at most `max_resets` times.
/// Any other outcome (success or error) is returned as is.
pub(crate) fn seek_with_resets<T: ResettableSeek + ?Sized>(
    target: &mut T,
    time: Time,
    max_resets: u32,
) -> std::result::Result<SeekedTo, SymphoniaError> {
    let mut resets = 0;
    loop {
        match target.try_seek(time) {
            Err(SymphoniaError::ResetRequired) if resets < max_resets => {
                resets += 1;
                target.rebuild_after_reset()?;
            }
            other => return other,
        }
    }
}

/// Duration in seconds of a whole chained Ogg stream, if the reader describes one.
///
/// symphonia's Ogg reader locates the links of a seekable chained stream up front and then
/// publishes the chain-level duration (sum of all links) in `MediaInfo` with a nanosecond
/// timebase, while `Track` only describes the current link. For anything else (including a
/// single-link Ogg stream, whose media info just mirrors the track) this returns `None` so the
/// per-track duration is used.
#[must_use]
pub fn chain_duration_secs(format: FormatId, media_info: &MediaInfo) -> Option<f64> {
    if format != FORMAT_ID_OGG {
        return None;
    }
    let tb = media_info.time_base?;
    if tb.numer.get() != 1 || tb.denom.get() != 1_000_000_000 {
        return None;
    }
    tb.calc_duration(media_info.duration?)
        .map(|t| t.as_secs_f64())
}

/// Convert a span of `ticks` of timebase `tb` into frames at `rate` Hz (rounded to nearest).
pub(crate) fn ticks_to_frames(ticks: u64, tb: TimeBase, rate: u32) -> u64 {
    let denom = u128::from(tb.denom.get());
    let num = u128::from(ticks) * u128::from(tb.numer.get()) * u128::from(rate);
    u64::try_from((num + denom / 2) / denom).unwrap_or(u64::MAX)
}

/// Frames to drop after a seek so playback starts exactly at the requested timestamp.
///
/// `FormatReader::seek` only lands on a packet at or before the target (`actual_ts <=
/// required_ts`); Matroska MP3/AAC/Vorbis additionally start a 200 ms seek pre-roll earlier. The
/// decoder must decode from there (it needs the pre-roll to warm up) and the frames before
/// `required_ts` are discarded, rather than trusting `actual_ts`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SeekSkip {
    /// The timestamp playback must start at, in the track timebase.
    required: Timestamp,
    /// Upper bound of frames to discard: the announced distance plus one second. Protects
    /// against a demuxer with unreliable packet timestamps eating the whole stream.
    budget: u64,
}

impl SeekSkip {
    /// `None` when nothing has to be discarded (landed exactly on, or after, the target).
    fn new(
        required: Timestamp,
        actual: Timestamp,
        tb: Option<TimeBase>,
        rate: u32,
    ) -> Option<Self> {
        let tb = tb?;
        let gap = required.duration_from(actual)?;
        if gap.is_zero() {
            return None;
        }
        let budget = ticks_to_frames(gap.get(), tb, rate).saturating_add(u64::from(rate));
        Some(Self { required, budget })
    }

    /// Account for one decoded buffer of `decoded_frames` frames whose first valid frame is at
    /// `valid_start`. Returns how many leading frames to discard and whether the target has
    /// been reached (so no more skipping is needed).
    fn advance(
        &mut self,
        valid_start: Timestamp,
        decoded_frames: usize,
        tb: TimeBase,
        rate: u32,
    ) -> (usize, bool) {
        let gap = match self.required.duration_from(valid_start) {
            Some(gap) if !gap.is_zero() => gap,
            // At or past the target.
            _ => return (0, true),
        };
        let gap_frames = ticks_to_frames(gap.get(), tb, rate);
        let skip = gap_frames.min(decoded_frames as u64).min(self.budget);
        self.budget -= skip;
        let reached = gap_frames <= decoded_frames as u64 || self.budget == 0;
        (skip as usize, reached)
    }
}

/// Object-safe wrapper trait for the ordinary PCM decode path. DSD-specific methods
/// (`is_dsd` / `enable_pcm_conversion` / `read_dsd_raw` / ...) are deliberately not part of this
/// trait, so DSD/DoP handling always goes through the concrete [`SymphoniaDecoder`] type.
pub trait Decoder: Send {
    /// Fill `buffer` with interleaved `f32` samples; returns the number written (0 = end).
    fn read(&mut self, buffer: &mut [f32]) -> Result<usize>;
    /// Seek to `position` seconds.
    fn seek(&mut self, position: f64) -> Result<()>;
    /// Current stream format.
    fn format(&self) -> AudioFormat;
    /// Total duration in seconds, if known.
    fn duration(&self) -> Option<f64>;
}

impl Decoder for SymphoniaDecoder {
    fn read(&mut self, buffer: &mut [f32]) -> Result<usize> {
        self.read(buffer)
    }
    fn seek(&mut self, position: f64) -> Result<()> {
        self.seek(position)
    }
    fn format(&self) -> AudioFormat {
        self.format()
    }
    fn duration(&self) -> Option<f64> {
        self.duration()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A 16-bit PCM WAV of `frames` stereo frames at 8 kHz; sample `i` of the left channel is
    /// `i % 1000` (so the position after a seek is recoverable), the right channel is `-left`.
    fn wav(frames: u32) -> Vec<u8> {
        let rate = 8000u32;
        let data_len = frames * 4;
        let mut v = Vec::new();
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&(36 + data_len).to_le_bytes());
        v.extend_from_slice(b"WAVEfmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&2u16.to_le_bytes());
        v.extend_from_slice(&rate.to_le_bytes());
        v.extend_from_slice(&(rate * 4).to_le_bytes());
        v.extend_from_slice(&4u16.to_le_bytes());
        v.extend_from_slice(&16u16.to_le_bytes());
        v.extend_from_slice(b"data");
        v.extend_from_slice(&data_len.to_le_bytes());
        for i in 0..frames {
            let l = (i % 1000) as i16;
            v.extend_from_slice(&l.to_le_bytes());
            v.extend_from_slice(&(-l).to_le_bytes());
        }
        v
    }

    fn open_wav(frames: u32) -> SymphoniaDecoder {
        let mut hint = Hint::new();
        hint.with_extension("wav");
        let len = wav(frames);
        let n = len.len() as u64;
        SymphoniaDecoder::open_source(
            Box::new(ReadSeekSource::new(Cursor::new(len), Some(n))),
            hint,
            DecoderOptions::default(),
        )
        .unwrap()
    }

    fn read_all(decoder: &mut SymphoniaDecoder) -> Vec<f32> {
        let mut out = Vec::new();
        let mut buf = vec![0.0f32; 1024];
        loop {
            let n = decoder.read(&mut buf).unwrap();
            if n == 0 {
                return out;
            }
            out.extend_from_slice(&buf[..n]);
        }
    }

    #[test]
    fn wav_decodes_with_the_reported_format_and_duration() {
        let mut d = open_wav(8000);
        assert_eq!(
            d.format(),
            AudioFormat {
                sample_rate: 8000,
                channels: 2,
                bits_per_sample: 16
            }
        );
        assert!((d.duration().unwrap() - 1.0).abs() < 1e-6);
        assert!(!d.is_dsd());
        assert!(d.stream_title().is_none());
        let samples = read_all(&mut d);
        assert_eq!(samples.len(), 16_000);
        // The left channel carries the ramp, the right its negation.
        assert!((samples[2 * 500] - 500.0 / 32768.0).abs() < 1e-6);
        assert!((samples[2 * 500 + 1] + 500.0 / 32768.0).abs() < 1e-6);
        // End of stream is sticky.
        let mut buf = [0.0f32; 16];
        assert_eq!(d.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn seek_is_sample_accurate() {
        let mut d = open_wav(8000);
        d.seek(0.5).unwrap();
        let samples = read_all(&mut d);
        assert_eq!(samples.len(), 2 * 4000);
        // Frame 4000 has left value 4000 % 1000 = 0, frame 4001 has 1.
        assert!(samples[0].abs() < 1e-6);
        assert!((samples[2] - 1.0 / 32768.0).abs() < 1e-6);
        // Seeking backwards works too.
        d.seek(0.0).unwrap();
        assert_eq!(read_all(&mut d).len(), 16_000);
    }

    #[test]
    fn seek_to_or_past_the_end_ends_the_stream_instead_of_failing() {
        let mut d = open_wav(8000);
        d.seek(1.0).unwrap();
        assert!(read_all(&mut d).is_empty());
        // A later seek revives the decoder.
        d.seek(0.75).unwrap();
        assert_eq!(read_all(&mut d).len(), 2 * 2000);
        d.seek(5.0).unwrap();
        assert!(read_all(&mut d).is_empty());
    }

    #[test]
    fn invalid_seek_positions_are_rejected() {
        let mut d = open_wav(800);
        assert_eq!(d.seek(-1.0).unwrap_err().message(), "Invalid seek position");
        assert_eq!(
            d.seek(f64::NAN).unwrap_err().message(),
            "Invalid seek position"
        );
    }

    #[test]
    fn live_stream_is_not_seekable() {
        let mut hint = Hint::new();
        hint.with_extension("wav");
        // Larger than the stream buffer, so a backward seek cannot be served from it.
        let frames = 200_000;
        let mut d = SymphoniaDecoder::open_source(
            Box::new(StreamSource::new(Cursor::new(wav(frames)))),
            hint,
            DecoderOptions::default(),
        )
        .unwrap();
        let mut buf = vec![0.0f32; 1024];
        for _ in 0..(frames as usize * 2 / 1024) {
            assert!(d.read(&mut buf).unwrap() > 0);
        }
        let err = d.seek(0.0).unwrap_err();
        assert_eq!(err.media_kind(), Some(crate::error::MediaErrorKind::Seek));
    }

    #[test]
    fn stream_title_comes_from_the_installed_callback() {
        let d = open_wav(80).with_stream_title(|| Some("Artist - Song".to_owned()));
        assert_eq!(d.stream_title().as_deref(), Some("Artist - Song"));
    }

    #[test]
    fn garbage_fails_to_probe_with_a_media_error() {
        let junk = vec![0x55u8; 4096];
        let err =
            SymphoniaDecoder::open_reader(Cursor::new(junk), None, None, DecoderOptions::default())
                .err()
                .expect("junk must not open");
        assert!(err.message().starts_with("Failed to probe format"));
        assert!(err.media_kind().is_some());
    }

    #[test]
    fn missing_file_is_a_plain_error() {
        let err = SymphoniaDecoder::open(
            Path::new("/nonexistent/definitely/missing.flac"),
            DecoderOptions::default(),
        )
        .err()
        .expect("missing file must not open");
        assert!(err.message().starts_with("Failed to open file"));
        assert_eq!(err.media_kind(), None);
    }

    /// Mono 44.1 kHz, 1 s sine encoded with LAME (carries a LAME tag with encoder
    /// delay/padding, so the MP3 demuxer reports `delay`, `padding` and `num_frames`).
    fn gapless_mp3() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gapless_lame.mp3")
    }

    /// `(num_frames, delay, padding)` of the fixture's audio track, per the demuxer.
    fn gapless_track_info() -> (u64, u64, u64) {
        let mss = MediaSourceStream::new(
            Box::new(std::fs::File::open(gapless_mp3()).unwrap()),
            Default::default(),
        );
        let mut hint = Hint::new();
        hint.with_extension("mp3");
        let reader = probe()
            .probe(
                &hint,
                mss,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .unwrap();
        let track = reader.default_track(TrackType::Audio).unwrap();
        (
            track.num_frames.expect("num_frames"),
            u64::from(track.delay.expect("delay")),
            u64::from(track.padding.expect("padding")),
        )
    }

    fn decoded_frames(gapless: bool) -> u64 {
        let mut decoder =
            SymphoniaDecoder::open(&gapless_mp3(), DecoderOptions { gapless }).unwrap();
        let channels = usize::from(decoder.channels());
        (read_all(&mut decoder).len() / channels) as u64
    }

    #[test]
    fn gapless_on_yields_exactly_the_track_frames() {
        let (n_frames, delay, padding) = gapless_track_info();
        assert!(delay > 0 && padding > 0, "fixture lacks LAME delay/padding");
        assert_eq!(decoded_frames(true), n_frames);
    }

    #[test]
    fn gapless_off_yields_the_untrimmed_frames() {
        let (n_frames, delay, padding) = gapless_track_info();
        assert_eq!(decoded_frames(false), n_frames + delay + padding);
    }

    /// A seek must land on the same audio whether or not the decoder trims: the frames after
    /// the seek target are identical, so `SeekSkip` is accounted against the trimmed output.
    #[test]
    fn seek_lands_on_the_target_with_and_without_gapless() {
        let target = 0.5f64;
        let read_rest = |gapless: bool| {
            let mut decoder =
                SymphoniaDecoder::open(&gapless_mp3(), DecoderOptions { gapless }).unwrap();
            decoder.seek(target).unwrap();
            read_all(&mut decoder).len() as u64
        };
        let (n_frames, _, padding) = gapless_track_info();
        let expected = n_frames - (target * 44_100.0) as u64;
        // Gapless on: exactly the frames from the target to the trimmed end.
        assert_eq!(read_rest(true), expected);
        // Off: the same span plus only the trailing padding (the delay lies before the target).
        assert_eq!(read_rest(false), expected + padding);
    }

    #[test]
    fn dsd_packet_skip_loop_handles_many_events() {
        let mut skipped = 100_000;
        let packet = next_dsd_packet(
            &mut skipped,
            |_| 1,
            |skipped| {
                if *skipped == 0 {
                    Ok(DsdPacketEvent::Packet {
                        track_id: 1,
                        packet: 7,
                    })
                } else {
                    *skipped -= 1;
                    Ok(DsdPacketEvent::Packet {
                        track_id: 2,
                        packet: 0,
                    })
                }
            },
            |_| Ok(()),
        )
        .unwrap();

        assert_eq!(packet, Some(7));
    }

    #[test]
    fn dsd_packet_skip_loop_rejects_repeated_resets() {
        let error = next_dsd_packet(
            &mut (),
            |_| 1,
            |_| Ok(DsdPacketEvent::<()>::Reset),
            |_| Ok(()),
        )
        .unwrap_err();

        assert_eq!(error.message(), "Too many consecutive DSD decoder resets");
    }

    #[test]
    fn dsd_reset_can_change_the_track_id() {
        // After the reset the track id is 5; packets of the old id 1 must be skipped.
        let mut state = (1u32, 0u32);
        let packet = next_dsd_packet(
            &mut state,
            |s| s.0,
            |s| {
                s.1 += 1;
                Ok(match s.1 {
                    1 => DsdPacketEvent::Reset,
                    2 => DsdPacketEvent::Packet {
                        track_id: 1,
                        packet: 0,
                    },
                    _ => DsdPacketEvent::Packet {
                        track_id: 5,
                        packet: 9,
                    },
                })
            },
            |s| {
                s.0 = 5;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(packet, Some(9));
    }
}

#[cfg(test)]
mod seek_tests {
    use super::*;
    use std::collections::VecDeque;

    type SeekResult = std::result::Result<SeekedTo, SymphoniaError>;

    fn seeked(required: i64, actual: i64) -> SeekedTo {
        SeekedTo {
            track_id: 0,
            required_ts: Timestamp::new(required),
            actual_ts: Timestamp::new(actual),
        }
    }

    /// Scripted reader: each seek pops the next result; records the targets and rebuilds.
    struct Mock {
        results: VecDeque<SeekResult>,
        seeks: Vec<Time>,
        rebuilds: u32,
        rebuild_fails: bool,
    }

    impl Mock {
        fn new(results: Vec<SeekResult>) -> Self {
            Self {
                results: results.into(),
                seeks: Vec::new(),
                rebuilds: 0,
                rebuild_fails: false,
            }
        }
    }

    impl ResettableSeek for Mock {
        fn try_seek(&mut self, time: Time) -> SeekResult {
            self.seeks.push(time);
            self.results.pop_front().expect("unexpected extra seek")
        }

        fn rebuild_after_reset(&mut self) -> std::result::Result<(), SymphoniaError> {
            self.rebuilds += 1;
            if self.rebuild_fails {
                Err(SymphoniaError::Unsupported("no track"))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn seek_without_reset_does_not_rebuild() {
        let mut m = Mock::new(vec![Ok(seeked(10, 8))]);
        let t = Time::from_millis(1500);
        let got = seek_with_resets(&mut m, t, MAX_SEEK_RESETS).unwrap();
        assert_eq!(got.actual_ts, Timestamp::new(8));
        assert_eq!(m.rebuilds, 0);
        assert_eq!(m.seeks, vec![t]);
    }

    #[test]
    fn reset_rebuilds_and_repeats_the_same_seek() {
        let mut m = Mock::new(vec![Err(SymphoniaError::ResetRequired), Ok(seeked(10, 10))]);
        let t = Time::from_millis(90_000);
        assert!(seek_with_resets(&mut m, t, MAX_SEEK_RESETS).is_ok());
        assert_eq!(m.rebuilds, 1);
        assert_eq!(m.seeks, vec![t, t], "the very same target must be retried");
    }

    #[test]
    fn two_resets_are_tolerated() {
        let mut m = Mock::new(vec![
            Err(SymphoniaError::ResetRequired),
            Err(SymphoniaError::ResetRequired),
            Ok(seeked(1, 1)),
        ]);
        assert!(seek_with_resets(&mut m, Time::from_millis(1), 2).is_ok());
        assert_eq!(m.rebuilds, 2);
    }

    #[test]
    fn endless_resets_are_bounded() {
        let mut m = Mock::new(vec![
            Err(SymphoniaError::ResetRequired),
            Err(SymphoniaError::ResetRequired),
            Err(SymphoniaError::ResetRequired),
        ]);
        let err = seek_with_resets(&mut m, Time::from_millis(1), 2).unwrap_err();
        assert!(matches!(err, SymphoniaError::ResetRequired));
        assert_eq!(m.rebuilds, 2);
        assert_eq!(m.seeks.len(), 3);
    }

    #[test]
    fn other_errors_pass_through_without_rebuild() {
        let mut m = Mock::new(vec![Err(SymphoniaError::SeekError(
            SeekErrorKind::OutOfRange,
        ))]);
        let err = seek_with_resets(&mut m, Time::from_millis(1), 2).unwrap_err();
        assert!(matches!(
            err,
            SymphoniaError::SeekError(SeekErrorKind::OutOfRange)
        ));
        assert_eq!(m.rebuilds, 0);
    }

    #[test]
    fn rebuild_failure_aborts_the_seek() {
        let mut m = Mock::new(vec![Err(SymphoniaError::ResetRequired)]);
        m.rebuild_fails = true;
        let err = seek_with_resets(&mut m, Time::from_millis(1), 2).unwrap_err();
        assert!(matches!(err, SymphoniaError::Unsupported(_)));
        assert_eq!(m.seeks.len(), 1);
    }

    fn tb(denom: u32) -> TimeBase {
        TimeBase::try_new(1, denom).unwrap()
    }

    /// Feed `packets` back-to-back (each `len` frames from `start`) through the skip state and
    /// return (total frames skipped, index of the packet where the target was reached).
    fn run(
        mut skip: SeekSkip,
        start: i64,
        len: usize,
        packets: usize,
        tb: TimeBase,
        rate: u32,
    ) -> (usize, Option<usize>) {
        let mut skipped = 0;
        for i in 0..packets {
            let pts = Timestamp::new(start + (i * len) as i64);
            let (n, done) = skip.advance(pts, len, tb, rate);
            skipped += n;
            if done {
                return (skipped, Some(i));
            }
        }
        (skipped, None)
    }

    #[test]
    fn skips_the_seek_pre_roll_exactly() {
        // Matroska audio (timebase 1/rate): 200 ms pre-roll before the requested timestamp.
        let rate = 48_000;
        let required = 5 * 48_000 + 123;
        let actual = required - 9_600;
        let skip = SeekSkip::new(
            Timestamp::new(required),
            Timestamp::new(actual),
            Some(tb(rate)),
            rate,
        )
        .unwrap();
        let (skipped, reached) = run(skip, actual, 1152, 64, tb(rate), rate);
        assert_eq!(skipped, 9_600, "exactly required - actual frames");
        assert!(reached.is_some());
    }

    #[test]
    fn skips_within_a_packet_after_a_coarse_landing() {
        // FLAC-like: packet of 4096 frames starts 1000 frames before the target.
        let rate = 44_100;
        let mut skip = SeekSkip::new(
            Timestamp::new(10_000),
            Timestamp::new(9_000),
            Some(tb(rate)),
            rate,
        )
        .unwrap();
        assert_eq!(
            skip.advance(Timestamp::new(9_000), 4096, tb(rate), rate),
            (1_000, true)
        );
    }

    #[test]
    fn converts_between_timebase_and_output_rate() {
        // A 1/1000 timebase (legacy Matroska audio) with 44.1 kHz output: 500 ms = 22050.
        let mut skip = SeekSkip::new(
            Timestamp::new(1_500),
            Timestamp::new(1_000),
            Some(tb(1000)),
            44_100,
        )
        .unwrap();
        assert_eq!(
            skip.advance(Timestamp::new(1_000), 100_000, tb(1000), 44_100),
            (22_050, true)
        );
    }

    #[test]
    fn nothing_to_skip_when_the_seek_landed_on_or_after_the_target() {
        let t = Some(tb(48_000));
        assert!(SeekSkip::new(Timestamp::new(100), Timestamp::new(100), t, 48_000).is_none());
        assert!(SeekSkip::new(Timestamp::new(100), Timestamp::new(200), t, 48_000).is_none());
        assert!(SeekSkip::new(Timestamp::new(100), Timestamp::new(0), None, 48_000).is_none());
    }

    #[test]
    fn packet_trim_start_moves_the_valid_start() {
        // The first valid frame of a trimmed packet is at pts + trim_start: only that distance
        // to the target remains.
        let rate = 48_000;
        let mut skip = SeekSkip::new(
            Timestamp::new(0),
            Timestamp::new(-312),
            Some(tb(rate)),
            rate,
        )
        .unwrap();
        let valid_start =
            Timestamp::new(-312).saturating_add(symphonia::core::units::Duration::new(312));
        assert_eq!(skip.advance(valid_start, 1024, tb(rate), rate), (0, true));
    }

    #[test]
    fn unreliable_timestamps_cannot_swallow_the_stream() {
        // Every packet claims to start at 0: skipping stops at the announced distance + 1 s.
        let rate = 48_000;
        let mut skip = SeekSkip::new(
            Timestamp::new(48_000),
            Timestamp::new(38_400),
            Some(tb(rate)),
            rate,
        )
        .unwrap();
        let mut skipped = 0usize;
        for _ in 0..1000 {
            let (n, done) = skip.advance(Timestamp::new(0), 1024, tb(rate), rate);
            skipped += n;
            if done {
                break;
            }
        }
        assert!(skipped as u64 <= 9_600 + u64::from(rate));
    }

    #[test]
    fn tick_conversion_rounds_to_nearest_frame() {
        // DSD64 timeline (1/2822400) to 44.1 kHz PCM: 64 DSD samples per PCM frame.
        assert_eq!(ticks_to_frames(64, tb(2_822_400), 44_100), 1);
        assert_eq!(ticks_to_frames(95, tb(2_822_400), 44_100), 1);
        assert_eq!(ticks_to_frames(97, tb(2_822_400), 44_100), 2);
        // Pass-through DSD frames (rate / 8): 8 DSD samples per frame.
        assert_eq!(ticks_to_frames(2_822_400, tb(2_822_400), 352_800), 352_800);
    }

    fn ns_media_info(secs: u64) -> MediaInfo {
        let mut mi = MediaInfo::new();
        mi.with_time_base(TimeBase::try_new(1, 1_000_000_000).unwrap());
        mi.with_duration(symphonia::core::units::Duration::new(secs * 1_000_000_000));
        mi
    }

    #[test]
    fn ogg_chain_duration_comes_from_media_info() {
        let d = chain_duration_secs(FORMAT_ID_OGG, &ns_media_info(95)).unwrap();
        assert!((d - 95.0).abs() < 1e-9);
    }

    #[test]
    fn chain_duration_is_ogg_and_ns_timebase_only() {
        use symphonia::core::formats::well_known::FORMAT_ID_FLAC;
        assert!(chain_duration_secs(FORMAT_ID_FLAC, &ns_media_info(95)).is_none());
        // Single-link Ogg: media info mirrors the track (sample-rate timebase).
        let mut mi = MediaInfo::new();
        mi.with_time_base(TimeBase::try_new(1, 48_000).unwrap());
        mi.with_duration(symphonia::core::units::Duration::new(48_000));
        assert!(chain_duration_secs(FORMAT_ID_OGG, &mi).is_none());
        assert!(chain_duration_secs(FORMAT_ID_OGG, &MediaInfo::new()).is_none());
    }
}
