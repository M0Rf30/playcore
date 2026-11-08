// SPDX-License-Identifier: MIT OR Apache-2.0

//! Error type shared by the decoder, DoP encoder and format helpers.
//!
//! Applications map [`Error`] into their own error type with a `From` impl; the
//! [`MediaErrorKind`] keeps the class of the underlying Symphonia failure so a
//! caller can still tell a truncated file from a malformed one without parsing text.

use std::fmt;

use symphonia::core::errors::Error as SymphoniaError;

/// Result alias used throughout the crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// What went wrong in a demuxer or decoder (mirrors Symphonia's error classes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MediaErrorKind {
    /// Reading the source failed; carries the I/O error kind
    /// (`UnexpectedEof` marks a truncated stream).
    Io(std::io::ErrorKind),
    /// Malformed data; the packet or frame can be skipped.
    Decode,
    /// The source could not be seeked.
    Seek,
    /// An unsupported container or codec feature.
    Unsupported,
    /// A decoder safety limit was reached.
    Limit,
    /// The demuxer or decoder must be rebuilt before continuing.
    ResetRequired,
    /// Any error class this enum does not know yet.
    Other,
}

/// An error raised by playcore.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A playback failure described by a plain message ("No audio tracks found",
    /// "Not seekable", "Unsupported DSD sample rate: 48000", ...).
    Other(String),
    /// A demuxer/decoder failure, with the kind of the underlying Symphonia error
    /// preserved. `message` carries the context prefix (`"Failed to read packet: ..."`).
    Media {
        /// Class of the underlying failure.
        kind: MediaErrorKind,
        /// Human-readable text, including the context.
        message: String,
        /// The original error, for `Error::source` walkers.
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

impl Error {
    /// A plain message error.
    #[must_use]
    pub fn other(message: impl Into<String>) -> Self {
        Self::Other(message.into())
    }

    /// Wrap a Symphonia error, prefixing its text with `context`
    /// (`"<context>: <error>"`) while keeping its kind and cause.
    #[must_use]
    pub fn from_symphonia_context(context: &str, err: SymphoniaError) -> Self {
        let message = format!("{context}: {err}");
        Self::from_symphonia_message(err, message)
    }

    fn from_symphonia_message(err: SymphoniaError, message: String) -> Self {
        let kind = match &err {
            SymphoniaError::IoError(e) => MediaErrorKind::Io(e.kind()),
            SymphoniaError::DecodeError(_) => MediaErrorKind::Decode,
            SymphoniaError::SeekError(_) => MediaErrorKind::Seek,
            SymphoniaError::Unsupported(_) => MediaErrorKind::Unsupported,
            SymphoniaError::LimitError(_) => MediaErrorKind::Limit,
            SymphoniaError::ResetRequired => MediaErrorKind::ResetRequired,
            _ => MediaErrorKind::Other,
        };
        Self::Media {
            kind,
            message,
            source: Some(Box::new(err)),
        }
    }

    /// The bare error text, without any category prefix.
    #[must_use]
    pub fn message(&self) -> &str {
        match self {
            Self::Other(message) | Self::Media { message, .. } => message,
        }
    }

    /// The media error kind, if this error came from a demuxer or decoder.
    #[must_use]
    pub fn media_kind(&self) -> Option<MediaErrorKind> {
        match self {
            Self::Media { kind, .. } => Some(*kind),
            Self::Other(_) => None,
        }
    }

    /// `true` for a truncated source: reading hit an unexpected end of file.
    #[must_use]
    pub fn is_unexpected_eof(&self) -> bool {
        self.media_kind() == Some(MediaErrorKind::Io(std::io::ErrorKind::UnexpectedEof))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Media {
                source: Some(source),
                ..
            } => Some(source.as_ref()),
            _ => None,
        }
    }
}

impl From<SymphoniaError> for Error {
    fn from(err: SymphoniaError) -> Self {
        let message = err.to_string();
        Self::from_symphonia_message(err, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symphonia::core::errors::SeekErrorKind;

    #[test]
    fn symphonia_error_kind_is_preserved() {
        let cases = [
            (
                SymphoniaError::DecodeError("bad frame"),
                MediaErrorKind::Decode,
            ),
            (
                SymphoniaError::SeekError(SeekErrorKind::Unseekable),
                MediaErrorKind::Seek,
            ),
            (
                SymphoniaError::Unsupported("codec"),
                MediaErrorKind::Unsupported,
            ),
            (SymphoniaError::LimitError("too big"), MediaErrorKind::Limit),
            (SymphoniaError::ResetRequired, MediaErrorKind::ResetRequired),
            (
                SymphoniaError::IoError(std::io::ErrorKind::UnexpectedEof.into()),
                MediaErrorKind::Io(std::io::ErrorKind::UnexpectedEof),
            ),
        ];
        for (err, kind) in cases {
            let text = err.to_string();
            let converted = Error::from(err);
            assert_eq!(converted.media_kind(), Some(kind), "{text}");
            assert!(std::error::Error::source(&converted).is_some(), "{text}");
        }
    }

    #[test]
    fn context_prefixes_the_message_and_keeps_the_kind() {
        let e = Error::from_symphonia_context(
            "Failed to decode packet",
            SymphoniaError::DecodeError("bad frame"),
        );
        assert_eq!(
            e.message(),
            "Failed to decode packet: malformed stream: bad frame"
        );
        assert_eq!(e.to_string(), e.message());
        assert_eq!(e.media_kind(), Some(MediaErrorKind::Decode));
        assert!(!e.is_unexpected_eof());
    }

    #[test]
    fn unexpected_eof_is_detected_by_kind_not_text() {
        let eof = Error::from_symphonia_context(
            "Failed to decode packet",
            SymphoniaError::IoError(std::io::ErrorKind::UnexpectedEof.into()),
        );
        assert!(eof.is_unexpected_eof());
        let other_io = Error::from(SymphoniaError::IoError(std::io::ErrorKind::NotFound.into()));
        assert!(!other_io.is_unexpected_eof());
        assert!(!Error::other("end of stream").is_unexpected_eof());
    }

    #[test]
    fn plain_error_has_no_media_kind() {
        let e = Error::other("Not seekable");
        assert_eq!(e.media_kind(), None);
        assert_eq!(e.message(), "Not seekable");
        assert!(std::error::Error::source(&e).is_none());
    }
}
