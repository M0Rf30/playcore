// SPDX-License-Identifier: MIT OR Apache-2.0

//! Adapters that turn plain `Read`/`Seek` byte sources into Symphonia [`MediaSource`]s, for
//! [`SymphoniaDecoder::open_source`](crate::decoder::SymphoniaDecoder::open_source).
//!
//! Symphonia already implements `MediaSource` for `std::fs::File` and `Cursor<Vec<u8>>`;
//! these cover everything else (HTTP range readers, in-memory slices, live radio streams).

use std::io::{self, Read, Seek, SeekFrom};

use symphonia::core::io::MediaSource;

/// A seekable `Read + Seek` source with a caller-supplied total length.
///
/// `byte_len` is supplied up front since arbitrary readers have no cheap, uniform way to
/// report their length the way a file's metadata does. Knowing it enables trailing-tag probing
/// and lets Symphonia seek accurately near the end of the stream.
pub struct ReadSeekSource<R> {
    inner: R,
    byte_len: Option<u64>,
}

impl<R> ReadSeekSource<R> {
    /// Wrap `inner`; `byte_len` is the total length in bytes if known.
    pub fn new(inner: R, byte_len: Option<u64>) -> Self {
        Self { inner, byte_len }
    }
}

impl<R: Read> Read for ReadSeekSource<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

impl<R: Seek> Seek for ReadSeekSource<R> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.inner.seek(pos)
    }
}

impl<R: Read + Seek + Send + Sync> MediaSource for ReadSeekSource<R> {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        self.byte_len
    }
}

/// A non-seekable `Read` source (e.g. an internet radio stream).
///
/// Unlike [`ReadSeekSource`], seeking is never supported: `is_seekable()` is always `false`
/// and the `Seek` impl always errors, matching what a live stream can actually do.
pub struct StreamSource<R> {
    inner: R,
}

impl<R> StreamSource<R> {
    /// Wrap `inner`.
    pub fn new(inner: R) -> Self {
        Self { inner }
    }
}

impl<R: Read> Read for StreamSource<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

impl<R> Seek for StreamSource<R> {
    fn seek(&mut self, _pos: SeekFrom) -> io::Result<u64> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "seeking is not supported on live streams",
        ))
    }
}

impl<R: Read + Send + Sync> MediaSource for StreamSource<R> {
    fn is_seekable(&self) -> bool {
        false
    }

    fn byte_len(&self) -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn read_seek_source_reports_seekability_and_length() {
        let mut s = ReadSeekSource::new(Cursor::new(vec![1u8, 2, 3, 4]), Some(4));
        assert!(s.is_seekable());
        assert_eq!(s.byte_len(), Some(4));
        s.seek(SeekFrom::Start(2)).unwrap();
        let mut b = [0u8; 2];
        s.read_exact(&mut b).unwrap();
        assert_eq!(b, [3, 4]);
    }

    #[test]
    fn stream_source_refuses_to_seek() {
        let mut s = StreamSource::new(Cursor::new(vec![9u8; 8]));
        assert!(!s.is_seekable());
        assert_eq!(s.byte_len(), None);
        assert_eq!(
            s.seek(SeekFrom::Start(0)).unwrap_err().kind(),
            io::ErrorKind::Unsupported
        );
        let mut b = [0u8; 3];
        s.read_exact(&mut b).unwrap();
        assert_eq!(b, [9, 9, 9]);
    }
}
