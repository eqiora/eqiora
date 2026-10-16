//! Shared size boundary for complete time checkpoints, Trajectories and Results.
use std::io::{self, Write};

use serde::Serialize;

pub(crate) const MAX_BYTES: usize = 512 * 1024 * 1024;

pub(crate) fn to_vec(value: &impl Serialize) -> serde_json::Result<Vec<u8>> {
    encode(value, MAX_BYTES)
}

fn encode(value: &impl Serialize, limit: usize) -> serde_json::Result<Vec<u8>> {
    let mut writer = BoundedWriter {
        bytes: Vec::new(),
        limit,
    };
    serde_json::to_writer(&mut writer, value)?;
    Ok(writer.bytes)
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
}

impl Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit - self.bytes.len() {
            return Err(io::Error::other(format!(
                "artifact exceeds its {} byte size limit",
                self.limit
            )));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_encoded_bytes_including_utf8_and_json_escaping() {
        let value = ["λ", "\n", "\"", "\\"];
        let expected = b"[\"\xce\xbb\",\"\\n\",\"\\\"\",\"\\\\\"]";
        assert_eq!(encode(&value, expected.len()).unwrap(), expected);
        assert!(encode(&value, expected.len() - 1).unwrap_err().is_io());
        assert!(encode(&value, 0).unwrap_err().is_io());
    }

    #[test]
    fn failed_write_does_not_append_a_partial_chunk() {
        let mut writer = BoundedWriter {
            bytes: vec![1, 2],
            limit: 3,
        };
        assert!(writer.write_all(&[3, 4]).is_err());
        assert_eq!(writer.bytes, [1, 2]);
        writer.write_all(&[3]).unwrap();
        assert_eq!(writer.bytes, [1, 2, 3]);
    }
}
