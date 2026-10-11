use eqiora_core::Diagnostic;

use crate::invalid_artifact;

/// Syntax-level limits applied before JSON deserialization.
///
/// Artifact families own every semantic work budget. This common contract
/// admits only encoded size and JSON nesting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonDecoderLimits {
    /// Maximum encoded bytes accepted.
    pub max_bytes: usize,
    /// Maximum JSON object/array nesting accepted before deserialization.
    pub max_nesting_depth: usize,
}

impl Default for JsonDecoderLimits {
    fn default() -> Self {
        Self {
            max_bytes: 16 * 1024 * 1024,
            max_nesting_depth: 64,
        }
    }
}

pub(crate) fn check_json_limits(bytes: &[u8], limits: JsonDecoderLimits) -> Result<(), Diagnostic> {
    if bytes.len() > limits.max_bytes {
        return Err(invalid_artifact(format!(
            "artifact has {} bytes, exceeding the {} byte decoder limit",
            bytes.len(),
            limits.max_bytes
        )));
    }

    let mut depth = 0_usize;
    let mut in_string = false;
    let mut escaped = false;
    for &byte in bytes {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth = depth.checked_add(1).ok_or_else(|| {
                    invalid_artifact("artifact JSON nesting depth overflowed usize")
                })?;
                if depth > limits.max_nesting_depth {
                    return Err(invalid_artifact(format!(
                        "artifact JSON nesting exceeds the {} level decoder limit",
                        limits.max_nesting_depth
                    )));
                }
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{JsonDecoderLimits, check_json_limits};

    #[test]
    fn byte_and_depth_boundaries_are_exact() {
        let bytes = br#"{"v":[0]}"#;
        check_json_limits(
            bytes,
            JsonDecoderLimits {
                max_bytes: bytes.len(),
                max_nesting_depth: 2,
            },
        )
        .expect("exact syntax boundaries must be admitted");

        assert!(
            check_json_limits(
                bytes,
                JsonDecoderLimits {
                    max_bytes: bytes.len() - 1,
                    max_nesting_depth: 2,
                },
            )
            .is_err()
        );
        assert!(
            check_json_limits(
                bytes,
                JsonDecoderLimits {
                    max_bytes: bytes.len(),
                    max_nesting_depth: 1,
                },
            )
            .is_err()
        );
    }

    #[test]
    fn delimiters_inside_strings_do_not_consume_depth() {
        check_json_limits(
            br#"{"v":"[[[[{{{{"}"#,
            JsonDecoderLimits {
                max_bytes: 64,
                max_nesting_depth: 1,
            },
        )
        .expect("quoted delimiters are not JSON structure");
    }
}
