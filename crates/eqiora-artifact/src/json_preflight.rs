use eqiora_core::Diagnostic;

use crate::invalid_artifact;

/// Syntax-level budgets for JSON artifact input and output.
///
/// Only encoded size and nesting depth are checked here; each artifact family
/// sets its own semantic limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonLimits {
    /// Maximum encoded bytes accepted.
    pub max_bytes: usize,
    /// Maximum JSON object/array nesting accepted before deserialization.
    pub max_nesting_depth: usize,
}

impl Default for JsonLimits {
    fn default() -> Self {
        Self {
            max_bytes: 16 * 1024 * 1024,
            max_nesting_depth: 64,
        }
    }
}

impl JsonLimits {
    /// Shared syntax budget for complete execution artifacts.
    pub const EXECUTION: Self = Self {
        max_bytes: 512 * 1024 * 1024,
        max_nesting_depth: 64,
    };

    /// Check byte and nesting limits before allocating a decoded artifact.
    /// This does not validate JSON syntax or artifact semantics.
    pub fn check(self, bytes: &[u8]) -> Result<(), Diagnostic> {
        if bytes.len() > self.max_bytes {
            return Err(invalid_artifact(format!(
                "artifact has {} bytes, exceeding the {} byte decoder limit",
                bytes.len(),
                self.max_bytes
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
                    if depth > self.max_nesting_depth {
                        return Err(invalid_artifact(format!(
                            "artifact JSON nesting exceeds the {} level decoder limit",
                            self.max_nesting_depth
                        )));
                    }
                }
                b'}' | b']' => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::JsonLimits;

    #[test]
    fn byte_and_depth_boundaries_are_exact() {
        let bytes = br#"{"v":[0]}"#;
        (JsonLimits {
            max_bytes: bytes.len(),
            max_nesting_depth: 2,
        })
        .check(bytes)
        .expect("exact syntax boundaries must be admitted");

        assert!(
            (JsonLimits {
                max_bytes: bytes.len() - 1,
                max_nesting_depth: 2,
            })
            .check(bytes)
            .is_err()
        );
        assert!(
            (JsonLimits {
                max_bytes: bytes.len(),
                max_nesting_depth: 1,
            })
            .check(bytes)
            .is_err()
        );
    }

    #[test]
    fn delimiters_inside_strings_do_not_consume_depth() {
        (JsonLimits {
            max_bytes: 64,
            max_nesting_depth: 1,
        })
        .check(br#"{"v":"[[[[{{{{"}"#)
        .expect("quoted delimiters are not JSON structure");
    }
}
