//! Lossless binary64 and immutable artifact capture for native reference producers.
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::Path;

pub(super) fn binary64_hex(value: f64) -> String {
    assert!(value.is_finite());
    let bits = value.to_bits();
    let sign = if bits >> 63 == 0 { "" } else { "-" };
    let mantissa = bits & ((1_u64 << 52) - 1);
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    if exponent == 0 && mantissa == 0 {
        format!("{sign}0x0.0p+0")
    } else if exponent == 0 {
        format!("{sign}0x0.{mantissa:013x}p-1022")
    } else {
        format!("{sign}0x1.{mantissa:013x}p{:+}", exponent - 1023)
    }
}

#[test]
fn binary64_hex_preserves_exact_finite_bits() {
    for (value, expected) in [
        (0., "0x0.0p+0"),
        (-0., "-0x0.0p+0"),
        (1., "0x1.0000000000000p+0"),
        (-2.5, "-0x1.4000000000000p+1"),
        (f64::from_bits(1), "0x0.0000000000001p-1022"),
        (f64::MIN_POSITIVE, "0x1.0000000000000p-1022"),
        (f64::MAX, "0x1.fffffffffffffp+1023"),
    ] {
        assert_eq!(binary64_hex(value), expected);
    }
}

pub(super) fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub(super) fn write(path: &Path, bytes: &[u8]) -> String {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    file.write_all(bytes).unwrap();
    digest(bytes)
}

pub(super) fn json_bytes(value: &serde_json::Value) -> Vec<u8> {
    serde_json::to_vec_pretty(value).unwrap()
}
