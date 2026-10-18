//! Shared exact binary normalization for scalar numerical actions.
pub(super) fn exponent(value: f64) -> i32 {
    if value == 0. {
        return 0;
    }
    let bits = value.abs().to_bits();
    let stored = ((bits >> 52) & 0x7ff) as i32;
    if stored == 0 {
        // A subnormal is its integer significand times 2^-1074.
        63 - bits.leading_zeros() as i32 - 1074
    } else {
        stored - 1023
    }
}

pub(super) fn scale(mut value: f64, mut exponent: i32) -> f64 {
    // Every factor is normal and exactly represented. Applying bounded chunks
    // avoids first constructing 2^1074 or 2^-2097 as an f64.
    while exponent != 0 && value != 0. && value.is_finite() {
        let part = exponent.clamp(-512, 512);
        value *= f64::from_bits(((part + 1023) as u64) << 52);
        exponent -= part;
    }
    value
}
