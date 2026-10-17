//! Compensated addition for real and complex sparse reductions.
use eqiora_core::Scalar;

pub(super) fn add<S: Scalar>(sum: &mut S, correction: &mut S, value: S) {
    // TwoSum does not assume an ordering of magnitudes. Complex addition is
    // componentwise, so the same identity applies independently to both parts.
    let next = *sum + value;
    let recovered = next - *sum;
    *correction = *correction + ((*sum - (next - recovered)) + (value - recovered));
    *sum = next;
}
