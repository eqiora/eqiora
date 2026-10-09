# Acceptance

The independently derived values in [references](../references/README.md) are
asserted by `crates/eqiora/tests/oriented_vector_calculus.rs`. Small integer inputs,
products and derivatives stay below binary64's exact integer range, so these
observations require exact values and units. No implementation snapshot or fitted
tolerance supplies an expected value. Model replay repeats the native observations;
the finite Plan/Result path additionally checks the point-bound curl value and
rejects a spatial Field's continuum representation.
