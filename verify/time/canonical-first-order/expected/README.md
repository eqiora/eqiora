# Acceptance criteria

- Canonical model validation and scalar Operator IR lowering succeed.
- State order is derived deterministically from derivative-symbol order.
- The permuted, signed, and scaled residual Jacobian normalizes to the expected
  right-hand side and state JVP.
- Tsitouras 5(4) samples agree with the analytic coupled trajectory within
  `4e-8` relative scale.
- The exact algebraic zero row produces a rank-deficient mass matrix,
  a common-initialized pair reported as `Provided`, and a BDF trajectory within `2e-6`; `x+z=1` holds within
  `5e-9`.
- The state-dependent derivative coefficient returns `EQ0705` from the
  first-order projection and never reaches Diffsol.
- Lowering and time-run artifacts round-trip with stable content digests;
  model/witness coefficient drift, plan/report method drift, and reference
  `ImplicitEuler` in the first-order v1 manifest are rejected.
