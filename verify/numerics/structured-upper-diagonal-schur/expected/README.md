# Acceptance contract

The [case contract](../case.toml) owns all predeclared thresholds. Each direct,
manual structured and ranked structured member must pass them independently.
Agreement between solvers is not an oracle.

For every consecutive mesh pair the order is `log2(E_h / E_(h/2))`.
Velocity L2 errors are divided by `U L`, velocity H1 and divergence L2 errors
by `U`, and pressure L2 errors by `P L`. The pressure integral is divided by
`P L^2`; equilibrium closure is divided by `P L`. The retained steady-Stokes
multiplier is already dimensionless. Every live solve also passes its original
independently checked true-residual target.

Canonical replay retains the complete composition and finite gauge metadata.
It does not supply independent numerical reauthentication of that retained
multiplier. There are no solver-generated expected-value tables.
