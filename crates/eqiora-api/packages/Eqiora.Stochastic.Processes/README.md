# Stochastic processes

The source declares two dimensionless Ito processes. `OrnsteinUhlenbeck` owns
additive noise and a drift towards `mean`; a positive `reversion_rate` gives
mean reversion. `GeometricBrownianMotion` owns multiplicative noise and an Ito
`growth_rate`. Each instance declares its own independent Wiener channel.

These components contain the equations and initial conditions. The ordinary
time Plan selects the numerical method, and each Run supplies its sample
identity and master seed. The package does not execute an ensemble or compute
its statistics. A numerical Euler–Maruyama trajectory has no blanket
positivity guarantee, even for geometric Brownian motion.

For physical quantities, use an explicit scale to interpret the dimensionless
`value`. Volatility has units of inverse square-root seconds; it is not a
per-step random perturbation.
