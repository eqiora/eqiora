# Retained serial Poisson arrivals

This case checks the constant-rate process used by the finite-time Run. Exact
retained cumulative thresholds determine right-continuous counts and the pending
next event. Queries do not advance shared RNG state; subinterval counts compose,
and a failed event budget cannot redraw a more convenient path. Zero intensity
requires neither an event nor an event budget.

For the independent mathematical reference `N~Poisson(2)`, the mean and variance
are two. The fourth central moment is `2+3*2²=14`, so the variance of `(N-2)²`
is ten. Two independent channels have centered cross moment zero, with variance
four. The fixed 4096 sample identities are checked against six predeclared standard
errors: `6*sqrt(2/n)`, `6*sqrt(10/n)`, and `12/sqrt(n)`, respectively. No expectation
or bound is taken from the implementation's sample output. Passing these bounded
moments does not prove independence of all streams.

Thresholds are exact sums of binary64 exponential draws from the declared sampler;
native logarithms retain an environment boundary. State-dependent intensity,
continuous exact-real random variates, distributed scheduling, cross-platform bit
identity, and installed Python execution are not claimed.
