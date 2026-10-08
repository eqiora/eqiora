# Typed finite-window observations

The case executes accepted common ODE trajectories through the existing Observable
owner and the componentwise direct DFT. The independently derived coefficients,
units, projections and falsifiers live in
`crates/eqiora-numerics/src/common_trajectory/spectrum/tests.rs`.

The transform is `C[k] = sum(w[n] x[n] exp(+2*pi*i*k*n/N))/N`, with inverse
`sum(C[k] exp(-2*pi*i*k*n/N))`. Phase is relative to the first included sample.
Actual accepted states must exist on the explicitly declared uniform grid;
extra output samples do not select the grid. Missing or irregular samples reject.
The interval is half-open `[t0,t0+N*dt)`. Event outputs use the accepted reset side.

Coefficients and amplitudes retain input units; bin power has squared units;
power density per Hz adds time to squared units. The rectangle-transform
estimator is `N*dt*C` with value-times-time units, not an exact continuous
integral. One-sided projections require mathematically real input. DC and the
even-N Nyquist bin are not doubled; interior amplitude and power density are
doubled independently. Windows receive no hidden coherent-gain or energy correction.

Inverse reconstruction means windowed sampled values only. Frequencies separated
by `1/dt` alias; off-bin signals leak. A finite spectrum does not establish a
steady harmonic solution. Derivatives, irregular grids, continuous reconstruction,
FFT acceleration and spatial trajectories are not claimed. Sample and component
counts in this evidence are witnesses, not mathematical limits; execution work
is bounded by the caller's explicit multiply-add budget.

The Python product test additionally runs the shared time backend, retains Model
Observable and Result provenance, and verifies replay through existing Result bytes.
That product test is separate from this case's narrow scientific runner.
