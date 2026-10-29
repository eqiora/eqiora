# Signals, controls, and system simulation

[Capability matrix](../capability-matrix.md) · [legend](../capability-matrix.md#how-to-read-the-matrix)

| Capability | C | X | V | M | Current implementation and planned extensions |
|---|:---:|:---:|:---:|:---:|---|
| Block diagram as semantic projection | 🟨 | 🟨 | 🟨 | ⬜ | Studio shows a relation projection; full block authoring remains |
| Algebraic loop admission | ✅ | ✅ | 🟨 | ⬜ | Implicit relations admit loops; canonical multirate loop case remains unverified |
| Continuous-time blocks | 🟨 | 🟨 | 🟨 | ⬜ | Relations can express them; reusable block library remains |
| Bounded periodic proportional control | ✅ | ✅ | ⬜ | ⬜ | One proportional controller executes atomically on one exact 10 ms clock and holds its output between ticks; reusable control libraries, multiple rates, and codegen semantics remain; [`hybrid.packaged-dc-motor-controller`](../../verify/hybrid/packaged-dc-motor-controller/README.md) |
| Multirate sample times | ✅ | 🟨 | 🟨 | ⬜ | Multiple clock domains are representable; full scheduler and rate-transition proof remain |
| Rate transition blocks | ⬜ | ⬜ | ⬜ | ⬜ | Buffering, determinism, latency, and codegen remain |
| Signal buses | ⬜ | ⬜ | ⬜ | ⬜ | Typed nested records, selectors, and compatibility remain |
| Vector and matrix signals | ⬜ | ⬜ | ⬜ | ⬜ | Shape inference and optimized execution remain |
| Complex signals | ⬜ | ⬜ | ⬜ | ⬜ | Type, math, serialization, and visualization remain |
| Fixed-point signals | ⬜ | ⬜ | ⬜ | ⬜ | Word length, scaling, overflow, rounding, and bit-true evidence remain |
| Enumerated signals | ⬜ | ⬜ | ⬜ | ⬜ | Language and wire support remain |
| Variant subsystems | ⬜ | ⬜ | ⬜ | ⬜ | Compile-time and runtime variants with provenance remain |
| Model references | 🟨 | ⬜ | ⬜ | ⬜ | Local component elaboration exists; package identity, reference boundaries, independent compilation, and execution semantics remain |
| Enabled and triggered subsystems | 🟨 | ⬜ | ⬜ | ⬜ | Activation ingredients exist; complete reset/hold semantics remain |
| Function-call subsystems | ⬜ | ⬜ | ⬜ | ⬜ | Deterministic invocation, reentrancy, and scheduling remain |
| Data stores and scoped state | ⬜ | ⬜ | ⬜ | ⬜ | Explicit state ownership and concurrency semantics remain |
| Sampled delay and discrete integration | ✅ | ✅ | ⬜ | ⬜ | `Eqiora.Controls.Sampled` provides UnitDelay and DiscreteIntegrator on the caller's clock with explicit initial memory; tests compare against hand-computed recurrences in source, packages and installed Python, and reject unit, initialization and foreign-clock errors. Runs use reference sessions, not Plan/Result replay. Not yet: generic component types, implicit sample-and-hold, rate transitions. |
| Lookup tables and interpolation | 🟨 | 🟨 | ⬜ | ⬜ | One-dimensional real tables give piecewise-linear values with declared validity, units and identity kept through source, packages, Python and replay; derivatives exist inside segments and are rejected at knots. Time-dependent tables drive ODEs, and `data.csv_array` loads measured columns for residual and loss comparisons. Not yet: higher-dimensional interpolation, extrapolation, wider sensor profiles. |
| Transfer functions and state space | ⬜ | ⬜ | ⬜ | ⬜ | Continuous/discrete MIMO library and conversion remain |
| PID and control primitives | ⬜ | ⬜ | ⬜ | ⬜ | Anti-windup, bumpless transfer, tuning, and codegen remain |
| Finite-window trajectory spectra | ✅ | ✅ | ✅ | ⬜ | A direct DFT of accepted ODE samples gives coefficients, amplitude and power with explicit window (rectangular or Hann), sign convention, normalization and units; the registered case checks small hand-derived examples. Not yet: irregular sampling, FFT backend, spatial trajectories, derivatives of spectra; [`time.finite-window-spectrum`](../../verify/time/finite-window-spectrum/README.md). |
| Filters and signal processing | ⬜ | ⬜ | ⬜ | ⬜ | FIR/IIR, multirate and streaming libraries remain; finite-window trajectory spectra are listed separately. |
| Control design and linear analysis | ⬜ | ⬜ | ⬜ | ⬜ | Operating points, linearization, frequency response, margins, and synthesis remain |
| Model predictive control | ⬜ | ⬜ | ⬜ | ⬜ | Online optimization, constraints, estimation, and deployment remain |
| State estimation | ⬜ | ⬜ | ⬜ | ⬜ | Kalman variants, observers, smoothing, and sensor models remain |
| Co-simulation master | ⬜ | ⬜ | ⬜ | ⬜ | Time negotiation, rollback, algebraic coupling, and error policy remain |
| FMI import/export | ⬜ | ⬜ | ⬜ | ⬜ | Model Exchange, Co-Simulation, Scheduled Execution, and version policy remain |
| Acausal physical network solve | ✅ | ✅ | ⬜ | ⬜ | Affine networks, including a packaged 10×10 circuit, solve through the general sparse faer path with an analytic check; one three-package electrical and rotational network runs transient mixed-signal dynamics with backward Euler. Not yet: general nonlinear or transient networks, broad libraries, production solvers, MPI, GPU; [`electrical.parallel-dc-network`](../../verify/electrical/parallel-dc-network/README.md), [`packages.hierarchical-physical-boundary`](../../verify/packages/hierarchical-physical-boundary/README.md), [`hybrid.packaged-dc-motor-controller`](../../verify/hybrid/packaged-dc-motor-controller/README.md) |
| Physical-domain connector library | 🟨 | 🟨 | ⬜ | ⬜ | Reusable scalar electrical `Pin` and rotational connector declarations are verified through ordinary exact packages; translational, thermal, fluid, multibody, vector/tensor, and broad component catalogs remain; [`hybrid.packaged-dc-motor-controller`](../../verify/hybrid/packaged-dc-motor-controller/README.md) |
| Physical units at ports | ✅ | ✅ | ⬜ | ⬜ | Nominal scalar electrical and rotational domains, dimensioned Across/Through coordinates, compatibility rejection, and typed physical observations are verified; conversions and broader domain/value shapes remain; [`hybrid.packaged-dc-motor-controller`](../../verify/hybrid/packaged-dc-motor-controller/README.md) |
| Canonical DC motor plus controller | ✅ | ✅ | ⬜ | ⬜ | Exactly three packages, one scalar ideal linear motor, viscous lumped load, proportional controller, one exact clock, host-serial `f64`, dense Newton, and backward Euler; no general DAE, Simulink/Simscape/Stateflow, production, MPI/GPU, real-time, or dynamic-plugin claim; [`hybrid.packaged-dc-motor-controller`](../../verify/hybrid/packaged-dc-motor-controller/README.md) |
| Fault statechart plus thermal plant | 🟨 | 🟨 | ⬜ | ⬜ | A focused fixed-equation enum/event controller test couples affine temperature evolution with fault hold, command retention, recovery and in-process restart. Full statechart semantics and the separate registered benchmark remain absent. |

