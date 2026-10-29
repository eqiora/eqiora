# Time, DAE, events, and hybrid semantics

[Capability matrix](../capability-matrix.md) · [legend](../capability-matrix.md#how-to-read-the-matrix)

| Capability | C | X | V | M | Current implementation and planned extensions |
|---|:---:|:---:|:---:|:---:|---|
| Exact rational clock domains | ✅ | ✅ | ✅ | ⬜ | Canonical nominal identity retains reduced rational period and phase. Concrete source quantities and bounded closed literal arithmetic lower without binary64; focused product tests cover this input boundary. |
| Borrowed periodic clock interfaces and static period projection | ✅ | ✅ | ⬜ | ⬜ | Signature `clock tick: periodic` borrows exact identity, including when two independent clocks have equal periods. `period(tick)` is a dimensioned static projection; the clock contract stays rational and ordinary scalar arithmetic receives its binary64 projection. Focused product tests cover this interface boundary; general multiclock execution remains separate. |
| Continuous activation | ✅ | ✅ | ✅ | ⬜ | Scalar reference relations |
| Periodic activation | ✅ | ✅ | ✅ | ⬜ | Sampled-controller conformance slice |
| Explicit sampling and retained-state hold | ✅ | ✅ | ⬜ | ⬜ | `sample(value, tick)` reads a continuous expression on a clock and `hold(memory)` keeps an initialized periodic State between updates; signal connections must share the same clock. Tests cover first ticks, pre- and post-tick values and holds between ticks. Not yet: implicit memory, arbitrary resampling, a multirate transfer library. |
| Bounded reference execution and in-process resume | ✅ | ✅ | ⬜ | ⬜ | `ExecutionSession` runs scalar and array input tables on exact Ports and Clocks up to a horizon; simultaneous ticks commit atomically, and an in-memory checkpoint resumes without reinitializing. Inputs and outputs are limited to one million components. Not yet: saved checkpoints, general array arithmetic, complex, spatial or distributed execution, production scheduling; the array behavior has no registered case. |
| Event activation | ✅ | ✅ | ✅ | ⬜ | Registered zero-crossing event slice |
| Guard activation | 🟨 | ⬜ | ⬜ | ⬜ | Planned canonical activation class; executable semantics remain |
| Explicit first-order ODE | ✅ | ✅ | ✅ | ⬜ | Scalar first-order models with initial values run without a mesh through Diffsol Tsitouras 5(4); you give per-coordinate tolerances, a horizon and output times, and Python reads series by `FieldRef`. Restart begins a fresh adaptive integration. Non-stiff problems only; DAE and events use other paths. |
| Higher-order scalar evolution | ✅ | ✅ | ⬜ | ⬜ | Nested time derivatives of scalar States are expanded into first-order coordinates with explicit companion equations and initial conditions. Tests run second- and third-order examples through explicit and residual-native solvers, step JVP/VJP, units and replay, and ODE Observables read the lower derivatives. Not yet: high-index reduction, resets of derivative coordinates, a registered higher-order case. |
| Stiff ODE | ✅ | ✅ | ✅ | ⬜ | BDF adapter exercised in bounded fixtures |
| Constant mass-matrix DAE | ✅ | ✅ | ✅ | ⬜ | Dense full/singular mass paths; parameter-dependent mass remains |
| Residual-native index-one DAE | ✅ | ✅ | ✅ | ⬜ | One semi-explicit reference fixture |
| General implicit DAE | 🟨 | 🟨 | ✅ | ⬜ | The verified claim is narrow; production residual backend and broad forms remain |
| Residual-native IDA adapter | 🟨 | 🟨 | ⬜ | ⬜ | Optional `eqiora-backend-sundials` runs implicit DAEs with SUNDIALS 7.8 IDA (dense BDF with index-one consistent initialization), and spatial Plans expose BDF in Python. Not yet: installed-package acceptance, scalable linear solvers, events, sensitivities, saved adaptive restart, registered verification. |
| Higher-index DAE | ⬜ | ⬜ | ⬜ | ⬜ | Index analysis/reduction and well-posed diagnostic remain |
| Consistent initialization | ✅ | ✅ | ⬜ | ⬜ | Initial equations jointly fix values, derivatives and clocked pre-state; mass-matrix and semi-explicit problems are checked for rank and consistency, and missing or contradictory conditions are rejected. Not yet: high-index DAE initialization, independent state selection; [`time.canonical-first-order`](../../verify/time/canonical-first-order/README.md), [`time.general-implicit-dae`](../../verify/time/general-implicit-dae/README.md), [`hybrid.packaged-dc-motor-controller`](../../verify/hybrid/packaged-dc-motor-controller/README.md) |
| General structural analysis | 🟨 | ⬜ | ⬜ | ⬜ | Non-square systems, Pantelides-like analysis, and diagnostics remain |
| Adaptive explicit integration | ✅ | ✅ | ✅ | ⬜ | Admitted Diffsol non-stiff path |
| Adaptive implicit integration | ✅ | ✅ | ✅ | ⬜ | Admitted BDF path; general residual-native production path remains |
| Fixed-step implicit integration | ✅ | ✅ | ✅ | ⬜ | Reference implicit Euler |
| Multistep method history | 🟨 | ✅ | 🟨 | ⬜ | Backend execution exists; portable history serialization/restart does not |
| Dense output | 🟨 | 🟨 | 🟨 | ⬜ | Adapter capability is bounded; canonical query contract remains |
| Zero-crossing localization | ✅ | ✅ | ✅ | ⬜ | Bounded registered roots |
| Directional event detection | ✅ | ✅ | ✅ | ⬜ | Direction-aware reference path |
| Diffsol identical-guard event grouping | ✅ | ✅ | ⬜ | ⬜ | One structurally identical scalar guard and direction groups the accepted implicit reset Relations atomically. Distinct simultaneous guards, periodic-tick coincidence, and priority remain; [`hybrid.registered-event`](../../verify/hybrid/registered-event/README.md) |
| Bounded reference event/tick coincidence | ✅ | ✅ | ⬜ | ⬜ | Events at the same time as a clock tick apply disjoint resets together and then run event-only microsteps; conflicting resets or overlapping roots are rejected and the boundary rolls back. Not yet: exact-real coincidence, many roots in one step, saved checkpoints; [`hybrid.bouncing-ball`](../../verify/hybrid/bouncing-ball/README.md). |
| Atomic reset | ✅ | ✅ | ✅ | ⬜ | Monomial implicit reset fixture and canonical next-state behavior |
| Event iteration | ✅ | ✅ | 🟨 | ⬜ | Bounded zero-time iteration with possible-Zeno diagnostic |
| Zeno detection and policy | 🟨 | 🟨 | 🟨 | ⬜ | Bound/diagnostic exists; general classification and continuation policy remain |
| Mode topology change | ⬜ | ⬜ | ⬜ | ⬜ | Structural state-dependent equation sets and safe reinitialization remain |
| Explicit reference event reset priority | ✅ | ✅ | ⬜ | ⬜ | Integer priorities decide between conflicting event resets in descending groups; equal-priority conflicts are rejected and failed resets roll back. The Diffsol path rejects nonzero priority. Covered by focused tests only. |
| Finite value-only mode control | ✅ | ✅ | ⬜ | ⬜ | Enum States and event resets run a thermostat and a fault/thermal controller in reference sessions with fixed equations; tests check event times, temperatures, resume and rejection of competing writes. Not yet: hierarchical or parallel states, switched equation sets, saved event checkpoints; the thermostat and fault-plant benchmarks are not claimed. |
| State machine execution | 🟨 | 🟨 | 🟨 | ⬜ | Basic event/reset ingredients exist; full statechart semantics remain |
| Hierarchical states | ⬜ | ⬜ | ⬜ | ⬜ | Entry/exit, ancestry, and transition priority remain |
| Parallel states | ⬜ | ⬜ | ⬜ | ⬜ | Deterministic orthogonal-region semantics remain |
| History states | ⬜ | ⬜ | ⬜ | ⬜ | Shallow/deep history and serialization remain |
| Temporal logic and assertions | ⬜ | ⬜ | ⬜ | ⬜ | Runtime monitors and property evidence remain |
| Bounded periodic hybrid scheduling | ✅ | ✅ | ⬜ | ⬜ | One exact continuous/phase-zero-tick/hold/physical-restoration reference slice is verified. The shared session additionally admits exact multiple-clock boundaries and the bounded event coincidence profile above; Guard execution and general scheduling remain; [`hybrid.packaged-dc-motor-controller`](../../verify/hybrid/packaged-dc-motor-controller/README.md) |
| Transient spatial-time coupling | ✅ | ✅ | ⬜ | ⬜ | 2D fixed-domain transient runs cover collocated FVM Navier–Stokes and conservative scalar transport (upwind/TPFA), with first-order time refinement checked for the flow case. Not yet: other time methods, restartable history, moving meshes, multiphysics trajectories, production runs; [`fluid.cell-centered-navier-stokes-fvm-2d`](../../verify/fluid/cell-centered-navier-stokes-fvm-2d/README.md), [`fluid.cartesian-advection-diffusion-fvm-2d`](../../verify/fluid/cartesian-advection-diffusion-fvm-2d/README.md) |
| Semantic checkpoint/restart | ✅ | ✅ | ✅ | ⬜ | Fixed-step implicit-Euler lineage |
| Adaptive solver continuation | 🟨 | 🟨 | ⬜ | ⬜ | Explicit Tsitouras45 runs save a checkpoint (controller state, dense output, events) that resumes bit-for-bit like an uninterrupted run, also from a file in a new process; Python offers `ExecutionCheckpoint`, `RunProgress` and `RunCancellation`. Not yet: BDF, sensitivities, MPI or GPU continuation, a registered continuation case. |
| Fault-tolerant trajectory continuation | ⬜ | ⬜ | ⬜ | ⬜ | Durable distributed/backend state and recovery semantics remain |

