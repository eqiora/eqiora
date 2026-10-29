# Differentiation, optimization, uncertainty, and data

[Capability matrix](../capability-matrix.md) · [legend](../capability-matrix.md#how-to-read-the-matrix)

| Capability | C | X | V | M | Current implementation and planned extensions |
|---|:---:|:---:|:---:|:---:|---|
| Scalar SSA primal evaluation | ✅ | ✅ | ✅ | ⬜ | Shared operator IR |
| Forward JVP | ✅ | ✅ | ✅ | ⬜ | Scalar/static and bounded spatial/time actions |
| Reverse VJP | ✅ | ✅ | ✅ | ⬜ | Scalar/static and bounded spatial/time actions |
| Fixed-history spatial step Parameter products | ✅ | 🟨 | ⬜ | ⬜ | `CommonSpatialPlan::differentiate_step` gives Parameter derivatives of one backward-Euler step with the previous State fixed, for real and complex Q1/TPFA and MINI/P1; candidates can be checked against the original equations. Not yet: differentiating the Run that produced the previous State (see next row), moving geometry, other time methods, registered verification. |
| Accepted spatial step history composition | ✅ | 🟨 | ⬜ | ⬜ | `CommonSpatialPlan::differentiate_trajectory` gives forward and adjoint derivatives through a backward-Euler or implicit-midpoint Run by composing accepted steps; Rust `DifferentiableProgram::compile_run`, Python, JAX and PyTorch use it. Q1/TPFA tests match independent controls. Not yet: adaptive or event derivatives, moving meshes, temporal Observables, registered verification. |
| Implicit forward sensitivity | ✅ | ✅ | ✅ | ⬜ | Nonsymmetric static systems and bounded time paths |
| Implicit adjoint sensitivity | ✅ | ✅ | ✅ | ⬜ | Scalar objectives use one row of `DenseOutputLinearization` and the same accepted-output pairing and `adjoint_output_gradient` as general outputs, with an explicit unit cotangent. Transpose solve and verified scalar objectives |
| Materialized direct-output differentiation | ✅ | ✅ | ⬜ | ⬜ | A real 5x5 nonsymmetric reference solves forward and adjoint derivative systems with faer `SparseLu` in both orientations and replays Relation JVP/VJP independently. Not yet: factor reuse, other backends, general materialization; [`differentiation.materialized-direct-output`](../../verify/differentiation/materialized-direct-output/README.md) |
| Complete-Field output linearization | ✅ | ✅ | ⬜ | ⬜ | Q1 vertex and TPFA cell Fields on 2D Cartesian meshes have primal, JVP and VJP at any Parameter point in Rust, Python, PyTorch and JAX, matching independent rebuilds and finite differences; any subset of Fields can be selected. Not yet: mixed spaces, non-rectangular or imported meshes, wider registered evidence; [`differentiation.spatial-poisson-fem-fvm`](../../verify/differentiation/spatial-poisson-fem-fvm/README.md), [`interfaces.python-differentiation`](../../verify/interfaces/python-differentiation/README.md), [`interfaces.python-pytorch-differentiation`](../../verify/interfaces/python-pytorch-differentiation/README.md), [`interfaces.python-jax-differentiation`](../../verify/interfaces/python-jax-differentiation/README.md) |
| Parameter sensitivities | ✅ | ✅ | ⬜ | ⬜ | Derivatives with respect to coefficients, sources, boundary values and time Parameters flow through coefficient expressions, including complex functions and coordinate maps, with domain checks kept; tests compare with hand-derived and rebuilt solves. Not yet: arbitrary-order implicit derivatives, user-defined operators, registered mixed derivatives; [`solid.mixed-boundary-elasticity-2d`](../../verify/solid/mixed-boundary-elasticity-2d/README.md), [`fluid.port-closed-si-mini-stokes-2d`](../../verify/fluid/port-closed-si-mini-stokes-2d/README.md), [`language.pure-calculus-support-map`](../../verify/language/pure-calculus-support-map/README.md), [RFC 0055](../../rfcs/0055-component-parameter-terms.md), [RFC 0056](../../rfcs/0056-pure-calculus-and-support-maps.md) |
| Shape sensitivities | ✅ | ✅ | ✅ | ⬜ | Cartesian geometry and fixed-topology simplex compliance |
| Real-linear complex derivative products | ✅ | ✅ | ⬜ | ⬜ | Complex expressions differentiate as real-linear maps with the pairing `Re(sum(conj(a)*b))`, including `conj`, `real`, `imag` and `abs2`; Programs accept real or complex Fields and Parameters with a real objective, and forward and adjoint solves agree with hand-derived results. Covered by product tests only. Not yet: external providers, branch continuation, complex trajectory derivatives. |
| Discrete time-step adjoint | ✅ | ✅ | ✅ | ⬜ | One residual-native implicit-Euler step |
| Multi-step trajectory adjoint | 🟨 | ✅ | ✅ | ⬜ | Four fixed steps across one checkpoint; general trajectories remain |
| Checkpoint scheduling for adjoints | ⬜ | ⬜ | ⬜ | ⬜ | Revolve-like scheduling, memory budgets, and distributed persistence remain |
| Adaptive/BDF adjoint | ⬜ | ⬜ | ⬜ | ⬜ | Accepted-step history and method-consistent reverse execution remain |
| Event-time sensitivity | ✅ | ✅ | ✅ | ⬜ | One transversal explicit-ODE event group |
| Reset Jacobian | ✅ | ✅ | ✅ | ⬜ | One grouped monomial implicit reset |
| Saltation operator | ✅ | ✅ | ✅ | ⬜ | One event slice; general hybrid modes remain |
| DAE event sensitivity | ⬜ | ⬜ | ⬜ | ⬜ | Consistent tangent spaces and mode reinitialization remain |
| Second derivatives/Hessian actions | ⬜ | ⬜ | ⬜ | ⬜ | Exact and Gauss-Newton actions remain |
| Differentiable linear/nonlinear solve reuse | 🟨 | 🟨 | 🟨 | ⬜ | Core contract exists; production caching and lifecycle remain |
| Gradient-based local optimization | 🟨 | ✅ | ✅ | ⬜ | One fixed-area aspect-ratio consumer; general optimizer API remains |
| Constrained nonlinear optimization | ⬜ | ⬜ | ⬜ | ⬜ | Bounds, nonlinear constraints, scaling, and KKT evidence remain |
| Topology optimization | ⬜ | ⬜ | ⬜ | ⬜ | Design fields, filters, regularization, and remeshing remain |
| Multidisciplinary optimization | ⬜ | ⬜ | ⬜ | ⬜ | Coupled derivatives, consistency, and workflow orchestration remain |
| Global optimization | ⬜ | ⬜ | ⬜ | ⬜ | Evolutionary, Bayesian, surrogate, and budget policies remain |
| Parameter estimation/calibration | 🟨 | 🟨 | ⬜ | ⬜ | The [measured-force notebook](../../examples/python/measured_force_calibration.ipynb) fits a mass with a hand-written Gauss–Newton loop over Model edits, ODE Parameter derivatives and typed speed residuals, checked against a known answer. Not yet: a reusable optimizer, spatial sensor fitting, likelihoods and priors, uncertainty estimates, hosted Colab validation. |
| Design of experiments | ⬜ | ⬜ | ⬜ | ⬜ | Experimental-design policies, aggregation, and statistical accounting remain |
| Uncertainty propagation | ⬜ | ⬜ | ⬜ | ⬜ | Monte Carlo, polynomial chaos, sparse grids, and error accounting remain |
| Reliability analysis | ⬜ | ⬜ | ⬜ | ⬜ | FORM/SORM, rare events, and confidence evidence remain |
| Sensitivity ranking | ⬜ | ⬜ | ⬜ | ⬜ | Local/global measures and uncertainty-aware reporting remain |
| Reduced-order modeling | ⬜ | ⬜ | ⬜ | ⬜ | POD/reduced basis/hyper-reduction, error indicators, and validity remain |
| Surrogate modeling | ⬜ | ⬜ | ⬜ | ⬜ | Training data identity, uncertainty, deployment, and retraining remain |
| System identification | ⬜ | ⬜ | ⬜ | ⬜ | Linear/nonlinear/state-space identification and validation remain |
| Derived remeshing-aware ML Dataset | ✅ | ✅ | ⬜ | ⬜ | A v4 dataset manifest derives typed features, targets, time-ordered train/validation/test windows and training-only standardization from the 2D remeshing trajectory, keeping mesh-local blocks ragged; replay checks the values. Not yet: interpolation or padding, dense batching, general splits, framework tensors, training, GPU loaders; [`artifacts.ml-dataset-remeshing-2d`](../../verify/artifacts/ml-dataset-remeshing-2d/README.md), [RFC 0067](../../rfcs/0067-derived-ml-dataset.md) |
| ML operator integration | 🟨 | ⬜ | ⬜ | ⬜ | Data-plane design anticipates it; no supported framework path |
| Physics-informed learning | ⬜ | ⬜ | ⬜ | ⬜ | Residual/objective integration, batching, and reproducible training remain |
| Ensemble and parameter sweeps | 🟨 | 🟨 | ⬜ | ⬜ | Parameter maps, stationary Run sweeps and finite-time Run sweeps share one host scheduler and keep order and duplicates; RC and scalar-wave frequency sweeps run this way, and time Results feed the moment estimator. Not yet: general Run graphs, caching, distributed scheduling. |

## Study and workflow layer

| Capability | C | X | V | M | Current implementation and planned extensions |
|---|:---:|:---:|:---:|:---:|---|
| Typed study definition | 🟨 | 🟨 | 🟨 | ⬜ | Realization and run artifacts exist; a study spanning many related runs remains |
| Independent evaluation-axis contract | ✅ | 🟨 | ⬜ | ⬜ | The [map-axis contract](../evaluation/independent-map-axes.md) defines mapped and shared inputs, zip/product composition, non-leading axes and empty outputs; native and Python plans support bounded point grids with one shared/mapped split. Not yet: sharing along selected axes only, general frontend transformations. |
| Ordered native evaluation composition | ✅ | ✅ | ✅ | ⬜ | Native maps run many points through one Q1/TPFA (and finite affine or Newton) program in bounded chunks and worker counts, keeping order and duplicates and reporting failed, cancelled or unstarted points; results match separate evaluations. Not yet: device or distributed scheduling; [`differentiation.bounded-parameter-study`](../../verify/differentiation/bounded-parameter-study/README.md), [`differentiation.bounded-parameter-study-private`](../../verify/differentiation/bounded-parameter-study-private/README.md). |
| Independent time trajectory mapping | ✅ | ✅ | ⬜ | ⬜ | `EvaluationMapPlan` runs many ordinary time Runs (deterministic, stochastic, adaptive with resets) with separate histories, keeping order, duplicates, cancellation and checkpoints; members match separately executed Runs in Rust and Python tests. Not yet: registered evidence, interacting particles, ragged arrays, trajectory AD in maps. |
| Ordered evaluation collection persistence | 🟨 | 🟨 | ⬜ | ⬜ | `EvaluationCollection` saves ordered request groups, Result references, sampling lineage and status, and Python reads and writes `.eqcollection` files that reopen in another process; statistics can be attached to complete groups. Incomplete groups stay incomplete. Not yet: registered persistence verification. [Collection persistence](../evaluation/independent-map-axes.md#recorded-collections). |
| Static analysis study | 🟨 | ✅ | ✅ | ⬜ | Scalar elliptic slices run; general physics study API remains |
| Accepted mapped first-order products | ✅ | ✅ | ✅ | ⬜ | Mapped Q1/TPFA (and finite affine or Newton) evaluations give JVP and VJP from retained or recomputed points, summing shared-Parameter cotangents in order; an analytic oracle checks them; [`differentiation.mapped-products`](../../verify/differentiation/mapped-products/README.md). Not yet: higher-order products. |
| Transient analysis study | 🟨 | ✅ | ✅ | ⬜ | Bounded ODE/DAE slices; general transient multiphysics workflow remains |
| Modal analysis study | ⬜ | ⬜ | ⬜ | ⬜ | Generalized eigenproblem, normalization, participation, and result semantics remain |
| Harmonic response study | 🟨 | 🟨 | ⬜ | ⬜ | RC and scalar-wave frequency sweeps run through Parameter maps and collections, keeping duplicate frequencies, failures, cancellation, retries and offline reopening; complex amplitudes come from `Result.output`. Not yet: factorization reuse across frequencies, harmonic derivatives. |
| Linear buckling study | ⬜ | ⬜ | ⬜ | ⬜ | Prestress, geometric stiffness, eigen extraction, and interpretation remain |
| Frequency-domain study | 🟨 | 🟨 | ⬜ | ⬜ | The shared ordered Parameter Run map supports the bounded harmonic RC and scalar-wave paths above. General ports/sources, postprocessing and prepared reuse remain. |
| Continuation study | ⬜ | ⬜ | ⬜ | ⬜ | Adaptive load/parameter continuation and branch tracking remain |
| Parametric sweep | 🟨 | 🟨 | ⬜ | ⬜ | Explicit ordered stationary Parameter points execute independent ordinary Runs with indexed failures, accepted sibling retention, saved per-member Plans and explicit retry groups. Cartesian design generation, execution caching, cross-Run prepared reuse and general aggregation remain. |
| Stable real Parameter sampling | ✅ | ✅ | ⬜ | ⬜ | Versioned SHA-256 uniform sampling assigns stable draws per sample and channel, with explicit common random numbers and fixed coordinates that do not shift other draws; tests cover replay, permutations, chunking and workers; [stable sampling](../evaluation/stable-sampling.md). |
| Finite ensemble statistics and estimator conventions | ✅ | ✅ | 🟨 | ⬜ | Complete maps compute weighted sum, mean, variance and covariance, with explicit IID conventions (n-1 moments, standard error) and gradients through mapped VJPs; a location-scale pathwise mean is available under declared assumptions. Not yet: general derivative-interchange results, weighted uncertainty, confidence intervals. [Estimator execution](../evaluation/stable-sampling.md#bound-finite-sample-moment-conventions) |
| Design-of-experiments study | ⬜ | ⬜ | ⬜ | ⬜ | Experimental-design policies, aggregation, and statistical accounting remain |
| Optimization study | 🟨 | 🟨 | ✅ | ⬜ | One shape consumer exists; reusable variables/objectives/constraints/algorithms remain |
| Calibration study | ⬜ | ⬜ | ⬜ | ⬜ | Measurements, likelihood, regularization, identifiability, and posterior artifacts remain |
| Uncertainty study | ⬜ | ⬜ | ⬜ | ⬜ | Random variables, correlations, samplers, estimators, and confidence reporting remain |
| Sensitivity study | 🟨 | ✅ | ✅ | ⬜ | Bounded implicit sensitivities; general study orchestration remains |
| Coupled multiphysics study | 🟨 | ⬜ | ⬜ | ⬜ | Coupling ontology exists; monolithic/partitioned execution remains |
| Co-simulation study | ⬜ | ⬜ | ⬜ | ⬜ | Participant lifecycle, time negotiation, rollback, and convergence remain |
| Workflow dependency DAG | 🟨 | 🟨 | 🟨 | ⬜ | Artifact lineage exists; a typed reusable study DAG remains |
| Content-addressed workflow cache | 🟨 | 🟨 | 🟨 | ⬜ | Individual artifacts are addressed; safe cross-run cache admission remains |
| Partial rerun/incremental solve | ⬜ | ⬜ | ⬜ | ⬜ | Change-impact analysis, state reuse, and invalidation proofs remain |
| Workflow pause/resume | 🟨 | 🟨 | ✅ | ⬜ | One semantic restart exists; general study-level persistence remains |
| Per-task retry and failure policy | ⬜ | ⬜ | ⬜ | ⬜ | Deterministic retry admission, partial evidence, and escalation remain |
| Resource-aware workflow scheduling | ⬜ | ⬜ | ⬜ | ⬜ | CPU/GPU/MPI requirements, queue placement, budgets, and priorities remain |
| Experiment tracking | 🟨 | 🟨 | 🟨 | ⬜ | Exact run provenance exists; searchable studies, tags, comparisons, and retention remain |
| Digital-twin deployment study | ⬜ | ⬜ | ⬜ | ⬜ | Offline model, reduced deployment, live data, estimation, updates, and audit remain |
| Online state/parameter estimation | ⬜ | ⬜ | ⬜ | ⬜ | Streaming assimilation, uncertainty, drift, and safe update remain |
| Scenario and fault campaign | ⬜ | ⬜ | ⬜ | ⬜ | Scenario language, combinatorics, coverage, injection, and result triage remain |
| Requirements-linked verification | ⬜ | ⬜ | ⬜ | ⬜ | Requirement identity, test linkage, coverage, waiver, and signed evidence remain |

## Machine learning and scientific-data integration

ML is an adapter and consumer of the same typed operators and artifacts, not a
second execution semantics. Device location, copies, differentiation, training
state, and learned-weight identity must always be explicit.

| Capability | C | X | V | M | Current implementation and planned extensions |
|---|:---:|:---:|:---:|:---:|---|
| Explicit tensor shape/dtype/device contract | ✅ | ✅ | ⬜ | ⬜ | PyTorch takes rank-one contiguous CPU `float64` Tensors; JAX takes `float64`/`complex128` pytrees on one CPU device and supports up to 32 `vmap` axes. Registered spatial claims are Q1/TPFA. Not yet: other precisions, devices, general tensors; [`interfaces.python-pytorch-differentiation`](../../verify/interfaces/python-pytorch-differentiation/README.md), [`interfaces.python-jax-differentiation`](../../verify/interfaces/python-jax-differentiation/README.md) |
| Zero-copy CPU tensor exchange | ✅ | ✅ | ✅ | ⬜ | Read-only dense NumPy `float64`; framework tensors remain |
| Zero-copy GPU tensor exchange | 🟨 | ⬜ | ⬜ | ⬜ | DLPack/stream ownership and lifetime remain |
| Explicit host/device copy policy | ✅ | ✅ | ✅ | ⬜ | Device transfers are typed in CUDA slices; ecosystem-wide policy remains |
| Framework DLPack interoperability | ✅ | ✅ | ⬜ | ⬜ | PyTorch 2.14 receives primal values as owned CPU Tensors and exchanges Parameters and cotangents through DLPack with one staging copy. Not yet: zero-copy output, GPU streams, other frameworks; [`interfaces.python-pytorch-differentiation`](../../verify/interfaces/python-pytorch-differentiation/README.md) |
| ONNX model import | ⬜ | ⬜ | ⬜ | ⬜ | Operator subset, shapes, devices, differentiation, and weight provenance remain |
| ONNX model export | ⬜ | ⬜ | ⬜ | ⬜ | Export applies only to admitted learned subgraphs; semantic mapping remains |
| Learned relation/operator node | 🟨 | ⬜ | ⬜ | ⬜ | Must remain a typed relation implementation with explicit validity and weights |
| Batched simulation | 🟨 | ✅ | ⬜ | ⬜ | One native map scheduler runs Parameter evaluations and independent time trajectories; spatial Programs compose step derivatives before mapping, JAX maps multi-step Programs and PyTorch backpropagates through trajectories. Not yet: adaptive or event-bearing member derivatives, ragged arrays, interacting particles. See the time mapping row and the [execution guide](../python/execution-and-arrays.md#independent-time-runs). |
| Batched parameter gradients | ✅ | ✅ | ⬜ | ⬜ | Native JVP/VJP separate point and seed axes; JAX composes nested `vmap`, `jit`, mapped gradients and bounded `jacfwd`/`jacrev`, summing shared-input covectors. Host CPU `float64` Q1/TPFA only; [`differentiation.mapped-products`](../../verify/differentiation/mapped-products/README.md), [`interfaces.python-jax-differentiation`](../../verify/interfaces/python-jax-differentiation/README.md) |
| Accepted-point differentiable program | ✅ | ✅ | ⬜ | ⬜ | A differentiable Program binds one Plan (2D Cartesian Q1/TPFA, finite affine or Newton problems, or a backward-Euler Run) and gives primal, JVP and VJP at chosen Parameter points with typed, unit-aware real or complex values, shared by Python, JAX and PyTorch. Not yet: other Plan kinds, non-rectangular meshes, adaptive trajectories, higher derivatives, devices; [`differentiation.spatial-poisson-fem-fvm`](../../verify/differentiation/spatial-poisson-fem-fvm/README.md), [`interfaces.python-differentiation`](../../verify/interfaces/python-differentiation/README.md), [`interfaces.python-pytorch-differentiation`](../../verify/interfaces/python-pytorch-differentiation/README.md), [`interfaces.python-jax-differentiation`](../../verify/interfaces/python-jax-differentiation/README.md) |
| End-to-end differentiable trajectory | 🟨 | ✅ | ✅ | ⬜ | Four fixed implicit steps; adaptive, long, distributed, and general hybrid trajectories remain |
| PyTorch custom autograd | ✅ | ✅ | ⬜ | ⬜ | An optional PyTorch 2.14 adapter registers a custom operator with first-order backward; eager mode, `opcheck`, `gradcheck` and `torch.compile(fullgraph=True)` pass for CPU `float64` Q1/TPFA. Not yet: double backward, `vmap`, AMP, GPU, export, performance; [`interfaces.python-pytorch-differentiation`](../../verify/interfaces/python-pytorch-differentiation/README.md) |
| JAX typed-FFI differentiation | ✅ | ✅ | ⬜ | ⬜ | JAX 0.11.0 calls one program through CPU typed-FFI primal, JVP and VJP with real or complex pytrees; `jit`, nested `vmap`, mapped gradients and bounded `jacfwd`/`jacrev` work for Q1/TPFA, without Python callbacks. Not yet: sharding, `pmap`, higher derivatives, accelerators, export, performance claims. [`interfaces.python-jax-differentiation`](../../verify/interfaces/python-jax-differentiation/README.md) |
| Neural ODE/DAE integration | ⬜ | ⬜ | ⬜ | ⬜ | Learned residuals, adjoints, events, stiffness, and training evidence remain |
| Physics-informed neural training | ⬜ | ⬜ | ⬜ | ⬜ | Domain sampling, residual loss, constraints, scaling, and reproducibility remain |
| Neural operator integration | ⬜ | ⬜ | ⬜ | ⬜ | Field discretization identity, mesh transfer, conditioning, and validity remain |
| Graph neural network on meshes | ⬜ | ⬜ | ⬜ | ⬜ | Typed graph/field extraction, batching, partitioning, and equivariance remain |
| Hybrid physics/learned constitutive model | ⬜ | ⬜ | ⬜ | ⬜ | State/history, thermodynamic constraints, tangent actions, and extrapolation remain |
| Learned closure model | ⬜ | ⬜ | ⬜ | ⬜ | Feature provenance, invariance, uncertainty, and solver coupling remain |
| Surrogate training pipeline | ⬜ | ⬜ | ⬜ | ⬜ | DOE, dataset split, normalization, training, metrics, and model artifact remain |
| Active learning loop | ⬜ | ⬜ | ⬜ | ⬜ | Acquisition, simulation budget, retraining, stopping, and audit remain |
| Dataset lineage and licenses | ⬜ | ⬜ | ⬜ | ⬜ | Source, transformation, split, consent/license, and retention remain |
| Feature normalization artifact | ⬜ | ⬜ | ⬜ | ⬜ | Unit-aware scaling and deployment identity remain |
| Learned-weight artifact | 🟨 | ⬜ | ⬜ | ⬜ | Common artifact identity can host it; typed schema and framework-neutral metadata remain |
| ML experiment tracking | ⬜ | ⬜ | ⬜ | ⬜ | Config, seeds, environment, data, metrics, checkpoints, and parent study remain |
| Distributed data-parallel training | ⬜ | ⬜ | ⬜ | ⬜ | Process/device topology, collectives, failure, and deterministic policy remain |
| Model-parallel training | ⬜ | ⬜ | ⬜ | ⬜ | Sharding, communication, checkpointing, and placement remain |
| Mixed-precision training | ⬜ | ⬜ | ⬜ | ⬜ | Precision policy, loss scaling, numerical gates, and reproducibility remain |
| Inference batching and serving | ⬜ | ⬜ | ⬜ | ⬜ | Latency/throughput budgets, version routing, monitoring, and rollback remain |
| Learned-model uncertainty | ⬜ | ⬜ | ⬜ | ⬜ | Ensembles/Bayesian approximations, calibration, and propagation remain |
| Out-of-distribution detection | ⬜ | ⬜ | ⬜ | ⬜ | Validity domain, alarms, fallback physics, and evidence remain |
| ML model registry | ⬜ | ⬜ | ⬜ | ⬜ | Immutable versions, stages, approvals, signatures, and deprecation remain |
| ML inference code generation | ⬜ | ⬜ | ⬜ | ⬜ | Admitted operator set, target kernels, quantization, and equivalence remain |
| Physics/ML co-optimization | ⬜ | ⬜ | ⬜ | ⬜ | Joint design/weight variables, implicit gradients, constraints, and scale remain |

