# Laminar cylinder-wake experience

Implement and publish this first as an **Unverified product example**. Promote it only after its
claim-local candidate, independent oracle, falsifiers, and acceptance requirements pass after
the current evidence freeze is explicitly lifted for that scientific scope; evidence work must
not block the earlier product path.

Status: the unverified product path now executes ten nonzero accepted startup
steps and is available as matching plain-Python and Colab sources, plus an
accessible static-site walkthrough. It does not advance the scientific
`fluid.flow-past-cylinder` claim.

## Current unverified product boundary

The checked-in sources compose one Python-authored channel-minus-circle
Geometry, Gmsh common Mesh, packaged steady and transient `.eqi` Components,
typed numerical policies, a steady nonzero bootstrap, ten transient accepted
States through 0.1 s, typed cell-average vorticity, two continuous-P1 pressure samples, the
signed intrinsic-2D cylinder force action pair, and a caller-owned Figure. They
retain Geometry, GeometrySelection, Mesh, Model, Plan, Trajectory, State,
Field, point, unit, support, and operator lineage and label the result as
unverified. Caller-owned presentation includes a poster, ten-frame WebM/MP4,
first/final reduced-motion still, and a complete visible text alternative from
the same product Result. The `0.1.0a4` site identity also enables a GitHub-backed
Open in Colab entry whose ephemeral runtime reconciles the pinned Eqiora release
without replacing a supported preloaded Matplotlib or using maintainer-owned Drive state. Boundary force reports both the action on the fluid domain and the
equal-and-opposite action on the cylinder, explicitly in N/m.

This is not a developed wake: its short multi-step startup is largely symmetric
and attached. It has no periodic trajectory, force/probe time series, benchmark
normalization or coefficient, benchmark value, tolerance, convergence result,
or scientific acceptance. The animation is presentation, not evidence.

## Target experience and future public claim

The film shows that one exact channel-minus-circle Model executes as transient
incompressible Navier--Stokes flow, reaches a resolved periodic laminar wake,
and publishes pressure, vorticity, cylinder force, and time-series observables
from one accepted lineage.

The scientific target is the Schäfer--Turek 2D-2 configuration. The steady
2D-1 case and a smooth transient analytic or manufactured case are prerequisite
verification, not substitute evidence for the wake.

The film does not claim turbulent flow, production scale, general curved
meshing, general drag/lift postprocessing, or validation merely from visual
similarity. The current steady Stokes Studio example is neither the result nor
the scientific precursor by itself.

## Storyboard

| Presentation time | Content |
|---|---|
| 0--2 s | Exact circle, channel, named inlet/walls/outlet, velocity profile, viscosity, and Reynolds number |
| 2--11 s | Fixed-camera vorticity field; a sparse deterministic streamline projection may appear without becoming evidence |
| 11--15 s | Lift trace and shedding phase, with drag and Strouhal number visible in their source conventions |
| 15--18 s | Phase-matched return to the poster frame |

Vorticity is the sole primary field. Pressure belongs in the detailed view and
poster comparison, not as a simultaneous overlay.

## Accepted-result evidence plan

The evidence plan owns these distinct obligations:

- a smooth transient case such as `fluid.taylor-green` verifies temporal and
  spatial accuracy before the non-box benchmark;
- Schäfer--Turek 2D-1 checks the force and pressure-difference convention on
  the same geometry family;
- 2D-2 compares the periodic drag, lift, pressure difference, and shedding
  frequency with the accepted community ranges;
- mass balance, time-step refinement, spatial refinement, and a complete
  force-balance defect remain visible in the dossier.

The decisive observable family is
`C_D(t)`, `C_L(t)`, front/back pressure difference, and Strouhal number in the
benchmark's normalization. The experience is rejected if the steady precursor
misses its independently derived comparison band, or if the reported shedding
frequency fails its precommitted time-step-refinement check. A plausible
vortex street cannot override either failure.

Expected community values and tolerances are owned outside the implementation
lane. Derivation-bearing convergence and balance fixtures use the dual
independent oracle gate.

## Capability and artifact dependencies

- non-box transient Navier--Stokes lowering over the accepted exact geometry
  and its mesh correspondence;
- physically scaled boundary traction/force and pressure-difference result
  projections;
- durable general 2D velocity/pressure trajectories and accepted vorticity;
- deterministic frame selection, 2D field playback, synchronized scalar
  traces, and the common gallery admission path.

The first film may implement only the renderer profile required by this fixed
2D scalar field and trace. It must not introduce a universal visualization
schema.

## Accessibility and promotion

The reduced-motion still shows one complete vortex pair, the current physical
time, the lift phase, Strouhal number, and the evidence route. Its text
alternative describes alternating shedding rather than relying on red/blue.

Promotion requires accepted smooth-transient, steady-cylinder, and periodic-
wake evidence; an accepted field trajectory with force and vorticity results;
and the common publication admission check.

## Primary source

M. Schäfer and S. Turek,
[“Benchmark Computations of Laminar Flow Around a Cylinder”](https://doi.org/10.1007/978-3-322-89849-4_39),
1996. The maintained
[FeatFlow 2D-2 definition and comparison tables](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html)
record the periodic procedure and reported quantities.
