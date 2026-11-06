# Proposed three-mesh correctness and ordinary timing protocol

This follow-up retains the same finite-channel linear Stokes problem, MINI
velocity/P1 pressure, full-stress traction outlet, force/flux definitions and
external-only engineering-budget algorithm in [the comparison](README.md).
The original single-mesh result and every original reference/candidate artifact
remain unchanged. No additional solves or measurements have started.

## Input-declared meshes

| Level | Maximum target size [m] | Maximum boundary error [m] | Maximum boundary facets |
| --- | --- | --- | --- |
| coarse | 0.05 | 0.0004 | 25 |
| standard | 0.025 | 0.0001 | 50 |
| fine | 0.0125 | 0.000025 | 100 |

Minimum mean ratio remains 1e-5. The standard controls are exactly the original
example controls. Halving volume size quarters the circle sagitta allowance
and doubles its facet budget: for radius R=0.05, the regular N-gon sagitta is
R(1-cos(pi/N)), below each declared allowance. These are input controls, not
predicted cell counts or an assertion of a convergence order. No mesh is
selected from a candidate result.

Generate each mesh through the ordinary public Gmsh provider; retain its exact
public mesh artifact, topology, named boundary identities and measured quality.
Independently calibrate A/B/C on each shared affine mesh, freezing its manifest
before that mesh's candidate runs. All original tolerance formulas, constants,
scales, residual norm, caps and falsifiers remain unchanged. A failed or
inconclusive level is reported rather than omitted or relaxed. Changes across
meshes are sensitivity observations; no exact continuum drag or convergence
claim follows without separate evidence.

## Ordinary execution boundary

Use the existing shared runner: one cold call, one separate diagnostic call,
two warmups and seven ordinary samples per fresh tool/mesh child. There are
11 executions per child in benchmark mode and two in correctness-only mode.
Keep one CPU thread/rank, hidden GPUs, explicit pinned CPU and original raw
attempts. A root-reviewed process collector will pass named case descriptors
and frozen reference identities; this directory does not own another process,
clock or sampling loop.

Eqiora times its ordinary `run(plan)` call. Public force/flux getters and field
inspection remain outside that clock. FEniCSx retains only mesh/space, compiled
UFL forms, immutable basis transformation/common layout and observation lifts
between calls. Every ordinary call allocates and assembles fresh stiffness and
reduced system/RHS, applies basis congruence and inlet lifting, allocates a new
MUMPS factorization/solver, solves from zero and recovers a fresh mixed field.
No matrix, factor, preconditioner or solved state is reused. FEniCSx force/flux
functional assembly and dots occur in untimed extraction, matching the public
observation boundary. No independently timed component is subtracted from an
ordinary total.

A separate FEniCSx diagnostic brackets assembly, direct solve and recovery.
The native Eqiora diagnostic reports actual assembly/linear-solve phases from
the qualified optimized producer. A missing required phase rejects timing;
it never triggers a synthetic timer or an inferred partition. Allocation,
true-residual acceptance, field recovery and object-destruction scopes must be
reported explicitly. Internal phase scopes/object lifetimes are not asserted
to be identical across tools.

Mesh/provider work, compilation and fresh JIT preparation are separate from
ordinary execution. Candidate exports, exact norms, frozen external-operator
checks and force/flux inspection are untimed. Cold memory is captured by the
shared owner immediately after ordinary field return; final child memory also
includes diagnostics and reporting. Reference calibration and its retained
objects never count as a fresh timing campaign.

This protocol and implementation require primary review before any new mesh
counts, calibration, candidate execution or timing collection is authorised.
