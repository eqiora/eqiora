# Fixed-topology ALE monolithic FSI in 2D

Status: verified for the bounded serial-host 2D slice below.

This case is the bounded evidence target for [RFC
0064](../../../rfcs/0064-fixed-topology-ale-fsi.md). It keeps reference
connectivity, entity order, Domain memberships, and the conforming interface
quotient immutable while accepted physical states produce exact current
geometry states.

One Realization-owned P1 harmonic action extends the accepted absolute solid
and interface displacement into a fluid mesh with at least one unconstrained
interior vertex. The common linear-solver contract executes that action. The
current coordinates, backward-difference mesh velocity, its spatial gradient,
the endpoint metric rate, and the affine GCL correction are derived together;
none is an independent input.

The fluid uses the ordinary conservative transient incompressible
Navier--Stokes meaning and a current-endpoint ALE weak action. The solid keeps
the ordinary first-order small-strain dynamic meaning in reference
configuration. Their velocity traces remain one algebraic quotient, and one
serial-host damped Newton path solves velocity, pressure, solid motion, and the
derived geometry dependency monolithically with backward Euler.

## Verified evidence

The Cargo integration target runs ordinary canonical lowering, resolved
Realization, numerical execution, and the moving artifact path. It establishes
all of the following in one reproducible case:

- direct canonical V5 source lowers to the complete typed ALE roles;
- zero motion reduces exactly to the fixed-domain local action;
- each moving affine cell satisfies `dJ/dt = J div(w)`; a nonempty set of
  GCL-active moving cells preserves the compatible constant-stream probe below
  `1e-12`, while omitting the correction leaves a witness above `1e-8`;
- the numerical trajectory independently reapplies harmonic interior motion
  and checks interface coordinates and consecutive-state mesh velocity before
  artifact publication;
- direct residual reassembly and centered differences check the accepted
  residual and every analytic Jacobian column; deterministic conservative
  coloring comes from typed cell closures, the quotient, topology, and sealed
  harmonic influence, whose driver columns remain singleton and whose
  aggregate singleton count is retained in accepted evidence;
- weak incompressibility, solid kinematics, shared interface velocity, and
  interface action and power balance close at every accepted step;
- current and whole-path orientation and quality gates hold;
- three accepted states publish exact mesh-bound Field snapshots,
  GeometryStates, moving SpatialStates, and an immutable two-generation
  trajectory prefix; and
- step refinement is first order in one common reference-topology mass norm.

GeometryState artifact validation replays exact resource lineage, immutable
topology, current/path quality evidence, and the complete displacement driver
leaves. The shared dimension-parametric meshing relation independently certifies
current coordinates against the graph-selected harmonic action and solver bound.
Both 2D and 3D publication use the same negative probes for a positively oriented
but nondriven coordinate state and a substituted driver coefficient block. The
artifact layer does not depend on the numerical execution crate.

Run:

```bash
cargo test --locked -p eqiora --features faer --test fixed_topology_ale_fsi_2d
cargo run --locked -p eqiora-verify -- run --case fsi.fixed-topology-ale-monolithic-2d
```

This case does not claim topology changes, remeshing, AMR, ALE sensitivity,
FSI adjoints, shape optimization, GPU or MPI ALE, an exact moving-volume
discrete energy identity, production mesh smoothing, finite-strain structure,
contact, performance, or scale.
