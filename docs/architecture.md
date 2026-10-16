# Architecture summary

Eqiora separates mathematical meaning from its realization and history.

The [glossary](glossary.md) defines project terms and the boundaries between
similarly named concepts.

```text
Semantic Model Graph ─┐
Ontology View Registry├─ typed transaction ─ immutable revision
Realization Graph ────┤
Evidence & Artifacts ─┤
Action & Provenance ──┘
```

## Semantic Kernel

Kernel nodes define model meaning: Domain, Representation, Field,
Parameter, Port, Relation, Activation, Connection, ClockDomain, FiniteSpace,
IndexSet, and PureOperator. A model is
a network of implicit residual relations activated continuously, periodically,
by events, or by guards and connected through causal signals or conserving
physical connections.

Relations carry a topologically ordered residual-expression DAG. References to
current values, derivatives, `pre`, and `next` remain expression-level symbols
rather than reintroducing a top-level continuous/discrete state tuple. Exact
rational ClockDomain periods preserve multi-rate coincidence independently of
floating-point realization. Field role and activation declare evolution separately
from spatial support. Initial Relations carry simultaneous fresh-initialization
equations; they are neither Field literals nor solver guesses. Restart consumes
accepted State/history rather than reapplying fresh initial equations.

Dimensionless integers retain signed 64-bit components through the same typed
expression DAG, transaction, and Model artifact as real and complex values.
Arithmetic and explicit conversions check their bounds; evaluation does not route
integers through binary64. FiniteSpace owns a nominal ordered basis for signed
coordinates and nonnegative counts. IndexSet owns a separate bounded ordinal
domain. Neither is a spatial discretization Space. Nominal value types retain the
exact declaration, and structural Parameter dependencies prevent edits that would
invalidate a compiled IndexSet or occurrence family.

Spatial Relations use the same DAG with shape-aware `grad`, `div`, `trace`, and
`normal` operators. Two physics-neutral tensor structure operators complete a
bounded continuum composition: `symmetric_part` accepts an exact spatial
Cartesian `[d,d]` tensor on a Cartesian volume, while `isotropic_lift` maps an
invariant supported scalar to `s I_d`. Both preserve physical dimension and
nominal support. Relation-scoped `coordinate(axis)` atoms and dimension-aware
unary mathematics express spatially varying data in the same DAG. Cartesian
volume and oriented boundary Domains define continuous geometry;
`DefinedOn`, `AppliesOn`, and `BoundaryOf` edges make field support and
relation scope explicit. A continuum Representation states continuous meaning
without choosing a mesh, basis, or solver.

## Common initialization derivative owner

Architecture decision for [#1002](https://github.com/eqiora/eqiora/issues/1002):
`eqiora-sem` may depend one way on `eqiora-ir` to check the accepted fresh scalar
initial point with the existing typed Operator IR derivative owner. This is an
explicit same-layer composition; it grants no reverse edge or permission for
other L2 dependencies.

SEM still owns canonical value evaluation, simultaneous initialization, the
initial solver's rank threshold, activation, and state commit. The IR result is
an additional accepted-point Jacobian check before commit. SEM does not consume
an optimized backend, device runtime, lowered time problem, or backend-produced
expected value. Derivative products have independent analytic product tests;
sharing their implementation does not itself establish independent evidence.

Forward secants cannot certify a derivative at a singular point (`x*x=0` at
zero is the minimal counterexample). A second differentiation implementation in
SEM would duplicate the existing calculus owner. Moving that owner into the
kernel/schema or adding a new crate would widen the architecture beyond this
current use. The chosen edge reuses the existing pure typed derivative boundary
without changing canonical Model bytes or introducing backend execution into the
reference interpreter. Layer checks retain every other dependency restriction.

The same common admission also consumes `eqiora-time`'s pure matrix proof to
check regular continuous equations separately from the joint initial system.
This explicit one-way SEM→time composition reuses the exact binary-rational rank
owner and the constant-mass null-space criterion already used by runtime time
adapters. SEM does not consume a time backend, lowered time problem, or integration
policy. Moving the proof into core would add arbitrary-precision arithmetic to
the identity/unit layer; copying it would create a competing rank oracle. Initial
conditions cannot repair a high-index regular system, and a solvable discrete
step is not an index-one certificate.

## Common finite Parameter evaluation owner

Architecture decision for [#938](https://github.com/eqiora/eqiora/issues/938):
`eqiora-compiler` consumes `eqiora-ir` in one direction to evaluate closed, typed
finite Parameter expressions through `ComponentScalarization`. The compiler owns
name resolution, dependencies, type checking, inactive-branch checking and source
lineage. IR owns component ordering and numerical finite-map actions, including
LU-based real inverses and scaled complex quotients. This composition does not
select a backend, construct a Plan or admit a solver in the compiler.

Static Hamiltonian composition and execution must not carry separate contraction,
factor-permutation or inverse algorithms. Duplicating those algorithms in the
compiler would create competing numerical owners; moving the component evaluator
into schema/core would move execution into the mathematical identity layer. A new
crate would split an existing owner for one consumer. Reusing pure IR retains one
implementation and its existing work bounds. Independent expected values remain
in product tests and claim-local evidence, not derived from the shared evaluator.

The dependency check admits only this directed edge. IR must not consume compiler
source, AST or elaboration, and other compiler dependencies remain constrained by
the existing layer policy.

## Standard Ontology

Model, Coupling, Scale, Objective, Solver, and EvidenceSet are typed named
subgraphs over kernel nodes. They use `OntologyId<S>`, not graph-node `Id<E>`,
and are first-class in APIs, transactions, revisions, and provenance without
acquiring independent execution semantics.

## Execution

Execution has one explicit trust boundary:

```text
Eqiora source → syntax AST → typed local elaboration → flat Graph Transaction
                                                        ↓
ModelView → immutable Snapshot → validated KernelProgram → evaluator/lowering
```

The source frontend retains bytes and recovery-oriented syntax without graph
knowledge. The compiler layer resolves names and SI dimensions, lowers the
recursive source expression into the canonical DAG, and emits the same typed
Transaction as the Rust API. Graph commit remains atomic; compilation cannot
partially mutate a store.

The [converged language specification](language/core.md) describes the adopted target surface
and its review specimens. It distinguishes specified syntax from delivered compiler behavior.

For the bounded local-source slice of
[RFC 0021](../rfcs/0021-component-hierarchy-and-instantiation.md), compilation-unit
Connector and Component declarations enter a resource-bounded staging area.
The compiler resolves public interfaces and scalar bindings, rejects recursion
and private access, assigns collision-checked identities from deterministic
instance paths, and expands nested instances into the same flat kernel
vocabulary. Component and instance are not kernel node kinds. Source locations
remain in an immutable in-memory provenance sidecar, outside model identity and
the public model wire. No Transaction is exposed until the whole expansion
succeeds.

The bounded exact-package slice of
[RFC 0022](../rfcs/0022-exact-package-identity-and-resolution.md) adds one
pre-elaboration barrier without changing kernel meaning:

```text
package manifest + exact file inventory + complete exact dependency releases
  → bounded source admission
  → compiler-derived semantic content
  → candidate release + manifest-derived exact lock
  → replay every source claim under final exact namespaces
  → ordinary package release

exact resolution record + content-addressed store
  → verified release graph
  → parse and index every exact model source
  → compare compiler canonical declarations with every release
  → select either a self-contained root Model or a root public Component
  → for a Component, bind abstract supports to caller-owned Geometry by exact name
  → elaborate through the common compiler-owned root occurrence
  → ordinary Transaction / Model artifact
  → package compilation sidecar
```

The installed package-to-execution path selects the Component form. Its
package source contains no concrete `box(...)` or caller application roles;
Geometry and Mesh remain caller authorities. The resulting Model enters the
same root `resolve(Model, Mesh, typed policies) -> Plan -> run/submit` path as
local source. Package compilation is retained as lineage on the in-memory
Model, Plan, and Run, never as a package-specific resolver or execution mode.

`eqiora-package` owns closed manifest, semantic content, source bundle,
resolution, store, and compilation-record contracts but has no compiler
dependency. `eqiora-compiler` owns neutral namespace, source, dependency, and
source-import inputs but has no package dependency. `eqiora-api` is the only
composition point. Package manifests and resolution records contain canonical
exact dependency targets; source-local names exist only in explicit imports.
The current compilation sidecar is v2 with semantic-canonicalization epoch 2;
the reader rejects pre-1.0 v1 records instead of migrating or retrying them.
The public preparation operation accepts no author semantic payload and does
no discovery: callers provide the complete exact release closure in memory.
One optional Rust input adapter, specified by
[RFC 0027](../rfcs/0027-capability-rooted-package-directory-admission.md),
retains an explicitly opened directory capability and constructs that bounded
source input from only `package.json` and its normalized inventory. Post-root
components are opened handle-relative, without following symbolic links, and
under dedicated manifest/per-file/total resource limits. This is neither
directory walking nor an atomic multi-file filesystem snapshot; identity
covers the owned bytes actually read. Release preparation derives the lock
from exact release identities and closed manifests, then uses the ordinary
resolver and compiler path to reject missing, duplicate, unreachable, cyclic,
or semantically dishonest inputs before returning the root release.
The optional application-level local-directory resolver composes these existing
owners for package development: callers explicitly supply one root, its complete
dependency-directory closure, and one store. It admits all inventories before
opening the store, prepares the graph leaf-first, requires every compiler-derived
dependency identity to equal its parent's exact request, rejects unused inputs,
and returns the ordinary canonical `ResolutionRecordV1`. Python projects use this
same Rust operation; no client owns a second graph resolver. Project manifest and
lockfile transactions, version selection, Git/network fetch, and vendoring remain
outside this local-source adapter.
The in-memory identity and resolution contract is the default build;
directory authoring, replay, and installation are available only through the
explicit `package-filesystem` facade feature.

The read-only restart adapter in
[RFC 0028](../rfcs/0028-retained-local-package-store-replay.md) consumes that
same `ResolutionRecordV1` from explicit bytes and retains either a caller-opened
`Dir` or one explicitly ambient-opened store root. It reads only the exact
source-digest entry, with no-follow/nonblocking final open, regular-file check,
fallible bounded allocation, and a one-byte growth probe. The ordinary resolver
then revalidates every release and dependency identity before compilation. The
adapter never enumerates, installs, updates, or selects a package; an unrelated
store entry or replacement ambient path cannot influence the locked graph.

The write boundary in
[RFC 0029](../rfcs/0029-atomic-package-store-installation.md) is a separate
`DirectoryPackageInstaller`, not a wider `PackageStore`. It canonicalizes one
already prepared release, stages the complete wire under a create-new name
(mode `0600` on Unix) in the retained root, synchronizes and closes it, and
publishes the exact source-digest filename through an atomic hard link that
cannot replace an existing entry. Equal existing content is idempotent;
invalid or different occupants fail closed through the ordinary bounded store
reader. The success receipt makes any deferred post-commit staging cleanup
visible. Unix v1 is a single-principal store contract; shared-store permission
policy remains separate.
This is one-release atomic visibility, not lock mutation, package selection,
multi-package commit, directory-entry crash durability, or staging garbage
collection.
The compilation sidecar binds the resulting Model digest to the exact root,
resolution digest, source bundles, and toolchain versions. It does not modify
the current Model bytes.

Package execution lineage is a second, optional identity edge.
`PackageRunBindingV1` binds that package-compilation digest and shared Model
digest to one caller-designated, model-matched `RunManifestV1` identity.
`eqiora-api` checks the Run's Model digest and semantic revision before creating
the edge, and replay revalidates the resolution-record identity and inventory,
compilation, Model, and Run identities. Neither the Model nor Run wire is
mutated, and the edge does not independently prove that execution occurred or
that numerical results were accepted. The registered evidence constructs it
only after its numerical and provenance checks pass. The
[`hybrid.packaged-dc-motor-controller`](../verify/hybrid/packaged-dc-motor-controller/README.md)
case applies that rule to an output-less Run v1 after accepting a joint
physical/periodic reference trajectory; its in-memory `PhysicalSample` values
are not a durable general result artifact.

[`PackageExecutionBindingV1`](../rfcs/0032-typed-package-execution-lineage.md)
is the distinct typed edge. It additionally names one exact semantic revision,
`RealizationEnvelope`, and package-neutral, Model-linked `RunManifestV2`. `eqiora-api`
validates the complete concrete Model/Realization/Run chain before construction
and repeats resolution plus artifact validation during replay. Package,
artifact, and run bytes remain unchanged; neither edge attests execution or
proves numerical acceptance by identity alone. The registered packaged
isotropic-balance application constructs this edge only after accepting the
unchanged 2D elasticity execution path.

`KernelProgram::from_snapshot` owns the selected definitions at one revision
and accumulates whole-model diagnostics. It requires a topology closed under
semantic edges, exact agreement between expression symbols and `DependsOn`
edges, dimensionally valid expressions, one Activation per Relation,
unambiguous periodic clocks, valid signal Connection nets, and nominal scalar physical Connection
networks. The reference interpreter and all compiler paths accept only this
validated form;
they never read a live mutable graph.

[RFC 0024](../rfcs/0024-scalar-conserving-connection-semantics.md) defines
nominal scalar physical Domains and Ports, explicit `Across` and `Through`
symbols, one closed-subsystem closure, and deterministic N-ary junction
residuals. `KernelProgram` validates that boundary and materializes the same
immutable `ComposedResidualSystem` for later execution. The current Model and
Transaction wires carry those values. Source exposes the same contract
through named connector members, for example `port.voltage` and `port.current`.
A flat domain uses `scalar_physical(across voltage: V, through current: A)`;
`port p: electrical` selects that exact nominal domain. Member names resolve source
lookup while the typed across/through roles own connection synthesis.

Within the hierarchy compiler, scalar physical `connect` declarations are
fragments, not additional Kernel Connections. That path first validates exact
nominal compatibility, retains definition-local public boundary partitions
without inventing occurrences, then normalizes the selected hierarchical
occurrence tree into pairwise-disjoint maximal sets. Each set emits one
ordinary flat N-ary Kernel Connection. Signal Connections do not enter this union and retain their
duplicate-use rules. Relation ownership is checked independently from
topological membership, so idempotent physical reconnection cannot hide a
second constitutive owner.

An ownerless public Port used only to forward a physical net is eliminated
before graph identity is sealed; retaining it would invent an unowned physical
unknown. Its source name is absent from ordinary entity symbols and is never
aliased to the resulting Connection or to an arbitrary retained Port. The
Connection's provenance retains every complete definition/instance/binding
origin, and its canonical owner scope is the least common ancestor of
contributing fragment-owner occurrence paths, with an explicit contributing
fragment required at that scope. A separate compiler sidecar preserves the
eliminated exposure's full identity, exact nominal contract, non-graph
provenance, final Connection, and the retained endpoint cut reached only
through fragments declared inside that occurrence.

[RFC 0036](../rfcs/0036-physical-exposure-projection-artifacts.md) seals those
compiler cuts for exact package compilation without changing Model meaning.
Each projection digest covers only the eliminated exposure, final Connection,
sorted interior cut, and scalar or field-boundary contract; presentation
selector, source provenance, Model, and package compilation do not enter that
meaning digest. The enclosing versioned catalog binds the complete projection
set and complete definition/instance/binding provenance to the exact Model,
semantic revision, and package-compilation digest. Artifact validation checks
the retained graph structure and connector/support contract. The public API
performs the stronger replay: it revalidates the resolution and compilation,
reseals the compiler-owned cuts and provenance, and requires exact catalog
equality. A structurally plausible subset cannot become authoritative by
itself.

`Common` denotes the class across value or field trace; `NetOutward` denotes
the through or parent-outward flux sum over the exact cut. A separate post-run
binding names one of those quantities, the exact catalog projection, one
closed Run v1 or Run v2 digest, and one output digest already present in that
Run. It is value-free and remains outside the Run output set, avoiding a
digest cycle. The registered scalar package case closes this binding through
Run v1. A sealed Model artifact reference lets the validated current Model
enter the current Realization and Run v2 identity chain while preserving
its exact wire-domain digest. The registered projection cases do not yet use
that chain: the field package case
proves exact catalog identity and boundary-support replay, not execution or
field result storage. Values, sampling, mesh identity, and transfer remain
separate result/Realization contracts. [RFC
0037](../rfcs/0037-version-neutral-model-artifact-reference.md) defines the
identity boundary and its nonclaims.

Direct flat source and hierarchy source use the same bounded normalizer.
Hierarchical canonical
equivalence, direct-flat maximal-set membership, compiler-side scalar cut
projection, and fail-closed admission are registered by
[`language.hierarchical-connection-sets`](../verify/language/hierarchical-connection-sets/README.md).
[`packages.hierarchical-physical-boundary`](../verify/packages/hierarchical-physical-boundary/README.md)
then crosses one exact dependency boundary and requires equal package semantic
identity, root-LCA Connection identity, canonical Model, ordered affine
residual system, and analytic solution for N-ary and partitioned source forms.
Their source and compilation lineage remain distinct. [RFC
0033](../rfcs/0033-hierarchical-conserving-connection-sets.md) records the
accepted bounded topology contract and its complete conformance map.

Occurrence-bound spatial support is a separate compiler contract. A Component
may declare public volume slots with an exact ambient dimension and boundary
slots with an exact parent volume slot. Each instance must bind every slot to
an existing Cartesian Domain of the same kind, dimension, and `BoundaryOf`
parentage.
The compiler carries those exact bindings through nested and exact-package
occurrences, then lowers Component-local scalar Fields and Relations directly
onto the bound Domains. Support slots create no Kernel entity, alias, mesh, or
realization choice; their binding spans remain only in occurrence provenance.
Missing, duplicate, unknown, private, kind-mismatched, dimension-mismatched,
and wrong-parent bindings fail before a Model or Transaction is exposed. The
bounded claim is registered by
[`packages.component-spatial-supports`](../verify/packages/component-spatial-supports/README.md)
and [RFC 0034](../rfcs/0034-occurrence-bound-spatial-supports.md). It does not
by itself define vector/tensor Fields, field-valued Ports, transfer operators,
result queries, numerical realization or solve, or fluid/structure physics.
Those are independent contracts rather than unresolved parts of a support
slot. Field-valued boundary meaning is registered below, while durable
eliminated-exposure projection identity and replay are specified by [RFC
0036](../rfcs/0036-physical-exposure-projection-artifacts.md).

Occurrence-bound Fields add the corresponding hierarchy obligation without
adding a hierarchy node to the Semantic Model. A required public continuum
Field slot is first specialized with its exact bound volume support, then
matched against one visible enclosing Field by physical dimension, value
shape, frame, representation, and exact support identity. Nested forwarding
preserves the same target identity. Expansion rewrites Component Relations to
that ordinary Field and emits no slot entity, edge, display alias, wire
payload, or numerical degree of freedom. Source identity and complete binding
provenance remain compiler-owned. The local and exact-package contract,
including fail-closed admission and legacy identity stability, is registered
by [`packages.occurrence-bound-fields`](../verify/packages/occurrence-bound-fields/README.md)
and [RFC 0040](../rfcs/0040-occurrence-bound-field-slots.md). Pure operators,
boundary partitions, and numerical execution remain independent gates at this
contract boundary.

The first packaged solid application closes one of those independent gates
without widening RFC 0040. The exact
`Eqiora.Solid.LinearElasticity@0.1.0` package exports one
`IsotropicBalanceWithPotential2d` Component containing only a 2D volume
support, displacement and load-potential Field slots, `mu` and `lambda`, and
the isotropic balance Relation. The root package owns the body, four boundary
Domains, both Fields, load definition, and four homogeneous displacement-trace
Relations. Occurrence expansion feeds the ordinary flat current Model to the
unchanged name-independent elasticity lowerer; package and Component names are
not dispatch keys. The registered
[`solid.packaged-isotropic-balance-2d`](../verify/solid/packaged-isotropic-balance-2d/README.md)
case proves a complete verification-private identity bijection to the existing
explicit-flat Model meaning, exact alias/declaration/binding/file-order
invariance, equal lowered coefficients and solutions, Q1 L2/H1 convergence,
and a nonzero affine-potential force/reaction balance. It then replays exact
package compilation, the current Model, current Realization, Run v2, and
`PackageExecutionBindingV1`. This is not a boundary-partition or traction Port,
an FSI interface, or a general package executor.

The exact `Eqiora.Solid.LinearElasticity@0.2.0` release adds the next semantic
gate without changing that accepted Component. `IsotropicMechanicalInterface2d` requires
the complete exact exterior of one occurrence body and exposes one nominal
displacement/traction Port per exact Boundary. Its Relations bind the
occurrence displacement trace and parent-outward isotropic traction to each
Port. Static family elaboration leaves only ordinary exact-boundary Ports,
Relations, Activations, and conserving Connections. The registered
[`solid.packaged-elastic-boundary-2d`](../verify/solid/packaged-elastic-boundary-2d/README.md)
case proves exact package resolution, member-order and import-alias
invariance, and the complete typed boundary expression. Mesh facets, trace
spaces, essential elimination, and coupled execution remain Realization
gates.

The exact `0.3.0` release adds only `FixedDisplacement2d` and
`ZeroTraction2d` semantic terminals; it does not change the accepted balance,
Connector, or interface Components. The package-neutral elasticity lowerer
normalizes direct Relations and exact two-Port terminal networks to the shared
canonical inventory `TraceZero | FluxZero | PortBinding`. The current Cartesian Q1
Realization converts only closed zero-data dispositions to a private complete-
side constraint mask, rejects live bindings before mesh construction, and
passes no Semantic Model or package identity to assembly. The registered
[`solid.mixed-boundary-elasticity-2d`](../verify/solid/mixed-boundary-elasticity-2d/README.md)
case proves direct/package CSR, solution, convergence, constrained reaction,
and global-balance equality.

The exact `0.4.0` release adds first-order dynamic-solid meaning without
changing those four declarations. `IsotropicElastodynamicsWithPotential2d`
requires distinct displacement and velocity Fields and contributes only
`derivative(displacement) - velocity = 0` plus density-weighted momentum.
`ElastodynamicMechanicalInterface2d` uses the neutral exact
`Eqiora.Mechanics.Interfaces@0.1.0::VelocityTractionBoundary`: its trace is
velocity while its outward traction derives from displacement stress. The
package-neutral lowerer makes those two Field roles explicit, verifies
positive density and volume/boundary coefficient agreement, and retains live
Ports without selecting a mesh or time method. Direct and exact-package
lowered-meaning equivalence and structural falsifiers are registered by
[`solid.dynamic-linear-solid-semantics-2d`](../verify/solid/dynamic-linear-solid-semantics-2d/README.md)
and [RFC 0048](../rfcs/0048-dynamic-linear-solid-semantics.md). Spatial mass,
shaped displacement/velocity initial-state artifacts, time stepping, and FSI
remain separate Realization gates.

The separate pair Realization now consumes exactly one live two-Port binding
without weakening that single-body gate. It lowers two bodies independently,
proves that their only live sides form one opposite-side coincident interface,
and keeps both semantic Domains and both Cartesian Q1 meshes distinct. A
Realization-owned topological bijection maps paired interface vertices to one
quotient displacement DOF. Both cell operators therefore scatter into the
same interface rows: trace continuity is unknown identity and weak traction
balance is assembled equilibrium. The same ordered assembly operation also
retains each body-local cut system and a free-interface-row mask, so finalized
evidence distinguishes free-row weak interface equilibrium, external support
reaction at constrained endpoints, and first-order raw stress-traction
recovery. The
registered
[`solid.conforming-elasticity-pair-2d`](../verify/solid/conforming-elasticity-pair-2d/README.md)
case proves direct/package four-target algebra equality, heterogeneous
piecewise convergence, interface balance, external reaction, and fail-closed
topology. [RFC 0042](../rfcs/0042-conforming-elasticity-interface-realization.md)
records the bounded conforming decision; nonmatching transfer, Stokes, and FSI
remain separate Realization and physics gates.

The first fluid execution gate is deliberately numerical-only. A 2D affine-
triangle MINI Realization uses the shared runtime-dimensional simplex spaces:
velocity is componentwise hierarchical P1 plus one normalized cell bubble,
pressure is continuous P1, and a single global multiplier selects the
zero-integral pressure representative. One local contribution is projected to
both the reduced solve system and unconstrained full reaction system. The
resulting symmetric-indefinite KKT operator keeps its mathematical property at
the solver boundary and is executed by the deterministic reference MINRES
path. Discrete prescribed boundary flux, mesh connectedness, gauge multiplier,
pressure mean, CSR symmetry, true residual, and global momentum balance are
all acceptance evidence. The bubble vanishes on the complete boundary, so the
velocity trace remains the same P1 topology used by the first conforming solid
slice; this is an intentional future FSI seam, not an FSI claim. This
numerical-only case retains convergence and stability ownership under [RFC
0043](../rfcs/0043-simplicial-mini-stokes-realization.md).

The complementary immutable first fluid-package gate is semantic-only. The exact
`Eqiora.Fluid.Incompressible@0.1.0` package exports
`SteadyStokesWithPotential2d`, which binds one root-owned 2D volume, velocity,
pressure, conservative-force-potential Field, and positive constant dynamic
viscosity. Expansion contributes only the ordinary momentum and
incompressibility Relations; the root owns the force-potential definition and
complete zero velocity trace. Direct-flat and exact-package forms pass the
same name-independent whole-Model recognizer and exact offline package replay.
The registered
[`fluid.packaged-steady-stokes-2d`](../verify/fluid/packaged-steady-stokes-2d/README.md)
case and [RFC 0044](../rfcs/0044-packaged-steady-incompressible-newtonian-2d.md)
own that claim. The package itself still selects no mesh, method, constraint,
scale, solver, Port, or FSI policy.

The separate registered
[`fluid.fieldwise-si-mini-stokes-2d`](../verify/fluid/fieldwise-si-mini-stokes-2d/README.md)
bridge fresh-lowers both direct and exact-package Models into exact
Domain/velocity/pressure roles, then binds those identities to one
`(P1+bubble)^2/P1` field-wise Realization graph. Its Stokes adapter derives gauge
and weak-functional scales from positive `L/U/P`, pulls the physical mesh and
force-potential tape into normalized coordinates, and directly assembles the
dimensionless symmetric-indefinite system. Only an accepted reference-MINRES
solution is reconstructed into exact-ID coherent-SI fields and physical
balance evidence. A verification-only physical assembly proves the complete
reduced CSR/RHS congruence coefficientwise. This bridge is equation-aware but
package-neutral; natural/open boundaries, other stable pairs, production
MINRES, transient flow, and FSI remain separate.

[RFC 0046](../rfcs/0046-power-conjugate-mechanical-boundaries.md) adds the
first public fluid boundary without changing that execution kernel. The exact
`Eqiora.Mechanics.Interfaces@0.1.0` package owns a nominal power-conjugate
velocity/traction Connector and zero terminals; the exact
`Eqiora.Fluid.Incompressible@0.2.0` package depends on it and binds complete-
exterior velocity trace and Newtonian parent-outward traction. A small shared
lowering seam normalizes only exact side/junction structure. Elasticity and
Stokes retain separate constitutive matchers. Four zero-velocity terminals and
four direct zero traces reach equal Field-wise MINI algebra and physical
evidence. RFC 0047 subsequently realizes `FluxZero` as zero normal pressure;
live `PortBinding` still fails before mesh access because it needs an exact
trace-space Realization. The solid package's
displacement/traction virtual-work Connector remains nominally and
dimensionally distinct; no implicit velocity conversion or FSI is claimed.

[RFC 0047](../rfcs/0047-mixed-stokes-static-pressure.md) closes the first
nonzero fluid boundary without putting load data in Realization. The exact
`Eqiora.Mechanics.BoundaryLoads@0.1.0` package contributes one method-neutral
normal-pressure terminal over the immutable velocity/traction Connector. A
distinct root-owned pressure-valued Field carries the load; the Stokes lowerer
recognizes the resulting Newtonian traction only after the shared boundary
normalizer has proved the package-neutral junction structure. Three
zero-velocity sides and one static-pressure side produce a partial P1 trace
closure and remove the constant pressure nullspace physically. Consequently
the Field-wise plan has no `ZeroIntegral` constraint, multiplier block, gauge
scale, row, or reconstructed placeholder. The numerical implementation keeps
the 11-by-11 volume relation, optional 4-by-4 pressure-integral relation, and
4-by-4 constant-traction facet relation separate. The registered
[`fluid.mixed-static-pressure-mini-stokes-2d`](../verify/fluid/mixed-static-pressure-mini-stokes-2d/README.md)
case proves direct/package equality, nonzero facet action, pressure integral,
SI load/reaction balance, and two-profile reconstruction. Coordinate-varying
traction, open-flow laws, live trace transfer, ALE, and FSI remain separate.

Field-valued physical interfaces build on those occurrence bindings without
adding a universal resource or a mesh payload. One nominal boundary Connector
owns an exact trace/flux dual pair, SI dimensions, value shape, component
frame, and pairing. Each Port stores only that Connector and one exact boundary
Domain; its parent and outward orientation are derived from the unique
`BoundaryOf` edge. A conserving Connection is admitted only when all members
share the same nominal Connector and coincident Cartesian point set. It means
pointwise trace continuity plus parent-outward flux balance. The shared pure
typing pass creates one opaque componentwise residual proof, which Operator IR
then scalarizes in deterministic root/component order. The current Model and
Transaction contract carries this vocabulary explicitly. The bounded
exact-package 2D `[2]` claim is
registered by
[`packages.field-valued-boundary-interface`](../verify/packages/field-valued-boundary-interface/README.md)
and [RFC 0035](../rfcs/0035-field-valued-boundary-interfaces.md). Its exact
wrapper-exposure catalog additionally replays Connector and boundary-support
identity through RFC 0036 without storing pointwise values. Meshes, trace
spaces, transfer, numerical coupling, and result payloads remain outside the
Semantic Model contract.

Canonical tensor structure is a separate Relation-expression contract. The
pure identity-parametric typing rules are shared by hierarchical source
checking and committed semantic validation; flat source lowering remains an
uncommitted projection until that semantic admission. Pointwise component
scalarization implements
the exact direct/transposed coordinate rule for `symmetric_part(T)` and the
diagonal rule for `isotropic_lift(s)`; it does not pretend to discretize the
surrounding `grad` or `div`. The current Model and Transaction contract carries
these two expression variants. The registered
[`language.canonical-tensor-operators`](../verify/language/canonical-tensor-operators/README.md)
case proves canonical source lowering, semantic typing, pointwise
scalarization, invalid-expression rejection, and current artifact replay. It
does not claim elasticity physics, numerical realization, or solve. [RFC
0038](../rfcs/0038-canonical-tensor-structure-operators.md)
defines the bounded contract.

One deliberately closed elasticity projection now proves that this canonical
tensor vocabulary reaches numerical execution without acquiring a second
physics semantics. The specialized lowerer accepts exactly one 2D Cartesian
Model with a length-valued `[2] SpatialCartesian` displacement, a
pressure-valued conservative potential on the same continuum Representation,
continuous balance/load/trace Relations, constant coercive Lamé expressions,
and complete homogeneous boundary closure. It retains Parameter identities in
immutable scalar tapes and rejects any Kernel node the closed execution would
otherwise ignore. A separate typed Realization selects componentwise Q1,
Gauss quadrature, ordered assembly, CSR, CG, and one host worker. The same
potential tape supplies `grad(q)` by coordinate JVP, including the complete
load used for physical reaction recovery. The registered
[`solid.isotropic-elasticity-2d`](../verify/solid/isotropic-elasticity-2d/README.md)
case proves rigid/shear/dilatation behavior, an exact affine patch,
manufactured L2/H1 convergence, componentwise equilibrium, and
current-Model-to-Realization/Run-v2 lineage. This explicit-flat case is not
itself a material package or an FSI interface; the separate packaged
application above reuses the same lowerer and preserves the remaining RFC 0039
nonclaims. [RFC
0039](../rfcs/0039-canonical-isotropic-elasticity-2d.md) records the bounded
projection contract.

The public `ModelDocument::compile`, `compile_module`, and `replay` operations own the
single current Model contract and accept no artifact-generation selector.
Source-text callers use `compile`, client-neutral `Module` callers use `compile_module`,
and persisted current bytes use `replay`; all three converge before artifact
acceptance.
Canonical bytes expose the persisted `eqiora.model-envelope/v48` schema as an
output fact; the suffix is not a selectable authoring profile. Historical
Model bytes reject, and replay never sniffs, retries, or migrates them.
The bounded value-edit and scalar-elliptic application workflows retain exact
current artifact identity as a checked capability boundary. [RFC
0083](../rfcs/0083-current-model-artifact-epoch.md) owns this pre-1.0 epoch
reset and its compatibility nonclaims.

For the admitted static affine slice, each retained Relation DAG and generated
junction DAG lowers through `ScalarOperatorIr::bind_affine`. Known Parameters
and model time are bound explicitly; structural propagation proves
`R(w) = A w + c` without numerical probing while preserving canonical unknown
and residual order. The resulting nonempty square system is captured once as a
`General` `CanonicalCsrSystemView`, solved through the backend-neutral
`SolverPlan`, and accepted again by evaluating the original composed DAGs. The
registered [`electrical.parallel-dc-network`](../verify/electrical/parallel-dc-network/README.md)
case closes the flat path with serial faer BiCGSTAB. The bounded
[`language.component-elaboration`](../verify/language/component-elaboration/README.md)
case reaches the same seam after deterministic root-to-system-to-leaf
elaboration and agrees with the explicit flat solution. The reference
time-independent affine specialization remains distinct from the dynamic
reference interpreter and rejects models outside its admitted class.

The separate
[`hybrid.packaged-dc-motor-controller`](../verify/hybrid/packaged-dc-motor-controller/README.md)
reference slice resolves three exact ordinary packages and flattens them into
one current Model network spanning electrical and rotational conserving
domains, a causal speed signal, and one exact 10 ms periodic controller. Its host-serial
`f64` interpreter solves a 23-by-23 consistency system, applies the phase-zero
tick atomically, restores physical consistency, and advances continuous
relations with backward Euler and bounded dense Newton while holding the
controller output between ticks. Dimensioned `PhysicalSample` observations
reaccept every component and junction residual at a non-tick boundary and
support an electromechanical power/energy balance. An independent two-state
matrix-exponential oracle and a 2 ms to 1 ms refinement check bound trajectory
error. This is one scalar ideal linear motor, viscous lumped load,
proportional controller, and exact clock—not general DAE, nonlinear device,
multiple-clock, production-solver, MPI, GPU, code-generation, fixed-point,
real-time scheduling, or dynamic-plugin support.

The flat electrical case also closes the client-neutral authoring path. Its
immutable Domain, Port, Relation, and N-ary Connection handles are validated by
Module-local identity, retained in the shared typed AST vocabulary, and
sent through the same lowerer. The application facade passes the lowered transaction
through the current Transaction wire, commits atomically, reconstructs the current
Model artifact, and produces the same analytic physical solution as the
source-authored model. Native construction is an authoring surface, not a
second semantics implementation.

The executable-kernel v0 reference path evaluates scalar expression DAGs,
uses backward Euler for continuous Relations, and solves each active square
implicit system with an intentionally small dense finite-difference Newton
method. Coincident periodic Activations are grouped by exact rational model
time; `Pre` reads the shared pre-activation state and all `Next` values commit
atomically. A causal signal input aliases its unique output, and a periodic
output is held between ticks. These are normative activation/connection
semantics coupled to deliberately replaceable reference numerics.

Event Activations detect directed guard crossings inside accepted continuous
steps and localize the earliest root by re-solving the same implicit step.
Events coincident within the declared time tolerance—and periodic ticks at that
instant—form one activation set: all `Pre` reads observe the shared left limit,
and all `Next` resets commit atomically. A bounded zero-time loop returns an
explicit possible-Zeno diagnostic. Guard activation, conserving across/through
composition with Event/Guard activation, spatial execution inside the time
interpreter, multiple-clock physical scheduling, and non-square structural
analysis remain later milestones, not silently approximated by this path. The
packaged DC-drive case above is the sole admitted continuous/conserving plus
exact-periodic reference composition. Spatial Relations instead pass through
a separate validated realization lowerer. Hybrid derivatives are a separate
lowered analysis and do not alter these reference execution semantics.

Compiler, CPU, GPU, distributed, and fixed-step paths may optimize realization
but must match reference trajectories and residuals within an explicit
numerical contract.

The first CPU conformance path lowers each residual DAG to dense symbol slots
and scalar SSA instructions. It evaluates those instructions independently
while sharing the reference activation calendar and Newton/time-step engine.
This isolates operator-lowering correctness before introducing a second
scheduler or optimized sparse solver; it is conformance scaffolding, not a
performance claim.

Production time integration is a separate realization boundary. A compiler
first classifies each continuous Relation subsystem. Constant derivative
Jacobians enter the narrower `TimeProblem` as an explicit ODE or a
full/rank-deficient mass-matrix system. Structurally nonconstant or nonlinear
derivative dependence enters the separate residual-native `ImplicitDaeProblem`
with `F(t,y,y_dot)` and paired state/derivative JVP actions; it is never
coerced into a mass matrix. `TimePlan` owns method, tolerances, initial step or
fixed-step bound, and requested output samples; internal steps never become
ClockDomains. The optional Diffsol 0.16 adapter currently
verifies Tsitouras 5(4), BDF, consistent initialization for an index-one
singular mass matrix, and continuous forward parameter sensitivities for ODEs
and Parameter-independent constant mass matrices. Canonical Parameter order
and `f_p dp` come from the same scalar SSA program; constant-derivative proof
also proves `M_p = 0`. Diffsol still rejects Parameter-dependent mass and
general residuals. A deterministic dense implicit-Euler/Newton oracle verifies
one semi-explicit index-one residual with a state-dependent derivative
coefficient and IDA-style consistent initialization. Production general-DAE
execution, arbitrary-index analysis, backend-history continuation, trajectory
sensitivities, and hybrid DAE events remain unclaimed. A narrow semantic
checkpoint/restart path records an accepted `(t, y, y_dot)` point, replays its
residual from canonical Operator IR, and links parent and child runs without a
digest cycle; it does not serialize adaptive-controller or BDF history. One
accepted implicit-Euler step is separately lowered as
`G(y_next; y_previous, p) = F(t_next, y_next, (y_next-y_previous)/h, p)` and
verified through paired JVP/VJP, forward sensitivity, and a transposed adjoint
solve. This is a discrete one-step derivative, not a continuous or trajectory
adjoint claim.

Backend root finding also remains below hybrid semantics. Diffsol receives
only a `RegisteredRootProblem`: callback actions, canonical Activation-group
proof, and the SHA-256 identity of a versioned root-registration artifact.
That artifact links callback order to the immutable model digest/revision and
time-lowering digest. Every `RootProposal` retains this identity with its
candidate time/index/pre-event state, so an index has no meaning across
registrations. `CanonicalRootSet` independently rebuilds callbacks from the
canonical guards in proof order and rejects incomplete, split, combined, or
mismatched groups. Eqiora owns direction filtering, simultaneous event/tick
grouping, atomic reset commit, and explicit restart from the post-event state;
Diffsol's automatic reset is not used. See
[RFC 0014](../rfcs/0014-production-time-backend-contracts.md).

Smooth implicit differentiation keeps the canonical Relation unchanged. At a
finite accepted point, scalar SSA binds each input explicitly as unknown,
selected parameter, or frozen and exposes one scalar-parametric
`LinearizedRelation` contract with primal, JVP, and VJP actions. `eqiora-differentiation`
projects those actions into matrix-free state/parameter Jacobians and composes
them with the sole solver plan for forward and adjoint analysis. Transpose is a
separate operator capability; an Eqiora-owned oriented view sends an actual
VJP action through the same solver and independent true-residual acceptance
path. The smooth implicit claim is currently static, host-local `f64` storage.
`ComponentScalarization::linearize` retains each original symbol, component and
real/imaginary coordinate while composing the same scalar derivative graph into
one `LinearizedRelation`. Its pairing is the real Euclidean product, equivalent
to `Re(sum(conj(a) * b))` on complex values. Thus `conj`, `real`, `imag` and `abs2`
use real differentials without a holomorphic assumption; the gradient of `abs2`
includes its factor of two. Transpose of this real representation is distinct
from algebraic complex transpose, conjugate transpose and a space-metric adjoint.
Focused source-to-implicit-solver tests compare a nonholomorphic complex linear
problem's forward and adjoint objective derivatives with an analytic solution.
This native derivative path does not extend the common Python Newton admission,
complex trajectory derivatives, branch continuation or external constitutive
providers. A provider must explicitly establish its mathematical scalar domain,
real-coordinate correspondence and paired derivative products; an existing real
kernel is insufficient. Unsupported component operations fail explicitly.
Application differentiation derives immutable input/output signatures from the
Plan's admitted kernel and coefficient layout. `KernelProgram::symbol_type`
owns mathematical typing; `CommonSpatialPlan::kernel`, `field_space` and
`field_coefficient_entities` retain Geometry admission and exact coefficient
associations. The Program shares these signatures with accepted points. Storage
dimensions and shared/mapped Parameter ranges derive from the signatures rather
than independent ID/count inventories. Batch axes remain outside value shape. Typed `ModelEntityRef<Id<E>>` selections
can erase their compile-time kind into a `ModelEntityRef<RawId>` while retaining
the exact Model owner and runtime entity kind. Finite Programs use an ordered
selection of Fields and Observables, whose coordinate ranges share one accepted
relation and output linearization.
Execution placement stays owned by the exact Plan; there is no second
application-specific device enum.
A separate materialized direct-output reference binds an accepted relation/output
pair to its exact canonical CSR coefficient source while keeping the primal
source RHS distinct from derivative RHS values. The solver request applies a
faer sparse-LU factorization of those coefficients in normal or transposed
orientation; after backend true-residual acceptance, differentiation
independently replays the solution
through the accepted relation JVP or VJP. Existing accepted-output pairs remain
matrix-free. This adds no prepared-factor lifecycle, explicit transpose CSR,
cross-call reuse, or identity inference from matching algebra. A
spatial slice now retains canonical Parameter identity and analytic Parameter
JVPs in the method-neutral scalar spatial-expression tape. An analysis
explicitly selects and orders its design Parameters; all other model
Parameters remain frozen. Q1 FEM and orthogonal TPFA FVM then assemble their
method-native state action and coefficient/source/essential-boundary
Parameter action into the same `AssembledLinearizedRelation`. A 2D Poisson
case checks every forward state component and an adjoint objective gradient
against independently recompiled centered differences under both
discretizations. A second 2D slice selects canonical Cartesian Domain bounds,
linearizes the affine reference-to-physical map, and propagates coordinate,
Jacobian, measure, basis-gradient, facet-area, and cell-distance actions
through both Q1 FEM and orthogonal TPFA FVM. Full-state sensitivities and
adjoint gradients agree with independently recompiled centered differences;
an area-preserving log-aspect consumer performs Armijo descent toward the
square stationary shape. A third slice retains the same coordinate and
relation contracts on a fixed-connectivity `SimplicialMesh`: runtime-dimensional
simplex strata and orientation are separate from vertex coordinates, every
full-dimensional affine cell must have positive signed Jacobian and pass an
explicit mean-ratio quality gate, and a realization-local vertex-velocity map
supplies geometry JVPs. Continuous P1 FEM and a typed
`DenseOutputLinearization` with one output row lower both the residual and quadrature-defined
compliance `integral(source * u_h) dx`. In 2D, full-state and adjoint objective
actions agree with independently recompiled, rebuilt, quality-checked centered
differences, while compliance converges toward an independent rectangular
Fourier-series value. This remains fixed-topology discretize-then-differentiate:
positive orientation proves only cell-local affine injectivity, not global
non-overlap, continuous shape calculus, remeshing/adaptivity, or topology
change. A separate residual-native slice composes one implicit-Euler step from the same
canonical residual linearization. It treats `y_next` as the solved unknown,
binds `[y_previous, canonical Parameters]` as the selected parameter space,
and keeps time and step size as frozen realization data. Its JVP and VJP apply
the discrete chain rule without observing Newton iterations, and its
nonsymmetric forward/adjoint solves agree with centered differences. A
`DiscreteStepLinearization` additionally exposes accepted boundary state/time
and canonical coordinate identity. Four fixed implicit-Euler steps now compose
in reverse across one separately validated content-addressed semantic restart;
all step residuals and state/time/Parameter continuity are checked before any
transpose solve, and the trajectory gradient agrees with centered differences.
A verified hybrid slice lowers one canonical explicit-ODE event group with a
shared scalar guard and constant full-monomial implicit `Next` Jacobian. It
derives guard and reset JVPs from the same scalar SSA, solves the grouped reset,
and produces event-time forward sensitivity and the complete saltation matrix.
The `hybrid.registered-event` slice connects that analysis to a content-linked
Diffsol proposal and explicitly restarts the unchanged ODE from the post-event
state. Crossing-direction mismatch, exactly grazing events, and registration
drift fail closed. Distinct guards that localize simultaneously, periodic-tick
coincidence, priority, DAE events, mode changes, BDF/adaptive trajectory
adjoints, checkpoint scheduling, and hybrid-event trajectory composition
remain milestones under RFC 0011.

Python is an L4 adapter over the public `eqiora` facade, not a parallel model
implementation. Parsed source and frozen native role-bearing `Field`, `Parameter`,
`Initial`, and `Relation`, scalar physical `PhysicalDomain` / `ConservingPort`, and anonymous
N-ary `Connection` declarations converge before one typed compiler lowerer.
The bounded spatial authoring surface adds runtime-dimensional Module-local
Cartesian volume and boundary `Domain` identities, compiler-supplied continuum
Representation, supported scalar Fields and Relations, and closed `grad` /
`div` / `trace` expression forms. Registered source/Python equivalence and
execution evidence is one-dimensional.
Module closure checks exact handle membership; dimension, shape, frame,
support, operator applicability, and residual rules remain owned by the same
identity-parametric Semantic Kernel typing used for source models.
The native path uses the client-neutral Rust `Module` graph; direct compilation
neither manufactures source text nor reparses formatted output. Parsed and constructed
declarations share the same AST owner, with real source spans or graph-path diagnostics
respectively. Python `compile(source=module)` and `Model.from_bytes` use the same
current contract as Rust and Studio without a generation argument. Historical Model bytes reject; replay
does not sniff, retry, or migrate them. Both authoring paths cross the current
bounded Transaction envelope, commit atomically, and reconstruct the immutable
current Model envelope. The data plane groups common explicit-ODE samples by
exact Field, keeps owned host `f64` vectors lazy until a read-only NumPy
projection is requested, and makes copying explicit through
`Array.numpy(copy=...)`. Synchronous and awaitable ODE and bounded
scalar-elliptic runs use native worker lifecycles and detach from Python while
blocking. Scalar-elliptic execution exposes its accepted-boundary progress and
cancellation contract; common ODE execution claims neither rather than
fabricating a boundary.

The public spatial path is the physics-independent root resolver. A caller
supplies one exact Model, its caller-owned Mesh, and typed spatial, solve,
scaling, and temporal policies; resolution returns one Model-bound common
`Plan`. Physics is recognized only from the Model. Synchronous and awaitable
execution consume that Plan through the same worker and publish one common
`Result`; typed physics observations project only from the exactly paired
native execution output. Specialized scalar, fluid, and solid requests and
model-plus-plan execution overloads are absent.

These slices claim ordinary-GIL CPython 3.11--3.14, lazily materialized dense
CPU NumPy for common explicit-ODE series and complete 1D--3D generated-
Cartesian scalar-elliptic primary Fields, versioned copy-on-export DLPack
snapshots, and one host-serial lifecycle for bounded FEM/FVM execution. The
accepted-point differentiation adapter additionally admits exact CPU:0 DLPack
Parameter-point/JVP/VJP inputs through a no-copy protocol view, validates that
view, and owns one staging copy before native work detaches from Python.
The optional PyTorch 2.14 adapter projects that same immutable program as a
functional CPU `float64` custom operator. Its fake kernel is metadata-only,
its registered first-order backward calls a separate native VJP operator, and
its fresh DLPack outputs never alias Tensor inputs or accepted evidence.
Coordinate/mesh and portable-Realization arrays, general Run inputs,
solver-iteration progress or cancellation, worker/backend/solver selection,
production/MPI/CUDA execution, a general Python graph/PDE surface, durable
result Artifacts, free-threaded builds, `abi3`, zero-copy DLPack execution, GPU,
distributed/sparse arrays, JAX, PyTorch double backward/batching/export, and
general ML-framework interop require separate evidence under RFC 0012.

The installed `eqiora check <MODEL_PATH>` command is a separate L4 terminal
adapter over the transport-neutral `ModelDocument::compile` operation. It
owns only bounded local-filesystem admission, exact UTF-8 decoding, fixed
terminal projection and process exits. The accepted document remains the
operation's same fresh occurrence; the CLI exposes only its structural
comparison fingerprint and never routes through control-v2, Python or Studio. It writes no artifact and defines no reusable CLI, operation, schema,
wire or registry abstraction.

Time execution follows the same one-way path. A validated continuous canonical
Relation first lowers to scalar Operator IR. The first-order projection proves
a complete constant derivative Jacobian from SSA structure and recomputes its
represented binary-rational rank exactly. Only a full monomial matrix
normalizes residual permutation/sign/scale to an explicit ODE; every other
non-zero-rank constant matrix remains a full or rank-deficient mass matrix,
with consistent initialization required for the latter. Those classes expose
primal/JVP actions through `TimeSystem`. A structural variable-coefficient or
nonlinear-derivative obstruction instead produces `GeneralImplicitProgram`,
an explicit differential/algebraic partition, and `ImplicitTimeSystem`
residual/JVP actions. Sample evaluation and floating-point rank tolerances are
never used for either admission. `time.canonical-first-order` and
`time.diffsol-adaptive` cover the production first-order path;
`time.general-implicit-dae` covers the narrow residual-native reference path.
The existing first-order lowering/run artifacts intentionally reject the
reference-only `ImplicitEuler` method. Residual-native provenance uses separate
versioned envelopes for the general-lowering proof, supplied initial
state/derivative data, backend-accepted consistent pair, and run linkage. This
keeps consistency-solve output distinct from the caller's guess and avoids
weakening the first-order wire. An accepted checkpoint is a separate
content-addressed artifact with no run reference. The parent run lists its
digest as an output, checkpoint-derived `Provided` initial data starts the
child, and a distinct restart manifest closes those edges. Separating the
edge avoids both a digest cycle and backend-specific checkpoint types.

Equation admission remains inspectable after compilation. A versioned time
lowering envelope binds the canonical model digest/revision to the Relation,
state order, equation class, complete derivative matrix, and exact rank. It
can independently recheck coefficients and replay rank against Operator IR. A
root-registration envelope binds that lowering to canonical ordered Event
Activation groups without duplicating guard expressions in the wire. A
time-run manifest links the lowering digest to the sole `TimePlan`,
adapter/version, accepted report, and output digests. Residual-native
checkpoint and restart artifacts compose with those output digests rather than
overloading the first run format. Backend-native durable payloads, adaptive or
BDF history, and adjoint trajectory checkpointing remain separate contracts.

Numerical realizations live at L3 and do not add model meaning. The first such
kernel is a uniform-line scalar diffusion operator using centered second
differences, Crank–Nicolson integration, and a tridiagonal solve. Its analytic
decay test demonstrates second-order spatial convergence while deliberately
remaining disconnected from canonical `Field`/`Representation` lowering.
It remains a numerical precursor; the separate scalar elliptic path below
provides the canonical connection without retroactively changing its claims.

Spatial realization follows one method-neutral factorization:

```text
oriented Mesh + GeometryMap + Quadrature
                    ↓
        entity-local operator
                    ↓ anonymous dense contribution
              AssemblyMap
                    ↓
             global algebra
```

Mesh and quadrature dimensions are explicit runtime artifact data so imported
and mixed-cell models remain inspectable. Backend lowering may specialize
validated dimensions later. FEM cell forms and FVM cell/face fluxes produce
the same `LocalContribution`; neither method defines the shared contract.
Essential constraints belong to the assembly map, not to physics operators.
The concrete `CartesianMesh` uses one free-axis/anchor construction for every
entity stratum and supplies affine geometry in runtime dimension.
`SimplicialMesh` realizes the same `MeshTopology + MeshGeometry` seam from
explicit fixed connectivity, deduplicated strata, orientation permutations,
and affine entity maps. It rejects duplicate cells, isolated vertices,
non-manifold facets, inverted cells, and cells below a recorded mean-ratio
gate before assembly. `eqiora-meshing` owns this L2 contract so numerical
realizations and artifact decoding reconstruct through the same invariants.
`eqiora.simplicial-mesh-envelope/v1` records affine `f64` coordinates,
connectivity, the quality gate, and recomputed quality evidence under bounded
decoding. Its domain-separated digest is the only mesh identity stored by an
`ImportedSimplicial` Realization policy. The verified 2D P1 entry point checks
that digest and admitted dimension before assembly. An isolated L3 Gmsh
adapter accepts a resource-bounded ASCII/binary MSH 4.1 subset containing
planar linear triangles or linear tetrahedra and reconstructs it through the
same L2 constructor. Gmsh paths, entity/physical/result semantics, and parser
types do not enter the artifact. The contract does not yet prove global
non-overlap or provide XDMF, mixed/curved cells, partitioning, or adaptivity.

`eqiora.cartesian-mesh-envelope/v1` is the generated counterpart: it stores
only strictly increasing axes plus exact last-axis-fastest entity order and
tensor-product local-node order. Decoding reconstructs the meshing-owned
`CartesianMesh` and accounts for implied coordinates and connectivity against
the shared mesh budgets. A basis remains Realization/Field meaning, so the
mesh artifact says `hypercube`, not Q1.
No Cartesian index enters a local operator or assembler.

The first canonical spatial vertical slice precedes that factorization:

```text
strong Relation + Domain/Representation + coordinate/math
                    ↓ shape/unit/support validation
   Cartesian scalar elliptic model + immutable data tapes
                    ↓ declared realization policy
     P1/Q1 FEM cells ─┬─ orthogonal TPFA cells/facets
                     ↓
       opaque finalized LinearSystem + SolverPlan
                     ↓
       execution adapter → accepted LinearSolution
                     ↓
         method-native field + balance evidence
```

The canonical lowerer, scalar spatial-expression tape, Cartesian mesh,
reference topology/geometry/quadrature, P0/P1/Q1 spaces, and local operators
carry explicit runtime dimension. Separate 1D, 2D, and 3D manufactured Poisson
cases close the same path through resolved FEM/FVM plans, continuous-L2
convergence, and global balance. The multidimensional FVM evidence uses an
explicit dual-grid Q1 reconstruction rather than a cell-center-only norm.
Capability negotiation for the resolved canonical FEM/FVM path therefore
admits dimensions one through three only. Runtime-dimensional simplex
topology, affine geometry, P1 basis, and local assembly are separately
implemented, with 2D end-to-end PDE/shape evidence and 3D topology/geometry
unit evidence. Two 2D imported affine-simplex cases additionally close the
canonical scalar/P1 solve path with an exact one-degree-of-freedom oracle; one
starts from an official Gmsh 4.15.2 fixture. Vector/tensor fields,
mixed/high-order elements, broader external mesh semantics, nonorthogonal FVM,
global mesh validity, and adaptivity remain unclaimed.

Linear solution has its own backend-neutral boundary. `eqiora-solver` owns the
single `SolverPlan`, the host-local, allocation-free `LinearOperator` action,
typed backend capabilities, reports for accepted solutions, and a small
deterministic reference CG that serves as an oracle. `eqiora-assembly` owns
anonymous local contributions, constraint-aware assembly maps, and CSR as an
assembly product; its CSR implements the solver action without adding a second
solver configuration. Numerical realizations and execution adapters both depend
on this L2 contract and not on each other. Production faer, Rayon/MPI, and
device integrations live in dedicated adapter crates. Every accepted report
includes a residual recomputed with the Eqiora operator, so acceptance never
relies only on a library's recursive estimate.

Cartesian Q1 FEM and TPFA connect to these contracts at a single point.
Assembly's reduced `LinearSystem` is temporary: finalization copies its CSR and
right-hand side once into a `CanonicalCsrSystemView` and drops the reduced
storage. From then on, that view is the only reduced matrix and action used by
host solves, CUDA solves, residual re-acceptance, and assembled JVP/VJP actions.
The finalized problem also exposes the asserted operator properties, the
`SolverPlan`, the vector layout, the method identity, and assembly evidence;
eliminated-boundary recovery, full FEM reaction state, TPFA facets, and
dual-grid reconstruction stay private. A solver adapter is chosen only after
finalization.

Before reconstructing a field, `finish` takes an accepted `LinearSolution` and
rechecks it against the same view, the complete plan, normal orientation, the
iteration limit, and the resolved producer and verifier topologies. This is a
numerical check, not proof of origin: a vector that satisfies two systems is
accepted by both. Origin is established separately. Typed distributed envelopes
and their content-DAG check bind the Model, Realization, Run, system, partition,
and layout artifacts, and semantic derivation replay proves that the linked
algebra was lowered from that Model and Realization. `finish` reserves its
residual and method-private output storage fallibly and moves the accepted
vector and report instead of copying them. The existing one-call entry points
are thin `finalize -> backend -> finish` compositions. This path is verified for
generated Cartesian scalar Poisson with Q1 FEM and orthogonal TPFA on the
reference CPU.

The optional public CUDA facade passes the same finalized system to the
single-device CUDA solver for both methods and reconstructs the field only after
the host has re-accepted the residual. The graph-bound adapter path binds a
logical device and `QueueSlot` before any device allocation, admits the
finalized CSR and its operator properties, and records the process-unique
runtime `QueueId`, typed transfers, value generations, and the CUDA fences it
successfully waited on. Its immutable in-memory receipt exposes the complete
host output only after both the native serial verifier and a further serial
true-residual replay have passed. The physical CUDA test is ignored in ordinary
CI; instead, the machine-readable v2 case replays one committed device
observation on an ordinary host. That replay revalidates the normalized Model,
Realization, and Run artifacts, rebuilds the execution trace, and re-accepts the
recorded candidates through the finalized host operator. The fences in that
reconstruction are synthetic, so the replay checks structural consistency but
cannot re-attest the physical waits. None of this is hardware attestation, and
it does not extend to general PDEs or accelerator assembly. RFC 0023 records the
durable Model/Realization/Run seam; the execution receipt itself has no durable
wire format.

Thread placement does not change solver identity. An operator may expose
allocation-free actions over disjoint, contiguous ranges of output rows, as
assembled CSR does. `eqiora-backend-rayon` evaluates those ranges in a bounded
Rayon pool owned by the run, without touching the global pool, and the wrapped
solver keeps its own plan and backend identity. P1/Q1 FEM, affine-simplex P1
FEM, and Cartesian TPFA assembly use a separate indexed-packet contract: batches
may be evaluated concurrently, but the L2 accumulator scatters packets, targets,
and local entries in one reference order. A FEM cell packet feeds both the
reduced solve system and the full reaction system, so local physics is
evaluated once. TPFA packets always arrive in cells-then-facets order, so source,
interior-flux, and boundary-flux operators share the same backend-neutral path.
`AssemblyReport` and `SolveReport` each record their execution adapter and
worker count, and spatial solution objects keep both reports. Reference-CG inner
products use a fixed logical partition of 1,024 elements: Rayon evaluates the
partial sums concurrently and the numerical contract combines them in order, so
the worker count does not change the floating-point expression tree.

The verified scope is generated-Cartesian P1/Q1 FEM and orthogonal TPFA in one
to three dimensions, plus imported affine-simplex P1 FEM in 2D, each followed by
replicated CSR and reference CG. With one and with four workers, fields,
reconstructions, reactions and fluxes, balances, artifact-bound packet counts,
and numerical evidence are bit-identical. Adaptive, distributed, device, fast,
and NUMA-aware assembly are not covered.

Device execution has its own L2 contract instead of extending host slices.
`RuntimeId` and `DeviceId` identify discovered placement. Typed buffer
descriptors carry allocation identity, shape, element representation, and
residency. `QueueSlot` selects a logical deployment position, and a
process-unique `QueueId` identifies one concrete runtime instance; order within
a command queue is defined only by that complete identity plus monotone
submission numbers. `Completion` is unrelated to hybrid-model events, and
`TransferPlan` makes transfer direction and byte count explicit. The optional L3
CUDA adapter keeps cudarc contexts, streams, and allocations, together with its
private cuSPARSE library, descriptor, and workspace handling and its cuBLAS
vector actions, below these contracts. Its action path transfers finalized `f64`
CSR, runs deterministic or backend-native cuSPARSE SpMV, and compares the result
with the host CSR oracle under an explicit tolerance. Its solver path keeps the
assembled CSR and Krylov vectors on the device for one of three fixed
combinations: CG with Jacobi, general BiCGSTAB with identity, or
symmetric-indefinite MINRES with identity. It returns one candidate, which a
separate serial host verifier accepts using CSR and a fixed-order true residual.
`SolveReport` records producer and verifier placement separately. Solver,
operator property, and preconditioner cannot be combined freely; only these
three combinations exist.

The first graph-bound CUDA execution supports less than the solver adapter. It
accepts only finalized Q1/TPFA CSR with matching properties, Jacobi-CG with
`Fast` reduction, one device and queue owned by the run, and an implicit zero
initial value. Binding checks the logical `QueueSlot` and the known minimum
device payload before allocating device memory. Execution creates a
process-unique `QueueId` and records seven typed transfers, a single-generation
solution update, and real successful waits after the inputs, the solve, and the
output transfer. The known payload and the separately reported sparse workspace
are checked against total device memory, not currently free memory. The fixed
nine-step DAG ends with native serial verification, an independent serial
receipt replay, and an immutable receipt for the complete host output, bound to
the output fingerprint:

```text
TransferInputsToCuda -> AwaitCudaInputsReady -> SolveOnCuda
  -> AwaitCudaSolveCompletion -> TransferCandidateToHost
  -> AwaitHostVisibility -> AcceptWithNativeHostVerification
  -> ReplayTrueResidualOnHost -> AcceptHostComplete
```

CUDA library and driver versions are recorded in the paired Run and adapter
evidence, not in the receipt.

The fixed-reference FSI composition reuses this protocol. A host-reproducible
Realization and a one-device `Fast` Realization each call the same
equation-aware finalizer, and both must produce the same complete CSR/RHS
fingerprint before admission. The CUDA graph then uses symmetric-indefinite
MINRES with identity. The accepted host output passes the same generic
true-residual check and the existing FSI finish. The CPU and CUDA Realizations
have different identities by design; the semantic Fields, mesh, scaling,
pressure closure, state elimination, and finalized algebra are the same.

Action and solve evidence both record the device's actual compute capability
next to its Eqiora device descriptor. The public facade exposes the solver
adapter only behind the `cuda` feature; the raw graph-bound admission path is
intentionally kept out of the curated facade. The adapter is loaded dynamically
and is absent from default builds. Setup, host-to-device transfer, action or
solve, device-to-host transfer, verification, and total time are measured
separately. Arbitrary initial values, free-memory reservation, persistent
residency, multiple streams or queues, GPU assembly, matrix-free kernels,
stronger preconditioners, reproducible device reductions, general FSI or
MINRES, scale, multi-GPU, and broader MPI-plus-CUDA work are separate gates
under RFC 0019, RFC 0058, RFC 0062, and RFC 0063. The host-staged MPI-plus-CUDA
composition is described with the distributed FSI handoff below.

Matrix-free work starts from a second L2 boundary rather than a device library
IR. `LocalLinearActionIr` is one shape-homogeneous batch of anonymous
entity-local linear maps with entity-major packed inputs and outputs, plus an
ordered CPU evaluator. Cartesian Q1 diffusion lowers to this form in one to
three dimensions, and assembled CSR agrees with the gathered, local, and
scattered action within the verification tolerance. Mesh identity, global
numbering, gather/scatter, and reduction are separate contracts. RFC 0020
explains why the isolated CubeCL 0.10.0 CUDA/HIP experiment fails the `f64` and
MSRV graduation gates; it provides no accelerator support.

Distributed algebra is its own contract, not a widened host slice.
`GlobalVectorSpace` and `Partition` assign one owner to every global index. Each
`LocalLayout` has sorted owned indices and disjoint ghost indices, CSR columns
owned by another partition produce an ordered `HaloPlan`, and each shard stores
only its owned rows. Construction counts every owned row, nonzero, ghost
candidate, and halo transfer with checked arithmetic before reserving memory
fallibly. The operator holds each layout once; `LocalCsrShard` borrows the
layout and CSR instead of carrying a second layout copy with its own identity. A
single-process loopback oracle reserves its complete workspace, splits the input
by owner, performs the declared halo transfers, applies local rows, and gathers
the unique outputs. It also reduces a dot product from unique-owner
contributions in partition order; the unsupported fast policy is rejected.
Exact agreement for one, two, and four partitions verifies the layout, the
protocol, and the reproducible collective invariants independently of any
transport.

The optional L3 MPI adapter executes the same immutable halo and shard contract
through a generic complete-CSR bridge, tested on one, two, and four ranks. The
application initializes and finalizes MPI; the adapter checks thread support and
duplicates the communicator, and no MPI types reach L2.
`MpiExecutionGroup::admit` collectively seals the system, the complete verifier,
the single `SolverPlan`, and a fingerprint that includes the plan into one run
token that holds a mutable borrow. All halo, Krylov, reduction, status, gather,
and host-verifier storage is reserved before admission, and raw apply, dot, and
solve operations are not public. Collective steps are numbered monotonically and
failure records are synchronized before any communication; fault-injection tests
after admission run under a parent timeout. Gathers by explicit global index
rebuild one vector per rank, which is accepted through the captured complete
host action, and the accepted vectors and reports must agree. The registered
one-host case verifies the bridge at one, two, and four ranks with both
reproducible and native-fast reductions.

Earlier physical two-node evidence covers only the lower-level halo, reduction,
and CG path. It does not extend to this bridge or to anything built on it. The
bridge has no multi-node evidence, scaling results, checkpoint/restart, or
process-failure recovery. One spatial composition built on it is described
below.

The graph-bound path puts this bridge behind the common execution boundary. A
transport-neutral `DistributedExecutorDescriptor` binds the provider, a logical
`ProcessGroupSlot`, the rank count, and one worker per partition to a portable
`Distributed`/`f64`/offline/SPD/CG/Jacobi/`Reproducible` graph. MPI must already
be initialized: `MpiExecutionGroup` checks thread support and duplicates the
communicator before the observed rank count and capabilities are used for
numerical binding. Admission therefore happens before any numerical workspace
or collective exists, but after the communicator does. The MPI implementation,
version, and provided thread support are recorded in typed Run provenance;
communicator and rank-local MPI handles stay private to L3.

`AdmittedExecution` seals the complete CSR fingerprint, partition identity,
derived layout and halo identity, the plan-inclusive admission fingerprint, the
single plan, and a preallocated independent host replay. The MPI adapter
consumes that token and has no other entry point for a graph, system, or plan.
Its normalized trace has a checked capacity of `32 * maximum_iterations + 64`,
reserved before collective admission. The trace records only the synchronized
boundaries that actually occur, and requires dense global ordinals, bounded
iterations, every Krylov phase family, and the ordered final gather and
acceptance steps. The immutable receipt exposes this fixed macro DAG:

```text
AgreeDistributedAdmission -> SolveDistributedKrylov
  -> AgreeDistributedProducerReport -> GatherDistributedOwnedCandidate
  -> AcceptWithNativeHostVerification -> AgreeDistributedAcceptedResult
  -> ReplayTrueResidualOnHost -> AgreeDistributedReceipt
  -> AcceptHostComplete
```

Every rank first rebuilds the same complete candidate and accepts it with the
native serial host verifier; the ranks then agree on the accepted result. L2
independently replays the complete-host true residual and binds the final
receipt to the normalized output fingerprint. L3 then all-gathers a fixed-size,
domain-separated summary that covers the operator, output, dimension, producer
report, the partition, layout, admission, and process-group identities, and the
full normalized trace. The receive buffer for that summary is allocated when the
communicator is duplicated, before numerical binding, and only byte-identical
receipts are exposed. The registered Q1/TPFA case verifies this path with live
MPI processes at one, two, and four ranks on one host.

[RFC 0026](../rfcs/0026-distributed-spatial-layout-and-replication.md)
specifies the first composition with spatial realization and keeps two
boundaries separate. The generic algebra boundary provides the object-safe L2
storage projection, the validated complete-CSR action, distributed system and
layout identities, and the MPI bridge described above. The spatial boundary
finalizes Q1 FEM and TPFA into the same captured representation: the reduced
assembly storage is dropped, the replicated or distributed layout is kept, and
distributed production is admitted only as `HostCpu { threads: 1 }` with one
worker per partition. Method-native reconstruction additionally requires
independent one-worker host verification.

Typed durable system, partition, and layout envelopes rebuild and compare the
complete derived algebra, and one external content-DAG check binds them to the
Model, Realization, and Run artifacts. The registered
`numerics.canonical-cartesian-poisson-mpi` case completes this composition:
every rank decodes one Model artifact created by the parent process; the decoded
Realization policy reproduces the system bytes and digest; all six artifact
digests agree across ranks; the replayed finalized view and the rotated-cyclic
layout derived from the artifacts enter the bridge; and explicit global indices
and values are gathered, re-accepted on the host, and passed to the existing
FEM/FVM finish. A changed right-hand side whose links have been correctly
updated passes the content-DAG check but fails semantic derivation replay. The
executable scope is generated Cartesian 2D `f64` Q1 FEM and TPFA,
`Reproducible` CG with Jacobi, one host, one worker per rank, and one, two, or
four ranks. Assembly, mesh state, reconstruction, and the final field are
replicated. Distributed assembly, sharded result fields, scaling, and hybrid
rank/thread execution are not covered.

## Geometry identity, geometry-to-mesh correspondence, and kernel-neutral CAD boundary

The Semantic Kernel still owns only `Domain`, `BoundaryOf`, `Port`, and
`Connection` meaning; RFC 0049 adds no geometry node. `eqiora-geometry`
provides the pure chain, within one revision, from a semantic Domain to an
entity in one geometry revision and on to entities in one mesh revision. L3
artifacts bind that chain to the Model, geometry, and mesh digests and replay
the Cartesian embedding against the affine-simplex coordinates.

The same L2 crate owns the kernel-neutral CAD design, observation, and adapter
contracts used by geometry producers. They describe source identity and units,
fully constrained design intent, explicit uncertainty and modeling tolerance,
observations, and adapter identity, without accepting a CAD-kernel object or an
entity-enumeration index. Truck objects, STEP parsing, and B-rep modeling stay
in `eqiora-cad-truck`; no CAD kernel is part of the L2 boundary.

Geometry consumes RFC 0037's sealed, replayable Model contract instead of
interpreting the persisted wire format. Replay yields the artifact identity and
a validated, immutable Kernel Program together; a reference that carries only an
identity cannot stand in for the content. An incompatible future change to Model
meaning would need a new schema and a compatibility decision, and geometry would
keep consuming the same replay boundary.

Body cell sets partition the mesh completely and without overlap. Each parent's
complete relative frontier is partitioned by its boundary Domains. Two distinct
interface Boundary Domains may reference the same geometry entity and the same
mesh facets only if they have different parents; incidence then gives each side
its own outward view from its parent. Callers never supply a physical normal
sign, and the mesh `OrientationCode` is only a local vertex permutation.

The Cartesian producer carries one finite, positive, coherent-SI classification
precision as part of its Geometry Identity. Correspondence uses that same
precision for point and facet membership and has no separate mesh-local
tolerance. Exact bounds still determine entity topology; mesh quality, solver
acceptance, CAD healing, and future import uncertainty are separate policies.

Planar circular-hole Geometry v2 is a separate identity domain that does not
depend on provenance and does not reinterpret the classifier-bearing Cartesian
contract. Its content is finite bounds, centre, radius, and complete named
membership; it stores no graph, build, provider, lineage, or `tolerance_m`.
Separately, the authored-CAD producer projects retained and newly created
lineage into private result topology through its build owner, then checks that
the naming is complete and atomic before producing the common Geometry. The
private v1/v2 representations stay closed, classification precision exists only
for the kinds that carry it, and no public result handle or classifier is added.
Imported or otherwise untracked topology still needs an explicit classification
policy, with its own identity, at its correspondence or Realization boundary.
V1 decoding and its classification semantics are unchanged.

Retention across revisions is a separate, explicit, total bijection between
bodies. Domain ULIDs are local to a revision; paired boundaries are derived from
the retained parent pair and `(axis, side)`. Missing, split, merged, ambiguous,
partial, or stale relations produce no retained association. CAD-kernel
objects, STEP parsing, B-rep modeling, concrete adapter identity values,
transfer, ALE, remeshing, and FSI equations are outside this L2 contract.

## Fixed-reference fluid--structure lowering

RFC 0050 adds one narrow consumer of this identity seam without adding an FSI
entity to the Semantic Kernel. The Semantic Model stays an ordinary flat
network: an inertial incompressible Newtonian relation, a first-order
linear-solid relation, and one conserving velocity/traction Connection between
two boundary Ports. Package and direct authoring lower to the same physical
roles, and neither selects a mesh, time method, pressure policy, solver, or
coupling algorithm.

The physics-neutral multi-Domain Realization contract records the Domain and
Field inventories, Field-wise spaces on one imported mesh, one conforming trace
quotient, one backward-Euler state that is represented but eliminated,
symmetric congruence scales, operator properties, solver, target, and schedule.
The current Realization envelope embeds these choices in the portable graph;
no family-specific envelope or historical decoder remains. The FSI-aware adapter then checks that the roles
select MINI/P1 fluid velocity and pressure, P1 solid velocity and displacement,
equal trace scales, no pressure gauge, and the reference CPU execution contract.

Backward Euler eliminates `d_next = d_previous + dt * v_next`, so displacement
does not become an artificial algebraic block. Fluid and solid P1 interface
velocities share one quotient row, and the fluid bubble has zero trace. The
complete coupled operator determines whether constant pressure is closed; no
standalone Stokes gauge rule is copied into the coupling path. Assembly produces
one dimensionless symmetric-indefinite CSR system
`A_hat = D^T A D / Theta`, with `Theta = P U L` for the intrinsic-2D case. The
typed in-memory result binds the reconstructed physical coefficients to the four
Field identities and their support inventories while reusing the same captured
CSR. It also keeps the mesh reference and the complete physics-neutral
Realization plan, so a durable projection cannot accept a foreign solution of
the same shape.

The registered
[`fsi.fixed-reference-monolithic-step-2d`](../verify/fsi/fixed-reference-monolithic-step-2d/README.md)
case defines what is verified. It does not cover advection, multiple steps,
partitioned coupling, moving meshes, GPU/MPI, nonlinear solids, or
sensitivities.

## Durable moving spatial observations

`DiscreteFieldEnvelopeV1` and `FieldSnapshotEnvelopeV2` remain the shared numeric and semantic leaves for moving and remeshing results. They keep the full lineage: Model, Realization, geometry, correspondence, mesh, Field, support, dimension, shape, frame, association, and coefficient block.

`GeometryStateEnvelope` owns one dimension-explicit v4 wire for triangles and tetrahedra. Its closed initial, continuous, and remesh origins retain exactly the applicable path, velocity, predecessor or semantic-transfer references. Initial and continuous states replay their graph-selected harmonic coordinate relation from exact driver coefficient leaves in both dimensions. Remesh origins remain the bounded 2D same-time representation change; overlap and transfer receipts prove that change. Retired Geometry State generations and their adapter trait are removed.

`SpatialStateEnvelope` owns the current dimension-explicit v4 wire. Initial, continuous and remesh origins share exact context, complete Field inventory, driver and fixed-step time validation. Every continuous state binds both its immediate spatial predecessor and geometry predecessor; a complete `RemeshStateAnchor` retains source, overlap and transfer identities after a representation change. `ValidatedMovingSpatialContext` owns replay of the shared resources. Retired V2/V3 state types and codecs are removed.

`SpatialTrajectoryEnvelope` and `SpatialTrajectorySegmentEnvelope` own the current v4 trajectory wire. One complete immutable prefix includes initial, continuous and remeshed representations. State-origin validation and exact state/geometry edges have one private owner shared by segment replay and root summaries. A remesh segment replaces the exact source tip at the same step/time; a continuous edge retains the complete reference context and transfer anchor. The V2/V3 trajectory codecs and their separate resource budgets are removed. The old fixed-mesh V1 context, State, trajectory, Dataset view, and application-specific fixed-reference projection have been removed; fixed-domain execution publishes through the common Model → Plan → Run/Result path.

`DiscreteFieldStorageEnvelopeV1` is still an optional low-level chunking projection over the leaf bytes. It is not part of the identity of a logical Field, moving State, trajectory, or ML Dataset.

[RFC 0067](../rfcs/0067-derived-ml-dataset.md) defines the ML Dataset boundary. The current `MlDatasetEnvelope` uses the v4 wire over the complete trajectory and derives topology meaning from the unified spatial state. The manifest derives one strictly increasing time
sequence from a complete current trajectory, in which the remesh target replaces the
source tip at the same time. Typed descriptors keep the feature or target
role, window offset, Field, support Domain, coherent-SI dimension, value shape,
and frame. Every sample keeps the identities of its state, snapshot, mesh, and
complete coefficient blocks. Training, validation, and test partitions are
ordered and cannot share a state artifact.

Normalization uses a single standard-score policy fitted to the training
population. Its per-descriptor, per-association, and per-component statistics
are computed from active training support only, so validation and test values
never enter the fit. A constant channel records a population deviation of zero
and an applied scale of one. The artifact names the ordered binary64 Welford
accumulator profile and distinguishes an exactly constant input from a
deviation that merely rounds to zero. The L4 CPU projection checks all block,
active-entity, scalar, and statistics work in advance, then copies data
explicitly into ragged entity-major blocks that carry the active indices and
mesh identity. It does not pad across remeshing and does not treat mesh-local
indices as persistent physical identity.

The complete source-to-target dependency replay is independent of storage and works
without XDMF/HDF5. The optional temporal exporter and the Dataset adapter both
consume that replay profile. XDMF paths, HDF5 layout, framework tensors, and
device placement are not part of Dataset identity. Interpolation, dense
batching, other split and transform policies, training, and framework or device
loaders are left to separate typed adapters.

## Realization selection

Realization is a separate typed layer, not a bag of optional fields on the
Semantic Model. The scalar, field-wise, coupled, and transient plan families
remain authoring contracts and frozen-wire compatibility contracts. Their
resolvers reject unknown policies and any contradiction between method and
space, identities, solver, and target, with no fallback. A successfully
resolved value is then normalized into the common portable DAG defined by [RFC
0058](../rfcs/0058-portable-realization-and-execution-graphs.md): Domain
discretizations and Field spaces feed typed numerical transformations, one
connected algebraic system, explicit linear and nonlinear solve roles, and
portable placement requirements.

The DAG uses typed references specific to each layer instead of a universal
node or payload. Its Phase-A projections drive scalar serial/Rayon execution,
steady field-wise Stokes, coupled FSI, and fixed-domain transient flow.
Equation-aware identity information fills in facts that the compatibility plans
lack; no anonymous Domain, Field, Relation, or Connection is invented. Graph
validation checks structural closure, and each execution finalizer compares
that information with its own lowering before executing. Repeated step count
stays a Run directive. The current envelope binds exact Model and layout artifacts to the graph’s
canonical bytes. One literal codec specimen fixes its field order; ordinary
scalar, mixed, coupled and ALE tests exercise current authoring and replay.
Retired envelope versions are rejected.

The first deployment contracts are linear and sit downstream of this portable
graph. A pure host binding checks backend and adapter capacity and the solver
combination before creating a worker pool or numerical system. The scalar
finalizer accepts the portable graph, validates it against equation-aware
lowering, and retains it beside one finalized CSR system; the curated facade does not expose raw graph or
system admission. An opaque token seals the system's fingerprint, operator
properties, single solver plan, binding, and preallocated serial verifier.
Acceptance returns the complete solution with an immutable receipt only after
the reports match and an independent true-residual replay passes, and binds the
receipt to the normalized output-vector fingerprint. Serial and Rayon execution
therefore share the fixed `SolveWithNativeAcceptance ->
ReplayTrueResidualOnHost -> AcceptHostComplete` DAG while keeping their own
bindings and solver-native verifier reports; the receipt records the extra
serial replay separately. This host slice has no mutable graph builder, no deep
CSR clone, and no durable execution wire format.

The CUDA binding is the graph-bound path described under device execution
above, with the same nine-step DAG. Because host evidence replay rebuilds the
trace with synthetic successful fences, it cannot re-attest the physical waits.

[RFC 0062](../rfcs/0062-cuda-fixed-mesh-fsi.md) reuses that binding and DAG for
one fixed-reference 2D FSI step. The finalized FSI object keeps the portable
graph and the CSR together, so CUDA admission cannot pair FSI provenance with a
system supplied by the caller. CPU and CUDA finalization must produce
bit-identical CSR/RHS fingerprints, and the returned generic receipt must match
the graph, operator, solver plan, device, dimension, report, and CUDA trace
before the unchanged physical finish can use the solution. Evidence from the
selected device compares Field identity, support, ordering, length,
dimensionless coefficients, and scale-normalized physical coefficients with an
independent CPU result. This is portable recorded evidence; it is not hardware
attestation and does not show that device reductions are reproducible.

The distributed binding uses `DistributedExecutorDescriptor` and a logical
`ProcessGroupSlot` instead of an `Mpi` target. It seals the complete system with
its owner, layout, and admission identities, reserves one bounded trace for the
collectives actually performed, and runs through the isolated MPI adapter. Its
nine-step macro DAG (shown above) covers admission, a repeating Krylov region of
halo exchange, action, reduction, and update, producer agreement, gather by
explicit owner index, native host acceptance, agreement on the accepted result,
independent receipt replay, agreement on the receipt summary across ranks, and
complete-host acceptance. Runtime MPI identity and thread support are recorded
as Run evidence, not in the receipt. This completes Phase B of RFC 0058;
distributed mesh ownership and assembly are defined separately by
[RFC 0060](../rfcs/0060-distributed-spatial-ownership-and-assembly.md).

That spatial work has one transport-neutral L2 composition seam. A mesh-revision
identity and unique ownership of top-dimensional cells determine where
lower-dimensional entities reside, without involving distributed algebra. The
same cell ownership selects each local assembly packet exactly once, and the
actual equation support determines which partition owns each target row.
Payload-bound routes are sealed before an unordered inbox is accepted; owner
folds restore target, row, and global-packet order, and complete systems are
rebuilt only from checked owner shards. The registered fixed-reference 2D FSI
loopback checks one, two, and four logical partitions against an independent
complete CPU assembly. The corresponding one-host MPI case runs the same stages
on one, two, and four physical ranks, including synchronized rejection of a
rank-local foreign mesh revision before variable-size transport. MPI is only an
adapter over these stages and adds no routing or accumulation rules of its own.
This completes RFC 0060's assembly composition. It is not a distributed solver
and has no multi-node or scaling evidence.

The composition specified by [RFC
0061](../rfcs/0061-mpi-fixed-mesh-fsi.md) moves RFC 0060's reduced owner-row
payloads directly into rank-local CSR/RHS storage, derives the solver-vector
halo from that sparsity, and solves the unchanged fixed-reference
symmetric-indefinite operator with reproducible identity-preconditioned MPI
MINRES. Gathering by explicit index, re-acceptance of the complete system on
every rank's host, and the existing FSI finish remain separate acceptance
stages. The full assembly target stays in the lineage even though only the
reduced target enters MINRES. The reconstructed complete CSR serves only as the
host verifier; it is never repartitioned for a second solve.

Reproducible execution fixes the ordering for one admitted process-group shape.
A different rank count can change the floating-point reduction tree, so
[RFC 0063](../rfcs/0063-mpi-cuda-fixed-mesh-fsi.md) requires the model and
operator meaning to stay the same and the result to agree with the independent
CPU result within a tolerance, rather than bit-for-bit across rank counts.
Partition, halo, assembly-receipt, deployment, and observed runtime provenance
are specific to the rank count. There is no durable distributed Run artifact for
symmetric-indefinite systems.

The next composition, also specified by RFC 0063, keeps the MPI Krylov state,
halo, reproducible reduction, gather, and complete-host acceptance unchanged and
delegates only each owner-row sparse action to CUDA. A separate L3 composition
crate turns the shard into deterministic rectangular `[owned | ghost]` columns,
uploads its matrix once to one CUDA device per rank, and stages each input and
owned output through the host across three waited completion boundaries.
Host-owned and CUDA-delegated admissions are different token variants, so the
ordinary MPI entry point cannot quietly substitute its host action for CUDA.

The one-host evidence maps one physical device selector to ordinal zero on each
rank, has the ranks agree on their distinct live device UUIDs, and runs the same
finalized operator at one, two, and four ranks before the unchanged FSI finish.
It does not provide GPU-aware MPI, device-resident Krylov iterations or
reductions, multi-node topology, performance or scale results, GPU assembly, a
durable composite receipt, transient FSI, or ALE.

Model time and deployment time are separate. `ClockDomain` defines activation
semantics. `ExecutionSchedule` may set deployment priority and deadline but has
no model period or phase. A future task lowerer may connect the two through
typed provenance, but scheduling policy must never change the equation network.
The packaged DC-drive evidence checks this boundary: changing its sample period
changes the package and Model meaning, while changing only the host execution
topology keeps the Model identity and changes the Run identity. This is not
real-time scheduling.

Capability negotiation is a predicate over both the compatibility plan and the
explicit problem requirements. It checks discretization method, mesh kind,
spatial dimension, scalar type, replicated or distributed vector layout, solver,
preconditioner, reduction policy, target availability, worker bound, and
scheduling profile. Each generic declaration is an exact set of nested spatial
and execution contexts paired with solver capabilities, not a set of independent
axes. A registered falsifier accepts an imported-simplex P1 FEM/CG combination
and a generated-Cartesian P0 FVM/BiCGSTAB combination, but rejects their
cross-combinations before execution even though each part is valid on its own.
Space, order, quadrature, and other method-specific facts are checked by typed
plan validation. Legacy resolution without operator properties keeps its
candidate set, and either the equation-aware portable projection or an explicit
legacy finalizer seal rejects anything outside that set; property-aware
field-wise resolution checks the set directly, and numerical finalization
settles equation identity and coefficients.

The current scalar-elliptic reference path declares exactly `1D..=3D + f64 +
replicated + one host worker`, with generated Cartesian and imported
affine-simplex meshes. Cross-validation of method, space, mesh, and quadrature
restricts the imported path to continuous P1 with simplex-centroid quadrature,
and the imported PDE evidence is 2D only. Lower-level contracts that accept any
dimension at runtime do not extend that evidence. The interval API also rejects
any resolved dimension other than one. Unsupported dimensions, thread counts,
mesh kinds, and undiscovered CUDA ordinals are rejected. That ordinal exists for
legacy compatibility and for later accelerator Deployment binding; the portable
graph keeps only a device count per partition. The Phase-B slices bind selected
serial/Rayon capacity, one CUDA device and queue, or one distributed process
group, and record only the Solve, Transfer, Halo, Collective, and Fence
dependencies they actually execute. Future heterogeneous placements must be
built by combining these observed resources, not by reopening a cross product of
targets.

RFC 0026 separately specifies a `2D + f64 + distributed + one host worker per
partition` generated-Cartesian Q1 FEM/TPFA composition. Its 1/2/4-rank case
verifies artifact and semantic replay, collective admission, distributed solve,
ordered gather, host re-acceptance, method-native finish, analytic balance and
error, and agreement with the serial reference, within the one-host scope above.

## Artifact and verification boundary

A numeric version suffix marks a real compatibility obligation, not just the
possibility of future change. Persisted or externally exchanged schemas,
protocols, digest domains, and their decoders keep `V1`, `V2`, and later names
for as long as those representations must stay distinguishable. Historical wire
DTOs stay inside their compatibility modules and are lowered to the current
typed API before they reach numerical code.

In-memory APIs use unversioned names unless two supported public contracts must
coexist. Existing suffixed public Rust names are never silently redefined; they
change under crate SemVer, either through a new unsuffixed name or through an
explicit compatibility decision. Private numerical policy and implementation
types use mathematical or operational names, and changing such a policy does not
create a new data-format version. Artifact and protocol versions separately
govern persisted bytes and external exchange.

The unversioned public `ModelEnvelope` and `ModelTransactionEnvelope` carry the
single current runtime contract and use the persisted v45 schema identifiers and
digest domains. They serialize the current Semantic Model through wire DTOs and
rebuild it through typed constructors, one graph transaction, and
`KernelProgram` validation. Canonical JSON ordering and schema-domain-separated
SHA-256 digests are deterministic; the source revision is kept as provenance
but excluded from semantic content identity. Each JSON decoder first applies a
syntax-only byte and depth preflight, then its family's shared node, edge,
expression, view, and operation budgets, before mutating the graph. Model
v1--v44 schemas are rejected without sniffing, retry, or migration. Semantic
limits are not kept in one shared bag: the mesh, geometry, field, Realization,
remesh, physical-exposure, distributed, resolved-array/import,
trajectory/storage, time, and ML Dataset decoders each receive only their own
family's budgets. An artifact that embeds another family composes that family's
named budget explicitly, so changing one family's admission policy cannot affect
another's.

The sealed `CanonicalModelArtifact` boundary projects the validated current
envelope to one `ModelArtifactReference`: the wire-domain digest, the typed
Model identity, and the semantic revision. `RealizationEnvelope` consumes that
reference without changing its fields or bytes. The reference does not detect
or upgrade schemas. The registered evidence builds a coherent lineage of current
Model, current Realization, and Run v2, but does not lower or execute the physical
Model. [RFC
0037](../rfcs/0037-version-neutral-model-artifact-reference.md) and
[`artifacts.model-reference-lineage`](../verify/artifacts/model-reference-lineage/README.md)
define this scope.

Different authoring routes can express the same relation network and still,
correctly, produce different Model identities. To compare such models,
`eqiora-artifact` projects the validated `KernelProgram` to a generation-tagged
`StructuralSemanticFingerprint` (current generation V41). The projection removes
Model and entity occurrence ULIDs, source presentation, package identity, and
artifact identity, and keeps distinct vertices, nominal references, Parameter
values, field roles and activation, initial equations, expression structure,
semantic edges, physical connections, and Model boundary membership. Exact
partition refinement with bounded individualization selects one canonical
labelling; unknown meaning or exhausted limits cause rejection. The comparison
API confirms equal digests against private canonical bytes. The fingerprint is
never accepted where an exact Model artifact reference, semantic revision,
replay input, provenance reference, or mutation precondition is required. [RFC
0073](../rfcs/0073-structural-semantic-fingerprint.md) and
[`interfaces.structural-semantic-fingerprint`](../verify/interfaces/structural-semantic-fingerprint/README.md)
define this scope.

`RunManifestV1` remains the compatible original format. New execution evidence
uses `RealizationEnvelope` plus `RunManifestV2`: model identity/revision,
problem requirements, exact plan, and layout/partition artifact references are
typed separately from resolved adapter/library/topology/reduction provenance.
Constructors and post-decode linkage checks reject model, revision, target,
layout, worker, device, or reduction drift. Neither format includes host paths
or wall-clock data. Run manifests are distinct from
`eqiora.verification-report/v6`: the latter records a repository runner's
ordered case outcomes, canonically ordered exact case filters, selected
environment and runner-kind filters, monotonic target durations in whole
milliseconds, and captured child streams. A duration is `null` when a target
did not start. Equal private execution keys run once per invocation and project
the same captured outcome to every selecting claim; case-owned table artifacts
and descriptive metadata do not enter that process identity. Verification
manifests can select only a closed typed target: a validated Cargo package/test
pair or a
repository-owned installed-wheel Python gate. The orthogonal environment and
runner-kind filters are applied only after complete registry validation.
Neither target form admits a shell command, free-form arguments, a working
directory, or a host-specific path.

The original bounded package path composes with `RunManifestV1` through
`PackageRunBindingV1`, a separate canonical sidecar containing the shared
Model digest, exact package-compilation digest, closed Run schema, and canonical
Run digest. Construction and replay reject Model, revision, compilation,
resolution, schema, or Run-digest substitution. This is content lineage, not
execution evidence. The packaged DC-drive case creates an output-less Run v1
and binding only after analytic trajectory, residual, power, convergence, and
package acceptance; that scalar sampled path still keeps its `PhysicalSample`
observations in memory and does not reinterpret RFC 0051's spatial trajectory.
The distinct
`PackageExecutionBindingV1` path explicitly validates compilation, Model,
current Realization, and Run v2; the packaged isotropic-balance case closes that
chain after numerical acceptance. Typed lineage is never inferred from
matching digests.

RFC 0026 specifies independent `linear-system-envelope/v1`,
`partition-envelope/v1`, and `distributed-layout-envelope/v1` schemas. The
layout artifact links the exact system and partition and must replay every
owned/ghost layout and halo exchange from those inputs. Existing distributed
digest slots in `RealizationEnvelope` and topology in `RunManifestV2` close
the content-link DAG without changing either wire. The canonical 2D claim must
also reload Model+Realization, deterministically re-finalize FEM/TPFA, and
reproduce the exact CSR/RHS/properties system artifact bytes and digest; digest
linkage alone is not derivation evidence. The three closed, bounded schemas
now reconstruct the complete CSR and unique-owner contracts and reject any
linked digest or freshly derived owned/ghost/halo mismatch. Model+Realization
re-finalization and the registered canonical FEM/FVM MPI case additionally
prove exact system bytes/digest before admission; an internally content-linked
changed RHS demonstrates that the content DAG alone remains insufficient.

An imported affine-simplex mesh is an independent
`SimplicialMeshEnvelopeV1`, not an array embedded in Semantic Model or run
output. Realization stores its SHA-256 identity; artifact linkage additionally
checks mesh dimension against the admitted problem requirements. Decoder
limits separately bound vertices, cells, coordinate values, and connectivity
indices before topology reconstruction, and stored quality evidence must equal
a fresh reconstruction exactly.

[RFC 0025](../rfcs/0025-discrete-field-and-import-provenance.md) specifies the
adjacent field and import-provenance boundary. `eqiora-meshing` owns one
immutable, entity-major `DiscreteFieldPayload`
checked against `MeshTopology`: Vertex or top-dimensional Cell association,
Scalar or fixed Vector shape, checked counts, finite canonical `f64` values,
and positive-zero normalization. `DiscreteFieldEnvelopeV1` binds the
payload specifically to the exact `SimplicialMeshEnvelopeV1`; no other typed
mesh artifact family exists yet. `ExternalImportManifestV1` separately retains
an ordered source, resolved-array, adapter, runtime, selection, and
accepted-artifact lineage assertion. Its constructor computes identities from
complete supplied values, and its bounded decoder and reference validator
recheck exact independent linkage. The manifest becomes derivation evidence
only after an L4 public import workflow invokes the named pure L3 format adapter,
deterministically replays those sources through the admitted resolver plan, and
reproduces every array and artifact. This composition keeps format adapters
independent of the sibling L3 artifact crate; only the L4 workflow owns the
opaque verified-lineage handle.
Source display names, paths, dataset layout, and format identity may change the
manifest but never accepted mesh/field content identity. Semantic `FieldDef`,
bounded `ScalarFieldSummary`, and a complete discrete array remain three
distinct contracts.

`eqiora-io-xml` owns the pure VTU and XDMF adapters, exposed through
`eqiora::io::xml`. Its private XML reader shares declaration, character,
attribute, tree and structural-path validation, fallible tree allocations,
and source/event/decoded-byte work accounting. XDMF retains its DataItem
quota and admitted attribute references; VTU rejects references. Format
semantics and normalized payloads remain separate. The XDMF replay preserves
the authority boundary. Bounded UTF-8 XDMF 3 metadata yields a pure, canonically
ordered Geometry/Topology/selected-Attribute request plan; the L3 adapter has no
filesystem or network authority. An explicit caller supplies one complete
source occurrence and typed value array per request. The adapter admits only
one Uniform Tri3/Tet4 grid, XY or compatible XYZ affine geometry, and Node/Cell
`f64` scalar or topological-dimension Vector Attributes, then reconstructs the
existing mesh and payload contracts. `eqiora-api` alone produces a fresh
`XdmfImportArtifactsV1`. Its separate replay API accepts independently loaded
expected manifest, mesh, and field artifacts, fresh-derives the import once,
and requires exact content, order, reference, and manifest identity before
returning `VerifiedXdmfImportV1`.

The VTU adapter in the same crate owns its VTK semantics. Its first bounded
path accepts one serial ASCII VTK XML `UnstructuredGrid` containing one `Piece`, `Float64` XYZ points, homogeneous affine Tri3 or Tet4 cells, and
explicitly selected PointData/CellData scalar or fixed-vector arrays. Exact
connectivity, offsets, cell types, tuple counts, finite values, planar Tri3
coordinates, XML structure, and resource budgets fail closed before the
shared `SimplicialMesh` and `DiscreteFieldPayload` constructors run. Structural
element paths select content; association and `Name` are validated metadata,
not lookup identity. The L4 facade records the complete source as metadata
ordinal zero, binds normalized arrays and accepted artifacts through the
existing `ExternalImportManifestV1`, and issues an opaque verified handle only
after independently persisted manifest/mesh/fields equal a fresh replay.
VTK ranges and the narrowly admitted official-writer norm-range annotations
are nonsemantic presentation metadata. Binary/appended payloads, compression,
multiple pieces, parallel VTU, mixed/high-order cells, time, and export remain
separate adapter claims.

`eqiora-io-hdf5` is an independent optional L3 adapter rather than an XDMF
subsystem or artifact dependency. Its sole source authority is one borrowed,
complete caller-owned file image; HDF5 opens those bytes through the Core VFD
and receives no path, directory, URL, or network capability. The first L4
XDMF/HDF5 composition requires every request to share one display locator,
translates the complete immutable plan into one native batch, and never
dereferences that locator. The manifest names the complete
`eqiora.xdmf-hdf5.file-image` composition rather than pretending the Rust
binding is the adapter. Exact `hdf5-metno` 0.13.0 and the observed statically
bundled HDF5 runtime, currently 2.1.0, are recorded separately in its ordered
runtime stack.

The serialized operation selects the native VOL, disables and later restores
plugin loading, audits the complete reachable hard-link tree, and preflights
all requested paths, scalar types, and shapes before its first value read. It
admits only bounded groups and internally stored, unfiltered, non-virtual
datasets with exact standard `u64` or IEEE binary64 `f64` datatypes; aliases,
cycles, non-hard links, attributes, committed datatypes whether linked or
unlinked, external storage, and filters fail closed. Native handles and binding
types remain private to the adapter.

The same L3 crate also owns a bounded primitive-array writer. It accepts no
path, emits one complete in-memory file image, disables object timestamps and
dynamic plugin loading, and fixes contiguous unfiltered dataset creation in
canonical path order. Raw-byte determinism is claimed only under the exact
recorded binding and native-library profile.

[RFC 0066](../rfcs/0066-remeshing-trajectory-xdmf-hdf5-export.md) composes that
writer with a pure XDMF 3 Temporal Collection renderer at L4. A complete
borrowed dependency catalog must replay the complete current trajectory,
every state and current geometry, every Field snapshot and coefficient block,
and the remesh association, overlap, and transfer receipt before projection.
At the equal-time remesh seam the remesh target replaces the source tip; no
epsilon time is invented. All coefficient blocks are stored at
content-addressed HDF5 paths, while the first presentation profile exposes
only genuine vertex values and retains MINI bubble coefficients as hidden
storage. The current v3 `XdmfHdf5TrajectoryStorageEnvelope` binds the canonical trajectory,
ordered frame/state/mesh/geometry/Field lineage, dataset paths and
presentation decisions, exact runtime stack, and both complete output hashes.
Neither external format becomes trajectory authority, and the caller alone
decides whether to persist the returned bytes.

These boundaries narrow in-process authority; they do not sandbox HDF5. A
hostile process environment established before library initialization,
defects or unbounded internal work in HDF5/the binding, multiple native source
images per import plan, temporal import, 3D or high-order trajectory export,
chunked/parallel storage, and cross-runtime raw-byte identity remain separate
claims. Complete hostile-file containment requires a future isolated worker
boundary.

## Studio projection boundary

Studio is currently an L4 browser presentation over fixed, runtime-validated
example projections. It does not call the application service, compile or
execute models, construct canonical CAD identity, or persist artifacts. The
GTK3 desktop shell was retired because its runtime dependency closure could
not satisfy the repository security policy.

Projection data and workspace state remain disjoint. Graph coordinates,
selection, and local revision navigation are bounded presentation state and
never claim canonical model, artifact, or Run identity. React component types
and renderer node shapes do not become canonical wire formats. The fixed
examples use explicit preview identities so they cannot be confused with
outputs from the public Eqiora facade.

The relation projection is semantic HTML around an accessible graph view.
Every graph operation has a keyboard and non-drag path, focus is visible, and
status is announced through semantic live regions. The local
`eqiora.studio.workspace/v1` schema stores only preview-digest-bound finite
view coordinates. Command definitions and availability are presentation data
over the same local actions.

Future compilation, execution, authored CAD, export, or persistence must call
the public Model-first application surface and earn claim-local evidence on
that ordinary path. The browser projection must not infer scientific support
or fabricate canonical identity while those capabilities are absent.

## Dependency layers

```text
L4  agent / verify / API / interface adapters
L3  numerics / runtime / differentiation / backend adapters / coupling / hybrid / ML / artifacts
L2  semantics / IR / compiler / codegen / geometry / meshing / assembly / spatial distribution / solver / time / realization / distributed algebra / device execution
L1  graph / language / type checking
L0  core / schemas / build-time manifest projection
```

`eqiora-build` is used only by adapter build scripts. It projects the exact
normal-dependency pin from the owning Cargo manifest into compiled provider
provenance. Workspace builds follow `workspace.dependencies`; packaged crates
use Cargo's normalized member manifest. CUDA, MPI, faer, Rayon, and Diffsol
therefore retain one version owner rather than hand-maintained release strings.
The helper carries no runtime numerical policy or library-discovery behavior.

Dependencies point downward only. The repository checks this rule through
`cargo xtask check-layers`. Directed L2 composition into the pure solver
contract is explicit and mechanically allowlisted: realization consumes the
sole solver plan, distributed algebra consumes scalar/reduction and complete-
CSR storage/view vocabulary, assembly implements only the solver storage
projection, and spatial distribution composes exact mesh ownership with that
assembly vocabulary and distributed partition identities. The solver-owned
view implements the operator action. RFC 0080 geometry admission also has two
one-way consumers of the same opaque, kind-erased canonical Geometry reference:
semantic admission derives spatial support, while the compiler now derives the
exact digest, dimensions, entity-set membership, and boundary-parent facts for
one externally bound Component occurrence. Neither consumer retains Geometry
content or admits caller-supplied fact tuples. No reverse edge is permitted.
