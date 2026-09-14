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
`RealizationEnvelopeV1`, and package-neutral, Model-linked `RunManifestV2`. `eqiora-api`
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
enter the unchanged Realization v1 and Run v2 identity chain while preserving
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
package compilation, the current Model, Realization v1, Run v2, and
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
`(P1+bubble)^2/P1` field-wise Realization v2. Its Stokes adapter derives gauge
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
current-Model-to-Realization-v1/Run-v2 lineage. This explicit-flat case is not
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
Canonical bytes expose the persisted `eqiora.model-envelope/v27` schema as an
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
path. The smooth implicit claim is currently static, host-local `f64`. A
separate materialized direct-output reference binds an accepted relation/output
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
`ScalarObjectiveLinearization` lower both the residual and quadrature-defined
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
The optional PyTorch 2.13 adapter projects that same immutable program as a
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
comparison fingerprint and never routes through control-v2, MCP, Python or
Studio. It writes no artifact and defines no reusable CLI, operation, schema,
wire or registry abstraction.

The local `eqiora-mcp` subprocess is a separate L4 projection of the accepted
transport-neutral `ModelDocument::compile` operation. Its closed MCP
`2026-07-28` surface uses newline-delimited stdio, requires version and client
capability metadata, advertises exactly one in-memory compile/check tool, and
admits one active call. Bounded framing and metadata fail before compilation;
accepted calls return only the current Model descriptor plus a structural
comparison fingerprint, while rejected calls return bounded structured
diagnostics. Best-effort response cancellation can suppress a result but does
not claim compiler cancellation. Direct-operation parity and black-box stdio
evidence own this projection. It is not a protocol layer shared by clients,
does not make Python an MCP client, and adds no Studio, remote transport,
execution, scientific-data, persistence, artifact inspection, or generic MCP
conformance capability.

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

Linear solution has a separate backend-neutral boundary. `eqiora-solver` owns
the only `SolverPlan`, the host-local allocation-free `LinearOperator` action,
typed backend capabilities, accepted-solution reports, and a deliberately
small deterministic reference CG oracle. `eqiora-assembly` owns anonymous
local contributions, constraint-aware assembly maps, and CSR as an assembly
artifact; its CSR implements the solver action but introduces no second solver
configuration. Numerical realizations and execution adapters both consume
this L2 contract, so neither depends on the other. Production faer,
Rayon/MPI, and device integrations live in dedicated adapter crates. Every
accepted report includes a residual recomputed through the Eqiora operator
rather than trusting only a recursive library estimate.

Canonical Cartesian Q1 FEM and TPFA meet those contracts at one explicit
boundary. Assembly's reduced `LinearSystem` is temporary storage: finalization
captures its CSR and right-hand side once into a `CanonicalCsrSystemView`, then
drops the reduced storage object. That captured view is the sole reduced
algebra owner and action used by host solves, CUDA solves, residual
reacceptance, and assembled JVP/VJP actions. The finalized problem also exposes
the asserted properties, exact `SolverPlan`, vector layout, method identity,
and assembly evidence, while keeping eliminated-boundary recovery, full FEM
reaction state, TPFA facets, and dual-grid reconstruction private. A solver
adapter is chosen only after finalization. `finish` consumes an accepted
`LinearSolution` and independently rechecks it against the same captured view,
complete plan, normal orientation, iteration limit, and resolved producer and
verifier topologies before reconstructing a field. This is numerical
reacceptance, not durable origin identity: a vector satisfying two systems is
admissible to both. Typed distributed envelopes and their content-DAG check
bind exact model, realization, run, system, partition, and layout artifacts;
semantic derivation replay remains the separate proof that the linked algebra
was lowered from that model and realization. Before field reconstruction,
`finish` reserves its residual and method-private output storage fallibly and
moves the accepted vector and report instead of copying them.
Existing one-call entry points are thin
`finalize -> backend -> finish` compositions. This is verified for generated
Cartesian scalar Poisson with Q1 FEM and orthogonal TPFA on the reference CPU.
The optional public CUDA facade composes the same finalized handoff with the
single-device CUDA solver for both methods, then reconstructs only after host
residual reacceptance. The graph-bound adapter path first binds a logical
device/`QueueSlot` before device allocation, admits the exact finalized CSR and
operator properties, and later records its process-unique runtime `QueueId`,
typed transfers, value generations, and successfully waited CUDA fences. Its
immutable in-memory receipt exposes the complete host output only after both
the native serial verifier and an additional serial true-residual replay. That
physical gate remains ignored in ordinary CI while the machine-readable v2
case replays one bounded committed device observation on an ordinary host.
The replay exactly revalidates normalized Model, Realization, and Run
artifacts and reconstructs the execution trace before independently
reaccepting the recorded candidates through the finalized host operator.
Synthetic successful fences used for host reconstruction prove structural
consistency only; they do not re-attest physical waits. This is not hardware
attestation or a general PDE or accelerator-assembly claim. RFC 0023 records
the durable model/realization/run seam; the execution receipt itself has no
durable wire.

Thread placement is orthogonal to solver identity. An operator may optionally
expose allocation-free actions over disjoint contiguous output rows; assembled
CSR does so. `eqiora-backend-rayon` evaluates those rows in a run-owned bounded Rayon
pool without touching the global pool, while the decorated solver retains its
own plan and backend identity. P1/Q1 FEM, affine-simplex P1 FEM, and Cartesian
TPFA assembly use a distinct indexed packet contract: bounded batches may
evaluate concurrently,
but the L2 accumulator scatters packets, targets, and local entries in one
reference order. A FEM cell packet feeds both the reduced solve system and the
full reaction system without re-evaluating local physics. TPFA packets have one
stable cells-then-facets order, so source, interior-flux, and boundary-flux
operators use the same backend-neutral path. `AssemblyReport` and `SolveReport`
independently record their execution adapter and worker count, and spatial
solution objects retain both complete reports. Reference-CG inner products use
an Eqiora-owned fixed 1,024-element logical partition: Rayon evaluates indexed
partials concurrently and the numerical contract combines them in order, so
worker count does not alter the floating-point expression tree. The verified claim is
currently generated-Cartesian P1/Q1 FEM and orthogonal TPFA in one through
three dimensions plus imported affine-simplex P1 FEM in verified 2D, followed
by replicated CSR/reference CG. One/four-worker fields, reconstructions,
reactions/fluxes, balances, artifact-bound packet counts, and numerical
evidence are bit-identical. Adaptive, distributed, device, fast, and NUMA-aware
assembly remain outside that claim.

Device execution has its own L2 contract instead of widening host slices.
`RuntimeId`/`DeviceId` identify discovered placement; typed buffer descriptors
carry allocation identity, shape, element representation, and residency;
`QueueSlot` selects a logical deployment position, while a process-unique
`QueueId` identifies one concrete runtime materialization; only that complete
identity plus monotone submissions defines order within one command queue.
`Completion` is explicitly unrelated to a hybrid-model event; and
`TransferPlan` makes direction and byte count visible. The optional L3 CUDA
adapter keeps cudarc contexts/streams/allocations and its private cuSPARSE
library/descriptor/workspace and cuBLAS vector-action boundaries below those
contracts. Its action slice transfers finalized `f64` CSR, runs deterministic
or backend-native cuSPARSE SpMV, and compares the returned vector with the host
CSR oracle under an explicit tolerance. Its solver slice keeps assembled CSR
and Krylov vectors resident for CG/Jacobi, general BiCGSTAB/identity, or
symmetric-indefinite MINRES/identity, returns one candidate, and accepts it
through a distinct serial-host CSR/fixed-order true-residual verifier.
`SolveReport` records producer and verifier placement separately. These are
three exact tuples, not independently interchangeable solver, property, and
preconditioner axes.

The first graph-bound CUDA execution is narrower than the solver adapter. It
accepts only the exact Q1/TPFA finalized CSR/property intersection with
Jacobi-CG/`Fast`, one run-owned device and queue, and an implicit zero initial
value. Binding checks the logical `QueueSlot` and known minimum device payload
before device allocation; adapter execution materializes a process-unique `QueueId`
and records seven typed transfers, an exact single-generation solution update,
and real successful waits after inputs, solve, and output transfer. The known
payload and separately reported external sparse workspace are checked against
total device capacity, not currently free memory. The fixed nine-step DAG ends
with native serial verification, an independent serial
receipt replay, and an immutable complete-host-output receipt bound to the
output fingerprint. CUDA library and driver versions stay in the paired Run
and adapter evidence rather than being copied into that receipt.

The fixed-reference FSI composition adds no second device execution protocol.
A host-reproducible and a one-device-`Fast` Realization independently invoke
the same equation-aware finalizer and must produce an identical complete
CSR/RHS fingerprint before admission. The CUDA graph then selects the exact
symmetric-indefinite MINRES/identity tuple. Its accepted complete host output
passes the same generic true-residual gate and the sole existing FSI finish.
The CPU and CUDA Realization identities intentionally differ; the semantic
Fields, mesh, scaling, pressure closure, state elimination, and finalized
algebra do not.

Both action and solve evidence retain the selected device's actual compute
capability beside its Eqiora-owned device descriptor. The public facade
exposes the established solver adapter only under `cuda`; the new raw
graph-bound admission seam is deliberately not a curated public facade API.
The adapter remains dynamically loaded and absent from default builds. Setup,
H2D, action/solve, D2H, verification, and total time are separate
observations. Arbitrary initial values, free-memory reservation, persistent
residency, multiple streams/queues, GPU assembly, matrix-free kernels,
stronger preconditioners, reproducible device reductions, general FSI or
general MINRES, scale, multi-GPU, and broader MPI plus CUDA remain separate
gates under RFC 0019, RFC 0058, RFC 0062, and RFC 0063. The bounded
host-staged composition is described with the distributed FSI handoff below.

Matrix-free experimentation begins from a second owned L2 boundary rather
than a device library IR. `LocalLinearActionIr` is one shape-homogeneous batch
of anonymous entity-local linear maps with entity-major packed inputs and
outputs and an ordered CPU evaluator. Cartesian Q1 diffusion lowers to this
form in one through three dimensions; assembled CSR and gathered/local/scattered
actions agree under the verification tolerance. Mesh identity, global
numbering, gather/scatter, and reduction remain separate contracts. RFC 0020
records why the isolated CubeCL 0.10.0 CUDA/HIP experiment fails the required
`f64` and MSRV graduation gates and creates no accelerator support claim.

Distributed algebra is a distinct contract rather than a wider meaning
of host slices. `GlobalVectorSpace` and `Partition` assign one owner to every
global index; each `LocalLayout` has sorted owned and disjoint ghost indices;
off-owner CSR columns derive an ordered `HaloPlan`; each shard stores only
owned rows. Construction pre-counts every owned row, nonzero, ghost candidate,
and halo transfer with checked arithmetic before fallible exact reservation.
The operator owns each layout once; `LocalCsrShard` is a borrowed layout/CSR
view rather than a second identity-bearing layout copy. A one-process loopback
oracle fallibly reserves its complete workspace, splits owner-local input,
performs the declared halo transfers, applies local rows, and gathers unique
output. It
also reduces a dot product from unique-owner contributions in partition order;
the unsupported fast policy fails closed. Exact agreement for one, two, and
four partitions verifies layout/protocol and reproducible collective
invariants independently of any transport.

The optional L3 MPI adapter executes that same immutable halo and shard
contract through a generic complete-CSR bridge on one, two, and four ranks in
an executable test. The application owns MPI initialization/finalization; the
adapter validates thread support and duplicates its communicator without
leaking MPI types into L2. `MpiExecutionGroup::admit` collectively seals the
system, complete verifier, sole `SolverPlan`, and plan-inclusive fingerprint
into one mutably borrowing run token. All dynamic halo, Krylov, reduction,
status, gather, and host-verifier storage is reserved before admission; raw
apply, dot, and solve operations are not public. Exact monotonic collective
steps and synchronized failure records precede communication, including
post-admission fault-injection tests guarded by a parent timeout. Explicit
global-index owner gathers reconstruct one vector per rank, which is accepted
through the captured complete host action; accepted vectors and reports must
then agree. The registered one-host case verifies this bridge at one, two, and
four ranks under reproducible and native-fast reductions. Earlier recorded
physical two-node evidence remains valid for the lower-level halo, reduction,
and CG algebra path, but the generic admitted bridge does not inherit that
physical claim. One canonical spatial composition is described below;
bridge-level multi-node evidence, scalability, checkpoint/restart, and
process-failure recovery remain absent.

The graph-bound path now places that bridge behind the common execution
boundary. A transport-neutral `DistributedExecutorDescriptor` binds the exact
provider, logical `ProcessGroupSlot`, rank count, and one worker per partition
to a portable `Distributed`/`f64`/offline/SPD/CG/Jacobi/`Reproducible` graph.
The application must already have initialized MPI; `MpiExecutionGroup`
validates thread support and duplicates the communicator before the observed
rank count and capabilities can be used for numerical binding. Thus this is a
pre-numerical-workspace and pre-collective admission boundary, not a
pre-communicator boundary. MPI implementation/version and provided thread
support stay in typed Run provenance, while communicator and rank-local MPI
handles remain private to L3.

`AdmittedExecution` seals the complete CSR fingerprint, partition identity,
derived layout/halo identity, plan-inclusive admission fingerprint, sole plan,
and preallocated independent host replay. The MPI adapter consumes that token
without an alternative graph/system/plan entry point. Its actual normalized
trace has a checked `32 * maximum_iterations + 64` capacity reserved before
collective admission, records only the synchronized boundaries that occur,
and requires dense global ordinals, bounded iterations, all Krylov phase
families, and the ordered terminal gather/acceptance suffix. The immutable
receipt exposes the fixed macro DAG:

```text
AgreeDistributedAdmission -> SolveDistributedKrylov
  -> AgreeDistributedProducerReport -> GatherDistributedOwnedCandidate
  -> AcceptWithNativeHostVerification -> AgreeDistributedAcceptedResult
  -> ReplayTrueResidualOnHost -> AgreeDistributedReceipt
  -> AcceptHostComplete
```

Every rank first reconstructs and natively serial-host-accepts the same
complete candidate, then agrees that accepted result. L2 independently replays
the complete-host true residual and binds the final receipt to the normalized
output fingerprint. L3 then all-gathers a domain-separated fixed-size summary
covering operator, output, dimension, producer report,
partition/layout/admission/process-group identities, and the full normalized
trace. The group allocates that summary receive storage during communicator
duplication, before numerical binding, and exposes only byte-identical
receipts. The registered canonical Q1/TPFA case verifies this path with live
MPI processes at one, two, and four ranks on one host. Earlier physical
two-node evidence still covers only the lower-level halo/reduction/CG case;
this graph-bound bridge does not inherit a physical multi-node claim.

[RFC 0026](../rfcs/0026-distributed-spatial-layout-and-replication.md)
specifies the first composition with canonical spatial realization. Its two
existing boundaries stay distinct. The generic algebra boundary has the
object-safe L2 storage projection, validated Eqiora-owned complete-CSR action,
distributed system/layout identities, and admitted MPI bridge described above.
The canonical spatial boundary now finalizes Q1 FEM and TPFA into that same
captured representation: reduced assembly storage is dropped, the admitted
replicated/distributed layout is retained, and distributed production is
admitted only as `HostCpu { threads: 1 }` with one worker per partition.
Method-native reconstruction additionally requires independent one-worker
host verification.

Typed durable system, partition, and layout envelopes reconstruct and compare
the complete derived algebra, while one external content-DAG check binds them
to exact Model, Realization, and Run artifacts. The registered
`numerics.canonical-cartesian-poisson-mpi` case closes the first composition:
one parent-created Model artifact is decoded on every rank; decoded
Realization policy freshly reproduces exact system bytes and digest; all six
artifact digests agree across ranks; the replayed finalized view and
artifact-derived rotated-cyclic layout enter the admitted bridge; and explicit
global indices and values are gathered, host-reaccepted, and passed to the
existing FEM/FVM finish. An honestly relinked changed RHS passes the content
DAG but fails semantic derivation replay. The executable envelope is exactly
generated Cartesian 2D `f64` Q1 FEM/TPFA, `Reproducible` CG/Jacobi, one host,
one worker per rank, and one/two/four ranks. Assembly, mesh state,
reconstruction, and the final field stay replicated; bridge-level physical
multi-node evidence, distributed assembly, sharded result fields, scaling,
and hybrid rank/thread execution remain unclaimed.

## Geometry identity, geometry-to-mesh correspondence, and kernel-neutral CAD boundary

The Semantic Kernel still owns only exact `Domain`, `BoundaryOf`, `Port`, and
`Connection` meaning. RFC 0049 adds no geometry node. `eqiora-geometry` closes
the pure revision-local chain from a semantic Domain to an entity in one exact
geometry revision and then to entities in one exact mesh revision. L3 artifacts
bind that chain to exact Model, geometry, and mesh digests and replay Cartesian
embedding against the accepted affine-simplex coordinates.

The same L2 crate owns the bounded, kernel-neutral CAD design, observation, and
adapter contracts consumed by geometry producers. These contracts describe
source identity and units, fully constrained design intent, explicit
uncertainty and modeling tolerance, accepted observations, and adapter
identity without admitting a CAD-kernel object or entity-enumeration index.
Concrete Truck objects, STEP parsing, and B-rep/modeling execution remain in
`eqiora-cad-truck`; no concrete CAD kernel is part of the L2 boundary.

Geometry consumes RFC 0037's sealed replayable current Model contract rather
than interpreting its persisted wire. Replay produces exact artifact identity
and a validated immutable Kernel Program together; an identity-only reference
cannot stand in for content. A future incompatible Model meaning requires a
new schema and compatibility decision while geometry continues to consume this
owned replay boundary.

Body cell sets form a total disjoint mesh partition. Each parent's complete
relative frontier is partitioned by its boundary Domains. Two distinct
interface Boundary Domains may reference one shared geometry entity and the
same mesh facets only when they have distinct parents; incidence derives one
parent-outward view per side. A caller never supplies a physical normal sign,
and mesh `OrientationCode` remains only a local vertex permutation.

The Cartesian producer owns one finite positive coherent-SI classification
precision as part of Geometry Identity. Correspondence reuses it for point and
facet membership and exposes no competing mesh-local tolerance. Exact bounds
still determine entity topology; mesh quality, solver acceptance, CAD healing,
and future import uncertainty remain separate policies.

Planar circular-hole Geometry v2 is a separate provenance-neutral identity
domain rather than a reinterpretation of that classifier-bearing contract. Its
content is finite bounds, centre, radius, and complete named membership; it
persists no graph, build, provider, lineage, or `tolerance_m`. Separately, the
accepted authored-CAD producer projects retained and created lineage into
private result topology through the accepted build owner, then validates
complete atomic naming into the common owned Geometry. The private v1/v2
representations remain closed, classification precision is present only for
kinds that own it, and no public result handle or classifier is introduced.
Imported or otherwise
untracked topology still requires an explicit identity-bearing classification
policy at its correspondence/Realization boundary. V1 decoding and its
classification semantics remain unchanged.

Cross-revision retention is a separate explicit total body bijection. Domain
ULIDs are revision-local; paired boundaries are derived from the retained
parent pair and `(axis, side)`. Missing, split, merged, ambiguous, partial, or
stale relations produce no retained association. Concrete CAD-kernel objects,
STEP parsing, B-rep/modeling execution, concrete adapter identity values,
transfer, ALE, remeshing, and FSI equations remain outside this L2 contract
boundary.

## Fixed-reference fluid--structure lowering

RFC 0050 closes one deliberately narrow consumer of that identity seam without
adding an FSI entity to the Semantic Kernel. The Semantic Model remains an
ordinary flat network: an inertial incompressible Newtonian relation, a
first-order linear-solid relation, and one conserving velocity/traction
Connection between two exact boundary Ports. The package and direct authoring
paths lower to the same physical roles; neither package selects a mesh, time
method, pressure policy, solver, or coupling algorithm.

The physics-neutral multi-Domain Realization contract records exact Domain and
Field inventories, Field-wise spaces over one imported mesh, one conforming
trace quotient, one represented-but-eliminated backward-Euler state, symmetric
congruence scales, operator properties, solver, target, and schedule.
Realization envelope v3 serializes those choices while preserving the frozen v1
and v2 wires. The FSI-aware adapter then proves that the exact roles select
MINI/P1 fluid velocity/pressure, P1 solid velocity/displacement, equal trace
scales, no pressure gauge, and the reference CPU execution contract.

Backward Euler eliminates `d_next = d_previous + dt * v_next`; it does not turn
displacement into a false algebraic block. Fluid and solid P1 interface
velocities address one quotient row, while the fluid bubble has zero trace.
The complete coupled operator determines whether constant pressure is closed;
no standalone Stokes gauge rule is copied into the coupling path. Assembly
produces one dimensionless symmetric-indefinite canonical CSR system under
`A_hat = D^T A D / Theta`, with `Theta = P U L` for the intrinsic-2D slice.
The typed in-memory result binds reconstructed physical coefficients back to
the exact four Field identities and support inventories while delegating the
same captured CSR. It also retains the exact mesh reference and complete
physics-neutral Realization plan so a durable projection cannot accept a
same-shaped foreign solution.

The registered
[`fsi.fixed-reference-monolithic-step-2d`](../verify/fsi/fixed-reference-monolithic-step-2d/README.md)
case is the claim boundary. It does not imply advection, multiple steps,
partitioned coupling, moving meshes, GPU/MPI, nonlinear solids, or
sensitivities.

## Durable moving spatial observations

`DiscreteFieldEnvelopeV1` and `FieldSnapshotEnvelopeV2` remain the shared numeric and semantic leaves for current moving and remeshing results. They retain exact Model, Realization, geometry, correspondence, mesh, Field, support, dimension, shape, frame, association, and coefficient-block lineage.

`SpatialStateEnvelopeV2` and `SpatialTrajectoryEnvelopeV2` own fixed-topology ALE continuation. `SpatialStateEnvelopeV3` and `SpatialTrajectoryEnvelopeV3` add explicit remesh origins, overlap and transfer receipts, and exact V2 source-prefix ancestry. The retired fixed-mesh V1 context, State, trajectory, Dataset view, and application-specific fixed-reference projection are absent; fixed-domain execution publishes through the common Model → Plan → Run/Result path.

`DiscreteFieldStorageEnvelopeV1` remains a low-level optional chunking projection over canonical leaf bytes. It does not enter logical Field, moving State, trajectory, or ML Dataset identity.

[RFC 0067](../rfcs/0067-derived-ml-dataset.md) defines the current
`MlDatasetEnvelopeV2` over V2/V3 trajectories. The manifest derives one
strict-time sequence from an exact V2-to-V3 trajectory:
the V3 remesh target replaces the equal-time V2 source tip. Typed descriptors
retain feature/target role, window offset, Field, support Domain, coherent-SI
dimension, value shape, and frame. Every sample retains exact state, snapshot,
mesh, and complete coefficient-block identities. Ordered training,
validation, and test partitions cannot share a state artifact.

Normalization is one closed training-population standard-score policy. Its
per-descriptor, association, and component statistics are recomputed from
active training support only; validation and test values cannot enter the
fit. A constant channel records zero population deviation and an applied
scale of one. The artifact names the ordered binary64 Welford accumulator
profile and distinguishes exact input constancy from a rounded deviation. The
L4 CPU projection preflights all block, active-entity, scalar, and statistics
work before performing bounded explicit copies into
ragged entity-major blocks carrying exact active indices and mesh identity.
It neither pads across remeshing nor treats mesh-local indices as persistent
physical identity.

The complete V2-to-V3 dependency replay is storage-independent and available
without XDMF/HDF5. The optional temporal exporter and the Dataset adapter are
two consumers of that exact replay profile. XDMF paths, HDF5 layout, framework
tensors, and device placement do not enter logical Dataset identity.
Interpolation, dense batching, wider split/transform policies, training, and
framework/device loaders remain separate typed adapters.

## Realization selection

Realization is a second typed layer, not an optional field bag on the Semantic
Model. The accepted scalar, field-wise, coupled, and transient plan families
remain bounded authoring and frozen-wire compatibility contracts. Their
resolvers reject unknown policy, method/space, exact-identity, solver, and
target contradictions without fallback. A successful resolved value then
normalizes into the common portable DAG defined by [RFC
0058](../rfcs/0058-portable-realization-and-execution-graphs.md): exact Domain
discretizations and Field spaces feed typed numerical transformations, one
connected algebraic system, explicit linear/nonlinear solve roles, and
portable placement requirements.

The DAG has typed layer-specific references rather than a universal node or
payload. Its bounded Phase-A projections drive scalar serial/Rayon, steady
field-wise Stokes, coupled FSI, and fixed-domain transient flow execution.
Equation-aware identity claims fill facts absent from compatibility plans; no
anonymous Domain, Field, Relation, or Connection is fabricated. Graph
validation proves structural closure, while each execution finalizer must
compare those claims with its exact accepted lowering before execution.
Repeated step count remains a Run directive. Committed artifact golden
fixtures independently freeze realization-envelope v1/v2/v3 bytes, and no
generic graph wire is introduced during migration.

The first bounded deployment contracts are linear and downstream of this
portable graph. A pure host binding validates exact backend/adapter capacity
and solver tuple before worker-pool or numerical-system materialization. The
scalar finalizer independently regenerates the equation-aware portable graph
and owns it beside one finalized canonical CSR system; the curated facade does
not expose raw graph/system admission. An opaque token seals that system's
fingerprint, operator properties, sole solver plan, binding, and preallocated
serial verifier. Acceptance returns the complete solution with an immutable
receipt only after exact report matching and independent true-residual replay,
and binds the receipt to the normalized output-vector fingerprint. Serial and
Rayon therefore expose the same fixed `SolveWithNativeAcceptance ->
ReplayTrueResidualOnHost -> AcceptHostComplete` DAG while retaining distinct
bindings and solver-native verifier reports; the receipt records its
additional serial replay separately. There is no mutable graph builder, deep
CSR clone, or durable execution wire in this host slice.

The corresponding CUDA binding selects one device and logical `QueueSlot`
before device allocation and admits only the exact finalized CSR/property and
Jacobi-CG/`Fast` tuple with an implicit zero initial value. Its adapter-owned
runtime materialization has a distinct process-unique `QueueId`. Seven typed
transfers, exact solution generations, and three real successfully waited
fences drive the fixed `TransferInputsToCuda -> AwaitCudaInputsReady ->
SolveOnCuda -> AwaitCudaSolveCompletion -> TransferCandidateToHost ->
AwaitHostVisibility -> AcceptWithNativeHostVerification ->
ReplayTrueResidualOnHost -> AcceptHostComplete` DAG. The final immutable
in-memory receipt retains a complete host output, native serial acceptance,
independent serial replay, and exact output fingerprint. Host evidence replay
reconstructs the trace with synthetic successful fences and therefore does
not re-attest the physical waits.

[RFC 0062](../rfcs/0062-cuda-fixed-mesh-fsi.md) reuses that exact binding and
nine-step DAG for one fixed-reference 2D FSI step. The finalized FSI owner
retains the portable graph and canonical CSR together, so CUDA admission cannot
pair FSI provenance with a caller-supplied system. CPU and CUDA finalization
must agree bit-for-bit on the CSR/RHS fingerprint; the returned generic receipt
must agree with the graph, operator, solver plan, device, dimension, report,
and CUDA trace before the unchanged physical finish can consume its solution.
Selected-device evidence compares Field identity, support, ordering, length,
dimensionless coefficients, and scale-normalized physical coefficients with an
independent CPU result. It is portable recorded evidence, not hardware
attestation or a claim of device-reproducible reduction.

The corresponding distributed binding uses `DistributedExecutorDescriptor`
and a logical `ProcessGroupSlot` rather than an `Mpi` target. It seals the
complete system plus owner/layout/admission identities, reserves one bounded
actual collective trace, and executes through the isolated MPI adapter. The
nine-step macro DAG covers admission, a repeating halo/action/reduction/update
Krylov region, producer agreement, explicit-index owner gather, native host
acceptance, accepted-result agreement, independent receipt replay, final
cross-rank receipt-summary agreement, and complete-host acceptance. Runtime
MPI identity and thread support remain Run evidence, not receipt fields. This
closes Phase B of RFC 0058; distributed mesh ownership and assembly are the
separate contract defined by
[RFC 0060](../rfcs/0060-distributed-spatial-ownership-and-assembly.md).

That spatial concern has one transport-neutral L2 composition seam. An exact
mesh-revision identity and unique top-cell ownership derive lower-entity
residency without entering distributed algebra. The same cell ownership then
selects exactly-once local assembly packets, while actual equation support
derives target-row ownership. Payload-bound routes are sealed before an
unordered inbox is admitted; owner folds restore target/row/global-packet
order and complete systems are reconstructed only from checked owner shards.
The registered fixed-reference 2D FSI loopback proves one/two/four logical
partitions against independent complete CPU assembly. The corresponding
one-host MPI case carries the same admitted stages through one, two, and four
physical ranks, including synchronized rejection of a rank-local foreign mesh
revision before variable-size transport. MPI is an adapter over these stages,
not a second routing or accumulation semantics. This closes RFC 0060's bounded
assembly composition without
itself claiming a distributed solver, multi-node execution, or scale.

The bounded composition specified by [RFC
0061](../rfcs/0061-mpi-fixed-mesh-fsi.md) promotes RFC 0060's accepted reduced
owner-row payloads directly into rank-local CSR/RHS storage, derives the
solver-vector halo from that accepted sparsity, and executes the unchanged
fixed-reference symmetric-indefinite operator with reproducible
identity-preconditioned MPI MINRES. Explicit-index gathering, complete-host
reacceptance on every rank, and the existing FSI finish remain distinct
acceptance stages. The full assembly target stays in lineage even though only
the reduced target enters MINRES. A reconstructed complete CSR is the host
verifier, not a source for a second solver partition.

Reproducible execution fixes ordering within one admitted process-group
shape. Different rank counts may change the floating-point reduction tree, so
[RFC 0063](../rfcs/0063-mpi-cuda-fixed-mesh-fsi.md) requires invariant
model/operator meaning and tolerance-based agreement
with the independent CPU result rather than cross-rank-count bit identity.
Partition, halo, assembly-receipt, deployment, and observed runtime provenance
correctly remain rank-count-specific. A durable symmetric-indefinite
distributed Run artifact remains outside this bounded composition.

The next bounded composition, specified by [RFC
0063](../rfcs/0063-mpi-cuda-fixed-mesh-fsi.md), keeps that MPI Krylov state,
halo, reproducible reduction, gather, and complete-host acceptance unchanged
while delegating only each admitted owner-row sparse action. A separate L3
composition crate captures the exact shard into deterministic rectangular
`[owned | ghost]` columns, uploads its matrix once to one rank-local CUDA
device, and host-stages each input and owned output across three waited
completion boundaries. Host-owned and delegated-action admissions are
different token variants, so the ordinary MPI entry point cannot silently
replace CUDA with its host action.

The one-host evidence masks one physical selector into ordinal zero per rank,
agrees distinct live UUIDs, and executes the same finalized operator at one,
two, and four ranks before invoking the unchanged FSI finish. This adds no
GPU-aware MPI, device-resident Krylov or reduction, multi-node topology,
performance, scale, GPU assembly, durable composite receipt, transient FSI,
or ALE claim.

Model-time and deployment-time remain separate. `ClockDomain` defines exact
activation semantics. `ExecutionSchedule` may define deployment priority and
deadline but has no model period or phase. A future task lowerer may connect
the layers through typed provenance without letting scheduling policy change
the canonical equation network. The packaged DC-drive evidence checks the
current identity boundary: changing its sample period changes package and
Model meaning, while changing only host execution topology preserves the Model
identity and changes the Run identity. This is not a real-time scheduling
claim.

Capability negotiation is a predicate over both the compatibility plan and explicit problem
requirements. It checks discretization method, mesh kind, spatial dimension,
scalar type, replicated/distributed vector layout, solver, preconditioner,
reduction policy, target availability, worker bound, and scheduling profile.
The generic declaration is an exact set of nested spatial/execution contexts
paired with exact solver capabilities, rather than independent axis sets. A
registered falsifier admits an imported-simplex P1 FEM/CG tuple and a generated-
Cartesian P0 FVM/BiCGSTAB tuple while rejecting their otherwise valid
cross-combination before execution. Space, order, quadrature, and other
method-specific facts remain owned by typed plan validation. Legacy
property-free resolution retains its admitted operator-property candidates,
and either the equation-aware portable projection or an explicit legacy
finalizer seal rejects a claim outside that set; property-aware field-wise
resolution checks it directly, while numerical finalization closes equation
identity and coefficients.
The current canonical scalar-elliptic reference path declares exactly `1D..=3D + f64 +
replicated + one host worker`, with generated Cartesian and imported affine-
simplex mesh kinds. Method/space/mesh/quadrature cross-validation restricts
the imported path to continuous P1 plus simplex-centroid quadrature; the
current imported PDE evidence is 2D only. Runtime-dimensional lower contracts
do not widen that evidence envelope. The interval API additionally rejects a
resolved dimension other than one. Unsupported dimensions/thread counts,
mesh kinds, and undiscovered CUDA ordinals fail closed. That ordinal belongs
to legacy compatibility and later accelerator Deployment binding; the
portable graph retains only a device count per partition. The bounded Phase-B
slices now bind selected serial/Rayon capacity, one CUDA device/queue, or one
distributed process group and record only the accepted
Solve/Transfer/Halo/Collective/Fence dependencies they execute. Future
heterogeneous placements must compose those observed resources rather than
reopening a target cross product.

RFC 0026 separately specifies a `2D + f64 + distributed + one host worker per
partition` generated-Cartesian Q1 FEM/TPFA composition. The named 1/2/4-rank
case now verifies exact artifact/semantic replay, collective admission,
distributed solve, ordered gather, host reacceptance, method-native finish,
analytic balance/error, and serial-reference conformance within the narrow
one-host envelope above.

## Artifact and verification boundary

Numeric version suffixes name actual compatibility obligations, not merely the
possibility of future change. Persisted or externally exchanged schemas,
protocols, digest domains, and their exact decoders retain `V1`, `V2`, and later
identities for as long as those representations must remain distinguishable.
Historical wire DTOs stay inside their owning compatibility modules and lower
to the current typed API before reaching numerical code.

Canonical in-memory APIs use versionless names unless simultaneously supported
public contracts must remain distinct. Existing suffixed public Rust names are
not silently redefined: they transition under crate SemVer through a forward
versionless spelling or an explicit compatibility decision. Private numerical
policy and implementation types use mathematical or operational names;
changing such a policy does not manufacture a data-format version. Artifact
and protocol versions independently govern persisted bytes and external
exchange.

The unversioned public `ModelEnvelope` and `ModelTransactionEnvelope` own the
single current runtime contract while retaining the persisted v8 schema
identifiers and digest domains. They serialize the current Semantic Model
through wire DTOs, then reconstruct through typed constructors, one graph
transaction, and `KernelProgram` validation. Canonical JSON order and
schema-domain-separated SHA-256 digests are deterministic; source revision is
retained as provenance but excluded from semantic content identity. Each JSON
decoder first applies one syntax-only byte/depth preflight, then the shared
family-owned node, edge, expression, view, and operation budgets before graph
mutation. Model v1--v7 schemas reject without sniffing, retry, or migration.
Semantic limits do not live in one universal
bag: mesh, geometry, field, Realization, remesh, physical-exposure,
distributed, resolved-array/import, trajectory/storage, time, and ML Dataset
decoders each receive only their owning family's budgets. An artifact that
embeds another family composes that family's named semantic budget explicitly;
changing one family cannot alter another family's admission policy.

The sealed `CanonicalModelArtifact` boundary projects the validated current
envelope to one `ModelArtifactReference`: exact wire-domain digest, typed Model
identity, and semantic revision. `RealizationEnvelopeV1` consumes that
reference without changing its existing fields or bytes. The reference does
not detect or upgrade schemas. The registered evidence constructs coherent
current Model, Realization v1, and Run v2 lineage but does not lower or execute
the physical Model. [RFC
0037](../rfcs/0037-version-neutral-model-artifact-reference.md) and
[`artifacts.model-reference-lineage`](../verify/artifacts/model-reference-lineage/README.md)
define this bounded claim.

Independent authoring routes may express the same accepted relation network
while correctly producing different exact Model identities. For that narrower
comparison purpose, `eqiora-artifact` projects the validated `KernelProgram`
to a generation-tagged `StructuralSemanticFingerprint` (current generation V9).
The projection removes Model and entity occurrence ULIDs, source presentation, package identity, and
exact artifact identity while retaining distinct vertices, nominal references,
Parameter values, field roles and activation, initial equations, expression
structure, semantic edges, physical connections,
and Model boundary membership. Exact partition refinement plus bounded
individualization selects one canonical labelling; unknown meaning or exhausted
limits fail closed. Equal digests are confirmed against private canonical
bytes by the comparison API. This fingerprint is not accepted anywhere that
requires an exact Model artifact reference, semantic revision, replay input,
provenance reference, or mutation precondition. [RFC 0073](../rfcs/0073-structural-semantic-fingerprint.md)
and [`interfaces.structural-semantic-fingerprint`](../verify/interfaces/structural-semantic-fingerprint/README.md)
define the bounded claim.

`RunManifestV1` remains the compatible original format. New execution evidence
uses `RealizationEnvelopeV1` plus `RunManifestV2`: model identity/revision,
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
Realization v1, and Run v2; the packaged isotropic-balance case closes that
chain after numerical acceptance. Typed lineage is never inferred from
matching digests.

RFC 0026 specifies independent `linear-system-envelope/v1`,
`partition-envelope/v1`, and `distributed-layout-envelope/v1` schemas. The
layout artifact links the exact system and partition and must replay every
owned/ghost layout and halo exchange from those inputs. Existing distributed
digest slots in `RealizationEnvelopeV1` and topology in `RunManifestV2` close
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

`eqiora-io-xdmf` implements the first format replay without widening that
boundary. Bounded UTF-8 XDMF 3 metadata yields a pure, canonically ordered
Geometry/Topology/selected-Attribute request plan; the L3 adapter has no
filesystem or network authority. An explicit caller supplies one complete
source occurrence and typed value array per request. The adapter admits only
one Uniform Tri3/Tet4 grid, XY or compatible XYZ affine geometry, and Node/Cell
`f64` scalar or topological-dimension Vector Attributes, then reconstructs the
existing mesh and payload contracts. `eqiora-api` alone produces a fresh
`XdmfImportArtifactsV1`. Its separate replay API accepts independently loaded
expected manifest, mesh, and field artifacts, fresh-derives the import once,
and requires exact content, order, reference, and manifest identity before
returning `VerifiedXdmfImportV1`.

`eqiora-io-vtu` is a sibling syntax adapter, not an XDMF profile. Its first
bounded path accepts one serial ASCII VTK XML `UnstructuredGrid` containing
one `Piece`, `Float64` XYZ points, homogeneous affine Tri3 or Tet4 cells, and
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
borrowed dependency catalog must replay the exact V2 prefix, V3 trajectory,
every state and current geometry, every Field snapshot and coefficient block,
and the remesh association, overlap, and transfer receipt before projection.
At the equal-time remesh seam the V3 target replaces the V2 source tip; no
epsilon time is invented. All coefficient blocks are stored at
content-addressed HDF5 paths, while the first presentation profile exposes
only genuine vertex values and retains MINI bubble coefficients as hidden
storage. `XdmfHdf5TrajectoryStorageEnvelopeV1` binds the canonical trajectory,
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
L0  core / schemas
```

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
