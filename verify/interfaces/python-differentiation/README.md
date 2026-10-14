# Python accepted-point differentiation

One opaque `DifferentiableProgram` binds one exact host-serial common scalar
Plan, its canonical Model artifact, its caller-supplied two-dimensional
rectangular Cartesian Mesh, an ordered set of canonical Parameter identities,
and the Plan's complete primary Field. The Plan admits Q1 FEM or cell-centred
TPFA with a typed linear solve policy. Those selected
Parameters form the program's ordered typed input leaves; unselected
Parameters stay frozen at their canonical Model values. The canonical values
are the default point. `evaluate(parameters)` accepts another complete point
without mutating the Model, creating a child revision, or replacing the Plan.

Each call returns a frozen opaque `DifferentiableEvaluation`. It owns its exact
Parameter point, accepted primal, paired linearization, output projection, and
execution receipt while retaining the program's static Model, Mesh, Plan, and
role identity. Its `point`, `primal`, `jvp`, and `vjp` cannot be retargeted by a
later evaluation of the same program. The registered case exercises an
alternate point followed by the default point and mutates the original Python
input after admission to falsify aliasing and hidden state.

Point inputs form an ordered sequence with one scalar or intrinsically shaped array
per selected Parameter. The native value signature owns its mathematical type and
real/complex packing. The registered Poisson profile has three real scalar leaves.
Its array leaves and the rank-one tangent/cotangent buffers admit exact native-endian,
aligned, C-contiguous CPU `float64` storage from NumPy, Eqiora `Array`, or a complete
DLPack producer already resident on CPU device 0. The DLPack path
preflights the device before export and asks NumPy's standard consumer for a
CPU view with `copy=False`; this forbids a hidden producer copy or transfer.
The descriptor gate precedes one Eqiora-owned staging copy per array leaf or
derivative buffer; the owned values then pass the common finiteness gate
before releasing the GIL. The temporary protocol view and producer lifetime
never enter detached native execution; the producer is responsible for not
mutating its buffer during this synchronous handoff. Accepted point values and
outputs use the existing immutable `Array` producer contract.

The registered two-dimensional Poisson case exercises both Q1 FEM and TPFA
FVM. It compares complete-Field JVP and VJP actions with independently
recompiled centered finite differences and checks
`<J v, c> = <v, J^T c>`. For FEM, the output projection reconstructs every
vertex and differentiates eliminated essential-boundary values directly;
method-native free unknowns are never presented as the complete Field.

The case also rejects foreign or substituted Plans, Mesh/Model role mismatch,
duplicate inputs, the retired Model-plus-Realization signature, incomplete
DLPack producers, foreign devices and CPU ordinals, wrong shapes, wrong dtypes,
non-native/non-contiguous inputs, and non-finite values. Normal and transposed
solve orientation, solver policy, accepted residual, exact state-system
identity, derivative implementation, and linearization reuse remain typed
in-memory evidence. The exact point is carried by the immutable evaluation,
while occurrence evidence remains bound to the same static program identity
and point-specific execution receipt. No persisted program, implicit or
partial Parameter override, multi-output packing, objective language, general
Run input, zero-copy consumer, GPU/stream contract, framework tracing, or
higher-order differentiation is claimed.

Run the registered evidence with:

```console
cargo test --locked -p eqiora-python --test python_differentiation
cargo run --locked -p eqiora-verify -- run --case interfaces.python-differentiation
```

The registered native-boundary case is paired with
`bindings/python/tests/test_differentiation.py`, which exercises the installed
wheel through the public `eqiora.diff.compile` wrapper.
