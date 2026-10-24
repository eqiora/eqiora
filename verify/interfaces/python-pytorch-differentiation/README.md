# Python PyTorch first-order differentiation

One optional `eqiora.torch` adapter projects an accepted framework-neutral
`DifferentiableProgram` as a functional PyTorch operator. The native program
retains the exact common scalar Plan, its Model, caller-supplied two-dimensional
rectangular Cartesian Mesh, ordered Parameter identities, and output Field.
The verified Plans use Q1 FEM or cell-centred TPFA with a typed linear solve
policy. A rank-one Tensor supplies only the numerical Parameter point.

The forward schema is
`eqiora::differentiable_solve(Tensor, str, int, int) -> Tensor`. It declares no
mutation and returns an owned CPU copy of the immutable primal NumPy leaf; neither the input Tensor nor
the immutable Eqiora evaluation/evidence is aliased. The registered autograd
rule invokes
`eqiora::_differentiable_solve_vjp(Tensor, Tensor, str, int, int) -> Tensor`.
It differentiates the accepted implicit relation through Eqiora's VJP rather
than recording solver iterations.

Fake kernels validate Tensor metadata without resolving or executing a native
program. The real kernels admit only native `float64`, rank-one,
C-contiguous CPU:0 Tensors of the exact declared shape. Parameter values and
cotangents cross the existing no-transfer DLPack admission boundary and then
make one Eqiora-owned staging copy per Parameter or output-cotangent leaf.
Primal and reverse products copy the selected read-only NumPy leaves into owned
Tensors; native result storage is never exposed for mutation.
The public wrapper restores the exact native output leaf shape with a Tensor
reshape outside these flat custom kernels. Fake execution and autograd use that
same reshape, preserving scalar and sampled output axes without another numerical
implementation. Binding requires real scalar input leaves and one real output;
multiple output leaves and complex Tensor products remain outside this adapter
profile.

Process-local opaque tokens keep custom-operator schemas free of Python/native
objects. PyTorch and JAX share one native registry. Registrations include the
exact initial State identity as well as Model, Plan and selected values, so
different nonlinear seeds cannot alias. Equal native identities retain one
program until interpreter exit so an
autograd or compiled graph can safely outlive its temporary `TorchProgram`
wrapper. Accepted evaluations and linearizations are never cached; losing
reuse cannot change a gradient. Unknown tokens, mismatched metadata, and
cross-process use fail closed.

The installed-wheel gate uses the declared `torch>=2.14,<2.15` extra and the
exact tested release 2.14.0. It exercises:

- Q1 FEM and TPFA FVM forward values against the native accepted primal;
- backward gradients against the native Eqiora VJP;
- `torch.library.opcheck` for schema, fake Tensor, autograd registration, and
  dynamic AOT dispatch;
- `torch.autograd.gradcheck` as an independent finite-difference oracle;
- eager and `torch.compile(fullgraph=True)` forward/backward agreement;
- retained-graph repeated backward, zero cotangents, and inputs that do not
  require gradients;
- dtype, rank, shape, layout, finiteness, token, and metadata falsifiers;
- ordinary seeded-root checks with independently derived opposite-branch values
  and derivatives, duplicate registration, and eager/compiled calls; these do
  not widen the registered Q1/TPFA scientific claim;
- a fresh non-aliasing output and a base `eqiora` import with no PyTorch import.

Registered host evidence obtains this profile from the same complete candidate
and manifest used by the base, typing, and JAX cases. Its PyTorch checks must be
present in that accepted manifest; the focused
`tools/ci/python_torch_gate.py` script remains available for standalone
development but is not a second registered artifact build.

The exact claim is in-process first-order CPU `float64` differentiation.
Double backward, Hessians, `vmap`, AMP/autocast, sparse or arbitrary Tensor
subclasses, CUDA, distributed autograd, `torch.export`, graph serialization,
AOT packaging, module reload, and multiprocessing fork/spawn are nonclaims.

Run the registered installed-wheel evidence with:

```console
cargo run --locked -p eqiora-verify -- run \
  --case interfaces.python-pytorch-differentiation
```
