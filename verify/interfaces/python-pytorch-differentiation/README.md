# Python PyTorch first-order differentiation

One optional `eqiora.torch` adapter projects an accepted framework-neutral
`DifferentiableProgram` as a functional PyTorch operator. The native program
retains the exact common scalar Plan, its Model, caller-supplied two-dimensional
rectangular Cartesian Mesh, ordered Parameter identities, and output Field.
The verified Plans use Q1 FEM or cell-centred TPFA with a typed linear solve
policy. A rank-one Tensor supplies only the numerical Parameter point.

The forward schema is
`eqiora::differentiable_solve(Tensor, str, int, int) -> Tensor`. It declares no
mutation and returns a fresh CPU DLPack snapshot; neither the input Tensor nor
the immutable Eqiora evaluation/evidence is aliased. The registered autograd
rule invokes
`eqiora::_differentiable_solve_vjp(Tensor, Tensor, str, int, int) -> Tensor`.
It differentiates the accepted implicit relation through Eqiora's VJP rather
than recording solver iterations.

Fake kernels validate Tensor metadata without resolving or executing a native
program. The real kernels admit only native `float64`, rank-one,
C-contiguous CPU:0 Tensors of the exact declared shape. Parameter values and
cotangents cross the existing no-transfer DLPack admission boundary and then
make one Eqiora-owned staging copy. Outputs cross the existing versioned
copy-on-export DLPack boundary.

Process-local opaque tokens keep custom-operator schemas free of Python/native
objects. Registrations are deduplicated by the complete static program
identity and retain only distinct programs until interpreter exit so an
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
