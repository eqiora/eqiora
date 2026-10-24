# Python PyTorch first-order differentiation

One optional `eqiora.torch` adapter projects an accepted framework-neutral
`DifferentiableProgram` as a functional PyTorch operator. The native program
retains the exact common scalar Plan, its Model, caller-supplied two-dimensional
rectangular Cartesian Mesh, ordered Parameter identities, and output Field.
The verified Plans use Q1 FEM or cell-centred TPFA with a typed linear solve
policy. An ordered tuple/list supplies the numerical Parameter leaves, each
with its exact native signature shape and scalar domain.

The forward schema is
`eqiora::differentiable_solve(Tensor[], str) -> Tensor[]`. It declares no
mutation and returns owned CPU copies of immutable primal NumPy leaves. The
registered autograd rule invokes
`eqiora::_differentiable_solve_vjp(Tensor[], Tensor[], str) -> Tensor[]`.
One native evaluation/VJP handles all leaves; Python does not loop over solves
or implement a coordinate derivative algorithm.

Fake kernels resolve the immutable retained registry entry's signatures and
validate tensor metadata without reading tensor data or executing a native
program. The real kernels admit contiguous CPU:0 tensors with exact leaf shape
and `float64`/`complex128` dtype. Inputs and cotangents cross the existing DLPack
admission boundary after detachment and resolution of lazy conjugate/negative
views. Native staging owns its copies; output tensors never alias result storage.
Scalar, intrinsic value, coefficient and sampled-time axes retain the native
signature layout without a second shape schema. PyTorch complex gradients use
the same real pairing as native products, `Re(conj(cotangent)*dy)`.

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
- dtype, leaf count, shape, layout, finiteness, token, and metadata falsifiers;
- ordinary mixed real/complex multi-output controls with independently derived
  values and cotangents, unused outputs, conjugate views and immutable binding;
  these do not widen the registered Q1/TPFA scientific claim;
- ordinary seeded-root checks with independently derived opposite-branch values
  and derivatives, duplicate registration, and eager/compiled calls; these do
  not widen the registered Q1/TPFA scientific claim;
- a fresh non-aliasing output and a base `eqiora` import with no PyTorch import.

Registered host evidence obtains this profile from the same complete candidate
and manifest used by the base, typing, and JAX cases. Its PyTorch checks must be
present in that accepted manifest; the focused
`tools/ci/python_torch_gate.py` script remains available for standalone
development but is not a second registered artifact build.

The registered scientific claim remains in-process first-order real CPU `float64`
Q1/TPFA differentiation. Ordinary product tests additionally cover structured
real/complex values and multiple outputs; they do not establish broader
scientific verification.
Double backward, Hessians, `vmap`, AMP/autocast, sparse or arbitrary Tensor
subclasses, CUDA, distributed autograd, `torch.export`, graph serialization,
AOT packaging, module reload, and multiprocessing fork/spawn are nonclaims.

Run the registered installed-wheel evidence with:

```console
cargo run --locked -p eqiora-verify -- run \
  --case interfaces.python-pytorch-differentiation
```
