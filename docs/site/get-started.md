# Get started

The public release supports ordinary-GIL CPython 3.11–3.14 on
manylinux x86-64.

## Install

Create a clean environment and install the exact release:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install eqiora==0.1.1
```

## Build and run a model

For this current-checkout example, follow the
[source installation guide](src/content/docs/get-started/index.mdx); the pinned
release above retains its published authoring contract.

This complete example compiles an Eqiora model, resolves a typed adaptive time
policy without inventing a Mesh, and runs the shared native lifecycle:

```python
import eqiora

model = eqiora.compile(source="""
model decay {
  state state: 1;
  initial { state = 1; }
  parameter rate: 1 / s = 1;
  relation flow {
    derivative(state) + rate * state = 0;
  }
}
""")
state = model.field("state")
plan = eqiora.resolve(
    model,
    temporal=eqiora.time.Tsitouras45(
        initial_step_s=0.01,
        relative_tolerance=1e-9,
        absolute_tolerances={state: 1e-11},
    ),
)
result = eqiora.run(
    plan,
    state=eqiora.State.initial(plan),
    until_s=1.0,
    output_times_s=(0.25, 0.5, 0.75, 1.0),
)
series = result.series(state)
time = series.time.numpy(copy=False)
values = series.values.numpy(copy=False)

print(eqiora.__version__)
print(model.digest)
print(time[-1], values[-1])
```

The NumPy arrays returned by `copy=False` are lifetime-safe and read-only. Use
`copy=True` when you need an independent writable array; Eqiora does not turn
an impossible zero-copy request into a hidden copy.

## Choose your next path

- [Python guide](python/index.md) covers spatial modeling, structured diagnostics,
  asynchronous runs, cancellation, and framework adapters.
- [Concepts](concepts.md) explains relations, realizations, and evidence.
- [Examples](examples.md) walks through runnable workflows and links their
  verified counterparts.
- [Capabilities](capabilities.md) summarizes current problem classes and
  maturity.

Rust users can run the small public-facade orientation examples from a source
checkout:

```bash
git clone https://github.com/nkiyohara/eqiora.git
cd eqiora
cargo run --locked -p eqiora --example quickstart
python examples/python/exact_cylinder_stokes.py
```

`quickstart` compiles and runs the decay model above. The exact-cylinder Python
path is its spatial counterpart: it authors Geometry, compiles the
equations-only component, resolves one typed root Plan, and runs it on the host
CPU. The [Examples](examples.md) page walks through that lifecycle.

Before relying on any numerical method or backend, trace its row in the
[capability matrix](capabilities.md) to a registered evidence case.
