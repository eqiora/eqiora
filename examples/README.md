# Examples

Examples are small, readable orientation paths for users.

Run them from the repository root:

```bash
cargo run --locked -p eqiora --example quickstart
cargo run --locked -p eqiora --example project_modules
cargo run --locked -p eqiora --features package-filesystem --example project_modules
python examples/python/exact_cylinder_geometry.py
python examples/python/exact_cylinder_mesh.py
python examples/python/steady_cylinder_source.py > steady-cylinder-authored.eqi
python examples/python/exact_cylinder_stokes.py
python examples/python/exact_cylinder_stokes.py \
  --pressure-png exact-cylinder-pressure.png
python examples/python/karman_vortex_street.py \
  --vorticity-png karman-vortex-street.png
python examples/python/mixed_boundary_elasticity.py
python examples/python/fixed_reference_fsi.py
python examples/python/coupled_scalar.py
python examples/heated-body/run.py
python examples/voltage-divider/run.py
```

| Example | Source | What it shows |
| --- | --- | --- |
| `quickstart` | [`decay.eqi`](decay.eqi) | Compile one scalar decay model and run it through the common root Plan lifecycle. |
| `project-modules` | [`modules/resistor-project`](modules/resistor-project/) | Compile a directly imported public Model from a closed, portable multi-file source inventory whose module identities come from paths below `src/`; the optional `package-filesystem` run discovers that same closure through bounded no-follow directory traversal. |
| `voltage-divider` | [`voltage-divider`](voltage-divider/README.md) | Solve the explicit-ground bundled-parts circuit through Plan/State/Run/Result and observe 4 mA, 8 V and power balance. |
| `heated-body` | [`heated-body`](heated-body/README.md) | Run steady and constant-capacity transient heat with complete 300 K data, Q1, BackwardEuler, exact local packages and offline replay. |
| `poisson` | [`crates/eqiora-api/packages/org.example.poisson`](../crates/eqiora-api/packages/org.example.poisson/) | Compile a 2D Poisson model and exercise its verification-only native reference solve. |
| `coupled-scalar` | [`python/coupled_scalar.py`](python/coupled_scalar.py) | Solve two diffusion/reaction equations with two-way coupling through one Q1 Plan and inspect both Fields in the common Result. |
| `exact-cylinder-geometry` | [`python/exact_cylinder_geometry.py`](python/exact_cylinder_geometry.py) | From an installed `eqiora` package, author the exact rectangle-with-one-circular-hole identity and inspect its fixed-role named selections. |
| `exact-cylinder-mesh` | [`python/exact_cylinder_mesh.py`](python/exact_cylinder_mesh.py) | From an installed `eqiora` package, realize the exact cylinder source with typed Gmsh policy and inspect Rust-derived selection counts. |
| `steady-cylinder-source` | [`python/steady_cylinder_source.py`](python/steady_cylinder_source.py) | From an installed `eqiora` package, author the complete equations-only steady-cylinder Component as bounded `eqiora.lang.Source` values and emit readable deterministic `.eqi` through the same compiler ingress used by hand-written source. |
| `exact-cylinder-stokes` | [`python/exact_cylinder_stokes.py`](python/exact_cylinder_stokes.py) | From an installed `eqiora` package, define the sole concrete Geometry in Python, compile it with the shipped equations-only `.eqi` Component, resolve the common MINI/P1 and linear-solve policies, inspect immutable pressure, solver, force, and flux evidence, and optionally save the pressure through `eqiora[gmsh,matplotlib]`. |
| `karman-vortex-street` | [`python/karman_vortex_street.py`](python/karman_vortex_street.py) | Run the DFG 2D-2 geometry and physical parameters at Reynolds number 100, discard chunked spin-up history, retain a developed wake window, and inspect vorticity, drag, lift, pressure difference, and sampled Strouhal number. The example uses MINI/P1 elements and backward Euler time stepping. |
| `mixed-boundary-elasticity` | [`python/mixed_boundary_elasticity.py`](python/mixed_boundary_elasticity.py) | Define the sole concrete rectangle in Python, compile the shipped equations-only Component, resolve Q1 and linear-solve policies, run the common Result path, and optionally save a caller-owned deformed-field Figure. |
| `fixed-reference-fsi` | [`python/fixed_reference_fsi.py`](python/fixed_reference_fsi.py) | Author the adjacent Geometry in Python, compile the equations-only FSI Component, scope MINI/P1 and P1 to exact Model Domains, initialize four exact Fields, and run the common root Plan/State/Run lifecycle. |
| `steady-flow-past-cylinder` | [`steady-flow-past-cylinder.eqi`](steady-flow-past-cylinder.eqi), [exact geometry](steady-flow-past-cylinder.geometry.json) | Python supplies the concrete Geometry to the equations-only `.eqi`, then uses the common root Plan lifecycle. |

Examples keep the Model, typed numerical Plan, and execution visibly separate;
their mesh, discretization, solver, and placement choices remain explicit. The
exact-cylinder Stokes example uses that same common lifecycle with one bounded
Geometry, Gmsh, MINI/P1, and linear-solve configuration.

The [numerical simulation lessons](https://eqiora.org/learn/numerical-simulation/)
explain how equations, discretization, and a numerical solve fit together.

The cylinder source and Python-authored exact geometry form one checked
workflow. Its 50-chord mesh is an error-controlled realization of the exact
circle.
