# Examples

Small, readable examples that show how the main parts of Eqiora fit together.

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
| `quickstart` | [`decay.eqi`](decay.eqi) | Compile a scalar decay model and run it through the standard Plan lifecycle. |
| `project-modules` | [`modules/resistor-project`](modules/resistor-project/) | Compile a public Model imported from a self-contained multi-file project whose module names come from paths under `src/`. With `package-filesystem`, the same files are found by a limited directory walk that does not follow links. |
| `voltage-divider` | [`voltage-divider`](voltage-divider/README.md) | Solve a grounded circuit built from bundled parts through Plan/State/Run/Result and check 4 mA, 8 V, and the power balance. |
| `heated-body` | [`heated-body`](heated-body/README.md) | Run steady and constant-capacity transient heat conduction with 300 K boundary data, Q1, BackwardEuler, local packages, and offline replay. |
| `poisson` | [`crates/eqiora-api/packages/org.example.poisson`](../crates/eqiora-api/packages/org.example.poisson/) | Compile a 2D Poisson model and run its native reference solve, which exists for verification only. |
| `coupled-scalar` | [`python/coupled_scalar.py`](python/coupled_scalar.py) | Solve two diffusion/reaction equations with two-way coupling through one Q1 Plan and inspect both Fields in the common Result. |
| `exact-cylinder-geometry` | [`python/exact_cylinder_geometry.py`](python/exact_cylinder_geometry.py) | Build the exact rectangle with one circular hole using the installed `eqiora` package and inspect its named selections. |
| `exact-cylinder-mesh` | [`python/exact_cylinder_mesh.py`](python/exact_cylinder_mesh.py) | Mesh the exact cylinder geometry with typed Gmsh settings and inspect the selection counts computed in Rust. |
| `steady-cylinder-source` | [`python/steady_cylinder_source.py`](python/steady_cylinder_source.py) | Write the steady-cylinder equations as `eqiora.Module` values in Python and emit readable, deterministic `.eqi` through the same compiler path as hand-written source. |
| `exact-cylinder-stokes` | [`python/exact_cylinder_stokes.py`](python/exact_cylinder_stokes.py) | Define the geometry in Python, compile it with the shipped `.eqi` equations, resolve MINI/P1 and linear-solve settings, inspect the pressure, solver, force, and flux results, and optionally save a pressure plot with `eqiora[gmsh,matplotlib]`. |
| `karman-vortex-street` | [`python/karman_vortex_street.py`](python/karman_vortex_street.py) | Run the DFG 2D-2 geometry and physical parameters at Reynolds number 100 with MINI/P1 elements and backward Euler, discard the spin-up history, keep a later observation window, and inspect vorticity, drag, lift, pressure difference, and the sampled Strouhal number. |
| `mixed-boundary-elasticity` | [`python/mixed_boundary_elasticity.py`](python/mixed_boundary_elasticity.py) | Define the rectangle in Python, compile the shipped `.eqi` equations, resolve Q1 and linear-solve settings, run the model, and optionally save a plot of the deformed body. |
| `fixed-reference-fsi` | [`python/fixed_reference_fsi.py`](python/fixed_reference_fsi.py) | Build the adjacent fluid and solid geometry in Python, compile the FSI equations, assign MINI/P1 and P1 to the model's domains, initialize four fields, and run the standard Plan/State/Run lifecycle. |
| `steady-flow-past-cylinder` | [`steady-flow-past-cylinder.eqi`](steady-flow-past-cylinder.eqi), [exact geometry](steady-flow-past-cylinder.geometry.json) | Python supplies the geometry to the `.eqi` equations and then follows the standard Plan lifecycle. |

Every example keeps the Model, the numerical Plan, and execution separate, and
states its mesh, discretization, solver, and placement choices explicitly.

The [numerical simulation lessons](https://eqiora.org/learn/numerical-simulation/)
explain how equations, discretization, and a numerical solve fit together.

The cylinder examples share one exact geometry. Their mesh approximates the
circle with 50 chords, within a controlled boundary error.
