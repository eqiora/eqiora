# Python SDK guide

Use Eqiora from Python to define equations, run models, inspect arrays, and
calculate derivatives with NumPy, PyTorch, or JAX.

Start with the
[package README](https://github.com/nkiyohara/eqiora/blob/main/bindings/python/README.md),
then use the focused guides:

- [Modeling and realization](modeling.md) covers native declarations,
  spatial support, immutable revisions, and scalar elliptic models.
- [Execution, diagnostics, and arrays](execution-and-arrays.md) covers
  synchronous and awaitable runs, cancellation, errors, NumPy, and DLPack.
- [Differentiation and framework adapters](differentiation.md) covers the
  framework-neutral program and PyTorch and JAX adapters.
- [Modeling and realization](modeling.md#exact-cylinder-pressure-rendering) also
  shows cylinder flow, mixed-boundary elasticity, and fixed-mesh fluid–structure
  interaction, including plotting results with Matplotlib.
- [Generated API reference](api.md) lists the available modules, classes, and functions.
- [Measured-force calibration notebook](../../examples/python/measured_force_calibration.ipynb)
  writes the CSV input, physical model, solve and parameter updates in notebook cells.
  It requires the development checkout installed in the kernel.
- [Measured heat sensor](../../examples/python/measured_heat_sensor.py) compares a
  unit-aware temperature CSV at explicit spatial positions and retained times.
  The script includes the complete model and requires the development checkout.

See the
[capability matrix](https://github.com/nkiyohara/eqiora/blob/main/docs/capability-matrix.md)
for the current implementation and verification status of each feature.
