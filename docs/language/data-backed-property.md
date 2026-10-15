# Example: a conductivity defined by a data table

This [local package](../../examples/property-composition/src/main.eqi) defines thermal
conductivity from a small table and uses it in two Components. `FourierFlux` computes heat flux
and `SlabConductance` computes the conductance of a slab, each at the temperature samples you
supply. Both run on an explicit periodic clock; nothing evolves in time.

The source, the table, its attribution, and the package manifest are in
[`examples/property-composition`](../../examples/property-composition/README.md), and the public
declarations are documented in the source. The
[Python test](../../bindings/python/tests/test_property_composition_specimen.py) compiles the
package, runs both Components, reopens the saved Model, moves the project together with its
vendored dependencies, and runs it again without the original package store.

## The two Components

`Conductivity` takes a temperature in kelvin and returns a conductivity in W/(m K).
`FourierFlux` multiplies it by a temperature gradient of 2 K/m with the minus sign of Fourier's
law. `SlabConductance` multiplies it by the area-to-thickness ratio 0.01 m² / 0.1 m. The root
Model feeds one input temperature to both and exposes their outputs. The source names the
property explicitly; nothing installs a material or picks a numerical provider implicitly.

Both are lumped evaluations at a single temperature: there is no spatial field, boundary
condition, stored state, or initial value. `gradient` is a signed scalar along one local
direction, not a three-dimensional vector. The slab conductance assumes the whole slab is at the
supplied temperature; it does not solve for a nonlinear temperature profile through the slab.

## The table

`conductivity` is bound to a synthetic teaching table:

| Temperature (K) | Conductivity (W/(m*K)) |
|---|---|
| 300 | 10 |
| 320 | 14 |
| 360 | 18 |

The values are exact decimals in the stated units. They describe an invented material, not
measurements of a real substance, and they are distributed under the repository license. A
published release records its provenance and license explicitly rather than inferring them from
a display name.

Between knots, conductivity is interpolated linearly in temperature, with no logarithmic scaling,
smoothing, gap filling, or fitted derivative table. The table is valid on the closed interval
`[300 K, 360 K]`, and evaluating it at a knot returns the tabulated value. Temperatures outside
the interval are rejected; the table is never extrapolated or clamped.

The derivative profile is `first_open_intervals`: values are available on the whole closed
interval, and first derivatives only strictly inside each segment. At 320 K the two segment
slopes differ, and at 300 K and 360 K there is no two-sided slope, so derivative requests there
are rejected instead of picking a one-sided value. The release states which derivatives it
provides, so a method that needs them can be rejected before it runs. No higher derivatives,
history, extra independent variables, uncertainty, or external callbacks are involved.

The data file must have exactly two named real columns, three rows, strictly increasing
temperatures, finite values, no missing entries, and the declared units. These shapes are
checked before anything is allocated. Missing content, a wrong digest, repeated or unordered
temperatures, non-finite values, and missing or extra columns are all rejected; nothing is
repaired. Exact coefficients use arbitrary-precision rationals under a bounded arithmetic work
budget, so small values such as micro-scale conductivity do not require rescaling or decimal
reinterpretation. Executable evaluation still uses binary64. Coefficient conversion rejects
overflow and nonzero values that would round to zero; evaluation rejects nonfinite results.
These limits describe this example's table, not a general limit on table size.

The property release records the identity of the data file's content together with the
interpolation, validity interval, derivative profile, and preprocessing. Large tables stay out of
`.eqi` files. The source refers to the release through its package declaration, never through a
file name, URL, or provider default. The [property declaration](properties.md) spells these
choices out without introducing a separate table file format.

## Expected values

The two linear segments are:

```text
k(T) = 10 W/(m*K) + (0.2 W/(m*K^2)) * (T - 300 K),  300 K <= T <= 320 K
k(T) = 14 W/(m*K) + (0.1 W/(m*K^2)) * (T - 320 K),  320 K <= T <= 360 K
```

Both give 14 W/(m*K) at 320 K. At 310 K, conductivity is 12 W/(m*K), the flux is -24 W/m^2,
and the conductance is 1.2 W/K. At 340 K they are 16 W/(m*K), -32 W/m^2, and 1.6 W/K. The
flux is negative because of the stated gradient and the minus sign in Fourier's law.

Inside the first segment, `dk/dT = 0.2 W/(m*K^2)`; inside the second, `dk/dT = 0.1 W/(m*K^2)`.
Multiplying by `area/thickness = 0.1 m` gives conductance slopes of 0.02 W/K^2 and 0.01 W/K^2.
At the end points the conductivity is exactly 10 and 18 W/(m*K). The tests compute these
expected values from the two line segments, not from the implementation.

## Using the property from Rust and Python

The [Rust test](../../crates/eqiora/tests/source_property_tables.rs) builds the package
`org.example.Table` from the array and attribution documents. The
[Python test](../../bindings/python/tests/test_table_property_authoring.py) builds
`org.example.PythonTable` from Python, writes its source and assets, resolves the project into a
package store, compiles it, and reopens the saved Model. Both evaluate the flux and conductance
Components at 310 K and 340 K and compare with the values above; the Python test also runs them
through an execution session with a supplied clock.

These are local example packages, not published standard-library releases; the
[standard package reference](../site/src/content/docs/reference/standard-packages/index.mdx)
lists them separately. [Python property authoring](../python/modeling.md#analytic-and-table-properties)
describes the Python API. A property is identified by its declaration: copying its name and units
into another package does not produce a compatible property.

| Substitution or request | Result |
|---|---|
| A scalar `12 [W/(m*K)]` instead of the release | Rejected: a constant is not a callable property |
| A release that also needs pressure as an input | Rejected: different input signature |
| A complex-valued or tensor-valued conductivity | Rejected: different result type |
| A property from another package with matching units | Rejected: different property identity |
| A history-dependent implementation | Rejected: properties must be pure |
| Temperature 299 K or 361 K | Rejected: outside the validity interval |
| Derivative at 320 K | Rejected: the knot is not smooth |
| A table with a different middle value | A new release with a different identity |
| A different interpolation or preprocessing policy | A new release, even if the table bytes are unchanged |

Reopening the project offline reproduces the same binding. Swapping in another numerical
provider that implements the same property changes only how it is computed, which is recorded in
the run's provenance. Changing the data or the interpolation policy is a different property, not
a provider swap.
