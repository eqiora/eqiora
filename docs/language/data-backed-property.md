# Specimen: consume a data-backed conductivity

This maintained [local package](../../examples/property-composition/src/main.eqi) binds one
exact property release into two ordinary Components. `FourierFlux` evaluates signed flux
and `SlabConductance` evaluates conductance at each supplied temperature sample. Both
receive an explicit periodic clock; this example has no thermal evolution or hidden time solver.

The complete runnable source, exact table, attribution and package manifest live in
[`examples/property-composition`](../../examples/property-composition/README.md).
Its public declarations carry their documentation in the source. The
[installed Python test](../../bindings/python/tests/test_property_composition_specimen.py)
compiles the package, executes both consumers, reopens Model bytes, moves the project with
its vendored lock and repeats execution without the original store.

## Contract and complete consumer

`Conductivity` takes temperature in kelvin and returns conductivity in W/(m K).
`FourierFlux` multiplies this value by the explicitly negative gradient, 2 K/m.
`SlabConductance` multiplies it by the explicit area/thickness ratio, 0.01 m² / 0.1 m.
The root connects its input temperature to both consumers and exposes their two outputs.
Source contains no implicit material installation or numerical provider selection.

`first_open_intervals` is a closed derivative profile: value evaluation on the closed validity
interval and first derivatives on open smooth segments. Endpoint and nonsmooth-knot derivative
requests reject. This is not a promise of arbitrary higher derivatives or automatic smoothing.
The release supplies the actual segment boundaries. The contract permits no history, hidden
independent variable, uncertainty reduction, or external callback.

These are sampled lumped constitutive evaluations, with no spatial support, boundary closure,
stored state, or initialization equation. `gradient` is a signed scalar gradient along a declared
one-dimensional local direction, not an interchangeable three-dimensional vector. The slab
conductance uses a uniform material evaluated at the supplied operating temperature; it is
not an exact nonlinear through-thickness temperature solution.

## Exact release binding

Bind `conductivity` to a synthetic instructional release with the following complete table:

| Temperature (K) | Conductivity (W/(m*K)) |
|---|---|
| 300 | 10 |
| 320 | 14 |
| 360 | 18 |

The data are exact decimal values in the specified coherent units. They describe a constructed
single-branch material, not measured values for a named substance. Scientific provenance is
this synthetic definition; redistribution follows the repository license. A published release
must retain the resolved provenance and license identities, not infer them from a display name.

Use piecewise affine interpolation of conductivity against temperature, with no logarithm,
normalization, filtering, missing-value filling, or fitted derivative table. The validity domain
is the closed interval `[300 K, 360 K]`. Value evaluation at a knot returns the tabulated value.
Outside-domain evaluation rejects; there is no extrapolation, clamping, or positivity floor.

The derivative is the segment slope strictly inside each segment. At 320 K the two slopes
differ, so a derivative request rejects. At both outer endpoints the selected profile also
rejects derivatives rather than choosing an undocumented one-sided convention. Value evaluation
there remains valid. The release exposes its derivative availability before a numerical method
that requires it is selected.

The accepted data artifact has exactly two named real-scalar columns, three rows, a strictly
increasing temperature axis, finite values, no missing entries, and the declared units. Decoder
bounds check those counts and shapes before allocation. Missing content, a mismatched digest,
duplicate or unordered abscissae, nonfinite values, and extra/missing columns reject rather than
being repaired. These are limits for this specimen's release, not universal table-size limits.

The property release binds the artifact's actual content identity through the existing artifact
owner, along with interpolation, validity, branch, derivative, and preprocessing meaning.
Large tables remain outside `.eqi`. Source refers to the release by its exact package declaration;
a mutable filename, network location, or provider default is not the accepted binding.
The [release declaration](properties.md) spells these choices with closed typed children and
an exact package asset reference, without inventing another table file format.

## Independent values and slopes

The two affine segments are:

```text
k(T) = 10 W/(m*K) + (0.2 W/(m*K^2)) * (T - 300 K),  300 K <= T <= 320 K
k(T) = 14 W/(m*K) + (0.1 W/(m*K^2)) * (T - 320 K),  320 K <= T <= 360 K
```

Both give 14 W/(m*K) at the shared knot. At 310 K, conductivity is 12 W/(m*K), the flux is
-24 W/m^2, and conductance is 1.2 W/K. At 340 K they are 16 W/(m*K), -32 W/m^2, and
1.6 W/K. The flux sign follows the stated gradient and the explicit minus sign in Fourier's
law, independently of table implementation.

Inside the first segment, `dk/dT = 0.2 W/(m*K^2)`; inside the second,
`dk/dT = 0.1 W/(m*K^2)`. Multiplication by `area/thickness = 0.1 m` gives conductance slopes
0.02 W/K^2 and 0.01 W/K^2. The endpoint conductivities are exactly 10 and 18 W/(m*K).
The maintained tests derive these expectations from the two line segments.

## Ordinary package use and substitutions

The maintained [Rust source/package specimen](../../crates/eqiora/tests/source_property_tables.rs)
constructs `org.example.Table` with the exact array and attribution documents. The
[installed Python specimen](../../bindings/python/tests/test_table_property_authoring.py)
authors `org.example.PythonTable`, writes its source and assets, resolves the local project
into an explicit store, compiles the resulting lock, and reopens Model bytes. Both use
separate Fourier-flux and slab-conductance Components at 310 K and 340 K, with the independent
values and slopes derived above. The Python specimen executes the four outputs through the
ordinary supplied-clock execution session.

These are reproducible local specimen packages, not published standard-library releases.
The [standard source reference](../site/src/content/docs/reference/standard-packages/index.mdx)
links the maintained examples separately from distributed packages. See
[Python property authoring](../python/modeling.md#exact-analytic-and-table-properties) for
its public construction seam. Contract identity must match the consumer's exact declaration;
copying its spelling and units under a different package identity does not satisfy it.

| Substitution or request | Required outcome |
|---|---|
| A scalar `12 [W/(m*K)]` instead of the release | Wrong binding kind; no callable input contract |
| A release requiring pressure as an additional input | Incompatible independent-variable signature |
| A complex-valued or spatial-tensor conductivity | Incompatible result type |
| A foreign nominal contract with matching units | Contract identity mismatch |
| A history-dependent implementation | Incompatible purity/state contract |
| Temperature 299 K or 361 K | Validity rejection, not extrapolation |
| Derivative at 320 K | Nonsmooth-knot rejection |
| Replacement table with a changed middle value | New release meaning and content identity |
| A different interpolation or preprocessing policy | New release meaning even with identical table bytes |

Reopening the accepted dependency closure offline must reproduce the same binding. Replacing
only a conforming numerical provider preserves the declared property mathematics while changing
execution provenance; replacing data or policy is not merely provider substitution.
