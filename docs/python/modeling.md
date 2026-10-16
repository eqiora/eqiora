# Modeling and realization

Use the environment from [Get started](/get-started/). The sections below show
how to declare a model, connect its geometry, and select numerical methods. For
a mathematical introduction, choose a subject in [Learn](/learn/).

## Native declarations

Define fields, parameters, initial conditions, and equations in Python, then
compile them into an immutable Model.

```python
import eqiora

x = eqiora.Field("x", role=eqiora.FieldRole.State)
rate = eqiora.Parameter(
    "rate",
    value=1.0,
    value_type=eqiora.ValueType.real(eqiora.Dimension(time=-1)),
)
flow = eqiora.Relation(
    "flow",
    equations=((eqiora.derivative(x) + rate * x, 0),),
)
module = eqiora.Module("decay", x, rate, eqiora.Initial((x, 1)), flow)
model = eqiora.compile(source=module)
```

`Field.value_type` holds the field's scalar domain (real or complex), physical
dimension, and component structure. `FieldRole.Variable` declares an algebraic
unknown, and `FieldRole.State` a quantity that evolves or has history, whatever
its support. An omitted type means a dimensionless real scalar.

The Field itself stores no initial value. `Initial` supplies simultaneous
equations, each an ordered left/right pair, that apply when a run starts from
scratch. For scalar ODEs and supported index-one DAEs they are solved together
with the regular equations. Missing state data, contradictory constraints, or an
unsupported case is an error; this is not a general high-index DAE solver. A
restart continues from a saved State and history without applying the initial
equations again.

```python
voltage = eqiora.ValueType.complex(eqiora.Dimension(mass=1, length=2, time=-3, current=-1))
body = eqiora.Domain.box("body", (0.0, 1.0), (0.0, 1.0))
channels = eqiora.Field(
    "channels",
    role=eqiora.FieldRole.Variable,
    domain=body,
    value_type=eqiora.ValueType.array(eqiora.ValueType.vector(voltage, 2), 3),
)
```

`ValueType.real(dimension)` and `ValueType.complex(dimension)` construct scalars.
`vector(scalar, extent)` and `tensor(scalar, *extents)` add spatial axes, while
`array(element, extent)` adds a channel axis and leaves the element's frame
unchanged. Types with the same number of components are still different types.
Spatial extents must match the Field's Domain. Complex execution is still under
development.

Dimensions accept rational exponents, for example
`eqiora.Dimension(length=Fraction(-1, 2))` with `Fraction` imported from `fractions`.
Initial equations are type-checked like any other expression, so a nonzero
dimensioned constant needs a quantity with compatible units. A scalar is not
broadcast into a vector, tensor, or array initial state. Distributed execution
requires explicit initial data.

The same `value_type=` objects work for `Parameter`,
`eqiora.lang.Component.field`, and `eqiora.lang.Component.parameter`.
`ValueType.to_eqi()` returns the type as Eqiora source.

`Field`, `Parameter`, and `Expression` handles support channel slices such as
`values[0:2]`, which return new expressions. Give both bounds as integers and no
step; negative or Boolean bounds and empty slices are errors, and bounds are never
clamped. The same `values[0:2]` syntax works in `.eqi`. A source declaration can
take its size from a static Parameter, as in `array<integer, n>`, bound at compile
time with
`eqiora.compile(source=source, entry="Channels", bindings={"n": 3, "data": (2, 3, 5)})`.
Changing a size or an index requires recompiling; ordinary coefficient changes
still use `preview_value_edit` and `commit`.

Parameter values can be real or complex scalars, or nested sequences matching the
declared shape. Reading a value returns nested tuples with every real and
imaginary component:

```python
coefficients = eqiora.Parameter(
    "coefficients",
    value_type=eqiora.ValueType.array(eqiora.ValueType.complex(), 2),
    value=[1 + 2j, 3 - 4j],
)
assert coefficients.value == (1 + 2j, 3 - 4j)
selected = coefficients[1]
```

A nonzero spatial vector or tensor coefficient needs a frame, taken from an
existing Domain. Its value is still a uniform Parameter:

```python
body = eqiora.Domain.box("body", (0, 1), (0, 1))
kind = eqiora.ValueType.tensor(eqiora.ValueType.real(), 2, 2)
coefficient = eqiora.Parameter(
    "coefficient", value_type=kind, value=((2, 3), (5, 7)), frame=body,
)
response = eqiora.Field(
    "response", role=eqiora.FieldRole.Variable, domain=body, value_type=kind,
)
law = eqiora.Relation("law", equations=((response, coefficient),), domain=body)
module = eqiora.Module("Coefficients", body, coefficient, response, law)
model = eqiora.compile(source=module)
assert model.parameter("coefficient").value == ((2, 3), (5, 7))
```

Include the frame Domain in the `Module`; leaving it out, or using one from
another Module, is an error. The Domain provides the Cartesian frame and the
number of spatial dimensions; it does not restrict where the Parameter applies.
Real and imaginary parts and axis order are preserved through inspection, edits,
and replay, and an array of tensors stays distinct from a single spatial tensor.
With `eqiora.lang`, the same value is written
`q.tensor_value(frame=body, components=((2, 3), (5, 7)))`, where `body` is a
`Component.volume` or `Component.boundary` handle from the same Component, and
`Component.set_default` can use the result as a Parameter default. The components
must be constant scalar expressions; they cannot refer to named model values,
including Parameter aliases.

Indices are fixed nonnegative integers. They may use a Parameter whose value is
known at compile time, but changing a Parameter used in an index, slice, or size
then requires recompiling. Typed value edits keep the full declared type and every
component through replay. Being able to write complex values does not mean that a
complex numerical solver is available.

Declare `value_type=eqiora.ValueType.integer()` to keep Python integers exact,
including values above `2**53`. Values must fit in a signed 64-bit integer;
booleans, floats, and out-of-range values are errors. Without an integer type,
numeric defaults are treated as real. `ParameterRef.value` returns the typed
value, also after edits and Model replay.

```python
species = eqiora.FiniteSpace("Species", labels=("A", "B"))
population = eqiora.Parameter(
    "population", value_type=eqiora.ValueType.counts(species),
    value=(2, 9007199254740993),
)
observed = eqiora.Field("observed", role=eqiora.FieldRole.Variable)
relation = eqiora.Relation("observation", equations=((observed, 0),))
module = eqiora.Module("Population", species, population, observed, relation)
model = eqiora.compile(source=module)
assert model.parameter("population").value == (2, 9007199254740993)
```

`ValueType.coordinates(ValueType.integer(), species)` holds signed integer
components in the same ordered basis, while `counts(species)` requires
nonnegative components. Another `FiniteSpace` with the same labels is a different
type. Likewise, `IndexSet("Rows", extent=3)` and `ValueType.index(rows)` define
their own type, whose values are the ordinals 0 to 2. Include each of these
declarations in the `Module`. Because such named types can only be written as
source inside their declaring scope, calling `to_eqi()` on one alone raises an
error.

With `eqiora.lang`, register spaces with `source.space(...)` and fixed-size index
sets with `component.index_set(..., extent=3)`. The `counts`, `coordinates`, and
`index` constructors require handles from the same Module or Component.
`eqiora.lang.quotient`, `remainder`, `to_real`, `to_integer`, and `ordinal`
follow the compiler's explicit conversion and arithmetic rules. These discrete
fields do not support products, dual spaces, general maps, or dynamic indexing.

Continuous coordinates use `ValueType.coordinates(ValueType.complex(), space)` or
a real scalar type with physical dimensions. `dual=True` gives the dual
coordinates, and `ValueType.linear_map(scalar, source, target)` records the map's
ordered endpoints, with explicit `source_dual` and `target_dual` flags. A
Component builds values with `coordinates` and `linear_map`; `linear_map` takes
target rows and source columns and is written `linear_map(A, B, rows)` in source.

`eqiora.lang.apply`, `compose`, `pair`, `transpose`, and `adjoint` correspond to
the finite operations of the same name in source. `pair` is bilinear, so pairing
`adjoint(x)` with `x` conjugates the first vector. `transpose` dualizes the
endpoints without conjugating, while the orthonormal adjoint swaps the endpoints
and conjugates the coefficients. Spaces from different declarations, channel
arrays, and spatial vectors are not interchangeable, even when their sizes agree.
Finite affine Plans support real and complex maps; tensor products and general
finite derivatives are not yet available.

Finite scalar reductions bind one symbolic index through `Component.sum`, `Component.product`,
`Component.min` or `Component.max`:

```python
rows = component.index_set("Rows", extent=3)
total = component.sum(lambda i: (q.ordinal(i) + 1) * (q.ordinal(i) + 1), over=rows)
component.let_alias("total", total)
```

Here `q` is `eqiora.lang`. The callback runs once to build the expression, and the
compiler expands the three terms to 14. A nested reduction needs its own `name`,
such as `name="j"`. The index cannot be used outside its callback or stored in
another declaration, and the index set and anything the callback refers to must
belong to the same Component. Inside the callback, an array can be indexed as
`values[q.ordinal(i)]`; general run-time indexing is not supported. The
[finite reduction rules](../language/numeric-catalog.md#finite-scalar-reductions)
cover term order, scalar types, units of products, size limits, and the
initializers where reductions are not allowed. `min` and `max` keep the integer or
real type and the dimension of their terms, evaluate every term, and return the
first term in a tie; they are not general conditional expressions.

A bare number as a Parameter default takes the coherent unit of the declared
dimension: `parameter rate: 1 / s = 1;` equals `parameter rate: 1 / s = 1[1 / s];`.
Write an explicit unit to convert. This applies only to numeric Parameter
defaults; nonzero literals elsewhere never acquire units implicitly. See
[Units and dimensions](../language/units.md#declaration-initializers).

A native Relation receives ordered `equations=((left, right), ...)` pairs.
Use `(residual, 0)` for a numerical residual equation. Named `equal`, `not_equal`,
`less`, `less_equal`, `greater`, and `greater_equal` functions produce predicates;
`logical_not`, `logical_and`, and `logical_or` combine them. Python's `==`, `<`,
and similar operators, and truth tests such as `if expression:`, raise errors on
symbolic expressions. Declarations and expressions are immutable, and Rust
validates them and creates the Model in a single step.

## Derived observables

To observe a value at a fixed point, use
`lang.evaluate(value, at=((coordinate, quantity), ...), side=None)`, giving each
coordinate of one support exactly once. Scalar Cartesian Q1 values and first
coordinate derivatives are reconstructed from the Result. `side="lower"` or
`"upper"` chooses the side of a one-dimensional limit, which gradients on cell
boundaries require. See
[coordinates and point evaluation](../language/coordinates.md#point-evaluation-and-one-sided-traces)
for what is supported.

Coordinate maps use the same operations as source:

```python
mapped = lang.pullback(x * x + x * y, from_=(xi, eta),
                       at=((x, 2 * xi + eta), (y, 3 * eta)))
scale = lang.volume_jacobian(from_=(xi, eta),
                             at=((x, 2 * xi + eta), (y, 3 * eta)))
```

Here `x`, `y`, `xi`, and `eta` are coordinate handles from the same model body.
Python uses `from_` because `from` is a reserved word; the generated source uses
`from`. `lang.jacobian_determinant` and `lang.map_orientation` take the same
keyword arguments. Use `mapped * scale` as the transformed integral density. Each
source and target coordinate must appear exactly once with compatible units, and
handles from another model body are rejected before compilation. A scalar pullback
does not transform vector frames or infer an embedded measure. See
[explicit coordinate pullbacks](../language/coordinates.md#explicit-coordinate-pullbacks)
for what currently executes and the reference values used to check it.

An `observable` adds a typed expression to the Model without adding an unknown or
an equation. Lumped values and spatial integrals use the same declaration:

```eqi
observable output: V = lower.positive.voltage - ground.terminal.voltage;
observable energy: J = integral(capacity * (temperature - reference), measure(body));
observable outward_heat: W = integral(normal(-conductivity * grad(temperature)), measure(wall));
```

`integral(expression, measure(domain))` uses a volume or surface measure
according to the kind of Domain, and remembers which body a boundary belongs to.
The result's dimension includes the measure. Boundary values of a Field are
written `trace(field)`, and outward flux uses the `normal` operator. Another
Domain of the same size cannot stand in for the declared one.

Each integral must be the whole expression of its own Observable; other
Observables can then combine the results, as in `observable total:J=bulk+surface;`.
References between Observables must not form a cycle. Coordinate-factor integral
densities may read Observables. In finite real scalar Plans, continuous Relations
other than initial conditions may read lumped Observables, including polynomial
coordinate integrals: `relation inventory { count=135/2; }`, for example, can fix
a density amplitude from its integrated count. The Plan chooses the polynomial
quadrature, and replaying the Model, Plan, or Result keeps the same integral.
Nonpolynomial integral coupling, spatial unknowns, and discrete or initial
Relations are not supported here.

A sum or difference of fixed functionals over the same volume can be used in
`variation(total, wrt=..., direction=..., holding=(...))`. The `holding` tuple
must name every other independent Field and Parameter the functionals depend on.
Nonlinear combinations can be observed and differentiated with a State JVP, but
cannot yet be used in a variation.

In native Python, use `eqiora.Observable(name, value_type=..., expression=...)`
and `eqiora.integral(expression, eqiora.measure(domain))`, and include each
declaration in its `Module`. Native Observables and `eqiora.lang.Observable`
handles can be combined with arithmetic operators to define another Observable.
Component bodies are checked whether or not they are instantiated. A private
Component Observable can still be inspected by its qualified name, but it is not
exported as a Port and cannot be used as a value in equations.

Observables are evaluated on a Result of the same Model. Spatial evaluation
currently covers real scalar Q1 Fields on a Cartesian mesh and planar P1 Fields
on an affine triangle mesh, explicit traces, scalar expressions, and outward
normal gradients. Volume and boundary integrals use the exact physical measure;
boundary normals point outward from the selected parent Region, including slanted
edges. A Result from the Cartesian Q1
elasticity Plan also reconstructs the two-component displacement for strain-energy
observations and State JVPs.
Only Fields used by the requested Observable are reconstructed. A mixed MINI/P1
Result can therefore integrate its P1 pressure or a constant physical measure
without reconstructing the unrelated MINI velocity. Integrals that request MINI
velocity reconstruction remain unsupported.

Installed Python P1 conformance covers `AffineTriangleMesher` output for both a
single rectangle and an adjacent-region partition, with a scoped
`P1().at(domain)` policy for each Region. Both use the same authenticated mesh
owner and retain physical integrals through Plan and Result replay.

An elastic energy can also provide a first variation. Declare a displacement
direction with units of m, hold the Lamé coefficients and the conservative-load
potential fixed, and list exactly the sides with essential conditions in
`zero_on`. For fixed 2D isotropic elasticity, the variation is compared with the
stress and load of the strong form before solving. Supported boundaries are
homogeneous essential sides, explicit zero traction, and constant traction given
by a real vector Parameter. For a nonzero traction `t`, add
`-integral(contract(t, trace(displacement), axes=((0,0),)), measure(boundary))`
over that boundary to the energy and hold `t` fixed. The surface term must use the
boundary law's sign and the same Parameter, not another Parameter that happens to
have equal values. The energy survives Plan replay, and the Result evaluates the
same bulk-plus-surface energy. Spatially varying traction is not supported. In a
scalar Q1 problem, a prescribed flux can likewise add surface work to the energy.

Spatial integration always uses an explicit quadrature rule and accepted Field
coefficients. Rendered values and output cadence do not select the quadrature. The State JVP uses the same basis and quadrature for Field and
normal-gradient directions, holding Parameters and geometry fixed. Sensitivities
through the solve and with respect to shape are separate features.

Look up the Observable on the Model, then evaluate it on the Result. For a
Component instance named `definition`:

```python
energy = model.observable("definition.energy")
observation = result.observe(energy, quadrature_points=2)
value = observation.value
lineage = observation.result_identity
```

For an ODE or fixed-geometry spatial Trajectory, pass `output_index` to select a
retained output position. `observe` evaluates finite ODE values and spatial
reductions; `observe_at` samples spatial fields. Both use the selected State,
and `time()` uses its accepted time. For example, `output_steps=(1, 3)` makes
`output_index=1` select step 3:

```python
observation = result.observe(energy, quadrature_points=2, output_index=1)
output_index, time_s, state_identity = observation.output
```

`observation.output` is `None` for an instantaneous Result, which rejects an
output index. A Trajectory requires an explicit valid index. ODE observations
retain real/complex shape and can evaluate stored derivatives or the highest
flow rate. An output index selects the requested output schedule; it does not
select an arbitrary time or the terminal instant of the Run. Current
queries reuse admitted Q1/P1 nodal reconstruction for fixed geometry; moving
geometry and State derivatives at retained outputs are not admitted. Sparse
outputs do not provide missing accepted intervals or continuous-time extrema.

For a finite real scalar ODE Observable, a Plan with `ForwardSensitivity` also
retains the native Parameter products at each requested output. Apply a direction
with the Parameter's units:

```python
sample_jvp = result.observe_parameter_jvp(sample, direction, output_index=1)
```

This returns the same output time, State identity and Result identity as `observe`;
`evaluation_kind` is `"parameter-jvp"`. The time stays fixed, and the value includes
both the Observable's direct Parameter dependence and the State sensitivity,
including reset propagation. A nonzero direction at an event exactly at the selected
time rejects: a zero first-order event-time product alone does not prove a stable
observation side under finite perturbations. Missing forward products, wrong direction
units, complex/vector derivative outputs and spatial Parameter products reject.
Saved Results retain these native products; the query adds no interpolation.

Finite ODE Observables can reference other finite Observables. For example, put
the measurement comparison and its units in the Model once:

```eqi
observable residual:m/s = velocity - measured(elapsed=time());
observable normalized:1 = residual / 0.01[m/s];
observable loss:1 = normalized * normalized / 2;
```

Here `measured` is a bound speed-history property and `velocity` is the solved
state. The normalization is an explicitly chosen speed uncertainty. Observing
`loss` or applying its Parameter direction follows the same dependencies for
retained outputs, terminal values, and time integrals. Conditional expressions
evaluate only the selected dependencies; an inactive measurement branch cannot
force an out-of-range table query. A predicate depending on a Parameter or State
still rejects differentiation at a tie, even when its first product is zero.
Time-only predicates remain fixed for Parameter products.

Own a fitted Parameter in the Model and pass it into a Component, for example
`parameter mass:kg=2; instance body:Driven(mass=mass, ...);`. The Component slot
then refers to that same Parameter. Within an instantiated Component, a literal
argument or local default is not a separate editable Parameter. Use `preview_value_edit` and `commit` to
change the Model, then resolve and run the new revision.

An integral over some coordinates keeps the others as its output support:
`observable density:1/m on position=integral(f,measure(velocity));`.
In the Component builder, `on=position` gives the output support and
`integrate_over=velocity` the measure. To evaluate this field-valued output at a
point, call
`result.observe_at(density, [(1.0, eqiora.Dimension(length=1))], quadrature_points=3)`;
the coordinates are given in order and checked for units. The returned
`observation.point` holds the output Domain ID and coordinates, while a lumped
`observe` result has `point=None`. Integrating another Observable composes the
reductions. A weighted mean is written as an explicit numerator over a
denominator, and a zero denominator is an error. This works for regular real
scalar coordinate densities and finite Result amplitudes. It does not reconstruct
arbitrary spatial phase or radial diffusion Fields, and it does not support
nonpolynomial coupling to a solver, general differentiation under the integral
sign, or State JVPs of these integrals.

In source, `spherical_measure(radius)` integrates with `4*pi*r^2 dr` over a
length-valued `CoordinateInterval(0, R, dimension=Dimension(length=1))`. The result
has volume units and uses the same quadrature and Result replay. Radial diffusion
Fields, their centre condition, and spherical functional derivatives are separate
features.

A first `partial` of one named polynomial integral over fixed coordinate
intervals, applied to the whole declaration, is supported with respect to an
independent Parameter or a coordinate of the remaining output support. It uses
the same Result quadrature and `observe_at`.

Finite explicit limits use the same source integral:

```eqiora
model Moving(support line: interval(m)) {
  coordinate x: m on line from line;
  parameter a: m = 1 [m];
  variable anchor: 1;
  relation fixed { anchor = 1; }
  observable total: m^3 = integral(x*x, measure(line), lower=0[m], upper=a);
  observable slope: m^2 = partial(total, wrt=a);
}
```

Compile this source with `entry="Moving"` and
`bindings={"line": eqiora.CoordinateInterval(-4, 4, dimension=eqiora.Dimension(length=1))}`.
With an ordinary finite Plan and two Gauss points, the Result gives
`total=1 m³/3` and `slope=1 m²`, and the saved Model, Plan, and Result keep these
definitions. The limits must have the coordinate's unit and lie inside the declared
support. Reversing the limits negates the integral, and equal limits give zero.
The Parameter derivative includes the derivative of the density and the signed
contributions of both endpoints. This requires a polynomial density, polynomial
limits, a single Cartesian coordinate interval, and no unknown Field in the
density. Moving geometry, derivatives through a solve, higher derivatives, and
solver constraints on the limits are not supported. The Component builder cannot
yet declare an abstract interval, so write this case in source.

`quadrature_points` sets the number of Gauss–Legendre points per axis. Physical
simplex measures use the Duffy-transformed rule for their exact reference cell;
Cartesian measures and coordinate factors use tensor-product rules. Point
boundaries always use one point, so an output made only of point integrals
requires `1`; mixed outputs apply the requested rule to each measure of positive
dimension. `observation.quadratures` maps each Domain ID to a
`(kind, dimension, points_per_axis)` tuple, with kind `"GaussLegendre"`,
`"SimplexDuffyGaussLegendre"`, or `"Point"`. Lumped values take no
`quadrature_points` and have an empty map.

Create a State direction with
`result.observable_state_tangent({field: (dimension, coefficients)})` and apply it
with `result.observe_state_jvp(energy, direction, quadrature_points=2)`. The
observation's `evaluation_kind` is `"state-jvp"`, and the direction cannot be used
with another Result. For a fixed polynomial functional,
`result.observe_state_second_variation(energy, first, second, wrt=field, quadrature_points=2)`
applies two ordered directions; its `evaluation_kind` is
`"state-second-variation"`. Both directions must come from this Result and hold
every other Field fixed. The compiler derives the same density as for authored
variations, and evaluation uses the Q1 basis on the integration Domain. Sums and
differences of spatial Observables are supported, but nonlinear combinations of
their values are not. Use enough quadrature points for the resulting polynomial,
for example three for a degree-four integrand on each cell. This is a derivative
with respect to the State only: it is not a Hessian through the solve, a stability
test, or a derivative with respect to a moving domain.

For vector Fields, direction coefficients follow the mesh's vertex order, with each
vertex's components stored together. A State direction need not satisfy the
boundary conditions, and a State JVP alone does not show that an authored
stationarity equation holds.

### Spatial time integrals

A finite Observable over fixed-geometry spatial States can be integrated through
covered accepted steps with the same `observe_time_integral` method:

```python
window_energy = result.observe_time_integral(
    energy,
    interpolation=eqiora.time.TimeObservationInterpolation.EndpointLinear,
    quadrature_points=2,
    interval_s=(0.5, 0.75),
)
```

The time rule integrates the linear interpolant of the Observable values at each
accepted step's endpoints. It does not interpolate State coefficients. Spatial
integrals reuse the same explicit quadrature and physical reconstruction as
`observe`; `spatial_quadratures` records those rules. The result's physical type
includes the time measure. ODE histories also admit this endpoint rule, in
addition to their native-midpoint Simpson rule.

Every requested interval must be covered by retained accepted steps before any
density is evaluated. For a step size of 0.25 and `output_steps=(1, 3)`, the exact
steps cover `[0, 0.25]` and `[0.5, 0.75]`. The saved predecessor at 0.5 does not
prove the missing interval `[0.25, 0.5]`. A query on `[0.5, 0.75]` succeeds; a
query across that gap fails. Omitting `interval_s` requests the entire Run and
therefore requires complete coverage. Moving geometry, spatial Parameter/State
derivatives, native spatial midpoints, and continuous-time extrema remain outside
this profile. Field-valued outputs must first be reduced or explicitly sampled
into a finite Observable.

### ODE trajectory functionals and Parameter directions

A finite ODE Observable can be evaluated at the final state or integrated over the
whole run with an explicit numerical rule:

```python
sample = model.observable("sample")
terminal = result.observe_terminal(sample)
integral = result.observe_time_integral(
    sample, interpolation=eqiora.time.TimeObservationInterpolation.NativeMidpointQuadratic
)
window_integral = result.observe_time_integral(
    sample, interpolation=eqiora.time.TimeObservationInterpolation.NativeMidpointQuadratic,
    interval_s=(0.2, 0.7),
)
```

The same interpolation policy also defines real scalar extrema:

```python
minimum = result.observe_time_minimum(
    sample, interpolation=eqiora.time.TimeObservationInterpolation.NativeMidpointQuadratic,
    interval_s=(0.1, 0.4),
)
maximum = result.observe_time_maximum(
    sample, interpolation=eqiora.time.TimeObservationInterpolation.NativeMidpointQuadratic,
    interval_s=(0.1, 0.4),
)
print(minimum.value, minimum.extremum_time_s)
```

These are extrema of the selected Observable interpolant, with the original
Observable units, rather than error bounds for the continuous solution. The closed
window includes pre-event and post-event values even at either boundary. Exactly
equal computed values choose the earliest physical time, then the earliest accepted
occurrence; no hidden comparison tolerance is used. Spatial queries use
`EndpointLinear` and the same explicit `quadrature_points` as spatial integrals.
Missing coverage, complex/vector extrema and unavailable spatial midpoints reject.
No restart State interpolation or extrema derivative is provided.

The integral applies Simpson's rule on each accepted solver step, using the
solver's values at the start, midpoint, and end of the step. It is therefore a
numerical approximation, and it does not depend on the output times you requested.
Its `value_type` is the Observable's dimension times seconds, and `interval_s`,
`interpolation`, and the Result and Trajectory identities record how it was computed.
The terminal value uses the run's end time even if that time was not a requested
output. A saved Result keeps the history needed for both.

`interval_s` selects fixed endpoints in seconds within the complete accepted
history. Empty, reversed, non-finite and incompletely covered intervals reject.
For a window cutting an accepted step, the rule integrates the quadratic through
that step's three Observable values over only the selected part. The same rule
is exact for quadratic-in-time Observables; general Observables retain quadrature
error. This does not construct a State interpolant or provide arbitrary-time
samples. Event resets separate the continuous pieces and never become a hidden
impulse. Omit `interval_s` to integrate the complete Run.

A Tsitouras45 Plan with events records the state just before and just after each
located reset. If the Plan also requests `ForwardSensitivity`, you can
differentiate the same terminal or integral value along a Parameter direction
with units:

```python
direction = {threshold: (eqiora.Dimension(), 1.0)}
terminal_jvp = result.observe_terminal_parameter_jvp(sample, direction)
integral_jvp = result.observe_time_integral_parameter_jvp(
    sample, direction,
    interpolation=eqiora.time.TimeObservationInterpolation.NativeMidpointQuadratic,
)
```

The integral JVP accepts the same `interval_s` keyword. Its endpoints stay fixed;
a moving event exactly at either endpoint has no admitted two-sided derivative.

The derivative includes the Observable's direct dependence on the Parameter, the
sensitivity of the continuous state, the shift in event times, and propagation
through resets. It is an error if the history or sensitivities are missing, if
the direction refers to another Model, if an event falls exactly at the end time,
or if a resumed run has no initial sensitivities. Space-time functionals,
adjoints, sub-intervals, moving endpoints, and grazing events are not supported.

## Normalizing numeric inputs

Use the same `eqiora.units` values for numerical inputs and source quantities.
`Unit.normalize` returns a coherent-SI float and its Dimension:

```python
from decimal import Decimal
from eqiora import units as u

seconds, time_dimension = u.s.prefixed("m").normalize(Decimal("500"))
force, force_dimension = u.N.prefixed("m").normalize(Decimal("2000"))
# seconds == 0.5; force == 2.0
```

The compiler composes the exact decimal value with the
unit scale, then rounds once to binary64. This also handles a decimal whose
unscaled value is outside binary64 range when its normalized value is finite.
Nonfinite values, nonzero underflow, invalid scale roots and unsupported units
reject. Floats use their shortest decimal spelling, just as `lang.quantity` does.
These multiplicative units do not convert Celsius or other affine temperatures.

For a numeric CSV, map its columns explicitly. The adapter parses each selected
cell as a `Decimal` and uses that same conversion:

```python
import eqiora
from eqiora import data, units as u

array, dimensions = data.csv_array(
    "elapsed_ms,force_mN\n0,0\n500,2000\n1000,0\n",
    columns=(("elapsed_ms", u.s.prefixed("m")), ("force_mN", u.N.prefixed("m"))),
)
assert dimensions == (eqiora.Dimension(time=1),
                      eqiora.Dimension(mass=1, length=1, time=-2))
# Write array.to_bytes() to the consuming package's data/force_samples.json.
```

The array contains row-major coherent-SI values and has shape `(3, 2)`. Its native
codec and digest are the same ones used by package resources. `ResolvedArray`
also accepts explicitly shaped `from_f64` and `from_u64` values, preserving exact
unsigned integers. `from_bytes` checks the current artifact's ordinary decoding
limits, and `to_bytes` emits canonical bytes.

Header names and mappings must be unique; selected missing or nonnumeric cells,
inconsistent row widths and malformed quoting reject. Unselected columns remain
uninterpreted. The adapter preserves row order and applies no time alignment,
sorting, filling or interpolation. The consuming property release declares
coherent-SI axis/value units, validity, interpolation and extrapolation; its
existing table owner rejects duplicate or unordered times and enforces derivative
rules. Check the returned dimensions against that consumer before binding data.

The array digest identifies normalized values and shape. To retain raw-source
provenance, include the original CSV and mapping/units in the package's cited
documentation. Conversion alone does not attest provenance or sensor support.

The [measured-force calibration notebook](../../examples/python/measured_force_calibration.ipynb)
builds this path from scratch: CSV columns, a force-driven model, speed residuals,
dimensionless loss, native Parameter products, immutable mass edits and artifact
reopening. Its scalar Gauss–Newton loop is written in the notebook and checked
against the hand-integrated force. Use a kernel with the development checkout
installed; the released 0.1.3 wheel predates these APIs.

For a spatial sensor, [measured_heat_sensor.py](../../examples/python/measured_heat_sensor.py)
contains the CSV mapping, full heat-model source, package assembly, Q1 solve and
comparison. Run it with `python measured_heat_sensor.py --output measured-heat`
in that same development environment. It normalizes time in milliseconds,
coordinates in millimetres and absolute temperature in kelvin. A stationary
sensor belongs to the explicitly selected body support. Missing cells and
duplicate times reject; recorded times must coincide exactly with the declared
step grid, without snapping or reconstructing unretained States. Supply
`--csv measurements.csv --license attribution.txt` to use your own data in
the same column format and retain its attribution.

The model defines temperature residual, uncertainty normalization and loss once;
`Result.observe_at(..., output_index=...)` evaluates those Observables at the
recorded positions and accepted times. Their metadata preserves the support,
coordinates, State and Result identity through replay. With four Q1 cells and a
0.25 s backward-Euler step, the center coefficients obey
`(1/9)(e_n-e_(n-1))/0.25 + (8/3)e_n = 1/4`, giving
`e_n = (3/32)(1-7^(-n))`. The example sensor sees half that center hat.
This independent discrete solution checks both zero and biased residuals; it is
not a claim of continuum discretization accuracy or spatial Parameter derivatives.

## Writing Eqiora source from Python

`eqiora.Module` builds Eqiora source from Python:

```python
import eqiora
from eqiora import lang as q
from eqiora import units as u

source = eqiora.Module("main")
component = source.component("Diffusion")
body = component.volume("body", dimensions=2)
value = component.field("value", on=body, role=eqiora.FieldRole.Variable, value_type=eqiora.ValueType.real(eqiora.Dimension(length=1)))
length = component.parameter("length", value_type=eqiora.ValueType.real(eqiora.Dimension(length=1)))
wave_number = component.let_alias("wave_number", q.math.pi / length)
component.relation(
    "balance", q.equation(q.div(q.grad(value)), -(wave_number**2) * value),
    on=body,
)

text = source.to_eqi()
model = eqiora.compile(
    source=source,
    entry="Diffusion",
    geometry=geometry,
    bindings={"body": geometry.selection("body"), "length": 1.0},
)
```

A Module Relation takes `q.equation(left, right)`, written `left = right;` in
source; use an explicit `0` on the right for a residual equation. Python's `==`
does not build an equation. The Module rejects handles from another Module and
anything over its size limits, and it becomes read-only the first time it is
written out or compiled. The output is ordinary UTF-8 `.eqi`. `doc=` values become
`///` comments attached to the declaration, with blank paragraphs written as empty
`///` lines; documentation is limited to 16,384 UTF-8 bytes. `write_eqi(path)`
writes to a temporary file in the same directory and then replaces the target,
so a failed write never leaves a partial file.

A Module can contain several Components.
Use `parent.instance(...)` to bind a child's requirements explicitly, and select the
entry with `eqiora.compile(source=source, entry="Parent", ...)` when the source
contains multiple public Components. Instance bindings and returned outputs use
their declared names as string keys.

`component.let_alias(name, expression)` names an expression inside a Component.
Its type and spatial support are inferred; pass `value_type=` or `on=` to check
that the inferred type or support (which must belong to this Component) is the one
you expect.
Aliases can use the component's Parameters, fields, and other aliases. For example,
`heat_flux = component.let_alias("heat_flux", coefficient * q.grad(potential))`
can appear in a volume relation as `q.div(heat_flux)`. A Parameter-only alias can also
supply a nested-instance Parameter argument; a field-dependent alias cannot.

An alias is not a parameter, an edit target, an unknown, or an equation. Wherever
it is used, it keeps its original dependencies, so Parameter edits and
differentiation see through it. Its spatial support must follow from the
expression itself, so keep `coordinate`, `trace`, and `normal`, whose meaning
depends on context, inside the relations that use them. `on=` cannot give a
constant a spatial support or choose a boundary the expression leaves open, and a
separately declared Support of the same shape is still a different support.
`at=clock` checks that everything the expression depends on at run time belongs to
that Clock of this Component; two Clocks with the same period are still different.
Static expressions, and expressions that mix continuous and clocked values, cannot
be tied to one clock. State operators keep their clock and initialization rules
inside aliases, and reading the current value of a state through an alias adds no
clock restriction.

Create a periodic clock with its period in seconds, given as an integer or a
`fractions.Fraction`:

```python
from fractions import Fraction

tick = component.clock("tick", period_s=Fraction(1, 10))
memory = component.field(
    "memory", on=body, role=eqiora.FieldRole.State,
    value_type=eqiora.ValueType.real(), at=tick,
)
component.initial(left=q.pre(memory), right=1)
observed = component.let_alias("observed", memory, on=body, at=tick)
component.relation(
    "update", q.equation(q.next(memory), q.pre(memory)), on=body, at=tick,
)
```

`phase_s` is optional and defaults to zero. The initial equations are solved
together, and the first tick comes after initialization. A clock handle can only
be used in the Component that declared it. `q.pre` and `q.next` are checked by the
same rules as in source. Before running, check that your execution backend
supports the spatial and time methods the model needs.

Python does not type-check a Module. Compiling a Module directly hands it to the
Rust compiler without generating and re-parsing text, and then applies the same
type checking, lowering, and geometry binding as for parsed source. `to_eqi()`
produces the source text, and `eqiora.Module.parse("main", text)` turns it back
into a Module; attach imports with `import_module`. Compiling a Module directly
and compiling its written-out file with the same bindings give the same
`structural_fingerprint`, although the artifacts are not byte-identical. Editing
documentation changes the source bytes and the package's source-bundle identity,
but not the Model. Errors in constructed declarations point to graph paths, and
errors in parsed declarations point to source locations.

`q.math.pi` is an expression written as `math.pi`, and `q.math.sin(expression)`
is written as the corresponding function. The top level of `eqiora.lang` holds
equation structure such as `q.grad` and `q.div`, and scalar functions and
constants are under `q.math`. These build expressions; they do not compute
numbers in Python, and the compiler defines their types and values.

A Module can also write the property declarations used in a Model Package:

```python
source = eqiora.Module("main")
contract = source.property_contract("Diffusivity", value_type=eqiora.ValueType.real())
release = source.property_release(
    "ReferenceDiffusivity",
    implements=contract,
    value=25,
    source_unit=u.one,
    source_scale=0.001,
    citation="org.example.measurement",
    license="spdx.CC0_1_0",
)
law = source.component("DiffusionLaw")
law_body = law.volume("body", dimensions=2)
diffusivity = law.property("diffusivity", contract=contract)
value = law.field("value", on=law_body, role=eqiora.FieldRole.Variable, value_type=eqiora.ValueType.real())
law.relation(
    "balance", q.equation(-q.div(diffusivity * q.grad(value)), 0), on=law_body,
)
material = source.material_composition(
    "ReferenceMaterial",
    properties={"diffusivity": release},
)
root = source.component("DiffusionProblem")
root_body = root.volume("body", dimensions=2)
root.instance(
    "equation",
    component=law,
    bindings={"body": root_body, "diffusivity": material["diffusivity"]},
)
source.write_eqi("src/property-diffusion.eqi")
```

The composition mapping may contain several releases. Each instance binding
selects a member explicitly, such as `diffusivity = ReferenceMaterial.diffusivity`.
Compilation checks that every required property is supplied exactly once and
that each release implements the required contract.

Contracts and releases in the same Module can be compiled directly with
`eqiora.compile(source=source, entry=..., bindings=...)`, with the same checks. A
handle of the same name from another Module is rejected before any source is
written.

To record where each property came from, write the `.eqi` into a Model Package,
lock it, and use `compile_package`; `property_bindings` on the resulting Model
then shows the provenance. Direct compilation does not record it.

### Analytic and table properties

A contract can declare ordered, named real inputs and a `first_partials` or
`first_open_intervals` derivative profile. Use `contract.input(name)` inside an
analytic release, and call the Component's property requirement with named
arguments. Imported contracts and releases keep their identity through
`ModuleRef.property_contract` and `ModuleRef.property_release`.

The [maintained table example](../../bindings/python/tests/test_table_property_authoring.py)
authors the complete offline package, its exact resolved array and citation/license documents,
and two independent consumers. Its property declaration uses:

```python
source = eqiora.Module("main", package="org.example.PythonTable")
temperature = eqiora.ValueType.real(eqiora.Dimension(temperature=1))
conductivity = eqiora.ValueType.real(
    eqiora.Dimension(mass=1, length=1, time=-3, temperature=-1)
)
contract = source.property_contract(
    "Conductivity", value_type=conductivity, inputs={"temperature": temperature},
    derivatives="first_open_intervals",
)
samples = source.property_table_release(
    "Samples", implements=contract, data="conductivity_samples",
    axis_unit=u.K, source_unit=u.kg * u.m / u.s**3 / u.K,
    validity=(q.quantity(300, u.K), q.quantity(360, u.K)),
    citation="synthetic_definition", license="repository_license",
)
```

The full example writes `data/conductivity_samples.json` with exact points
`(300, 10), (320, 14), (360, 18)` and both attribution files under `docs/` before
`resolve_local_project` and `compile_package`. At 310 K and 340 K it checks Fourier
fluxes −24 and −32 W/m² and conductances 1.2 and 1.6 W/K. Table values enforce the
closed validity interval; active first derivatives reject knots and endpoints.
The [independent derivation](../language/data-backed-property.md) states the arithmetic
and non-claims. This local specimen is not a published standard package.

The complete current vocabulary and steady-cylinder Component are shown in
[`examples/python/steady_cylinder_source.py`](../../examples/python/steady_cylinder_source.py).
The source authoring API supports one public Component, public volume/parent-boundary
supports and parameters, typed scalar, spatial-vector/tensor and channel-array
continuum fields, continuous residual Relations, structural SI units, constants,
coordinates, arithmetic,
powers, gradient, divergence, trace, normal contraction, symmetric part, and
isotropic lift. Package authoring supports multiple scalar contracts
and constant releases, one material composition, one consumer plus one root
Component, and complete direct or composed bindings.

### Enum declarations, values and exhaustive cases

`eqiora.Enum` constructs an exact native declaration. Its `member(name)` method returns
an immutable `EnumValue`; include the declaration in the native Model alongside its users:

```python
mode = eqiora.Enum("Mode", members=("Heating", "Cooling", "Fault"))
heating = mode.member("Heating")
parameter = eqiora.Parameter("mode", value_type=mode.value_type, value=heating)
```

`mode.id`, `mode.members` and `mode.value_type` expose its identity and type.
`EnumValue.enum_id` and `EnumValue.value_type` retain that same owner. Repeated member
selection compares equal, but a separately constructed enum with the same spelling is
foreign. Strings, integers and Booleans are not enum values; `bool(heating)` rejects.
Values cannot be ordered, used in arithmetic, or substituted for a numerical zero.

The Module route authors symbolic members and complete cases through the same compiler:

```python
from eqiora import lang as q

source = eqiora.Module("main")
mode = source.enum("Mode", members=("Heating", "Cooling", "Fault"))
owner = source.model("Controller")
tick = owner.clock("tick", period_s=1)
drive = owner.input("drive", value_type=mode.value_type, at=tick)
level = owner.output("level", value_type=eqiora.ValueType.real(), at=tick)
owner.relation(
    "classify",
    q.equation(level, q.case(drive, [
        (mode.member("Heating"), 2),
        (mode.member("Cooling"), -3),
        (mode.member("Fault"), 0),
    ])),
    at=tick,
)
model = eqiora.compile(source=source, entry="Controller")
compiled_mode = model.enum("Mode")
session = model.execution_session(
    end_time_s=0, max_step_s=0.1,
    inputs={"drive": ("tick", [compiled_mode.member("Cooling")])},
)
session.advance_ticks(1)
# session.output("level", 0) is (Fraction(0), -3.0).
```

`q.case` takes an ordered sequence of symbolic enum-member/value pairs. Compilation
requires every member exactly once, rejects foreign declarations, and checks all branch
types. There is no wildcard arm. Runtime selection is lazy; Python constructs the symbolic
arms without using Python truthiness or retaining a callback in the Model.

Module enum handles are authoring identities. After compilation, obtain runtime values
from `model.enum("Mode")`, not from a separately constructed native declaration.
`Model.enum` also accepts an exact enum ID, which is useful after artifact replay when
lexical names may be absent; a replayed declaration's `name` can be `None`.
Typed Parameter edits, execution-session inputs, outputs, fields and checkpoints retain
exact enum values and reject foreign members. Initial equations must select an explicit
member. Enum values remain scalar and dimensionless; enum arrays, records, buses and
enabled-mode transition semantics are outside this profile. The scalar numerical trajectory
API is not the discrete-value transport.

## Named scalar weak forms

A Component may own one named scalar weak form with an explicit dimensionless test.
The test names its trial Field and the exact boundary supports on which it vanishes:

```eqiora
form weak_heat for heat_balance {
  test w: 1 for temperature zero_on surface;
  integrate(body, dot(grad(w), conductivity * grad(temperature)))
    = integrate(body, w * heating);
}
```

`surface` is a caller-bound complete-exterior support. A comma-separated list of
individual boundary support names is also accepted. The compiler resolves exact
Model boundary identities and rejects foreign parents and duplicate members.
Resolution requires this restriction to equal the admitted homogeneous essential
boundary inventory; missing essential sides or extra natural sides are rejected.
Other sides may carry explicit homogeneous natural flux laws such as
`normal(conductivity * grad(temperature)) = 0`. Their exact constitutive flux
discharges the boundary term without imposing a zero test trace.
The current execution profile is steady scalar diffusion with Cartesian Q1. Tests
vanish on essential sides even when the prescribed Field trace is nonzero.
Nonzero natural loads require surface work and remain outside this authored profile,
as do storage, arbitrary weak forms,
authored integral-conservative and mixed forms remain unsupported.

Python uses `w = component.test("w", for_=temperature, zero_on=surface)` followed by
`component.weak_form("weak_heat", [heat_balance], equations=[(left, right)])`. Use
`component.boundaries(left, right, bottom, top)` for an explicit boundary collection.
The former `primal_form` and global `test(field)` entry points are removed.
`model.authored_formulations` exposes the form name, test name, exact trial and
`zero_on_domain_ids`. Plan replay preserves and rechecks the same restriction.

The closed scalar certificate establishes strong-to-weak correspondence conditionally
on a fixed domain, classical divergence and boundary traces, and an admissible H1 test
with zero essential trace. It accounts for every boundary and the physical flux sign.
It does not prove those regularity hypotheses, the reverse implication, coercivity,
convergence or solver suitability. Model identity remains independent of authored forms;
the authored source/projection identity changes when its test restriction changes.

## Resolve and lock a local package project

For a complete walkthrough of directory layout, public declarations, local imports,
and GitHub distribution, start with [Create and share a library](/guides/creating-libraries/).

An installed Eqiora distribution can add an exact standard fluid or solid
dependency to an existing project through the same manifest/lock transaction:

```python
import eqiora

resolution = eqiora.add_bundled_dependency(
    ".", "package-store", "Eqiora.Fluid.Incompressible", version="0.6.0"
)
```

Create the store directory first. The request must match the release
shipped in the distribution. The solid package is
`Eqiora.Solid.LinearElasticity`, version `0.6.0`.
The manifest records `version = "0.6.0"` and `sources = [{ bundled = true }]`
for the fluid dependency. Each dependency lists its explicit candidate sources.

`eqiora.toml` is the author-maintained project and package manifest. It owns the
canonical name, exact version, source root, entry module, and direct
dependencies:

```toml
[package]
name = "org.example.application"
version = "0.1.0"
source = "src"
entry = "models.main"

[dependencies."org.example.materials"]
version = "1"
sources = [{ path = "packages/materials-1.2" }, { path = "packages/materials-1.3" }]

[dependencies."org.example.components"]
version = "2.1.0"
sources = [{ path = "packages/components" }]
```

Each dependency directory contains its own `eqiora.toml`. Dependency paths are
relative to the declaring manifest; `../library` explicitly selects a sibling
outside the project directory. Absolute paths and parent segments after named
directories are rejected. Each package's source root remains confined to that
package directory. Package names, not local aliases, authorize imports.

Requests are literal constraints, not caret compatibility promises:

| Request | Matching releases |
| --- | --- |
| `1` | Stable `1.x.x` |
| `1.2` | Stable `1.2.x` |
| `1.2.3` | Exactly `1.2.3` |
| `0.3` | Stable `0.3.x`; bare `0` is rejected |
| `1.2.3-rc.1` | Exactly that prerelease |
| `>=1.2.3,<2.0.0` | Stable versions between the stated endpoints |

Bounded ranges require a complete lower and upper stable version, using `>` or
`>=` followed by `<` or `<=`. Endpoints cannot contain prerelease or build
metadata. Exact requests retain build metadata. There is no `latest`, wildcard,
caret or automatic prerelease advancement.

Explicit update freezes all configured candidate manifests, source files and
README bytes before searching. Canonical package-name order and descending
SemVer precedence select the first complete transitive solution, backtracking
when requests conflict. One canonical name has one release throughout the graph.
Identical content mirrors coalesce; different content for the same release and
unresolved equal-precedence alternatives are rejected. Diagnostics include the
conflicting requests and dependency paths. A cache hit never changes selection.

Version matching is not scientific equivalence. The existing compiler still
checks schema, public declarations, units, types and signatures before accepting
the selected closure. Acquisition or validation failure does not select an older
release. A failed selection, installation or publication preserves the accepted
manifest/lock pair.

`entry` selects a module relative to the source root: `models.main` selects
`src/models/main.eqi` with the default source root. The generated package
manifest retains this selection for offline compilation. `entry_model` selects
a Model in that module or through one of its explicit imports.

Python resolves that project into the store and atomically writes its current
project lock to `eqiora.lock`:

```python
from pathlib import Path

import eqiora

store_root = Path("package-store")
store_root.mkdir()
resolution = eqiora.resolve_local_project(".", store_root)
assert eqiora.open_project(".", store_root) == resolution

model = eqiora.compile_package(
    store_root,
    resolution,
    entry="materials.Calibration",
)
```

To move the project offline, create a destination directory and copy the accepted
closure with `eqiora.vendor_project(".", store_root, "vendor")`. After moving the
project, `eqiora.open_project(".", "vendor")` returns the validated resolution for
`compile_package`. Reopening checks current root sources and every locked package
without reading external dependency paths or selecting bundled releases.

`fetch_project` fills a store from the explicit source requests only if they still
match the accepted lock. `update_project` explicitly re-derives that lock from the
current sources. Compilation and reopening never update requests or fetch packages.
The CLI uses the same operations: `package add --bundled`, `package fetch`,
`package update`, `package vendor --destination`, and `package check`, each with
the project path and `--store`.

Inspect a frozen update before committing it:

```python
proposal = eqiora.preview_local_project(".")
print(proposal.explanation)
proposed_lock = proposal.lock
resolution = proposal.commit(store_root)
assert Path("eqiora.lock").read_bytes() == proposed_lock
```

`proposal.resolution` contains the exact semantic closure; `proposal.lock` also
records authored requests and Git provenance. Commit consumes the proposal and
publishes the already validated bytes, without selecting or downloading again.
If the manifest or lock changed after preview, create a fresh proposal. The CLI's
`eqiora package preview .` prints the same canonical proposed lock as JSON without
installing packages or writing a lock. `package update` performs a fresh explicit
preview and commit through the same API.

Ordinary reopen, compile and run use the existing exact lock. Adding a newer
candidate does not advance it; fetch only materializes its selected content.
Changing a request, even to a wider range containing the current selection,
requires an explicit update. Transport relocation alone does not change package
identity or the accepted semantic edge.

### Explicit Git sources

On Linux, add a public HTTPS repository or an explicit absolute/`./`/`../` local
repository path through the project API:

```python
resolution = eqiora.add_git_dependency(
    ".", "package-store", "org.example.Materials", version="1.0.0",
    repository="https://example.org/materials.git", revision="refs/heads/main",
)
```

The CLI equivalent is `eqiora package add . org.example.Materials --version 1.0.0
--git https://example.org/materials.git --rev refs/heads/main --store package-store`.
Revisions are lowercase full 40-digit commit IDs or explicit `refs/heads/...` /
`refs/tags/...` names. Arbitrary revision expressions are rejected.

`eqiora.lock` uses the current `eqiora.project-lock.v2` envelope, containing authored
requests, exact selected semantic edges and separate immutable Git commits.
HTTPS commit provenance retains the explicit public repository locator so sources
using the same ref name cannot borrow each other's pin. Changing that locator
requires an explicit update. Local Git paths are not stored in the lock: a
relocated explicit local source may supply the same accepted commit. Transport
errors and rejected objects are failures, not permission to select an older version.
API return bytes remain the semantic resolution
accepted by `compile_package`; they are not the entire project lock. `fetch_project`
retains the accepted commit even if its branch moves. `update_project` explicitly
resolves the authored request again. `open_project`, compile and run never invoke Git.

Git acquisition requires `/usr/bin/git`, `/usr/bin/prlimit` and home-backed `TMPDIR`.
Each acquisition has a 90-second deadline; Git runs with 1 GiB address space,
64 MiB per-file storage, 60 CPU seconds, 64 open descriptors and bounded output.
Admission limits are 64 MiB stored inventory / 10,000 entries, 4,096 source files,
32 directory levels, 8 MiB per expanded blob and 32 MiB total source bytes.
One project acquires at most 16 repositories with nesting depth 8.

There is no checkout, hook/filter execution or submodule acquisition. Local repository
configuration and object alternates are not accepted. Git tree symlinks, gitlinks,
path escapes and conflicting paths fail before publication. Fetched package-local
dependencies stay inside the fetched tree; fetched packages cannot select ambient
local Git repositories. HTTPS is unauthenticated, with redirects and credential
helpers disabled; userinfo, query strings and fragments are rejected. Unsupported
containment environments fail instead of running an uncontained fetch.

The package builder opens manifest-relative paths without following symbolic
links, finds `.eqi` files, generates each closed package
manifest, prepares the exact graph leaf-first, and publishes the lock only
after the complete closure is installed.
An optional package-root `README.md` is retained as documentation in the exact
source bundle; it is never compiled as model source.

Use `eqiora.add_local_dependency(project_root, store_root, name, version="1.0.0",
path="packages/library")` to add or replace a direct dependency, and
`eqiora.remove_local_dependency(project_root, store_root, name)` to remove it.
Both return the new semantic resolution bytes. The complete candidate is validated before the
manifest and lock are published; a failed update preserves the accepted pair.
Remove source imports before removing a dependency they require.

The CLI uses the same operations:

```bash
eqiora package lock . --store package-store
eqiora package add . org.example.Library --version 1.0.0 --path packages/library --store package-store
eqiora package remove . org.example.Library --store package-store
eqiora package check . --store package-store --entry-model Main
```

If a process stops during publication, locked compilation reads the previous
accepted lock. The next explicit update recovers the saved pair before resolving.
Concurrent project writes are rejected; retry after the other operation finishes.

For portable package source, create modules with their explicit package identity:
`Module("main", package="org.example.Exterior")` and
`Module("parts", package="org.example.Exterior")`. `import_module` emits that exact
package-qualified source import; `Module.parse` accepts the same `package=` when
reopening emitted source. Standalone modules default to `eqiora.local_project`.
The existing package and normalized virtual-path validators own these names.
Attached modules in other packages create explicit direct dependency edges; no
ambient or transitive import is added. A namespace identifies authored source;
locked package compilation separately authenticates the bundle, version and digest.

## Author exact boundary families

`Module.field_connector` declares named trace and flux quantities. Their
`ValueType`s determine shape and frame; `spatial_vector=True` uses the existing
ambient-dimension-dependent spatial vector contract with scalar quantity types.
`Component.port(..., connector=connector, on=boundary)` binds an individual port.

Use `Component.complete_exterior("exterior", parent=body)` for an exact exterior
signature requirement. Its `member("face")` handle scopes a boundary family:
`Component.port(..., on=face)`, `relation(..., on=face)`, and
`connect(..., over=face)` share that exact binder. Select family ports with
`port[face]` before reading their named quantities. An ordinary instance binds
an exterior with `Component.boundaries(left, right, bottom, top)`; the compiler
checks exact parent identity, duplicate members, and completeness.

When the selected root itself requires an exterior, supply its explicit
Geometry selections and parent through the same `bindings` argument used by
`compile` and `compile_package`:

```python
parent = geometry.selection("body")
bindings = {
    "body": parent,
    "exterior": (
        tuple(geometry.selection(name) for name in ("left", "right", "bottom", "top")),
        parent,
    ),
}
```

These Geometry bindings retain the exact canonical revision across emitted
source and artifact replay. Complete-exterior admission currently uses validated
Cartesian box and planar rectangle primitive topology.
`Component.connect_periodic(first, second)` on a Model declaration explicitly
authors periodic topology; its endpoints must satisfy the compiler and Geometry periodic pairing contract.

`ModuleRef.connector(name)` returns a public imported connector descriptor, and
`ModuleRef.component(name)` retains its named physical port interfaces. Imported
ports keep their declaring connector's nominal identity, including boundary
family selection binders. A locally declared connector with equal quantity types
is a distinct connector.

`ModuleRef.record(name)`, `ModuleRef.enum(name)`, and `ModuleRef.space(name)`
return read-only nominal type handles for the exact imported declaration.
Component fields, parameters, enum members, record constructors, and finite-space
counts keep the qualified import path in emitted source; the provider declaration
is not copied into the consumer.

`ModuleRef.operator(name)` returns an immutable callable for a public pure operator
in the exact imported Module. Calls supply every formal by name and preserve the
provider's declared argument order and types. For example,
`main.import_module("laws", provider).operator("conductivity")(x=temperature,
k0=base, a=slope)` authors a qualified call in `main`; it does not copy the
provider's declaration. Explicit parsed Modules and local `.eqi` imports use the
same path. The existing compiler checks units, purity, and bounded definition
closure. Imported signatures retain concrete scalar dimensions and generic spatial
tensor rank; channel arrays cannot replace spatial vectors or tensors.
Numerical execution remains within the admitted real scalar profile;
imported calls can participate in `lang.partial` with explicit independent and
held bindings. Calls across package boundaries inside operator definitions,
Boolean/integer operator signatures and complex execution remain outside that profile.

## Compile one exact locked package Model or Component

Python can bind an existing content-addressed package's public Component to
Geometry and produce an immutable `Model`. Here `support_bindings` explicitly maps every support
name in the selected signature to a Geometry selection; each boundary maps to
`(boundary_selection, parent_selection)`:

```python
from pathlib import Path

import eqiora

store_root = Path("package-store")
resolution = Path("resolution.canonical.json").read_bytes()
model = eqiora.compile_package(
    store_root,
    resolution,
    geometry=geometry,
    entry="PoissonRectangle",
    bindings={**support_bindings, "wave_number": 3.14159, "source_scale": 19.7392},
)

print(model.digest)
print(model.package_compilation_digest)
for binding in model.property_bindings:
    print(binding.contract, binding.release, binding.normalized_value)
    print(binding.validity, binding.citation, binding.license)
```

The caller selects one explicit store directory and supplies the exact bytes
from `ResolutionRecordV1.canonical_json()`. The required `entry=` names the
selected public Model or Component. `bindings=` explicitly supplies its required
signature inputs, with `geometry=` authenticating any Geometry selections. Eqiora verifies the complete locked dependency graph.
Human-formatted, reordered, newline-terminated, duplicate-key, or
store-mismatched resolution bytes fail closed. Missing or ambiguous support
bindings fail instead of matching Geometry by bounds, coordinates, or digest.

`package_compilation_digest` identifies the package compilation.
`property_bindings` is read-only metadata about exact constant, analytic and table
releases: contract, consuming Component, input names, derivative profile, value type,
validity, citation and license. `normalized_value` is present only for constant releases.
The resulting `Model` enters ordinary `eqiora.resolve(model, mesh=..., ...)` and
`eqiora.run(plan)`; its `Plan` and `Run` retain the same digest. Bare Model JSON
still carries Model/Geometry meaning but not the package sidecar, so replayed
Models use the same resolver with `package_compilation_digest is None` while retaining
the exact `property_bindings` carried by Model expression occurrences. Supply the store and lock bytes explicitly; this function works locally.

## Check one exact package structurally

Check that a locked package compiles and can be serialized and reopened:

```python
report = eqiora.check_package_conformance(
    store_root,
    resolution,
    entry_model="Main",
    profile="eqiora.package.structural-conformance-v1",
)

print(report.packages)
print(report.package_compilation_digest)
print(report.model_digest)
```

The read-only operation accepts one explicit store, exact canonical resolution
bytes, one bare root-local Model selector, and the exact profile token shown
above. It compiles and replays the closure twice through the existing package
and current Model boundaries, then returns immutable in-process facts only
after package-compilation and Model identity agree. Rejections raise the
existing structured `EqioraError` family and return no partial report.

This check examines structural compatibility. Assess the equations and
numerical results separately when deciding whether a package suits your problem.

## Authored CAD to exact geometry

Construct a channel with a circular hole using a rectangle, a circle, and
subtraction:

```python
import eqiora

graph = eqiora.geometry.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.0, 2.2), y_bounds=(0.0, 0.41))
circle = graph.circle(center=(0.2, 0.2), radius=0.05)
fluid = graph.subtract(rectangle, circle)
geometry = graph.build(
    fluid,
    named_topology={
        "fluid": fluid.region,
        "inlet": rectangle.boundaries[0],
        "outlet": rectangle.boundaries[1],
        "walls": rectangle.boundaries[2:],
        "cylinder": circle.boundaries[0],
    },
)

assert geometry.selection_dimension("fluid") == 2
assert geometry.selection_dimension("cylinder") == 1
```

Coordinates and radii use metres. `GeometryGraph` also provides
`rectangle_extrusion(...)` and `circular_through_cut(solid, ...)` for solid
authoring. The solid depth and CAD tolerances are separate from the derived
2D geometry's classification tolerance. Meshing later approximates the circle
with straight segments; its centre and radius remain unchanged.

## Generate a Gmsh mesh

The matching meshing operation is an explicit typed provider choice:

```python
request = eqiora.meshing.GmshMesher(
    maximum_boundary_error=1e-4,
    maximum_target_size=0.05,
    minimum_mean_ratio=1e-5,
    maximum_boundary_facets=50,
)
plan = eqiora.meshing.resolve(geometry, request)
mesh = eqiora.meshing.generate(plan)

```

`resolve` plans the boundary subdivision without launching Gmsh. `generate`
runs Gmsh 4.15.2 and returns a mesh of linear triangles with named selections.
Missing Gmsh, a different version, or invalid output raises an error.

`maximum_target_size=None` lets Gmsh choose the global characteristic size.
Set a finite positive value to choose it yourself; this target does not guarantee
the length of every edge. This operation supports a rectangle with one circular
hole and affine 2D triangles.

## Explicit linear solver intent

`eqiora.solve.Linear` requires either `objective=eqiora.solve.Robust` (or `Fast`
or `LowMemory`) or a complete `algorithm`, `preconditioner`, `reduction`, and
`provider` tuple. The examples below use exact manual requests. Their algorithms
and complete provider release identities survive Plan replay without substitution.
`SolverProvider.reference()` and `SolverProvider.faer()` describe the corresponding
compiled backend, including its implementation version and library inventory.

An objective ranks only admissible candidates. A diagonal that has not been
established excludes Jacobi; an identity-preconditioned candidate can still be
selected. An objective does not guarantee performance, memory usage, or that a
reproducible-reduction candidate is available. `Plan.solve` exposes the actual
selected tuple and its reasons; manual requests have no ranking objective.

## Exact-cylinder steady Stokes result

The first fluid application keeps the component's equations, fields,
dimensions, Parameters, and abstract support names in the installed `.eqi`
source. Python defines the concrete shape and size. `compile` checks
that exact Geometry selections close the selected public Component, derives
Parameter dimensions from its declarations, and returns the ordinary immutable
`Model` used by every resolver:

```python
from importlib.resources import files

model = eqiora.compile(
    path=files(eqiora).joinpath("examples", "steady-flow-past-cylinder.eqi"),
    entry="SteadyFlowPastCylinder",
    geometry=geometry,
    bindings={
        "fluid": geometry.selection("fluid"),
        **{
            name: (geometry.selection(name), geometry.selection("fluid"))
            for name in ("inlet", "outlet", "walls", "cylinder")
        },
        "dynamic_viscosity": 1.0e-3,
        "zero_pressure": 0.0,
        "inlet_speed": 0.3,
        "channel_height": geometry.bounds[1][1] - geometry.bounds[1][0],
    },
)
linear = eqiora.solve.Linear(
    algorithm=eqiora.solve.LinearSolver.SparseLu,
    preconditioner=eqiora.solve.Preconditioner.Identity,
    reduction=eqiora.solve.Reduction.Fast,
    provider=eqiora.solve.SolverProvider.faer(),
    relative_tolerance=1e-6,
    absolute_tolerance=1e-13,
    maximum_iterations=10_000,
)
plan = eqiora.resolve(
    model,
    mesh=mesh,
    spatial=eqiora.fem.MiniP1(),
    solve=linear,
    scaling=None,
)
result = eqiora.run(plan)

pressure = result.output(plan.capability.pressure)
pressure_values = pressure.values("vertex")
print(pressure.coefficient_count("vertex"))
print(result.solve)
print(min(pressure_values), max(pressure_values))
force = result.boundary_force(geometry.selection("cylinder"))
inlet = result.boundary_flux(geometry.selection("inlet"))
outlet = result.boundary_flux(geometry.selection("outlet"))
print(force.on_domain, inlet.value + outlet.value)
```

`compile` is keyword-only and accepts exactly one of `path=` or `source=`;
`filename=` labels diagnostics when using `source=`.

Select the velocity and pressure outputs with their field handles. Field values,
mesh coordinates, and connectivity provide read-only NumPy views in matching
mesh order. Use geometry selections to calculate cylinder force and inlet/outlet
flux from the result.

This example uses a 2D steady Stokes component, a channel with a circular hole,
MINI/P1 elements, and SparseLU. The runnable file is
[`examples/python/exact_cylinder_stokes.py`](../../examples/python/exact_cylinder_stokes.py).

## Exact-cylinder pressure rendering

Install the Gmsh and Matplotlib adapters and ask the same runnable file to save
the pressure field:

```console
uv add 'eqiora[gmsh,matplotlib]'
uv run python eqiora-source/examples/python/exact_cylinder_stokes.py \
  --pressure-png exact-cylinder-pressure.png
```

The example files come from the [repository download](https://eqiora.org/get-started/#example-files);
the Eqiora dependency comes from PyPI. The equivalent composition API is:

```python
import eqiora.matplotlib as eqplot

pressure = result.output(plan.capability.pressure)
figure = eqplot.plot_scalar_field(result, field=pressure.field)
figure.savefig("exact-cylinder-pressure.png")
```

The adapter plots P1 pressure using the mesh coordinates and triangle
connectivity, with a scale in pascals. Gouraud shading interpolates the
vertex values for display.

Matplotlib remains optional and is not imported by base `eqiora`. This
helper currently renders scalar vertex or cell fields as still images.

## Mixed-boundary structural result

The installed package also carries the mixed-boundary elasticity
source. Compile the equations, select the numerical settings, and run the
resulting Plan:

```python
from importlib.resources import files

graph = eqiora.geometry.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
geometry = graph.build(
    rectangle,
    named_topology={
        "body": rectangle.region,
        "x_lower": rectangle.boundaries[0],
        "x_upper": rectangle.boundaries[1],
        "y_lower": rectangle.boundaries[2],
        "y_upper": rectangle.boundaries[3],
    },
)
mesh_plan = eqiora.meshing.resolve(
    geometry,
    eqiora.meshing.CartesianMesher(cells=(16, 16)),
)
mesh = eqiora.meshing.generate(mesh_plan)
model = eqiora.compile(
    path=files(eqiora).joinpath("examples", "mixed-boundary-elasticity.eqi"),
    entry="MixedBoundaryElasticity2d",
    geometry=geometry,
    bindings={
        "body": geometry.selection("body"),
        **{
            name: (geometry.selection(name), geometry.selection("body"))
            for name in ("x_lower", "x_upper", "y_lower", "y_upper")
        },
        "mu": 3.0, "lambda": 0.0, "length_scale": 1.0,
    },
)
plan = eqiora.resolve(
    model,
    mesh=mesh,
    spatial=eqiora.fem.Q1(),
    solve=eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.ConjugateGradient,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Reproducible,
        provider=eqiora.solve.SolverProvider.reference(),
        relative_tolerance=1.0e-10,
        absolute_tolerance=1.0e-12,
        maximum_iterations=10_000,
    ),
)
result = eqiora.run(plan)

displacement = result.output(plan.fields[0])
mesh = displacement.mesh
```

The Plan contains the mesh, Q1 elements, solver settings, and execution
placement. Unsupported combinations raise an error.

Select the displacement field from the Result. Its values, mesh coordinates,
and Q1 connectivity provide read-only NumPy views in matching order. This
example uses small-strain linear elasticity.

The optional still displays original and explicitly scaled deformed edges:

```python
import eqiora.matplotlib as eqplot

figure = eqplot.plot_deformed_field(
    result,
    field=plan.fields[0],
    scale=1.0,
)
figure.savefig("mixed-boundary-displacement.png")
```

The complete runnable workflow is
[`examples/python/mixed_boundary_elasticity.py`](../../examples/python/mixed_boundary_elasticity.py).

## Fixed-mesh monolithic FSI result

Define adjacent fluid and solid regions, generate their mesh, and compile
the FSI component. Select the elements for each domain:

```python
model = eqiora.compile(
    path=files(eqiora).joinpath("examples", "fixed-reference-fsi.eqi"),
    geometry=geometry,
    entry="FixedReferenceFsi2d",
    bindings={
        **parameters,
        **{region: geometry.selection(region) for region in ("fluid", "solid")},
        **{
            f"{region}_{side}": (
                geometry.selection(f"{region}_{side}"), geometry.selection(region)
            )
            for region in ("fluid", "solid")
            for side in ("x_lower", "x_upper", "y_lower", "y_upper")
        },
    },
)
plan = eqiora.resolve(
    model,
    mesh=mesh,
    spatial=(
        eqiora.fem.MiniP1().at(model.domain("fluid")),
        eqiora.fem.P1().at(model.domain("solid")),
    ),
    temporal=eqiora.time.BackwardEuler(step_s=0.05),
    solve=eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.MinimumResidual,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Reproducible,
        provider=eqiora.solve.SolverProvider.reference(),
        relative_tolerance=1.0e-11,
        absolute_tolerance=1.0e-13,
        maximum_iterations=20_000,
    ),
    scaling=None,
)
state = eqiora.State.initial(
    plan,
    time_s=0.0,
    fields=(
        eqiora.InitialField(model.field("definition.fluid_velocity"), vertex_values=..., cell_values=...),
        eqiora.InitialField(model.field("definition.fluid_pressure"), vertex_values=...),
        eqiora.InitialField(model.field("definition.solid_velocity"), vertex_values=...),
        eqiora.InitialField(model.field("definition.solid_displacement"), vertex_values=...),
    ),
)
result = eqiora.run(plan, state=state, steps=2, output_steps=(1, 2))
```

This FSI model requires MINI/P1 elements for the fluid and P1 elements for
the solid, with Backward Euler time stepping and a MINRES solve.
`scaling=None` chooses coupled scales automatically; supply a complete
`IncompressibleScaling` value to choose them manually.

Initial coefficients are immutable, exact-Field assignments in coherent SI.
They must be complete and association-correct; pressure has no auxiliary
zero-mean restriction in this fixed-reference formulation. A compatible State
can restart a freshly resolved Plan even when solve or scaling policies differ,
while a foreign Model, Geometry, field, or state space is rejected.

The complete runnable workflow is
[`examples/python/fixed_reference_fsi.py`](../../examples/python/fixed_reference_fsi.py).
It uses a fixed-reference 2D affine-triangle formulation: the mesh stays fixed
while fluid and solid variables are solved together.

## Conserving connections

Scalar conserving connections use nominal physical-domain identity:

```python
voltage = eqiora.Dimension(mass=1, length=2, time=-3, current=-1)
current = eqiora.Dimension(current=1)
electrical = eqiora.PhysicalDomain(
    "electrical",
    across_name="voltage",
    across_type=eqiora.ValueType.real(voltage),
    through_name="current",
    through_type=eqiora.ValueType.real(current),
)
left = eqiora.ConservingPort("left", domain=electrical)
right = eqiora.ConservingPort("right", domain=electrical)
component = eqiora.Relation(
    "component",
    equations=((eqiora.across(left), 0), (eqiora.through(right), 0)),
)
physical_model = eqiora.compile(source=eqiora.Module(
    "physical_pair",
    electrical,
    left,
    right,
    component,
    eqiora.connect(left, right),
))
```

The quantity names become the source members `left.voltage` and `right.current`.
The native `eqiora.across` and `eqiora.through` functions select physical roles,
so renaming a quantity does not change the connection law. Equal names and
dimensions do not make separately constructed domains interchangeable.

## Spatial declarations

Domain, boundary, Field support, and Relation support are
immutable handles. Use the same domain handles when declaring fields and
relations; support is not inferred from names.

```python
interval = eqiora.Domain.box("interval", (0.0, 1.0))
lower = interval.boundary(
    "lower",
    axis=0,
    side=eqiora.BoundarySide.Lower,
)
upper = interval.boundary(
    "upper",
    axis=0,
    side=eqiora.BoundarySide.Upper,
)
potential = eqiora.Field(
    "potential",
    role=eqiora.FieldRole.Variable,
    domain=interval,
)
source = eqiora.Parameter(
    "source",
    value=1.0,
    value_type=eqiora.ValueType.real(eqiora.Dimension(length=-2)),
)
model = eqiora.compile(source=eqiora.Module(
    "poisson",
    interval,
    lower,
    upper,
    potential,
    source,
    eqiora.Relation(
        "balance",
        domain=interval,
        equations=((-eqiora.div(eqiora.grad(potential)) - source, 0),),
    ),
    eqiora.Relation(
        "lower_value",
        domain=lower,
        equations=((eqiora.trace(potential), 0),),
    ),
    eqiora.Relation(
        "upper_value",
        domain=upper,
        equations=((eqiora.trace(potential), 0),),
    ),
))
```

Compilation checks the shape, frame, dimensions, and support of `grad`, `div`,
`trace`, and the resulting relations.

## Typed spatial Plan

Spatial execution uses the same root lifecycle as the examples above: author
one concrete Geometry, resolve a typed meshing provider, compile an
equations-only component with that Geometry, and call
`eqiora.resolve(model, mesh=..., spatial=..., solve=...)`. Run the returned
`Plan` with `eqiora.run(plan)` or `eqiora.submit(plan)`.

The same resolved Plan can be moved as one exact local artifact without its
producer process:

```python
plan.write("case.eqplan")
portable = eqiora.Plan.read("case.eqplan")
result = eqiora.run(portable)
```

`.eqplan` contains exactly `plan.to_bytes()`. Reading re-resolves the Plan
against the locally available providers and rejects unknown,
noncanonical, oversized, non-regular, symlinked, or wrongly suffixed inputs.
Use it to move one exact Plan between local processes.

Save results and spatial trajectories to files. Reopening requires the Plan
used to produce them:

```python
result.write("run.eqresult")
reopened = eqiora.Result.read(portable, "run.eqresult")

trajectory = reopened.trajectory
trajectory.write("run.eqtrajectory")
same_trajectory = eqiora.trajectory.Trajectory.read(
    portable, "run.eqtrajectory"
)
```

The files contain exactly `result.to_bytes()` and `trajectory.to_bytes()`.
For a dynamic Result, the Result remains the single complete root and owns its
Trajectory; the separate Trajectory file is an optional spatial projection,
not a second occurrence record. Process-local Runs, restart checkpoints,
archives, and cloud transport remain outside this boundary.

## Exact revisions and compiled Model files

`Model` owns one immutable canonical artifact. Previewing an edit never
mutates it, and committing a valid edit returns a child:

```python
base = model
edit = base.preview_value_edit("source", 2.0)
child = base.commit(edit)

assert base.revision != child.revision
assert base.digest == edit.base_digest
```

Commit checks the edit's exact base digest and graph revision atomically.
Stale or foreign plans produce no partial child. Ordinary authoring, edits,
and byte/file decoding all use the single current artifact contract:

```python
restored = eqiora.Model.from_bytes(child.to_bytes())
assert restored == child

child.write("child.eqmodel")
same = eqiora.Model.read("child.eqmodel")
assert same.revision == child.revision
```

The canonical bytes still expose the persisted
`eqiora.model-envelope/v48` schema, but callers do not select that suffix.
`.eqi` remains source text; `.eqmodel` is the canonical compiled Model artifact.
Only the current schema is accepted; decoding never sniffs, retries, or silently
migrates an older artifact.

Independent definitions allocate fresh canonical occurrence identities, so
exact equality and digest equality are intentionally stronger than structural
comparison:

```python
source_model = eqiora.compile(
    source="""
    model decay(parameter rate: 1 / s = 1) {
      state x: 1;
      initial { x = 1; }
      relation flow {
        derivative(x) + rate * x = 0;
      }
    }
    """
)
x = eqiora.Field("x", role=eqiora.FieldRole.State)
rate = eqiora.Parameter(
    "rate",
    value=1.0,
    value_type=eqiora.ValueType.real(eqiora.Dimension(time=-1)),
)
native_model = eqiora.compile(source=eqiora.Module(
    "decay",
    x,
    rate,
    eqiora.Initial((x, 1)),
    eqiora.Relation(
        "flow",
        equations=((eqiora.derivative(x) + rate * x, 0),),
    ),
))

assert source_model != native_model
assert source_model.digest != native_model.digest
assert source_model.structurally_equivalent(native_model)
assert (
    source_model.structural_fingerprint
    == native_model.structural_fingerprint
)
```

The structural fingerprint omits names, formatting, source spans, occurrence
IDs, package provenance, and artifact codec. It is comparison evidence, not a
replacement for exact identity in execution, replay, provenance, or edits.


Quantity inputs use the `eqiora.units` catalog. For example,
`q.quantity(Decimal("998.2"), u.kg / u.m**3)` preserves the exact decimal input
until compiler normalization. Import `Decimal` from Python's `decimal` module.
Integer inputs retain their decimal digits. Float inputs use Python's shortest
round-trip decimal spelling; this is a source-authoring policy, not a claim of
exact binary-ratio rescaling. Native numerical inputs remain binary64 values in
coherent SI. Quantity literal spellings are limited to 256 bytes.


## Finite inequalities and complementarity

`eqiora.lang.inequality(left, right)` declares `left >= right`. A Boolean
comparison remains an expression and cannot stand in for a Relation condition.
`eqiora.lang.complementarity` takes two explicit nonnegativity predicates. Its
operands can have different dimensions, such as gap in metres and force in newtons:

```python
import eqiora

q = eqiora.lang
length = eqiora.Dimension(length=1)
force_unit = eqiora.Dimension(mass=1, length=1, time=-2)
source = eqiora.Module("main")
component = source.model("Contact")
gap = component.field("gap", value_type=eqiora.ValueType.real(length),
                      role=eqiora.FieldRole.Variable)
force = component.field("force", value_type=eqiora.ValueType.real(force_unit),
                        role=eqiora.FieldRole.Variable)
component.relation(
    "contact",
    q.equation(q.quantity(2, eqiora.units.N / eqiora.units.m) * gap - force,
               q.quantity(-6, eqiora.units.N)),
    q.complementarity(q.greater_equal(gap, q.quantity(0, eqiora.units.m)),
                      q.greater_equal(force, q.quantity(0, eqiora.units.N))),
)
model = eqiora.compile(source=source, entry="Contact")
reference = model.constraint("contact", 1)
enforcement = eqiora.solve.ActiveSet(
    tolerances=(eqiora.solve.ConstraintTolerance.complementarity(
        reference, 1e-10, length, 1e-10, force_unit,
    ),),
    max_active_sets=2,
)
linear = eqiora.solve.Linear(
    relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=8,
    algorithm=eqiora.solve.LinearSolver.SparseLu,
    preconditioner=eqiora.solve.Preconditioner.Identity,
    reduction=eqiora.solve.Reduction.Fast,
    provider=eqiora.solve.SolverProvider.faer(),
)
plan = eqiora.resolve(model, solve=linear, enforcement=enforcement)
result = eqiora.run(plan, state=eqiora.State.initial(plan))
receipt = result.constraints[0]
assert receipt.activity == "active"
assert abs(receipt.left_value) <= receipt.left_tolerance
assert abs(receipt.right_value - 6) <= receipt.right_tolerance
```

Here `2 gap - force = -6`, so the admissible solution is zero gap and 6 N force.
Changing the load to +6 N gives gap 3 m and zero force, the inactive branch.
Both operands must be nonnegative within their respective declared tolerances;
at least one must be zero within its own tolerance. “Biactive” means both meet
that zero test. These are bounded numerical acceptance statements, not a claim
of exact satisfaction. Each inequality also requires its own dimensioned
`ConstraintTolerance.inequality(reference, value, dimension)`.

The Model retains the mathematical conditions. The Plan separately retains the
explicit active-set algorithm, operand tolerances, search bound and linear solver
provider. Results expose original operand values, dimensions and activity.
Plan, State and Result replay preserve their exact ownership. Result replay
reevaluates the original expressions and selected branch rather than trusting a
persisted mask, solver success or claimed residual target.

This profile admits static affine real scalar Fields without spatial support,
with at most 256 unknowns, 16 complementarity pairs and 65,536 candidate sets.
The number of equalities plus complementarity pairs must equal the number of
unknowns. Every operand must be affine, including inactive branches. Mixed
conserving Ports and Fields, spatial or dynamic enforcement, nonlinear contact,
friction, inclusions and regularized penalties are unsupported. Missing or
wrong-unit tolerances, omitted nonnegativity and reversed nonnegativity signs
are errors. Equality-only mathematical rendering rejects constrained Relations.

## Mathematical coordinate factors

Bind a source interval with explicit coherent-SI bounds and coordinate dimensions. The source
owns nominal identity and ordered product structure; these inputs require no Geometry or Mesh:

```python
model = eqiora.compile(
    source="""model Distribution(support position: interval(m), support velocity: interval(m/s)) {
      support phase: product(position, velocity);
      variable f: s/m^2 on phase;
      relation retain on phase { f = 0[s/m^2]; }
      observable count: 1 = integral(f, measure(phase));
    }""",
    entry="Distribution",
    bindings={
        "position": eqiora.CoordinateInterval(0, 1, dimension=eqiora.Dimension(length=1)),
        "velocity": eqiora.CoordinateInterval(-2, 2, dimension=eqiora.Dimension(length=1, time=-1)),
    },
)
```

The measure is `m * (m/s)`, so integrating a density in `s/m^2` is dimensionless.
Model replay retains exact factor order and units. Bounds are finite and strictly increasing;
substituting position bounds for velocity bounds rejects. To realize one prescribed polynomial
scalar Field, use the same exact Domain to construct a coordinate grid:

```python
mesh = eqiora.meshing.Mesh.coordinate_factors(model, model.domain("phase"), [2, 4])
plan = eqiora.resolve(
    model,
    mesh=mesh,
    spatial=eqiora.fvm.CellCentered(),
    solve=eqiora.solve.Linear(
        relative_tolerance=1e-12,
        absolute_tolerance=1e-12,
        maximum_iterations=20,
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast,
        provider=eqiora.solve.SolverProvider.faer(),
    ),
)
result = eqiora.run(plan)
count = result.observe(model.observable("count"), quadrature_points=2)
assert abs(count.value) < 1e-12
```

The static profile admits one equality `f = prescribed_polynomial`, with degree at most
three per exact factor and one to three bounded interval factors. It solves cell-integrated
equalities using a constant Field basis and two-point tensor Gauss quadrature. Field values
are cell averages, not point samples. Partial Result integrals split at grid faces; an
interior face belongs to the upper cell and the last endpoint belongs to the final cell.
`Mesh`, `Plan` and `Result` each support their ordinary byte replay. This Mesh has no physical
Geometry or Geometry correspondence; the corresponding digest properties are `None`.
The generic `CellCentered` request exposes its cell-constant `space`; Field roles and
numerical integration follow the resolved Plan. Physical Mesh viewing, general product
PDEs, nonpolynomial Field projection, transient radial diffusion and Field derivatives remain outside
this profile. The [registered derivation](../../verify/language/factor-integrals/README.md)
separates moment projection error from quadrature error.

A Model or Component can name physical coordinates for `q.partial`:

```python
module = eqiora.Module("main")
component = module.component("Derivatives")
body = component.volume("body", dimensions=2)
length = eqiora.ValueType.real(dimension=eqiora.Dimension(length=1))
x = component.coordinate("x", value_type=length, on=body, factor=body, axis=0)
y = component.coordinate("y", value_type=length, on=body, factor=body, axis=1)
component.relation("slope", q.equation(q.partial(x*y, wrt=x), y), on=body)
```

Here `q` is `eqiora.lang`. The exact support and factor handles must belong to the same
Component. Axes are zero-based within the factor; only a scalar factor permits omission.
The compiler checks coordinate units and factor membership. Source also admits independently
dimensioned position/velocity factors. Polynomial partials use the shared calculus evaluator;
unknown continuous scalar Field partials retain their request until an admitted Cartesian Q1
Result supplies basis derivatives. Higher unknown-Field partials, intrinsic boundary charts
and TPFA derivative observations reject. See the [coordinate profile](../language/coordinates.md).

## Fields on physical boundaries

Source passed to `eqiora.compile` may declare a Field directly on a Cartesian boundary,
or use a `boundary(parent=...)` support bound to an exact Geometry selection. A line density
in `kg/m` has an integral in `kg`, even when the line is embedded in 2D. Spatial Cartesian
vector components follow the ambient dimension, independently of that line measure.
See the [source boundary Field example](../language/coordinates.md#fields-on-affine-physical-boundaries).

Geometry bindings use the existing `(boundary_selection, parent_selection)` pair. A selection
from another exact Geometry revision rejects even if its name or coordinates match. This
path provides Model typing and artifact reference replay; numerical boundary-Field solving,
curved/grouped boundary Fields and physical `grad`/`div` on boundary Fields remain unsupported.


Static radial diffusion uses the same coordinate-grid factory with `model.domain("radius")`
and `eqiora.fvm.CellCentered()`. Bind the explicit equations and center/surface conditions in
[the radial model](../../verify/language/factor-integrals/models/radial-diffusion.eqi), then
observe its `average` through the ordinary Result. Two quadrature points integrate the
represented cell concentration against the spherical weight exactly. The
[registered derivation](../../verify/language/factor-integrals/README.md#static-radial-diffusion-and-particle-average)
separates its discretization error from quadrature; time evolution and pointwise
cell-constant reconstruction remain unsupported.
