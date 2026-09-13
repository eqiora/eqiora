# Modeling and realization

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

`Field.value_type` holds its mathematical scalar domain, physical dimension and
component roles. `FieldRole.Variable` declares an algebraic unknown;
`FieldRole.State` declares evolution or history independently of support.
Omitted types are dimensionless real scalars. `Initial` supplies simultaneous
ordered left/right equation pairs for fresh initialization; the Field stores no initial literal.
Fresh scalar ODE and supported index-one DAE initialization checks the initial and
regular equations together. Missing state data, contradictory constraints, or an
unsupported initialization profile reject; this is not a general high-index DAE
solver. Restart uses an accepted State/history without reapplying these equations.

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
`vector(scalar, extent)` and `tensor(scalar, *extents)` introduce spatial axes;
`array(element, extent)` adds a channel axis without changing the element's frame.
Equal component counts do not make these types interchangeable. Spatial extents
must match the Field's exact Domain. Complex execution is still under development.

Dimensions accept exact rational exponents, for example
`eqiora.Dimension(length=Fraction(-1, 2))` with `Fraction` imported from `fractions`.
Initial equations follow expression typing: nonzero dimensioned constants need
an explicit compatible quantity. A scalar is not broadcast into a vector, tensor,
or array initial state. Distributed execution requires explicit initial data.

The same `value_type=` objects apply to `Parameter`,
`eqiora.lang.Component.field`, and `eqiora.lang.Component.parameter`.
`ValueType.to_eqi()` emits the canonical type through the Rust formatter.

Native `Field`, `Parameter` and `Expression` handles support immutable channel slices such as
`values[0:2]`. Supply both integer bounds and no step; negative bounds, Boolean bounds,
clamping and empty slices are not accepted. The same `values[0:2]` syntax works in `.eqi`.
Source declarations can use `array<integer, n>` with an exact static size Parameter, including
`eqiora.compile(source=source, entry="Channels", bindings={"n": 3, "data": (2, 3, 5)})`.
Changes to a compiled size or selector require recompilation; ordinary coefficient edits
continue to use `preview_value_edit` and `commit`.

Parameters accept real or complex scalars and nested channel sequences matching the declared
shape. Inspection returns immutable nested tuples with every real/imaginary component:

```python
coefficients = eqiora.Parameter(
    "coefficients",
    value_type=eqiora.ValueType.array(eqiora.ValueType.complex(), 2),
    value=[1 + 2j, 3 - 4j],
)
assert coefficients.value == (1 + 2j, 3 - 4j)
selected = coefficients[1]
```

Nonzero spatial coefficients carry an explicit frame context from an existing
Domain. Their values remain uniform Parameters:

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

Include the frame Domain in `Module`; foreign or omitted declarations reject.
The Domain supplies the model-global Cartesian frame and ambient dimension, not
Parameter support. Values retain real/imaginary components and axis order through
inspection, edits, and replay. A channel array of tensors remains distinct from one
spatial tensor. Python Module authoring uses the same constructor as
`q.tensor_value(frame=body, components=((2, 3), (5, 7)))`, where `body` is an exact
`Component.volume` or `Component.boundary` handle from that Component. The resulting
expression can supply a Parameter default through `Component.set_default`. Constructor
components must be closed scalar expressions; referencing a named model value, including
a Parameter alias, inside the constructor rejects.

Indices are static exact nonnegative integers. Module expressions may use a Parameter
with an exact compile-time value; changing a Parameter used by a compiled index, slice or
extent requires recompilation.
Typed value edits preserve the complete declared type and all components through replay.
This authoring support does not establish a complex numerical solver.

Declare `value_type=eqiora.ValueType.integer()` to retain Python integers exactly,
including values above `2**53`. Values must be signed 64-bit integers; booleans,
floating-point values and overflow reject. Without an explicit integer type,
ordinary numeric defaults keep their real interpretation. `ParameterRef.value`
returns the exact typed value, including after edits and Model replay.

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

`ValueType.coordinates(species)` holds signed integer components in the same
ordered basis; `counts(species)` requires nonnegative components. Equal labels in
another `FiniteSpace` do not establish the same type. `IndexSet("Rows", extent=3)`
and `ValueType.index(rows)` similarly retain a distinct nominal identity and admit
only ordinals from zero through two. Include each native declaration in
`Module`. Nominal types need their declaration's lexical scope for source
rendering, so their standalone `to_eqi()` rejects.

Python Module registers spaces with `source.space(...)` and constant-sized sets
with `component.index_set(..., extent=3)`. Its `counts`, `coordinates`, and `index`
constructors require handles from the owning Module or Component. The closed
`eqiora.lang.quotient`, `remainder`, `to_real`, `to_integer`, and `ordinal`
expressions use the shared compiler's explicit conversion and arithmetic rules.
Products, dual spaces, general maps, and dynamic indexing remain
unsupported for these discrete fields.

Finite scalar reductions bind one symbolic index through `Component.sum`, `Component.product`,
`Component.min` or `Component.max`:

```python
rows = component.index_set("Rows", extent=3)
total = component.sum(lambda i: (q.ordinal(i) + 1) * (q.ordinal(i) + 1), over=rows)
component.let_alias("total", total)
```

Here `q` is `eqiora.lang`. The callback runs once to author the body; the compiler expands
its three terms and obtains 14. A nested reduction needs a distinct `name`, such as
`name="j"`. Binders cannot escape their callback or capture another declaration, and the
set and captured declarations must belong to the same Component. An array expression accepts
`values[q.ordinal(i)]` within this scope. General runtime indexing remains unsupported.
The [finite reduction rules](../language/numeric-catalog.md#finite-scalar-reductions) define
ordering, scalar types, product units, expansion bounds and unsupported initializer contexts.
`min` and `max` preserve ordinary integer or real scalar types and dimensions, evaluate all
terms and retain the first term on a tie; they do not supply generic conditional expressions.

A numeric Parameter default uses the declared dimension's coherent unit.
For example, `parameter rate: 1 / s = 1;` gives the same value as
`parameter rate: 1 / s = 1[1 / s];`. Explicit input units still express compatible
conversions. This context applies only to numeric Parameter defaults;
nonzero literals in general expressions do not silently acquire units.

A native Relation receives ordered `equations=((left, right), ...)` pairs.
Use `(residual, 0)` for a numerical residual equation. Named `equal`, `not_equal`,
`less`, `less_equal`, `greater`, and `greater_equal` functions produce predicates;
`logical_not`, `logical_and`, and `logical_or` compose them. Symbolic Python equality,
ordering, and truth testing reject. Declarations and expressions
are frozen; validation and artifact creation happen atomically in Rust.

## Derived observables

An `observable` retains a typed expression in Model meaning without adding a
Field unknown or solving equation. Finite values and spatial integrals use the
same declaration:

```eqi
observable output: V = lower.positive.voltage - ground.terminal.voltage;
observable energy: J = integral(capacity * (temperature - reference), measure(body));
observable outward_heat: W = integral(normal(-conductivity * grad(temperature)), measure(wall));
```

`integral(expression, measure(domain))` infers volume or surface measure from the exact Domain kind and retains the exact
parent boundary. The output dimension includes that measure. Boundary Field
values require `trace(field)`; oriented flux uses the existing outward `normal`
operator. A same-sized foreign Domain does not substitute for the declared one.
These reductions currently occur only at the root of an Observable expression.

Native Python uses `eqiora.Observable(name, value_type=..., expression=...)`,
`eqiora.integral(expression, eqiora.measure(domain))`. Include each declaration in its
owning `Module`. The source graph checks both instantiated and unused Component
bodies. A private Component Observable remains inspectable by exact qualified
identity and does not become an exported Port or a value symbol in equations.

Evaluation belongs to an accepted Result with the exact Model meaning. The
initial spatial execution profile covers real scalar Q1 Fields on an authenticated
Cartesian mesh, explicit traces, scalar expressions and oriented normal gradients.
Spatial evaluation requires an explicit numerical quadrature rule; it never uses
rendered values or output cadence as an integration authority. Its State JVP uses
the same basis and quadrature for Field and normal-gradient variations, holding
Parameters and Geometry fixed. Reduced-solve and geometry-shape sensitivities
require separate admission.

Select the declaration through its exact Model handle, then evaluate it on the
accepted Result. For a Component instance named `definition`:

```python
energy = model.observable("definition.energy")
observation = result.observe(energy, quadrature_points=2)
value = observation.value
lineage = observation.result_identity
```

`quadrature_points` selects Gauss–Legendre points per axis; a point boundary
requires `1`. Finite values omit this argument. A State direction is created with
`result.observable_state_tangent({field: (dimension, coefficients)})` and applied
with `result.observe_state_jvp(energy, direction, quadrature_points=2)`. Its
`evaluation_kind` is `"state-jvp"`, and a different Result cannot reuse that direction.

### Smooth trajectory functionals

For a scalar ODE Observable, evaluate the accepted terminal state or integrate over
its complete Run interval with an explicit numerical rule:

```python
sample = model.observable("sample")
terminal = result.observe_terminal(sample)
integral = result.observe_time_integral(
    sample, quadrature=eqiora.time.TimeFunctionalQuadrature.AcceptedStepSimpson
)
```

The integral uses each accepted solver step's native start, midpoint, and end
values. Requested output times do not select the integration samples. This is
Simpson quadrature over retained native history, not an exact analytic integral.
`value_type` includes the Observable dimension multiplied by seconds;
`interval_s`, `quadrature`, and exact Result/Trajectory identities retain the
numerical scope. Terminal evaluation uses the fixed terminal time even when it
was omitted from requested outputs. Result persistence retains the same history.

The current profile admits smooth real scalar ODE expressions only. Missing
history, foreign Model references, and event/reset histories are rejected.
Spatial-time composition, interval clipping, moving endpoints, and trajectory
derivatives remain outside this profile.

## Author Eqiora Language source

`eqiora.Module` owns the shared equations-language graph for Python authoring:

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

Module Relations receive an immutable `q.equation(left, right)`, including an
explicit zero right operand for a residual equation. They emit `left = right;`;
Python `==` does not construct an equation. The Module owns exact supports and
expressions, rejects foreign handles and resource-limit violations, and freezes
on its first emission or compile.
It emits ordinary readable UTF-8 `.eqi`; `doc=` values become attached `///`
documentation. A blank paragraph is emitted as an empty `///` line, keeping the
block attached to its declaration. Documentation is limited to 16,384 UTF-8 bytes.
`write_eqi(path)` uses same-directory staging and atomic replacement, so an I/O
failure does not publish a partly written source file.

A Module can contain multiple Components within its existing declaration bound.
Use `parent.instance(...)` to bind a child's requirements explicitly, and select the
entry with `eqiora.compile(source=source, entry="Parent", ...)` when the source
contains multiple public Components. Instance bindings and returned outputs use
their declared names as string keys.

`component.let_alias(name, expression)` declares a private immutable expression alias.
Its type and spatial support are inferred; `value_type=` asserts the inferred type,
and `on=` asserts the exact inferred Support owned by this Component.
Aliases can use the component's Parameters, fields, and other aliases. For example,
`heat_flux = component.let_alias("heat_flux", coefficient * q.grad(potential))`
can appear in a volume relation as `q.div(heat_flux)`. A Parameter-only alias can also
supply a nested-instance Parameter argument; a field-dependent alias cannot.

Aliases do not become required parameters, independent edit targets, unknowns, or equations.
Each occurrence retains the original dependencies, so Parameter edits and differentiation
pass through the expression. Expressions must have an intrinsically inferable spatial support:
keep context-dependent `coordinate`, `trace`, and `normal` expressions in their relations.
Writing `on=` cannot give a constant spatial support or select an unspecified boundary.
An equal-shaped, separately declared Support is still a different nominal support.
`at=clock` asserts that the expression's runtime dependencies belong to that exact
Component-owned Clock. Equal periods do not make different Clocks interchangeable.
Static expressions and mixtures of continuous and clocked dependencies cannot assert a
single clock. State operators retain their exact-clock and initialization rules through
aliases; reading a current state through an alias adds no clock restriction.

Create nominal periodic clocks with exact seconds, using an integer or `fractions.Fraction`:

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

The optional `phase_s` defaults to zero. Initial equations are simultaneous, and the
clock's first tick follows initialization. Clock handles belong to their declaring
Component; foreign handles are rejected before changing a declaration. `q.pre` and
`q.next` use the same Rust state-role and use-context checks as emitted source.
Check the spatial and time methods supported by your execution backend before running the model.

Module values do not type-check or lower equations in Python. Direct compile
consumes the Rust Module graph without formatting or reparsing text, then uses
the same type checker, lowerer and Geometry/support binder as parsed source.
`to_eqi()` explicitly formats the module; `eqiora.Module.parse("main", text)`
reconstructs it, with imports attached explicitly through `import_module`.
Direct and emitted-file compilation with identical bindings can be compared by
`structural_fingerprint`, not exact artifact identity. Prose changes affect source
bytes and package source-bundle identity, not the physical Model. Constructed
declarations report graph paths; parsed declarations retain real source spans.
`q.math.pi` is an immutable expression that emits exactly
`math.pi`; `q.math.sin(expression)` emits the matching scalar
operation. Use these expressions with fields and parameters from your Module. The top-level `eqiora.lang` vocabulary is reserved for equation
structure such as `q.grad` and `q.div`, while scalar functions and constants
live under `q.math`. They are not Python numerical operations, and the native
compiler remains the authority for their typing and value semantics.

A Module can emit the constant property declarations used by
an exact Model Package:

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
that each release implements the required nominal contract.

Contracts and releases authored in the same Module can compile locally through
`eqiora.compile(source=source, entry=..., bindings=...)`. The shared Rust compiler
checks each exact release or composition member against its nominal contract;
a same-spelled handle from another Module is rejected before emission.

To retain exact package provenance, emit the `.eqi` into a Model Package, lock
it, and use `compile_package`. That route exposes the existing immutable
`property_bindings` inspection. Local compilation does not retain package provenance.

### Exact analytic and table properties

Contracts can declare ordered named real inputs and `first_partials` or
`first_open_intervals` derivative profiles. Use `contract.input(name)` in an analytic
release and apply the Component requirement with named arguments. Imported contracts and
releases retain the exact public declaration identity through `ModuleRef.property_contract`
and `ModuleRef.property_release`.

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

## Resolve and lock a local package project

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
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python '.[gmsh,matplotlib]'
uv run --no-project --python .venv/bin/python examples/python/exact_cylinder_stokes.py \
  --pressure-png exact-cylinder-pressure.png
```

These commands build the checked-out source with its matching example. The
equivalent composition API is:

```python
import eqiora.matplotlib as eqplot

pressure = result.snapshots[0]
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

displacement = result.output(plan.capability.displacement)
mesh = displacement.mesh
evidence = eqiora.solid.linear_elasticity_evidence(result)
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
    field=plan.capability.displacement,
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
        eqiora.InitialField(model.field("fluid_velocity"), vertex_values=..., cell_values=...),
        eqiora.InitialField(model.field("fluid_pressure"), vertex_values=...),
        eqiora.InitialField(model.field("solid_velocity"), vertex_values=...),
        eqiora.InitialField(model.field("solid_displacement"), vertex_values=...),
    ),
)
result = eqiora.run(plan, state=state, steps=2, output_steps=(1, 2))
evidence = eqiora.fsi.evidence(result)
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
`eqiora.model-envelope/v26` schema, but callers do not select that suffix.
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
