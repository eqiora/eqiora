# Fixed-domain conservation Laws

A `law` retains physical flux, source and optional storage as mathematical terms. Numerical methods
consume these terms through the same Model as ordinary Relations.

```eqiora
model HeatedInterval() {
  domain body = box(0, 1);
  domain left = boundary(body, axis = 0, side = lower);
  domain right = boundary(body, axis = 0, side = upper);
  variable temperature: K on body;
  parameter conductivity: kg * m / s^3 / K = 2;
  parameter heating: kg / m / s^3 = 4;
  law heat_balance on body {
    flux -conductivity * grad(temperature);
    source heating;
  }
  relation left_temperature on left { trace(temperature) = 0; }
  relation right_temperature on right { trace(temperature) = 0; }
}
```

A Law specifies the steady balance `div(flux) = source`. Flux is the
physical outward flux; diffusion therefore uses `-conductivity * grad(temperature)`.
The compiler requires exactly one flux and one source, with compatible dimensions
and the Law's exact volume support. Write `source 0;` for no production.
Boundary and interface conditions remain ordinary Relations on their exact supports.

Python Module authoring uses `component.law("heat_balance", on=body,
flux=-conductivity * eqiora.lang.grad(temperature), source=heating)`. It produces the
same source AST and retains the same physical terms in Model artifacts and package
composition. Changing a physical term changes Model meaning even when an equation
could otherwise be rewritten to an equivalent residual.

Real scalar storage Laws on a fixed volume retain `storage c * u;` and the checked
accumulation `derivative(c * u)`. Constant positive capacity, complete initial data
and essential boundary data execute through the common scalar Backward Euler path.
Its numerical profile admits Q1 on Cartesian cells and P1 on planar triangles;
nonlinear capacity and moving-domain transport remain outside that execution profile.

The Cartesian Q1 path also admits an explicit first-time-derivative weak pairing:

```eqi
form weak_storage for balance {
  test w: 1 for u zero_on left, right;
  integrate(body, w * c * derivative(u))
    + integrate(body, dot(grad(w), grad(u))) = integrate(body, w * heating);
}
```

Here `balance` declares `storage c * u; flux -grad(u); source heating;`, and `u`
is a continuous scalar State with its own initial and boundary equations. The
correspondence check distinguishes the rate from the State value, retains exact
coefficient identity and dimensions, and rejects a missing or changed storage pairing.
Current authored-form v16 bytes and resolved Plan replay retain that distinction.
The focused Rust profile exercises accepted steps and State restart; it does not
claim a general transient Formulation checker or a moving-volume transport rule.

A fixed Cartesian Q1 storage Law can also retain prescribed conservative transport:

```eqi
law balance on body {
  storage c * u;
  flux -k * grad(u) + u * (bx * grad(x) + by * grad(y));
  source f;
}
```

Here `x` and `y` are declared physical coordinates on `body`, and `bx` and `by`
are prescribed scalar coefficients with the units of capacity times velocity.
They may vary in space. The weak transport contribution is
`-integrate(body, dot(grad(w), u * (bx * grad(x) + by * grad(y))))`.
Keeping the complete flux retains the divergence of its coefficient as well as
transport of `u`; replacing it by a velocity-times-gradient expression would lose
that density term. Source and authored weak terms must preserve the same coordinate
axes and coefficients. The common Region solver handles the resulting nonsymmetric
matrix with the selected linear solver.

Focused Rust tests cover interval and box Q1 execution, fixed essential boundaries,
constant positive capacity, Plan/State replay and restart. The profile remains on a
fixed Mesh; it does not infer a material velocity, mesh motion or moving-volume
Jacobian from coefficient units. Unknown-dependent transport, transport interfaces,
natural transport boundaries and stabilization are not admitted by this profile.

A steady real scalar Law on an exact one-dimensional Geometry support can also
carry an authored mathematical conservation form:

```eqi
form conservative for balance {
  interval segment(a, b) on body;
  outward_flux(segment, a, -k * grad(T))
    + outward_flux(segment, b, -k * grad(T)) = integrate(segment, s);
}
```

Here `balance` retains `flux -k * grad(T); source s;`, and `k` and `s` are
explicit Component parameter bindings. The binder quantifies every ordered
interval `(a,b)` contained in `body`. Its endpoints are mathematical variables;
they are not mesh cells or the exterior endpoints of the parent support.
`outward_flux` applies normal `-1` at `a` and `+1` at `b`.

Fresh compilation checks dimensions and exact physical terms. The separate
projection checker reads the live Law, Field support and authenticated Geometry
again, without generating an expected form. The conditional implication assumes
a fixed one-dimensional domain and classical divergence and boundary traces;
it does not establish these regularity assumptions or the reverse implication.

Native AST construction, source formatting, Python inspection and mathematical
rendering retain the interval. The projection uses one tagged v6 wire for plural Relation-owned equations,
weak-test inventories and interval binders; displaced projection decoders are removed. Forms
remain outside Model identity. The ordinary single-region scalar TPFA path admits
this form on a fixed 1D Geometry with the existing positive diffusion and supported boundary conditions.
Automatic and exact integral-conservative requests derive mathematical content
and pass the same checker; only authored requests retain authored source identity.
Plan and Result replay preserve this distinction and rerun admission. Storage,
multidimensional, mixed and complex forms remain unavailable.

An interval form can retain an explicit constant scalar gauge after its binder:

```eqi
gauge potential {
  reference integrate(body, potential) = 0;
  compatibility integrate(body, source_value) + lower_load + upper_load = 0;
}
```

The reference and compatibility equalities are separately dimension-checked and
bound to the exact trial Field and parent support. Here the endpoint loads denote
`n k grad(potential)`, the negative of physical outward flux, so their sum enters
compatibility with the source integral using a plus sign. The numerical admission
checker matches this condition to the original source and both natural boundary
relations; a written condition alone does not establish that their values balance.
The ordinary fixed-1D TPFA path admits the explicit zero-integral reference with
two natural endpoints. Plan resolution retains the existing `ZeroIntegral`
constraint and requires a solver for the resulting symmetric-indefinite bordered
system. Before solving, execution checks the constant vector against the actual
original matrix and checks the assembled load balance without projection. Result
acceptance and replay reassemble the original equations and reference, retaining
separate original-equation, compatibility and gauge residuals and the numerical
multiplier. The scalar Field contains no multiplier entry. Python exposes these
as `original_residual_norm`, `compatibility_residual`,
`gauge_residual` and `gauge_multiplier`; the solve report describes
the bordered system. Constrained differentiation is not admitted.

The native P1 and TPFA paths also support an explicit spatial mean. General
nullspaces, automatic mode discovery and finite floating-network Formulations
remain outside this interval profile.

A bounded real steady 2D Stokes system may retain two weak equations in one form.
`form weak for momentum, continuity` pairs each equation with its declared Relation;
velocity and pressure have distinct named tests. The velocity test carries the exact
complete homogeneous essential boundary, while the pressure test has no boundary
restriction. `frobenius` contracts equal real rank-two shapes and remains distinct
from vector `dot`. Compilation through the public Model API checks every produced
term against the live Stokes certificate before exposing Python inspection or
rendering. This profile supports inspection only; authored mixed numerical
execution and stability claims remain unavailable.


A finite global scalar system can retain an explicit constant-shift reference without
Geometry or a Mesh. Each equation corresponds to the Field in the same position
of the finite binder:

```eqi
form floating for first, second {
  finite voltage(v1, v2);
  gauge voltage {
    reference v1 = offset;
    compatibility i1 + i2 = 0;
  }
  g * (v1 - v2) = i1;
  g * (v2 - v1) = i2;
}
```

Here the original global Relations `first` and `second` contain these same
operand pairs; `g`, `i1`, `i2` and `offset` are explicit Component parameter
bindings. Trial Fields must be invariant real scalars with one physical dimension.
The first finite execution profile requires homogeneous affine left operands,
Field-independent right operands and an actual symmetric matrix with the declared
uniform shift mode. Compatibility must state the sum of original loads equals
zero; execution also checks their numerical balance independently of solver tolerance.
The reference fixes one listed Field to a literal or exact scalar Parameter with
its dimension. SparseLU with Identity/Fast controls solves the bordered system.
No original equation is removed, pinned or shifted.

The authored-form v6 projection retains the ordered finite coordinates and the two
conditions separately from Model identity. Plan replay checks these operands and
identities against the live Model. Result v9 rechecks the original equations,
reference and multiplier. Its `original_residual_norm`, `compatibility_residual`,
`gauge_residual` and `gauge_multiplier` properties apply to both finite and spatial
gauges; displaced scalar-prefixed properties are removed. General left/right
nullspaces, multiple independent modes, non-affine gauges, complex coordinates,
constrained differentiation and distributed/device execution remain outside this profile.
