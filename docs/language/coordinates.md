# Coordinates, measures, and local evaluation

This section of the [target grammar](core.md) covers exact coordinate factors and the
foundation audit's position/velocity derivative specimen. It specifies mathematical nodes;
their current numerical admission is not established by the examples.

The current implementation admits abstract dimensioned interval slots and owned products in
Model and Component source, including nested products, physical Cartesian region factors, and
whole-product Observable measures.
The source support owns factor identity; native `StaticBindingValue::CoordinateInterval` and
Python `CoordinateInterval` supply checked finite bounds. Model/Transaction v33 and structural
fingerprint v28 retain these factors. No ambient physical frame or numerical realization is
inferred. Coordinate binders/partials, factor-wise integration, non-Cartesian product factors,
and curved embedded-field extensions remain separate work.

## Exact factors and coordinate bindings

`support position: interval(m)` requires a bounded one-dimensional position factor.
`support velocity: interval(m / s)` requires an independently identified velocity factor.
An interval retains its coordinate unit, finite bounds, and canonical increasing-coordinate
measure. Physical position bindings retain their exact Geometry relationship; a velocity
interval is mathematical data and requires no fabricated CAD object.

`support phase: product(position, velocity);` in an owning body constructs the ordered product
from those exact bound factors. It is not another external requirement and cannot choose new
factor bounds. This mathematical support is not a two-dimensional physical surface or a
tensor product of finite component spaces. Nested products retain their factor structure.

`coordinate x: m on phase from position;` names the coordinate projection of that exact factor
on the product. It introduces no unknown or equation. The dimension must match the selected
factor, which must occur uniquely in the declared support. A repeated factor requires an
explicit factor-occurrence selector before this projection can be admitted; matching by name
or unit is not a fallback. `coordinate` permits notation after the name, requires `on` and
`from`, and does not accept `at` or an initializer.

## Products containing physical position

A physical factor may be a source `domain position = box(...)` or an exact Geometry selection
bound to `support position: volume(ambient_dimension=2)`. Geometry products currently admit
canonical boxes and individual axis-aligned rectangular faces, including straight subdivided
sides. The selected region must itself be Cartesian; its bounding box alone is insufficient.

```eqiora
model Distribution(support velocity: interval(m/s)) {
  domain position = box(0, 2, 0, 3);
  support phase: product(position, velocity);
  variable f: s/m^3 on phase;
  relation retain on phase { f = 0[s/m^3]; }
  observable count: 1 = integral(f, measure(phase));
}
```

Here `position` is one exact factor with two axes, and `velocity` has one axis. The product has
intrinsic dimension 3 and measure units `m² × (m/s)`, but no ambient three-dimensional physical
frame. A spatial `vector<...,3>` or physical `grad` on the product therefore does not follow
from that intrinsic dimension. The existing physical Domain retains its own ambient frame.
Model replay retains factor identity/order and requires the exact Geometry artifact for semantic
admission of Geometry factors. No mesh or numerical quadrature is manufactured.

Boundary, curved, nonrectangular, and grouped Geometry factors remain unsupported. This path
establishes typing and measure semantics, not coordinate partials or a product-space solver.

## Complete analytic derivative specimen

```eqiora
model CoordinatePartials(
  support position: interval(m),
  support velocity: interval(m / s),
  parameter length: m = 2 [m],
  parameter speed: m / s = 4 [m / s],
  parameter scale: s / m^2 = 1 [s / m^2]
) {
  support phase: product(position, velocity);
  coordinate x: m on phase from position;
  coordinate v: m / s on phase from velocity;

  let polynomial: s / m^2 on phase = scale * (x / length) * (v / speed);
  observable position_partial: s / m^3 on phase = partial(polynomial, wrt = x);
  observable velocity_partial: s^2 / m^3 on phase = partial(polynomial, wrt = v);
  observable mixed_partial: s^2 / m^4 on phase =
    partial(partial(polynomial, wrt = x), wrt = v);
  observable position_density: 1 / m on position = integral(polynomial, measure(velocity));
}
```

Bind position to `[0, 2 m]` and velocity to `[0, 4 m/s]`, with the stated increasing
Cartesian measures. These positive bounds make the polynomial nonnegative, but the example
is an analytic calculus probe, not an executed probability or transport model. It has no
unknown, evolution equation, initial condition, or boundary law to solve.

For general positive `L = length`, `V0 = speed`, and `F0 = scale`, the independently derived
derivatives are:

```text
f_x  = F0*v/(L*V0)                 dimension s/m^3
f_v  = F0*x/(L*V0)                 dimension s^2/m^3
f_xv = F0/(L*V0)                   dimension s^2/m^4
n(x) = integral_0^V0 f(x,v) dv
     = F0*x*V0/(2*L)               dimension 1/m
```

At `x = 1 m`, `v = 2 m/s`, the field is 0.25 s/m^2, its position partial is 0.25 s/m^3,
its velocity partial is 0.125 s^2/m^3, and the mixed partial is 0.125 s^2/m^4. The reduced
field at `x = 1 m` is 1/m. Its full position integral is 2, a dimensionless inventory.
Using `1/m` as the derivative factor for velocity would fail the type check before evaluation.

`partial(unknown_field, wrt = x)` retains the same requested coordinate operation but requires
a field representation during execution. The compiler must not substitute the analytic formula
above for an unknown distribution or claim a classical derivative from a piecewise-constant
reconstruction. Cartesian `grad` uses the declared spatial coordinate/frame order; position
and velocity partials cannot be assembled into one homogeneous spatial gradient vector.

## Integral scope and output support

`integral(expression, measure(support))` integrates only the exact factors identified by that
measure. Integrating a full product uses `measure(phase)`; integrating its velocity factor
uses `measure(velocity)` and retains the position support. Remaining factor order and identity
are unchanged. Integrating a foreign same-sized factor rejects.

An integral binds coordinate occurrences by exact factor identity inside its integrand; it
does not capture all identifiers with a matching spelling in the surrounding scope. Coordinates
projecting the integrated factor cease to be free dependencies of the result. A remaining
coordinate such as position stays free. Consistently renaming a declared coordinate and its
references preserves this mathematical projection. Replacing it with a foreign coordinate
named identically does not.

The initial profile uses fixed finite integration limits. Parameterized bounds are fixed for
the selected Model binding; time-varying endpoints and shape derivatives require their later
owner. Differentiation under an integral requires the admitted regularity and fixed-domain
conditions, not just a syntactically movable `partial` node.

Measures multiply dimensions. A velocity integral contributes `m/s`, a physical line integral
`m`, a surface integral `m^2`, and a physical volume integral `m^3`. An embedded line in 2D
still has intrinsic measure dimension `m`. Metric/Jacobian weights remain mathematical data;
quadrature points and numerical weights implement that data rather than define it.

An integral is not a normalized average. Write a denominator explicitly, and reject a zero
measure before division. For a radial spherical-symmetry profile the measure is `4*pi*r^2 dr`,
not `dr`; a constant concentration `c` has total `c*4*pi*R^3/3` and average `c`. The profile
must carry center regularity and the radial measure explicitly before it is admitted.

Finite sums have separate syntax `sum(expression, over = (i in index_set))`. The index is a
fresh exact bounded binder scoped only over the integrand. It shadows no existing binding
silently: a conflicting declaration name rejects. The sum preserves element dimension and
does not acquire a physical quadrature measure. Canonical iteration follows the index set's
declared order. Alpha-renaming `i` preserves the sum; capturing an outer value does not.

## Point evaluation and one-sided traces

`evaluate(expression, at = (x = value, v = value))` binds exact coordinate projections for
point evaluation. Each named coordinate occurs once, has a dimension-compatible value inside
its declared bounds, and belongs to the expression's support. Partial point evaluation removes
only the bound factors; supplying every factor returns a lumped value. This `at` is a named
argument of a structural operation, not a declaration's temporal activation clause.

`trace(expression)` inside a boundary relation uses that relation's exact boundary context.
Outside such a context, spell `trace(expression, on = boundary)` explicitly. For an interface
with two parent fields, identify the side with `from = parent_support`; a same-name neighbor
or nearest point cannot determine it. One-sided traces retain the chosen parent and boundary
orientation. A normal flux additionally uses that parent's outward normal. Equal traces and
conserving signed fluxes are separate interface laws, not automatic consequences of taking
a trace.

Point evaluation requires an admitted pointwise representation. A weak field with no admitted
point value, a discontinuous value with no selected side, foreign point/support data, and a
stale Geometry binding reject before evaluation. No renderer interpolation substitutes for
the requested mathematical observation.

## Fields on affine physical boundaries

A Field may live directly on a Cartesian boundary or on one edge of an exact straight-edged planar Geometry
artifact. For example, a density in `kg/m` on a line in 2D has an integral in `kg`:

```eqiora
model LineDensity() {
  domain body = box(0, 2, 0, 3);
  domain wall = boundary(body, axis = 0, side = lower);
  variable density: kg/m on wall;
  relation retain on wall { density = 1[kg/m]; }
  observable mass: kg = integral(density, measure(wall));
}
```

The line has intrinsic dimension 1 and ambient dimension 2. A spatial Cartesian vector on
it therefore has two components; its line measure contributes one power of length.
Geometry-bound variants use the existing `volume` and `boundary(parent=...)` support slots,
with exact caller Geometry selections. An oblique edge retains its actual geometry and
parent incidence. Equal dimensions or coordinates do not substitute a foreign boundary.

This is Model typing and replay. Curved/grouped boundary Fields, physical `grad`/`div` of
boundary Fields, and numerical realization of these Fields remain unsupported.
