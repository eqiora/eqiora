# Coordinates, measures, and local evaluation

This section of the [target grammar](core.md) covers exact coordinate factors and the
foundation audit's position/velocity derivative specimen. It specifies mathematical nodes;
their current numerical admission is not established by the examples.

The current implementation admits abstract dimensioned interval slots and owned products in
Model and Component source, including nested products, physical Cartesian region factors, and
whole-product and selected-factor Observable measures.
The source support owns factor identity; native `StaticBindingValue::CoordinateInterval` and
Python `CoordinateInterval` supply checked finite bounds. Model/Transaction v40 and structural
fingerprint v35 retain these factors. No ambient physical frame or numerical realization is
inferred. Exact coordinate binders and real scalar polynomial partials execute through the
shared calculus evaluator, including independently dimensioned position and velocity factors.
First coordinate derivatives of continuous scalar Fields remain explicit Model nodes; the
Cartesian Q1 observation path evaluates their basis derivatives and polynomial chain rules.
Focused tests cover a two-dimensional Q1 bilinear field, exact analytic values, factor/axis/unit
rejection, and replay. Bounded real scalar factor integrals execute coordinate densities with
finite Result amplitudes, exact selected measures and explicit remaining output support;
[registered evidence](../../verify/language/factor-integrals/README.md) covers polynomial moments
and finite Gaussian error bounds. Non-Cartesian product factors, product-domain Field/PDE
realizations, transient radial diffusion, nonpolynomial integral solver coupling, general differentiation under integrals,
higher unknown-Field partials and curved embedded-field extensions remain separate work. The complete examples below include target operations beyond this bounded implementation.

## Explicit coordinate pullbacks

`pullback(value, from=(xi, eta), at=(x=2*xi+eta, y=3*eta))` composes an
invariant scalar on the target support with the declared map from the source support.
Both selector inventories must be complete and unique. Matching units or sampled values do
not identify coordinates or maps. All assignments are evaluated before binding any target
coordinate, so mappings between aliases on the same support are simultaneous.

The same `from` and `at` arguments define `jacobian_determinant`, `volume_jacobian`, and
`map_orientation`. For a square map with Jacobian J these return det(J), |det(J)|, and
its orientation sign respectively. The first two retain the product of target coordinate
units divided by the product of source coordinate units; orientation is dimensionless.
A singular map has zero signed determinant, but cannot supply an invertible volume scale
or orientation. Numerical conditioning and execution resource limits are distinct from
mathematical dimension: these operators have no fixed maximum matrix dimension.

For x=2ξ+η and y=3η, u=x²+xy pulls back to 4ξ²+10ξη+4η².
The chain rule gives its reference derivatives (8ξ+10η, 10ξ+8η); applying J⁻ᵀ
recovers the physical derivatives (4ξ+5η, 2ξ+η). The volume scale is 6, so the
integral over the image of the unit square is 31 m⁴. Omitting the scale instead
integrates over the reference measure and gives 31/6 m⁴. A reflection changes orientation,
not the positive volume measure. The map and its factor must be written explicitly in the
density; `pullback` alone does not change the integration measure.

The current ordinary Result integral path proves spatially affine maps, with finite solved
scalar coefficients, and uses the canonical coordinate evaluator at quadrature points.
Non-affine maps remain expressible and locally evaluable in the admitted differentiable
profile; general nonlinear map quadrature is not yet implemented. Point evaluation checks
exact admitted planar Geometry membership, including holes, rather than only a bounding box.
That check is not a proof of global map coverage or bijectivity.

Scalar pullbacks may have different source and target dimensions, but the three determinant
factors require a square map. They do not infer an embedded metric, a chart atlas, vector or
covector frame conversion, or a Piola transform. Unknown spatial Field derivatives require
an admitted reconstruction; higher composed pullback derivatives and general map sensitivity
remain separate execution work. Existing declared radial measures and center regularity
retain their own contracts below.

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
`from`, and does not accept `at` or an initializer. For a multi-axis physical factor, select an
axis explicitly: `coordinate y: m on phase from position[1];`. Axes are zero-based within that
factor and follow its declared Cartesian/Geometry order. Omitting the axis is admitted only for
a scalar factor; it never flattens a product into an ambient coordinate frame.

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
establishes typing, measure semantics and analytic coordinate partials; a product-space solver
remains separate work.

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

The current Q1 path supplies classical derivatives in cell interiors and the corresponding weak
first derivative almost everywhere. It does not promise a continuous gradient across cells.
Observation admission rejects a piecewise-constant/TPFA Result instead of treating its Field as
coordinate-independent. Boundary embedding coordinates remain readable, but boundary partials
require an intrinsic chart and are rejected. A physical scalar Field uses the same axis order for
`grad` components and coordinate partials.

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

Omitting endpoint options integrates the complete fixed support. Explicit finite limits use
`integral(expression, measure(line), lower=lo, upper=hi)` on one complete Cartesian
coordinate interval. Both endpoints are required, lumped real scalars with that exact
coordinate unit; they cannot capture a bound coordinate. They must stay within the fixed
support. Reversing endpoints negates the integral; equal endpoints yield zero. Numerical
quadrature remains an explicit Result choice, separate from the mathematical limits.

For a named polynomial integral, an independent Parameter partial uses the Leibniz rule:
the integral of the density partial, plus the density evaluated at the upper endpoint times
its partial, minus the analogous lower contribution. Existing polynomial calculus and exact
point evaluation admit these terms. For example, `integral(x*x, measure(line), lower=0[m],
upper=a)` has derivative `a*a` with respect to length Parameter `a`. Even at equal bounds,
the derivative need not vanish: `integral(1, measure(line), lower=a, upper=2*a)` is `a`
and has derivative 1 at a=0. This path rejects unknown Field densities, singular or
nonpolynomial differentiation, partial products, weighted measures and moving Geometry.
Explicit limits are observations, not an admitted integral solver constraint.

The current first-partial profile admits `partial(density, wrt=remaining_x)` as an
Observable declaration root when `density` is one named polynomial integral over fixed
bounded coordinate intervals. Declare `remaining_x` on the output support from the exact
remaining factor; an integrated coordinate is bound and cannot be a free selector.
An independent Parameter is also a valid selector. The compiler differentiates the density
with the existing polynomial calculus and retains the same measure and output support.
Signed nonzero real literal divisors are allowed; variable denominators, higher/composite
integral derivatives, nonpolynomial densities and Cartesian Geometry measures are outside
this initial derivative profile.

Measures multiply dimensions. A velocity integral contributes `m/s`, a physical line integral
`m`, a surface integral `m^2`, and a physical volume integral `m^3`. An embedded line in 2D
still has intrinsic measure dimension `m`. Metric/Jacobian weights remain mathematical data;
quadrature points and numerical weights implement that data rather than define it.

An integral is not a normalized average. Write a denominator explicitly, and reject a zero
measure before division. Use `spherical_measure(radius)` to declare the spherical-symmetry measure
`4*pi*r^2 dr` on one length-valued coordinate interval `[0,R]`, with `R > 0`.
A constant concentration `c` has total `c*4*pi*R^3/3` and explicit average `c`.
The measure alone does not define a radial PDE or center condition. Static radial diffusion
now admits the explicit pair `partial(r*r*j,wrt=r)=q*r*r`, `j=-D*partial(c,wrt=r)` with
fixed scalar Parameters, positive `D`, `evaluate(j,at=(r=0[m]),side=upper)=0` and an explicit
surface concentration at `R` approached from below. Its cell-centered coordinate-grid Plan
retains radial face areas and cell volumes; ordinary Result observations use this spherical
measure. See the [executable model](../../verify/language/factor-integrals/models/radial-diffusion.eqi)
and [independent refinement derivation](../../verify/language/factor-integrals/README.md#static-radial-diffusion-and-particle-average).
The stored cell-constant fields have no admitted pointwise reconstruction. Time evolution,
variable coefficients, nonlinear laws and shells with a positive inner radius remain unsupported.
Spherical functional variations and integral partials remain unsupported.

Continuous noninitial condition Relations can constrain a lumped Observable, for example
`relation inventory { mass=45; }` where `mass` is a declared integral. Finite real scalar
Plans admit polynomial coordinate densities with finite Field/Parameter coefficients,
including nested integrals and explicit normalized ratios. The Model retains exact
Observable references and measures. The Plan selects tensor-product Gauss rules with
one to seven points per axis from a conservative polynomial degree; the spherical
Jacobian adds degree two. Each integral has at most 4096 points and expansion is limited
to 65536 expression operations and dependency depth 32. Coefficients can be nonlinear
in finite unknowns when the existing Newton policy admits them. Original operand checks
recompute integral values at the exact candidate Field/Parameter point.

Coordinate-dependent denominators and nonpolynomial coordinate densities reject in this
finite coupling profile, even when explicit Result quadrature can observe them. Spatial
unknowns, remaining output coordinates, product-domain PDEs and discrete/initial or
conservation-law coupling require their own numerical admission.

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

The implemented point profile requires **every** axis of one exact volume or coordinate
support, each exactly once; partial binding remains a target operation. Bindings are
simultaneous and produce a lumped value. The ordinary static Result evaluates analytic
ramp/sinusoidal expressions and reconstructs scalar Cartesian Q1 Fields with the same basis
used by its integral observations. In one dimension this is piecewise affine P1. The source
Model retains the binding expression; the accepted Result/Plan retains mesh, coefficients,
Geometry and reconstruction identity. No nearest vertex, display resolution or output frame
chooses the value.

```eqiora
coordinate x: m on body from body[0];
observable probe: K = evaluate(temperature, at=(x=0.125[m]));
observable slope: K/m = evaluate(partial(temperature, wrt=x), at=(x=0.125[m]));
observable left_slope: K/m =
  evaluate(partial(temperature, wrt=x), at=(x=0.5[m]), side=lower);
```

`side=lower` and `side=upper` select approaches from lower and higher coordinate values,
respectively. They currently require a one-dimensional point. An outward approach at a
Domain endpoint rejects. Q1 values are continuous; Q1 derivatives at cell boundaries require
an explicit side. Analytic piecewise/branch limits and piecewise-constant Field values are
not admitted by this profile. A requested side never silently substitutes the value of a
conditional at its branch boundary. Analytic polynomial coordinate/Parameter partials can
be evaluated inside `evaluate`; differentiating the enclosing binding itself, moving points,
State JVPs, transient time selection, reductions inside point bindings and arbitrary
reconstruction spaces remain unsupported.
Unsupported `time` or side syntax rejects rather than choosing an output frame.

The Python constructor uses the same structured AST:
`eqiora.lang.evaluate(value, at=((x, eqiora.lang.quantity(0.125, eqiora.units.m)),), side=None)`.
Coordinates and expressions retain their lexical owners. Source emission, native compilation,
and Model/Plan/Result replay share the same evaluation path. The registered
[exact point case](../../verify/language/exact-point-evaluation/README.md) gives independent
values, coordinate slopes, interface-side and admission falsifiers.

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

## Bounded nonlocal kernel actions

The [executable interaction model](../../verify/language/factor-integrals/models/nonlocal-interaction.eqi)
uses the existing product, integral and Observable owners:

```eqiora
support pair: product(target, source);
coordinate x: m on pair from target;
coordinate y: m on pair from source;
let kernel: 1/m on pair = x*y/1[m^3];
let u: 1 on pair = amplitude*y/1[m];
observable action: 1 on target = integral(kernel*u, measure(source));
observable inventory: m = integral(action, measure(target));
relation prescribed_inventory { inventory = 2[m]; }
```

Supply bounded `interval(m)` bindings for both supports and a finite dimensionless variable
`amplitude`. The source coordinate is integrated while the target remains an output
coordinate. Changing the source bound changes the action; no outside-domain values or
infinite-domain limit are implied. The Plan owns polynomial quadrature for the relation;
Result observation requires explicit quadrature on the exact selected measure.

A nominal `indexset` and explicit `sum` can instead declare a finite atomic action, with
physical measure units carried by each mass. This is its own discrete measure, not an
implicit continuous quadrature rule. Typed kernel applications and discrete ordinal
intermediates retain their exact meaning during coordinate observations.

Adjoints require declared pairings and both source and target measures. The
[independent derivation](../../verify/language/factor-integrals/README.md#bounded-nonlocal-actions)
checks a nonsymmetric example with unequal domain extents. It also derives the finite
residual, output and total JVP/VJP products of an integral-coupled solve. This bounded path
does not construct general function-space adjoints, solve arbitrary nonlocal Fields,
admit singular kernels or supply coordinate-observation State tangents.
