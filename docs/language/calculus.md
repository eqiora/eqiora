# Explicit calculus, time, and branches

These target-language decisions define explicit calculus and branching. The examples are specified source,
not claims of current parser, derivative, or numerical admission.

## Independent bindings and partial derivatives

The canonical operation is `partial(expression, wrt = binding)`. A binding is an exact
operator formal, independent input, or admitted coordinate, not a string or a matching name.
An optional `holding = (binding, ...)` is a compile-time set of independent bindings. It is
not a tuple of runtime values. The differentiated binding cannot also be held fixed; duplicate,
foreign, or dependent entries reject. Unlisted independent inputs remain fixed as well.

Result dimensions divide the expression dimension by the differentiated binding's dimension.
Shape is retained for a scalar independent input. A zero derivative retains this result type,
including support, instead of becoming an unrelated dimensionless numeric zero.

An `operator` has a typed input signature, a result type, and a pure expression definition:

```eqiora
operator polynomial(input x: 1, input y: 1): 1 = x * x * y;
operator polynomial_dx(input x: 1, input y: 1): 1 =
  partial(polynomial(x = x, y = y), wrt = x, holding = (y));
operator polynomial_dy(input x: 1, input y: 1): 1 =
  partial(polynomial(x = x, y = y), wrt = y);

model PartialExample(input x: 1, input y: 1) {
  let squared: 1 = x * x;
  observable dx: 1 = partial(squared * y, wrt = x);
  observable dy: 1 = partial(squared * y, wrt = y);
}
```

Both paths give `2*x*y` and `x*x`. At `x = 2`, `y = 3` they give 12 and 4. The alias
`squared` is not a fresh independent binding: holding it fixed while differentiating with
respect to `x` rejects. Substitution through the operator call retains the caller's exact
binding and the chain rule. An equation that implicitly determines `x` does not make this
operation a sensitivity of the solved value of `x`.

For a dimensioned constitutive example, let the temperature input explicitly be a difference:

```eqiora
operator conductivity(
  input delta_temperature: K,
  input k0: W / (m * K),
  input a: 1 / K
): W / (m * K) =
  k0 * (1 + a * delta_temperature + a * a * delta_temperature^2);

operator conductivity_slope(
  input delta_temperature: K,
  input k0: W / (m * K),
  input a: 1 / K
): W / (m * K^2) = partial(
  conductivity(delta_temperature = delta_temperature, k0 = k0, a = a),
  wrt = delta_temperature,
  holding = (k0, a)
);
```

The result is `k0*(a + 2*a*a*delta_temperature)`. With `k0 = 10 W/(m*K)`, `a = 0.01/K`,
and a 20 K difference, conductivity is 12.4 W/(m*K) and its slope is 0.14 W/(m*K^2).
The name does not encode an absolute-temperature type: once absolute/difference quantities
are admitted, this input must use the difference contract, never affine absolute-point algebra.

Ordered nested operations spell second partials, for example
`partial(partial(x*x*y + y^3, wrt = x), wrt = y)`. The inner operation differentiates first;
the ordered binding list remains in the graph. For this smooth polynomial the Hessian is
`[[2*y, 2*x], [2*x, 6*y]]`. Symmetry follows from this polynomial's smoothness, not from
sorting derivative bindings. Heterogeneous input dimensions produce typed blocks rather than
an implicitly homogeneous matrix. The initial profile rejects derivative orders above two.

## Typed Jacobian actions

For the admitted real scalar polynomial profile, `jvp(value, [x, y], [dx, dy])`
applies the Jacobian to an ordered list of input directions. The two lists are structural
arguments: they do not construct a homogeneous array. Each direction must have the exact
scalar type and dimension of its selected independent binding. Repeated bindings, aliases
used as independent inputs, and mismatched direction units are rejected. Other inputs are
held fixed, and the result has the original output dimension.

`vjp(value, x, seed)` returns the dual block for the selected input. This initial scalar
output profile uses the real dimensionless dual pairing: `seed` has reciprocal output units,
and the returned block has reciprocal `x` units. Separate input blocks retain separate types;
a position block and a velocity block are not packed into one spatial vector. Python spells
these operations `lang.jvp(value, wrt=(x, y), tangent=(dx, dy))` and
`lang.vjp(value, wrt=x, cotangent=seed)`.

A Hessian action is formed by applying `jvp` to a first partial. For
`f=x*x*y+y*y*y`, its two blocks are
`jvp(partial(f, wrt=x), [x,y], [dx,dy])` and
`jvp(partial(f, wrt=y), [x,y], [dx,dy])`. This evaluates actions without requiring a dense
Hessian. At `(x,y)=(3,5)` and `(dx,dy)=(2,-1)`, the blocks are `14` and `-18`.
Applying another derivative is outside the order-two bound, including through aliases and
operator composition. These explicit-expression actions do not differentiate implicit solves,
unknown spatial fields, complex inputs, or nonsmooth branches.

For an analytic Cartesian field, a differentiated polynomial operator can bind its
length inputs to `coordinate(0)` and `coordinate(1)` inside a spatial Observable.
The retained operator is expanded by the same calculus projection during ordinary
Q1 Result quadrature. Differentiating the independent operator inputs precedes
coordinate substitution; this does not introduce general coordinate-factor partials
or derivatives of an unknown field representation. Exact declared coordinate selectors and
the separate Q1 first-Field derivative path are described in the
[coordinate profile](coordinates.md).

## Continuous time and higher-order evolution

`time()` is the time coordinate, with dimension `s`, of the enclosing continuous timeline.
Fresh initialization explicitly binds its origin and starting coordinate. Restart retains
that timeline and the accepted coordinate; it does not restart time at zero. A numerical
step counter, sample number, periodic clock identity, or wall clock cannot substitute for it.
A pure operator has no enclosing evolving context and must receive time as a declared input.
`time()` outside an applicable continuous context rejects.

`derivative(expression)` differentiates along that timeline, including all admitted evolving
dependencies. For a distributed field it holds the declared spatial coordinates fixed. It
does not mean a material derivative or differentiate through mesh motion. Parameters are
constant during a Run, even when another Run uses a different study value.
The [prescribed moving-volume Law profile](conservation-laws.md#prescribed-moving-volume-balance)
retains the coordinate map, its rate and material-minus-mesh transport explicitly. Its density
balance includes the volume Jacobian; a scalar trajectory derivative alone cannot supply it.

```eqiora
model Oscillator(
  parameter mass: kg = 1 [kg],
  parameter stiffness: N / m = 4 [N / m],
  parameter initial_displacement: m = 1 [m],
  parameter initial_velocity: m / s = 0 [m / s]
) {
  state displacement: m;
  initial {
    displacement = initial_displacement;
    derivative(displacement) = initial_velocity;
  }
  relation motion {
    mass * derivative(derivative(displacement)) + stiffness * displacement = 0 [N];
  }
  observable squared_displacement_rate: m^2 / s =
    derivative(displacement * displacement);
  observable elapsed: s = time();
}
```

Bind the timeline origin and initial coordinate to zero for this specimen. With the written
defaults, `x(t) = (1 m)*cos((2/s)*t)` and `dx/dt = -(2 m/s)*sin((2/s)*t)`.
The squared-displacement rate is `2*x*dx/dt`, not zero and not `2*x` alone. At `t = pi/8 s`
it equals `-2 m^2/s`. The total mechanical energy is 2 J at every time.

A first-order Formulation introduces the velocity auxiliary and retains `dx/dt = v`,
`mass*dv/dt = -stiffness*x`, and both initial conditions. The source still owns one
second-order displacement state declaration; numerical lowering cannot discard its initial
velocity or secretly add another author-controlled state. A restart at `pi/8 s` keeps that
accepted position, velocity, and time instead of reapplying the defaults.

The smooth path rejects `pre`, `next`, clocked values, stochastic increments, and derivatives
across an unhandled jump. Differentiating a parameter-only expression yields a typed zero;
changing a Parameter between Runs is not continuous evolution.

## Predicates and lazy piecewise values

Use Boolean literals `true` and `false`, `not`, `and`, `or`, and one conditional spelling:
`if predicate then expression else expression`. Both branches are mandatory and must have
compatible types, support, and activation. The predicate is Boolean, never a nonzero number.
`and` and `or` short-circuit, and only the selected conditional branch is evaluated. Batched
execution masks inactive elements before evaluating an otherwise invalid branch.

```eqiora
operator guarded_root(input x: 1): 1 =
  if x >= 0 then math.sqrt(x) else 0;

operator piecewise_current(
  input voltage: V,
  input negative_conductance: A / V,
  input positive_conductance: A / V
): A = if voltage < 0 [V]
  then negative_conductance * voltage
  else positive_conductance * voltage;

model PiecewiseExample(input voltage: V) {
  observable current: A = piecewise_current(
    voltage = voltage,
    negative_conductance = 1 [A / V],
    positive_conductance = 2 [A / V]
  );
}
```

`guarded_root(-1)` is zero without evaluating a negative real square root; `guarded_root(4)`
is 2. Its ordinary derivative at zero rejects. The current is -1 A at -1 V, zero at exactly
zero voltage, and 2 A at +1 V. Its open-branch slopes are 1 A/V and 2 A/V. At zero, selecting
the second value branch does not establish a derivative: the two one-sided slopes disagree.

A piecewise law is memoryless. It neither manufactures hysteresis nor inserts an event into
the timeline. A Run requiring crossing localization must consume an explicit admitted event
or reject that execution profile. Numerical tolerances never move the mathematical threshold.
Complex ordering, incompatible branches, missing branches, and an ordinary derivative at a
kink reject at their typed owner.

All examples on this page are lumped. No Geometry, spatial boundary, or mesh is needed for
the explicit-expression derivative path. Coordinate derivatives, functional variations, and
sensitivities of implicit solutions keep their distinct mathematical owners.
