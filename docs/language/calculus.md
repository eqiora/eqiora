# Explicit calculus, time, and branches

This page specifies explicit calculus and branching in the language. The examples show the
specified source form; the parser, derivative support, or numerical execution may not yet
accept every one of them.

## Independent bindings and partial derivatives

A partial derivative is written `partial(expression, wrt = binding)`. The binding must be an
operator formal, an independent input, or a supported coordinate; a string or a name that
merely matches is not accepted. The optional `holding = (binding, ...)` is a compile-time set
of independent bindings, not a tuple of runtime values. The differentiated binding cannot also
be held fixed, and duplicate, foreign, or dependent entries are rejected. Independent inputs
that are not listed are held fixed too.

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
respect to `x` is rejected. Calling an operator substitutes the caller's binding and applies
the chain rule. If an equation implicitly determines `x`, this operation still does not give
the sensitivity of the solved value of `x`.

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
The name alone does not give the input a temperature-difference type. Once the language
distinguishes absolute temperatures from differences, this input must use the difference type
rather than affine arithmetic on absolute temperatures.

Ordered nested operations spell second partials, for example
`partial(partial(x*x*y + y^3, wrt = x), wrt = y)`. The inner operation differentiates first;
the ordered binding list remains in the graph. For this smooth polynomial the Hessian is
`[[2*y, 2*x], [2*x, 6*y]]`. Symmetry follows from this polynomial's smoothness, not from
sorting derivative bindings. Inputs with different dimensions produce separately typed blocks
rather than one homogeneous matrix. Repeated polynomial partials are bounded by the calculus
node and depth budgets, without a separate derivative-order limit.

## Typed Jacobian actions

`jvp(value, [x, y], [dx, dy])` applies the real-linear Jacobian to an ordered
list of input directions. The two lists are structural arguments and do not build a homogeneous
array. Real and complex scalar or channel-array inputs retain their complete values. Each
direction must have the exact type, shape and dimension of its selected independent binding.
Repeated bindings, aliases
used as independent inputs, and mismatched direction units are rejected. Other inputs are
held fixed, and the result has the original output type.

`vjp(value, x, seed)` returns the dual block for the selected input. It uses the real
dimensionless pairing `Re(sum(conj(seed) * delta_value))`: `seed` has the output's complete
shape and scalar domain with reciprocal output units. The returned block has `x`'s complete
shape and scalar domain with reciprocal `x` units. Reverse actions contract every output
component. Separate input blocks retain separate types;
a position block and a velocity block are not packed into one spatial vector. Python spells
these operations `lang.jvp(value, wrt=(x, y), tangent=(dx, dy))` and
`lang.vjp(value, wrt=x, cotangent=seed)`.

A Hessian action is formed by applying `jvp` to a first partial. For
`f=x*x*y+y*y*y`, its two blocks are
`jvp(partial(f, wrt=x), [x,y], [dx,dy])` and
`jvp(partial(f, wrt=y), [x,y], [dx,dy])`. This evaluates actions without requiring a dense
Hessian. At `(x,y)=(3,5)` and `(dx,dy)=(2,-1)`, the blocks are `14` and `-18`.
For a complex example, `f=z*math.conj(z)+math.complex(0,1)*z` has real Jacobian
`[[2*a, 2*b-1], [1, 0]]` at `z=a+ib`. At `z=2+5i`, the direction `3-4i` gives
`jvp(f,[z],[direction])=-24+3i`; the seed `7+11i` gives `vjp(f,z,seed)=39+63i`.
Conjugate dependence is retained; these are real-linear actions, not holomorphic derivatives.
Whole typed operator results compose with actions, and static channel indexing such as
`jvp(z,[z],[direction])[1]` retains the value's local axes. Nested actions use the same calculus.

Indexing retains the complete original output's domain, even if a later derivative is zero.
For an array-valued operator `f(x,z)=x*math.log(component(z,0))`, with
`z:array<1,2>`, `partial(f(x=x,z=z)[0],wrt=x)` still requires both entries of `z`
to be positive. Discarding an output component does not discard its logarithm's domain.
Formal derivatives retain these mathematical domains without requiring an unnecessary
numerical evaluation of the original polynomial value, which can overflow even when its
selected derivative is finite.

Conditionals over the admitted numeric inputs also compose when the predicate is
independent of every selected input.
For `f = if k>0 then math.log(x) else math.log(-x)`, holding `k` fixed gives
`partial(f,wrt=x)=1/x` and `partial(partial(f,wrt=x),wrt=x)=-1/(x*x)` on
the selected branch's domain. Only that branch's source and derivative domains are checked,
including when the result is zero. Differentiating with respect to `k` is rejected;
an inner zero partial does not erase the original branch dependency. Units of an absent
branch derivative come from the typed expression, without evaluating the other branch.

These actions apply to admitted explicit expressions only. Array construction inside a
differentiated expression, general conditional derivatives, implicit solves and general
unknown-spatial-field variations remain outside this executable profile. A scalar `partial`
selector still requires an independent real scalar. Source and derivative-domain checks remain
in force even when the selected input is unused and the derivative is zero.

Model coordinate actions use the same physical dependencies as coordinate `partial`.
For a represented Field `u(x,y)`, `jvp(u,[x],[x])` means `partial(u,wrt=x)*x`;
the Field is not held fixed with respect to its spatial coordinate. VJP contracts
the corresponding coordinate partial with the seed using the same real pairing.
The admitted first-order Field profile and its reconstruction remain authoritative;
nested mixed Field derivatives require a supported representation and otherwise reject.

For an analytic Cartesian field, a differentiated polynomial operator can bind its
length inputs to `coordinate(0)` and `coordinate(1)` inside a spatial Observable.
The operator is expanded with the same calculus rules during Q1 Result quadrature. Its inputs
are differentiated before the coordinates are substituted, so this provides neither general
derivatives with respect to coordinates nor derivatives of an unknown field. Declared
coordinate selectors and the separate reconstruction-based first-Field derivative are described in the
[coordinate profile](coordinates.md).

## Continuous time and higher-order evolution

`time()` is the time coordinate, with dimension `s`, of the enclosing continuous timeline.
Fresh initialization explicitly binds its origin and starting coordinate. Restart retains
that timeline and the accepted coordinate; it does not restart time at zero. A numerical
step counter, sample number, periodic clock identity, or wall clock cannot substitute for it.
A pure operator has no enclosing evolving context and must receive time as a declared input.
`time()` outside an applicable continuous context rejects.

`derivative(expression)` differentiates along that timeline, through every evolving quantity
the expression depends on. For a distributed field it holds the declared spatial coordinates
fixed; it is not a material derivative and does not differentiate through mesh motion.
Parameters are constant during a Run, even when another Run uses a different value.
The [prescribed moving-volume balance](conservation-laws.md#prescribed-moving-volume-balance)
keeps the coordinate map, its rate, and the material-minus-mesh transport explicit. Its density
balance includes the volume Jacobian, which a scalar time derivative cannot supply.

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

A first-order Formulation introduces an auxiliary velocity and keeps `dx/dt = v`,
`mass*dv/dt = -stiffness*x`, and both initial conditions. The source still declares a single
second-order displacement state; lowering can neither drop its initial velocity nor add a state
the author did not declare. A restart at `pi/8 s` keeps the position, velocity, and time
reached there instead of reapplying the defaults.

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
the timeline. A Run that needs to locate a crossing must use an explicit event; otherwise it is
rejected. Numerical tolerances never move the mathematical threshold. Ordering complex values,
incompatible or missing branches, and ordinary derivatives at a kink are all rejected.

All examples on this page are lumped. No Geometry, spatial boundary, or mesh is needed for
derivatives of explicit expressions. Coordinate derivatives, functional variations, and
sensitivities of implicit solutions are separate operations, documented on their own pages.
