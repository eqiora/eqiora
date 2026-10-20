# Typed operators

`operator` declares a bounded, side-effect-free mathematical definition. Inputs and the
result carry their types, and calls bind arguments by formal name:

```eqiora
operator conductivity(
  input delta_temperature: K,
  input k0: W / (m * K),
  input a: 1 / K
): W / (m * K) =
  k0 * (1 + a * delta_temperature + a * a * delta_temperature^2);
```

With a temperature difference of 20 K, `k0 = 10 W/(m*K)` and `a = 0.01/K`, the
result is 12.4 W/(m*K). Both the constant and temperature-dependent terms have the
same complete dimensions. A call uses
`conductivity(delta_temperature = temperature, k0 = coefficient, a = curvature)`;
authored named-argument order does not change the formal binding.

Concrete signatures retain the complete shared value type: real/complex domain, physical
dimensions, ordered axes, channel rank, frame and nominal bases. They coexist with the
operator-local generic `scalar` and `spatial[rank]` classes. Those generic classes remain dimension-polymorphic;
`scalar` is not a synonym for the dimensionless concrete type `1`. Existing exact tensor
bodies retain `rational`, `component` and `delta`. The former `pure operator` and `->`
declaration spellings are not part of the current source surface.

Polynomial bodies use exact decimal or rational coefficients, formal references and bounded
arithmetic. Coefficient precision is independent of machine integer width. The shared exact
coefficient owner bounds input and arithmetic work and records reduced decimal integer parts
in artifacts. Executable lowering rounds each coefficient to binary64 and rejects nonzero
coefficients that round to zero or infinity. Exact symbolic meaning does not authorize
reassociation or guarantee finite results for every numerical evaluation.
Powers use positive integer literal exponents up to 255 and share the
calculus work and depth limits. Zero, negative or computed exponents are outside this
profile; write the constant `1` for a degree-zero term. This preserves the general value-power
rule that `0^0` rejects instead of erasing its domain check during polynomial expansion.
Local calls with scalar formal inputs compose through the existing exact calculus owner,
including shaped outputs when the caller satisfies the callee's component-layout contract. Lexical names and formal bindings are resolved before expansion; cycles, capture,
unknown or missing arguments, incompatible types and excess work reject locally. Operators retain declared real-to-complex result embeddings during composition, including physical
units and the positive-zero imaginary component that selects a complex branch. Operators cannot read hidden evolving state or invoke host callbacks. Cross-package composition inside
operator definitions, composition with shaped formal inputs and unrestricted analytic functions
remain outside this profile;
existing qualified package calls at Model and Component use sites retain their visibility rules.

Definition identity retains exact arithmetic and complete input/result type constraints. The
current Model stores the operator definition and application identity. Execution projects an
admitted scalar application into the ordinary expression graph and uses the existing evaluator;
proof normalization does not authorize reassociating floating-point operations.

The native scalar calculus owner exposes inspectable first and second partial derivative
graphs. For the example, the temperature derivatives at 20 K are 0.14 W/(m*K^2) and
0.002 W/(m*K^3). A zero partial retains its derived dimensions, support and original source domain.
Derivatives also reject points where their own value is undefined, such as a square-root slope at zero. Unsupported
profiles and expressions exceeding calculus resource budgets reject rather than substituting finite differences.
Repeated polynomial partials have no separate derivative-order ceiling. Source
`partial(expression, wrt = binding, holding = (...))` admits real scalar polynomial
arithmetic, quotients, `math.sin`, `math.cos`, `math.exp`, `math.log`, `math.sqrt`,
and local operator composition at exact declared formals, Parameters, or continuous
state Fields. Let aliases retain dependencies and cannot be differentiation or holding bindings.
A Component Parameter slot preserves a directly bound parent Parameter direction; a literal
or derived binding does not introduce a new independent value.
The [formal partial evidence](../../verify/language/formal-partials/README.md) checks analytic
values and quotient dimensions. General smooth functions, unknown-field coordinate derivatives and solved sensitivities
remain outside this executable profile.

Python `Module.operator` authors a local typed definition from formal handles and a callback
invoked once. Type admission uses the shared compiler. `eqiora.lang.component(value, *axes)`
and `eqiora.lang.delta(left_axis, right_axis)` construct the same component calculus as source.
Integer axes name coordinates of the current result, not fixed component indices. For example,
`component(left, 0) * component(right, 1)` defines a dyadic result from two vector formals;
`delta(0, 1)` selects its diagonal. Axis bounds, shape, frame and nominal basis requirements
remain compiler checks. Its returned handle accepts named arguments. Formal handles
cannot escape their scope, and a foreign Module cannot acquire a call merely because every
argument is a literal. `ModuleRef.operator(name)` binds an imported public operator to its
exact provider and preserves named formal order through the same compiler.

## Guarded scalar profile

Concrete real scalar signatures also admit the [conditional scalar profile](conditionals.md):
Boolean predicates, lazy value branches, square roots and the bounded nonsmooth math recipes.
Typed exact constants retain their SI dimensions, including zero thresholds. Each operation
checks its actual operand constraints through the common component type proof. Unit-preserving
functions and rational dimension powers can retain symbolic input units; trigonometric,
exponential and logarithmic functions require provably dimensionless operands.
The body derives the result's dimensions and domain;
declared result constraints are checked, while a generic result layout can be fixed by its
typed caller. Predicates require real numeric operands or Boolean values. Guarded definitions share the same retained
calculus, composition and execution projection. Polynomial normalization rejects the guarded
profile instead of treating a branch or domain requirement as an algebraic identity.

## Continuous expression derivatives

`time()` denotes the enclosing continuous timeline coordinate in seconds. Initial and restart
time are supplied through the ordinary time lifecycle. For a common ODE Plan,
`State.initial(plan, time_s=...)` evaluates the authored initial equations at that
explicit coordinate (default zero); `State.from_result(..., time_s=...)` retains
the accepted restart coordinate. Neither operation shifts `time()` back to zero.
Fresh initial Parameter tangents are validated at the chosen initial coordinate,
not by attempting an unrelated initialization at zero during Plan resolution. A pure operator receives time as a named
input; it cannot read an ambient clock. Bare `time` is not a source expression.

`derivative(expression)` applies the total time chain rule to the admitted real scalar calculus
expressions above, let aliases and operator composition over continuous States and fixed Parameters.
For example, `derivative(q*q)` expands through the shared formal partial owner to the two product
contributions, and `derivative(time()*q)` includes both explicit time and State evolution.
The default distributed-field derivative holds the declared spatial coordinates fixed.
Nested total derivatives expand through the same ordered partial transform; products of State
values and rates keep every chain-rule term. A higher-order State retains its source identity
and lowers to value/derivative coordinates with explicit companion equations and initial data.
The resolved ODE `Plan.formulation` exposes `FirstOrderEvolution`, its source Relation identity,
and the ordered `(FieldRef, derivative order)` state coordinates. Consecutive coordinates of
the same Field obey the companion equation `derivative(q_k) = q_(k+1)`. This correspondence
is reconstructed and checked when the Plan is restored.
Clocked values, `pre`/`next`, algebraic variables and unsupported nonsmooth rules reject.
The [expression chain-rule case](../../verify/time/expression-chain-rule/README.md) executes a
nonlinear stored quantity through the common implicit lifecycle and exact artifact replay.

Continuous ODE Observables may read these derivatives. At an accepted smooth point the
execution owner supplies stored lower derivatives and the highest flow rate with their
original units. Terminal and time-integral Parameter variations include both state dependence
and direct flow-Parameter dependence. A derivative order not supplied by the admitted ODE
is unavailable; no extra evolution state or initial condition is invented for observation.
Reset maps, impulses and stochastic increments are not differentiated as smooth expressions.
