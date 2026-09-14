# Typed scalar operators

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

Concrete ordinary real scalar signatures coexist with the existing operator-local generic
`scalar` and `spatial[rank]` classes. Those generic classes remain dimension-polymorphic;
`scalar` is not a synonym for the dimensionless concrete type `1`. Existing exact tensor
bodies retain `rational`, `component` and `delta`. The former `pure operator` and `->`
declaration spellings are not part of the current source surface.

Polynomial bodies use exact decimal or rational coefficients, formal references and bounded
arithmetic. Powers use positive integer literal exponents up to 255 and share the
calculus work and depth limits. Zero, negative or computed exponents are outside this
profile; write the constant `1` for a degree-zero term. This preserves the general value-power
rule that `0^0` rejects instead of erasing its domain check during polynomial expansion.
Calls between local scalar operators compose through the existing exact calculus owner. Lexical names and formal bindings are resolved before expansion; cycles, capture,
unknown or missing arguments, incompatible types and excess work reject locally. Operators
cannot read hidden evolving state or invoke host callbacks. Cross-package composition inside
operator definitions and the general analytic/complex catalog remain outside this profile;
existing qualified package calls at Model and Component use sites retain their visibility rules.

Definition identity retains exact arithmetic and input/result scalar-domain and dimension
constraints. The
current Model stores the operator definition and application identity. Execution projects an
admitted scalar application into the ordinary expression graph and uses the existing evaluator;
proof normalization does not authorize reassociating floating-point operations.

The native scalar calculus owner exposes inspectable first and second partial derivative
graphs. For the example, the temperature derivatives at 20 K are 0.14 W/(m*K^2) and
0.002 W/(m*K^3). A zero partial retains its derived dimensions and support. Unsupported
orders or profiles reject rather than substituting finite differences. Source
`partial(expression, wrt = binding, holding = (...))` admits real scalar polynomial
arithmetic and local operator composition at exact declared formals, Parameters, or continuous
state Fields. Let aliases retain dependencies and cannot be differentiation or holding bindings.
A Component Parameter slot preserves a directly bound parent Parameter direction; a literal
or derived binding does not introduce a new independent value.
The [formal partial evidence](../../verify/language/formal-partials/README.md) checks analytic
values and quotient dimensions. General smooth functions, coordinate derivatives, mixed source
partials, and solved sensitivities remain outside this executable profile.

Python `Module.operator` authors a local concrete scalar definition from typed formal handles
and a callback invoked once. Its returned handle accepts named arguments. Formal handles
cannot escape their scope, and a foreign Module cannot acquire a call merely because every
argument is a literal. Qualified imported operator calls remain available through source
compilation rather than a raw qualified-name handle constructor.

## Guarded scalar profile

Concrete real scalar signatures also admit the [conditional scalar profile](conditionals.md):
Boolean predicates, lazy value branches, square roots and the bounded nonsmooth math recipes.
Typed exact constants retain their SI dimensions, including zero thresholds. Generic scalar
and tensor polynomial definitions retain their existing contract; the guarded profile requires
concrete dimensions and real scalar formals/results. Guarded definitions share the same retained
calculus, composition and execution projection. Polynomial normalization rejects the guarded
profile instead of treating a branch or domain requirement as an algebraic identity.

## Continuous expression derivatives

`time()` denotes the enclosing continuous timeline coordinate in seconds. Initial and restart
time are supplied through the ordinary time lifecycle. A pure operator receives time as a named
input; it cannot read an ambient clock. Bare `time` is not a source expression.

`derivative(expression)` applies the total time chain rule to admitted real scalar polynomial
expressions, let aliases and operator composition over continuous States and fixed Parameters.
For example, `derivative(q*q)` expands through the shared formal partial owner to the two product
contributions, and `derivative(time()*q)` includes both explicit time and State evolution.
The default distributed-field derivative holds the declared spatial coordinates fixed.
Clocked values, `pre`/`next`, algebraic variables and unsupported derivative products reject.
The [expression chain-rule case](../../verify/time/expression-chain-rule/README.md) executes a
nonlinear stored quantity through the common implicit lifecycle and exact artifact replay.
