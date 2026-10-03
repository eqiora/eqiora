# Numeric domains and scalar operations

This target-language catalog defines the bounded scalar operations. Each implementation slice must expose
only the rows it actually supports. The table does not establish current execution coverage.

## Closed enum values and case expressions

An enum declares a nonempty, bounded set of unique members at source-file scope:

```eqiora
enum Mode { Heating, Cooling, Fault }
model Controller(parameter mode: Mode = Mode.Heating) {
  variable command: 1;
  relation classify {
    command = case mode {
      Mode.Heating => 2,
      Mode.Cooling => -3,
      Mode.Fault => 0
    };
  }
}
```

Enum values are dimensionless invariant scalars with exact nominal declaration identity.
Two import aliases of the same declaration denote the same type; another declaration with
identical names and members is a different type. Members are not integers or ordinals.
There is no arithmetic, ordering, numeric conversion, Boolean coercion, implicit zero or
ordinary continuous derivative. Enum arrays and spatial enum values are not admitted.
State initialization and Parameter defaults must name a member of the exact declaration.

A `case` has exactly one arm for each member of its scrutinee's enum. Missing, duplicate or
foreign members reject; wildcard/default arms and payload patterns are unsupported.
Every value arm must pass the common branch type, dimension, support and activation checks.
The compiler lowers the exhaustive case to exact enum equality and lazy value selection:
only the selected value arm executes. Arm order does not establish execution priority.
The retained declaration and member identity survive Model replay.

Enums can be Parameters, explicitly initialized State and clocked input/output values.
Use the typed execution session for discrete values; the legacy scalar trajectory interface
is not an enum transport. [Finite event-driven modes](events.md#finite-mode-controlled-components)
reuse these values with explicit reset and hold equations. Authored transition priority and
hierarchical, parallel or history statecharts remain separate capabilities.

## Closed records and typed buses

A record declares ordered, uniquely named mathematical members. Each member keeps its
own dimension, shape and scalar domain. Record declarations have exact nominal identity;
matching member names and types do not make two declarations interchangeable.

```eqi
enum Mode { Off, On }
record Config { gain: 1, mode: Mode }
record Bus { voltage: V, valid: bool, mode: Mode }

component Controller(parameter config: Config) {
  clock tick = periodic(1[s]);
  state bus: Bus at tick;
  initial {
    bus.voltage = 0[V];
    bus.valid = false;
    bus.mode = Mode.Off;
  }
  relation update at tick {
    next(bus.voltage) = pre(bus.voltage) + config.gain * 1[V];
    next(bus.valid) = true;
    next(bus.mode) = config.mode;
  }
}
model Example() {
  instance controller: Controller(config = Config(mode = Mode.On, gain = 2));
}
```

A constructor supplies every member exactly once by name. Declaration order determines
canonical member order, independently of argument order. Static record Parameters may
forward another Parameter of the exact record type. Derived numeric members retain their
original Parameter expressions and differentiation dependence. Model-owned record defaults
create ordinary independently editable Parameter members, like scalar model defaults.

A Field bus shares one role and temporal owner. A sampled bus names one explicit clock
for all members. Member selection preserves that ownership: `pre`, `next` and initial equations follow the ordinary Field rules. Members
cannot carry individual clock clauses, and wrong-clock updates reject before execution.
Boolean, enum and integer members remain discrete. Selecting a real or complex member uses
its ordinary mathematical type; the heterogeneous record itself is not a numeric vector.

Components can exchange bus members through ordinary borrowed Field slots. For example,
`instance control: BusController(tick = tick, voltage = sensor.voltage, valid = sensor.valid,
drive = command.drive)` binds a controller directly to existing sensor and command bus
members. Its signature declares those scalar Fields at its borrowed `tick`. Both producers
and consumers retain the same exact member identities; no copy or clock conversion occurs.
The controller can read the previous sensor sample with `pre(voltage)` and write its command
with `next(drive)`. Binding an expression, a missing member, a different mathematical type,
or a member owned by a different clock is rejected. This explicit member route does not
introduce a record-valued signal Port.

Closed records are available through source and `Module.record(name, members=...)` in Python.
Python Field and Parameter handles expose `.member(name)`, and a record descriptor constructs
values with named arguments. Current Model artifacts retain the record declaration and its
ordered typed member expressions across replay and execution-session restart. The current
surface covers flat records with support-independent mathematical member types and owned
Field buses. Invariant arrays retain their declared shape. Support-dependent spatial vector
or tensor declarations, nested records, record arrays, record-valued signal Ports and payload
variants are not admitted.

## Exact integers

`integer` is a signed exact 64-bit value, ranging from -9223372036854775808 through
9223372036854775807. Checked addition, subtraction, multiplication, and negation reject overflow;
they do not wrap or saturate. Decimal integer literal parsing never passes through binary64.
A leading sign is applied before checking the signed literal's range, so the minimum value
is representable without admitting its positive magnitude as an integer value.

```eqiora
parameter particles: integer = 9007199254740993;
let adjacent: integer = particles + 1;
```

`adjacent` is exactly 9007199254740994. Integer arithmetic receives integer operands; there is
no implicit integer/real promotion in an expression. A literal can use a uniquely required
integer context. Without that context ordinary numeric expressions retain the real-scalar
default, so `1/2` in a real value expression is one half, not integer zero.

`quotient(a, b)` truncates toward zero; `remainder(a, b)` satisfies
`a = b*quotient(a,b) + remainder(a,b)` and has the dividend's sign when nonzero.
For example, `quotient(-7, 3) = -2` and `remainder(-7, 3) = -1`.
A zero divisor rejects. Minimum-integer divided by -1 rejects overflow. Ordinary `/` does
not silently select this integer quotient operation; explicitly convert to real for a real ratio.

`to_real(n)` explicitly converts an integer to a dimensionless real scalar. The canonical
binary64 literal/value boundary rounds once to nearest, ties to even, and may lose integer
precision. In particular converting 9007199254740993 produces 9007199254740992; preserving
an exact count requires retaining the integer type. `to_integer(x)` accepts only a finite,
dimensionless real scalar with integral value inside the exact integer range. It does not
round a fractional value or recover precision already lost by a previous conversion.

`index<set>` is a bounded index tied to an exact declared finite index set; it is not an
interchangeable integer from another set. Its checked constructor is `index(set, integer)`.
Physical counts retain their separate species/count contract and nonnegative range; an
integer representation alone supplies neither species identity nor a count-to-amount law.
Runtime counts cannot determine model extent. Integers, indexes, Booleans, and enums have
no ordinary continuous derivative, even if an integer value happens to remain constant.

### Finite index sets and component families

`indexset Stages = range(n);` declares the ordered indexes from zero through `n - 1`.
The extent is a positive exact static integer and obeys the existing expansion budget before
allocation. A body declaration may depend on its selected static parameters; runtime State
cannot determine it. Distinct declarations retain distinct nominal identity even when their
extents agree. Duplicate names and cyclic static dependencies reject.

```eqiora
indexset Stages = range(3);
instance cell[i in Stages]: Cell(value = to_real(ordinal(i)));
```

The binder has type `index<Stages>` and is scoped to the instance's arguments. It cannot
silently shadow another declaration. `ordinal(i)` explicitly projects its exact integer
position; ordinary integer arithmetic does not operate on nominal indexes implicitly.
`cell[index(Stages, 0)].output` selects the indexed instance's ordinary member. A foreign
same-sized set or an out-of-range ordinal rejects. Boundary Relation families keep their
separate exact boundary-member meaning.

Expansion produces ordinary fixed instances in declared index order. It does not introduce
a runtime loop or resize the Model. An edit to a static parameter that would invalidate
elaborated structure rejects; changing the structure requires compilation with new bindings.
Array expressions are immutable. `array<V, n>` uses an exact static integer extent;
`values[lower:upper]` selects a nonempty half-open interval within that extent. Both bounds
are explicit static integers. Steps, clipping, wrapping and runtime indexing reject.

Each instance family admits one local binder. Nested families and child IndexSets may depend
on exact Parameter values, including the enclosing member's ordinal. Every actual context is
specialized and resource-checked before occurrence allocation; selected Components use the
same specialization. Unselected generic definitions retain symbolic validation, without
invented extents. Explicit and indexed descriptions can be compared by their mathematical
equations and occurrence structure; their distinct authored Source need not have equal bytes.

### Indexed equations and connections

Models and Components may expand Relations and ordinary connections over an exact IndexSet.
The binder follows the Relation name or connection kind, before any support or clock clause:

```eqiora
relation drive[i in Stages] at tick {
  driver[index(Stages, ordinal(i))].y = ordinal(i) + 1;
}
connect [i in Stages]
  driver[index(Stages, ordinal(i))].y -> cell[index(Stages, ordinal(i))].u;
```

Each member produces ordinary equations and connections with its own exact occurrence.
A Relation keeps its declared support and activation; sharing a period does not equate
separately declared clocks. The binder is local to the family and cannot capture a declaration.
Boundary families retain their distinct exact boundary-member selectors and support rules.

A conserving connection family can join adjacent component instances:

```eqiora
indexset Stages = range(3);
indexset Links = range(2);
instance cell[i in Stages]: Resistor(resistance = 2[Ohm]);
connect [j in Links]
  cell[index(Stages, ordinal(j))].negative,
  cell[index(Stages, ordinal(j) + 1)].positive;
```

`j` belongs to `Links`. The explicit `index(Stages, ...)` constructor selects a member of
`Stages`; equal extents do not make the sets interchangeable. Every neighbor must be in range.
There is no wrapping, clipping, runtime topology change or implicit broadcast. Family extent,
expanded work and neighbor selection are checked before materializing the family.

The existing physical connection owner derives conservation constraints. Resistor laws remain
owned by their component occurrences. With a 12 V source across these three 2 ohm resistors,
the ordinary DC equations give 2 A through the chain and a 4 V drop across each resistor.
Indexed and explicit forms can be compared by their equations and exact endpoint sets after
mapping explicit instance names to ordinals; whole authored Model identities need not agree.

Python consumes this source through `eqiora.compile(source=...)` or a source file. Python
Source currently exposes exact IndexSet handles and finite reduction callbacks; it does not
provide new indexed instance/connection handles in this profile.

### Finite scalar reductions

A reduction binds one exact finite IndexSet over its expression:

```eqiora
model Polynomial() {
  indexset Terms = range(3);
  variable result: integer;
  relation value {
    result = sum((ordinal(i) + 1) * (ordinal(i) + 1), over = (i in Terms));
  }
}
```

The terms are 1, 4 and 9, so the result is 14. `product(expression, over = (i in Terms))`
uses the same binder grammar. Both expand in ascending ordinal order into a left-associated
chain of ordinary additions or multiplications. The first term starts the chain; no numeric
identity is inserted. Extents must be positive, so empty sums and products reject with the
existing empty-IndexSet policy.

Bodies admit ordinary integer, real and complex scalars. A sum preserves the element's
unit; a product over `n` equal-dimensional terms multiplies that dimension `n` times.
For example, three factors of `2 [m]` have product `8 [m^3]`. Boolean, nominal, spatial
and channel-array results are outside this scalar reduction profile. Complex typing does
not establish complex numerical execution.

`min(expression, over = (i in Terms))` and `max(...)` use the same finite binder
and ascending left fold. They admit ordinary integer or real scalars with identical complete
types and preserve the element's dimensions. The first element initializes the fold; empty
sets reject. Comparisons retain exact integer values, including adjacent values above 2^53.
On an exact tie the earlier element wins. Every term is evaluated, so an error in a term
cannot be hidden by an earlier winner. Mixed integer/real types, complex values, Booleans,
nominal indexes/counts and shaped values reject.

Finite extrema retain live Parameter dependencies and lower to ordinary comparison and
selection nodes. Exact sampled values and admitted real equations use the existing
execution profiles. The [typed point derivative profile](conditionals.md) admits real
extrema away from demanded ties and rejects at a tie; the untyped numerical SSA entry
still requires typed execution for selection. These reductions add no smoothing or
general nonsmooth solver.

The binder has the exact set's nominal index type. Use `ordinal(i)` for integer arithmetic
or channel indexing. Nested reductions use distinct lexical binder names and cannot capture
another declaration. Expansion charges the body size and nested extent products against the
existing resource bounds before allocation. Ordinary Parameter references remain live expression
dependencies; structural extent dependencies retain the existing edit restrictions.

Reduction extents must be resolved during definition checking. A selected local Model uses
its supplied static Parameter bindings before that check. Components reached through authored
Model instances use each concrete static binding context, including nested parameter forwarding.
Every context must type-check; one valid instance does not excuse an invalid second instance.
Dimensioned products must match the declared result dimension in each context. The existing
physical contract projection must agree across these specialized contexts; context-dependent
physical topology is not admitted by this reduction-specialization path. Uninstantiated
unresolved-reduction definitions still receive symbolic validation and reject; standalone selected
Component specialization and symbolic product-dimension inference remain outside this profile.

Reductions are admitted in Relations and runtime expression aliases. Parameter defaults and
IndexSet extent definitions cannot contain reductions in this profile. Runtime-sized
reductions and tensor contractions remain separate capabilities. Indexed
equation and connection families follow the rules above.

### Nominal particle counts

The [finite-space owner](finite-spaces.md#exact-counts-and-signed-changes) supplies ordered
species identity. `counts<Species>` is a vector of nonnegative particle cardinalities, each
bounded by the maximum signed integer. It is distinct from integer channels, signed component
changes and amount concentrations. Integer storage alone does not supply a species or units
conversion. A signed stoichiometric change uses `coordinates<integer, Species>`; adding it
to counts requires the exact same Space and rejects any component underflow or overflow.
Every component and the enclosing tick are validated before committing state or outputs.

## Boolean predicates

`bool` is a dimensionless invariant scalar with the values `true` and `false`.
It has no numerical zero, arithmetic, implicit numeric conversion, or continuous derivative.
The current scalar profile rejects Boolean arrays and spatial vectors.

`==` and `!=` produce Boolean values. Ordinary integers compare exactly, including adjacent
values above 2^53; real and complex scalars may share equality at compatible dimensions.
`<`, `<=`, `>`, and `>=` admit dimension-compatible real scalars or ordinary exact integers
of the same domain. Complex and Boolean ordering reject. Nominal indexes support equality
only within the same declared set; use `ordinal` explicitly for integer ordering. Counts,
coordinates, and channel arrays have no comparison in this bounded profile.

`not`, `and`, and `or` require Boolean operands. Conjunction evaluates its right operand only
when its left value is true; disjunction does so only when its left value is false. Both sides
must still pass static type, support, and activation checks. Sharing a skipped expression
with a separately requested output does not exempt that output from evaluation.

A Relation equality remains `left = right`; it does not become the predicate `left == right`.
Direct Boolean initialization and sampled assignments use the existing typed state and output
owners. Conditional values and bounded batches use the separate
[scalar branch profile](conditionals.md). A Boolean implicit solver, predicate derivatives
and automatic crossing events remain outside these profiles.

## Real elementary functions

Except where stated otherwise, the following functions require real dimensionless scalar
arguments and return a real dimensionless scalar. First and second derivatives are admitted
only on the open smooth domain in the table. Value admission at an endpoint does not grant
a derivative there. Typed vectorized evaluation follows the same component domains, without
changing branch decisions or introducing broadcasting.

| Operation | Value domain and result rule | Derivative boundary |
|---|---|---|
| `+`, `-` | Compatible dimensions; preserve dimension | Smooth on finite admitted values |
| `*` | Multiply dimensions | Smooth on finite admitted values |
| `/` | Nonzero denominator; divide dimensions | Singular at zero denominator |
| `math.sqrt(x)` | Real `x >= 0`; halve exact dimension exponents | Reject ordinary derivatives at zero |
| `math.exp(x)`, `math.expm1(x)` | All finite real inputs; reject nonfinite output | Smooth where numerical evaluation is admitted |
| `math.log(x)` | `x > 0` | Zero and negative arguments reject |
| `math.log1p(x)` | `x > -1` | `x <= -1` rejects |
| `math.sin(x)`, `math.cos(x)` | Dimensionless radians | Smooth |
| `math.tan(x)` | Exclude odd multiples of `pi/2` | Poles reject |
| `math.asin(x)`, `math.acos(x)` | `-1 <= x <= 1` | Derivatives require `-1 < x < 1` |
| `math.atan(x)` | All finite real inputs | Smooth |
| `math.atan2(y, x)` | Equal dimensions, not both zero; result in `(-pi, pi]` | Reject at the branch cut and origin |
| `math.sinh(x)`, `math.cosh(x)`, `math.tanh(x)` | All finite real inputs; reject nonfinite output | Smooth where numerical evaluation is admitted |
| `math.asinh(x)` | All finite real inputs | Smooth |
| `math.acosh(x)` | `x >= 1` | Derivatives require `x > 1` |
| `math.atanh(x)` | `-1 < x < 1` | Endpoints and outside reject |

`math.pi` is the dimensionless circle constant. Trigonometric arguments do not infer degrees
from magnitude or a variable name. At a negative `x` with zero `y`, `atan2(y,x)` selects
`pi`; signed numerical zero does not select a second mathematical branch. This value
convention does not make the derivative continuous across the cut.

`expm1(x)` means `exp(x)-1` mathematically and `log1p(x)` means `log(1+x)`, but their numerical
implementations use the appropriate cancellation-resistant kernels. They are not required
to reproduce the rounding error of a literal subtraction or addition in another expression.
Invalid real arguments never silently promote to complex. Complex functions retain their
separately admitted branch and real-linear derivative rules.

### Current real/complex scalar evaluation

Source constant folding and the typed reference evaluator share `sin`, `cos`, `exp`,
`log`, `sqrt`, `conj`, `real`, `imag`, `abs`, `abs2` and `arg` for invariant numeric
scalars. This profile does not establish complex Plan/Run execution or derivatives of
these complex functions. Scalar Operator IR retains its separately admitted real
`sin`, `sqrt` and `abs` paths; other functions reject there until explicitly admitted.

`conj` preserves the mathematical domain and dimension. `real` and `imag` explicitly
return real scalars with the input dimension; `abs` returns real magnitude with that
dimension, and `abs2` returns real squared magnitude with twice the dimension exponents.
`arg` returns a dimensionless real angle in `(-pi, pi]` and rejects zero amplitude.
It accepts dimensioned input without changing its physical meaning.

`sin`, `cos`, `exp` and `log` require dimensionless input. Real `sqrt` requires a
nonnegative input, and real `log` requires a positive input; neither promotes an invalid
real value to complex. Complex `log` rejects zero and selects the principal argument
in `(-pi, pi]`. Complex `sqrt` selects the root with nonnegative real part and, on the
negative real axis, nonnegative imaginary part. Signed zero is normalized before branch
selection. Every result component must be finite. These value conventions confer no
branch-cut, zero-root or complex differentiation capability.

## Value powers

Dimension exponents remain exact rational arithmetic. Value powers additionally check numerical
domain. A dimensioned base requires a statically exact reduced rational exponent `p/q`; a
runtime exponent requires a dimensionless base. Rational syntax in a dimensional power must
retain the fraction before numerical rounding, so a literal `1/3` cannot become an approximate
dimension exponent.

An integer power admits a negative real base. Zero with a positive exponent is zero; zero
with a negative exponent rejects; `0^0` rejects. For a rational exponent with denominator
greater than one, the initial real profile requires a nonnegative base. A negative real base
rejects even for an odd denominator: no alternate real-root branch is inferred. To request
a complex principal power, explicitly construct a complex base and use its admitted profile.

A noninteger runtime real exponent requires a positive base. Derivatives divide dimensions
according to the exact exponent and reject singular or unadmitted endpoint behavior; a zero
result alone is not proof that the derivative exists. Polynomial integer powers remain smooth
at zero where their actual derivative formula is defined.

## Memoryless nonsmooth operations

These operations use the same lazy branch owner as `if ... then ... else ...`. Their real
arguments must have compatible dimensions and shapes. No numerical tolerance changes ties.

| Operation | Exact value rule | Ordinary derivative rule |
|---|---|---|
| `math.abs(x)` | `x` for `x >= 0`, otherwise `-x`; preserve dimension | Reject at zero |
| `math.min(a,b)` | Lower value; choose first operand at equality | Reject at a tie unless the admitted expression proof establishes smoothness |
| `math.max(a,b)` | Higher value; choose first operand at equality | Same tie rule |
| `math.clamp(x, lower, upper)` | Require `lower <= upper`; return the bounded value | Reject at switching boundaries; no implicit smoothing |
| `math.sign(x)` | Dimensionless -1, 0, or 1 according to the exact sign | Reject at zero |
| `math.step(x)` | Dimensionless zero for `x < 0`, one for `x >= 0` | Reject at zero |

Complex magnitude and squared magnitude use their explicit complex operations; they do not
inherit real ordering. A smoothness assertion supplied by a caller is not a proof that a tie
can be differentiated. A method needing generalized derivatives or crossing events must
request those separately.

## Error and verification boundaries

All operations retain source-local diagnostics for wrong dimensions/domains, overflow,
nonfinite output, unsupported derivatives, and invalid conversions. Never clip a pole,
replace a singular inverse, add a positivity floor, or silently switch branches to make a
numerical evaluation succeed. Mathematical domain and numerical representability are checked
separately; execution precision is not a new mathematical scalar domain.

Useful independent reference points include `sqrt(4)=2`, `sqrt'(4)=1/4`, `exp(0)=1`,
`expm1(0)=0`, `log(1)=0`, `log1p'(0)=1`, `sin(0)=0`, `cos(0)=1`, and
`atan2(1,1)=pi/4`. Boundary tests must also reach rejected domains and inactive conditional
branches. Small-argument accuracy tests for `expm1` and `log1p` require a separately justified
error bound, not comparison against the cancellation-prone expression they are meant to replace.
