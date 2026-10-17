# Converged language: core rules

This specification defines the adopted target language. Its grammar and examples do not
claim current compiler acceptance: implementation lands in dependency-closed feature slices.
Delivered behavior is indexed in the [capability matrix](../capability-matrix.md).
The rules here supersede conflicting source sketches in older frontend discussions.

Specimens: [resistor divider](divider.md), [heated body](heated-body.md),
[sampled state](sampled-state.md), [data-backed property](data-backed-property.md),
[finite-state mathematics](finite-state.md), [harmonic RC response](harmonic-rc.md).
The [1D wavefunction](wavefunction.md) covers stationary normalization and complex time evolution.
The [Maxwell cavity](maxwell.md) covers vector evolution and oriented boundary traces.
The [stochastic specimen](stochastic.md) specifies explicit calculus and noise identity.
The [phase-separation specimen](phase-separation.md) specifies functional variations and mixed dynamics.
The [free-streaming specimen](free-streaming.md) closes a bounded position–velocity transport problem.
The [ion-transport specimen](ion-transport.md) combines species identity, molar flux, and Poisson coupling.

The [calculus and branching rules](calculus.md) include the foundation audit's explicit
partials, continuous time, second-order oscillator, and piecewise constitutive examples.
The [finite constraint rules](constraints.md) define equality, ordered inequality and
complementarity meaning separately from numerical enforcement.
The [coordinate and measure rules](coordinates.md) specify product supports and partial integrals.
The [tensor and local-map rules](tensors.md) include the rank-four constitutive specimen.
The [numeric catalog](numeric-catalog.md) defines exact integers and bounded scalar operations.
The [unit catalog](units.md) fixes admitted symbols, prefixes, and structural dimension aliases.

## Source and modules

A source file is UTF-8. Identifiers are case-sensitive. Whitespace separates tokens but does
not terminate a statement. Braces delimit bodies, semicolons terminate statements, and commas
separate signature entries and arguments. There is no implicit multiplication.

Identifiers match `[A-Za-z_][A-Za-z0-9_]*`; documentation may contain Unicode, while notation
uses the explicit symbol commands below. Decimal tokens require a digit before any fractional point and digits after it,
with an optional `e`/`E` exponent and signed integer exponent value. Signs remain operators.
The [grammar productions](grammar.md) collect the shared syntax; specialized child rules
are defined on their linked owner pages.

`//` introduces a line comment; `///` attaches documentation to a declaration. Declaration
notation uses `@{...}` immediately after the declaration name. Its contents are a bounded
notation AST, not executable source or arbitrary TeX. Neither documentation nor notation
introduces a mathematical value or changes name resolution.

### Declaration notation

```eqi
component Material(parameter viscosity @{\mu}: Pa * s) {
  state estimate @{\hat{x}}: 1;
  variable stress @{\sigma_{ij}}: Pa;
}
```

An island describes one symbol, optionally styled, accented, and intrinsically scripted.
The initial vocabulary is closed:

| Kind | Admitted spelling |
| --- | --- |
| Latin | `a`–`z`, `A`–`Z` |
| Greek | `\alpha`, `\beta`, `\gamma`, `\delta`, `\epsilon`, `\zeta`, `\eta`, `\theta`, `\iota`, `\kappa`, `\lambda`, `\mu`, `\nu`, `\xi`, `\omicron`, `\pi`, `\rho`, `\sigma`, `\tau`, `\upsilon`, `\phi`, `\chi`, `\psi`, `\omega`; uppercase uses the same names with an initial capital, such as `\Alpha` and `\Omega` |
| Greek variants | `\varepsilon`, `\vartheta`, `\varkappa`, `\varpi`, `\varrho`, `\varsigma`, `\varphi` |
| Distinguished symbols | `\hbar`, `\ell`, `\aleph` |
| Styles | `\mathrm{x}`, `\mathit{x}`, `\mathbf{x}`, `\mathsf{x}`, `\mathtt{x}`, `\mathcal{X}`, `\mathbb{R}` |
| Accents | `\hat{x}`, `\tilde{x}`, `\bar{x}`, `\vec{x}`, `\dot{x}`, `\ddot{x}` |
| Intrinsic scripts | At most one `_` and one `^` per symbol, in either input order; script contents admit symbols, digits `0`–`9`, and the marks below |
| Script marks | `\prime`, `\star`, `\top` (transpose), `\dagger`, `\plus`, `\minus`, `\pm`, `\mp` |

Canonical formatting emits the lower script before the upper script and braces every script.
Atoms in a script are separated by spaces: `\sigma_{ij}` becomes `\sigma_{i j}`.
This also keeps `\alpha i` distinct from an unknown command `\alphai`.
The shorthand `x''` becomes `x^{\prime \prime}`; script `*`, `+`, and `-` become their named
marks. Redundant grouping is removed. Other aliases, including `\bf`, `\boldsymbol`, and
`\overline`, are rejected rather than interpreted by a TeX engine.

Each complete island is limited to 1,024 UTF-8 bytes, 256 significant notation tokens,
eight recursive symbol/group levels, 32 script entries (including nested decorations/groups),
and 4,096 emitted bytes. Admission checks precede the corresponding allocation or recursive
descent. Errors retain the original island or offending command's exact byte range.
Text commands, macros, file operations, environments, layout commands, math delimiters, and
formula operators are not part of this algebra. Greek letters use the command table rather
than Unicode aliases.

These scripts are part of a declaration's chosen symbol: `x^{2}` does not square its value,
and `\sigma_{ij}` does not index a tensor. The mathematical type and expressions still own
those operations. Rendering from mathematical types, including default vector bold/arrow
styles, is separate; an explicit style records the author's override without invoking a renderer.

Rust authoring uses `Notation::parse("@{...}")` and a declaration's `with_notation` method.
Python uses the same native admission owner:

```python
source = eqiora.Module("main")
material = source.component("Material")
material.parameter("viscosity", value_type=eqiora.ValueType.real())
material.set_notation("viscosity", eqiora.lang.Notation(r"@{\mu}"))
```

`Module.set_notation` targets an existing top-level declaration; `Component.set_notation`
targets an existing declaration in that component or model. Emission freezes these metadata
with the source. Notation survives source formatting, declaration cloning and source-package
reopening. Editing it changes exact source bytes and source-bundle identity, but not the
compiler's semantic local-source identity, physical type checking, or structural model comparison.

Module identity is the exact package identity plus portable relative source path. Source
does not declare or rename its own module. Imports name the full target and require an
explicit local `as` binding. A local import name never changes the imported declaration's
identity. There are no wildcard imports, textual includes, ambient package aliases, inherited
equations, or equation overrides. Standard packages use the same manifest and exact lock
resolution as other packages.

The parser retains source ranges and invalid fragments. Compilation publishes no partial
Model when a source, type, or binding error remains.

### Occurrence labels

Compilation resolves declaration labels once over the complete Model. Two resistor
instances may both declare `resistance @{R}`: their labels become `R_{left}` and
`R_{right}`. Existing intrinsic scripts stay on the symbol; qualification extends
the same lower-script node rather than emitting a second subscript.

The resolver applies declared notation, explicit instance notation, the shortest
distinguishing instance-path suffix, then declaration, connector-role and family-member
qualifiers. If those still collide or exceed the generated-label budget, the exact
resolved identity supplies a total deterministic fallback. Collision checks include
LaTeX, MathML, Unicode, plain and speech profiles; losing boldface or a distinct Greek glyph must not
make two quantities indistinguishable. Adding occurrences can change labels.

```python
labels = model.notation_labels("plain")
for quantity in labels:
    print(quantity.selector, quantity.role, quantity.label)
detail = model.notation_labels("plain", identities=[labels[0].identity])
```

`QuantityLabel.identity` includes the full Model scope, declaration occurrence,
connector role and exact family member. A repeated identity has one label, and
subviews inherit it without recomputing collisions. Source definition and instance
spans remain available even for constant-substituted parameters and eliminated
public physical ports. A forwarded parameter's graph target can be shared while
its declaration occurrences remain distinct.

Rust callers inspect `CompiledModel::notation()` or `ModelDocument::notation()`;
`ModelNotation::view` selects existing entries and `NotationProfile` chooses their
projection. `NotationLabel` owns bounded structural qualification independently of
source admission: at most 512 generated nodes and 16,384 emitted bytes per label,
checked before output allocation. The source notation limits above are unchanged.

Source/package recompilation retains authored labels. Bare canonical Model artifacts
carry physical meaning, not presentation metadata, so reopening them produces
exact-Kernel-identity labels with no invented source locations. Value edits preserve
labels when the occurrence inventory is unchanged; structural edits rebuild their
complete identity-only catalog.

Selected `model` and `component` declarations can place authored `form` blocks
following their ordinary declarations. Both use the same test, trial, dimension
and correspondence checks; attaching a form preserves the original equations.
Python `Module.model(...).weak_form(...)` follows this path too. Product tests
cover finite Hermitian forms and scalar Q1 forms with complex boundary loads,
including selected parameter and Geometry bindings and offline package replay.
`harmonic_form(...)` uses the same Model/Component owner for explicit angular frequency,
negative-exponential peak convention, input excitations and original/amplitude Field mappings.
The admitted fixed-domain real LTI profile derives a separate complex problem while retaining
the original initial conditions. Common finite and scalar-Q1 Plans reconstruct real responses;
they do not claim arbitrary initial-value equivalence. See the [RC specimen](harmonic-rc.md)
and its bounded registered evidence for the precise execution boundary.

### Mathematical rendering

`model.render_equations("law", "mathml")` presents the retained left and right
sides of each equation in authored order. `model.render_formulations("latex")`
presents authored forms, including their exact test, trial and integration-support
references. Both use the same typed projection as declaration labels; gradient,
time derivative, power, array index and inner-product notation comes from semantic
operators, never from identifier spellings. Pure operators without a retained
source alias are named by their exact definition digest and keep argument order.

Each immutable `MathRendering` contains `text`, `plain`, `speech` and `references`.
Profiles are `latex`, `mathml`, `unicode`, `plain` and `speech`. MathML output is a
complete `<math>` element. If `used_fallback` is true, `text` is plain text instead
of rich markup. Renderers should display that text normally, not interpret it as
HTML or TeX. Generated output is bounded to 65,536 bytes per profile.

References retain the exact graph target, quantity role and declaration identities.
When several authored declarations share one physical target, all are retained and
the equation uses an identity label rather than choosing one alias. Intrinsic
prime or transpose marks remain part of a declaration label; an expression power
or index is a separate operation. `value_type.render("speech")` describes the
checked scalar domain, channel and spatial axes, frame, nominal basis and exact
rational dimensions independently of the symbol's style.

### Declaration documentation

Place a contiguous block of `///` lines immediately before a declaration or
signature entry. The first nonempty paragraph is its hover summary. Keep blank
paragraphs inside the block with an empty `///` line:

```eqiora
/// A decay component.
///
/// The rate is supplied by each instance.
```

The declaration follows the final comment line directly. An ordinary `//` line
remains a comment rather than declaration documentation.

A blank source line between the block and declaration leaves the documentation
unattached. A `///` after code is a trailing comment; a standalone block on the next
line starts fresh. `////` is an ordinary line comment. Documentation at the end of
a scope, or before syntax discarded during error recovery, remains unattached.

Documentation prose is limited to 16,384 UTF-8 bytes. Hover renders paragraphs and
emphasis; links, HTML, lists and code fences are displayed literally. Source ranges
refer to the original UTF-8 file, including the comment markers. Formatting a cloned
syntax node retains that provenance; parse the generated source for its new ranges.

Canonical formatting normalizes code while preserving comments with their owning
declarations and syntax gaps. Native document reconstruction retains comments on
the reused declarations. Detached documentation stays in its enclosing scope and
is separated from following declarations so recovery cannot attach it accidentally.
Editor symbols expose documentation for declarations and signature entries;
reference hover follows the existing compiler-resolved top-level declaration identity.

## Declaration heads

The shared order is:

```text
kind name [@{notation}] [: type] [on support] [at activation]
```

Brackets here mean optional grammar, not literal source brackets. Optional syntax does not
make every clause legal for every kind. The tables are closed; specialized children are typed
constructs, not arbitrary attributes.

| Kind | Type | `on` | `at` | Definition |
|---|---|---|---|---|
| `parameter` | Required | No | No | `= expression` in a body; optional default in a signature |
| `variable` | Required | Optional | Optional | No declaration initializer |
| `state` | Required | Optional | Optional | No declaration initializer |
| `let` | Optional assertion | Optional assertion | Optional assertion | Required `= expression` |
| `input` | Required | Optional | Optional | External causal input in a signature |
| `output` | Required | Optional | Optional | Owned causal output in a signature; equations in the body |
| `port` | Required connector type | Optional | Optional | Owned connector occurrence; role laws come from its connector |
| `clock` | `periodic` for a requirement | No | No | Signature requirement or concrete `= periodic(...)` in an owning scope |
| `support` | Required support contract | No | No | Signature requirement or exact derived product/boundary in a body |
| `observable` | Required | Optional assertion | Optional assertion | Derived `= expression`; no solve unknown |
| `test` | Required | Inferred from trial field | Inferred from trial field | `for field`, optional `in h1/hcurl/hdiv/l2`, and an optional `zero_on` boundary restriction |
| `property` requirement | Required exact contract reference | No | No | Signature requirement bound to an exact release |
| `coordinate` | Required coordinate dimension | Required | No | `from` exact coordinate factor; no initializer |
| `space` | None | No | No | `= orthonormal(...)` or `= product(...)` |
| `noise` | Required admitted noise contract | No in the scalar profile | No | Exact process-channel declaration |
| `amplitude` | Required complex value type | Inherited/asserted | No | Form child `for` an exact original value |

| Container or definition | Header after name/notation | Contents |
|---|---|---|
| `component`, `model` | Required parenthesized signature, including `()` | Braced private body |
| `connector` | No value type or support/activation clauses | Named typed member roles |
| `operator` | Parenthesized input signature, `: result-type` | `= expression;` |
| `property contract` | Parenthesized input signature, `: result-type` | Closed contract children |
| `property release` | `: qualified-contract` | Closed definition, validity, provenance, and license children |
| `relation` | Optional boundary-family binder, optional `on`, then optional `at` | Braced simultaneous equalities |
| Conservation `law` | Required `on support` | `storage`, `flux`, `source` children |
| Stochastic `law` | `for state` | `calculus`, `drift`, `diffusion` children |
| `form` | `for` exact Law or Relation | Typed tests, relations, or selected reduction children |
| `instance` | `: qualified-component(named-bindings)` | Semicolon; no overriding body |

Notation always follows the declared name, before its signature or type. Instance support
and clock requirements are named bindings; `on` or `at` does not override a component body.
Omitting an instance argument list is not another canonical spelling: use `Ground()`.

A property requirement binds a release, not the scalar obtained by evaluating that release.
The [property declaration rules](properties.md) define its contract and release children.
The contract owns the independent-variable signature and result type; each call supplies its
named inputs. State-dependent input does not make a pure property call mutable. Support and
activation come from the typed call arguments, subject to the contract; the release cannot
perform sampling, history updates, or support conversion implicitly.

`relation name [family] [on support] [at activation] { ... }` contains simultaneous equations.
The optional family uses the restricted boundary binder defined below.
`law name on support { ... }` contains one physical `flux`, one `source`, and optional
`storage` expression, each terminated by a semicolon. Omitting storage means a steady balance,
not inferred zero initial energy. Its fixed-domain convention is
`derivative(storage) + div(flux) = source`; moving-domain transport needs its own admitted
contract. `form name for law { ... }` owns mathematical trial/test roles and equations,
not mesh or solver configuration. These are closed typed children, not string-valued attributes.

Signature `variable` and `state` entries borrow exact external occurrences. They do not allocate
private unknowns or transfer ownership of state initialization and updates. A `variable`
requirement reads an exact unknown, including an owned state without changing its role; a `state`
requirement additionally requires state ownership and the exact declared activation. Definition
checking and nested forwarding use the declared requirement: binding a state to a `variable`
formal does not grant that formal time-derivative, `pre`, or `next` capability. Signature `input`
entries require a compatible driver; signature `output` entries expose values defined by the
body. A `port` exposes its connector's members and participates in typed connection equations.
An `on` or `at` clause on a port must satisfy that connector's admitted support and activation
contract. It cannot change its member roles.

An omitted `on` on a declared unknown means a lumped value, not an inferred spatial field.
An omitted `at` on a declared unknown means continuous time, permanently. `on` and `at` are
independent: a clocked spatial state and a continuous lumped state are both meaningful.
A parameter is fixed mathematical input; a spatially varying constitutive expression belongs
in a typed property, operator, or derived expression, not an implicitly mutable parameter.

A `let` derives its support and activation from its expression. Written clauses assert those
derived facts; they cannot sample, hold, broadcast, or relocate the expression. Parameter-only
aliases are static. A runtime-dependent alias cannot define an array extent, clock period,
package identity, or other static requirement.

An alias's inferred activation describes its declared runtime dependencies, before algebraic
simplification. Static inputs do not add an activation. A current state read contributes its
exact declared clock or continuous activation; `time` contributes continuous activation.
Combining dependencies on different nominal clocks, or continuous and clocked dependencies,
does not produce a single clock. `at clock_name` asserts that all runtime dependencies have
that exact clock. It rejects static expressions and mixed activations, even when two clocks
have equal periods. Omitting `at` leaves the inferred dependencies unchanged. A statically
selected output of an indexed instance family contributes its resolved output clock,
including when the enclosing component binds that clock separately at each occurrence.

This dependency assertion is separate from evolution-use requirements. Reading the current
value of a clocked state through an alias remains valid wherever the equivalent direct read
is valid, including a continuous equation. The assertion does not execute a transition or
restrict a retained current-value read to its update ticks. An alias containing `pre`, `next`,
or `derivative` still checks the operator's state role and initialization or relation context
at each use; a matching assertion cannot discharge those requirements. An event-local
alias with a real reset obligation retains its exact event context separately from the
continuous state declaration; [crossing events](events.md) define that bounded exception.
It does not retag a static expression or ordinary current-state read.

```eqiora
parameter viscosity @{\mu}: Pa * s = 1.002e-3;
variable pressure @{p}: Pa on fluid;
state temperature @{T}: K on solid;
state memory: V at control;
initial { memory = 0 [V]; }
```

`variable` introduces an algebraic unknown; `state` owns evolution or history. An `initial`
block contains simultaneous mathematical initialization equations. It is neither an ordered
assignment program nor a collection of solver guesses. Restart consumes accepted State and
does not execute initialization again. Numerical guesses, scales, time steps, and output
schedules belong to the numerical/execution interface.

Clock identity is nominal and exact. Two clocks with equal periods are not interchangeable.
An activation must resolve to an admitted typed clock or event. Crossing activation boundaries
requires an explicit admitted transition such as sample or hold.

`clock control = periodic(10 [ms], phase = 0 [s]);` creates a clock in its owning scope.
The first argument is a positive exact period; the optional nonnegative exact phase defaults
to zero. Ticks occur at `phase + k * period` for nonnegative integer `k`, measured from fresh
initialization's time origin. A delayed first tick uses an explicit positive phase. Initialization
precedes a tick at zero. `period(control)` is an exact static time quantity. A signature entry
`clock tick: periodic` borrows a supplied clock and does not create a second schedule.

At a tick, `pre(state)` reads committed pre-tick state and `next(state)` denotes the candidate
post-tick state. All same-clock update equations are solved together and committed atomically.
A rejected attempt changes no accepted state. An owned state with no admitted update retains
its committed value; it does not become zero or acquire another owner's update. Clocked inputs
and outputs exist at ticks, not as implicit continuous held signals.

The target has no source `field`, `field slot`, `as continuum`, postfix `shape`, explicit
`continuous`, or initialized unknown declaration. Kernel Fields and continuum semantics retain
their existing owners.

## Boundary Relation families

Inside a Component, `[member in exterior]` expands a Relation once per exact member of
that Component's `complete_exterior` support requirement. This retains the existing
[complete-exterior owner](../../rfcs/0041-complete-exterior-port-families.md) with the
converged header; it is not an array index or a runtime loop.

For example, this component applies the same prescribed temperature to every supplied
exterior member:

```eqiora
component PrescribedExteriorTemperature(
  support body: volume(ambient_dimension = 3),
  support exterior: complete_exterior(parent = body),
  variable temperature: K on body,
  parameter value: K
) {
  relation prescribed[face in exterior] on face {
    trace(temperature) = value;
  }
}
```

The signature borrows the temperature occurrence; the component adds boundary equations,
not another temperature unknown or an initialization rule. The caller supplies the exact
complete exterior. Missing, duplicate, or foreign-parent members reject before expansion;
neither mesh facets nor coordinate proximity select members. Member order is non-semantic.

The binder follows optional notation and precedes `on`. It is visible only in that Relation's
support clause and body, cannot shadow an existing declaration, and must be the Relation's
`on` support. Nested binders, arithmetic on members, subset selection, and families over a
parameter or singular boundary reject. The ordinary activation rules still apply: omitting
`at` means continuous, and a family does not admit an otherwise unsupported activation.

Expansion retains the exact member and source occurrence for diagnostics, then produces ordinary
Relations through the existing elaborator. Expanded members and expressions count toward the
source unit's shared budgets in the [resource profile](resources.md), and each generated Relation
obeys the per-Relation equality limit. Expansion must be bounded before allocation and publishes
no partial Model on failure.
The example specifies the new header; current parser acceptance arrives with its migration.

## Mathematical types

| Source type | Meaning |
|---|---|
| `V` | Real scalar with voltage dimension |
| `1` | Dimensionless real scalar |
| `m^(-1/2)` | Real scalar with an exact rational dimension exponent |
| `complex<V>` | Complex scalar with voltage dimension |
| `vector<V, 3>` | Three-component spatial vector |
| `tensor<complex<Pa>, 3, 3>` | Rank-two spatial tensor with complex pressure components |
| `array<V, 3>` | Three indexed voltage channels, not a spatial vector |

There is no redundant `real<...>` constructor. A vector or tensor contains a scalar type;
an array contains a checked element type and exact static extent. Spatial frame information
may be inferred only from a unique exact support contract. Without that context a frame
requirement must be supplied explicitly; an extent alone cannot select a frame.

Finite component spaces and maps use the [finite-space grammar](finite-spaces.md).
They are distinct from both arrays and
spatial vectors. Equal size does not permit substitution between two bases, between bases
and spatial frames, or between a basis and a coordinate domain.

Array indexes are zero-based exact integers. An extent is a positive static integer; zero-sized
arrays are rejected. Indexing preserves element type and does not choose a component basis or
perform a coordinate transformation. Expansion and element-count limits are checked before
allocation or elaboration. There is no implicit broadcasting, reshaping, or basis conversion.

An exact dimensionless integer Parameter may determine a channel extent:

```eqiora
public component Channels(parameter n: integer = 3, parameter data: array<integer, n>) {
  variable values: array<integer, n>;
  relation copy { values = data; }
}
```

Each occurrence specializes its signature and body from its own static bindings, including
nested declaration families and a Component selected directly for compilation. Argument
expressions resolve in the caller; defaults and type extents resolve in the child declaration.
Unbound sizes, cycles, nonintegral sizes and excessive expansion reject before elaboration.

`values[lower:upper]` selects an immutable half-open channel slice. Both bounds are explicit
exact static integers, with `0 <= lower < upper <= extent`; steps and empty slices are rejected.
The result retains the element type, dimension, support and ordinal order. A Parameter used
for a compiled shape, family, index or slice is a structural dependency: changing it requires
recompilation. Ordinary numerical Parameters remain editable. These dependencies survive
Model and Transaction replay.

A uniform spatial coefficient uses an explicit constructor:

```eqiora
model Coefficients() {
  domain body = box(0, 1, 0, 1);
  parameter K: tensor<1, 2, 2> =
    tensor_value(frame = body, components = [[2, 3], [5, 7]]);
  variable response: tensor<1, 2, 2> on body;
  relation assignment on body { response = K; }
}
```

Here `frame = body` references the model-global Cartesian frame through an exact
support. It supplies ambient dimension and frame context; it does not give the
uniform Parameter spatial support or create another Field. Cartesian boundaries
supply their ambient frame as well. Components retain their nested axis order,
with the final axis varying fastest. Channel arrays remain outer arrays of
explicitly constructed spatial values. Constructor components must be closed scalar
expressions; named model values, including Parameter aliases, reject. Arithmetic using an
already framed Parameter retains its ordinary expression graph.

A spatial Parameter requirement or contextual zero may infer its frame only when
one exact support context is available. Missing or ambiguous context requires an
explicit frame reference; matching extents alone cannot select a support.
Wrong ambient extents, foreign support references, implicit array-to-tensor
conversion, and incompatible component dimensions reject. This is value authoring
and replay, not tensor contraction, local-frame conversion, or complex execution.

Dimensions use exact reduced rational exponents of the SI base dimensions. Dimension aliases
are structural: they neither scale a value nor introduce nominal quantity identity. Rational
normalization has a positive denominator, coprime numerator and denominator, and one zero.
Precision, storage layout, and backend do not enter a mathematical scalar type.

Addition and equality require compatible dimensions, shapes, frames, supports, and activation.
Real-to-complex embedding preserves dimension and value. Complex-to-real conversion requires
an explicit mathematical projection. Ordered predicates require real scalars or exact ordinary integers of the same domain;
complex values can be compared for equality but not ordered.

A bare literal zero can take the numeric scalar domain, dimension, and shape uniquely required
by its context. It cannot stand for the Boolean value `false`. For example, `voltage = 0;` uses a voltage zero. An unconstrained zero is dimensionless
real scalar zero. Contextual zero does not create a frame, support, clock, or basis conversion.
A numeric initializer of an explicitly dimension-typed declaration uses that dimension's
coherent unit: `parameter density: kg / m ^ 3 = 1;` needs no repeated unit. This also
applies to a numeric parameter default or type-annotated `let`; it does not apply to a
nonzero literal in an initial equation.
Explicit input units still undergo conversion and dimension checking. This declaration-only
rule does not assign units to arbitrary expressions or instance arguments; nonzero literals
elsewhere remain dimensionless.

## Numbers, units, and brackets

A leading sign is an operator applied to a number; it is not part of the number token.
Decimal literals, with or without an exponent, keep their exact written value until they are
converted to a quantity. A number followed by `[`, with or without whitespace between them,
starts a unit annotation. Names inside the brackets are looked up in the unit catalog, never
among value names.

```eqiora
10 [ms]
210 [GPa]
998.2 [kg / m^3]
```

After a name, brackets mean indexing (`samples[2]`), and on their own in an expression they
form an array literal (`[a, b]`). A number cannot be indexed, so `10[2]` is an invalid unit
annotation. Multiplication must always be written with `*`.

Dimension and unit expressions use `*`, `/`, parentheses, and `^` with an exact exponent:

```text
exponent = signed-integer | "(" signed-integer "/" positive-integer ")"
```

So `m^(-1/2)` is a dimension, while `x^(1/2)` in a value expression is ordinary numerical
exponentiation. Dimension exponents are computed with exact rational arithmetic. Two dimensions
are equal when their reduced exponents are equal, however they were written.

Angle brackets belong to a type constructor only where a type is expected; in an expression they
are comparisons. Function calls never take angle-bracket arguments. This is how the parser tells
`complex<V>` from `a < b` without consulting the symbol table.

All scaled units convert through one catalog in the compiler. The exact decimal value is
multiplied by the exact unit scale and then rounded to binary64 once. The compiler rejects
overflow, non-finite results, and nonzero values that would round to zero. Clocks keep their times
as exact rationals instead. A fractional power of a scaled unit is allowed only when the root of
its scale is exact. Units with an offset, such as degrees Celsius, use the separate rules for
absolute and difference quantities.

## Expressions and equations

From strongest to weakest binding:

| Operation | Associativity |
|---|---|
| Parenthesized expression; call, member access, indexing | Postfix, left to right |
| `^` | Right |
| Unary `+`, `-` | Prefix; power binds inside its operand |
| `*`, `/` | Left |
| `+`, `-` | Left |
| `<`, `<=`, `>`, `>=`, `==`, `!=` | Non-associative |
| `not` | Prefix Boolean negation, below comparisons |
| `and` | Left, short-circuit |
| `or` | Left, short-circuit |
| `if predicate then value else value` | Right-nested conditional expression |

The exponent of `^` may carry a sign, so `-x^2` means `-(x^2)`, `x^-2` means `x^(-2)`, and
`x^y^z` means `x^(y^z)`. Comparisons cannot be chained: write `a < b and b < c`, because
`a < b < c` is an error.

`=` is not an expression operator. In a Relation or `initial` block, `lhs = rhs;` states one
equation of a simultaneous system. In a parameter default, alias definition, or named argument,
the surrounding construct gives `=` its meaning. `==` is a Boolean comparison and never states a
physical equation. Booleans do not convert to numbers.

Equations are not assignments, and the order in which they or their operands are written does
not imply an order of execution. Each Relation keeps both sides of every equation as written.
Boolean and integer equations are checked directly and must be simple assignments; no solver is
introduced for them. For numerical equations, the compiler forms the residual `lhs - rhs` after
checking both sides. When the right-hand side is a literal zero (possibly in parentheses or with
unary minus) and dropping it would not change the type or support of the residual, the zero is
omitted. A typed zero keeps its units, shape, frame, and scalar domain, so a complex zero still
forces a real left-hand side to become complex. Zeros that are computed, such as `0 * y`,
`y - y`, or a parameter whose value is zero, are not omitted.

A nonzero decimal that would underflow to zero in binary64, either as written or after unit
conversion, is an error. Exact zero (including `0e-999`) and representable subnormal numbers
are allowed. These checks happen before the literal-zero rule.

`math.i` is the dimensionless imaginary unit. `math.complex(real_part, imaginary_part)` combines
two real-valued operands with matching dimensions, shapes, frames and exact finite bases.
It preserves their component axes and constructs a complex value; there are no `i`/`j` literal
suffixes, and an imaginary part is never dropped silently. For example,
`math.complex(2 [V], 3 [V])` has type `complex<V>`.

`math.conj`, `math.real` and `math.imag` act on each component of scalars, channel arrays,
spatial vectors/tensors and finite-coordinate values. They preserve units, axes, frame and
basis identity; `real` and `imag` return real components. Conjugation neither transposes a
matrix nor exchanges the endpoints of a finite map.

In a declaration initializer, plain numbers inside an array literal or `math.complex` take the
declared dimension. `parameter channels: array<V, 2> = [1, 2];` and
`parameter voltage: complex<V> = math.complex(1, 2);` therefore both hold volts. Each array axis
must have exactly its declared length; the initializer never broadcasts a scalar or reshapes an
array. Numbers that carry their own units must have a compatible dimension, and arithmetic
inside the literal keeps its usual dimensions. This only applies to declaration initializers: in
other expressions and in instance bindings, both real-valued operands must already have the
same dimension and complete component type.

A dimensionally valid expression can still be outside a function's domain. Real square roots need
nonnegative arguments, and real logarithms need positive dimensionless arguments. Complex
square roots and logarithms use the principal branch with argument in `(-pi, pi]`, and the
logarithm is undefined at zero. Derivatives on a branch cut or at a singularity are checked
separately from value evaluation. A power of a dimensioned quantity needs a rational exponent
known at compile time; an exponent computed at run time requires a dimensionless base.

`inner(a, b)` conjugates its first argument; plain contraction does not. Transpose and adjoint
are different operations, and an adjoint that depends on a metric keeps that metric. Operators
are pure: they cannot perform I/O, call back into the host language, change history, or draw
random numbers. Unsupported value or derivative operations are reported as errors.

## Scope and resolution

A Component or Model signature lists everything a caller must supply. The body is private,
although tools can inspect it. All names in the signature, and then all names in the body, are
known before any default, type, or expression is resolved, so declarations can refer to names
that appear later in the file. The order of declarations does not determine evaluation order.

Duplicate names in one scope, cyclic aliases or defaults, recursive operator calls, and recursive
component expansion are errors. Equations that depend on each other are not a cycle in this
sense: coupled unknowns are what a simultaneous system is for. Whether such a system can
actually be solved is decided later, when a formulation and realization are chosen.

Instance arguments are written `name = value`. The signature says whether the name is a
parameter, support, clock, property, port, or reference, so the caller does not repeat it. Every
required input must be bound exactly once; only inputs with defaults may be left out. Ports and
outputs that the component itself creates are not passed in. Binding borrowed state passes the
caller's state through instead of creating a new one. Private members cannot be imported or
bound from outside.

Signature names and body names share one namespace, so a body cannot shadow a signature
parameter. An operator's parameters live in their own scope, separate from the caller's, and
substitution follows the bound declaration rather than the spelling of a parameter or import
alias.

A pure scalar operator is applied pointwise when all its arguments are defined on the same
support with the same activation. Compile-time scalar coefficients act as constants there;
a run-time value on a different support or clock cannot be used. Scalars are never broadcast to
vectors, product supports are never built implicitly, and values are never sampled or held
behind your back. The result has the same support and activation as its arguments.

## Diagnostics, recovery, and formatting

Syntax diagnostics use the codes in the [diagnostic registry](../diagnostics.md): `EQ0601` for
invalid tokens, `EQ0602` for grammar errors, and `EQ0603` for unresolved names or static types.
Each diagnostic points at the smallest offending UTF-8 byte range, and a duplicate binding also
points at the original one. An error at end of file uses the empty range at the end of the input.

After an error, the parser skips ahead while respecting nested brackets, so a malformed unit
annotation or type does not swallow the next declaration and later errors are still reported.
The [resource and diagnostic profile](resources.md) gives the size limits and examples of
rejection and recovery.

The formatter keeps the parsed mathematical structure, documentation, and comments. Running it
twice gives the same result, and it never changes how an expression groups, including around
unary minus, powers, and the two sides of an equation. Only whitespace and source positions
change. Source with errors still gets diagnostics and partial editor support, but it is never
formatted as if it were a valid Model.
