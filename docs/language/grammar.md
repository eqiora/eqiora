# Shared grammar productions

This grammar goes with the [core language rules](core.md). Square brackets mark optional parts,
braces mark repetition, quoted text is literal source, and `|` separates alternatives. Scope and
type rules restrict what the grammar allows, so source that parses is not necessarily valid.
Specialized constructs are defined on their own pages, linked below.

```text
qualified-name = identifier {"." identifier}
import = "import" qualified-name "as" identifier ";"
annotation = "@{" notation-ast "}"
dimension-declaration = ["public"] "dimension" identifier [annotation] "=" dimension ";"
value-head = value-kind identifier [annotation] [":" type] ["on" qualified-name]
             ["at" qualified-name]
value-kind = "parameter" | "variable" | "state" | "let" | "input" | "output"
           | "port" | "observable" | "property"
signature = "(" [signature-entry {"," signature-entry}] ")"
signature-entry = value-head ["=" expression] | support-requirement | clock-requirement
support-requirement = "support" identifier [annotation] ":" support-contract
clock-requirement = "clock" identifier [annotation] ":" "periodic"
container = ["public"] ("component" | "model") identifier [annotation] signature body
body = "{" {body-item} "}"
value-declaration = value-head ["=" expression] ";"
event-declaration = "event" identifier "=" "crossing" "(" expression ","
                    "direction" "=" crossing-direction ")" ";"
crossing-direction = "any" | "rising" | "falling"
indexset-declaration = "indexset" identifier [annotation] "=" "range" "(" static-extent ")" ";"
index-family = "[" identifier "in" qualified-name "]"
instance = "instance" identifier [annotation] [index-family] ":" qualified-name "("
           [named-argument {"," named-argument}] ")" ";"
named-argument = identifier "=" expression-or-exact-reference
initial = "initial" "{" {equation} "}"
boundary-family = "[" identifier "in" qualified-name "]"
relation = "relation" identifier [annotation] [index-family] ["on" qualified-name]
           ["at" qualified-name] "{" {equation} "}"
equation = expression "=" expression ";"
scalar-connector = ["public"] "connector" identifier "{"
                   "across" identifier ":" type ";"
                   "through" identifier ":" type ";" "}"
field-connector = ["public"] "connector" identifier "{"
                  "trace" identifier ":" dimension ";"
                  "flux" identifier ":" dimension ";"
                  "shape" connector-shape ";" "frame" connector-frame ";"
                  "pairing" "euclidean_boundary_duality" ";"
                  "orientation" "parent_outward" ";" "}"
connector-port = "port" identifier ":" qualified-name ["over" qualified-name] ";"
connector-shape = "scalar" | "spatial_vector" | "[" positive-integer {"," positive-integer} "]"
connector-frame = "invariant" | "spatial"
connection = "connect" [index-family] expression "->" expression {"," expression} ";"
           | "connect" [index-family] expression "," expression {"," expression} ";"
           | "connect" "periodic" qualified-name "," qualified-name ";"
record = ["public"] "record" identifier [annotation] "{" record-member {"," record-member} [","] "}"
record-member = identifier [annotation] ":" type
operator = ["public"] "operator" identifier [annotation] signature ":" type "=" expression ";"
```

An operator signature may contain only pure inputs and compile-time values. Component and Model
signatures accept the roles listed in the core rules. A reference argument keeps its role: the
grammar does not let a support, clock, property, or state be used as a number. Instance
arguments are always named, and the caller never repeats the argument's category.

A Relation family ranges over an IndexSet or over a Component's complete exterior boundary. A
boundary family comes after the notation and before `on`, and `on` must name the boundary member
the family binds. Each Relation in the family keeps its ordinary support and activation; the
[core family rules](core.md#boundary-relation-families) define scope and expansion. An instance
family ranges over a finite index set instead, and its index cannot be used as a boundary member.
The [numeric catalog](numeric-catalog.md) defines index sets, their sizes, and indexed member
references.

Events are private to their Model or Component and follow the [crossing-event rules](events.md).
The name after `at` in a Relation refers to a declared clock or event; the parser does not decide
which. The guard's type comes from its expression.

Imports come before declarations. A library file may contain declarations without any Model; a
run names its Model explicitly. The order of top-level declarations does not affect name
resolution, and a component cannot redefine an imported equation. Module cycles and the set of
reachable modules are handled by the package resolver; importing never runs code.

```text
scalar-type = dimension | "complex" "<" dimension ">"
type = scalar-type | "bool" | "integer" | "index" "<" qualified-name ">"
     | "vector" "<" scalar-type "," static-extent ">"
     | "tensor" "<" scalar-type "," static-extent "," static-extent
       {"," static-extent} ">"
     | "array" "<" type "," static-extent ">"
     | coordinate-type | map-type | "counts" "<" qualified-name ">" | qualified-nominal-type
dimension = dimension-product
dimension-product = dimension-power {("*" | "/") dimension-power}
dimension-power = dimension-atom ["^" dimension-exponent]
dimension-atom = "1" | qualified-dimension-name | "(" dimension ")"
dimension-exponent = signed-integer | "(" signed-integer "/" positive-integer ")"
quantity = number "[" unit-expression "]"
```

Unit expressions use the same product and power syntax, but names in them are looked up in the
unit catalog rather than among dimensions or values. A static extent is a positive integer
expression known at compile time; anything that depends on run-time state cannot be an extent.
Once a number is followed by `[`, the parser treats the brackets as units and reports an error if
they are invalid, rather than trying indexing instead. Integer expressions inside type arguments
cannot contain comparisons or Boolean operators, so a closing `>` is never ambiguous.

```text
expression = conditional
conditional = "if" expression "then" expression "else" conditional | disjunction
disjunction = conjunction {"or" conjunction}
conjunction = negation {"and" negation}
negation = "not" negation | comparison
comparison = additive [("<" | "<=" | ">" | ">=" | "==" | "!=") additive]
additive = multiplicative {("+" | "-") multiplicative}
multiplicative = signed-power {("*" | "/") signed-power}
signed-power = ("+" | "-") signed-power | power
power = postfix ["^" signed-power]
postfix = primary {call-arguments | "[" expression "]" | "." identifier}
primary = quantity | number | qualified-name | "true" | "false" | finite-reduction
        | "(" expression ")" | "[" [expression {"," expression}] "]"
finite-reduction = ("sum" | "product" | "min" | "max") "(" expression "," "over" "="
                   "(" identifier "in" qualified-name ")" ")"
call-arguments = "(" [argument {"," argument}] ")"
argument = expression | named-argument | typed-structural-argument
```

Special arguments such as `holding`, contraction axes, coordinate assignments, and the binder of a
finite sum each follow the grammar of their own operation. They are not general assignments or
free-form keyword arguments. A call cannot mix positional and named arguments or bind the same
parameter twice. User-defined operators take named inputs; built-in operators take the
positional and named arguments documented for them. The recursive `power` rule makes `^`
right-associative and allows a signed exponent.

Specialized syntax is defined in [properties](properties.md), [finite spaces](finite-spaces.md),
[coordinates](coordinates.md), [calculus](calculus.md), [harmonic forms](harmonic-rc.md),
[eigenpairs](wavefunction.md), [variations](phase-separation.md), and
[stochastic Laws](stochastic.md). The [resource limits](resources.md) are checked before any
recursion or expansion.

Conditional expressions follow the [conditional value rules](conditionals.md). Both branches of
`if ... then ... else ...` are type-checked, and only one is evaluated at run time. Put a
conditional in parentheses when it is an operand of arithmetic or `not`.

Typed operator definitions use the [operator profile](operators.md):

```text
operator-declaration = ["public"] "operator" name "(" formal {"," formal} ")"
                       ":" operator-type "=" expression ";"
formal = "input" name ":" operator-type
operator-type = "scalar" | "spatial" "[" integer "]" | value-type
named-call = qualified-name "(" name "=" expression {"," name "=" expression} ")"
```

Calls to user-defined operators name every argument. The compiler checks that the body is pure,
that its dimensions are consistent, and that it stays within the resource limits.
