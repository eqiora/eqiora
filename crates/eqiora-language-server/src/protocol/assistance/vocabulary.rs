//! Help for source constructs, using the existing grammar and type contracts.
use super::Entry;
use lsp_types::CompletionItemKind;

pub(super) fn entries() -> Vec<Entry> {
    let mut entries = Vec::new();
    for (name, syntax, description) in [
        (
            "model",
            "model Name(...) { ... }",
            "Declare an executable model with an explicit signature and a private body. Even an empty signature uses (). The body owns fields, component instances and simultaneous Relations.\n\nExample: `model Decay() { state x: 1; initial { x = 1; } relation evolution { derivative(x) = -x / 1[s]; } }`",
        ),
        (
            "component",
            "component Name(...) { ... }",
            "Declare a reusable component with an explicit interface and a private body. Signature entries describe its parameters, inputs, outputs, ports, supports and clocks; an instance supplies named bindings.\n\nExample: `component Gain(parameter gain: 1, input x: V, output y: V) { relation law { y = gain * x; } }`",
        ),
        (
            "operator",
            "operator name(input x: Type, ...): ResultType = expression;",
            "Declare a pure mathematical operator with typed inputs and result. Calls bind inputs by name. Its body must fit the admitted pure calculus and cannot read hidden evolving state or perform I/O.\n\nExample: `operator squared(input x: 1): 1 = x * x;`",
        ),
        (
            "relation",
            "relation name [on support] [at activation] { lhs = rhs; }",
            "Declare simultaneous mathematical equalities. The equals sign expresses an equation, not an imperative assignment. Optional on and at clauses select the exact support and activation.\n\nExample: `relation ohm { voltage = resistance * current; }`",
        ),
        (
            "parameter",
            "parameter name: Type = expression;",
            "Declare a static typed parameter. A body parameter requires a value; a signature parameter may have a default or require a named binding. Parameters do not evolve as State fields.\n\nExample: `parameter resistance: Ohm = 10;`",
        ),
        (
            "variable",
            "variable name: Type [on support] [at activation];",
            "Declare an algebraic unknown determined by Relations. A variable has no declaration initializer; it is distinct from an evolving State.\n\nExample: `variable voltage: V;`",
        ),
        (
            "state",
            "state name: Type [on support] [at activation];",
            "Declare a State governed by continuous evolution, clocked updates or an admitted event reset. Supply mathematical initial conditions in an initial block; a State has no declaration initializer.\n\nExample: `state temperature: K; initial { temperature = 300[K]; }`",
        ),
        (
            "let",
            "let name [: Type] [on support] [at activation] = expression;",
            "Name a derived expression without adding a solve unknown or a new equation. An optional type, support or activation is an assertion about the inferred expression. Aliases do not introduce sampling or hold behavior.\n\nExample: `let area = width * height;`",
        ),
        (
            "input",
            "input name: Type [on support] [at activation]",
            "Declare a named pure-operator input or an external causal input in a Model/Component signature. Calls or instances supply its typed binding; it is not an imperative read operation.",
        ),
        (
            "output",
            "output name: Type [on support] [at activation]",
            "Declare an owned causal output in a Model/Component signature. Relations in the body determine its value; an output is not a function return statement.",
        ),
        (
            "initial",
            "initial { lhs = rhs; }",
            "State simultaneous mathematical initial conditions. Equations in this block are not sequential assignments. Use it for State initialization rather than adding an initializer to a State declaration.",
        ),
        (
            "instance",
            "instance name: Component(argument = value, ...);",
            "Create one occurrence of a Component with named interface bindings. Even an argument-free occurrence uses (). Bindings retain their parameter, support, clock, field or port roles.\n\nExample: `instance amplifier: Gain(gain = 2, x = voltage);`",
        ),
        (
            "import",
            "import package.module as alias;",
            "Import an explicitly named source module under a local alias. Access its public declarations through that alias. Imports use the resolved module/package graph and do not create ambient or transitive aliases.",
        ),
        (
            "as",
            "import package.module as alias;",
            "Choose the local alias for an explicit module import. The alias changes source spelling, not the identity of imported declarations.",
        ),
        (
            "public",
            "public component Name(...) { ... }",
            "Expose an admitted top-level declaration to importing modules. Its body members remain private; visibility does not bypass package resolution or type checks.",
        ),
        (
            "private",
            "private component Name(...) { ... }",
            "Keep an admitted top-level declaration local to its module. Other modules cannot access it through an import alias.",
        ),
        (
            "dimension",
            "dimension Name = dimension_expression;",
            "Declare a structural physical-dimension alias. Dimensions use exact rational exponents and are distinct from scaled input units.\n\nExample: `dimension Speed = m / s;`",
        ),
        (
            "enum",
            "enum Name { Member, ... }",
            "Declare a nonempty set of named values with nominal identity. Members are not integers and have no implicit numeric conversion. A case expression must cover every member exactly once.",
        ),
        (
            "record",
            "record Name { member: Type, ... }",
            "Declare a closed ordered set of typed members with nominal identity. A constructor supplies every member exactly once by name; identical member shapes do not make different Record declarations interchangeable.",
        ),
        (
            "indexset",
            "indexset Name = range(extent);",
            "Declare an exact finite index set with a static extent. Family binders and reductions retain this set's nominal ownership; equal extents alone do not identify two sets.",
        ),
        (
            "space",
            "space Name = orthonormal(extent);",
            "Declare a mathematical component space or an admitted product of spaces. Component bases are distinct from channel arrays, spatial frames and coordinate domains.",
        ),
        (
            "domain",
            "domain name = box(lower, upper);",
            "Declare an admitted spatial domain. Spatial fields and operations retain their exact support identity; a domain declaration does not itself choose a mesh or numerical solver.",
        ),
        (
            "support",
            "support name: support_contract",
            "Declare an exact support requirement in a signature or an admitted derived support in a body. A support identifies where a field or Relation lives; it is not an interchangeable geometric extent.",
        ),
        (
            "on",
            "variable name: Type on support;",
            "Attach a field or Relation to an exact spatial support where the construct admits it. On a let or observable, this asserts the expression's inferred support; it does not relocate a field.",
        ),
        (
            "at",
            "relation name at clock { ... }",
            "Select an exact admitted activation, such as a declared Clock or event. On an alias it asserts the inferred dependency profile; it does not implicitly sample a continuous value.",
        ),
        (
            "clock",
            "clock name = periodic(period, phase = offset);",
            "Declare a nominal Clock with an exact periodic schedule, or a periodic requirement in a signature. Equal numerical periods do not merge distinct Clock identities.\n\nExample: `clock tick = periodic(10[ms]);`",
        ),
        (
            "periodic",
            "periodic(period, phase = offset)",
            "Describe an exact periodic Clock schedule using durations. Phase is optional. Also appears in periodic Clock requirements and explicit periodic connections.\n\nExample: `clock tick = periodic(10[ms], phase = 0[s]);`",
        ),
        (
            "event",
            "event name = crossing(expression, direction = rising);",
            "Declare a zero-crossing event with an explicit admitted direction: any, rising or falling. Event Relations describe simultaneous accepted resets through pre and next; a conditional value alone does not create an event.",
        ),
        (
            "connector",
            "connector Name { across effort: Type; through flow: Type; }",
            "Declare named typed connection roles. Scalar connectors pair across equality with through conservation; field connectors retain explicit trace/flux, shape, frame, pairing and orientation contracts.",
        ),
        (
            "port",
            "port name: Connector;",
            "Declare an owned occurrence of a Connector interface. The connector determines its physical roles and connection laws; a conserving port is distinct from a causal input or output.",
        ),
        (
            "connect",
            "connect first, second;",
            "Connect compatible typed interfaces. Conserving connections impose role laws; causal connections use a directed source -> destination form. Connections do not erase support, clock or connector identity.",
        ),
        (
            "across",
            "across name: Type;",
            "Declare a scalar connector's effort-like role. A connection equates compatible across values, such as electrical voltage.",
        ),
        (
            "through",
            "through name: Type;",
            "Declare a scalar connector's flow-like role. A connection imposes the corresponding signed conservation law, such as conservation of electrical current.",
        ),
        (
            "observable",
            "observable name: Type = expression;",
            "Declare a typed derived quantity for observation. It adds no solve unknown; optional support or activation clauses assert the expression's inferred profile.",
        ),
        (
            "law",
            "law name on support { storage ...; flux ...; source ...; }",
            "Declare an admitted physical law. A fixed-domain conservation law retains storage, physical flux and source; a stochastic law for a State instead has explicit calculus, drift and diffusion children. Each profile has its own admission rules.",
        ),
        (
            "form",
            "form name for law { ... }",
            "Declare a mathematical form for an exact Law or Relation, retaining trial/test roles and admitted reduction children. A form is distinct from selecting a numerical Plan.",
        ),
        (
            "test",
            "test name: Type for field;",
            "Declare a mathematical test role associated with an exact trial field. Support and activation follow the trial field; zero_on can impose an admitted homogeneous boundary restriction.",
        ),
        (
            "property",
            "property contract Name(input x: Type): ResultType { ... }",
            "Declare a pure typed property contract or an exact property release. A signature requirement binds a release of the specified contract, not a scalar value returned by evaluating it.",
        ),
        (
            "material",
            "material composition Name { ... }",
            "Declare an admitted composition of exact property releases. Property identity, input types, validity and provenance remain explicit rather than being inferred from a material name.",
        ),
        (
            "if",
            "if predicate then value else value",
            "Choose a value using a Boolean predicate. Both branches must type-check, while only the selected branch is evaluated. This creates no event, history or imperative control-flow block.",
        ),
        (
            "then",
            "if predicate then value else value",
            "Introduce the value selected when an if predicate is true. Both value branches retain the same complete type, dimension, shape, support and activation requirements.",
        ),
        (
            "else",
            "if predicate then value else value",
            "Introduce the value selected when an if predicate is false. A conditional value requires this branch; an inactive branch is not evaluated.",
        ),
        (
            "case",
            "case value { Enum.Member => expression, ... }",
            "Select a value by an exact Enum member. Cover each member exactly once; wildcard arms, missing members and foreign nominal members reject.",
        ),
        (
            "and",
            "predicate and predicate",
            "Boolean conjunction with short-circuit evaluation. Booleans do not implicitly convert to dimensionless numeric values.",
        ),
        (
            "or",
            "predicate or predicate",
            "Boolean disjunction with short-circuit evaluation. Booleans do not implicitly convert to dimensionless numeric values.",
        ),
        (
            "not",
            "not predicate",
            "Boolean negation. It binds below comparisons and above and/or; parentheses can make the intended predicate grouping explicit.",
        ),
        (
            "true",
            "true: bool",
            "The Boolean true value. It is not the numeric value 1 and is not implicitly coerced to a number.",
        ),
        (
            "false",
            "false: bool",
            "The Boolean false value. It is not a special numeric zero and is not implicitly coerced to a number.",
        ),
    ] {
        entries.push(entry(
            name,
            syntax,
            description,
            CompletionItemKind::KEYWORD,
        ));
    }
    for (name, syntax, description) in [
        (
            "bool",
            "bool",
            "Exact Boolean scalar type. Values are true or false; there is no implicit numeric coercion.",
        ),
        (
            "integer",
            "integer",
            "Exact dimensionless integer type with checked arithmetic and explicit real conversion. A static integer Parameter can determine an admitted channel-array extent.",
        ),
        (
            "complex",
            "complex<Dimension>",
            "Complex scalar type with a physical dimension. Use math.complex(real_part, imaginary_part) for explicit construction; imaginary parts are never silently discarded.\n\nExample: `parameter voltage: complex<V> = math.complex(2, 3);`",
        ),
        (
            "array",
            "array<ElementType, extent>",
            "Indexed channel array with an exact positive static extent. A channel axis is not a spatial vector axis. Indexing is zero-based; no implicit broadcasting or reshaping is admitted.\n\nExample: `parameter channels: array<V, 3> = [1, 2, 3];`",
        ),
        (
            "vector",
            "vector<ScalarType, extent>",
            "Spatial vector type. Its spatial axis and frame are distinct from channel-array axes and nominal component bases. An extent alone cannot choose a frame; the enclosing support must determine it or require one explicitly.",
        ),
        (
            "tensor",
            "tensor<ScalarType, extent, ...>",
            "Spatial tensor type with a scalar element type and explicit spatial extents. Retains physical dimensions and frame requirements; it is distinct from a channel array.",
        ),
        (
            "scalar",
            "operator f(input x: scalar): scalar = expression;",
            "Dimension-polymorphic scalar class in the admitted generic pure-operator profile. It is not a synonym for dimensionless real type 1. Ordinary real scalar types are written as dimensions, such as V or m/s; there is no real<...> constructor.",
        ),
        (
            "spatial",
            "operator f(input x: spatial[rank]): spatial[rank] = expression;",
            "Generic spatial tensor class in the admitted pure-operator profile. Rank and exact calculus restrictions are checked by the compiler; this is distinct from an ordinary concrete field's physical type.",
        ),
    ] {
        entries.push(entry(name, syntax, description, CompletionItemKind::CLASS));
    }
    entries
}

fn entry(name: &str, syntax: &str, description: &str, kind: CompletionItemKind) -> Entry {
    Entry {
        name: name.into(),
        label: syntax.into(),
        documentation: Some(description.into()),
        kind,
        parameters: None,
    }
}
