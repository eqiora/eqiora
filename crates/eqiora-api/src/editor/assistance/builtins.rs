//! Editor prose for the currently admitted source vocabulary. These descriptions
//! do not define typing or evaluation; those remain owned by the compiler.
use super::{EditorSymbol, EditorSymbolKind as Kind, documented};

pub(super) fn entries() -> Vec<EditorSymbol> {
    let mut entries = vec![
        function(
            "math.sin",
            "math.sin(x)",
            "Sine of a dimensionless real scalar, with the argument in radians. Returns a dimensionless scalar.\n\nExample: `math.sin(math.pi / 2)`.",
            &[("x", "Dimensionless real scalar angle in radians.")],
        ),
        function(
            "math.sqrt",
            "math.sqrt(x)",
            "Real square root. Halves each exact dimension exponent: the square root of an area has length dimension. Values require x >= 0; first derivatives require x > 0.\n\nExample: `math.sqrt(4[m^2])`.",
            &[("x", "Real scalar with a nonnegative value.")],
        ),
        function(
            "math.complex",
            "math.complex(real_part, imaginary_part)",
            "Construct a complex scalar from two real scalars with equal dimensions. Declaration initializers may supply the dimension to literal operands.\n\nExample: `math.complex(2[V], 3[V])`.",
            &[
                ("real_part", "Real component."),
                (
                    "imaginary_part",
                    "Imaginary component, with the same dimension as the real component.",
                ),
            ],
        ),
        function(
            "math.abs",
            "math.abs(x)",
            "Absolute value of a real scalar; preserves its dimension. Uses the nonnegative branch at zero.",
            &[("x", "Real scalar.")],
        ),
        function(
            "math.min",
            "math.min(x, y)",
            "Select the smaller real scalar. Operands must have compatible types and dimensions; ties select the first operand.",
            &[
                ("x", "First real scalar."),
                ("y", "Second compatible real scalar."),
            ],
        ),
        function(
            "math.max",
            "math.max(x, y)",
            "Select the larger real scalar. Operands must have compatible types and dimensions; ties select the first operand.",
            &[
                ("x", "First real scalar."),
                ("y", "Second compatible real scalar."),
            ],
        ),
        function(
            "math.clamp",
            "math.clamp(x, lower, upper)",
            "Restrict a real scalar to an inclusive interval with matching dimensions. Requires lower <= upper. At either endpoint, selects x.",
            &[
                ("x", "Real scalar to restrict."),
                ("lower", "Inclusive lower bound."),
                (
                    "upper",
                    "Inclusive upper bound; must not be less than lower.",
                ),
            ],
        ),
        function(
            "math.sign",
            "math.sign(x)",
            "Return dimensionless -1, 0, or 1 according to the sign of a real scalar. A nonsmooth operation, not a smooth derivative promise.",
            &[("x", "Real scalar of any admitted dimension.")],
        ),
        function(
            "math.step",
            "math.step(x)",
            "Return dimensionless 0 for x < 0 and 1 for x >= 0. The value at zero is 1.",
            &[("x", "Real scalar of any admitted dimension.")],
        ),
        function(
            "time",
            "time()",
            "Continuous timeline coordinate in seconds. Initial and restart time come from the enclosing timeline; pure operators cannot read an ambient clock.",
            &[],
        ),
        function(
            "derivative",
            "derivative(expression)",
            "Total time derivative of an admitted continuous State or real scalar polynomial expression. Divides dimensions by time. Fixed Parameters are constant; clocked values and unsupported expression profiles reject.",
            &[(
                "expression",
                "Continuous State or admitted expression over continuous States and fixed Parameters.",
            )],
        ),
        function(
            "partial",
            "partial(expression, wrt = binding, holding = (...))",
            "Partial derivative with respect to an explicit independent binding. Divides expression dimensions by binding dimensions. Formal polynomial and declared-coordinate profiles have their own admission limits.",
            &[
                ("expression", "Expression to differentiate."),
                (
                    "wrt",
                    "Explicit formal, Parameter, State or admitted coordinate binding.",
                ),
                (
                    "holding",
                    "Optional tuple of independent bindings held fixed.",
                ),
            ],
        ),
        function(
            "pullback",
            "pullback(expression, from = (...), at = (...))",
            "Compose an invariant scalar with an explicit coordinate map. Source selectors and target bindings must each cover their exact support once. Coordinate derivatives use the chain rule; this operation does not convert vector frames.",
            &[
                (
                    "from",
                    "Complete ordered tuple of source coordinate selectors.",
                ),
                (
                    "at",
                    "Complete ordered target coordinate assignments evaluated on the source support.",
                ),
            ],
        ),
        function(
            "jacobian_determinant",
            "jacobian_determinant(from = (...), at = (...))",
            "Signed determinant of an explicit square coordinate map. Its unit is the product of target coordinate units divided by the product of source coordinate units. A singular map has determinant zero.",
            &[
                (
                    "from",
                    "Complete ordered tuple of source coordinate selectors.",
                ),
                (
                    "at",
                    "Complete ordered target coordinate assignments evaluated on the source support.",
                ),
            ],
        ),
        function(
            "volume_jacobian",
            "volume_jacobian(from = (...), at = (...))",
            "Absolute Jacobian determinant for an invertible square coordinate map. Use this factor explicitly in a pulled-back scalar integral. Embedded surface measure is not inferred.",
            &[
                (
                    "from",
                    "Complete ordered tuple of source coordinate selectors.",
                ),
                (
                    "at",
                    "Complete ordered target coordinate assignments evaluated on the source support.",
                ),
            ],
        ),
        function(
            "map_orientation",
            "map_orientation(from = (...), at = (...))",
            "Dimensionless orientation sign of an invertible square coordinate map, kept separate from its positive volume scale. This does not supply a flux or vector frame conversion.",
            &[
                (
                    "from",
                    "Complete ordered tuple of source coordinate selectors.",
                ),
                (
                    "at",
                    "Complete ordered target coordinate assignments evaluated on the source support.",
                ),
            ],
        ),
        function(
            "pre",
            "pre(state)",
            "Read the committed left value of a State in its admitted clock or event context. Preserves the State's type and dimensions.",
            &[("state", "State owned by the active clock or event context.")],
        ),
        function(
            "next",
            "next(state)",
            "Denote a State's accepted right value in a clocked update or event reset. The surrounding Relation is simultaneous, not an imperative assignment.",
            &[("state", "State owned by the active update context.")],
        ),
        function(
            "sample",
            "sample(expression, clock)",
            "Sample an admitted continuous expression at the explicit Clock's tick. Coincident event and tick reads use the same committed left state.",
            &[
                ("expression", "Continuous expression to sample."),
                ("clock", "Exact nominal Clock for the sample."),
            ],
        ),
        function(
            "hold",
            "hold(state)",
            "Read a directly named periodic State continuously between ticks. Its explicit initial equation supplies the value before the first tick; hold creates no memory.",
            &[("state", "Periodic State with an explicit initial equation.")],
        ),
        function(
            "period",
            "period(clock)",
            "Return the exact declared Clock period projected to a scalar duration in seconds.",
            &[("clock", "Declared exact periodic Clock name.")],
        ),
        function(
            "coordinate",
            "coordinate(axis)",
            "Read a coordinate of the enclosing admitted spatial domain. Returns a length-valued scalar; the axis must exist in that domain.",
            &[("axis", "Zero-based spatial axis index.")],
        ),
        function(
            "grad",
            "grad(field)",
            "Spatial gradient on the field's admitted support. Divides dimensions by length and adds a spatial axis. Channel axes and spatial axes remain distinct.",
            &[(
                "field",
                "Spatial scalar or vector field admitted by the selected profile.",
            )],
        ),
        function(
            "div",
            "div(field)",
            "Spatial divergence on an admitted support. Contracts a spatial derivative axis and divides dimensions by length.",
            &[("field", "Admitted spatial flux expression.")],
        ),
        function(
            "symmetric_part",
            "symmetric_part(tensor)",
            "Symmetric part of an admitted spatial rank-two tensor. Preserves dimensions; used with vector gradients for strain.",
            &[("tensor", "Spatial rank-two tensor.")],
        ),
        function(
            "isotropic_lift",
            "isotropic_lift(scalar)",
            "Lift a scalar to an isotropic spatial tensor using the admitted spatial context. Preserves dimensions.",
            &[(
                "scalar",
                "Scalar expression in the admitted spatial context.",
            )],
        ),
        function(
            "trace",
            "trace(field)",
            "Take a full trace on an exact boundary or physical interface, selected by on or the enclosing Relation. The field's parent support selects its side. Preserves value dimensions and requires admitted trace regularity.",
            &[("field", "Field on the corresponding parent support.")],
        ),
        function(
            "normal",
            "normal(flux)",
            "Contract with the normal of the exact target selected by on or the enclosing Relation. Exterior boundaries use their parent-outward normal; physical interfaces use the common normal of their first declared boundary. Preserves flux dimensions and requires admitted trace regularity.",
            &[(
                "flux",
                "Spatial flux expression on an exact adjacent parent, or its admitted full trace on the target.",
            )],
        ),
        function(
            "ordinal",
            "ordinal(index)",
            "Convert a finite-space index to its exact integer ordinal. Nominal index ownership is checked before conversion.",
            &[("index", "Index of an admitted finite space.")],
        ),
        function(
            "quotient",
            "quotient(x, y)",
            "Checked integer quotient, truncating toward zero. Division by zero and unrepresentable results reject.",
            &[
                ("x", "Integer dividend."),
                ("y", "Nonzero compatible integer divisor."),
            ],
        ),
        function(
            "remainder",
            "remainder(x, y)",
            "Checked integer remainder, with the dividend's sign. Division by zero and unrepresentable results reject.",
            &[
                ("x", "Integer dividend."),
                ("y", "Nonzero compatible integer divisor."),
            ],
        ),
        function(
            "to_real",
            "to_real(value)",
            "Explicit integer scalar to binary64 conversion, rounded once to nearest with ties to even.",
            &[("value", "Integer scalar to convert.")],
        ),
        function(
            "to_integer",
            "to_integer(value)",
            "Explicit checked conversion from a dimensionless real scalar to an integer. Requires an integral, representable value; does not silently round.",
            &[("value", "Real scalar with an exact integral value.")],
        ),
    ];
    for (name, label, doc) in [
        (
            "math.pi",
            "math.pi: 1",
            "Dimensionless circle constant pi, represented by the compiler's canonical binary64 value.",
        ),
        (
            "math.i",
            "math.i: complex<1>",
            "Dimensionless imaginary unit. Use math.complex(real_part, imaginary_part) to construct dimensioned complex scalars.",
        ),
    ] {
        entries.push(documented(name, label, doc, Kind::Let));
    }
    entries
}

fn function(name: &str, label: &str, doc: &str, parameters: &[(&str, &str)]) -> EditorSymbol {
    let mut symbol = documented(name, label, doc, Kind::Operator);
    symbol.callable = true;
    symbol.children = parameters
        .iter()
        .map(|(name, doc)| documented(name, name, doc, Kind::Formal))
        .collect();
    symbol
}
