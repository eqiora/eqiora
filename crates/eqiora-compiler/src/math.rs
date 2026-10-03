pub(crate) mod finite;
pub(crate) mod piecewise;
pub(crate) mod tensor;
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_lang::{NamePath, TextRange};

use crate::diagnostics::source_error;

pub(crate) const ROOT: &str = "math";

pub(crate) fn model_name_diagnostics(
    file: &str,
    model_name: &str,
    range: TextRange,
) -> Vec<Diagnostic> {
    (model_name == ROOT)
        .then(|| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                range,
                "identifier `math` is reserved for compiler-owned scalar mathematics",
            )
        })
        .into_iter()
        .collect()
}

/// Whether a path belongs to the compiler-owned scalar-mathematics namespace.
pub(crate) fn is_namespaced(path: &NamePath) -> bool {
    path.segments().next() == Some(ROOT)
}

/// Whether a path names an admitted scalar mathematical function.
pub(crate) fn is_function(path: &NamePath) -> bool {
    unary_function(path.as_str()).is_some()
        || piecewise::arity(path.as_str()).is_some()
        || finite::Operation::named(path.as_str()).is_some()
}

pub(crate) fn unary_function(name: &str) -> Option<eqiora_schema::kernel::UnaryMathFunction> {
    use eqiora_schema::kernel::UnaryMathFunction;
    Some(match name {
        "math.sin" => UnaryMathFunction::Sin,
        "math.sqrt" => UnaryMathFunction::Sqrt,
        "math.cos" => UnaryMathFunction::Cos,
        "math.exp" => UnaryMathFunction::Exp,
        "math.log" => UnaryMathFunction::Log,
        "math.conj" => UnaryMathFunction::Conj,
        "math.real" => UnaryMathFunction::Real,
        "math.imag" => UnaryMathFunction::Imag,
        "math.abs" => UnaryMathFunction::Abs,
        "math.abs2" => UnaryMathFunction::Abs2,
        "math.arg" => UnaryMathFunction::Arg,
        _ => return None,
    })
}

/// Returns the compiler-owned value of a canonical mathematical constant.
///
/// Constants remain source paths for formatting and provenance, but lower to
/// the same dimensionless scalar expression as the equivalent numeric literal.
pub(crate) fn constant(path: &NamePath) -> Option<f64> {
    match path.as_str() {
        "math.pi" => Some(f64::from_bits(0x4009_21fb_5444_2d18)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use eqiora_lang::{NamePath, TextRange};

    #[test]
    fn pi_has_the_canonical_binary64_value() {
        let path = NamePath::from_segments(["math", "pi"], TextRange::default()).unwrap();
        assert_eq!(
            super::constant(&path).unwrap().to_bits(),
            0x4009_21fb_5444_2d18
        );
    }
}
