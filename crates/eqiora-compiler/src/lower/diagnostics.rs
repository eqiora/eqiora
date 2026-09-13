use super::*;

pub(super) fn unresolved(file: &str, range: TextRange, name: &str, expected: &str) -> Diagnostic {
    source_error(
        codes::LANGUAGE_TYPE_ERROR,
        file,
        range,
        format!("unresolved {expected} `{name}`"),
    )
}

pub(super) fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}
