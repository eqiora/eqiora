//! Checked scalar storage access and numerical validation.
use super::*;

pub(super) fn read(values: &[f64], id: ValueId, instruction: usize) -> Result<f64, Diagnostic> {
    usize::try_from(id.0)
        .ok()
        .and_then(|index| values.get(index))
        .copied()
        .ok_or_else(|| {
            Diagnostic::error(
                codes::INVALID_OPERATOR_IR,
                format!(
                    "SSA value {} is unavailable at instruction {instruction}",
                    id.0
                ),
            )
            .with_graph_path(ir_path(instruction))
        })
}

pub(super) fn collect_roots(
    roots: &[ValueId],
    instructions: &[Instruction],
    values: &[f64],
) -> Result<Vec<f64>, Diagnostic> {
    roots
        .iter()
        .map(|root| read(values, *root, instructions.len()))
        .collect()
}

pub(super) fn slot_index(slot: SymbolSlot, instruction: usize) -> Result<usize, Diagnostic> {
    usize::try_from(slot.0).map_err(|_| {
        Diagnostic::error(codes::INVALID_OPERATOR_IR, "symbol slot exceeds usize")
            .with_graph_path(ir_path(instruction))
    })
}

pub(super) fn write_roots(
    roots: &[ValueId],
    values: &[f64],
    output: &mut [f64],
) -> Result<(), Diagnostic> {
    for (output, root) in output.iter_mut().zip(roots) {
        *output = read(values, *root, values.len())?;
    }
    Ok(())
}

pub(super) fn accumulate(
    values: &mut [f64],
    id: ValueId,
    contribution: f64,
    instruction: usize,
) -> Result<(), Diagnostic> {
    let index = usize::try_from(id.0).map_err(|_| invalid_value_index(id, instruction))?;
    let value = values
        .get_mut(index)
        .ok_or_else(|| invalid_value_index(id, instruction))?;
    let next = *value + contribution;
    require_finite_value(next, "VJP", instruction)?;
    *value = next;
    Ok(())
}

pub(super) fn accumulate_coordinate(
    values: &mut [f64],
    coordinate: usize,
    contribution: f64,
    name: &str,
) -> Result<(), Diagnostic> {
    let next = values[coordinate] + contribution;
    if !next.is_finite() {
        return Err(invalid_linearization(format!(
            "{name} coordinate {coordinate} evaluated to {next}"
        )));
    }
    values[coordinate] = next;
    Ok(())
}

pub(super) fn require_length(
    values: &[f64],
    expected: usize,
    name: &str,
) -> Result<(), Diagnostic> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(invalid_linearization(format!(
            "{name} expects {expected} values, received {}",
            values.len()
        )))
    }
}

pub(super) fn require_finite(values: &[f64], name: &str) -> Result<(), Diagnostic> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid_linearization(format!(
            "{name} must contain only finite values"
        )))
    }
}

pub(super) fn require_finite_value(
    value: f64,
    name: &str,
    instruction: usize,
) -> Result<(), Diagnostic> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Diagnostic::error(
            codes::NONFINITE_EVALUATION,
            format!("scalar Operator IR {name} instruction {instruction} evaluated to {value}"),
        )
        .with_graph_path(ir_path(instruction)))
    }
}

pub(super) fn powi_derivative(base: f64, exponent: i32) -> f64 {
    match exponent {
        0 => 0.0,
        i32::MIN => f64::from(exponent) * base.powi(exponent) / base,
        _ => f64::from(exponent) * base.powi(exponent - 1),
    }
}

pub(super) fn invalid_value_index(id: ValueId, instruction: usize) -> Diagnostic {
    Diagnostic::error(
        codes::INVALID_OPERATOR_IR,
        format!(
            "SSA value {} is unavailable at reverse instruction {instruction}",
            id.0
        ),
    )
    .with_graph_path(ir_path(instruction))
}

pub(super) fn invalid_linearization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_LINEARIZATION, message)
        .with_graph_path(GraphPath::new(["operator-ir", "linearization"]))
}

pub(super) fn ir_size_error() -> Diagnostic {
    Diagnostic::error(
        codes::INVALID_OPERATOR_IR,
        "scalar Operator IR exceeds the u32 slot limit",
    )
}

pub(super) fn ir_builder_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_OPERATOR_IR, message)
        .with_graph_path(GraphPath::new(["operator-ir", "input-slots"]))
}

pub(super) fn ir_path(index: usize) -> GraphPath {
    GraphPath::new(["operator-ir".to_owned(), index.to_string()])
}

/// Validate the numerical instruction profile and finite role-bound point together.
pub(super) fn validate_linearization_inputs(
    instructions: &[Instruction],
    input_count: usize,
    inputs: &[f64],
    roles: &[DifferentiationRole],
) -> Result<(), Diagnostic> {
    if instructions.iter().any(|instruction| {
        matches!(
            instruction,
            Instruction::Select { .. }
                | Instruction::Require { .. }
                | Instruction::PureOperator { .. }
                | Instruction::Array { .. }
                | Instruction::Index(_, _)
        )
    }) {
        return Err(invalid_linearization(
            "retained pure applications, ordered selections and channel operations require explicit scalar projection before automatic differentiation",
        ));
    }
    if inputs.len() != input_count || roles.len() != input_count {
        return Err(invalid_linearization(format!(
            "scalar linearization expects {} point values and roles, received {} values and {} roles",
            input_count,
            inputs.len(),
            roles.len()
        )));
    }
    require_finite(inputs, "linearization point")?;
    Ok(())
}
