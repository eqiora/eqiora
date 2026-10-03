//! Finite algebra shares typed admission and the reference evaluator's demand/work budget.
use super::{check_component_work, component_budget_error};
use eqiora_core::{Diagnostic, ValueLiteral, diagnostic::codes};
use eqiora_schema::kernel::{FiniteBinaryOperation, FiniteUnaryOperation, typing::ExpressionType};
use num_complex::Complex64;

fn invalid(error: impl std::fmt::Display) -> Diagnostic {
    Diagnostic::error(codes::DIMENSION_MISMATCH, error.to_string())
}
fn ty(value: &ValueLiteral) -> ExpressionType<()> {
    ExpressionType::new(value.value_type().clone(), None)
}

pub(super) fn unary(
    operation: FiniteUnaryOperation,
    value: &ValueLiteral,
) -> Result<ValueLiteral, Diagnostic> {
    let output = ty(value)
        .finite_unary(operation)
        .map_err(invalid)?
        .value_type;
    let count = value.component_count();
    check_component_work(0, count)?;
    let input_columns = value
        .value_type()
        .map_bases()
        .map(|(source, _)| source.extent() as usize);
    let output_columns = output
        .map_bases()
        .map(|(source, _)| source.extent() as usize);
    ValueLiteral::new(
        output,
        (0..count).map(|index| {
            let source = if let (Some(input), Some(output)) = (input_columns, output_columns) {
                (index % output) * input + index / output
            } else {
                index
            };
            let (real, imaginary) = value.component(source).expect("typed finite component");
            (
                real,
                if operation == FiniteUnaryOperation::Adjoint {
                    -imaginary
                } else {
                    imaginary
                },
            )
        }),
    )
    .map_err(invalid)
}

pub(super) fn binary(
    operation: FiniteBinaryOperation,
    left: &ValueLiteral,
    right: &ValueLiteral,
    used: &mut usize,
) -> Result<ValueLiteral, Diagnostic> {
    let output = ty(left)
        .finite_binary(operation, ty(right))
        .map_err(invalid)?
        .value_type;
    let count = output
        .shape()
        .component_count()
        .expect("checked finite shape");
    let inner = if let Some((source, _)) = left.value_type().map_bases() {
        source.extent() as usize
    } else {
        left.value_type()
            .coordinate_basis()
            .expect("typed pairing")
            .extent() as usize
    };
    let columns = output
        .map_bases()
        .map_or(1, |(source, _)| source.extent() as usize);
    let cost = count
        .checked_mul(inner)
        .ok_or_else(component_budget_error)?;
    check_component_work(*used, cost)?;
    *used += cost;
    ValueLiteral::new(
        output,
        (0..count).map(|index| {
            let (row, column) = (index / columns, index % columns);
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..inner {
                let a = left
                    .component(if operation == FiniteBinaryOperation::Pair {
                        k
                    } else {
                        row * inner + k
                    })
                    .expect("typed finite left component");
                let b = right
                    .component(if operation == FiniteBinaryOperation::Compose {
                        k * columns + column
                    } else {
                        k
                    })
                    .expect("typed finite right component");
                sum += Complex64::new(a.0, a.1) * Complex64::new(b.0, b.1);
            }
            (sum.re, sum.im)
        }),
    )
    .map_err(invalid)
}
