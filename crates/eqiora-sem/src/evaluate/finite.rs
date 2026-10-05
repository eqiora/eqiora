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
    if matches!(
        operation,
        FiniteUnaryOperation::MatrixTrace
            | FiniteUnaryOperation::Determinant
            | FiniteUnaryOperation::Inverse
    ) {
        use eqiora_schema::kernel::{
            ExprDagBuilder,
            typing::{RootContract, TypedResidual},
        };
        let mut builder = ExprDagBuilder::new();
        let input = builder.constant(value.clone())?;
        let root = builder.finite_unary(operation, input)?;
        let dag = builder.finish([root])?;
        let typed =
            TypedResidual::<()>::infer(dag, None, RootContract::ComponentwiseResidual, |_| {
                Err::<ExpressionType<()>, _>(())
            })
            .map_err(|_| invalid("finite map invariant typing failed"))?;
        let components = eqiora_ir::ComponentScalarization::lower(&typed)?.evaluate(|_| None)?;
        if output.scalar_domain() == eqiora_core::ScalarDomain::Complex {
            return ValueLiteral::new(
                output,
                components.chunks_exact(2).map(|pair| (pair[0], pair[1])),
            )
            .map_err(invalid);
        }
        return ValueLiteral::new(output, components.into_iter().map(|value| (value, 0.0)))
            .map_err(invalid);
    }
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
            let source = if let FiniteUnaryOperation::PermuteFactors(order) = operation {
                if let Some((input, target)) = value.value_type().map_bases() {
                    let columns = input.extent() as usize;
                    permuted_component(target, index / columns, order) * columns
                        + permuted_component(input, index % columns, order)
                } else {
                    permuted_component(
                        value
                            .value_type()
                            .coordinate_basis()
                            .expect("typed coordinates"),
                        index,
                        order,
                    )
                }
            } else if let (Some(input), Some(output)) = (input_columns, output_columns) {
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
    if operation == FiniteBinaryOperation::TensorProduct {
        check_component_work(*used, count)?;
        *used += count;
        return ValueLiteral::new(
            output,
            (0..count).map(|index| {
                let (a, b) = if let (Some((ls, _)), Some((rs, rt))) = (
                    left.value_type().map_bases(),
                    right.value_type().map_bases(),
                ) {
                    let (lc, rc, rr) = (
                        ls.extent() as usize,
                        rs.extent() as usize,
                        rt.extent() as usize,
                    );
                    let (row, col) = (index / (lc * rc), index % (lc * rc));
                    ((row / rr) * lc + col / rc, (row % rr) * rc + col % rc)
                } else {
                    (
                        index / right.component_count(),
                        index % right.component_count(),
                    )
                };
                let (ar, ai) = left.component(a).expect("typed tensor factor");
                let (br, bi) = right.component(b).expect("typed tensor factor");
                let value = Complex64::new(ar, ai) * Complex64::new(br, bi);
                (value.re, value.im)
            }),
        )
        .map_err(invalid);
    }
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

fn permuted_component(basis: eqiora_core::FiniteBasis, index: usize, order: [u8; 2]) -> usize {
    if order == [0, 1] {
        return index;
    }
    let [left, right] = basis.factors().expect("typed product basis");
    (index % left.extent() as usize) * right.extent() as usize + index / left.extent() as usize
}
