use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use eqiora_ir::{DifferentiationRole, LinearizedRelation, RelationTangent, ScalarOperatorIr};
use eqiora_schema::kernel::{ExprNode, KernelNode, SymbolRef};
use eqiora_sem::{Interpreter, KernelProgram, ReferenceConfig};
use eqiora_time::ImplicitDaeInitialization;

use crate::time::invalid_time;

pub(crate) fn initialize(
    kernel: &KernelProgram,
    fields: &[Id<kinds::Field>],
    relation: Id<kinds::Relation>,
    config: ReferenceConfig,
) -> Result<ImplicitDaeInitialization, Diagnostic> {
    let initial = Interpreter::new()
        .initialize(kernel, config)
        .map_err(|diagnostics| {
            diagnostics.into_iter().next().unwrap_or_else(|| {
                invalid_time(relation, "fresh initialization failed without a diagnostic")
            })
        })?;
    let state = fields
        .iter()
        .map(|field| {
            initial
                .fields()
                .get(&field.erase())
                .and_then(eqiora_core::ValueLiteral::real_scalar_value)
                .map(|value| value.value())
                .ok_or_else(|| {
                    invalid_time(
                        relation,
                        "fresh initialization omitted a required state coordinate",
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let derivative = fields.iter().map(|field| {
        if let Some(value) = initial.derivatives().get(&field.erase()) {
            return Ok(*value);
        }
        // Algebraic coordinates have no derivative unknown in this index-one
        // partition. Zero is the residual-adapter convention for that absent
        // coordinate, not an inferred initial condition or physical derivative.
        let has_derivative = kernel.nodes().any(|node| match node {
            KernelNode::Relation(definition) => definition.expression().nodes().iter().any(|node| {
                matches!(node, ExprNode::Symbol(SymbolRef::Derivative(candidate)) if candidate == field)
            }),
            _ => false,
        });
        if has_derivative {
            Err(invalid_time(relation, "fresh initialization omitted a differential coordinate"))
        } else {
            Ok(0.0)
        }
    }).collect::<Result<Vec<_>, _>>()?;
    ImplicitDaeInitialization::accepted(state, derivative)
}

/// Certify the existing zero initial-state Parameter action from the complete
/// linearized initial constraints, including regular descriptor equations.
/// Free derivative directions are allowed only when they cannot change state.
pub(crate) fn require_zero_parameter_tangent(
    kernel: &KernelProgram,
    fields: &[Id<kinds::Field>],
    parameters: &[Id<kinds::Parameter>],
    relation: Id<kinds::Relation>,
) -> Result<(), Diagnostic> {
    let unsupported = || {
        invalid_time(
            relation,
            "initial constraints do not prove a unique zero initial-state Parameter tangent",
        )
    };
    let initial = initialize(kernel, fields, relation, ReferenceConfig::new(0.0, 1.0)?)?;
    let n = fields.len();
    let width = 2 * n + parameters.len();
    let mut rows = Vec::new();
    for node in kernel.nodes() {
        let KernelNode::Relation(definition) = node else {
            continue;
        };
        if definition.id() != relation && !definition.is_initial() {
            continue;
        }
        let typed = kernel
            .typed_relation_residual(definition.id())
            .map_err(|errors| errors.into_iter().next().expect("typing failure"))?;
        let operator = ScalarOperatorIr::lower_typed_scalar(&typed)?;
        let mut inputs = Vec::new();
        let mut roles = Vec::new();
        let mut coordinates = Vec::new();
        for symbol in operator.symbols() {
            let (value, coordinate) = match symbol {
                SymbolRef::Field(field) => {
                    let index = fields
                        .iter()
                        .position(|candidate| candidate == field)
                        .ok_or_else(unsupported)?;
                    (initial.state()[index], Some(index))
                }
                SymbolRef::Derivative(field) => {
                    let index = fields
                        .iter()
                        .position(|candidate| candidate == field)
                        .ok_or_else(unsupported)?;
                    (initial.derivative()[index], Some(n + index))
                }
                SymbolRef::Parameter(parameter) => {
                    let value = kernel
                        .value(parameter.erase())
                        .ok_or_else(unsupported)?
                        .value();
                    let coordinate = parameters
                        .iter()
                        .position(|candidate| candidate == parameter)
                        .map(|index| 2 * n + index);
                    (value, coordinate)
                }
                SymbolRef::Time => (0.0, None),
                _ => return Err(unsupported()),
            };
            inputs.push(value);
            roles.push(if coordinate.is_some() {
                DifferentiationRole::Unknown
            } else {
                DifferentiationRole::Frozen
            });
            if let Some(coordinate) = coordinate {
                coordinates.push(coordinate);
            }
        }
        let linearization = operator.linearize(&inputs, &roles)?;
        let mut block = vec![vec![0.0; width]; operator.residual_count()];
        for (local, coordinate) in coordinates.iter().copied().enumerate() {
            let mut direction = vec![0.0; coordinates.len()];
            direction[local] = 1.0;
            let mut output = vec![0.0; block.len()];
            linearization.jvp(RelationTangent::Unknown(&direction), &mut output)?;
            for (row, value) in block.iter_mut().zip(output) {
                row[coordinate] = value;
            }
        }
        rows.extend(block);
    }
    let derivative_rank = rank(&rows, n..2 * n)?;
    if rank(&rows, 0..2 * n)? != n + derivative_rank || rank(&rows, n..width)? != derivative_rank {
        return Err(unsupported());
    }
    Ok(())
}

fn rank(rows: &[Vec<f64>], columns: std::ops::Range<usize>) -> Result<usize, Diagnostic> {
    // Padding changes neither column span nor rank and reuses the existing
    // exact binary-rational matrix proof rather than a second rank threshold.
    let size = rows.len().max(columns.len());
    let mut square = vec![0.0; size * size];
    for (row, values) in rows.iter().enumerate() {
        for (column, source) in columns.clone().enumerate() {
            square[row * size + column] = values[source];
        }
    }
    Ok(eqiora_time::ConstantDerivativeMatrixProof::new(size, square)?.exact_rank())
}
