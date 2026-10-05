use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use eqiora_ir::{DifferentiationRole, LinearizedRelation, RelationTangent, ScalarOperatorIr};
use eqiora_schema::kernel::{ExprNode, KernelNode, SymbolRef};
use eqiora_sem::{Interpreter, KernelProgram, ReferenceConfig};
use eqiora_time::ImplicitDaeInitialization;

use crate::time::invalid_time;

pub(crate) fn initialize(
    kernel: &KernelProgram,
    fields: &[eqiora_core::TimeStateCoordinate],
    relation: Id<kinds::Relation>,
    initial_time: f64,
    config: ReferenceConfig,
) -> Result<ImplicitDaeInitialization, Diagnostic> {
    let initial = Interpreter::new()
        .initialize(kernel, initial_time, config)
        .map_err(|diagnostics| {
            diagnostics.into_iter().next().unwrap_or_else(|| {
                invalid_time(relation, "fresh initialization failed without a diagnostic")
            })
        })?;
    let state = fields
        .iter()
        .map(|coordinate| {
            let (field, order) = (coordinate.field(), coordinate.derivative_order());
            let value = if let Some(order) = std::num::NonZeroU32::new(order) {
                initial
                    .derivatives()
                    .get(&(field.erase(), order))
                    .and_then(|value| coordinate_value(*coordinate, value))
            } else {
                initial
                    .fields()
                    .get(&field.erase())
                    .and_then(|value| coordinate_value(*coordinate, value))
            };
            value.ok_or_else(|| {
                invalid_time(
                    relation,
                    "fresh initialization omitted a required state coordinate",
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let derivative = fields.iter().map(|coordinate| {
            let (field, order) = (coordinate.field(), coordinate.derivative_order());
        let order = order.checked_add(1).and_then(std::num::NonZeroU32::new)
            .ok_or_else(|| invalid_time(relation, "state coordinate derivative order overflows"))?;
        if let Some(value) = initial.derivatives().get(&(field.erase(), order)) {
            return coordinate_value(*coordinate, value).ok_or_else(|| {
                invalid_time(relation, "time lowering requires the complete initial derivative coordinate")
            });
        }
        // Only an algebraic coordinate with no authored time rate uses zero
        // for its unused adapter rate. Missing differential data is an error.
        let has_derivative = kernel.nodes().any(|node| match node {
            KernelNode::Relation(definition) => definition.expression().nodes().iter().any(|node| {
                matches!(node, ExprNode::Symbol(SymbolRef::Derivative(candidate, candidate_order)) if *candidate == field && *candidate_order >= order)
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

fn coordinate_value(
    coordinate: eqiora_core::TimeStateCoordinate,
    value: &eqiora_core::ValueLiteral,
) -> Option<f64> {
    if coordinate.is_imaginary()
        && value.value_type().scalar_domain() != eqiora_core::ScalarDomain::Complex
    {
        return None;
    }
    let (real, imaginary) = value.component(coordinate.component())?;
    Some(if coordinate.is_imaginary() {
        imaginary
    } else {
        real
    })
}

/// Certify the existing zero initial-state Parameter action from the complete
/// linearized initial constraints, including regular descriptor equations.
/// Free derivative directions are allowed only when they cannot change state.
pub(crate) fn require_zero_parameter_tangent(
    kernel: &KernelProgram,
    fields: &[eqiora_core::TimeStateCoordinate],
    parameters: &[Id<kinds::Parameter>],
    relation: Id<kinds::Relation>,
    initial_time: f64,
) -> Result<(), Diagnostic> {
    let unsupported = || {
        invalid_time(
            relation,
            "initial constraints do not prove a unique zero initial-state Parameter tangent",
        )
    };
    let initial = initialize(
        kernel,
        fields,
        relation,
        initial_time,
        ReferenceConfig::new(0.0, 1.0)?,
    )?;
    let n = fields.len();
    let width = 2 * n + parameters.len();
    let rows = linearized_constraints(
        kernel,
        fields,
        parameters,
        relation,
        initial_time,
        &initial,
        true,
    )?;
    let derivative_rank = rank(&rows, n..2 * n)?;
    if rank(&rows, 0..2 * n)? != n + derivative_rank || rank(&rows, n..width)? != derivative_rank {
        return Err(unsupported());
    }
    Ok(())
}

/// At a fresh initial point, a constant-mass system needs its algebraic
/// constraints to determine the null directions of the mass matrix locally.
/// This neither differentiates a constraint nor certifies an entire trajectory.
pub(crate) fn require_constant_mass_regularity(
    kernel: &KernelProgram,
    fields: &[eqiora_core::TimeStateCoordinate],
    relation: Id<kinds::Relation>,
    initial_time: f64,
    initial: &ImplicitDaeInitialization,
    mass: &eqiora_time::ConstantDerivativeMatrixProof,
) -> Result<(), Diagnostic> {
    let n = fields.len();
    if mass.exact_rank() == n {
        return Ok(());
    }
    let rows = linearized_constraints(kernel, fields, &[], relation, initial_time, initial, false)?;
    let state_jacobian = rows
        .iter()
        .flat_map(|row| row[..n].iter().copied())
        .collect::<Vec<_>>();
    mass.require_index_one_regularity(&state_jacobian)
        .map_err(|error| invalid_time(relation, error.message()))
}

/// The declared residual-native partition must locally determine differential
/// rates and algebraic values with differential values held fixed.
pub(crate) fn require_implicit_regularity(
    kernel: &KernelProgram,
    fields: &[eqiora_core::TimeStateCoordinate],
    relation: Id<kinds::Relation>,
    initial_time: f64,
    initial: &ImplicitDaeInitialization,
    kinds: &[eqiora_time::DaeVariableKind],
) -> Result<(), Diagnostic> {
    let n = fields.len();
    let rows = linearized_constraints(kernel, fields, &[], relation, initial_time, initial, false)?;
    let selected = rows
        .iter()
        .map(|row| {
            kinds
                .iter()
                .enumerate()
                .map(|(column, kind)| {
                    row[match kind {
                        eqiora_time::DaeVariableKind::Differential => n + column,
                        eqiora_time::DaeVariableKind::Algebraic => column,
                    }]
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let actual = rank(&selected, 0..n)?;
    if actual != n {
        return Err(invalid_time(
            relation,
            format!(
                "fresh residual-native initialization has an unsupported high-index or singular differential/algebraic partition: local regularity rank {actual}, required {n}; explicit state selection or index reduction may be needed"
            ),
        ));
    }
    Ok(())
}

fn linearized_constraints(
    kernel: &KernelProgram,
    fields: &[eqiora_core::TimeStateCoordinate],
    parameters: &[Id<kinds::Parameter>],
    relation: Id<kinds::Relation>,
    initial_time: f64,
    initial: &ImplicitDaeInitialization,
    include_initial: bool,
) -> Result<Vec<Vec<f64>>, Diagnostic> {
    let unsupported = || {
        invalid_time(
            relation,
            "initial constraint linearization contains an unsupported symbol",
        )
    };
    let n = fields.len();
    let width = 2 * n + parameters.len();
    let mut rows = Vec::new();
    for node in kernel.nodes() {
        let KernelNode::Relation(definition) = node else {
            continue;
        };
        if definition.id() != relation && !(include_initial && definition.is_initial()) {
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
                        .position(|candidate| {
                            *candidate == eqiora_core::TimeStateCoordinate::new(*field, 0, 0, false)
                        })
                        .ok_or_else(unsupported)?;
                    (initial.state()[index], Some(index))
                }
                SymbolRef::Derivative(field, order) => {
                    if let Some(index) = fields.iter().position(|candidate| {
                        *candidate
                            == eqiora_core::TimeStateCoordinate::new(*field, order.get(), 0, false)
                    }) {
                        (initial.state()[index], Some(index))
                    } else {
                        let index = fields
                            .iter()
                            .position(|candidate| {
                                *candidate
                                    == eqiora_core::TimeStateCoordinate::new(
                                        *field,
                                        order.get() - 1,
                                        0,
                                        false,
                                    )
                            })
                            .ok_or_else(unsupported)?;
                        (initial.derivative()[index], Some(n + index))
                    }
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
                SymbolRef::Time => (initial_time, None),
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
    for (coordinate, source) in fields.iter().enumerate() {
        let (field, order) = (source.field(), source.derivative_order());
        if let Some(next) = fields.iter().position(|candidate| {
            *candidate == eqiora_core::TimeStateCoordinate::new(field, order + 1, 0, false)
        }) {
            let mut row = vec![0.; width];
            row[n + coordinate] = 1.;
            row[next] = -1.;
            rows.push(row);
        }
    }
    Ok(rows)
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

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, ScalarDomain, TimeStateCoordinate, ValueLiteral, ValueType};

    #[test]
    fn initial_projection_keeps_imaginary_channels_and_rejects_foreign_parts() {
        let field = Id::<kinds::Field>::new();
        let ty = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
            .unwrap()
            .array(2)
            .unwrap();
        let value = ValueLiteral::new(ty, [(1., 2.), (3., 4.)]).unwrap();
        let projected =
            [(1, false), (1, true), (0, false), (0, true)].map(|(component, imaginary)| {
                coordinate_value(
                    TimeStateCoordinate::new(field, 0, component, imaginary),
                    &value,
                )
                .unwrap()
            });
        assert_eq!(projected, [3., 4., 1., 2.]);
        assert!(coordinate_value(TimeStateCoordinate::new(field, 0, 2, false), &value).is_none());
        let real = ValueLiteral::from_real(
            ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap(),
            7.,
        )
        .unwrap();
        assert!(coordinate_value(TimeStateCoordinate::new(field, 0, 0, true), &real).is_none());
    }
}
