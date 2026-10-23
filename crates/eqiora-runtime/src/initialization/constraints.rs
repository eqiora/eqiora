//! Initial constraint actions and demand-local time compatibility.
use super::*;
use eqiora_ir::{DifferentiationRole, LinearizedRelation, RelationTangent, ScalarOperatorIr};
use num_traits::{ToPrimitive, Zero};

pub(super) enum ConstraintAction<'a> {
    Jacobian {
        include_initial: bool,
    },
    Time {
        mass: &'a eqiora_time::ConstantDerivativeMatrixProof,
        mixed: Option<(&'a [f64], &'a [f64])>,
    },
}

pub(super) fn linearized_constraint_actions(
    kernel: &KernelProgram,
    fields: &[eqiora_core::TimeStateCoordinate],
    parameters: &[Id<kinds::Parameter>],
    relation: Id<kinds::Relation>,
    point: (f64, &ImplicitDaeInitialization),
    action: ConstraintAction<'_>,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), Diagnostic> {
    let (initial_time, initial) = point;
    let (include_initial, time_action) = match action {
        ConstraintAction::Jacobian { include_initial } => (include_initial, None),
        ConstraintAction::Time { mass, mixed } => (false, Some((mass, mixed))),
    };
    let unsupported = || {
        invalid_time(
            relation,
            "initial constraint linearization contains an unsupported symbol",
        )
    };
    let n = fields.len();
    let width = 2 * n + parameters.len() + 1;
    let mut rows = Vec::new();
    let mut mixed_rows = Vec::new();
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
        let mut symbol_columns = Vec::new();
        let mut typed_inputs = Vec::new();
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
            let ty = typed
                .expression()
                .nodes()
                .iter()
                .zip(typed.node_types())
                .find_map(|(node, ty)| {
                    matches!(node, ExprNode::Symbol(candidate) if candidate == symbol)
                        .then_some(&ty.value_type)
                })
                .ok_or_else(unsupported)?;
            typed_inputs.push(
                eqiora_core::ValueLiteral::from_real(ty.clone(), value)
                    .map_err(|_| unsupported())?,
            );
            symbol_columns.push(if *symbol == SymbolRef::Time {
                Some(width - 1)
            } else {
                coordinate
            });
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
        let mut mixed = vec![0.0; block.len()];
        if let Some((mass, direction)) = time_action {
            let (time, action) = projected_time_actions(
                &operator,
                &typed_inputs,
                &symbol_columns,
                mass,
                direction,
                n,
                relation,
            )?;
            for (row, value) in block.iter_mut().zip(time) {
                row[width - 1] = value;
            }
            mixed = action;
        }
        mixed_rows.extend(mixed);
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
            mixed_rows.push(0.0);
        }
    }
    Ok((rows, mixed_rows))
}

/// Differentiate a representative modulo the constant mass column space.
/// Keep all symbol dependence inside atoms: freezing a Parameter coefficient
/// before cancellation would lose its derivative away from the initial point.
fn projected_time_actions(
    operator: &ScalarOperatorIr,
    inputs: &[eqiora_core::ValueLiteral],
    columns: &[Option<usize>],
    mass: &eqiora_time::ConstantDerivativeMatrixProof,
    mixed: Option<(&[f64], &[f64])>,
    n: usize,
    relation: Id<kinds::Relation>,
) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
    let mut temporal = vec![BigRational::zero(); operator.residual_count()];
    let mut action = temporal.clone();
    for (atom, mut coefficients) in operator.additive_terms(inputs, operator.symbols())? {
        coefficients.resize(mass.dimension(), BigRational::zero());
        if mass.is_compatible(&[], &coefficients)? {
            continue;
        }
        let slots = atom
            .symbols()
            .iter()
            .map(|symbol| {
                operator
                    .symbols()
                    .iter()
                    .position(|candidate| candidate == symbol)
                    .expect("atom retains original symbol")
            })
            .collect::<Vec<_>>();
        let point = slots
            .iter()
            .map(|&slot| inputs[slot].clone())
            .collect::<Vec<_>>();
        let roles = slots
            .iter()
            .map(|&slot| {
                if columns[slot].is_some() {
                    DifferentiationRole::Unknown
                } else {
                    DifferentiationRole::Frozen
                }
            })
            .collect::<Vec<_>>();
        let linearization = atom.linearize_typed(&point, &roles)?;
        let active = slots
            .iter()
            .filter_map(|&slot| columns[slot].map(|column| (slot, column)))
            .collect::<Vec<_>>();
        if let Some((physical, parameter)) = mixed {
            let along_time = active
                .iter()
                .map(|&(slot, column)| {
                    if operator.symbols()[slot] == SymbolRef::Time {
                        1.0
                    } else if column < n {
                        physical[column]
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>();
            let along_parameter = active
                .iter()
                .map(|&(slot, column)| {
                    if operator.symbols()[slot] != SymbolRef::Time && column >= 2 * n {
                        parameter[column - 2 * n]
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>();
            let mut value = [0.0];
            linearization.mixed_jvp(
                RelationTangent::Unknown(&along_time),
                RelationTangent::Unknown(&along_parameter),
                RelationTangent::Unknown(&vec![0.0; active.len()]),
                &mut value,
            )?;
            for (result, coefficient) in action.iter_mut().zip(&coefficients) {
                *result += coefficient * exact(value[0]);
            }
        } else {
            let direction = active
                .iter()
                .map(|&(slot, _)| {
                    if operator.symbols()[slot] == SymbolRef::Time {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>();
            let mut value = [0.0];
            linearization.jvp(RelationTangent::Unknown(&direction), &mut value)?;
            for (result, coefficient) in temporal.iter_mut().zip(&coefficients) {
                *result += coefficient * exact(value[0]);
            }
        }
    }
    let rounded = |values: Vec<BigRational>| {
        values
            .into_iter()
            .map(|value| {
                value
                    .to_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| {
                        invalid_time(
                            relation,
                            "projected initial compatibility action exceeds binary64 range",
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()
    };
    Ok((rounded(temporal)?, rounded(action)?))
}
