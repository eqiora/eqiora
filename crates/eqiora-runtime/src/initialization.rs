mod constraints;
use constraints::{ConstraintAction, linearized_constraint_actions};

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use eqiora_schema::kernel::{ExprNode, KernelNode, SymbolRef};
use eqiora_sem::{Interpreter, KernelProgram, ReferenceConfig};
use eqiora_time::ImplicitDaeInitialization;
use num_rational::BigRational;
use num_traits::Signed;

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
    let initial = initialize(
        kernel,
        fields,
        relation,
        initial_time,
        ReferenceConfig::new(0.0, 1.0)?,
    )?;
    let n = fields.len();
    let rows = linearized_constraints(
        kernel,
        fields,
        parameters,
        relation,
        initial_time,
        &initial,
        true,
    )?;
    require_zero_state_parameter_tangent(&rows, n, parameters.len(), relation)
}

fn require_zero_state_parameter_tangent(
    rows: &[Vec<f64>],
    n: usize,
    parameters: usize,
    relation: Id<kinds::Relation>,
) -> Result<(), Diagnostic> {
    let derivative_rank = rank(rows, n..2 * n)?;
    if rank(rows, 0..2 * n)? != n + derivative_rank
        || rank(rows, n..2 * n + parameters)? != derivative_rank
    {
        return Err(invalid_time(
            relation,
            "initial constraints do not prove a unique zero initial-state Parameter tangent",
        ));
    }
    Ok(())
}

/// Differentiate the same initial equations, preserving the zero unused-rate convention.
pub(crate) fn initial_rate_parameter_tangent(
    kernel: &KernelProgram,
    fields: &[eqiora_core::TimeStateCoordinate],
    parameters: &[Id<kinds::Parameter>],
    relation: Id<kinds::Relation>,
    time: f64,
    initial: &ImplicitDaeInitialization,
    direction: &[f64],
) -> Result<Vec<f64>, Diagnostic> {
    let n = fields.len();
    if initial.state().len() != n
        || direction.len() != parameters.len()
        || direction.iter().any(|value| !value.is_finite())
    {
        return Err(invalid_time(
            relation,
            "initial rate Parameter action has invalid shape or values",
        ));
    }
    let rows = linearized_constraints(kernel, fields, parameters, relation, time, initial, true)?;
    require_zero_state_parameter_tangent(&rows, n, parameters.len(), relation)?;
    let authored = fields
        .iter()
        .map(|coordinate| {
            kernel.nodes().any(|node| match node {
                KernelNode::Relation(definition) => {
                    definition.expression().nodes().iter().any(|node| {
                        matches!(node, ExprNode::Symbol(SymbolRef::Derivative(field, order))
                    if *field == coordinate.field() && order.get() > coordinate.derivative_order())
                    })
                }
                _ => false,
            })
        })
        .collect::<Vec<_>>();
    let mut action_rows = Vec::new();
    let mut rhs = Vec::new();
    for row in &rows {
        let mut mass = row[n..2 * n].to_vec();
        mass.resize(2 * n, 0.0);
        action_rows.push(mass);
        rhs.push(
            -row[2 * n..]
                .iter()
                .zip(direction)
                .map(|(&a, &b)| exact(a) * exact(b))
                .sum::<BigRational>(),
        );
    }
    // Demand time compatibility only for rate directions that the continuous
    // equations leave undetermined. In particular a full-rank rate
    // block needs no time derivative of forcing at a table knot or branch.
    // Unused adapter rates have their existing zero convention.
    let mut determined_rows = action_rows.clone();
    let mut determined_rhs = rhs.clone();
    for (index, &active) in authored.iter().enumerate() {
        if !active {
            let mut row = vec![0.0; 2 * n];
            row[index] = 1.0;
            determined_rows.push(row);
            determined_rhs.push(exact(0.0));
        }
    }
    let continuous_rows =
        linearized_constraints(kernel, fields, parameters, relation, time, initial, false)?;
    // Initial equations may fix otherwise free rates at only one Parameter
    // value. They cannot erase the continuous constraint's tangent condition.
    // Skip that condition only when the continuous mass block itself fixes
    // every authored rate; unused algebraic rates remain existential.
    if rank(&continuous_rows, n..2 * n)? == authored.iter().filter(|&&active| active).count() {
        let mut values = solve_initial_action(&determined_rows, &determined_rhs, relation)?;
        values.truncate(n);
        return Ok(values);
    }
    let mass = matrix_proof(&continuous_rows, n..2 * n)?;
    let (continuous, _) = linearized_constraint_actions(
        kernel,
        fields,
        parameters,
        relation,
        (time, initial),
        ConstraintAction::Time {
            mass: &mass,
            mixed: None,
        },
    )?;
    // Recover the physical time direction, including algebraic coordinates
    // whose adapter rates are unused. For constant M, differentiating F=0
    // gives A*r + M*a + F_t=0. The acceleration is only an elimination
    // variable: the existing exact projection owns this compatibility solve.
    let mut physical_rows = Vec::new();
    let mut physical_rhs = Vec::new();
    for row in &continuous {
        let mut mass = row[n..2 * n].to_vec();
        mass.resize(2 * n, 0.0);
        physical_rows.push(mass);
        physical_rhs.push(
            row[n..2 * n]
                .iter()
                .zip(initial.derivative())
                .map(|(&a, &b)| exact(a) * exact(b))
                .sum(),
        );
        physical_rows.push(row[..2 * n].to_vec());
        physical_rhs.push(-exact(*row.last().expect("time column")));
    }
    for (index, &active) in authored.iter().enumerate() {
        if active {
            let mut row = vec![0.0; 2 * n];
            row[index] = 1.0;
            physical_rows.push(row);
            physical_rhs.push(exact(initial.derivative()[index]));
        }
    }
    let physical = solve_initial_action(&physical_rows, &physical_rhs, relation)?;
    let (_, mixed) = linearized_constraint_actions(
        kernel,
        fields,
        parameters,
        relation,
        (time, initial),
        ConstraintAction::Time {
            mass: &mass,
            mixed: Some((&physical[..n], direction)),
        },
    )?;
    // Differentiate both F=0 and its time compatibility equation. With
    // parameter-independent M and the certified zero initial state tangent:
    // M*q=-F_p*dp; A*q+M*b=-D²F[(r,1),dp].
    for (row, value) in continuous.iter().zip(mixed) {
        action_rows.push(row[..2 * n].to_vec());
        rhs.push(-exact(value));
    }
    let mut values = solve_initial_action(&action_rows, &rhs, relation)?;
    values.truncate(n);
    for (value, active) in values.iter_mut().zip(authored) {
        if !active {
            *value = 0.0;
        }
    }
    Ok(values)
}

fn exact(value: f64) -> BigRational {
    BigRational::from_float(value).expect("validated finite initial Jacobian")
}

fn solve_initial_action(
    rows: &[Vec<f64>],
    rhs: &[BigRational],
    relation: Id<kinds::Relation>,
) -> Result<Vec<f64>, Diagnostic> {
    let width = rows.first().map_or(0, Vec::len);
    let matrix = matrix_proof(rows, 0..width)?;
    // The auxiliary acceleration may be free, but the physical rate/action
    // must be uniquely fixed. A basic column witness alone cannot prove this.
    let n = width / 2;
    if matrix.exact_rank() != n + rank(rows, n..width)? {
        return Err(invalid_time(
            relation,
            "initial compatibility equations do not determine a unique rate action",
        ));
    }
    let mut padded_rhs = rhs.to_vec();
    padded_rhs.resize(matrix.dimension(), BigRational::from_integer(0.into()));
    let mut values = matrix
        .compatibility_projection(&[], &padded_rhs)?
        .coefficients()?;
    values.truncate(width);
    // Check rounded coefficients in the original rows. Each correctly rounded
    // coefficient's error is bounded by epsilon*|q| + one minimum subnormal.
    // All products/sums here are exact, so no row-scale tolerance is invented.
    let epsilon = exact(f64::EPSILON);
    let tiny = exact(f64::from_bits(1));
    for (row, target) in rows.iter().zip(rhs) {
        let mut residual = -target;
        let mut bound = BigRational::from_integer(0.into());
        for (&coefficient, &value) in row.iter().zip(&values) {
            let coefficient = exact(coefficient);
            let value = exact(value);
            residual += &coefficient * &value;
            bound += coefficient.abs() * (&epsilon * value.abs() + &tiny);
        }
        if residual.abs() > bound {
            return Err(invalid_time(
                relation,
                "rounded initial rate Parameter tangent leaves the original initial equations",
            ));
        }
    }
    Ok(values)
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
    Ok(linearized_constraint_actions(
        kernel,
        fields,
        parameters,
        relation,
        (initial_time, initial),
        ConstraintAction::Jacobian { include_initial },
    )?
    .0)
}

fn matrix_proof(
    rows: &[Vec<f64>],
    columns: std::ops::Range<usize>,
) -> Result<eqiora_time::ConstantDerivativeMatrixProof, Diagnostic> {
    // Padding changes neither column span nor rank and reuses the existing
    // exact binary-rational matrix proof rather than a second rank threshold.
    let size = rows.len().max(columns.len());
    let mut square = vec![0.0; size * size];
    for (row, values) in rows.iter().enumerate() {
        for (column, source) in columns.clone().enumerate() {
            square[row * size + column] = values[source];
        }
    }
    eqiora_time::ConstantDerivativeMatrixProof::new(size, square)
}

fn rank(rows: &[Vec<f64>], columns: std::ops::Range<usize>) -> Result<usize, Diagnostic> {
    Ok(matrix_proof(rows, columns)?.exact_rank())
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
