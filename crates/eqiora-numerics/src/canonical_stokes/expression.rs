use eqiora_core::{Diagnostic, DimExponents, RawId};
use eqiora_ir::{OperatorApplicationProof, StandardPureOperator};
use eqiora_schema::kernel::typing::TypedResidual;
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, KernelNode};
use eqiora_sem::KernelProgram;

use crate::spatial_expression::{self, ScalarSpatialExpression};

use super::support::{is_field, lowering_error};
use crate::additive_residual::AdditiveResidualView;

/// Exact constitutive carrier selected by one private semantic lowerer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum IncompressibleStressForm {
    SymmetricNewtonian,
}

pub(super) fn momentum_viscous_root(
    residual: &TypedResidual<RawId>,
    root: ExprId,
    velocity: RawId,
    pressure: RawId,
    force_potential: RawId,
    owner: RawId,
) -> Result<Option<ExprId>, Diagnostic> {
    let expression = residual.expression();
    let view = AdditiveResidualView::derive(expression, root, owner)?;
    if view.leaves().len() != 2 {
        return Ok(None);
    }
    let divergences = view
        .leaves()
        .iter()
        .filter_map(|leaf| match expression.node(leaf.value()) {
            Some(ExprNode::Divergence(stress)) => Some((leaf, *stress)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let forcings = view
        .leaves()
        .iter()
        .filter(|leaf| {
            matches!(
                expression.node(leaf.value()),
                Some(ExprNode::Gradient(force_value))
                    if is_field(expression, *force_value, force_potential)
            )
        })
        .collect::<Vec<_>>();
    let ([(divergence, stress)], [forcing]) = (divergences.as_slice(), forcings.as_slice()) else {
        return Ok(None);
    };
    let Some(viscous) =
        newtonian_stress_viscous_root(residual, *stress, velocity, pressure, owner)?
    else {
        return Ok(None);
    };
    if divergence.sign() != forcing.sign() {
        return Err(view.mismatch(
            "Stokes momentum divergence and force-gradient roles must have the same sign, up to whole-equation reversal",
        ));
    }
    Ok(Some(viscous))
}

/// Lower the dynamic-viscosity tape from one exact Newtonian stress value.
///
/// Equation-specific callers use this both for volume momentum recognition
/// and for matching the constitutive outward traction on a boundary.
pub(super) fn lower_newtonian_stress_viscosity(
    program: &KernelProgram,
    residual: &TypedResidual<RawId>,
    stress: ExprId,
    velocity: RawId,
    pressure: RawId,
    owner: RawId,
) -> Result<Option<ScalarSpatialExpression<f64>>, Diagnostic> {
    lower_incompressible_stress_viscosity(
        program,
        residual,
        stress,
        velocity,
        pressure,
        owner,
        IncompressibleStressForm::SymmetricNewtonian,
    )
}

pub(super) fn lower_incompressible_stress_viscosity(
    program: &KernelProgram,
    residual: &TypedResidual<RawId>,
    stress: ExprId,
    velocity: RawId,
    pressure: RawId,
    owner: RawId,
    form: IncompressibleStressForm,
) -> Result<Option<ScalarSpatialExpression<f64>>, Diagnostic> {
    let Some(viscous) =
        incompressible_stress_viscous_root(residual, stress, velocity, pressure, owner, form)?
    else {
        return Ok(None);
    };
    match form {
        IncompressibleStressForm::SymmetricNewtonian => {
            lower_exact_twice_viscosity(program, residual, viscous, velocity, owner)
        }
    }
}

fn incompressible_stress_viscous_root(
    residual: &TypedResidual<RawId>,
    stress: ExprId,
    velocity: RawId,
    pressure: RawId,
    owner: RawId,
    form: IncompressibleStressForm,
) -> Result<Option<ExprId>, Diagnostic> {
    match form {
        IncompressibleStressForm::SymmetricNewtonian => {
            newtonian_stress_viscous_root(residual, stress, velocity, pressure, owner)
        }
    }
}

fn newtonian_stress_viscous_root(
    residual: &TypedResidual<RawId>,
    stress: ExprId,
    velocity: RawId,
    pressure: RawId,
    owner: RawId,
) -> Result<Option<ExprId>, Diagnostic> {
    let expression = residual.expression();
    let Some(ExprNode::Sub(viscous, isotropic_pressure)) = expression.node(stress) else {
        return Ok(None);
    };
    let Some(pressure_proof) = OperatorApplicationProof::classify(
        residual,
        *isotropic_pressure,
        StandardPureOperator::IsotropicLift,
    )
    .map_err(|error| calculus_error(owner, *isotropic_pressure, "isotropic_lift", error))?
    else {
        return Ok(None);
    };
    Ok((is_field(expression, pressure_proof.operand(), pressure)
        && contains_symmetric_gradient(residual, *viscous, velocity, owner)?)
    .then_some(*viscous))
}

pub(super) fn lower_exact_twice_viscosity(
    program: &KernelProgram,
    residual: &TypedResidual<RawId>,
    value: ExprId,
    velocity: RawId,
    owner: RawId,
) -> Result<Option<ScalarSpatialExpression<f64>>, Diagnostic> {
    let spatial_dimension = vector_field_dimension(program, velocity, owner)?;
    let expression = residual.expression();
    let mut factors = Vec::new();
    flatten_product(expression, value, &mut factors);
    if factors.len() < 3 {
        return Ok(None);
    }
    let mut symmetric_gradients = Vec::new();
    for (index, factor) in factors.iter().enumerate() {
        if is_symmetric_gradient(residual, *factor, velocity, owner)? {
            symmetric_gradients.push(index);
        }
    }
    let twos = factors
        .iter()
        .enumerate()
        .filter(|(_, factor)| {
            matches!(
                expression.node(**factor),
                Some(ExprNode::Constant(value))
                    if value.real_scalar_value().is_some_and(|value|
                        value.value() == 2.0 && value.dim() == DimExponents::DIMENSIONLESS)
            )
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if symmetric_gradients.len() != 1 || twos.is_empty() {
        return Ok(None);
    }
    let gradient = symmetric_gradients[0];
    let two = twos[0];
    let mut coefficient_factors = factors
        .into_iter()
        .enumerate()
        .filter_map(|(index, factor)| (index != gradient && index != two).then_some(factor));
    let Some(first) = coefficient_factors.next() else {
        return Ok(None);
    };
    let mut viscosity =
        spatial_expression::lower(program, expression, first, owner, spatial_dimension)?;
    for factor in coefficient_factors {
        viscosity = viscosity.multiply(spatial_expression::lower(
            program,
            expression,
            factor,
            owner,
            spatial_dimension,
        )?);
    }
    if viscosity.constant_value().is_none() {
        return Err(lowering_error(
            owner,
            "dynamic viscosity coefficient has no finite revision-local value",
        ));
    }
    Ok(Some(viscosity))
}

fn vector_field_dimension(
    program: &KernelProgram,
    field: RawId,
    owner: RawId,
) -> Result<usize, Diagnostic> {
    let Some(KernelNode::Field(definition)) = program.node(field) else {
        return Err(lowering_error(owner, "Newtonian velocity Field is missing"));
    };
    let [extent] = definition.shape().extents() else {
        return Err(lowering_error(
            owner,
            "Newtonian velocity must have one vector-shape extent",
        ));
    };
    let dimension = usize::try_from(extent.get()).map_err(|_| {
        lowering_error(
            owner,
            "Newtonian velocity component count exceeds the local target",
        )
    })?;
    if !matches!(dimension, 2 | 3) {
        return Err(lowering_error(
            owner,
            format!("Newtonian velocity requires dimension two or three, received {dimension}"),
        ));
    }
    Ok(dimension)
}

fn flatten_product(expression: &ExprDag, value: ExprId, factors: &mut Vec<ExprId>) {
    if let Some(ExprNode::Mul(left, right)) = expression.node(value) {
        flatten_product(expression, *left, factors);
        flatten_product(expression, *right, factors);
    } else {
        factors.push(value);
    }
}

fn contains_symmetric_gradient(
    residual: &TypedResidual<RawId>,
    value: ExprId,
    velocity: RawId,
    owner: RawId,
) -> Result<bool, Diagnostic> {
    if is_symmetric_gradient(residual, value, velocity, owner)? {
        return Ok(true);
    }
    let Some(ExprNode::Mul(left, right)) = residual.expression().node(value) else {
        return Ok(false);
    };
    Ok(
        contains_symmetric_gradient(residual, *left, velocity, owner)?
            || contains_symmetric_gradient(residual, *right, velocity, owner)?,
    )
}

fn is_symmetric_gradient(
    residual: &TypedResidual<RawId>,
    value: ExprId,
    velocity: RawId,
    owner: RawId,
) -> Result<bool, Diagnostic> {
    let Some(proof) =
        OperatorApplicationProof::classify(residual, value, StandardPureOperator::SymmetricPart)
            .map_err(|error| calculus_error(owner, value, "symmetric_part", error))?
    else {
        return Ok(false);
    };
    Ok(matches!(
        residual.expression().node(proof.operand()),
        Some(ExprNode::Gradient(argument))
            if is_field(residual.expression(), *argument, velocity)
    ))
}

fn calculus_error(
    owner: RawId,
    value: ExprId,
    operation: &str,
    error: impl std::fmt::Display,
) -> Diagnostic {
    lowering_error(
        owner,
        format!(
            "{operation} calculus proof failed at expression node {}: {error}",
            value.index()
        ),
    )
}

pub(super) fn is_divergence_of_field(expression: &ExprDag, value: ExprId, field: RawId) -> bool {
    matches!(
        expression.node(value),
        Some(ExprNode::Divergence(argument)) if is_field(expression, *argument, field)
    )
}

pub(super) fn is_additive_divergence_of_field(
    expression: &ExprDag,
    root: ExprId,
    field: RawId,
    owner: RawId,
) -> Result<bool, Diagnostic> {
    let view = AdditiveResidualView::derive(expression, root, owner)?;
    Ok(matches!(
        view.leaves(),
        [leaf]
            if matches!(
                expression.node(leaf.value()),
                Some(ExprNode::Divergence(argument)) if is_field(expression, *argument, field)
            )
    ))
}

pub(super) fn load_definition_root(
    expression: &ExprDag,
    root: ExprId,
    field: RawId,
) -> Option<ExprId> {
    let ExprNode::Sub(left, right) = expression.node(root)? else {
        return None;
    };
    is_field(expression, *left, field).then_some(*right)
}

pub(super) fn additive_load_definition_root(
    expression: &ExprDag,
    root: ExprId,
    field: RawId,
    owner: RawId,
) -> Result<Option<ExprId>, Diagnostic> {
    match expression.node(root) {
        Some(ExprNode::Sub(left, right)) if is_field(expression, *left, field) => {
            return Ok(Some(*right));
        }
        Some(ExprNode::Sub(left, right)) if is_field(expression, *right, field) => {
            return Ok(Some(*left));
        }
        Some(ExprNode::Add(left, right)) if is_field(expression, *left, field) => {
            if let Some(ExprNode::Neg(source)) = expression.node(*right) {
                return Ok(Some(*source));
            }
        }
        Some(ExprNode::Add(left, right)) => match expression.node(*left) {
            Some(ExprNode::Neg(negated_field)) if is_field(expression, *negated_field, field) => {
                return Ok(Some(*right));
            }
            _ => {}
        },
        _ => {}
    }
    let view = AdditiveResidualView::derive(expression, root, owner)?;
    if view.leaves().len() != 2 {
        return Ok(None);
    }
    let fields = view
        .leaves()
        .iter()
        .filter(|leaf| is_field(expression, leaf.value(), field))
        .collect::<Vec<_>>();
    let [field_leaf] = fields.as_slice() else {
        return Ok(None);
    };
    let sources = view
        .leaves()
        .iter()
        .filter(|leaf| leaf.value() != field_leaf.value())
        .collect::<Vec<_>>();
    let [source] = sources.as_slice() else {
        return Ok(None);
    };
    if !field_leaf.sign().is_opposite(source.sign()) {
        return Err(view.mismatch(
            "Stokes field-definition roles must have opposite signs, up to whole-equation reversal",
        ));
    }
    Ok(Some(source.value()))
}
