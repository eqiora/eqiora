//! Prescribed scalar and vector components shared by boundaries and initial states.
use super::*;
use eqiora_core::{Id, entity::kinds};
use eqiora_ir::{OperatorApplicationProof, StandardPureOperator};
use eqiora_schema::kernel::{ExprId, ExprNode, SymbolRef, typing::TypedResidual};

#[derive(Debug, Clone, PartialEq)]
pub(in crate::form_compiler) enum PrescribedDatum<S: Coefficient> {
    Components(Vec<Data<S>>),
    NormalMultiple(Data<S>),
    ParameterComponents {
        parameter: RawId,
        values: Vec<S>,
        scale: Data<S>,
    },
}

impl<S: Coefficient> PrescribedDatum<S> {
    pub(in crate::form_compiler) fn derive(
        context: &Context<'_, S>,
        typed: &TypedResidual<RawId>,
        value_type: &ValueType,
        expression: Option<ExprId>,
    ) -> Result<Self, Diagnostic> {
        let count = components(value_type, context.dimension)?;
        Ok(match expression {
            None => PrescribedDatum::Components(vec![
                Data::constant(
                    context.dimension,
                    <S as From<f64>>::from(0.0)
                );
                count
            ]),
            Some(id) if value_type.shape().is_scalar() => {
                PrescribedDatum::Components(vec![context.data(id, 0)?])
            }
            Some(id) => {
                let (inner, scale) = vector_datum_operand(context, id)?;
                match context.dag.node(inner) {
                    Some(ExprNode::Symbol(SymbolRef::Parameter(parameter))) => {
                        let literal =
                            context
                                .program
                                .typed_value(parameter.erase())
                                .ok_or_else(|| {
                                    invalid("boundary Parameter has no revision-local value")
                                })?;
                        if literal.component_count() != count
                            || literal.value_type().shape() != value_type.shape()
                            || literal.value_type().frame() != value_type.frame()
                        {
                            return Err(invalid(
                                "boundary Parameter has a foreign value shape or frame",
                            ));
                        }
                        let values = (0..count).map(|component| {
                                let (real, imag) = literal.component(component).expect("validated component");
                                let mut value = <S as From<f64>>::from(real);
                                if imag != 0.0 {
                                    value += S::imaginary_unit().ok_or_else(|| {
                                        invalid("complex boundary Parameter requires complex coefficients")
                                    })? * <S as From<f64>>::from(imag);
                                }
                                Ok(value)
                            }).collect::<Result<Vec<_>, Diagnostic>>()?;
                        PrescribedDatum::ParameterComponents {
                            parameter: parameter.erase(),
                            values,
                            scale,
                        }
                    }
                    Some(ExprNode::Gradient(potential)) => {
                        let potential = context.data(*potential, 0)?;
                        let primal = potential.clone().multiply(Data::constant(
                            context.dimension,
                            <S as From<f64>>::from(0.0),
                        ));
                        PrescribedDatum::Components(
                            (0..count)
                                .map(|axis| {
                                    Ok(primal
                                        .clone()
                                        .add(
                                            potential
                                                .coordinate_derivative(axis, context.dimension)?,
                                        )
                                        .multiply(scale.clone()))
                                })
                                .collect::<Result<_, Diagnostic>>()?,
                        )
                    }
                    Some(ExprNode::NormalComponent { value: tensor, .. }) => {
                        let proof = OperatorApplicationProof::classify(
                            typed,
                            *tensor,
                            StandardPureOperator::IsotropicLift,
                        )
                        .map_err(|_| {
                            invalid("boundary normal datum lacks an exact isotropic-lift proof")
                        })?
                        .ok_or_else(|| {
                            invalid("boundary normal datum requires an isotropic lift")
                        })?;
                        PrescribedDatum::NormalMultiple(
                            context.data(proof.operand(), 0)?.multiply(scale),
                        )
                    }
                    _ => {
                        return Err(invalid(
                            "vector boundary datum requires a potential gradient, vector Parameter or isotropic normal lift",
                        ));
                    }
                }
            }
        })
    }
    pub(in crate::form_compiler) fn evaluate(
        &self,
        point: &[f64],
        normal: &[f64],
    ) -> Result<Vec<S>, Diagnostic> {
        match self {
            PrescribedDatum::ParameterComponents { values, scale, .. } => {
                let scale = scale.evaluate(point)?;
                Ok(values.iter().map(|value| *value * scale).collect())
            }
            PrescribedDatum::Components(values) => {
                values.iter().map(|value| value.evaluate(point)).collect()
            }
            PrescribedDatum::NormalMultiple(value) => {
                let value = value.evaluate(point)?;
                if normal.len() != point.len() || normal.iter().any(|value| !value.is_finite()) {
                    return Err(invalid("boundary datum requires the exact parent normal"));
                }
                Ok(normal
                    .iter()
                    .map(|normal| <S as From<f64>>::from(*normal) * value)
                    .collect())
            }
        }
    }

    pub(in crate::form_compiler) fn bind_parameter_point(
        &mut self,
        fields: &[Id<kinds::Parameter>],
        values: &[S],
    ) -> Result<(), Diagnostic> {
        match self {
            PrescribedDatum::ParameterComponents { .. } => {
                return Err(invalid(
                    "vector boundary Parameters require component-aware rebinding",
                ));
            }
            PrescribedDatum::Components(components) => {
                for component in components {
                    *component = component.bind_parameter_point(fields, values)?;
                }
            }
            PrescribedDatum::NormalMultiple(value) => {
                *value = value.bind_parameter_point(fields, values)?
            }
        }
        Ok(())
    }
}

/// Separate prescribed scalar factors without differentiating them as part of
/// a potential, preserving the distinction between a grad(g) and grad(a*g).
fn vector_datum_operand<S: Coefficient>(
    context: &Context<'_, S>,
    mut id: ExprId,
) -> Result<(ExprId, Data<S>), Diagnostic> {
    let mut scale = Data::constant(context.dimension, <S as From<f64>>::from(1.0));
    for depth in 0..128 {
        match context.dag.node(id) {
            Some(ExprNode::Trace { value, .. }) => id = *value,
            Some(ExprNode::Neg(value)) => {
                scale = scale.multiply(Data::constant(
                    context.dimension,
                    <S as From<f64>>::from(-1.0),
                ));
                id = *value;
            }
            Some(ExprNode::Mul(left, right)) => {
                if let Ok(factor) = context.data(*left, depth + 1) {
                    scale = scale.multiply(factor);
                    id = *right;
                } else {
                    scale = scale.multiply(context.data(*right, depth + 1)?);
                    id = *left;
                }
            }
            Some(ExprNode::Div(value, divisor)) => {
                scale = scale.divide(context.data(*divisor, depth + 1)?);
                id = *value;
            }
            _ => return Ok((id, scale)),
        }
    }
    Err(invalid("vector boundary datum nesting exceeds 128"))
}
