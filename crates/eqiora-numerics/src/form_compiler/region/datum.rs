//! Prescribed scalar and vector components shared by boundaries and initial states.
use super::*;
use eqiora_ir::{OperatorApplicationProof, StandardPureOperator};
use eqiora_schema::kernel::{ExprId, ExprNode, typing::TypedResidual};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PrescribedDatum<S: Coefficient> {
    Components(Vec<Data<S>>),
    NormalMultiple(Data<S>),
    Neg(Box<Self>),
    Multiply {
        value: Box<Self>,
        scalar: Data<S>,
        scalar_first: bool,
    },
    Divide {
        value: Box<Self>,
        scalar: Data<S>,
    },
    Add(Box<Self>, Box<Self>),
    Derivative {
        primal: Box<Self>,
        tangent: Box<Self>,
    },
}

impl<S: Coefficient> PrescribedDatum<S> {
    /// Bind one exact boundary RHS using a previously authenticated optional carrier.
    pub(crate) fn from_boundary(
        program: &KernelProgram,
        relation: RawId,
        rhs: ExprId,
        carrier: Option<(RawId, RawId, ExprId)>,
        dimension: usize,
    ) -> Result<
        (
            Self,
            Vec<eqiora_core::Id<eqiora_core::entity::kinds::Parameter>>,
        ),
        Diagnostic,
    > {
        let mut coefficients = BTreeMap::new();
        let mut parameters = BTreeMap::new();
        if let Some((field, definition, source)) = carrier {
            let typed = super::super::scalar::typed_relation(program, definition)?;
            let Some(KernelNode::Field(field_node)) = program.node(field) else {
                return Err(invalid("prescribed carrier is not an exact Field"));
            };
            let context = Context {
                time_s: None,
                program,
                dag: typed.expression(),
                owner: definition,
                dimension,
                coefficients: &coefficients,
            };
            let value = super::super::spatial::data::Components::prescribed(
                &context,
                field_node.value_type(),
                Some(source),
            )?;
            coefficients.insert(field, value);
            collect_parameters(typed.expression(), source, &mut parameters)?;
        }
        let typed = super::super::scalar::typed_relation(program, relation)?;
        let ty = typed
            .node_type(rhs)
            .ok_or_else(|| invalid("boundary RHS has no exact type"))?;
        let context = Context {
            time_s: None,
            program,
            dag: typed.expression(),
            owner: relation,
            dimension,
            coefficients: &coefficients,
        };
        let datum = Self::derive(&context, &typed, &ty.value_type, Some(rhs))?;
        collect_parameters(typed.expression(), rhs, &mut parameters)?;
        let mut parameters = parameters.into_values().collect::<Vec<_>>();
        parameters.sort_by_key(eqiora_core::Id::ulid);
        Ok((datum, parameters))
    }

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
            Some(id) => Self::derive_vector(context, typed, id, 0)?,
        })
    }

    fn derive_vector(
        context: &Context<'_, S>,
        typed: &TypedResidual<RawId>,
        id: ExprId,
        depth: usize,
    ) -> Result<Self, Diagnostic> {
        if depth > 128 {
            return Err(invalid("vector boundary datum nesting exceeds 128"));
        }
        let nested = |id| Self::derive_vector(context, typed, id, depth + 1);
        let scalar = |id| {
            typed.node_type(id).is_some_and(|ty| {
                ty.value_type.array_rank() == 0
                    && ty.shape().is_scalar()
                    && ty.frame() == eqiora_core::ValueFrame::Invariant
            })
        };
        Ok(match context.dag.node(id) {
            Some(ExprNode::Trace { value, .. }) => {
                let ty = typed
                    .node_type(id)
                    .ok_or_else(|| invalid("missing typed boundary trace"))?;
                if !matches!(
                    ty.support,
                    Some(
                        eqiora_schema::kernel::typing::SpatialSupport::Boundary { .. }
                            | eqiora_schema::kernel::typing::SpatialSupport::PhysicalInterface { .. }
                    )
                ) {
                    return Err(invalid(
                        "prescribed trace lacks authenticated boundary support",
                    ));
                }
                nested(*value)?
            }
            Some(ExprNode::Neg(value)) => Self::Neg(Box::new(nested(*value)?)),
            Some(ExprNode::Mul(left, right)) if scalar(*left) && !scalar(*right) => {
                Self::Multiply {
                    scalar: context.data(*left, 0)?,
                    value: Box::new(nested(*right)?),
                    scalar_first: true,
                }
            }
            Some(ExprNode::Mul(left, right)) if scalar(*right) && !scalar(*left) => {
                Self::Multiply {
                    value: Box::new(nested(*left)?),
                    scalar: context.data(*right, 0)?,
                    scalar_first: false,
                }
            }
            Some(ExprNode::Div(value, divisor)) if scalar(*divisor) => Self::Divide {
                value: Box::new(nested(*value)?),
                scalar: context.data(*divisor, 0)?,
            },
            Some(ExprNode::Gradient(potential)) => {
                let potential = context.data(*potential, 0)?;
                let primal = potential.clone().multiply(Data::constant(
                    context.dimension,
                    <S as From<f64>>::from(0.),
                ));
                Self::Components(
                    (0..context.dimension)
                        .map(|axis| {
                            Ok(primal
                                .clone()
                                .add(potential.coordinate_derivative(axis, context.dimension)?))
                        })
                        .collect::<Result<_, Diagnostic>>()?,
                )
            }
            Some(ExprNode::NormalVector { .. }) => Self::NormalMultiple(Data::constant(
                context.dimension,
                <S as From<f64>>::from(1.),
            )),
            Some(ExprNode::NormalComponent { value: tensor, .. }) => {
                let proof = OperatorApplicationProof::classify(
                    typed,
                    *tensor,
                    StandardPureOperator::IsotropicLift,
                )
                .map_err(|_| invalid("boundary normal datum lacks an exact isotropic-lift proof"))?
                .ok_or_else(|| invalid("boundary normal datum requires an isotropic lift"))?;
                Self::NormalMultiple(context.data(proof.operand(), 0)?)
            }
            _ => Self::Components(context.components(id, 0)?),
        })
    }

    pub(crate) fn requires_normal(&self) -> bool {
        match self {
            Self::Components(_) => false,
            Self::NormalMultiple(_) => true,
            Self::Neg(value) | Self::Multiply { value, .. } | Self::Divide { value, .. } => {
                value.requires_normal()
            }
            Self::Add(left, right) => left.requires_normal() || right.requires_normal(),
            Self::Derivative { primal, tangent } => {
                primal.requires_normal() || tangent.requires_normal()
            }
        }
    }

    pub(crate) fn evaluate(&self, point: &[f64], normal: &[f64]) -> Result<Vec<S>, Diagnostic> {
        self.evaluate_depth(point, normal, 0)
    }

    fn evaluate_depth(
        &self,
        point: &[f64],
        normal: &[f64],
        depth: usize,
    ) -> Result<Vec<S>, Diagnostic> {
        if depth > 128 {
            return Err(invalid("vector boundary datum nesting exceeds 128"));
        }
        let evaluate = |value: &Self| value.evaluate_depth(point, normal, depth + 1);
        let values = match self {
            Self::Components(values) => values
                .iter()
                .map(|value| value.evaluate(point))
                .collect::<Result<Vec<_>, _>>()?,
            Self::NormalMultiple(value) => {
                let value = value.evaluate(point)?;
                if normal.len() != point.len() || normal.iter().any(|value| !value.is_finite()) {
                    return Err(invalid("boundary datum requires the exact parent normal"));
                }
                normal
                    .iter()
                    .map(|normal| <S as From<f64>>::from(*normal) * value)
                    .collect()
            }
            Self::Neg(value) => evaluate(value)?.into_iter().map(|value| -value).collect(),
            Self::Multiply {
                value,
                scalar,
                scalar_first,
            } => {
                let (values, scalar) = if *scalar_first {
                    let scalar = scalar.evaluate(point)?;
                    (evaluate(value)?, scalar)
                } else {
                    (evaluate(value)?, scalar.evaluate(point)?)
                };
                values
                    .into_iter()
                    .map(|value| {
                        if *scalar_first {
                            scalar * value
                        } else {
                            value * scalar
                        }
                    })
                    .collect()
            }
            Self::Divide { value, scalar } => {
                let values = evaluate(value)?;
                let scalar = scalar.evaluate(point)?;
                values.into_iter().map(|value| value / scalar).collect()
            }
            Self::Add(left, right) => {
                let left = evaluate(left)?;
                let right = evaluate(right)?;
                if left.len() != right.len() {
                    return Err(invalid("boundary datum component counts differ"));
                }
                left.into_iter().zip(right).map(|(a, b)| a + b).collect()
            }
            Self::Derivative { primal, tangent } => {
                evaluate(primal)?;
                evaluate(tangent)?
            }
        };
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid("non-finite vector boundary datum"));
        }
        Ok(values)
    }

    /// Differentiate prescribed inputs while holding the supplied geometry/normal fixed.
    pub(in crate::form_compiler) fn input_partial(
        &mut self,
        coordinate: &eqiora_ir::ScalarSymbolCoordinate,
        dimension: usize,
    ) -> Result<(), Diagnostic> {
        let tangent = self.partial_tree(coordinate, dimension, 0)?;
        *self = Self::Derivative {
            primal: Box::new(self.clone()),
            tangent: Box::new(tangent),
        };
        Ok(())
    }

    fn partial_tree(
        &self,
        coordinate: &eqiora_ir::ScalarSymbolCoordinate,
        dimension: usize,
        depth: usize,
    ) -> Result<Self, Diagnostic> {
        if depth > 128 {
            return Err(invalid("vector boundary derivative nesting exceeds 128"));
        }
        if self.constant_in_direction(coordinate, depth)? {
            return Ok(Self::Derivative {
                primal: Box::new(self.clone()),
                tangent: Box::new(Self::Components(vec![
                    Data::constant(
                        dimension,
                        <S as From<f64>>::from(0.)
                    );
                    self.component_count(dimension)
                ])),
            });
        }
        let partial = |value: &Self| value.partial_tree(coordinate, dimension, depth + 1);
        let data_partial =
            |value: &Data<S>| value.derivative(&|input| f64::from(input == coordinate), dimension);
        let data_constant =
            |value: &Data<S>| value.constant_in_direction(&|input| f64::from(input == coordinate));
        let multiply = |value: Self, scalar: Data<S>, scalar_first| Self::Multiply {
            value: Box::new(value),
            scalar,
            scalar_first,
        };
        Ok(match self {
            Self::Components(values) => {
                Self::Components(values.iter().map(data_partial).collect::<Result<_, _>>()?)
            }
            Self::NormalMultiple(value) => Self::NormalMultiple(data_partial(value)?),
            Self::Neg(value) => Self::Neg(Box::new(partial(value)?)),
            Self::Multiply {
                value,
                scalar,
                scalar_first,
            } => {
                if data_constant(scalar)? {
                    return Ok(multiply(partial(value)?, scalar.clone(), *scalar_first));
                }
                if value.constant_in_direction(coordinate, depth + 1)? {
                    return Ok(multiply(
                        (**value).clone(),
                        data_partial(scalar)?,
                        *scalar_first,
                    ));
                }
                let (left, right) = if *scalar_first {
                    (
                        multiply((**value).clone(), data_partial(scalar)?, true),
                        multiply(partial(value)?, scalar.clone(), true),
                    )
                } else {
                    (
                        multiply(partial(value)?, scalar.clone(), false),
                        multiply((**value).clone(), data_partial(scalar)?, false),
                    )
                };
                Self::Add(Box::new(left), Box::new(right))
            }
            Self::Divide { value, scalar } if data_constant(scalar)? => Self::Divide {
                value: Box::new(partial(value)?),
                scalar: scalar.clone(),
            },
            Self::Divide { value, scalar } => Self::Divide {
                value: Box::new(Self::Add(
                    Box::new(multiply(partial(value)?, scalar.clone(), false)),
                    Box::new(Self::Neg(Box::new(multiply(
                        (**value).clone(),
                        data_partial(scalar)?,
                        false,
                    )))),
                )),
                scalar: scalar.clone().multiply(scalar.clone()),
            },
            Self::Add(left, right) => {
                Self::Add(Box::new(partial(left)?), Box::new(partial(right)?))
            }
            Self::Derivative { tangent, .. } => Self::Derivative {
                primal: Box::new(self.clone()),
                tangent: Box::new(partial(tangent)?),
            },
        })
    }

    fn component_count(&self, dimension: usize) -> usize {
        match self {
            Self::Components(values) => values.len(),
            Self::NormalMultiple(_) => dimension,
            Self::Neg(value) | Self::Multiply { value, .. } | Self::Divide { value, .. } => {
                value.component_count(dimension)
            }
            Self::Add(left, _) => left.component_count(dimension),
            Self::Derivative { tangent, .. } => tangent.component_count(dimension),
        }
    }

    /// Activity is structural; an incidental sampled zero never erases a branch.
    fn constant_in_direction(
        &self,
        coordinate: &eqiora_ir::ScalarSymbolCoordinate,
        depth: usize,
    ) -> Result<bool, Diagnostic> {
        if depth > 128 {
            return Err(invalid("vector boundary activity nesting exceeds 128"));
        }
        let constant = |value: &Self| value.constant_in_direction(coordinate, depth + 1);
        let data_constant =
            |value: &Data<S>| value.constant_in_direction(&|input| f64::from(input == coordinate));
        Ok(match self {
            Self::Components(values) => {
                let mut result = true;
                for value in values {
                    result &= data_constant(value)?;
                }
                result
            }
            Self::NormalMultiple(value) => data_constant(value)?,
            Self::Neg(value) => constant(value)?,
            Self::Multiply { value, scalar, .. } | Self::Divide { value, scalar } => {
                constant(value)? && data_constant(scalar)?
            }
            Self::Add(left, right) => constant(left)? && constant(right)?,
            Self::Derivative { tangent, .. } => constant(tangent)?,
        })
    }

    pub(in crate::form_compiler) fn bind_inputs<'a>(
        &mut self,
        point: impl Into<crate::form_compiler::spatial::InputPoint<'a>>,
    ) -> Result<(), Diagnostic> {
        let point = point.into();
        self.map_data(&|value| value.bind_inputs(point))
    }

    pub(in crate::form_compiler) fn map_data(
        &mut self,
        map: &impl Fn(&Data<S>) -> Result<Data<S>, Diagnostic>,
    ) -> Result<(), Diagnostic> {
        match self {
            Self::Components(components) => {
                for component in components {
                    *component = map(component)?;
                }
            }
            Self::NormalMultiple(value) => *value = map(value)?,
            Self::Neg(value) => value.map_data(map)?,
            Self::Multiply { value, scalar, .. } | Self::Divide { value, scalar } => {
                value.map_data(map)?;
                *scalar = map(scalar)?;
            }
            Self::Add(left, right) => {
                left.map_data(map)?;
                right.map_data(map)?;
            }
            Self::Derivative { primal, tangent } => {
                primal.map_data(map)?;
                tangent.map_data(map)?;
            }
        }
        Ok(())
    }
}

fn collect_parameters(
    dag: &eqiora_schema::kernel::ExprDag,
    root: ExprId,
    parameters: &mut BTreeMap<RawId, eqiora_core::Id<eqiora_core::entity::kinds::Parameter>>,
) -> Result<(), Diagnostic> {
    let mut pending = vec![root];
    let mut seen = BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id.index()) {
            continue;
        }
        let node = dag
            .node(id)
            .ok_or_else(|| invalid("prescribed input references an absent node"))?;
        if let ExprNode::Symbol(eqiora_schema::kernel::SymbolRef::Parameter(id)) = node {
            parameters.insert(id.erase(), *id);
        }
        node.try_for_each_operand(|operand| {
            pending.push(operand);
            Ok::<_, Diagnostic>(())
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
