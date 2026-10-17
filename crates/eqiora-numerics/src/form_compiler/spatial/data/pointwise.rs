//! Prescribed components use the existing typed component Operator IR.
use super::*;
use eqiora_core::{Id, ScalarDomain, entity::kinds};
use eqiora_ir::{
    ComponentScalarRow, ComponentScalarization, DifferentiationRole, LinearizedRelation,
    RelationTangent,
};

#[derive(Debug, Clone, PartialEq)]
enum Input<S: Coefficient> {
    Scalar {
        data: Data<S>,
        imaginary: bool,
    },
    ParameterComponent {
        parameter: Id<kinds::Parameter>,
        value: f64,
    },
}

impl<S: Coefficient> Input<S> {
    fn evaluate(&self, point: &[f64]) -> Result<f64, Diagnostic> {
        match self {
            Self::Scalar { data, imaginary } => {
                let value = data.evaluate(point)?;
                Ok(if *imaginary { value.im() } else { value.re() })
            }
            Self::ParameterComponent { value, .. } => Ok(*value),
        }
    }
    fn spatial(&self) -> bool {
        matches!(self, Self::Scalar { data, .. } if data.spatial())
    }
    fn bind(&self, fields: &[Id<kinds::Parameter>], values: &[S]) -> Result<Self, Diagnostic> {
        Ok(match self {
            Self::Scalar { data, imaginary } => Self::Scalar {
                data: data.bind_parameter_point(fields, values)?,
                imaginary: *imaginary,
            },
            Self::ParameterComponent { parameter, .. } if fields.contains(parameter) => {
                return Err(super::super::invalid(
                    "structured Parameter rebinding requires complete component coordinates",
                ));
            }
            value => value.clone(),
        })
    }
    fn derivative(&self, axis: usize, dimension: usize) -> Result<Self, Diagnostic> {
        Ok(match self {
            Self::Scalar { data, imaginary } => Self::Scalar {
                data: data.coordinate_derivative(axis, dimension)?,
                imaginary: *imaginary,
            },
            Self::ParameterComponent { .. } => Self::Scalar {
                data: Data::constant(dimension, <S as From<f64>>::from(0.)),
                imaginary: false,
            },
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct PointwiseData<S: Coefficient> {
    row: ComponentScalarRow,
    inputs: Vec<Input<S>>,
    tangent: Option<Vec<Input<S>>>,
}

impl<S: Coefficient> PointwiseData<S> {
    pub(super) fn components(
        context: &Context<'_, S>,
        id: ExprId,
        depth: usize,
    ) -> Result<Vec<Data<S>>, Diagnostic> {
        let typed = super::super::super::scalar::typed_relation(context.program, context.owner)?;
        if typed.expression() != context.dag {
            return Err(super::super::invalid(
                "pointwise coefficient requires its exact typed expression",
            ));
        }
        let ty = &typed
            .node_type(id)
            .ok_or_else(|| super::super::invalid("missing typed coefficient"))?
            .value_type;
        if !matches!(
            ty.scalar_domain(),
            ScalarDomain::Real | ScalarDomain::Complex
        ) || (ty.scalar_domain() == ScalarDomain::Complex && S::DOMAIN != ScalarDomain::Complex)
        {
            return Err(super::super::invalid(
                "pointwise coefficient requires its complete numeric scalar domain",
            ));
        }
        let lowered = ComponentScalarization::lower_selected(&typed, &[id])?;
        let rows = lowered
            .rows()
            .iter()
            .map(|row| {
                Self::from_row(context, row, depth)
                    .map(|data| Data(Arc::new(Node::Pointwise(data))))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if ty.scalar_domain() == ScalarDomain::Real {
            return Ok(rows);
        }
        let unit = Data::constant(
            context.dimension,
            S::imaginary_unit().expect("checked complex domain"),
        );
        let (pairs, remainder) = rows.as_chunks::<2>();
        if !remainder.is_empty() {
            return Err(super::super::invalid(
                "complex coefficient requires paired real/imaginary rows",
            ));
        }
        Ok(pairs
            .iter()
            .map(|[real, imaginary]| real.clone().add(imaginary.clone().multiply(unit.clone())))
            .collect())
    }

    fn from_row(
        context: &Context<'_, S>,
        row: &ComponentScalarRow,
        depth: usize,
    ) -> Result<Self, Diagnostic> {
        let inputs = row
            .symbols()
            .iter()
            .map(|coordinate| {
                if let SymbolRef::Parameter(parameter) = coordinate.symbol() {
                    let literal =
                        context
                            .program
                            .typed_value(parameter.erase())
                            .ok_or_else(|| {
                                super::super::invalid("coefficient Parameter has no exact value")
                            })?;
                    if !coordinate.component_index().is_empty()
                        || (literal.value_type().scalar_domain() == ScalarDomain::Complex
                            && S::DOMAIN != ScalarDomain::Complex)
                    {
                        let index = literal
                            .value_type()
                            .shape()
                            .flat_index(coordinate.component_index())
                            .ok_or_else(|| {
                                super::super::invalid(
                                    "coefficient Parameter component differs from its exact shape",
                                )
                            })?;
                        let (real, imaginary) = literal.component(index).ok_or_else(|| {
                            super::super::invalid("coefficient Parameter component is unavailable")
                        })?;
                        return Ok(Input::ParameterComponent {
                            parameter,
                            value: if coordinate.is_imaginary() {
                                imaginary
                            } else {
                                real
                            },
                        });
                    }
                }
                if !coordinate.component_index().is_empty() {
                    return Err(super::super::invalid(
                        "structured prescribed Field requires its retained component expressions",
                    ));
                }
                let index = context.dag.nodes().iter().position(|node|
                matches!(node, ExprNode::Symbol(symbol) if *symbol == coordinate.symbol()))
                .ok_or_else(|| super::super::invalid("pointwise input has no retained symbol"))?;
                Ok(Input::Scalar {
                    data: context.data(
                        context.dag.node_id(index as u32).expect("symbol index"),
                        depth + 1,
                    )?,
                    imaginary: coordinate.is_imaginary(),
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        Ok(Self {
            row: row.clone(),
            inputs,
            tangent: None,
        })
    }

    pub(super) fn bind_parameter_point(
        &self,
        fields: &[Id<kinds::Parameter>],
        values: &[S],
    ) -> Result<Self, Diagnostic> {
        let bind = |inputs: &[Input<S>]| {
            inputs
                .iter()
                .map(|input| input.bind(fields, values))
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(Self {
            row: self.row.clone(),
            inputs: bind(&self.inputs)?,
            tangent: self.tangent.as_deref().map(bind).transpose()?,
        })
    }
    pub(super) fn spatial(&self) -> bool {
        self.inputs
            .iter()
            .chain(self.tangent.iter().flatten())
            .any(Input::spatial)
    }
    pub(super) fn coordinate_derivative(
        &self,
        axis: usize,
        dimension: usize,
    ) -> Result<Self, Diagnostic> {
        if self.tangent.is_some() {
            return Err(super::super::invalid(
                "pointwise coefficient admits one spatial derivative",
            ));
        }
        let tangent = self
            .inputs
            .iter()
            .map(|input| input.derivative(axis, dimension))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            row: self.row.clone(),
            inputs: self.inputs.clone(),
            tangent: Some(tangent),
        })
    }
    pub(super) fn evaluate(&self, point: &[f64]) -> Result<S, Diagnostic> {
        let evaluate = |inputs: &[Input<S>]| {
            inputs
                .iter()
                .map(|input| input.evaluate(point))
                .collect::<Result<Vec<_>, _>>()
        };
        let inputs = evaluate(&self.inputs)?;
        let value = if let Some(tangent) = &self.tangent {
            let tangent = evaluate(tangent)?;
            let roles = vec![DifferentiationRole::Unknown; inputs.len()];
            let action = self.row.linearize(&inputs, &roles)?;
            let mut result = [0.0];
            action.jvp(RelationTangent::Unknown(&tangent), &mut result)?;
            result[0]
        } else {
            self.row.evaluate(&inputs)?
        };
        Ok(<S as From<f64>>::from(value))
    }
}
