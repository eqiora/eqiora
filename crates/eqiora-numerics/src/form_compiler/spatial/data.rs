use std::collections::BTreeMap;
use std::sync::Arc;

use eqiora_core::{Diagnostic, RawId};
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, SymbolRef};
use eqiora_sem::KernelProgram;

use crate::spatial_expression::{self, Coefficient, validate_coordinates as validate_point};
use eqiora_ir::ScalarSymbolCoordinate;
use input::Input;

/// Directions of physical coordinates and retained real Parameter components.
/// These coefficient partials hold Time and geometry bindings fixed; geometry
/// maps and their own Parameter dependence have separate partial actions.
type Direction<'a> = &'a dyn Fn(&ScalarSymbolCoordinate) -> f64;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Data<S: Coefficient>(Arc<Node<S>>);

mod activity;
pub(in crate::form_compiler) mod affine;
mod coefficient;
mod components;
mod coordinates;
pub(in crate::form_compiler) use components::Components;
mod degree;
pub(in crate::form_compiler) mod field_components;
mod input;
mod pointwise;

#[derive(Debug, PartialEq)]
enum Node<S: Coefficient> {
    Constant {
        dimension: usize,
        value: S,
    },
    Input {
        dimension: usize,
        coordinate: ScalarSymbolCoordinate,
        input: Input<S>,
    },
    Pointwise(pointwise::PointwiseData<S>),
    Affine(affine::AffineData<S>),
    Chart(Data<S>, super::motion::UniformChart),
    // Retain primal domains and Parameter identities when an input derivative is zero.
    Derivative(Data<S>, Data<S>),
    Add(Data<S>, Data<S>),
    Mul(Data<S>, Data<S>),
    Div(Data<S>, Data<S>),
    Part {
        imaginary: bool,
        data: Data<S>,
    },
}

impl<S: Coefficient> Data<S> {
    pub(in crate::form_compiler) fn bind_parameter_point(
        &self,
        point: &crate::parameter_point::ParameterPoint,
    ) -> Result<Self, Diagnostic> {
        let bind = |data: &Self| data.bind_parameter_point(point);
        Ok(Self(Arc::new(match self.0.as_ref() {
            Node::Constant { dimension, value } => Node::Constant {
                dimension: *dimension,
                value: *value,
            },
            Node::Input {
                dimension,
                coordinate,
                input,
            } => Node::Input {
                dimension: *dimension,
                coordinate: coordinate.clone(),
                input: input.bind(coordinate, point)?,
            },
            Node::Pointwise(data) => Node::Pointwise(data.bind_parameter_point(point)?),
            Node::Affine(data) => Node::Affine(data.bind_parameter_point(point)?),
            Node::Chart(data, chart) => Node::Chart(bind(data)?, chart.clone()),
            Node::Derivative(primal, tangent) => Node::Derivative(bind(primal)?, bind(tangent)?),
            Node::Add(a, b) => Node::Add(bind(a)?, bind(b)?),
            Node::Mul(a, b) => Node::Mul(bind(a)?, bind(b)?),
            Node::Div(a, b) => Node::Div(bind(a)?, bind(b)?),
            Node::Part { imaginary, data } => Node::Part {
                imaginary: *imaginary,
                data: bind(data)?,
            },
        })))
    }

    pub(in crate::form_compiler) fn on_uniform_chart(
        &self,
        chart: &super::motion::UniformChart,
    ) -> Self {
        Self(Arc::new(Node::Chart(self.clone(), chart.clone())))
    }
    pub(in crate::form_compiler) fn constant(dimension: usize, value: S) -> Self {
        Self(Arc::new(Node::Constant { dimension, value }))
    }
    pub(in crate::form_compiler) fn add(self, right: Self) -> Self {
        let zero = |data: &Self| matches!(data.0.as_ref(), Node::Constant { value, .. } if *value == <S as From<f64>>::from(0.0));
        if zero(&self) {
            return right;
        }
        if zero(&right) {
            return self;
        }
        Self(Arc::new(Node::Add(self, right)))
    }
    pub(in crate::form_compiler) fn multiply(self, right: Self) -> Self {
        Self(Arc::new(Node::Mul(self, right)))
    }
    pub(in crate::form_compiler) fn divide(self, right: Self) -> Self {
        Self(Arc::new(Node::Div(self, right)))
    }
    /// Differentiate coefficient data through its shared inputs and exact rules.
    pub(in crate::form_compiler) fn coordinate_derivative(
        &self,
        axis: usize,
        dimension: usize,
    ) -> Result<Self, Diagnostic> {
        if axis >= dimension {
            return Err(super::invalid("gradient axis exceeds physical dimension"));
        }
        self.derivative(
            &|coordinate| f64::from(matches!(
                coordinate.symbol(), SymbolRef::Coordinate { axis: selected, .. } if selected == axis
            )),
            dimension,
        )
    }

    pub(in crate::form_compiler) fn derivative(
        &self,
        direction: Direction<'_>,
        dimension: usize,
    ) -> Result<Self, Diagnostic> {
        if self.constant_in_direction(direction)? {
            return Ok(Self(Arc::new(Node::Derivative(
                self.clone(),
                Self::constant(dimension, <S as From<f64>>::from(0.0)),
            ))));
        }
        let derivative = |value: &Self| value.derivative(direction, dimension);
        Ok(match self.0.as_ref() {
            Node::Constant { .. } => Self(Arc::new(Node::Derivative(
                self.clone(),
                Self::constant(dimension, <S as From<f64>>::from(0.0)),
            ))),
            Node::Input {
                dimension: source_dimension,
                coordinate,
                input,
            } => {
                if *source_dimension != dimension {
                    return Err(super::invalid(
                        "coefficient gradient dimension differs from its source",
                    ));
                }
                Self(Arc::new(Node::Derivative(
                    self.clone(),
                    Self(Arc::new(Node::Input {
                        dimension,
                        coordinate: coordinate.clone(),
                        input: input.derivative(coordinate, direction, dimension)?,
                    })),
                )))
            }
            Node::Pointwise(data) => Self(Arc::new(Node::Pointwise(
                data.derivative(direction, dimension)?,
            ))),
            Node::Affine(data) => Self(Arc::new(Node::Affine(
                data.derivative(direction, dimension)?,
            ))),
            Node::Chart(data, chart) => {
                // The coefficient partial holds this geometry binding fixed.
                // Transport only physical coordinate directions through the chart;
                // Parameter components retain their original real-coordinate scale.
                let pulled_back = |coordinate: &ScalarSymbolCoordinate| {
                    let value = direction(coordinate);
                    if matches!(coordinate.symbol(), SymbolRef::Coordinate { .. }) {
                        value / chart.scale
                    } else {
                        value
                    }
                };
                data.derivative(&pulled_back, dimension)?
                    .on_uniform_chart(chart)
            }
            Node::Add(a, b) => derivative(a)?.add(derivative(b)?),
            Node::Mul(a, b) => derivative(a)?
                .multiply(b.clone())
                .add(a.clone().multiply(derivative(b)?)),
            Node::Div(a, b) => derivative(a)?
                .multiply(b.clone())
                .add(
                    a.clone()
                        .multiply(derivative(b)?)
                        .multiply(Self::constant(dimension, <S as From<f64>>::from(-1.0))),
                )
                .divide(b.clone().multiply(b.clone())),
            Node::Part { imaginary, data } => Self(Arc::new(Node::Part {
                imaginary: *imaginary,
                data: derivative(data)?,
            })),
            Node::Derivative(_, tangent) => {
                // The marker retains the source domain and input identities; its
                // numerical value is the tangent, so further derivatives act there.
                Self(Arc::new(Node::Derivative(
                    self.clone(),
                    derivative(tangent)?,
                )))
            }
        })
    }
    pub(in crate::form_compiler) fn spatial(&self) -> bool {
        match self.0.as_ref() {
            Node::Pointwise(data) => data.spatial(),
            Node::Affine(data) => data.spatial(),
            Node::Chart(data, _) | Node::Derivative(data, _) => data.spatial(),
            Node::Constant { .. } => false,
            Node::Input { input, .. } => input.spatial(),
            Node::Add(a, b) | Node::Mul(a, b) | Node::Div(a, b) => a.spatial() || b.spatial(),
            Node::Part { data, .. } => data.spatial(),
        }
    }
    pub(in crate::form_compiler) fn evaluate(&self, point: &[f64]) -> Result<S, Diagnostic> {
        let value = match self.0.as_ref() {
            Node::Constant { dimension, value } => {
                validate_point(point, *dimension)?;
                *value
            }
            Node::Input {
                dimension, input, ..
            } => {
                validate_point(point, *dimension)?;
                <S as From<f64>>::from(input.evaluate(point)?)
            }
            Node::Derivative(primal, tangent) => {
                primal.evaluate(point)?;
                tangent.evaluate(point)?
            }
            Node::Pointwise(data) => data.evaluate(point)?,
            Node::Affine(data) => data.evaluate(point)?,
            Node::Chart(data, chart) => {
                if point.len() != 2 {
                    return Err(super::invalid(
                        "moving chart coefficient requires a planar point",
                    ));
                }
                data.evaluate(&[
                    (point[0] - chart.offset[0]) / chart.scale,
                    (point[1] - chart.offset[1]) / chart.scale,
                ])?
            }
            Node::Add(a, b) => a.evaluate(point)? + b.evaluate(point)?,
            Node::Mul(a, b) => a.evaluate(point)? * b.evaluate(point)?,
            Node::Div(a, b) => a.evaluate(point)? / b.evaluate(point)?,
            Node::Part { imaginary, data } => {
                let value = data.evaluate(point)?;
                <S as From<f64>>::from(if *imaginary { value.im() } else { value.re() })
            }
        };
        if value.is_finite() {
            Ok(value)
        } else {
            Err(super::invalid("non-finite linear coefficient data"))
        }
    }
}

pub(in crate::form_compiler) struct Context<'a, S: Coefficient> {
    pub(in crate::form_compiler) time_s: Option<f64>,
    pub(in crate::form_compiler) program: &'a KernelProgram,
    pub(in crate::form_compiler) dag: &'a ExprDag,
    pub(in crate::form_compiler) owner: RawId,
    pub(in crate::form_compiler) dimension: usize,
    pub(in crate::form_compiler) coefficients: &'a BTreeMap<RawId, Components<S>>,
}

impl<S: Coefficient> Context<'_, S> {
    pub(in crate::form_compiler) fn components(
        &self,
        id: ExprId,
        depth: usize,
    ) -> Result<Vec<Data<S>>, Diagnostic> {
        if depth > 128 {
            return Err(super::invalid("linear expression nesting exceeds 128"));
        }
        let typed = super::super::scalar::typed_relation(self.program, self.owner)?;
        if typed
            .node_type(id)
            .is_some_and(|ty| ty.value_type.shape().is_scalar())
        {
            return self.data(id, depth).map(|data| vec![data]);
        }
        pointwise::PointwiseData::lower(self, id, depth)?.values(self.dimension)
    }

    pub(in crate::form_compiler) fn data(
        &self,
        id: ExprId,
        depth: usize,
    ) -> Result<Data<S>, Diagnostic> {
        if depth > 128 {
            return Err(super::invalid("linear expression nesting exceeds 128"));
        }
        Ok(match self.dag.node(id) {
            Some(ExprNode::Symbol(SymbolRef::Time)) => {
                let time = self
                    .time_s
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| {
                        super::invalid(
                            "coefficient Time requires an explicit finite evaluation point",
                        )
                    })?;
                Data::constant(self.dimension, <S as From<f64>>::from(time))
            }
            Some(ExprNode::Constant(value)) => Data::constant(
                self.dimension,
                S::literal(value).ok_or_else(|| {
                    super::invalid("constant requires its exact scalar numeric domain")
                })?,
            ),
            Some(ExprNode::Symbol(
                symbol @ (SymbolRef::Coordinate { .. } | SymbolRef::Parameter(_)),
            )) => {
                let typed = super::super::scalar::typed_relation(self.program, self.owner)?;
                if typed.expression() != self.dag {
                    return Err(super::invalid(
                        "coefficient input requires its exact typed expression",
                    ));
                }
                let ty = &typed
                    .node_type(id)
                    .ok_or_else(|| super::invalid("missing coefficient input type"))?
                    .value_type;
                if !ty.shape().is_scalar() {
                    return Err(super::invalid(
                        "scalar coefficient requires an exact scalar value type",
                    ));
                }
                let coordinates = ScalarSymbolCoordinate::for_value(*symbol, ty)?
                    .into_iter()
                    .map(|coordinate| {
                        let input = Input::from_coordinate(self, &coordinate, depth)?;
                        Ok(Data(Arc::new(Node::Input {
                            dimension: self.dimension,
                            coordinate,
                            input,
                        })))
                    })
                    .collect::<Result<Vec<_>, Diagnostic>>()?;
                Components::new(ty.clone(), coordinates)?.scalar(self.dimension)?
            }
            Some(ExprNode::Symbol(SymbolRef::Field(field))) => self
                .coefficients
                .get(&field.erase())
                .ok_or_else(|| {
                    super::invalid(
                        "unknown-dependent coefficient or unresolved coefficient definition",
                    )
                })?
                .scalar(self.dimension)?,
            _ => pointwise::PointwiseData::lower(self, id, depth)?.scalar(self.dimension)?,
        })
    }
}

#[cfg(test)]
mod tests;
