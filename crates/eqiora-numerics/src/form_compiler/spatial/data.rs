use std::collections::BTreeMap;
use std::sync::Arc;

use eqiora_core::{Diagnostic, RawId};
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, SymbolRef, UnaryMathFunction};
use eqiora_sem::KernelProgram;

use crate::spatial_expression::{self, Coefficient, ScalarSpatialExpression};

#[derive(Debug, Clone, PartialEq)]
pub(in crate::form_compiler) struct Data<S: Coefficient>(Arc<Node<S>>);

mod mapped;
mod pointwise;

#[derive(Debug, PartialEq)]
enum Node<S: Coefficient> {
    Tape(ScalarSpatialExpression<S>),
    Pointwise(pointwise::PointwiseData<S>),
    MapFactor(mapped::MapFactorData),
    Chart(Data<S>, super::motion::UniformChart),
    CoordinateDerivative(ScalarSpatialExpression<S>, usize),
    Add(Data<S>, Data<S>),
    Mul(Data<S>, Data<S>),
    Div(Data<S>, Data<S>),
    Pow(Data<S>, i32),
    Math(UnaryMathFunction, Data<S>),
    Cos(Data<S>),
    SqrtDerivativeRoot(Data<S>),
}

impl<S: Coefficient> Data<S> {
    pub(in crate::form_compiler) fn bind_parameter_point(
        &self,
        fields: &[eqiora_core::Id<eqiora_core::entity::kinds::Parameter>],
        values: &[S],
    ) -> Result<Self, Diagnostic> {
        let bind = |data: &Self| data.bind_parameter_point(fields, values);
        Ok(Self(Arc::new(match self.0.as_ref() {
            Node::Tape(tape) => Node::Tape(tape.bind_parameter_point(fields, values)?),
            Node::Pointwise(data) => Node::Pointwise(data.bind_parameter_point(fields, values)?),
            Node::MapFactor(map) => Node::MapFactor(map.bind_parameter_point(fields, values)?),
            Node::Chart(data, chart) => Node::Chart(bind(data)?, chart.clone()),
            Node::CoordinateDerivative(tape, axis) => {
                Node::CoordinateDerivative(tape.bind_parameter_point(fields, values)?, *axis)
            }
            Node::Add(a, b) => Node::Add(bind(a)?, bind(b)?),
            Node::Mul(a, b) => Node::Mul(bind(a)?, bind(b)?),
            Node::Div(a, b) => Node::Div(bind(a)?, bind(b)?),
            Node::Pow(a, power) => Node::Pow(bind(a)?, *power),
            Node::Math(function, a) => Node::Math(*function, bind(a)?),
            Node::Cos(a) => Node::Cos(bind(a)?),
            Node::SqrtDerivativeRoot(a) => Node::SqrtDerivativeRoot(bind(a)?),
        })))
    }

    /// Compare symbolic coefficient products without sampling or erasing Parameters.
    pub(in crate::form_compiler) fn same_coefficient(&self, other: &Self) -> bool {
        fn product<'a, S: Coefficient>(
            data: &'a Data<S>,
            scale: &mut S,
            factors: &mut Vec<&'a Data<S>>,
        ) {
            match data.0.as_ref() {
                Node::Mul(a, b) => {
                    product(a, scale, factors);
                    product(b, scale, factors);
                }
                Node::Tape(tape) if tape.parameter_fields().is_empty() => {
                    if let Some(value) = tape.constant_value() {
                        *scale = *scale * value;
                    } else {
                        factors.push(data);
                    }
                }
                _ => factors.push(data),
            }
        }
        fn factor<S: Coefficient>(a: &Data<S>, b: &Data<S>) -> bool {
            match (a.0.as_ref(), b.0.as_ref()) {
                (Node::Tape(a), Node::Tape(b)) => a.is_same_coefficient_as(b),
                (Node::Pointwise(a), Node::Pointwise(b)) => a == b,
                (Node::MapFactor(a), Node::MapFactor(b)) => a == b,
                (Node::Chart(a, c), Node::Chart(b, d)) => c == d && a.same_coefficient(b),
                (Node::CoordinateDerivative(a, i), Node::CoordinateDerivative(b, j)) => {
                    i == j && a.is_same_coefficient_as(b)
                }
                (Node::Add(a, b), Node::Add(c, d)) => {
                    (a.same_coefficient(c) && b.same_coefficient(d))
                        || (a.same_coefficient(d) && b.same_coefficient(c))
                }
                (Node::Div(a, b), Node::Div(c, d)) => {
                    a.same_coefficient(c) && b.same_coefficient(d)
                }
                (Node::Pow(a, n), Node::Pow(b, m)) => n == m && a.same_coefficient(b),
                (Node::Math(f, a), Node::Math(g, b)) => f == g && a.same_coefficient(b),
                (Node::Cos(a), Node::Cos(b))
                | (Node::SqrtDerivativeRoot(a), Node::SqrtDerivativeRoot(b)) => {
                    a.same_coefficient(b)
                }
                _ => false,
            }
        }
        let (mut left_scale, mut right_scale) =
            (<S as From<f64>>::from(1.0), <S as From<f64>>::from(1.0));
        let (mut left, mut right) = (Vec::new(), Vec::new());
        product(self, &mut left_scale, &mut left);
        product(other, &mut right_scale, &mut right);
        if !left_scale.is_finite() || left_scale != right_scale || left.len() != right.len() {
            return false;
        }
        for candidate in left {
            let Some(index) = right.iter().position(|other| factor(candidate, other)) else {
                return false;
            };
            right.remove(index);
        }
        true
    }

    pub(in crate::form_compiler) fn on_uniform_chart(
        &self,
        chart: &super::motion::UniformChart,
    ) -> Self {
        Self(Arc::new(Node::Chart(self.clone(), chart.clone())))
    }
    pub(in crate::form_compiler) fn constant(dimension: usize, value: S) -> Self {
        Self(Arc::new(Node::Tape(ScalarSpatialExpression::constant(
            dimension, value,
        ))))
    }
    pub(in crate::form_compiler) fn add(self, right: Self) -> Self {
        let zero = |data: &Self| matches!(data.0.as_ref(), Node::Tape(tape) if tape.parameter_fields().is_empty() && tape.constant_value() == Some(<S as From<f64>>::from(0.0)));
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
    /// Differentiate coefficient data through its existing scalar tape and exact rules.
    pub(in crate::form_compiler) fn coordinate_derivative(
        &self,
        axis: usize,
        dimension: usize,
    ) -> Result<Self, Diagnostic> {
        if axis >= dimension {
            return Err(super::invalid("gradient axis exceeds physical dimension"));
        }
        let derivative = |value: &Self| value.coordinate_derivative(axis, dimension);
        Ok(match self.0.as_ref() {
            Node::Tape(tape) => Self(Arc::new(Node::CoordinateDerivative(tape.clone(), axis))),
            Node::Pointwise(data) => Self(Arc::new(Node::Pointwise(
                data.coordinate_derivative(axis, dimension)?,
            ))),
            Node::Chart(data, chart) => data
                .coordinate_derivative(axis, dimension)?
                .on_uniform_chart(chart)
                .multiply(Self::constant(
                    dimension,
                    <S as From<f64>>::from(1.0 / chart.scale),
                )),
            Node::MapFactor(_) => self
                .clone()
                .multiply(Self::constant(dimension, <S as From<f64>>::from(0.0))),
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
            Node::Pow(a, exponent) => {
                if *exponent == 0 {
                    // Demand the primal too: differentiation must not erase an undefined base.
                    self.clone()
                        .multiply(Self::constant(dimension, <S as From<f64>>::from(0.0)))
                } else {
                    let previous = match exponent.checked_sub(1) {
                        Some(power) => Self(Arc::new(Node::Pow(a.clone(), power))),
                        None => self.clone().divide(a.clone()),
                    };
                    Self::constant(dimension, <S as From<f64>>::from(f64::from(*exponent)))
                        .multiply(previous)
                        .multiply(derivative(a)?)
                }
            }
            Node::Math(UnaryMathFunction::Sqrt, a) => derivative(a)?.divide(
                Self::constant(dimension, <S as From<f64>>::from(2.0))
                    .multiply(Self(Arc::new(Node::SqrtDerivativeRoot(a.clone())))),
            ),
            Node::Math(UnaryMathFunction::Conj, a) => Self(Arc::new(Node::Math(
                UnaryMathFunction::Conj,
                derivative(a)?,
            ))),
            Node::Math(UnaryMathFunction::Sin, a) => {
                Self(Arc::new(Node::Cos(a.clone()))).multiply(derivative(a)?)
            }
            Node::Math(_, _)
            | Node::CoordinateDerivative(_, _)
            | Node::Cos(_)
            | Node::SqrtDerivativeRoot(_) => {
                return Err(super::invalid(
                    "coefficient gradient requires an admitted first-derivative rule",
                ));
            }
        })
    }
    pub(in crate::form_compiler) fn spatial(&self) -> bool {
        match self.0.as_ref() {
            Node::Pointwise(data) => data.spatial(),
            Node::MapFactor(_) => false,
            Node::Chart(data, _) => data.spatial(),
            Node::Tape(tape) | Node::CoordinateDerivative(tape, _) => {
                tape.is_coordinate_dependent()
            }
            Node::Add(a, b) | Node::Mul(a, b) | Node::Div(a, b) => a.spatial() || b.spatial(),
            Node::Pow(a, _) | Node::Math(_, a) | Node::Cos(a) | Node::SqrtDerivativeRoot(a) => {
                a.spatial()
            }
        }
    }
    pub(in crate::form_compiler) fn evaluate(&self, point: &[f64]) -> Result<S, Diagnostic> {
        let value = match self.0.as_ref() {
            Node::Tape(tape) => tape.evaluate(point)?,
            Node::Pointwise(data) => data.evaluate(point)?,
            Node::MapFactor(map) => <S as From<f64>>::from(map.value()),
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
            Node::CoordinateDerivative(tape, axis) => {
                let mut direction = vec![0.0; point.len()];
                *direction
                    .get_mut(*axis)
                    .ok_or_else(|| super::invalid("gradient axis exceeds physical dimension"))? =
                    1.0;
                tape.evaluate_tangent(
                    point,
                    &direction,
                    &vec![<S as From<f64>>::from(0.0); tape.parameter_fields().len()],
                )?
                .1
            }
            Node::Add(a, b) => a.evaluate(point)? + b.evaluate(point)?,
            Node::Mul(a, b) => a.evaluate(point)? * b.evaluate(point)?,
            Node::Div(a, b) => a.evaluate(point)? / b.evaluate(point)?,
            Node::Pow(a, n) => a.evaluate(point)?.powi(*n),
            Node::Math(UnaryMathFunction::Conj, a) => a.evaluate(point)?.conj(),
            Node::Math(UnaryMathFunction::Sin, a) => a.evaluate(point)?.sin(),
            Node::Cos(a) => a.evaluate(point)?.cos(),
            Node::SqrtDerivativeRoot(a) => {
                spatial_expression::sqrt_derivative_root(a.evaluate(point)?)?
            }
            Node::Math(UnaryMathFunction::Sqrt, a) => a.evaluate(point)?.sqrt(),
            _ => return Err(super::invalid("unsupported coefficient mathematics")),
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
    pub(in crate::form_compiler) coefficients: &'a BTreeMap<RawId, Data<S>>,
}

impl<S: Coefficient> Context<'_, S> {
    pub(in crate::form_compiler) fn data(
        &self,
        id: ExprId,
        depth: usize,
    ) -> Result<Data<S>, Diagnostic> {
        if depth > 128 {
            return Err(super::invalid("linear expression nesting exceeds 128"));
        }
        let data = |id| self.data(id, depth + 1);
        Ok(match self.dag.node(id) {
            Some(ExprNode::PureOperatorApplication(_)) => Data(Arc::new(Node::Pointwise(
                pointwise::PointwiseData::new(self, id, depth)?,
            ))),
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
            Some(ExprNode::CoordinateMapFactor { .. }) => Data(Arc::new(Node::MapFactor(
                mapped::MapFactorData::new(self, id)?,
            ))),
            Some(
                ExprNode::Constant(_)
                | ExprNode::Symbol(SymbolRef::Coordinate { .. })
                | ExprNode::Symbol(SymbolRef::Parameter(_)),
            ) => Data(Arc::new(Node::Tape(spatial_expression::lower(
                self.program,
                self.dag,
                id,
                self.owner,
                self.dimension,
            )?))),
            Some(ExprNode::Symbol(SymbolRef::Field(field))) => self
                .coefficients
                .get(&field.erase())
                .cloned()
                .ok_or_else(|| {
                    super::invalid(
                        "unknown-dependent coefficient or unresolved coefficient definition",
                    )
                })?,
            Some(ExprNode::Complex { real, imag }) => {
                let unit = S::imaginary_unit().ok_or_else(|| {
                    super::invalid(
                        "complex coefficient construction requires a complex scalar domain",
                    )
                })?;
                data(*real)?.add(data(*imag)?.multiply(Data::constant(self.dimension, unit)))
            }
            Some(ExprNode::Neg(a)) => {
                data(*a)?.multiply(Data::constant(self.dimension, <S as From<f64>>::from(-1.0)))
            }
            Some(ExprNode::Add(a, b)) => data(*a)?.add(data(*b)?),
            Some(ExprNode::Sub(a, b)) => data(*a)?.add(
                data(*b)?.multiply(Data::constant(self.dimension, <S as From<f64>>::from(-1.0))),
            ),
            Some(ExprNode::Mul(a, b)) => data(*a)?.multiply(data(*b)?),
            Some(ExprNode::Div(a, b)) => data(*a)?.divide(data(*b)?),
            Some(ExprNode::PowI(a, n)) => Data(Arc::new(Node::Pow(data(*a)?, *n))),
            Some(ExprNode::UnaryMath(function, a)) => {
                Data(Arc::new(Node::Math(*function, data(*a)?)))
            }
            _ => return Err(super::invalid("unsupported linear coefficient expression")),
        })
    }
}

#[cfg(test)]
mod tests;
