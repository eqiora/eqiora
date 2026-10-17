use std::collections::BTreeMap;
use std::sync::Arc;

use eqiora_core::{Diagnostic, RawId};
use eqiora_schema::kernel::{ExprDag, ExprId, ExprNode, SymbolRef, UnaryMathFunction};
use eqiora_sem::KernelProgram;

use crate::spatial_expression::{self, Coefficient, validate_coordinates as validate_point};
use eqiora_ir::ScalarSymbolCoordinate;
use input::Input;

/// Directions of physical coordinates and retained real Parameter components.
/// These coefficient partials hold Time and geometry bindings fixed; geometry
/// maps and their own Parameter dependence have separate partial actions.
type Direction<'a> = &'a dyn Fn(&ScalarSymbolCoordinate) -> f64;

#[derive(Debug, Clone, PartialEq)]
pub(in crate::form_compiler) struct Data<S: Coefficient>(Arc<Node<S>>);

pub(in crate::form_compiler) mod affine;
mod components;
pub(in crate::form_compiler) use components::Components;
mod degree;
mod input;
mod mapped;
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
    MapFactor(mapped::MapFactorData),
    Chart(Data<S>, super::motion::UniformChart),
    // Retain primal domains and Parameter identities when an input derivative is zero.
    Derivative(Data<S>, Data<S>),
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
            Node::MapFactor(map) => Node::MapFactor(map.bind_parameter_point(point)?),
            Node::Chart(data, chart) => Node::Chart(bind(data)?, chart.clone()),
            Node::Derivative(primal, tangent) => Node::Derivative(bind(primal)?, bind(tangent)?),
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
                Node::Constant { value, .. } => *scale = *scale * *value,
                _ => factors.push(data),
            }
        }
        fn factor<S: Coefficient>(a: &Data<S>, b: &Data<S>) -> bool {
            match (a.0.as_ref(), b.0.as_ref()) {
                (
                    Node::Input {
                        dimension: a,
                        coordinate: c,
                        input: e,
                    },
                    Node::Input {
                        dimension: b,
                        coordinate: d,
                        input: f,
                    },
                ) => a == b && c == d && e == f,
                (Node::Pointwise(a), Node::Pointwise(b)) => a == b,
                (Node::Affine(a), Node::Affine(b)) => a == b,
                (Node::MapFactor(a), Node::MapFactor(b)) => a == b,
                (Node::Chart(a, c), Node::Chart(b, d)) => c == d && a.same_coefficient(b),
                (Node::Derivative(a, b), Node::Derivative(c, d)) => {
                    a.same_coefficient(c) && b.same_coefficient(d)
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
            Node::MapFactor(map) => {
                if !map.constant_in_direction(direction)? {
                    return Err(super::invalid(
                        "map coefficient derivatives require the exact coordinate-map action",
                    ));
                }
                Self(Arc::new(Node::Derivative(
                    self.clone(),
                    Self::constant(dimension, <S as From<f64>>::from(0.0)),
                )))
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
            Node::Math(
                function @ (UnaryMathFunction::Conj
                | UnaryMathFunction::Real
                | UnaryMathFunction::Imag),
                a,
            ) => Self(Arc::new(Node::Math(*function, derivative(a)?))),
            Node::Math(UnaryMathFunction::Sin, a) => {
                Self(Arc::new(Node::Cos(a.clone()))).multiply(derivative(a)?)
            }
            Node::Math(_, _)
            | Node::Derivative(_, _)
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
            Node::Affine(data) => data.spatial(),
            Node::MapFactor(_) => false,
            Node::Chart(data, _) | Node::Derivative(data, _) => data.spatial(),
            Node::Constant { .. } => false,
            Node::Input { input, .. } => input.spatial(),
            Node::Add(a, b) | Node::Mul(a, b) | Node::Div(a, b) => a.spatial() || b.spatial(),
            Node::Pow(a, _) | Node::Math(_, a) | Node::Cos(a) | Node::SqrtDerivativeRoot(a) => {
                a.spatial()
            }
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
            Node::Add(a, b) => a.evaluate(point)? + b.evaluate(point)?,
            Node::Mul(a, b) => a.evaluate(point)? * b.evaluate(point)?,
            Node::Div(a, b) => a.evaluate(point)? / b.evaluate(point)?,
            Node::Pow(a, n) => a.evaluate(point)?.powi(*n),
            Node::Math(UnaryMathFunction::Conj, a) => a.evaluate(point)?.conj(),
            Node::Math(UnaryMathFunction::Real, a) => {
                <S as From<f64>>::from(a.evaluate(point)?.re())
            }
            Node::Math(UnaryMathFunction::Imag, a) => {
                <S as From<f64>>::from(a.evaluate(point)?.im())
            }
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
        let data = |id| self.data(id, depth + 1);
        Ok(match self.dag.node(id) {
            Some(
                ExprNode::PureOperatorApplication(_)
                | ExprNode::Index { .. }
                | ExprNode::UnaryMath(UnaryMathFunction::Real | UnaryMathFunction::Imag, _),
            ) => pointwise::PointwiseData::lower(self, id, depth)?.scalar(self.dimension)?,
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
