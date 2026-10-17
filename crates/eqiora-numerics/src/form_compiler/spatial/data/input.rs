//! One typed input binding for scalar, component and affine coefficient expressions.
use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Input<S: Coefficient> {
    Scalar { data: Data<S>, imaginary: bool },
    ParameterComponent { value: f64 },
    Coordinate { axis: usize, dimension: usize },
}

impl<S: Coefficient> Input<S> {
    pub(super) fn same_binding(
        &self,
        coordinate: &ScalarSymbolCoordinate,
        other: &Self,
        other_coordinate: &ScalarSymbolCoordinate,
    ) -> bool {
        // from_coordinate authenticates a physical coordinate's support against
        // its exact parent factor. Restricting that coordinate to a Boundary
        // changes its support, not its physical value or derivative direction.
        // Keep the factor, axis, component and scalar part; Parameters retain
        // their exact identity even when their bound values happen to agree.
        let physical_source = |coordinate: &ScalarSymbolCoordinate| match coordinate.symbol() {
            SymbolRef::Coordinate { factor, axis, .. } => {
                coordinate.with_symbol(SymbolRef::Coordinate {
                    support: factor,
                    factor,
                    axis,
                })
            }
            _ => coordinate.clone(),
        };
        self == other && physical_source(coordinate) == physical_source(other_coordinate)
    }

    pub(super) fn from_coordinate(
        context: &Context<'_, S>,
        coordinate: &eqiora_ir::ScalarSymbolCoordinate,
        depth: usize,
    ) -> Result<Self, Diagnostic> {
        if let SymbolRef::Field(field) = coordinate.symbol() {
            let value = context.coefficients.get(&field.erase()).ok_or_else(|| {
                super::super::invalid("unknown-dependent or unresolved prescribed Field")
            })?;
            let mut data = value.coordinate(
                coordinate.component_index(),
                coordinate.is_imaginary(),
                context.dimension,
            )?;
            for &axis in coordinate.spatial_derivative_axes() {
                data = data.coordinate_derivative(axis, context.dimension)?;
            }
            return Ok(Self::Scalar {
                data,
                imaginary: false,
            });
        }
        if let SymbolRef::Parameter(parameter) = coordinate.symbol() {
            let literal = context
                .program
                .typed_value(parameter.erase())
                .ok_or_else(|| super::super::invalid("coefficient Parameter has no exact value"))?;
            return Ok(Input::ParameterComponent {
                value: coordinate.read_value(literal)?,
            });
        }
        if let SymbolRef::Coordinate {
            support,
            factor,
            axis,
        } = coordinate.symbol()
        {
            if axis >= context.dimension
                || !spatial_expression::physical_coordinate(context.program, support, factor)
                || !coordinate.component_index().is_empty()
                || coordinate.is_imaginary()
                || !coordinate.spatial_derivative_axes().is_empty()
            {
                return Err(super::super::invalid(
                    "coefficient coordinate is outside its physical support",
                ));
            }
            return Ok(Input::Coordinate {
                axis,
                dimension: context.dimension,
            });
        }
        if !coordinate.component_index().is_empty() {
            return Err(super::super::invalid(
                "non-scalar coefficient coordinate requires an exact Field or Parameter",
            ));
        }
        let index = context
            .dag
            .nodes()
            .iter()
            .position(
                |node| matches!(node, ExprNode::Symbol(symbol) if *symbol == coordinate.symbol()),
            )
            .ok_or_else(|| super::super::invalid("pointwise input has no retained symbol"))?;
        Ok(Input::Scalar {
            data: context.data(
                context.dag.node_id(index as u32).expect("symbol index"),
                depth + 1,
            )?,
            imaginary: coordinate.is_imaginary(),
        })
    }
    pub(super) fn evaluate(&self, point: &[f64]) -> Result<f64, Diagnostic> {
        match self {
            Self::Scalar { data, imaginary } => {
                let value = data.evaluate(point)?;
                Ok(if *imaginary { value.im() } else { value.re() })
            }
            Self::ParameterComponent { value, .. } => Ok(*value),
            Self::Coordinate { axis, dimension } => {
                validate_point(point, *dimension)?;
                Ok(point[*axis])
            }
        }
    }
    pub(super) fn spatial(&self) -> bool {
        matches!(self, Self::Coordinate { .. })
            || matches!(self, Self::Scalar { data, .. } if data.spatial())
    }
    pub(super) fn bind(
        &self,
        coordinate: &eqiora_ir::ScalarSymbolCoordinate,
        point: &crate::parameter_point::ParameterPoint,
    ) -> Result<Self, Diagnostic> {
        Ok(match self {
            Self::Coordinate { .. } => self.clone(),
            Self::Scalar { data, imaginary } => Self::Scalar {
                data: data.bind_parameter_point(point)?,
                imaginary: *imaginary,
            },
            Self::ParameterComponent { .. } => {
                let SymbolRef::Parameter(parameter) = coordinate.symbol() else {
                    return Err(super::super::invalid(
                        "component binding is not a Parameter",
                    ));
                };
                let value = point.get(parameter).ok_or_else(|| {
                    super::super::invalid("coefficient Parameter point omits an exact identity")
                })?;
                Self::ParameterComponent {
                    value: coordinate.read_value(value)?,
                }
            }
        })
    }
    pub(super) fn derivative(
        &self,
        coordinate: &ScalarSymbolCoordinate,
        direction: Direction<'_>,
        dimension: usize,
    ) -> Result<Self, Diagnostic> {
        let constant = |value: f64| Self::Scalar {
            data: Data::constant(dimension, <S as From<f64>>::from(value)),
            imaginary: false,
        };
        Ok(match self {
            Self::Coordinate {
                dimension: source_dimension,
                ..
            } => {
                if *source_dimension != dimension {
                    return Err(super::super::invalid(
                        "coordinate derivative dimension differs from its source",
                    ));
                }
                constant(direction(coordinate))
            }
            Self::Scalar { data, imaginary } => Self::Scalar {
                data: data.derivative(direction, dimension)?,
                imaginary: *imaginary,
            },
            Self::ParameterComponent { .. } => constant(direction(coordinate)),
        })
    }
}
