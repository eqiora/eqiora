//! Real-linear Jacobian actions built on independent real-coordinate partials.
use super::composition::coordinates;
use super::*;
use eqiora_core::{ScalarDomain, ValueType};
#[cfg(test)]
mod contract_tests;
mod contraction;
mod linearization;
mod pairing;
use crate::kernel::UnaryMathFunction;
use linearization::Linearization;

/// A derived map over formal slots in one calculus arena. Each domain block keeps its
/// own units and order; the codomain is one numeric component. The forward map is
/// applied as an action using the shared real-coordinate linearization; it is
/// neither stored as a separate graph nor assembled as a matrix.
struct Jacobian<'a> {
    calculus: &'a mut CalculusBuilder,
    root: CalculusNodeId,
    inputs: Vec<(u16, PureValueClass)>,
    output: ValueType,
}

impl<'a> Jacobian<'a> {
    fn new(
        calculus: &'a mut CalculusBuilder,
        root: CalculusNodeId,
        formals: &[u16],
    ) -> Result<Self, PureOperatorError> {
        if formals.is_empty() || formals.len() > MAX_FORMALS {
            return Err(PureOperatorError::FormalLimit);
        }
        let output = calculus.value_type(root)?;
        if !output.shape().is_scalar() {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut inputs = Vec::with_capacity(formals.len());
        for formal in formals {
            if !seen.insert(*formal) {
                return Err(PureOperatorError::ArityMismatch);
            }
            let selected = calculus
                .formals
                .get(usize::from(*formal))
                .ok_or(PureOperatorError::InvalidFormal(*formal))?;
            if selected.exact_type().is_none() {
                return Err(PureOperatorError::FormalTypeMismatch);
            }
            inputs.push((*formal, selected.clone()));
        }
        Ok(Self {
            calculus,
            root,
            inputs,
            output,
        })
    }

    fn jvp(
        self,
        directions: &[CalculusArgument],
        parameter_dimension: DimExponents,
    ) -> Result<CalculusNodeId, PureOperatorError> {
        if directions.len() != self.inputs.len() {
            return Err(PureOperatorError::ArityMismatch);
        }
        let output = self
            .output
            .clone()
            .with_dimension(
                self.output
                    .dimension()
                    .div(parameter_dimension)
                    .ok_or(PureOperatorError::ResultDimensionOverflow)?,
            )
            .map_err(|_| PureOperatorError::FormalTypeMismatch)?;
        for ((_, selected), direction) in self.inputs.iter().zip(directions) {
            let input = input_type(selected);
            let rate = input
                .value_type
                .clone()
                .with_dimension(
                    input
                        .dimension()
                        .div(parameter_dimension)
                        .ok_or(PureOperatorError::ResultDimensionOverflow)?,
                )
                .map_err(|_| PureOperatorError::FormalTypeMismatch)?;
            ExpressionType::<()>::new(output.clone(), None)
                .jvp(
                    &ExpressionType::new(rate.clone(), None),
                    &ExpressionType::new(direction.value_type(self.calculus)?, None),
                )
                .map_err(action_type_error)?;
            direction.validate(self.calculus, &PureValueClass::from_value_type(rate)?)?;
        }
        let staged = self.calculus;
        let linear = Linearization::bind(staged, self.root, &self.inputs)?;
        let mut result = None;
        for coordinate in &linear.coordinates {
            let selected = self
                .inputs
                .iter()
                .position(|(formal, _)| *formal == coordinate.formal)
                .expect("selected coordinate");
            let direction = directions[selected].project(staged, &coordinate.axes)?;
            let direction = match coordinate.part {
                Some(part) => staged.push(CalculusNode::UnaryMath(part, direction))?,
                None => direction,
            };
            let partial = staged.push(CalculusNode::Differentiated {
                source: linear.root,
                wrt: coordinate.value,
            })?;
            let action = staged.push(CalculusNode::Mul(partial, direction))?;
            result = Some(match result {
                Some(prior) => staged.push(CalculusNode::Add(prior, action))?,
                None => action,
            });
        }
        let result = result.expect("nonempty checked domain");
        if staged.value_type(result)? != output {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        Ok(result)
    }
}

fn input_type(class: &PureValueClass) -> ExpressionType<()> {
    ExpressionType::new(
        class.exact_type().expect("checked exact input").clone(),
        None,
    )
}

fn action_type_error(error: crate::kernel::typing::TypeViolation<()>) -> PureOperatorError {
    match error {
        crate::kernel::typing::TypeViolation::DimensionOverflow { .. } => {
            PureOperatorError::ResultDimensionOverflow
        }
        _ => PureOperatorError::FormalTypeMismatch,
    }
}

impl CalculusBuilder {
    /// Apply a complete value's Jacobian to ordered, independently typed input blocks.
    /// Each direction has its selected input's units; the result has the output's units.
    /// Formals that are not selected are held fixed. No matrix is assembled.
    ///
    /// # Errors
    /// Rejects empty, duplicate, or invalid selections, incompatible direction types,
    /// unsupported derivatives, and calculus resource limits. On error nothing is appended.
    pub fn jvp(
        &mut self,
        value: &CalculusArgument,
        directions: &[(u16, CalculusArgument)],
    ) -> Result<CalculusNodeId, PureOperatorError> {
        self.jacobian_action(value, directions, None)
    }

    /// Differentiate with respect to a real scalar coordinate, including typed rates
    /// of dependent inputs. Each rate has its input's units divided by the coordinate's
    /// units; complex and shaped inputs use the same real-linear action as `jvp`.
    /// Unlisted inputs remain fixed. The output has its original units per coordinate.
    ///
    /// # Errors
    /// Rejects invalid coordinates, duplicate or incompatible rates, unsupported
    /// derivatives and resource limits. On error nothing is appended.
    pub fn total_derivative(
        &mut self,
        value: &CalculusArgument,
        coordinate: u16,
        rates: &[(u16, CalculusArgument)],
    ) -> Result<CalculusNodeId, PureOperatorError> {
        self.jacobian_action(value, rates, Some(coordinate))
    }

    fn jacobian_action(
        &mut self,
        value: &CalculusArgument,
        directions: &[(u16, CalculusArgument)],
        coordinate: Option<u16>,
    ) -> Result<CalculusNodeId, PureOperatorError> {
        let count = directions
            .len()
            .saturating_add(usize::from(coordinate.is_some()));
        if count == 0 || count > MAX_FORMALS {
            return Err(PureOperatorError::FormalLimit);
        }
        let parameter_dimension = match coordinate {
            Some(formal) => {
                let selected = self
                    .formals
                    .get(usize::from(formal))
                    .ok_or(PureOperatorError::InvalidFormal(formal))?;
                let ty = selected
                    .exact_type()
                    .ok_or(PureOperatorError::FormalTypeMismatch)?;
                if !selected.is_invariant_scalar() || ty.scalar_domain() != ScalarDomain::Real {
                    return Err(PureOperatorError::FormalTypeMismatch);
                }
                ty.dimension()
            }
            None => DimExponents::DIMENSIONLESS,
        };
        let output = PureValueClass::from_value_type(value.value_type(self)?)?;
        if !output.is_invariant_scalar()
            && !composition::compatible_component_layout(&output, &self.result)
        {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        let mut staged = CalculusBuilder {
            formals: self.formals.clone(),
            result: self.result.clone(),
            nodes: self.nodes.clone(),
            depths: self.depths.clone(),
        };
        let axes = (0..output.rank())
            .map(|axis| ComponentIndex::Result(axis as u16))
            .collect::<Vec<_>>();
        let root = value.project(&mut staged, &axes)?;
        let mut directions = directions.to_vec();
        if let Some(coordinate) = coordinate {
            let unit = staged.push(CalculusNode::Rational {
                value: ExactRational::integer(1),
                dimension: DimExponents::DIMENSIONLESS,
            })?;
            directions.insert(0, (coordinate, unit.into()));
        }
        let formals = directions
            .iter()
            .map(|(formal, _)| *formal)
            .collect::<Vec<_>>();
        let values = directions
            .iter()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        let result =
            Jacobian::new(&mut staged, root, &formals)?.jvp(&values, parameter_dimension)?;
        *self = staged;
        Ok(result)
    }

    /// Pull a complete output cotangent back to one independently typed input.
    /// The real pairing is dimensionless: seed and result have reciprocal output
    /// and input units. Whole values retain their own axes during contraction.
    ///
    /// # Errors
    /// Rejects invalid selections, incomplete or incompatible numeric contracts,
    /// unsupported derivatives, and calculus resource limits. A failure appends nothing.
    pub fn vjp(
        &mut self,
        value: &CalculusArgument,
        cotangent: &CalculusArgument,
        formal: u16,
    ) -> Result<CalculusNodeId, PureOperatorError> {
        contraction::vjp(self, value, cotangent, formal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heterogeneous_jvp_keeps_output_units_and_rejects_wrong_direction_atomically() {
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        let speed = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
        let output = length.mul(speed).unwrap();
        let class = |dimension| {
            PureValueClass::invariant_scalar()
                .with_dimension(dimension)
                .with_scalar_domain(ScalarDomain::Real)
                .unwrap()
        };
        let mut builder = CalculusBuilder::new(
            [class(length), class(speed), class(length), class(speed)],
            class(output),
        )
        .unwrap();
        let inputs = (0..4)
            .map(|formal| {
                builder
                    .push(CalculusNode::FormalComponent {
                        formal,
                        axes: Box::new([]),
                    })
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let root = builder
            .push(CalculusNode::Mul(inputs[0], inputs[1]))
            .unwrap();
        let before = builder.nodes.clone();
        assert_eq!(
            builder.jvp(&root.into(), &[(0, inputs[3].into())]),
            Err(PureOperatorError::FormalTypeMismatch)
        );
        assert_eq!(builder.nodes, before);
        assert_eq!(
            builder.jvp(
                &root.into(),
                &[(0, inputs[2].into()), (0, inputs[2].into())]
            ),
            Err(PureOperatorError::ArityMismatch)
        );
        assert_eq!(builder.nodes, before);
        let action = builder
            .jvp(
                &root.into(),
                &[(0, inputs[2].into()), (1, inputs[3].into())],
            )
            .unwrap();
        assert_eq!(builder.value_type(action).unwrap().dimension(), output);
        let before = builder.nodes.clone();
        assert_eq!(
            builder.vjp(&root.into(), &inputs[2].into(), 0),
            Err(PureOperatorError::FormalTypeMismatch)
        );
        assert_eq!(builder.nodes, before);
        let seed = builder
            .push(CalculusNode::Rational {
                value: ExactRational::integer(1),
                dimension: output.pow(-1, 1).unwrap(),
            })
            .unwrap();
        let cotangents =
            [1, 0].map(|formal| builder.vjp(&root.into(), &seed.into(), formal).unwrap());
        assert_eq!(
            builder.value_type(cotangents[0]).unwrap().dimension(),
            speed.pow(-1, 1).unwrap()
        );
        assert_eq!(
            builder.value_type(cotangents[1]).unwrap().dimension(),
            length.pow(-1, 1).unwrap()
        );
        builder.finish(action).unwrap();
    }
}
