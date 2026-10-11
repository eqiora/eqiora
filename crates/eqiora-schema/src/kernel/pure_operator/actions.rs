//! Typed scalar Jacobian maps reuse the ordered first-partial transform.
use super::*;
use eqiora_core::{ScalarDomain, ValueType};

/// A derived map over exact formal slots in one calculus arena. Domain blocks
/// retain their own units and order; the codomain is one real scalar tangent.
/// Both actions use the dimensionless real dual pairing. This is an action view,
/// not a second persisted graph or a homogeneous matrix materialization.
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
        if output.scalar_domain() != ScalarDomain::Real || !output.shape().is_scalar() {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut inputs = Vec::with_capacity(formals.len());
        for formal in formals {
            if !seen.insert(*formal) {
                return Err(PureOperatorError::ArityMismatch);
            }
            let selected = *calculus
                .formals
                .get(usize::from(*formal))
                .ok_or(PureOperatorError::InvalidFormal(*formal))?;
            if !selected.is_invariant_scalar()
                || selected.scalar_domain() != Some(ScalarDomain::Real)
                || selected.dimension().is_none()
            {
                return Err(PureOperatorError::FormalTypeMismatch);
            }
            inputs.push((*formal, selected));
        }
        Ok(Self {
            calculus,
            root,
            inputs,
            output,
        })
    }

    fn staged(&self) -> CalculusBuilder {
        CalculusBuilder {
            formals: self.calculus.formals.clone(),
            result: self.calculus.result,
            nodes: self.calculus.nodes.clone(),
            depths: self.calculus.depths.clone(),
        }
    }

    fn jvp(self, directions: &[CalculusNodeId]) -> Result<CalculusNodeId, PureOperatorError> {
        if directions.len() != self.inputs.len() {
            return Err(PureOperatorError::ArityMismatch);
        }
        for ((_, selected), direction) in self.inputs.iter().zip(directions) {
            let actual = self.calculus.value_type(*direction)?;
            if actual.scalar_domain() != ScalarDomain::Real
                || !actual.shape().is_scalar()
                || selected.dimension() != Some(actual.dimension())
            {
                return Err(PureOperatorError::FormalTypeMismatch);
            }
        }
        let mut staged = self.staged();
        let mut result = None;
        for ((formal, _), direction) in self.inputs.iter().zip(directions) {
            let partial = staged.partial(self.root, *formal)?;
            let action = staged.push(CalculusNode::Mul(partial, *direction))?;
            result = Some(match result {
                Some(prior) => staged.push(CalculusNode::Add(prior, action))?,
                None => action,
            });
        }
        let result = result.expect("nonempty checked domain");
        if staged.value_type(result)? != self.output {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        *self.calculus = staged;
        Ok(result)
    }

    fn vjp(self, cotangent: CalculusNodeId) -> Result<Vec<CalculusNodeId>, PureOperatorError> {
        let seed = self.calculus.value_type(cotangent)?;
        let dual = self
            .output
            .dimension()
            .pow(-1, 1)
            .ok_or(PureOperatorError::ResultDimensionOverflow)?;
        if seed.scalar_domain() != ScalarDomain::Real
            || !seed.shape().is_scalar()
            || seed.dimension() != dual
        {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        let mut staged = self.staged();
        let mut result = Vec::with_capacity(self.inputs.len());
        for (formal, selected) in &self.inputs {
            let partial = staged.partial(self.root, *formal)?;
            let value = staged.push(CalculusNode::Mul(cotangent, partial))?;
            let expected = selected
                .dimension()
                .expect("checked domain block")
                .pow(-1, 1)
                .ok_or(PureOperatorError::ResultDimensionOverflow)?;
            if staged.value_type(value)?.dimension() != expected {
                return Err(PureOperatorError::FormalTypeMismatch);
            }
            result.push(value);
        }
        *self.calculus = staged;
        Ok(result)
    }
}

impl CalculusBuilder {
    /// Apply the scalar output Jacobian to ordered, independently typed input blocks.
    /// Each direction has its selected input's units; the result has output units.
    /// Omitted formals are held fixed. No homogeneous matrix is materialized.
    ///
    /// # Errors
    /// Rejects empty, duplicate or invalid selections, incompatible direction types,
    /// unsupported derivatives and calculus resource bounds. The append is atomic.
    pub fn jvp(
        &mut self,
        root: CalculusNodeId,
        directions: &[(u16, CalculusNodeId)],
    ) -> Result<CalculusNodeId, PureOperatorError> {
        if directions.is_empty() || directions.len() > MAX_FORMALS {
            return Err(PureOperatorError::FormalLimit);
        }
        let formals = directions
            .iter()
            .map(|(formal, _)| *formal)
            .collect::<Vec<_>>();
        let values = directions
            .iter()
            .map(|(_, value)| *value)
            .collect::<Vec<_>>();
        Jacobian::new(self, root, &formals)?.jvp(&values)
    }

    /// Pull back a scalar output cotangent through ordered typed input blocks.
    /// The real dual pairing is dimensionless: the seed has reciprocal output
    /// units, and each returned block has reciprocal selected-input units.
    /// Both actions use the same typed map and ordered first-partial transform.
    ///
    /// # Errors
    /// Rejects empty/duplicate selections, invalid formals, a nondual seed, unsupported
    /// derivatives and calculus resource bounds. Failure leaves the builder intact.
    pub fn vjp(
        &mut self,
        root: CalculusNodeId,
        cotangent: CalculusNodeId,
        formals: &[u16],
    ) -> Result<Vec<CalculusNodeId>, PureOperatorError> {
        Jacobian::new(self, root, formals)?.vjp(cotangent)
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
            builder.jvp(root, &[(0, inputs[3])]),
            Err(PureOperatorError::FormalTypeMismatch)
        );
        assert_eq!(builder.nodes, before);
        assert_eq!(
            builder.jvp(root, &[(0, inputs[2]), (0, inputs[2])]),
            Err(PureOperatorError::ArityMismatch)
        );
        assert_eq!(builder.nodes, before);
        let action = builder
            .jvp(root, &[(0, inputs[2]), (1, inputs[3])])
            .unwrap();
        assert_eq!(builder.value_type(action).unwrap().dimension(), output);
        let before = builder.nodes.clone();
        assert_eq!(
            builder.vjp(root, inputs[2], &[0, 1]),
            Err(PureOperatorError::FormalTypeMismatch)
        );
        assert_eq!(builder.nodes, before);
        let seed = builder
            .push(CalculusNode::Rational {
                value: ExactRational::integer(1),
                dimension: output.pow(-1, 1).unwrap(),
            })
            .unwrap();
        let cotangents = builder.vjp(root, seed, &[1, 0]).unwrap();
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
