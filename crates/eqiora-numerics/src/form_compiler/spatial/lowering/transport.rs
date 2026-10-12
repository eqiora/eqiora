//! Conservative scalar fluxes share the Region gradient-test pairing.
use super::*;

impl<S: Coefficient> Context<'_, S> {
    pub(super) fn flux_terms(&self, id: ExprId, depth: usize) -> Result<Terms<S>, Diagnostic> {
        if depth > 128 {
            return Err(super::super::invalid("scalar flux nesting exceeds 128"));
        }
        let negative = || Data::constant(self.dimension, <S as From<f64>>::from(-1.0));
        let flux = |id| self.flux_terms(id, depth + 1);
        match self.dag.node(id) {
            Some(ExprNode::Add(a, b)) => Ok(flux(*a)?.add(flux(*b)?)),
            Some(ExprNode::Sub(a, b)) => Ok(flux(*a)?.add(flux(*b)?.scale_flux(negative()))),
            Some(ExprNode::Neg(a)) => Ok(flux(*a)?.scale_flux(negative())),
            Some(ExprNode::Mul(a, b)) => {
                if let Ok(data) = self.data(*a, depth + 1) {
                    return Ok(flux(*b)?.scale_flux(data));
                }
                if let Ok(data) = self.data(*b, depth + 1) {
                    return Ok(flux(*a)?.scale_flux(data));
                }
                let (trial, factor, transport) =
                    if let Ok((trial, factor)) = self.scalar_trial(*a, depth + 1) {
                        (trial, factor, *b)
                    } else if let Ok((trial, factor)) = self.scalar_trial(*b, depth + 1) {
                        (trial, factor, *a)
                    } else {
                        // Keep the established nonlinear-diffusion diagnostic at its owner.
                        self.flux(id, depth + 1)?;
                        return Err(super::super::invalid("unsupported scalar transport factor"));
                    };
                let transport = self.prescribed_vector(transport, depth + 1)?;
                let mut result =
                    Terms::data(Data::constant(self.dimension, <S as From<f64>>::from(0.0)));
                // div(v*u) pairs with -grad(test).v*u, including variable v.
                result
                    .transport
                    .extend(
                        transport
                            .into_iter()
                            .enumerate()
                            .map(|(axis, coefficient)| {
                                (
                                    (trial, axis),
                                    coefficient.multiply(factor.clone()).multiply(negative()),
                                )
                            }),
                    );
                Ok(result)
            }
            Some(ExprNode::Div(a, b)) => Ok(flux(*a)?.scale_flux(
                Data::constant(self.dimension, <S as From<f64>>::from(1.0))
                    .divide(self.data(*b, depth + 1)?),
            )),
            _ => {
                let (trial, coefficient) = self.flux(id, depth + 1)?;
                let mut result =
                    Terms::data(Data::constant(self.dimension, <S as From<f64>>::from(0.0)));
                result
                    .diffusion
                    .insert(trial, coefficient.multiply(negative()));
                Ok(result)
            }
        }
    }

    fn scalar_trial(&self, id: ExprId, depth: usize) -> Result<(RawId, Data<S>), Diagnostic> {
        if depth > 128 {
            return Err(super::super::invalid("scalar trial nesting exceeds 128"));
        }
        let trial = |id| self.scalar_trial(id, depth + 1);
        match self.dag.node(id) {
            Some(ExprNode::Symbol(SymbolRef::Field(field)))
                if !self.coefficients.contains_key(&field.erase()) =>
            {
                Ok((
                    field.erase(),
                    Data::constant(self.dimension, <S as From<f64>>::from(1.0)),
                ))
            }
            Some(ExprNode::Mul(a, b)) => {
                let (data, value) = if let Ok(data) = self.data(*a, depth + 1) {
                    (data, *b)
                } else {
                    (self.data(*b, depth + 1)?, *a)
                };
                let (field, coefficient) = trial(value)?;
                Ok((field, coefficient.multiply(data)))
            }
            Some(ExprNode::Div(a, b)) => {
                let (field, coefficient) = trial(*a)?;
                Ok((field, coefficient.divide(self.data(*b, depth + 1)?)))
            }
            Some(ExprNode::Neg(a)) => {
                let (field, coefficient) = trial(*a)?;
                Ok((
                    field,
                    coefficient
                        .multiply(Data::constant(self.dimension, <S as From<f64>>::from(-1.0))),
                ))
            }
            _ => Err(super::super::invalid(
                "transport requires one exact scalar trial",
            )),
        }
    }

    fn prescribed_vector(&self, id: ExprId, depth: usize) -> Result<Vec<Data<S>>, Diagnostic> {
        if depth > 128 {
            return Err(super::super::invalid(
                "prescribed transport nesting exceeds 128",
            ));
        }
        let vector = |id| self.prescribed_vector(id, depth + 1);
        let scale = |values: Vec<Data<S>>, coefficient: Data<S>| {
            values
                .into_iter()
                .map(|value| value.multiply(coefficient.clone()))
                .collect()
        };
        let negative = || Data::constant(self.dimension, <S as From<f64>>::from(-1.0));
        match self.dag.node(id) {
            Some(ExprNode::Gradient(potential)) => {
                let potential = self.data(*potential, depth + 1)?;
                let primal = potential
                    .clone()
                    .multiply(Data::constant(self.dimension, <S as From<f64>>::from(0.0)));
                (0..self.dimension)
                    .map(|axis| {
                        Ok(primal
                            .clone()
                            .add(potential.coordinate_derivative(axis, self.dimension)?))
                    })
                    .collect()
            }
            Some(ExprNode::Add(a, b) | ExprNode::Sub(a, b)) => {
                let right = vector(*b)?;
                let right = if matches!(self.dag.node(id), Some(ExprNode::Sub(..))) {
                    scale(right, negative())
                } else {
                    right
                };
                Ok(vector(*a)?
                    .into_iter()
                    .zip(right)
                    .map(|(a, b)| a.add(b))
                    .collect())
            }
            Some(ExprNode::Neg(a)) => Ok(scale(vector(*a)?, negative())),
            Some(ExprNode::Mul(a, b)) => {
                if let Ok(data) = self.data(*a, depth + 1) {
                    Ok(scale(vector(*b)?, data))
                } else {
                    Ok(scale(vector(*a)?, self.data(*b, depth + 1)?))
                }
            }
            Some(ExprNode::Div(a, b)) => Ok(scale(
                vector(*a)?,
                Data::constant(self.dimension, <S as From<f64>>::from(1.0))
                    .divide(self.data(*b, depth + 1)?),
            )),
            _ => Err(super::super::invalid(
                "prescribed transport transport requires exact coefficient gradients and scalar combinations",
            )),
        }
    }
}
