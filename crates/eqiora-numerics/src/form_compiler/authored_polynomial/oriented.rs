//! Expand admitted first-gradient forms with the Model calculus definitions.
use super::*;
use eqiora_schema::kernel::pure_operator::PureOperatorDefinition;

impl Context<'_> {
    pub(super) fn oriented_definition(
        &mut self,
        value: &E,
        depth: usize,
    ) -> Option<PureOperatorDefinition> {
        self.step(depth)?;
        match value {
            E::Curl { value } => {
                let shape = self.physical_shape(value, depth + 1)?;
                if shape.iter().any(|n| *n != self.dimensions) {
                    return None;
                }
                PureOperatorDefinition::curl_from_gradient(
                    u32::try_from(self.dimensions).ok()?,
                    u16::try_from(shape.len()).ok()?,
                )
                .ok()
            }
            E::Cross { left, right }
                if self.dimensions == 3
                    && self.physical_shape(left, depth + 1)? == [3]
                    && self.physical_shape(right, depth + 1)? == [3] =>
            {
                PureOperatorDefinition::cross_product().ok()
            }
            _ => None,
        }
    }

    // The coefficient ring forgets frames and supports. This local profile
    // admits operands on the trial's exact volume, before any cancellation.
    pub(super) fn physical_shape(&mut self, value: &E, depth: usize) -> Option<Vec<usize>> {
        self.step(depth)?;
        let shape = self.shape(value, depth + 1)?;
        match value {
            E::Field { ulid } | E::TimeDerivative { field_ulid: ulid } => {
                self.physical_field(ulid, &shape)?;
            }
            E::Parameter { ulid } => {
                self.physical_frame(ulid, &shape)?;
            }
            E::Test { field_ulid } | E::Direction { field_ulid, .. } => {
                self.physical_field(field_ulid, &shape)?;
            }
            E::Neg { value }
            | E::Conjugate { value }
            | E::Component { value, .. }
            | E::Gradient { value } => {
                self.physical_shape(value, depth + 1)?;
            }
            E::Add { left, right }
            | E::Sub { left, right }
            | E::Mul { left, right }
            | E::Dot { left, right }
            | E::Inner { left, right } => {
                self.physical_shape(left, depth + 1)?;
                self.physical_shape(right, depth + 1)?;
            }
            E::Complex { real, imag } => {
                self.physical_shape(real, depth + 1)?;
                self.physical_shape(imag, depth + 1)?;
            }
            E::Number { .. } | E::Rational { .. } => {}
            E::Curl { .. } | E::Cross { .. } => {}
            _ => return None,
        }
        Some(shape)
    }

    fn physical_frame(&self, symbol: &str, shape: &[usize]) -> Option<()> {
        (shape.is_empty()
            || self.symbols.get(symbol)?.frame() == eqiora_core::ValueFrame::SpatialCartesian)
            .then_some(())
    }

    fn physical_field(&self, symbol: &str, shape: &[usize]) -> Option<()> {
        self.physical_frame(symbol, shape)?;
        let support = self.supports.get(symbol)?;
        (matches!(support,
            eqiora_schema::kernel::typing::SpatialSupport::Volume { dimensions, .. }
                if *dimensions == self.dimensions)
            && support == self.supports.get(self.field)?)
        .then_some(())
    }

    pub(super) fn oriented_component(
        &mut self,
        value: &E,
        coordinate: &[usize],
        depth: usize,
    ) -> Option<Polynomial> {
        let definition = self.oriented_definition(value, depth + 1)?;
        let result = definition.result_rule();
        if coordinate.len() != result.rank()
            || coordinate
                .iter()
                .any(|i| result.spatial_extent().is_none_or(|n| *i >= n as usize))
        {
            return None;
        }
        self.pure_component(
            &definition,
            coordinate,
            depth + 1,
            |context, formal, indices, depth| match value {
                E::Curl { value } if formal == 0 => context.atom(match value.as_ref() {
                    E::Field { ulid } => Atom::FieldGradient(ulid.clone(), indices.to_vec()),
                    E::Test { field_ulid } if field_ulid == context.field => {
                        Atom::TestGradient(indices.to_vec())
                    }
                    E::Direction { name, field_ulid }
                        if name == context.name && field_ulid == context.field =>
                    {
                        Atom::TestGradient(indices.to_vec())
                    }
                    _ => return None,
                }),
                E::Cross { left, right } => {
                    let argument = match formal {
                        0 => left,
                        1 => right,
                        _ => return None,
                    };
                    let [axis] = indices else {
                        return None;
                    };
                    context.vector(argument, *axis, depth)
                }
                _ => None,
            },
        )
    }
}

#[cfg(test)]
mod tests;
