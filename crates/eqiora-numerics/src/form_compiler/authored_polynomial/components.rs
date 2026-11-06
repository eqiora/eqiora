//! Exact matrix components, including arithmetic and shared tensor calculus.
use super::*;

impl Context<'_> {
    pub(super) fn tensor(
        &mut self,
        value: &E,
        component: usize,
        axis: usize,
        depth: usize,
    ) -> Option<Polynomial> {
        self.step(depth)?;
        match value {
            E::LinearMap { values, .. } | E::SpatialTensor { values, .. } => {
                let shape = self.shape(value, depth + 1)?;
                self.closed_component(&shape, values, &[component, axis])
            }
            E::Parameter { ulid } => {
                self.atom(Atom::Parameter(ulid.clone(), vec![component, axis]))
            }
            E::SymmetricPart { .. } | E::IsotropicLift { .. } => {
                self.structure_component(value, &[component, axis], depth + 1)
            }
            E::Neg { value } => self
                .tensor(value, component, axis, depth + 1)?
                .checked_neg()
                .ok(),
            E::Add { left, right } => self
                .tensor(left, component, axis, depth + 1)?
                .checked_add(&self.tensor(right, component, axis, depth + 1)?)
                .ok(),
            E::Sub { left, right } => self
                .tensor(left, component, axis, depth + 1)?
                .checked_add(
                    &self
                        .tensor(right, component, axis, depth + 1)?
                        .checked_neg()
                        .ok()?,
                )
                .ok(),
            E::Mul { left, right } => {
                let (scalar, tensor) = if self.shape(left, depth + 1)?.is_empty() {
                    (left, right)
                } else {
                    (right, left)
                };
                self.scalar(scalar, depth + 1)?
                    .checked_mul(&self.tensor(tensor, component, axis, depth + 1)?)
                    .ok()
            }
            E::Div { left, right } => self
                .tensor(left, component, axis, depth + 1)?
                .divide_constant(&self.scalar(right, depth + 1)?),
            E::Conjugate { value } => self
                .tensor(value, component, axis, depth + 1)?
                .conjugate()
                .ok(),
            E::Gradient { value } if component < self.dimensions && axis < self.dimensions => self
                .atom(match value.as_ref() {
                    E::Field { ulid } => Atom::FieldGradient(ulid.clone(), vec![component, axis]),
                    E::Direction { name, field_ulid }
                        if name == self.name && field_ulid == self.field =>
                    {
                        Atom::TestGradient(vec![component, axis])
                    }
                    E::Test { field_ulid } if field_ulid == self.field => {
                        Atom::TestGradient(vec![component, axis])
                    }
                    _ => return None,
                }),
            _ => None,
        }
    }
}
