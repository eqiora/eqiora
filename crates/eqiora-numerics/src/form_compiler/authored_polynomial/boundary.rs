//! Exact boundary identity accompanies every formal outward-normal component.
use super::*;
use eqiora_core::RawId;
use eqiora_schema::kernel::pure_operator::PureOperatorDefinition;
use eqiora_schema::kernel::typing::{SpatialSupport, TypedResidual};
use eqiora_schema::kernel::{ExprId, ExprNode, SymbolRef};

impl Context<'_> {
    pub(super) fn boundary(&self) -> Option<String> {
        let boundary = self.integration_domain.as_ref()?;
        let SpatialSupport::Boundary {
            parent, dimensions, ..
        } = self.domains.get(boundary)?
        else {
            return None;
        };
        let SpatialSupport::Volume {
            domain,
            dimensions: trial_dimensions,
        } = self.supports.get(self.field)?
        else {
            return None;
        };
        (parent == domain && *dimensions == self.dimensions && dimensions == trial_dimensions)
            .then(|| boundary.clone())
    }

    pub(super) fn tangential_definition(
        &mut self,
        value: &E,
        depth: usize,
    ) -> Option<PureOperatorDefinition> {
        self.boundary()?;
        (self.physical_shape(value, depth + 1)? == [self.dimensions]).then_some(())?;
        PureOperatorDefinition::tangential_lift(u32::try_from(self.dimensions).ok()?).ok()
    }

    pub(super) fn tangential_component(
        &mut self,
        value: &E,
        coordinate: &[usize],
        depth: usize,
    ) -> Option<Polynomial> {
        self.step(depth)?;
        let boundary = self.boundary()?;
        let definition = self.tangential_definition(value, depth + 1)?;
        let rank = if self.dimensions == 2 { 0 } else { 1 };
        if coordinate.len() != rank || coordinate.iter().any(|i| *i >= self.dimensions) {
            return None;
        }
        let mut sum = Polynomial::constant(ExactRational::integer(0));
        for axis in 0..self.dimensions {
            let mut lifted_coordinate = coordinate.to_vec();
            lifted_coordinate.push(axis);
            let component = self.pure_component(
                &definition,
                &lifted_coordinate,
                depth + 1,
                |context, formal, indices, _| {
                    (formal == 0).then_some(())?;
                    context.trace(value, indices.to_vec())
                },
            )?;
            sum = sum
                .checked_add(
                    &component
                        .checked_mul(&Polynomial::atom(Atom::Normal(boundary.clone(), axis)))
                        .ok()?,
                )
                .ok()?;
        }
        Some(sum)
    }
}

impl Context<'_> {
    pub(super) fn source_normal(
        &mut self,
        typed: &TypedResidual<RawId>,
        id: ExprId,
        value: ExprId,
        coordinate: &[usize],
        depth: usize,
    ) -> Option<Polynomial> {
        self.step(depth)?;
        let boundary = self.boundary()?;
        (typed.node_type(id)?.support.as_ref() == self.domains.get(&boundary)).then_some(())?;
        let mut sum = Polynomial::constant(ExactRational::integer(0));
        for axis in 0..self.dimensions {
            let mut indices = coordinate.to_vec();
            indices.push(axis);
            let value = self.source_restricted(typed, value, &indices, depth + 1)?;
            sum = sum
                .checked_add(
                    &value
                        .checked_mul(&Polynomial::atom(Atom::Normal(boundary.clone(), axis)))
                        .ok()?,
                )
                .ok()?;
        }
        Some(sum)
    }

    pub(super) fn source_restricted(
        &mut self,
        typed: &TypedResidual<RawId>,
        id: ExprId,
        coordinate: &[usize],
        depth: usize,
    ) -> Option<Polynomial> {
        self.step(depth)?;
        self.boundary()?;
        let ty = typed.node_type(id)?;
        if coordinate.len() != ty.shape().rank()
            || coordinate
                .iter()
                .zip(ty.shape().extents())
                .any(|(i, n)| *i >= n.get() as usize)
        {
            return None;
        }
        match typed.expression().node(id)? {
            ExprNode::Symbol(SymbolRef::Field(field)) => {
                let ulid = field.ulid().to_string();
                self.physical_shape(&E::Field { ulid: ulid.clone() }, depth + 1)?;
                self.atom(Atom::TraceField(ulid, coordinate.to_vec()))
            }
            ExprNode::PureOperatorApplication(application) => {
                let definition = typed.expression().definition(application.definition())?;
                let arguments = application.arguments();
                let types = arguments
                    .iter()
                    .map(|id| typed.node_type(*id).cloned())
                    .collect::<Option<Vec<_>>>()?;
                definition.instantiate(&types).ok()?;
                self.pure_component(
                    definition,
                    coordinate,
                    depth + 1,
                    |context, formal, indices, depth| {
                        context.source_restricted(
                            typed,
                            *arguments.get(usize::from(formal))?,
                            indices,
                            depth,
                        )
                    },
                )
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests;
