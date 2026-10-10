//! Source correspondence for the planar uniform-chart ALE transport projection.
use super::TimeDerivativeProofError as Error;
use super::*;
use crate::kernel::{CoordinateMapFactor, ExprNode, SymbolRef, typing::TypedResidual};
use eqiora_core::{DimExponents, ScalarDomain, ValueType};

#[cfg(test)]
mod tests;

impl<I> TypedResidual<I> {
    /// Prove the advective projection `c*u*lambda*(v-d(chi)/dt)` of a planar Law.
    ///
    /// `stored` must equal `c*u*J` for the exact supplied volume map; `c` may
    /// contain only constants and Parameters. The map must be polynomial uniform
    /// scaling plus translation, with axes selected by coordinate identity.
    /// Every advective axis must retain an explicit `v - mesh_rate` subtraction,
    /// whose rate independently equals the derivative of that map row. Return
    /// those exact real scalar material-velocity expressions (m/s) in physical
    /// axis order. A subtraction with different dimensions cannot supply a
    /// relative-velocity witness even when its normalized numbers match.
    ///
    /// This checks only the advective projection. Gradients of `field` are
    /// omitted here: the caller must separately admit the diffusive flux, types,
    /// supports, positive map, and prescribed scalar velocity expressions.
    /// Other vector forms and nonlinear coordinate maps reject. Bounds and exact
    /// arithmetic are shared with `ExprDag::verify_time_derivative`; no numeric samples
    /// or executable differentiation establish this correspondence.
    pub fn verify_uniform_ale_transport(
        &self,
        map: ExprId,
        stored: ExprId,
        flux: ExprId,
        field: Id<kinds::Field>,
    ) -> Result<[ExprId; 2], Error> {
        let dag = self.expression();
        if dag.nodes().len() > MAX_PROOF_NODES {
            return Err(Error::Limit);
        }
        let Some(ExprNode::CoordinateMapFactor {
            factor: CoordinateMapFactor::VolumeScale,
            source,
            at,
        }) = dag.node(map)
        else {
            return Err(Error::UnsupportedExpression);
        };
        if source.len() != 2 || at.len() != 2 {
            return Err(Error::UnsupportedExpression);
        }
        let velocity_type = ValueType::scalar(
            ScalarDomain::Real,
            DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0])
                .expect("physical velocity dimension"),
        )
        .expect("real scalar velocity");
        let mut proof = TransportProof {
            dag,
            velocity_nodes: self
                .node_types()
                .iter()
                .enumerate()
                .filter(|(_, ty)| ty.value_type == velocity_type)
                .map(|(index, _)| dag.node_id(index as u32).expect("typed node"))
                .collect(),
            budget: Budget(MAX_PROOF_WORK),
            maps: projection::Maps::default(),
            field,
            coordinates: [None; 2],
            rates: [zero(), zero()],
            velocities: [None, None],
            flux: [zero(), zero()],
        };
        for &id in source {
            let (atom, axis) = coordinate(dag, id)?;
            if proof.coordinates[axis].replace(atom).is_some() {
                return Err(Error::Mismatch);
            }
        }
        let [
            Some(Atom::Coordinate(a, b, _)),
            Some(Atom::Coordinate(c, d, _)),
        ] = proof.coordinates
        else {
            return Err(Error::Mismatch);
        };
        if a != b || a != c || c != d {
            return Err(Error::Mismatch);
        }
        let mut scales = [None, None];
        let mut target = None;
        for &(selector, row) in at {
            let (Atom::Coordinate(support, factor, axis), _) = coordinate(dag, selector)? else {
                unreachable!()
            };
            if support != factor || target.is_some_and(|prior| prior != support) {
                return Err(Error::Mismatch);
            }
            target = Some(support);
            let row = proof.normalize(row)?;
            let scale = uniform_row_scale(&row, proof.coordinates[axis].unwrap())?;
            if scales[axis].replace(scale).is_some() {
                return Err(Error::Mismatch);
            }
            proof.rates[axis] = derivative(&row, &mut proof.maps, &mut proof.budget)?;
        }
        let [Some(scale), Some(other)] = scales else {
            return Err(Error::Mismatch);
        };
        if scale != other || scale == zero() {
            return Err(Error::Mismatch);
        }
        let map = proof.normalize(map)?;
        let map_terms = map.terms().collect::<Vec<_>>();
        let [(atoms, coefficient)] = map_terms[..] else {
            return Err(Error::Mismatch);
        };
        let [map_atom @ Atom::Map(_)] = atoms else {
            return Err(Error::Mismatch);
        };
        if coefficient != ExactRational::integer(1) {
            return Err(Error::Mismatch);
        }
        let mut capacity = zero();
        for (atoms, coefficient) in proof.normalize(stored)?.terms() {
            let mut rest = atoms.to_vec();
            for required in [Atom::Field(field), *map_atom] {
                let index = rest
                    .iter()
                    .position(|atom| *atom == required)
                    .ok_or(Error::Mismatch)?;
                rest.remove(index);
            }
            if rest.iter().any(|atom| !matches!(atom, Atom::Parameter(_))) {
                return Err(Error::UnsupportedExpression);
            }
            capacity.add_term(rest, coefficient)?;
        }
        let prefactor = capacity
            .checked_mul(&Polynomial::atom(Atom::Field(field)))?
            .checked_mul(&scale)?;
        proof.vector(
            flux,
            Polynomial::constant(ExactRational::integer(1)),
            &[],
            0,
        )?;
        let mut result = [flux; 2];
        for (axis, result) in result.iter_mut().enumerate() {
            let Some((id, velocity)) = &proof.velocities[axis] else {
                return Err(Error::Mismatch);
            };
            let relative = velocity.checked_add(&proof.rates[axis].checked_neg()?)?;
            if proof.flux[axis] != prefactor.checked_mul(&relative)? {
                return Err(Error::Mismatch);
            }
            *result = *id;
        }
        Ok(result)
    }
}

fn zero() -> Polynomial {
    Polynomial::constant(ExactRational::integer(0))
}

fn coordinate(dag: &ExprDag, id: ExprId) -> Result<(Atom, usize), Error> {
    match dag.node(id) {
        Some(ExprNode::Symbol(SymbolRef::Coordinate {
            support,
            factor,
            axis,
        })) if *axis < 2 => Ok((Atom::Coordinate(*support, *factor, *axis), *axis)),
        _ => Err(Error::UnsupportedExpression),
    }
}

fn uniform_row_scale(row: &Polynomial, coordinate: Atom) -> Result<Polynomial, Error> {
    let mut scale = zero();
    for (atoms, coefficient) in row.terms() {
        let mut remaining = Vec::new();
        let mut found = false;
        for &atom in atoms {
            match atom {
                Atom::Parameter(_) | Atom::Time => remaining.push(atom),
                _ if atom == coordinate && !found => found = true,
                _ => return Err(Error::UnsupportedExpression),
            }
        }
        if found {
            scale.add_term(remaining, coefficient)?;
        }
    }
    Ok(scale)
}

struct TransportProof<'a> {
    dag: &'a ExprDag,
    velocity_nodes: std::collections::BTreeSet<ExprId>,
    budget: Budget,
    maps: projection::Maps,
    field: Id<kinds::Field>,
    coordinates: [Option<Atom>; 2],
    rates: [Polynomial; 2],
    velocities: [Option<(ExprId, Polynomial)>; 2],
    flux: [Polynomial; 2],
}

impl TransportProof<'_> {
    fn normalize(&mut self, id: ExprId) -> Result<Polynomial, Error> {
        projection::normalize(self.dag, id, false, &mut self.budget, &mut self.maps)
    }

    fn vector(
        &mut self,
        id: ExprId,
        multiplier: Polynomial,
        factors: &[ExprId],
        depth: usize,
    ) -> Result<(), Error> {
        self.budget.charge(1)?;
        if depth > 128 {
            return Err(Error::Limit);
        }
        match self.dag.node(id).ok_or(Error::InvalidExpression)? {
            ExprNode::Add(a, b) | ExprNode::Sub(a, b) => {
                let (a, b, subtract) =
                    (*a, *b, matches!(self.dag.node(id), Some(ExprNode::Sub(..))));
                self.vector(a, multiplier.clone(), factors, depth + 1)?;
                self.vector(
                    b,
                    if subtract {
                        multiplier.checked_neg()?
                    } else {
                        multiplier
                    },
                    factors,
                    depth + 1,
                )
            }
            ExprNode::Neg(value) => {
                self.vector(*value, multiplier.checked_neg()?, factors, depth + 1)
            }
            ExprNode::Mul(a, b) => {
                let (a, b) = (*a, *b);
                let (scalar, vector, polynomial) = match self.normalize(a) {
                    Ok(value) => (a, b, value),
                    Err(Error::UnsupportedExpression) => (b, a, self.normalize(b)?),
                    Err(error) => return Err(error),
                };
                let multiplier = multiplier.checked_mul(&polynomial)?;
                self.budget.polynomial(&multiplier)?;
                let mut factors = factors.to_vec();
                factors.push(scalar);
                self.vector(vector, multiplier, &factors, depth + 1)
            }
            ExprNode::Gradient(value) if matches!(self.dag.node(*value), Some(ExprNode::Symbol(SymbolRef::Field(field))) if *field == self.field) => {
                Ok(())
            }
            ExprNode::Gradient(value) => {
                let (atom, axis) = coordinate(self.dag, *value)?;
                if self.coordinates[axis] != Some(atom) {
                    return Err(Error::Mismatch);
                }
                self.flux[axis] = self.flux[axis].checked_add(&multiplier)?;
                self.budget.polynomial(&self.flux[axis])?;
                for &factor in factors {
                    self.relative_velocity(factor, axis, 0)?;
                }
                Ok(())
            }
            _ => Err(Error::UnsupportedExpression),
        }
    }

    fn relative_velocity(&mut self, id: ExprId, axis: usize, depth: usize) -> Result<(), Error> {
        self.budget.charge(1)?;
        if depth > 128 {
            return Err(Error::Limit);
        }
        let node = self.dag.node(id).ok_or(Error::InvalidExpression)?;
        if let ExprNode::Sub(material, rate) = node {
            let material = *material;
            if self.velocity_nodes.contains(&material) && self.normalize(*rate)? == self.rates[axis]
            {
                let velocity = self.normalize(material)?;
                if velocity.terms().any(|(atoms, _)| {
                    atoms.iter().any(|atom| {
                        !matches!(atom, Atom::Parameter(_) | Atom::Time | Atom::Coordinate(..))
                    })
                }) {
                    return Err(Error::UnsupportedExpression);
                }
                if self.velocities[axis]
                    .as_ref()
                    .is_some_and(|(_, prior)| *prior != velocity)
                {
                    return Err(Error::Mismatch);
                }
                self.velocities[axis] = Some((material, velocity));
                return Ok(());
            }
        }
        let mut operands = Vec::new();
        node.try_for_each_operand(|id| {
            operands.push(id);
            Ok::<_, Error>(())
        })?;
        for operand in operands {
            self.relative_velocity(operand, axis, depth + 1)?;
        }
        Ok(())
    }
}
