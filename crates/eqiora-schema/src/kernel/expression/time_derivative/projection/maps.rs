//! Proof-local map identity; no executable differentiation or persistent registry.
use super::super::derivative;
use super::{Atom, Budget, Error, ExactRational, Polynomial};
use crate::kernel::{CoordinateMapFactor, ExprDag, ExprId, ExprNode, SymbolRef};

#[derive(PartialEq, Eq)]
struct Map {
    factor: Option<CoordinateMapFactor>,
    source: Vec<Atom>,
    targets: Vec<Atom>,
    rows: Vec<Polynomial>,
    rates: Vec<Polynomial>,
}

#[derive(Default)]
pub(in super::super) struct Maps {
    maps: Vec<Map>,
    partials: Vec<(crate::kernel::SymbolRef, Atom)>,
    fields: Vec<MappedField>,
}

mod fields;
use fields::MappedField;

impl Maps {
    pub(in super::super) fn dynamic(&self, index: usize) -> bool {
        let map = &self.maps[index];
        map.factor != Some(CoordinateMapFactor::Orientation)
            && map.rates.iter().any(|rate| rate.terms().len() != 0)
    }

    pub(super) fn intern<'a>(
        &mut self,
        dag: &ExprDag,
        factor: Option<CoordinateMapFactor>,
        source: &[ExprId],
        at: &[(ExprId, ExprId)],
        get: &impl Fn(ExprId) -> Result<&'a Polynomial, Error>,
        budget: &mut Budget,
    ) -> Result<usize, Error> {
        budget.charge(source.len() + at.len())?;
        let source = source
            .iter()
            .map(|id| selector(dag, *id))
            .collect::<Result<_, _>>()?;
        let targets = at
            .iter()
            .map(|(id, _)| selector(dag, *id))
            .collect::<Result<_, _>>()?;
        let mut rows = Vec::with_capacity(at.len());
        for (_, id) in at {
            require_profile(dag, *id, true, false, budget)?;
            let row = get(*id)?;
            // Nested map factors and derivative-valued motion need a separate proof.
            if row.terms().any(|(atoms, _)| {
                atoms.iter().any(|atom| {
                    !matches!(
                        atom,
                        Atom::Field(_) | Atom::Parameter(_) | Atom::Time | Atom::Coordinate(..)
                    )
                })
            }) {
                return Err(Error::UnsupportedExpression);
            }
            budget.polynomial(row)?;
            rows.push(row.clone());
        }
        let rates = rows
            .iter()
            .map(|row| derivative(row, self, budget))
            .collect::<Result<Vec<_>, _>>()?;
        let candidate = Map {
            factor,
            source,
            targets,
            rows,
            rates,
        };
        // Bound structural comparisons as well as polynomial arithmetic.
        let size = candidate
            .rows
            .iter()
            .chain(&candidate.rates)
            .map(|row| 1 + row.terms().len() + row.factor_count())
            .sum::<usize>()
            + candidate.source.len()
            + candidate.targets.len();
        for (index, prior) in self.maps.iter().enumerate() {
            budget.charge(size)?;
            if prior == &candidate {
                return Ok(index);
            }
        }
        let index = self.maps.len();
        self.maps.push(candidate);
        Ok(index)
    }

    pub(super) fn action<'a>(
        &self,
        dag: &ExprDag,
        factor: &Polynomial,
        parameter: ExprId,
        directions: &[ExprId],
        get: &impl Fn(ExprId) -> Result<&'a Polynomial, Error>,
        budget: &mut Budget,
    ) -> Result<Polynomial, Error> {
        if !matches!(dag.node(parameter), Some(ExprNode::Symbol(SymbolRef::Time))) {
            return Err(Error::Mismatch);
        }
        let mut terms = factor.terms();
        let Some(([Atom::Map(index)], coefficient)) = terms.next() else {
            return Err(Error::UnsupportedExpression);
        };
        if terms.next().is_some() || coefficient != ExactRational::integer(1) {
            return Err(Error::UnsupportedExpression);
        }
        let map = &self.maps[*index];
        if directions.len() != map.rates.len() {
            return Err(Error::Mismatch);
        }
        for (direction, expected) in directions.iter().zip(&map.rates) {
            budget.polynomial(expected)?;
            if get(*direction)? != expected {
                return Err(Error::Mismatch);
            }
        }
        Ok(if self.dynamic(*index) {
            Polynomial::atom(Atom::MapRate(*index))
        } else {
            Polynomial::constant(ExactRational::integer(0))
        })
    }
}

fn selector(dag: &ExprDag, id: ExprId) -> Result<Atom, Error> {
    match dag.node(id) {
        Some(ExprNode::Symbol(SymbolRef::Coordinate {
            support,
            factor,
            axis,
        })) => Ok(Atom::Coordinate(*support, *factor, *axis)),
        _ => Err(Error::UnsupportedExpression),
    }
}

// Check the authored dependency profile before polynomial cancellation can hide
// an unsupported Field or a factor-valued mapped row.
fn require_profile(
    dag: &ExprDag,
    root: ExprId,
    allow_fields: bool,
    allow_rates: bool,
    budget: &mut Budget,
) -> Result<(), Error> {
    let mut pending = vec![(root, allow_fields, allow_rates)];
    let mut visited = std::collections::BTreeSet::new();
    while let Some((id, allow_fields, allow_rates)) = pending.pop() {
        if !visited.insert((id, allow_fields, allow_rates)) {
            continue;
        }
        budget.charge(1)?;
        let node = dag.node(id).ok_or(Error::InvalidExpression)?;
        match node {
            ExprNode::Symbol(SymbolRef::Field(_)) if allow_fields => {}
            ExprNode::Symbol(SymbolRef::Derivative(_, std::num::NonZeroU32::MIN))
                if allow_rates => {}
            ExprNode::Symbol(
                SymbolRef::Coordinate { .. } | SymbolRef::Parameter(_) | SymbolRef::Time,
            ) => {}
            ExprNode::Symbol(_)
            | ExprNode::CoordinateMapFactor { .. }
            | ExprNode::CoordinateMapFactorAction { .. } => {
                return Err(Error::UnsupportedExpression);
            }
            _ => {}
        }
        // Nested explicit polynomial pullbacks remain admissible. Unknown
        // mapped Fields are a first-map profile, even if cancellation hides them.
        let nested = matches!(node, ExprNode::Pullback { .. });
        node.try_for_each_operand(|operand| {
            budget.charge(1)?;
            pending.push((operand, allow_fields && !nested, allow_rates && !nested));
            Ok::<_, Error>(())
        })?;
    }
    Ok(())
}
