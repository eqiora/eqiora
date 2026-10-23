//! Equation-derived roles for the currently admitted primal and mixed forms.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::{Diagnostic, RawId, ValueType};
use eqiora_schema::kernel::{ExprDag, ExprNode, KernelNode, SymbolRef};
use eqiora_sem::KernelProgram;

use crate::canonical::{continuum_fields_on, lowering_error, relations_on};

pub(super) mod expression;
use expression::{coefficient_dependencies, field, kinematic, principal, strip_sign};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EquationRoles {
    pub(crate) fields: BTreeMap<RawId, (RawId, ValueType)>,
    pub(crate) relations: BTreeMap<RawId, EquationRole>,
    /// Exact constraint Relation -> (constrained trial, tested multiplier).
    pub(crate) constraints: BTreeMap<RawId, (RawId, RawId)>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EquationRole {
    pub(crate) domain: RawId,
    pub(crate) kind: Role,
    pub(crate) dependencies: BTreeSet<RawId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    Coefficient { field: RawId },
    Kinematic { state: RawId, rate: RawId },
    Residual { tested: RawId },
}

impl EquationRoles {
    pub(crate) fn derive(
        program: &KernelProgram,
        domains: impl IntoIterator<Item = RawId>,
    ) -> Result<Self, Diagnostic> {
        Self::derive_with_tests(program, domains, &BTreeMap::new())
    }

    pub(crate) fn derive_with_tests(
        program: &KernelProgram,
        domains: impl IntoIterator<Item = RawId>,
        tests: &BTreeMap<RawId, RawId>,
    ) -> Result<Self, Diagnostic> {
        let mut roles = Self {
            fields: BTreeMap::new(),
            relations: BTreeMap::new(),
            constraints: BTreeMap::new(),
        };
        for domain in domains {
            roles.domain(program, domain, tests)?;
        }
        for (relation, tested) in tests {
            if roles.relations.get(relation).map(|role| role.kind)
                != Some(Role::Residual { tested: *tested })
            {
                return Err(lowering_error(
                    *relation,
                    "explicit equation/test association is outside the admitted Domains",
                ));
            }
        }
        Ok(roles)
    }

    fn domain(
        &mut self,
        program: &KernelProgram,
        domain: RawId,
        tests: &BTreeMap<RawId, RawId>,
    ) -> Result<(), Diagnostic> {
        let fields = continuum_fields_on(program, domain)
            .into_iter()
            .collect::<BTreeSet<_>>();
        for id in &fields {
            let Some(KernelNode::Field(definition)) = program.node(*id) else {
                unreachable!()
            };
            if self
                .fields
                .insert(*id, (domain, definition.value_type().clone()))
                .is_some()
            {
                return Err(lowering_error(*id, "equation Field has duplicate support"));
            }
        }
        let mut equations = BTreeMap::new();
        for id in relations_on(program, domain) {
            let typed = super::scalar::typed_relation(program, id)?;
            if typed.expression().roots().len() != 1 {
                return Err(lowering_error(
                    id,
                    "equation role requires one residual root",
                ));
            }
            equations.insert(id, typed.expression().clone());
        }
        let mut state_rates = BTreeMap::new();
        for (id, dag) in &equations {
            if tests.contains_key(id) {
                continue;
            }
            if let Some((state, rate)) = kinematic(dag, dag.roots()[0]) {
                if !fields.contains(&state)
                    || !fields.contains(&rate)
                    || state == rate
                    || state_rates.insert(state, rate).is_some()
                {
                    return Err(lowering_error(
                        *id,
                        "kinematic state requires one exact local rate",
                    ));
                }
                self.insert(program, *id, domain, Role::Kinematic { state, rate }, dag)?;
            }
        }
        let mut coefficients = BTreeSet::new();
        loop {
            let mut changed = false;
            for (id, dag) in &equations {
                if self.relations.contains_key(id) || tests.contains_key(id) {
                    continue;
                }
                let root = strip_sign(dag, dag.roots()[0]);
                let candidates = match dag.node(root) {
                    // Canonical simplification represents f = 0 by the bare Field.
                    Some(ExprNode::Symbol(SymbolRef::Field(target)))
                        if fields.contains(&target.erase()) =>
                    {
                        vec![target.erase()]
                    }
                    Some(ExprNode::Sub(left, right)) => [(*left, *right), (*right, *left)]
                        .into_iter()
                        .filter_map(|(lhs, rhs)| {
                            let target = field(dag, lhs)?;
                            let dependencies = coefficient_dependencies(dag, rhs)?;
                            (fields.contains(&target)
                                && !dependencies.contains(&target)
                                && dependencies.is_subset(&coefficients))
                            .then_some(target)
                        })
                        .collect::<Vec<_>>(),
                    _ => Vec::new(),
                };
                if candidates.is_empty() {
                    continue;
                }
                if candidates.len() != 1
                    || state_rates.contains_key(&candidates[0])
                    || !coefficients.insert(candidates[0])
                {
                    return Err(lowering_error(
                        *id,
                        "coefficient Field requires one unambiguous definition",
                    ));
                }
                self.insert(
                    program,
                    *id,
                    domain,
                    Role::Coefficient {
                        field: candidates[0],
                    },
                    dag,
                )?;
                changed = true;
            }
            if !changed {
                break;
            }
        }
        if state_rates.values().any(|rate| coefficients.contains(rate)) {
            return Err(lowering_error(
                domain,
                "kinematic rate cannot be coefficient data",
            ));
        }
        let unknowns = fields
            .iter()
            .copied()
            .filter(|field| !coefficients.contains(field) && !state_rates.contains_key(field))
            .collect::<BTreeSet<_>>();
        let mut trials = BTreeMap::new();
        let mut residuals = BTreeMap::new();
        let mut constraints = Vec::new();
        for (id, dag) in &equations {
            if self.relations.contains_key(id) {
                continue;
            }
            if let Some(&tested) = tests.get(id) {
                if !unknowns.contains(&tested) || trials.insert(tested, BTreeSet::new()).is_some() {
                    return Err(lowering_error(
                        *id,
                        "explicit test must own one live unknown Field on this Domain",
                    ));
                }
                self.insert(program, *id, domain, Role::Residual { tested }, dag)?;
                let root = strip_sign(dag, dag.roots()[0]);
                if let Some(ExprNode::Divergence(value)) = dag.node(root)
                    && let Some(constrained) = field(dag, *value)
                {
                    self.constraints.insert(*id, (constrained, tested));
                }
                continue;
            }
            let root = strip_sign(dag, dag.roots()[0]);
            if let Some(ExprNode::Divergence(value)) = dag.node(root)
                && let Some(trial) = field(dag, *value)
            {
                constraints.push((*id, trial));
                continue;
            }
            let principal_root = match program.node(*id) {
                Some(KernelNode::Relation(relation)) => match relation.meaning() {
                    eqiora_schema::kernel::RelationMeaning::Conservation(terms) => {
                        dag.nodes().iter().position(|node| matches!(node, ExprNode::Divergence(flux) if *flux == terms.flux()))
                            .and_then(|index| dag.node_id(index as u32))
                            .unwrap_or(root)
                    }
                    _ => root,
                },
                _ => root,
            };
            let (principal_fields, multipliers) = principal(dag, principal_root, &coefficients)?;
            let principal_fields = principal_fields
                .into_iter()
                .map(|field| state_rates.get(&field).copied().unwrap_or(field))
                .filter(|field| !coefficients.contains(field))
                .collect::<BTreeSet<_>>();
            if principal_fields.is_empty() || !principal_fields.is_subset(&unknowns) {
                return Err(lowering_error(
                    *id,
                    "equation has no unique supported principal trial; cyclic or missing coefficient definitions are not inferred",
                ));
            }
            let multipliers = multipliers
                .into_iter()
                .filter(|field| {
                    unknowns.contains(field) && self.fields[field].1.shape().is_scalar()
                })
                .collect::<BTreeSet<_>>();
            residuals.insert(*id, (principal_fields, multipliers));
        }
        // Infer the unique equation-to-test-space assignment across the whole
        // domain. Cross diffusion can leave a row ambiguous in isolation while
        // another equation fixes one of its candidate Fields. Never break a tie
        // by declaration order or Field identity.
        while !residuals.is_empty() {
            for (candidates, _) in residuals.values_mut() {
                candidates.retain(|field| !trials.contains_key(field));
            }
            let id = residuals
                .iter()
                .find_map(|(id, (candidates, _))| (candidates.len() == 1).then_some(*id))
                .ok_or_else(|| {
                    lowering_error(
                        domain,
                        "equations have no unique supported principal trial assignment; cyclic or missing coefficient definitions are not inferred and ambiguous test spaces require an explicit decision",
                    )
                })?;
            let (candidates, multipliers) = residuals.remove(&id).expect("selected residual");
            let tested = *candidates.first().expect("one remaining trial");
            trials.insert(tested, multipliers);
            self.insert(
                program,
                id,
                domain,
                Role::Residual { tested },
                &equations[&id],
            )?;
        }
        for (id, constrained) in constraints {
            let candidates = trials.get(&constrained).ok_or_else(|| {
                lowering_error(id, "divergence constraint has no paired principal equation")
            })?;
            if candidates.len() != 1 {
                return Err(lowering_error(
                    id,
                    "divergence constraint requires one unique gradient multiplier",
                ));
            }
            let tested = *candidates.first().expect("one multiplier");
            self.constraints.insert(id, (constrained, tested));
            self.insert(
                program,
                id,
                domain,
                Role::Residual { tested },
                &equations[&id],
            )?;
        }
        let tested = self
            .relations
            .values()
            .filter(|entry| entry.domain == domain)
            .filter_map(|entry| match entry.kind {
                Role::Residual { tested } => Some(tested),
                _ => None,
            })
            .collect::<Vec<_>>();
        if tested.len() != unknowns.len()
            || tested.iter().copied().collect::<BTreeSet<_>>() != unknowns
        {
            return Err(lowering_error(
                domain,
                "equation test spaces must cover every unknown exactly once",
            ));
        }
        Ok(())
    }

    fn insert(
        &mut self,
        program: &KernelProgram,
        id: RawId,
        domain: RawId,
        kind: Role,
        dag: &ExprDag,
    ) -> Result<(), Diagnostic> {
        let dependencies = checked_relation_dependencies(program, id, dag).ok_or_else(|| {
            lowering_error(
                id,
                "Relation dependency inventory differs from its equation",
            )
        })?;
        self.relations.insert(
            id,
            EquationRole {
                domain,
                kind,
                dependencies,
            },
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests;

pub(super) fn checked_relation_dependencies(
    program: &KernelProgram,
    relation: RawId,
    dag: &ExprDag,
) -> Option<BTreeSet<RawId>> {
    let dependencies = dag
        .nodes()
        .iter()
        .filter_map(|node| match node {
            ExprNode::Symbol(SymbolRef::Field(id) | SymbolRef::Derivative(id, _)) => {
                Some(id.erase())
            }
            ExprNode::Symbol(SymbolRef::Parameter(id)) => Some(id.erase()),
            ExprNode::Symbol(SymbolRef::Coordinate { support, .. }) => Some(support.erase()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let graph = program
        .edges()
        .iter()
        .filter(|edge| edge.from() == relation && edge.kind() == eqiora_graph::EdgeKind::DependsOn)
        .map(|edge| edge.to())
        .collect();
    (dependencies == graph).then_some(dependencies)
}
