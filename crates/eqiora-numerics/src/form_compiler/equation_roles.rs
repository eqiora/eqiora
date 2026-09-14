//! Equation-derived roles for the currently admitted primal and mixed forms.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::{Diagnostic, RawId, ValueType};
use eqiora_schema::kernel::{ExprDag, ExprNode, KernelNode, SymbolRef};
use eqiora_sem::KernelProgram;

use crate::canonical::{continuum_fields_on, lowering_error, relations_on};

mod expression;
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
        let mut roles = Self {
            fields: BTreeMap::new(),
            relations: BTreeMap::new(),
            constraints: BTreeMap::new(),
        };
        for domain in domains {
            roles.domain(program, domain)?;
        }
        Ok(roles)
    }

    fn domain(&mut self, program: &KernelProgram, domain: RawId) -> Result<(), Diagnostic> {
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
                if self.relations.contains_key(id) {
                    continue;
                }
                let root = strip_sign(dag, dag.roots()[0]);
                let Some(ExprNode::Sub(left, right)) = dag.node(root) else {
                    continue;
                };
                let candidates = [(*left, *right), (*right, *left)]
                    .into_iter()
                    .filter_map(|(lhs, rhs)| {
                        let target = field(dag, lhs)?;
                        let dependencies = coefficient_dependencies(dag, rhs)?;
                        (fields.contains(&target)
                            && !dependencies.contains(&target)
                            && dependencies.is_subset(&coefficients))
                        .then_some(target)
                    })
                    .collect::<Vec<_>>();
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
        let mut constraints = Vec::new();
        for (id, dag) in &equations {
            if self.relations.contains_key(id) {
                continue;
            }
            let root = strip_sign(dag, dag.roots()[0]);
            if let Some(ExprNode::Divergence(value)) = dag.node(root)
                && let Some(trial) = field(dag, *value)
            {
                constraints.push((*id, trial));
                continue;
            }
            let (principal_fields, multipliers) = principal(dag, root)?;
            let principal_fields = principal_fields
                .into_iter()
                .map(|field| state_rates.get(&field).copied().unwrap_or(field))
                .filter(|field| !coefficients.contains(field))
                .collect::<BTreeSet<_>>();
            if principal_fields.len() != 1 || !principal_fields.is_subset(&unknowns) {
                return Err(lowering_error(
                    *id,
                    "equation has no unique supported principal trial; cyclic or missing coefficient definitions are not inferred",
                ));
            }
            let tested = *principal_fields.first().expect("one trial");
            let multipliers = multipliers
                .into_iter()
                .filter(|field| {
                    unknowns.contains(field) && self.fields[field].1.shape().is_scalar()
                })
                .collect::<BTreeSet<_>>();
            if trials.insert(tested, multipliers).is_some() {
                return Err(lowering_error(
                    *id,
                    "multiple residuals require an explicit test-space decision",
                ));
            }
            self.insert(program, *id, domain, Role::Residual { tested }, dag)?;
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
        let dependencies = dag
            .nodes()
            .iter()
            .filter_map(|node| match node {
                ExprNode::Symbol(SymbolRef::Field(id) | SymbolRef::Derivative(id)) => {
                    Some(id.erase())
                }
                ExprNode::Symbol(SymbolRef::Parameter(id)) => Some(id.erase()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let graph = program
            .edges()
            .iter()
            .filter(|edge| edge.from() == id && edge.kind() == eqiora_graph::EdgeKind::DependsOn)
            .map(|edge| edge.to())
            .collect();
        if dependencies != graph {
            return Err(lowering_error(
                id,
                "Relation dependency inventory differs from its equation",
            ));
        }
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
