//! Typed equation-derived region forms, before global Field DOF mapping.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::{Diagnostic, RawId, ScalarDomain, ValueFrame, ValueType};
use eqiora_sem::KernelProgram;

use super::bilinear::Pairing;
use super::equation_roles::{EquationRoles, Role};
use super::linear::data::{Context, Data};
use super::scalar::{continuous_activations, require_closed_dag, typed_relation};

mod binding;
mod boundary;
pub(crate) use boundary::RegionBoundaryLaw;
mod boundary_integral;
mod evaluate;
mod integration;
mod scalar;
pub(super) use integration::integrate_scalar;
pub(super) use scalar::ScalarRow;
mod flux;
mod lowering;
mod nonlinear;
pub(crate) use nonlinear::RegionLinearization;
#[cfg(test)]
mod tests;

pub(crate) use binding::{
    BoundRegionForm, RegionFieldBinding, RegionFieldLayout, RegionTimeBinding, basis, state_layout,
    state_layouts,
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompiledRegionForm {
    domain: RawId,
    dimension: usize,
    roles: EquationRoles,
    rows: Vec<Row>,
}

#[derive(Debug, Clone, PartialEq)]
struct Row {
    relation: RawId,
    tested: RawId,
    value_type: ValueType,
    terms: Vec<Term>,
    dyadics: Vec<nonlinear::DyadicTerm>,
    flux: Vec<flux::FluxTerm>,
    forcing: Vec<Data>,
}

#[derive(Debug, Clone, PartialEq)]
struct Term {
    trial: RawId,
    derivative: bool,
    pairing: Pairing,
    coefficient: Data,
    positive_diffusion: bool,
}

impl CompiledRegionForm {
    pub(crate) fn derive(
        program: &KernelProgram,
        domain: RawId,
        dimension: usize,
    ) -> Result<Self, Diagnostic> {
        if !(1..=3).contains(&dimension) {
            return Err(invalid("region dimension must be one through three"));
        }
        let roles = EquationRoles::derive(program, [domain])?;
        for (_, value_type) in roles.fields.values() {
            components(value_type, dimension)?;
        }
        let coefficients = super::linear::coefficients(program, dimension, &roles)?;
        let mut rows = BTreeMap::new();
        for (relation, role) in &roles.relations {
            let typed = typed_relation(program, *relation)?;
            require_closed_dag(typed.expression(), *relation)?;
            for node_type in typed.node_types() {
                if let Some(support) = &node_type.support
                    && (*support.domain() != domain || support.dimensions() != dimension)
                {
                    return Err(invalid("region equation has mismatched typed support"));
                }
            }
            let Role::Residual { tested } = role.kind else {
                continue;
            };
            let root = typed.expression().roots()[0];
            let value_type = typed
                .node_type(root)
                .expect("typed root")
                .value_type
                .clone();
            components(&value_type, dimension)?;
            let test_type = &roles.fields[&tested].1;
            if value_type.shape() != test_type.shape() || value_type.frame() != test_type.frame() {
                return Err(invalid(
                    "residual and test Field have different component roles",
                ));
            }
            let context = Context {
                program,
                dag: typed.expression(),
                owner: *relation,
                dimension,
                coefficients: &coefficients,
            };
            let mut row = Row {
                relation: *relation,
                tested,
                value_type: value_type.clone(),
                terms: Vec::new(),
                dyadics: Vec::new(),
                flux: Vec::new(),
                forcing: vec![Data::constant(dimension, 0.0); components(&value_type, dimension)?],
            };
            lowering::lower(&context, root, Data::constant(dimension, 1.0), &mut row, 0)?;
            for term in &mut row.dyadics {
                let field = roles
                    .fields
                    .get(&term.field)
                    .ok_or_else(|| invalid("dyadic trial is not a local Field"))?;
                if components(&field.1, dimension)? != dimension
                    || value_type.shape() != field.1.shape()
                    || value_type.frame() != field.1.frame()
                {
                    return Err(invalid(
                        "dyadic flux requires a spatial vector row and trial",
                    ));
                }
                term.split = !term.coefficient.spatial()
                    && roles
                        .constraints
                        .values()
                        .any(|(trial, _)| *trial == term.field);
            }
            for term in &row.terms {
                let trial = roles
                    .fields
                    .get(&term.trial)
                    .ok_or_else(|| invalid("region trial is not a local Field"))?;
                let test_scalar = test_type.shape().is_scalar();
                let trial_scalar = trial.1.shape().is_scalar();
                let valid = match term.pairing {
                    Pairing::Value | Pairing::Gradient => {
                        test_type.shape() == trial.1.shape() && test_type.frame() == trial.1.frame()
                    }
                    Pairing::SymmetricGradient | Pairing::Divergence => {
                        !test_scalar && !trial_scalar
                    }
                    Pairing::TestDivergenceTrialValue => !test_scalar && trial_scalar,
                    Pairing::TestValueTrialDivergence => test_scalar && !trial_scalar,
                };
                if !valid {
                    return Err(invalid(
                        "weak operator has mismatched test/trial component roles",
                    ));
                }
            }
            rows.insert(tested, row);
        }
        if rows.is_empty() {
            return Err(invalid("region needs a residual equation"));
        }
        continuous_activations(
            program,
            &roles.relations.keys().copied().collect::<BTreeSet<_>>(),
        )?;
        Ok(Self {
            domain,
            dimension,
            roles,
            rows: rows.into_values().collect(),
        })
    }

    pub(crate) const fn domain(&self) -> RawId {
        self.domain
    }

    pub(crate) fn rows(&self) -> impl Iterator<Item = (RawId, RawId, &ValueType)> {
        self.rows
            .iter()
            .map(|row| (row.relation, row.tested, &row.value_type))
    }

    pub(crate) fn fields(&self) -> impl Iterator<Item = (RawId, &ValueType)> {
        self.rows
            .iter()
            .map(|row| (row.tested, &self.roles.fields[&row.tested].1))
    }
}

pub(crate) fn components(value_type: &ValueType, dimension: usize) -> Result<usize, Diagnostic> {
    if value_type.scalar_domain() == ScalarDomain::Real && value_type.array_rank() == 0 {
        if value_type.shape().is_scalar() && value_type.frame() == ValueFrame::Invariant {
            return Ok(1);
        }
        if value_type.shape().rank() == 1
            && value_type.frame() == ValueFrame::SpatialCartesian
            && value_type.shape().component_count() == Some(dimension)
        {
            return Ok(dimension);
        }
    }
    Err(invalid(
        "region execution requires real invariant scalars or spatial vectors",
    ))
}

fn invalid(message: &str) -> Diagnostic {
    Diagnostic::error(
        eqiora_core::diagnostic::codes::INVALID_DISCRETIZATION,
        message,
    )
}
