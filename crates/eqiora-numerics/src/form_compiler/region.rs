//! Typed equation-derived region forms, before global Field DOF mapping.

use crate::spatial_expression::Coefficient;
use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::{Diagnostic, RawId, ScalarDomain, ValueFrame, ValueType};
use eqiora_sem::KernelProgram;

use super::bilinear::Pairing;
use super::equation_roles::{EquationRoles, Role};
use super::scalar::{continuous_activations, require_closed_dag, typed_relation};
use super::spatial::data::{Context, Data};

pub(crate) mod balance;
mod binding;
mod graph;
pub(crate) use graph::RegionGraph;
mod boundary;
mod datum;
pub(in crate::form_compiler) use datum::PrescribedDatum;
mod compatibility;
mod conservative;
pub(crate) use boundary::RegionBoundaryLaw;
pub(crate) use conservative::ConservativeSystem;
mod boundary_integral;
mod prepared_facet;
pub(crate) use prepared_facet::PreparedRegionFacet;
mod evaluate;
pub(super) mod integration;
mod orientation;
mod quadrature;
mod scalar;
pub(super) use integration::integrate_scalar;
pub(super) use scalar::ScalarRow;
mod flux;
mod lowering;
mod nonlinear;
pub(crate) use nonlinear::RegionLinearization;
mod error_estimate;
mod prepared;
pub(crate) use prepared::PreparedRegionCell;
#[cfg(test)]
mod tests;

pub(crate) use binding::{
    BoundRegionForm, RegionFieldBinding, RegionFieldLayout, RegionTimeBinding, basis, state_layout,
    state_layouts,
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompiledRegionForm<S: Coefficient> {
    domain: RawId,
    dimension: usize,
    roles: EquationRoles,
    rows: Vec<Row<S>>,
}

#[derive(Debug, Clone, PartialEq)]
struct Row<S: Coefficient> {
    relation: RawId,
    tested: RawId,
    value_type: ValueType,
    terms: Vec<Term<S>>,
    dyadics: Vec<nonlinear::DyadicTerm<S>>,
    flux: Vec<flux::FluxTerm<S>>,
    forcing: Vec<Data<S>>,
}

#[derive(Debug, Clone, PartialEq)]
struct Term<S: Coefficient> {
    trial: RawId,
    derivative: bool,
    pairing: Pairing,
    coefficient: Data<S>,
    positive_diffusion: bool,
}

impl<S: Coefficient> Term<S> {
    fn require_coefficient(&self, value: S) -> Result<S, Diagnostic> {
        if !value.is_finite()
            || (self.positive_diffusion && (value.im() != 0.0 || value.re() <= 0.0))
        {
            return Err(invalid(
                "region requires positive finite diffusion and finite reaction/forcing",
            ));
        }
        Ok(value)
    }
}

impl<S: Coefficient> CompiledRegionForm<S> {
    pub(crate) fn derive(
        program: &KernelProgram,
        domain: RawId,
        dimension: usize,
    ) -> Result<Self, Diagnostic> {
        if !(1..=3).contains(&dimension) {
            return Err(invalid("region dimension must be one through three"));
        }
        let roles = EquationRoles::derive(program, [domain])?;
        let coefficients = super::spatial::coefficients(program, dimension, &roles)?;
        for (field, (_, value_type)) in &roles.fields {
            if !coefficients.contains_key(field)
                && value_type.scalar_domain() == ScalarDomain::Complex
                && S::DOMAIN != ScalarDomain::Complex
            {
                return Err(invalid(
                    "complex region Fields require a complex coefficient representation",
                ));
            }
            components(value_type, dimension)?;
        }
        let mut rows = BTreeMap::new();
        for (relation, role) in &roles.relations {
            let typed = typed_relation(program, *relation)?;
            require_closed_dag(typed.expression(), *relation)?;
            for node_type in typed.node_types() {
                if let Some(support) = &node_type.support
                    && (*support.domain() != domain
                        || support.ambient_dimensions() != Some(dimension))
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
            if !same_component_type(&value_type, test_type) {
                return Err(invalid(
                    "residual and test Field have different component roles",
                ));
            }
            let context = Context {
                time_s: None,
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
                forcing: vec![
                    Data::constant(dimension, <S as From<f64>>::from(0.0));
                    components(&value_type, dimension)?
                ],
            };
            lowering::lower(
                &context,
                root,
                Data::constant(dimension, <S as From<f64>>::from(1.0)),
                &mut row,
                0,
            )?;
            for term in &mut row.dyadics {
                let field = roles
                    .fields
                    .get(&term.field)
                    .ok_or_else(|| invalid("dyadic trial is not a local Field"))?;
                if !spatial_vector(&field.1, dimension)
                    || !same_component_type(&value_type, &field.1)
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
                let test_vector = spatial_vector(test_type, dimension);
                let trial_vector = spatial_vector(&trial.1, dimension);
                let valid = match term.pairing {
                    Pairing::Value | Pairing::Gradient => same_component_type(test_type, &trial.1),
                    Pairing::Components { .. } => term.pairing.accepts(
                        dimension,
                        components(test_type, dimension)?,
                        components(&trial.1, dimension)?,
                    ),
                    Pairing::SkewGradient(axis) => {
                        axis < dimension && same_component_type(test_type, &trial.1)
                    }
                    Pairing::TestGradientTrialValue(axis) => {
                        axis < dimension && test_scalar && trial_scalar
                    }
                    Pairing::Curl => dimension == 3 && test_vector && trial_vector,
                    Pairing::SymmetricGradient | Pairing::Divergence => test_vector && trial_vector,
                    Pairing::TestDivergenceTrialValue => test_vector && trial_scalar,
                    Pairing::TestValueTrialDivergence => test_scalar && trial_vector,
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

    pub(crate) fn is_nonlinear(&self) -> bool {
        self.rows.iter().any(|row| !row.dyadics.is_empty())
    }

    pub(crate) fn rows(&self) -> impl Iterator<Item = (RawId, RawId, &ValueType)> {
        self.rows
            .iter()
            .map(|row| (row.relation, row.tested, &row.value_type))
    }

    /// Inventory of weak operations actually retained by the compiled rows.
    pub(crate) fn formulation_rules(&self) -> BTreeSet<&'static str> {
        use super::vocabulary::{FormulationRule, MixedFormulationRule};
        let mut rules = BTreeSet::from([
            if S::DOMAIN == ScalarDomain::Complex {
                FormulationRule::ConjugatedTestPairing.id()
            } else {
                FormulationRule::TestPairing.id()
            },
            FormulationRule::SourcePairing.id(),
        ]);
        for term in self.rows.iter().flat_map(|row| &row.terms) {
            let rule = match term.pairing {
                Pairing::Value
                | Pairing::Components {
                    test: super::bilinear::BasisComponent::Value(_),
                    ..
                }
                | Pairing::TestValueTrialDivergence => FormulationRule::ValuePairing,
                Pairing::Curl => FormulationRule::VectorCurlCurlByParts,
                Pairing::Components {
                    test: super::bilinear::BasisComponent::Gradient { .. },
                    ..
                }
                | Pairing::Gradient
                | Pairing::SymmetricGradient
                | Pairing::Divergence
                | Pairing::SkewGradient(_)
                | Pairing::TestGradientTrialValue(_)
                | Pairing::TestDivergenceTrialValue => FormulationRule::DivergenceByParts,
            };
            rules.insert(rule.id());
        }
        if !self.roles.constraints.is_empty() {
            rules.insert(MixedFormulationRule::ContinuityConstraintPairing.id());
        }
        rules
    }

    pub(crate) fn constraint_pairs(&self) -> impl Iterator<Item = (RawId, RawId)> + '_ {
        self.roles.constraints.values().copied()
    }

    pub(crate) fn requires_boundary(&self, field: RawId) -> bool {
        self.rows
            .iter()
            .any(|row| row.tested == field && (!row.flux.is_empty() || !row.dyadics.is_empty()))
            && !self
                .constraint_pairs()
                .any(|(_, multiplier)| multiplier == field)
    }

    pub(crate) fn fields(&self) -> impl Iterator<Item = (RawId, &ValueType)> {
        self.rows
            .iter()
            .map(|row| (row.tested, &self.roles.fields[&row.tested].1))
    }
}

pub(crate) fn components(value_type: &ValueType, dimension: usize) -> Result<usize, Diagnostic> {
    let spatial_axes = &value_type.shape().extents()[value_type.array_rank()..];
    if !matches!(
        value_type.scalar_domain(),
        ScalarDomain::Real | ScalarDomain::Complex
    ) || (value_type.frame() == ValueFrame::SpatialCartesian
        && spatial_axes
            .iter()
            .any(|axis| axis.get() as usize != dimension))
    {
        return Err(invalid(
            "region components require numeric values with exact Cartesian extents",
        ));
    }
    value_type
        .shape()
        .component_count()
        .ok_or_else(|| invalid("region component count overflow"))
}

fn same_component_type(left: &ValueType, right: &ValueType) -> bool {
    left.clone()
        .with_dimension(right.dimension())
        .is_ok_and(|ty| &ty == right)
}

pub(super) fn spatial_vector(value_type: &ValueType, dimension: usize) -> bool {
    value_type.array_rank() == 0
        && value_type.frame() == ValueFrame::SpatialCartesian
        && value_type.shape().rank() == 1
        && value_type.shape().component_count() == Some(dimension)
}

fn invalid(message: &str) -> Diagnostic {
    Diagnostic::error(
        eqiora_core::diagnostic::codes::INVALID_DISCRETIZATION,
        message,
    )
}
