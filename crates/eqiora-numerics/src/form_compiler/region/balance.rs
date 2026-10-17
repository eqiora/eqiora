//! Physical volume work over one fixed-geometry backward-Euler step.
//!
//! This is a signed quadratic identity, not a positivity certificate or a full
//! conservation claim. Boundary work must be supplied by the owning step's
//! authenticated boundary/interface acceptance. No normalized CSR is consumed.
use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, DimExponents, RawId};
use eqiora_meshing::{AffineGeometryMap, GeometryMap, QuadratureRule};
use eqiora_schema::kernel::{ExprNode, KernelNode, SymbolRef};
use eqiora_sem::KernelProgram;

use super::{BoundRegionForm, Row, Term, invalid};

mod sampling;
use sampling::sample_fields;
#[cfg(test)]
mod tests;

/// Mathematical role of a contribution; none implies positive definiteness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialBalanceRole {
    Storage,
    StatePotential,
    Dissipation,
    ConstraintWork,
    AppliedVolumeWork,
}

/// Directed term lineage is retained even for reciprocal cross-field terms.
/// The two directed halves together give one off-diagonal quadratic term.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialBalanceTerm {
    pub(crate) relation: RawId,
    pub(crate) tested: RawId,
    pub(crate) trial: Option<RawId>,
    pub(crate) term: usize,
    pub(crate) role: SpatialBalanceRole,
    pub(crate) dimension: DimExponents,
    pub(crate) previous: f64,
    pub(crate) current: f64,
    pub(crate) increment: f64,
    pub(crate) work: f64,
}

impl SpatialBalanceTerm {
    /// Relation that contributed this directed term.
    pub fn relation(&self) -> RawId {
        self.relation
    }
    /// Field whose test function is paired with the term.
    pub fn tested_field(&self) -> RawId {
        self.tested
    }
    /// Trial Field, absent for a prescribed volume load.
    pub fn trial_field(&self) -> Option<RawId> {
        self.trial
    }
    /// Position in the retained directed term inventory of this Relation.
    pub fn term_index(&self) -> usize {
        self.term
    }
    /// Mathematical role in the backward-Euler work identity.
    pub fn role(&self) -> SpatialBalanceRole {
        self.role
    }
    /// Quadratic value before the step for storage or state-potential terms.
    pub fn previous(&self) -> eqiora_core::DynQuantity {
        eqiora_core::DynQuantity::new(self.previous, self.dimension)
    }
    /// Quadratic value after the step for storage or state-potential terms.
    pub fn current(&self) -> eqiora_core::DynQuantity {
        eqiora_core::DynQuantity::new(self.current, self.dimension)
    }
    /// Backward-Euler quadratic increment for storage or state-potential terms.
    pub fn increment(&self) -> eqiora_core::DynQuantity {
        eqiora_core::DynQuantity::new(self.increment, self.dimension)
    }
    /// Time-integrated directed work. The role determines its sign in the balance.
    pub fn work(&self) -> eqiora_core::DynQuantity {
        eqiora_core::DynQuantity::new(self.work, self.dimension)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegionStepBalance {
    pub(crate) domain: RawId,
    pub(crate) contributions: Vec<SpatialBalanceTerm>,
}

impl RegionStepBalance {
    /// Volume residual work, grouped by units. Surface work is deliberately not
    /// inferred: callers must compare each dimension with its boundary work.
    pub(crate) fn volume_work(&self) -> Result<Vec<(DimExponents, f64)>, Diagnostic> {
        let mut sums = BTreeMap::<[(i32, i32); 7], f64>::new();
        for term in &self.contributions {
            let value = match term.role {
                SpatialBalanceRole::Storage | SpatialBalanceRole::StatePotential => {
                    term.current - term.previous + term.increment
                }
                SpatialBalanceRole::Dissipation | SpatialBalanceRole::ConstraintWork => term.work,
                SpatialBalanceRole::AppliedVolumeWork => -term.work,
            };
            let sum = sums.entry(term.dimension.exponents()).or_insert(0.0);
            *sum += value;
            if !sum.is_finite() {
                return Err(invalid("volume balance accumulation is nonfinite"));
            }
        }
        Ok(sums
            .into_iter()
            .map(|(dimension, value)| {
                (
                    DimExponents::from_rationals(dimension).expect("retained dimensions"),
                    value,
                )
            })
            .collect())
    }
}

impl BoundRegionForm<f64> {
    /// Evaluate physical, basis-major/component-minor coefficients on the same
    /// affine cell at both endpoints. The caller authenticates the fixed mesh.
    /// Every algebraic and eliminated state Field must occur exactly once in
    /// both maps. Coefficient scales and numerical test signs are not physical
    /// fields and are intentionally absent from this evaluation.
    pub(crate) fn volume_step_balance(
        &self,
        program: &KernelProgram,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
        previous: &BTreeMap<RawId, Vec<f64>>,
        current: &BTreeMap<RawId, Vec<f64>>,
    ) -> Result<RegionStepBalance, Diagnostic> {
        self.validate_geometry(geometry, quadrature)?;
        let dt = self
            .time_step()
            .ok_or_else(|| invalid("volume balance requires BackwardEuler"))?;
        // Time is folded into Data at lowering. Inspect retained source before
        // claiming the constant-coefficient quadratic identity.
        if super::EquationRoles::derive(program, [self.form.domain])? != self.form.roles {
            return Err(invalid(
                "volume balance source has different equation roles",
            ));
        }
        for id in self.form.roles.relations.keys() {
            let Some(KernelNode::Relation(relation)) = program.node(*id) else {
                return Err(invalid("volume balance is missing its source Relation"));
            };
            if relation
                .expression()
                .nodes()
                .iter()
                .any(|node| matches!(node, ExprNode::Symbol(SymbolRef::Time)))
            {
                return Err(invalid(
                    "volume balance does not admit time-dependent coefficient or load data",
                ));
            }
        }
        if self.form.rows.iter().any(|row| !row.dyadics.is_empty()) {
            return Err(invalid("volume balance does not admit nonlinear fluxes"));
        }
        let states = self
            .eliminations
            .iter()
            .map(|(&state, &rate)| (rate, state))
            .collect::<BTreeMap<_, _>>();
        if states.len() != self.eliminations.len() {
            return Err(invalid(
                "volume balance requires a unique state for each eliminated rate",
            ));
        }
        let multipliers = self
            .form
            .constraint_pairs()
            .map(|(_, field)| field)
            .collect::<std::collections::BTreeSet<_>>();
        let mut contributions = Vec::new();
        for row in &self.form.rows {
            let units = row
                .value_type
                .dimension()
                .mul(self.form.roles.fields[&row.tested].1.dimension())
                .and_then(|units| {
                    units.mul(DimExponents::from_integers([
                        0,
                        self.form.dimension as i32,
                        1,
                        0,
                        0,
                        0,
                        0,
                    ])?)
                })
                .ok_or_else(|| invalid("volume balance dimensions overflow"))?;
            for (index, term) in row.terms.iter().enumerate() {
                let role = if multipliers.contains(&row.tested) || multipliers.contains(&term.trial)
                {
                    if term.derivative || self.eliminations.contains_key(&term.trial) {
                        return Err(invalid("volume balance requires algebraic constraint work"));
                    }
                    SpatialBalanceRole::ConstraintWork
                } else if term.derivative && !self.eliminations.contains_key(&term.trial) {
                    SpatialBalanceRole::Storage
                } else if !term.derivative && self.eliminations.contains_key(&term.trial) {
                    SpatialBalanceRole::StatePotential
                } else {
                    SpatialBalanceRole::Dissipation
                };
                if role != SpatialBalanceRole::ConstraintWork {
                    require_reciprocal(&self.form.rows, &self.eliminations, row, term, role)?;
                    let other_field = self
                        .eliminations
                        .get(&term.trial)
                        .copied()
                        .unwrap_or(term.trial);
                    let other = self
                        .form
                        .rows
                        .iter()
                        .find(|row| row.tested == other_field)
                        .expect("proved reciprocal row");
                    if other
                        .value_type
                        .dimension()
                        .mul(self.form.roles.fields[&other_field].1.dimension())
                        != row
                            .value_type
                            .dimension()
                            .mul(self.form.roles.fields[&row.tested].1.dimension())
                    {
                        return Err(invalid(
                            "reciprocal balance terms have unlike physical work dimensions",
                        ));
                    }
                }
                if role == SpatialBalanceRole::StatePotential && !states.contains_key(&row.tested) {
                    return Err(invalid(
                        "potential work lacks the exact tested state/rate pair",
                    ));
                }
                contributions.push(SpatialBalanceTerm {
                    relation: row.relation,
                    tested: row.tested,
                    trial: Some(term.trial),
                    term: index,
                    role,
                    dimension: units,
                    previous: 0.,
                    current: 0.,
                    increment: 0.,
                    work: 0.,
                });
            }
            contributions.push(SpatialBalanceTerm {
                relation: row.relation,
                tested: row.tested,
                trial: None,
                term: row.terms.len(),
                role: SpatialBalanceRole::AppliedVolumeWork,
                dimension: units,
                previous: 0.,
                current: 0.,
                increment: 0.,
                work: 0.,
            });
        }
        for point in quadrature.points() {
            let samples = sample_fields(self, geometry, &point.coordinates, previous, current)?;
            let mut physical = vec![0.; self.form.dimension];
            geometry.map_point(&point.coordinates, &mut physical)?;
            let weight = point.weight * geometry.measure_scale();
            let mut output = contributions.iter_mut();
            for row in &self.form.rows {
                for term in &row.terms {
                    let result = output.next().expect("one contribution per term");
                    let coefficient = term.coefficient.evaluate(&physical)?;
                    let test = if result.role == SpatialBalanceRole::StatePotential {
                        states[&row.tested]
                    } else {
                        row.tested
                    };
                    let trial = if term.derivative {
                        self.eliminations
                            .get(&term.trial)
                            .copied()
                            .unwrap_or(term.trial)
                    } else {
                        term.trial
                    };
                    let left = &samples[&test];
                    let right = &samples[&trial];
                    if matches!(
                        result.role,
                        SpatialBalanceRole::Storage | SpatialBalanceRole::StatePotential
                    ) {
                        result.previous += 0.5
                            * weight
                            * coefficient
                            * term
                                .pairing
                                .entry(left.previous.basis(), right.previous.basis());
                        result.current += 0.5
                            * weight
                            * coefficient
                            * term
                                .pairing
                                .entry(left.current.basis(), right.current.basis());
                        result.increment += 0.5
                            * weight
                            * coefficient
                            * term
                                .pairing
                                .entry(left.difference.basis(), right.difference.basis());
                    } else {
                        result.work += dt
                            * weight
                            * coefficient
                            * term
                                .pairing
                                .entry(left.current.basis(), right.current.basis());
                    }
                }
                let result = output.next().expect("one applied contribution per row");
                let sample = &samples[&row.tested].current;
                let mut work = 0.;
                for (component, datum) in row.forcing.iter().enumerate() {
                    work += datum.evaluate(&physical)? * sample.value[component];
                }
                for flux in &row.flux {
                    if let super::flux::FluxTerm::Isotropic(datum) = flux {
                        if sample.value.len() != self.form.dimension {
                            return Err(invalid("isotropic volume work requires a spatial vector"));
                        }
                        work -= datum.evaluate(&physical)?
                            * (0..self.form.dimension)
                                .map(|axis| sample.gradient[axis * self.form.dimension + axis])
                                .sum::<f64>();
                    }
                }
                result.work += dt * weight * work;
            }
        }
        if contributions.iter().any(|term| {
            [term.previous, term.current, term.increment, term.work]
                .iter()
                .any(|value| !value.is_finite())
        }) {
            return Err(invalid("physical volume balance is nonfinite"));
        }
        Ok(RegionStepBalance {
            domain: self.form.domain,
            contributions,
        })
    }
}

fn require_reciprocal(
    rows: &[Row<f64>],
    rates: &BTreeMap<RawId, RawId>,
    row: &Row<f64>,
    term: &Term<f64>,
    role: SpatialBalanceRole,
) -> Result<(), Diagnostic> {
    let transpose = term
        .pairing
        .transpose()
        .ok_or_else(|| invalid("volume balance lacks an adjoint gradient/value pairing"))?;
    let trial = rates.get(&term.trial).copied().unwrap_or(term.trial);
    let Some(other) = rows.iter().find(|other| other.tested == trial) else {
        return Err(invalid("volume balance lacks a reciprocal row"));
    };
    let matches = other
        .terms
        .iter()
        .filter(|candidate| {
            let candidate_role = if candidate.derivative && !rates.contains_key(&candidate.trial) {
                SpatialBalanceRole::Storage
            } else if !candidate.derivative && rates.contains_key(&candidate.trial) {
                SpatialBalanceRole::StatePotential
            } else {
                SpatialBalanceRole::Dissipation
            };
            candidate_role == role
                && rates
                    .get(&candidate.trial)
                    .copied()
                    .unwrap_or(candidate.trial)
                    == row.tested
                && candidate.pairing == transpose
                && candidate.coefficient.same_coefficient(&term.coefficient)
        })
        .count();
    if matches != 1 {
        return Err(invalid("volume balance requires one exact reciprocal term"));
    }
    Ok(())
}
