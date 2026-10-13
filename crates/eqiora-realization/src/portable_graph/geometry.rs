//! Compose moving geometry with an already resolved spatial graph.

use std::collections::BTreeSet;

use super::*;
use crate::{BackwardEulerRelationStep, P1HarmonicMeshMotionPolicy};

impl PortableRealizationGraph {
    /// Add harmonic coordinate extension and explicit endpoint GCL pullbacks.
    ///
    /// `current_domains` selects current coordinates independently of harmonic
    /// or driven membership. Other selected Domains use reference coordinates.
    /// `pullbacks` selects the exact transported Relation/Field pairs and their
    /// shared Backward Euler duration. The graph owns their derivative and GCL
    /// transformations; no physical-family name selects a numerical action.
    ///
    /// This checks graph structure. Equation lowering and mesh binding still
    /// authenticate each selected Field, transport law and conforming trace.
    ///
    /// # Errors
    /// Returns `EQ0807` for absent or duplicate selections, conflicting geometry,
    /// inconsistent durations or derivatives, or invalid graph closure.
    pub fn with_harmonic_geometry(
        mut self,
        motion: &P1HarmonicMeshMotionPolicy,
        current_domains: impl IntoIterator<Item = Id<kinds::Domain>>,
        pullbacks: impl IntoIterator<Item = BackwardEulerRelationStep>,
    ) -> Result<Self, Diagnostic> {
        let domain_reference = |id| {
            self.domains
                .iter()
                .position(|domain| domain.domain == id)
                .map(DomainDiscretizationId::new)
                .ok_or_else(|| invalid_realization("harmonic graph selection has an absent Domain"))
        };
        let mut harmonic_domains = motion
            .harmonic_domains()
            .iter()
            .map(|id| domain_reference(*id))
            .collect::<Result<Vec<_>, _>>()?;
        harmonic_domains.sort();
        let mut drivers = Vec::new();
        let mut selected = harmonic_domains.iter().copied().collect::<BTreeSet<_>>();
        for &(domain, field) in motion.drivers() {
            let domain = domain_reference(domain)?;
            let driver = self
                .fields
                .iter()
                .position(|node| node.field == field && node.domain == domain)
                .map(FieldRepresentationId::new)
                .ok_or_else(|| {
                    invalid_realization(
                        "harmonic graph driver has an absent or mismatched Field/Domain",
                    )
                })?;
            drivers.push(driver);
            selected.insert(domain);
        }
        drivers.sort();
        let mut current = BTreeSet::new();
        for domain in current_domains {
            let domain = domain_reference(domain)?;
            if !selected.contains(&domain) || !current.insert(domain) {
                return Err(invalid_realization(
                    "current geometry Domains must be unique members of the motion policy",
                ));
            }
        }
        let mut pullbacks = pullbacks.into_iter().collect::<Vec<_>>();
        pullbacks.sort_by_key(|step| step.state().ulid());
        let duration = pullbacks
            .first()
            .ok_or_else(|| invalid_realization("harmonic graph action requires a GCL pullback"))?
            .duration();
        if pullbacks.iter().any(|step| step.duration() != duration)
            || pullbacks
                .windows(2)
                .any(|pair| pair[0].state() == pair[1].state())
        {
            return Err(invalid_realization(
                "harmonic graph pullbacks require unique Fields and one shared duration",
            ));
        }
        let action = GeometryActionId::new(self.geometry_actions.len());
        for domain in selected {
            let node = &mut self.domains[domain.index()];
            if matches!(
                node.configuration,
                DomainConfiguration::CurrentAleGeometry { .. }
            ) {
                return Err(invalid_realization(
                    "harmonic graph Domain already has a current geometry action",
                ));
            }
            node.configuration = if current.contains(&domain) {
                DomainConfiguration::CurrentAleGeometry { action }
            } else {
                DomainConfiguration::ReferenceConfiguration
            };
        }
        for step in pullbacks {
            let state = self
                .fields
                .iter()
                .position(|field| field.field == step.state())
                .map(FieldRepresentationId::new)
                .ok_or_else(|| {
                    invalid_realization("harmonic graph pullback selects an absent Field")
                })?;
            let derivative = TransformationNode::BackwardEulerDerivative {
                relation: step.relation(),
                state,
                duration,
            };
            let existing = self.transformations.iter().filter(|node| matches!(node,
                TransformationNode::BackwardEulerDerivative { relation, state: candidate, .. }
                        if *candidate == state && *relation == step.relation()
            )).collect::<Vec<_>>();
            match existing.as_slice() {
                [] => self.transformations.push(derivative),
                [existing] if **existing == derivative => {}
                _ => {
                    return Err(invalid_realization(
                        "harmonic graph pullback conflicts with its existing derivative",
                    ));
                }
            }
            self.transformations
                .push(TransformationNode::GclCompatibleAlePullback {
                    relation: step.relation(),
                    velocity: state,
                    geometry: action,
                });
        }
        self.geometry_actions
            .push(GeometryActionNode::P1HarmonicExtension {
                harmonic_domains,
                drivers,
                interfaces: motion.interfaces().to_vec(),
                duration,
                quality_gate: motion.quality_gate(),
                solver: motion.solver(),
            });
        self.systems[0].transformations = (0..self.transformations.len())
            .map(TransformationId::new)
            .collect();
        self.validate()?;
        Ok(self)
    }
}
