//! Geometry inventory and numerical-configuration closure.

use std::collections::{BTreeMap, BTreeSet};

use super::super::*;
use crate::{ConformingTraceSource, P1HarmonicMeshMotionPolicy};

pub(in super::super) fn validate_geometry_actions(
    graph: &PortableRealizationGraph,
) -> Result<(), Diagnostic> {
    let mut current_domains = vec![BTreeSet::new(); graph.geometry_actions.len()];
    for (index, domain) in graph.domains.iter().enumerate() {
        if let DomainConfiguration::CurrentAleGeometry { action } = domain.configuration {
            let Some(domains) = current_domains.get_mut(action.index()) else {
                return Err(invalid_realization(
                    "portable current ALE geometry references an absent Geometry Action",
                ));
            };
            domains.insert(DomainDiscretizationId::new(index));
        }
    }
    for (index, action) in graph.geometry_actions.iter().enumerate() {
        let GeometryActionNode::P1HarmonicExtension {
            harmonic_domains,
            drivers,
            interfaces,
            duration,
            quality_gate,
            solver,
        } = action;
        if !strictly_sorted_unique_by(harmonic_domains, |id| *id)
            || !strictly_sorted_unique_by(drivers, |id| *id)
            || !strictly_sorted_unique_by(interfaces, |id| id.ulid())
        {
            return Err(invalid_realization(
                "portable Geometry Action inventories must be strictly ordered and unique",
            ));
        }
        let harmonic = harmonic_domains
            .iter()
            .map(|id| {
                graph
                    .domain(*id)
                    .map(|domain| domain.domain)
                    .ok_or_else(|| {
                        invalid_realization(
                            "portable Geometry Action references an absent harmonic Domain",
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let driver_fields = drivers
            .iter()
            .map(|id| {
                graph.field(*id).ok_or_else(|| {
                    invalid_realization(
                        "portable Geometry Action references an absent driver Field",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        // The semantic policy owns nonempty/disjoint groups, unique drivers and SPD admission.
        P1HarmonicMeshMotionPolicy::new(
            harmonic,
            driver_fields
                .iter()
                .map(|field| (graph.domains[field.domain.index()].domain, field.field)),
            interfaces.iter().copied(),
            *quality_gate,
            *solver,
        )?;
        let driven = driver_fields
            .iter()
            .map(|field| field.domain)
            .collect::<BTreeSet<_>>();
        let harmonic = harmonic_domains.iter().copied().collect::<BTreeSet<_>>();
        let selected = harmonic.union(&driven).copied().collect::<BTreeSet<_>>();
        if current_domains[index].is_empty() || !current_domains[index].is_subset(&selected) {
            return Err(invalid_realization(
                "portable Geometry Action must supply its selected current ALE Domains",
            ));
        }
        for domain in &selected {
            match graph.domains[domain.index()].configuration {
                DomainConfiguration::ReferenceConfiguration => {}
                DomainConfiguration::CurrentAleGeometry { action }
                    if action == GeometryActionId::new(index) => {}
                _ => {
                    return Err(invalid_realization(
                        "portable Geometry Action Domain requires reference or matching current configuration",
                    ));
                }
            }
        }
        crate::BackwardEulerRelationStep::validate_duration(*duration)?;
        for transformation in &graph.transformations {
            if let TransformationNode::BackwardEulerElimination {
                state,
                duration: step,
                ..
            } = transformation
                && drivers.contains(state)
                && step != duration
            {
                return Err(invalid_realization(
                    "portable Geometry Action driver elimination must share its duration",
                ));
            }
        }
        let mut connections = BTreeMap::new();
        let mut crossing = BTreeSet::new();
        for transformation in &graph.transformations {
            let TransformationNode::ConformingTraceQuotient { source, endpoints } = transformation
            else {
                continue;
            };
            let [Some(first), Some(second)] = endpoints.map(|id| graph.field(id)) else {
                return Err(invalid_realization(
                    "portable trace quotient references an absent Field node",
                ));
            };
            let mut domains = [first.domain, second.domain];
            if !domains.iter().any(|id| selected.contains(id)) {
                continue;
            }
            domains.sort();
            if let ConformingTraceSource::ConservingConnection(connection) = source {
                if connections
                    .insert(connection.erase(), domains)
                    .is_some_and(|previous| previous != domains)
                {
                    return Err(invalid_realization(
                        "portable Geometry Action Connection has inconsistent endpoint Domains",
                    ));
                }
                if (harmonic.contains(&domains[0]) && driven.contains(&domains[1]))
                    || (driven.contains(&domains[0]) && harmonic.contains(&domains[1]))
                {
                    crossing.insert(connection.erase());
                }
            }
        }
        if crossing != interfaces.iter().map(|id| id.erase()).collect() {
            return Err(invalid_realization(
                "portable Geometry Action interfaces must exactly cover cross-group Connections",
            ));
        }
    }
    Ok(())
}
