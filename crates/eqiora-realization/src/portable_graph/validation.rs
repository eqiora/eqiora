//! Structural closure checks for one portable realization graph.

use std::collections::BTreeSet;

use super::*;

mod geometry;
pub(super) use geometry::validate_geometry_actions;

pub(super) fn validate_system(
    graph: &PortableRealizationGraph,
    system: &AlgebraicSystemNode,
) -> Result<(), Diagnostic> {
    if system.blocks.is_empty() {
        return Err(invalid_realization(
            "portable algebraic system requires at least one block",
        ));
    }
    let mut field_blocks = BTreeSet::new();
    let mut constraints = BTreeSet::new();
    let mut actual_scales = Vec::with_capacity(system.blocks.len());
    for block in &system.blocks {
        match *block {
            SystemBlock::Field(id) => {
                let Some(field) = graph.field(id) else {
                    return Err(invalid_realization(
                        "portable algebraic block references an absent Field node",
                    ));
                };
                if !field_blocks.insert(field.field().ulid()) {
                    return Err(invalid_realization(
                        "portable algebraic system contains a duplicate Field block",
                    ));
                }
                actual_scales.push(AlgebraicBlock::Field(field.field()));
            }
            SystemBlock::ConstraintMultiplier(constraint) => {
                if !constraints.insert(constraint.field().ulid()) {
                    return Err(invalid_realization(
                        "portable algebraic system contains a duplicate constraint multiplier",
                    ));
                }
                if !graph
                    .fields
                    .iter()
                    .any(|field| field.field == constraint.field())
                {
                    return Err(invalid_realization(
                        "portable constraint multiplier refers to an absent Field node",
                    ));
                }
                actual_scales.push(AlgebraicBlock::ConstraintMultiplier {
                    field: constraint.field(),
                });
            }
        }
    }
    if let SystemScaling::SymmetricCongruence(scaling) = &system.scaling {
        let expected_scales = scaling
            .block_scales()
            .iter()
            .map(|entry| entry.block())
            .collect::<Vec<_>>();
        if actual_scales != expected_scales {
            return Err(invalid_realization(
                "portable algebraic blocks and congruence scales must have exact equal coverage and order",
            ));
        }
    }
    let mut seen_transformations = BTreeSet::new();
    for id in &system.transformations {
        if graph.transformation(*id).is_none() || !seen_transformations.insert(*id) {
            return Err(invalid_realization(
                "portable algebraic system has an absent or duplicate transformation",
            ));
        }
    }
    if seen_transformations.len() != graph.transformations.len() {
        return Err(invalid_realization(
            "portable Realization graph contains an unreachable transformation",
        ));
    }
    let mut represented = BTreeSet::new();
    let mut used_geometry_actions = BTreeSet::new();
    let mut gcl_fields = BTreeSet::new();
    for block in &system.blocks {
        if let SystemBlock::Field(id) = *block {
            represented.insert(id);
        }
    }
    for transformation in &graph.transformations {
        match *transformation {
            TransformationNode::BackwardEulerDerivative { state, .. } => {
                if graph.field(state).is_none() {
                    return Err(invalid_realization(
                        "portable Backward Euler transformation references an absent Field node",
                    ));
                }
                represented.insert(state);
            }
            TransformationNode::EnergySkewConvection { velocity, .. } => {
                if graph.field(velocity).is_none() {
                    return Err(invalid_realization(
                        "portable energy-skew transformation references an absent Field node",
                    ));
                }
                represented.insert(velocity);
            }
            TransformationNode::CellCenteredConvection { state, .. } => {
                if graph.field(state).is_none() {
                    return Err(invalid_realization(
                        "portable cell-centered convection transformation references an absent Field node",
                    ));
                }
                represented.insert(state);
            }
            TransformationNode::OrthogonalTwoPointDiffusion { state, .. } => {
                if graph.field(state).is_none() {
                    return Err(invalid_realization(
                        "portable orthogonal two-point diffusion references an absent Field node",
                    ));
                }
                represented.insert(state);
            }
            TransformationNode::ImplicitCenteredMomentumConvection { velocity, .. } => {
                if graph.field(velocity).is_none() {
                    return Err(invalid_realization(
                        "portable centered momentum convection references an absent velocity Field node",
                    ));
                }
                represented.insert(velocity);
            }
            TransformationNode::CartesianCentralNewtonianTraction {
                velocity, pressure, ..
            }
            | TransformationNode::MomentumWeightedLinearExactCoupling {
                velocity, pressure, ..
            } => {
                if graph.field(velocity).is_none() || graph.field(pressure).is_none() {
                    return Err(invalid_realization(
                        "portable collocated fluid transformation references an absent velocity or pressure Field node",
                    ));
                }
                if velocity == pressure {
                    return Err(invalid_realization(
                        "portable collocated fluid transformation requires distinct velocity and pressure Field nodes",
                    ));
                }
                represented.insert(velocity);
                represented.insert(pressure);
            }
            TransformationNode::GclCompatibleAlePullback {
                relation,
                velocity,
                geometry,
            } => {
                let (Some(velocity_field), Some(action)) =
                    (graph.field(velocity), graph.geometry_action(geometry))
                else {
                    return Err(invalid_realization(
                        "portable GCL-compatible ALE pullback references an absent Field or Geometry Action",
                    ));
                };
                if !gcl_fields.insert(velocity) {
                    return Err(invalid_realization(
                        "a portable Field must have at most one GCL-compatible ALE pullback",
                    ));
                }
                used_geometry_actions.insert(geometry);
                let GeometryActionNode::P1HarmonicExtension { duration, .. } = action;
                if !matches!(
                    graph.domain(velocity_field.domain).map(|domain| domain.configuration),
                    Some(DomainConfiguration::CurrentAleGeometry { action }) if action == geometry
                ) {
                    return Err(invalid_realization(
                        "portable GCL-compatible ALE Field requires the selected current Domain geometry",
                    ));
                }
                if !graph.transformations.iter().any(|candidate| {
                    matches!(
                        candidate,
                        TransformationNode::BackwardEulerDerivative {
                            relation: candidate_relation,
                            state,
                            duration: candidate_duration,
                        } if *candidate_relation == relation
                            && *state == velocity
                            && candidate_duration == duration
                    )
                }) {
                    return Err(invalid_realization(
                        "portable GCL-compatible ALE pullback must share its Relation, velocity, and duration with Backward Euler",
                    ));
                }
                represented.insert(velocity);
            }
            TransformationNode::BackwardEulerElimination { state, rate, .. } => {
                let (Some(state_field), Some(rate_field)) = (graph.field(state), graph.field(rate))
                else {
                    return Err(invalid_realization(
                        "portable Backward Euler elimination references an absent Field node",
                    ));
                };
                if state == rate || state_field.domain != rate_field.domain {
                    return Err(invalid_realization(
                        "portable Backward Euler state and rate must be distinct Fields on one Domain",
                    ));
                }
                if represented.contains(&state) || !represented.contains(&rate) {
                    return Err(invalid_realization(
                        "portable Backward Euler elimination requires a non-algebraic state and algebraic rate",
                    ));
                }
                represented.insert(state);
                represented.insert(rate);
            }
            TransformationNode::ConformingTraceQuotient { endpoints, .. } => {
                let [Some(first), Some(second)] =
                    [graph.field(endpoints[0]), graph.field(endpoints[1])]
                else {
                    return Err(invalid_realization(
                        "portable trace quotient references an absent Field node",
                    ));
                };
                if first.domain == second.domain {
                    return Err(invalid_realization(
                        "portable conforming trace quotient must join distinct Domains",
                    ));
                }
                represented.extend(endpoints);
            }
        }
    }
    if used_geometry_actions.len() != graph.geometry_actions.len() {
        return Err(invalid_realization(
            "portable Realization graph contains an unreachable Geometry Action",
        ));
    }
    if represented.len() != graph.fields.len() {
        return Err(invalid_realization(
            "portable Realization graph contains an unreachable Field representation",
        ));
    }
    let represented_domains = graph
        .fields
        .iter()
        .map(|field| field.domain)
        .collect::<BTreeSet<_>>();
    if represented_domains.len() != graph.domains.len() {
        return Err(invalid_realization(
            "portable Realization graph contains an unreachable Domain discretization",
        ));
    }
    Ok(())
}

pub(super) fn strictly_sorted_unique_by<T, K: Ord>(values: &[T], key: impl Fn(&T) -> K) -> bool {
    values.windows(2).all(|pair| key(&pair[0]) < key(&pair[1]))
}

/// Moment coefficient meaning is checked on construction and canonical decode.
pub(super) fn validate_moment_spaces(graph: &PortableRealizationGraph) -> Result<(), Diagnostic> {
    let mut moments = false;
    for field in &graph.fields {
        if matches!(
            field.space.family(),
            crate::SpaceFamily::TetrahedralEdge | crate::SpaceFamily::TetrahedralFace
        ) {
            moments = true;
            graph
                .domain(field.domain)
                .expect("validated Domain reference")
                .discretization
                .validate_space(field.space)?;
        }
    }
    if moments
        && (!graph.transformations.is_empty()
            || !graph.geometry_actions.is_empty()
            || !graph.nonlinear_solves.is_empty()
            || graph.systems.iter().any(|system| {
                system.scaling != SystemScaling::Dimensional
                    || system
                        .blocks
                        .iter()
                        .any(|block| !matches!(block, SystemBlock::Field(_)))
            }))
    {
        return Err(invalid_realization(
            "tetrahedral moments require a static dimensional linear graph without unadmitted transformations or constraints",
        ));
    }
    Ok(())
}
