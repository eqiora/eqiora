//! Replay the graph-selected harmonic coordinate action from exact driver leaves.
use crate::{
    DiscreteFieldEnvelopeV1, FieldSnapshotEnvelopeV2, ReplayableCanonicalModelArtifact,
    ValidatedMovingSpatialContext, invalid_artifact,
};
use eqiora_core::Diagnostic;
use eqiora_meshing::{CellId, DiscreteFieldAssociation, FacetId, P1HarmonicCoordinateRelation};

pub(super) fn validate_coordinate_derivation<
    const D: usize,
    M: ReplayableCanonicalModelArtifact,
>(
    context: &ValidatedMovingSpatialContext<'_, M>,
    driver_snapshot: &FieldSnapshotEnvelopeV2,
    driver_snapshot_blocks: &[&DiscreteFieldEnvelopeV1],
    current_coordinates_m: &[Vec<f64>],
) -> Result<(), Diagnostic> {
    let graph = context.realization().graph();
    let [
        eqiora_realization::GeometryActionNode::P1HarmonicExtension {
            harmonic_domains,
            drivers,
            interfaces,
            solver,
            ..
        },
    ] = graph.geometry_actions()
    else {
        return Err(invalid_artifact(
            "geometry-state requires one exact harmonic action",
        ));
    };
    let ([fluid], [driver]) = (harmonic_domains.as_slice(), drivers.as_slice()) else {
        return Err(invalid_artifact(
            "geometry-state requires one harmonic Domain and one driven Domain",
        ));
    };
    let harmonic_domain = graph
        .domain(*fluid)
        .ok_or_else(|| invalid_artifact("harmonic Domain is absent"))?
        .domain();
    let driver = graph
        .field(*driver)
        .ok_or_else(|| invalid_artifact("geometry driver is absent"))?;
    let driven_domain = graph
        .domain(driver.domain())
        .ok_or_else(|| invalid_artifact("driver Domain is absent"))?
        .domain();
    if driver_snapshot.field() != context.driver()
        || driver_snapshot.support_domain() != driven_domain
    {
        return Err(invalid_artifact(
            "geometry-state driver or material roles differ from the exact Realization graph",
        ));
    }

    let vertex_block = driver_snapshot_blocks
        .iter()
        .find(|block| block.association() == DiscreteFieldAssociation::Vertex)
        .ok_or_else(|| {
            invalid_artifact(
                "geometry-state driver snapshot has no complete vertex coefficient block",
            )
        })?;
    if driver_snapshot_blocks.len() != 1
        || vertex_block
            .component_shape()
            .component_count()
            .map_err(|error| invalid_artifact(error.message()))?
            != D
    {
        return Err(invalid_artifact(
            "geometry-state driver must be one P1 spatial-vector vertex block",
        ));
    }
    let driver_values = vertex_block.values().as_chunks::<D>().0;

    let [connection] = interfaces.as_slice() else {
        return Err(invalid_artifact(
            "geometry-state requires one exact motion interface",
        ));
    };
    let interface = context.correspondence().derive_conserving_interface(
        context.geometry(),
        context.model(),
        context.mesh(),
        *connection,
    )?;
    let harmonic_cells = context
        .correspondence()
        .body_cells(harmonic_domain)
        .ok_or_else(|| invalid_artifact("ALE fluid Domain has no exact mesh-cell inventory"))?
        .into_iter()
        .map(CellId::new)
        .collect();
    let driven_cells = context
        .correspondence()
        .body_cells(driven_domain)
        .ok_or_else(|| invalid_artifact("ALE solid Domain has no exact mesh-cell inventory"))?
        .into_iter()
        .map(CellId::new)
        .collect();
    let interface_facets = interface
        .facet_indices()
        .iter()
        .copied()
        .map(FacetId::new)
        .collect();
    let relation = P1HarmonicCoordinateRelation::<D>::new(
        context.mesh().mesh(),
        harmonic_cells,
        driven_cells,
        interface_facets,
    )
    .map_err(|error| invalid_artifact(error.message()))?;
    let driver_residual_targets = relation
        .driver_rhs_norms()
        .iter()
        .map(|&right_hand_side_norm| {
            solver
                .residual_target(right_hand_side_norm)
                .map_err(|error| invalid_artifact(error.message()))
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    relation
        .validate_current_coordinates(
            driver_values,
            current_coordinates_m,
            &driver_residual_targets,
        )
        .map_err(|error| invalid_artifact(error.message()))
}
