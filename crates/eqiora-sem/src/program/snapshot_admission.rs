//! Snapshot selection and whole-program admission orchestration.

use std::collections::BTreeMap;

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, OntologyId};
use eqiora_geometry::CanonicalGeometryV1;
use eqiora_graph::Snapshot;
use eqiora_schema::Model;

use super::geometry_admission::{
    admit_entity_sets, admit_geometry_boundary_junctions, index_closed_bundle,
};
use super::spatial_domains::{
    declared_spatial_supports, resolve_cartesian_bounds, validate_domains, validate_fields,
    validate_geometry_support_uses,
};
use super::{
    KernelProgram, kernel_error, kernel_path, model_path, validate_activations,
    validate_closed_topology, validate_connections, validate_relations,
};
use crate::conserving::validate_scalar_physical_networks;

impl KernelProgram {
    /// Select a `ModelView` from an immutable snapshot and validate all
    /// cross-node invariants required for execution.
    ///
    /// Validation is diagnostic-accumulating: independent faults are returned
    /// together in deterministic graph order.
    ///
    /// # Errors
    /// Returns diagnostics when the view is absent, its semantic topology is
    /// not closed, a symbol is unresolved, dimensions conflict, activation or
    /// clock wiring is ambiguous, or a connection contract is invalid.
    pub fn from_snapshot(
        snapshot: &Snapshot,
        model: OntologyId<Model>,
    ) -> Result<Self, Vec<Diagnostic>> {
        Self::from_snapshot_impl(snapshot, model, None)
    }

    /// Select and validate a Model together with its exact closed canonical
    /// geometry bundle.
    ///
    /// Artifact order is irrelevant. Every distinct geometry digest named by
    /// the Model must be supplied exactly once, and no unreferenced artifact is
    /// accepted. The returned program retains only internally derived spatial
    /// support facts, never artifact bytes or borrowed references.
    ///
    /// # Errors
    /// Returns diagnostics for the ordinary model invariants, an inexact
    /// artifact bundle, invalid entity-set selections, or unsupported geometry
    /// use.
    pub fn from_snapshot_with_geometry(
        snapshot: &Snapshot,
        model: OntologyId<Model>,
        geometry: &[&CanonicalGeometryV1],
    ) -> Result<Self, Vec<Diagnostic>> {
        Self::from_snapshot_impl(snapshot, model, Some(geometry))
    }

    fn from_snapshot_impl(
        snapshot: &Snapshot,
        model: OntologyId<Model>,
        geometry: Option<&[&CanonicalGeometryV1]>,
    ) -> Result<Self, Vec<Diagnostic>> {
        let raw_model = model.erase();
        let Some(view) = snapshot.ontology_view(&raw_model) else {
            return Err(vec![
                Diagnostic::error(
                    codes::ONTOLOGY_VIEW_NOT_FOUND,
                    format!("ModelView {raw_model} does not exist at the selected revision"),
                )
                .with_graph_path(model_path(model)),
            ]);
        };

        let mut diagnostics = Vec::new();
        let mut nodes = BTreeMap::new();
        let mut values = BTreeMap::new();
        for &member in view.members() {
            match snapshot.node(member) {
                Some(node) => match node.kernel_definition() {
                    Some(definition) => {
                        nodes.insert(member, definition.clone());
                        if let Some(value) = node.value() {
                            if !matches!(
                                definition,
                                eqiora_schema::kernel::KernelNode::Parameter(_)
                            ) {
                                diagnostics.push(kernel_error(member, "only Parameter nodes own revision-local values; use initial Relations for Fields"));
                            }
                            values.insert(member, value.clone());
                        }
                    }
                    None => diagnostics.push(kernel_error(
                        member,
                        "ModelView member has no complete Semantic Kernel definition",
                    )),
                },
                None => diagnostics.push(
                    Diagnostic::error(
                        codes::NODE_NOT_FOUND,
                        format!("ModelView member {member} is absent from the snapshot"),
                    )
                    .with_graph_path(kernel_path(member)),
                ),
            }
        }

        let edges = snapshot
            .edges()
            .filter(|edge| {
                view.members().contains(&edge.from()) && view.members().contains(&edge.to())
            })
            .copied()
            .collect::<Vec<_>>();

        validate_closed_topology(snapshot, view.members(), &mut diagnostics);
        super::nominal_values::validate(&nodes, &mut diagnostics);
        super::record_admission::validate(&nodes, &edges, &mut diagnostics);
        let cartesian_bounds = resolve_cartesian_bounds(&nodes, &values, &edges, &mut diagnostics);
        let invalid_domains = validate_domains(&nodes, &edges, &cartesian_bounds, &mut diagnostics);
        let mut spatial_supports = declared_spatial_supports(&nodes, &edges, &cartesian_bounds);
        let artifacts_admitted = geometry.is_some();
        let mut geometry_boundary_embeddings = BTreeMap::new();
        let mut geometry_boundary_junctions = BTreeMap::new();
        let mut admitted_geometry_ports = std::collections::BTreeSet::new();
        if let Some(geometry) = geometry {
            let artifacts = match index_closed_bundle(&nodes, geometry) {
                Ok(artifacts) => artifacts,
                Err(bundle_faults) => {
                    diagnostics.extend(bundle_faults);
                    return Err(diagnostics);
                }
            };
            let admission = admit_entity_sets(&nodes, &edges, &invalid_domains, &artifacts);
            diagnostics.extend(admission.diagnostics);
            spatial_supports.extend(admission.supports);
            geometry_boundary_embeddings = admission.boundary_embeddings;
            let (junctions, ports, junction_diagnostics) = admit_geometry_boundary_junctions(
                &nodes,
                &edges,
                &artifacts,
                &geometry_boundary_embeddings,
            );
            geometry_boundary_junctions = junctions;
            admitted_geometry_ports = ports;
            diagnostics.extend(junction_diagnostics);
        }
        validate_geometry_support_uses(
            &nodes,
            &edges,
            artifacts_admitted,
            &admitted_geometry_ports,
            &mut diagnostics,
        );
        validate_fields(&nodes, &edges, &spatial_supports, &mut diagnostics);
        validate_relations(&nodes, &edges, &spatial_supports, &mut diagnostics);
        super::observables::validate(&nodes, &edges, &spatial_supports, &mut diagnostics);
        super::signal_activation::validate(&nodes, &edges, &mut diagnostics);
        validate_activations(&nodes, &edges, &spatial_supports, &mut diagnostics);
        validate_connections(
            &nodes,
            &edges,
            view.boundary(),
            &cartesian_bounds,
            &geometry_boundary_junctions,
            &geometry_boundary_embeddings,
            &mut diagnostics,
        );
        crate::boundary_physical::validate_networks(
            &nodes,
            &edges,
            &cartesian_bounds,
            &geometry_boundary_embeddings,
            &mut diagnostics,
        );
        validate_scalar_physical_networks(&nodes, &edges, view.boundary(), &mut diagnostics);

        if diagnostics.is_empty() {
            Ok(Self {
                revision: snapshot.revision(),
                model,
                nodes,
                values,
                edges,
                boundary: view.boundary().clone(),
                spatial_supports,
                cartesian_bounds,
                geometry_boundary_junctions,
            })
        } else {
            Err(diagnostics)
        }
    }
}
