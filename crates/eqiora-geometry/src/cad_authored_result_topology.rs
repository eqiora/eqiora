//! Immutable result topology projected from accepted authored-CAD lineage.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use crate::cad_authored_selection::FaceKey;
use crate::{
    CircularHoleGeometry, EDGE_DIMENSION, FACE_DIMENSION, GeometryBuildReceipt, GeometryFaceHandle,
    GeometrySolidOperation, NamedEntitySet,
};

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_ARTIFACT, message)
}

/// Opaque one-dimensional handle owned by one exact accepted result topology.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct CadAuthoredResultEdgeHandle {
    owner_graph_digest: [u8; 32],
    member: usize,
    source: FaceKey,
}

impl CadAuthoredResultEdgeHandle {
    /// Exact authored-graph identity that owns this result handle.
    #[must_use]
    pub const fn owner_graph_digest_bytes(&self) -> [u8; 32] {
        self.owner_graph_digest
    }

    /// Construction-lineage provenance from which this result edge was projected.
    #[must_use]
    pub const fn source_provenance_key(&self) -> &'static str {
        self.source.provenance_key()
    }
}

/// Opaque two-dimensional handle owned by one exact accepted result topology.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct CadAuthoredResultFaceHandle {
    owner_graph_digest: [u8; 32],
    member: usize,
    source: FaceKey,
}

impl CadAuthoredResultFaceHandle {
    /// Exact authored-graph identity that owns this result handle.
    #[must_use]
    pub const fn owner_graph_digest_bytes(&self) -> [u8; 32] {
        self.owner_graph_digest
    }

    /// Construction-lineage provenance from which this result face was projected.
    #[must_use]
    pub const fn source_provenance_key(&self) -> &'static str {
        self.source.provenance_key()
    }
}

/// Dimension-carrying handle in one admitted planar result topology.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CadAuthoredResultTopologyHandle {
    /// One exact boundary edge.
    Edge(CadAuthoredResultEdgeHandle),
    /// The exact rectangle-minus-circle region face.
    Face(CadAuthoredResultFaceHandle),
}

impl CadAuthoredResultTopologyHandle {
    /// Topological dimension retained by the handle type.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        match self {
            Self::Edge(_) => EDGE_DIMENSION,
            Self::Face(_) => FACE_DIMENSION,
        }
    }

    /// Exact authored-graph identity that owns this result handle.
    #[must_use]
    pub const fn owner_graph_digest_bytes(&self) -> [u8; 32] {
        match self {
            Self::Edge(handle) => handle.owner_graph_digest_bytes(),
            Self::Face(handle) => handle.owner_graph_digest_bytes(),
        }
    }

    /// Construction-lineage provenance from which this result member was projected.
    #[must_use]
    pub const fn source_provenance_key(&self) -> &'static str {
        match self {
            Self::Edge(handle) => handle.source_provenance_key(),
            Self::Face(handle) => handle.source_provenance_key(),
        }
    }

    const fn member(&self) -> usize {
        match self {
            Self::Edge(handle) => handle.member,
            Self::Face(handle) => handle.member,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LineageProjection {
    retained_unchanged: Vec<FaceKey>,
    retained_modified: Vec<FaceKey>,
    created: Vec<FaceKey>,
    deleted: Vec<FaceKey>,
    split: Vec<FaceKey>,
    merged: Vec<FaceKey>,
}

/// Immutable planar result topology owned by one admitted authored graph.
///
/// This is deliberately not a generic B-rep. It closes exactly the positive-z
/// transverse rectangle-minus-circle result needed by the accepted construction
/// graph and retains no provider-local indices or coordinate classifier.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CadAuthoredResultTopology {
    owner_graph_digest: [u8; 32],
    bounds: [[f64; 2]; 2],
    circle_center: [f64; 2],
    circle_radius_m: f64,
}

impl CadAuthoredResultTopology {
    pub(crate) fn from_build(
        graph: &GeometrySolidOperation,
        build: &GeometryBuildReceipt,
    ) -> Result<Self, Diagnostic> {
        let lineage = LineageProjection {
            retained_unchanged: keys(build.retained_unchanged()),
            retained_modified: keys(build.retained_modified()),
            created: keys(build.created()),
            deleted: keys(build.deleted()),
            split: keys(build.split()),
            merged: keys(build.merged()),
        };
        Self::admit(graph, build.graph_digest_bytes(), lineage)
    }

    fn admit(
        graph: &GeometrySolidOperation,
        build_graph_digest: [u8; 32],
        lineage: LineageProjection,
    ) -> Result<Self, Diagnostic> {
        if build_graph_digest != graph.digest_bytes() {
            return Err(invalid(
                "result topology build belongs to a foreign authored graph identity",
            ));
        }
        let Some((bounds, circle_center, circle_radius_m)) = graph.planar_cut_parts() else {
            return Err(invalid(
                "result topology requires the admitted circular through-cut graph",
            ));
        };
        let expected = LineageProjection {
            retained_unchanged: vec![
                FaceKey::profile_x_lower(),
                FaceKey::profile_x_upper(),
                FaceKey::profile_y_lower(),
                FaceKey::profile_y_upper(),
            ],
            retained_modified: vec![FaceKey::start_cap(), FaceKey::end_cap()],
            created: vec![FaceKey::cut_wall()],
            deleted: Vec::new(),
            split: Vec::new(),
            merged: Vec::new(),
        };
        if lineage != expected {
            return Err(invalid(
                "result topology lineage is incomplete, mutated, deleted, split, merged, or ambiguous",
            ));
        }
        Ok(Self {
            owner_graph_digest: graph.digest_bytes(),
            bounds,
            circle_center,
            circle_radius_m,
        })
    }

    /// Exact authored-graph identity owning this topology.
    #[must_use]
    pub const fn owner_graph_digest_bytes(&self) -> [u8; 32] {
        self.owner_graph_digest
    }

    /// Project one admitted source-owned construction handle into result topology.
    ///
    /// # Errors
    /// Returns `EQ0901` for a foreign, stale, deleted, or non-section handle.
    pub fn project(
        &self,
        source: &GeometryFaceHandle,
    ) -> Result<CadAuthoredResultTopologyHandle, Diagnostic> {
        if source.graph_digest_bytes() != self.owner_graph_digest || source.is_v1() {
            return Err(invalid(
                "construction handle is foreign or stale for this result topology",
            ));
        }
        let source_key = source.face_key();
        let projected = if source_key == FaceKey::end_cap() {
            CadAuthoredResultTopologyHandle::Face(CadAuthoredResultFaceHandle {
                owner_graph_digest: self.owner_graph_digest,
                member: 0,
                source: source_key,
            })
        } else if let Some(member) = edge_member(source_key) {
            CadAuthoredResultTopologyHandle::Edge(CadAuthoredResultEdgeHandle {
                owner_graph_digest: self.owner_graph_digest,
                member,
                source: source_key,
            })
        } else {
            return Err(invalid(
                "construction handle does not survive in the positive-z planar result topology",
            ));
        };
        Ok(projected)
    }

    /// Atomically bind complete result membership into canonical Geometry v2.
    ///
    /// Every group must be nonempty and dimension-homogeneous. Every result
    /// member must be owned by this topology and occur exactly once across the
    /// complete mapping. No coordinates, proximity, provider IDs, mesh labels,
    /// or classification tolerance participate.
    ///
    /// # Errors
    /// Returns `EQ0901` for foreign, incomplete, duplicate, empty, or
    /// mixed-dimensional membership, or invalid semantic names.
    pub(crate) fn canonical_geometry(
        &self,
        named_topology: &BTreeMap<String, Vec<CadAuthoredResultTopologyHandle>>,
    ) -> Result<CircularHoleGeometry, Diagnostic> {
        let mut covered = BTreeSet::new();
        let mut entity_sets = Vec::with_capacity(named_topology.len());
        for (name, handles) in named_topology {
            let Some(first) = handles.first() else {
                return Err(invalid("named result-topology group must not be empty"));
            };
            let dimension = first.dimension();
            let mut members = Vec::with_capacity(handles.len());
            for handle in handles {
                if handle.owner_graph_digest_bytes() != self.owner_graph_digest {
                    return Err(invalid(
                        "named result-topology handle belongs to a foreign owner identity",
                    ));
                }
                if handle.dimension() != dimension {
                    return Err(invalid(
                        "one result-topology name cannot group mixed dimensions",
                    ));
                }
                let identity = (handle.dimension(), handle.member());
                if !covered.insert(identity) {
                    return Err(invalid(
                        "result-topology membership must be named exactly once",
                    ));
                }
                members.push(handle.member());
            }
            entity_sets.push(NamedEntitySet::new(name, dimension, members));
        }
        let expected = BTreeSet::from([
            (EDGE_DIMENSION, 0),
            (EDGE_DIMENSION, 1),
            (EDGE_DIMENSION, 2),
            (EDGE_DIMENSION, 3),
            (EDGE_DIMENSION, 4),
            (FACE_DIMENSION, 0),
        ]);
        if covered != expected {
            return Err(invalid(
                "named result topology must cover the complete planar result exactly once",
            ));
        }
        CircularHoleGeometry::new(
            self.bounds,
            self.circle_center,
            self.circle_radius_m,
            entity_sets,
        )
    }
}

fn keys(handles: &[GeometryFaceHandle]) -> Vec<FaceKey> {
    handles.iter().map(GeometryFaceHandle::face_key).collect()
}

fn edge_member(source: FaceKey) -> Option<usize> {
    if source == FaceKey::profile_x_lower() {
        Some(0)
    } else if source == FaceKey::profile_x_upper() {
        Some(1)
    } else if source == FaceKey::profile_y_lower() {
        Some(2)
    } else if source == FaceKey::profile_y_upper() {
        Some(3)
    } else if source == FaceKey::cut_wall() {
        Some(4)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GeometryGraph;

    fn scaled_graph(scale: f64) -> GeometrySolidOperation {
        let graph = GeometryGraph::new();
        let base = graph
            .rectangle_extrusion(
                (0.0, 2.2 * scale),
                (0.0, 0.41 * scale),
                0.0,
                scale,
                1.0e-10 * scale,
            )
            .unwrap();
        graph
            .circular_through_cut(
                &base,
                [0.2 * scale, 0.2 * scale],
                0.05 * scale,
                1.0e-10 * scale,
            )
            .unwrap()
    }

    fn named(
        graph: &GeometrySolidOperation,
        topology: &CadAuthoredResultTopology,
    ) -> BTreeMap<String, Vec<CadAuthoredResultTopologyHandle>> {
        let project = |name| topology.project(&graph.face_handle(name).unwrap()).unwrap();
        BTreeMap::from([
            ("fluid".to_owned(), vec![project("end-cap")]),
            ("inlet".to_owned(), vec![project("profile-x-lower")]),
            ("outlet".to_owned(), vec![project("profile-x-upper")]),
            (
                "walls".to_owned(),
                vec![project("profile-y-lower"), project("profile-y-upper")],
            ),
            ("cylinder".to_owned(), vec![project("cut-wall")]),
        ])
    }

    #[test]
    fn scale_family_projects_identical_typed_membership_and_replays() {
        let mut membership = None;
        for exponent in [-40, 0, 40] {
            let graph = scaled_graph(2.0_f64.powi(exponent));
            let build = GeometryBuildReceipt::from_graph(&graph).unwrap();
            let topology = CadAuthoredResultTopology::from_build(&graph, &build).unwrap();
            let named = named(&graph, &topology);
            let actual_membership = named
                .iter()
                .map(|(name, handles)| {
                    (
                        name.clone(),
                        handles
                            .iter()
                            .map(|handle| (handle.dimension(), handle.source_provenance_key()))
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            if let Some(expected) = &membership {
                assert_eq!(&actual_membership, expected);
            } else {
                membership = Some(actual_membership);
            }
            let geometry = topology.canonical_geometry(&named).unwrap();
            assert_eq!(geometry.bounds().len(), 2);
            assert_eq!(
                geometry
                    .entity_sets()
                    .iter()
                    .find(|set| set.name() == "cylinder")
                    .map(NamedEntitySet::dimension),
                Some(1)
            );
            assert_eq!(
                geometry
                    .entity_sets()
                    .iter()
                    .find(|set| set.name() == "fluid")
                    .map(NamedEntitySet::dimension),
                Some(2)
            );
            assert_eq!(
                CircularHoleGeometry::decode_canonical(
                    geometry.canonical_bytes(),
                    Default::default(),
                )
                .unwrap(),
                geometry
            );
        }
    }

    #[test]
    fn owner_dimension_and_complete_membership_fail_closed() {
        let graph = scaled_graph(1.0);
        let topology = CadAuthoredResultTopology::from_build(
            &graph,
            &GeometryBuildReceipt::from_graph(&graph).unwrap(),
        )
        .unwrap();
        assert!(
            topology
                .project(&graph.face_handle("start-cap").unwrap())
                .is_err()
        );

        let foreign_graph = scaled_graph(2.0);
        assert!(
            topology
                .project(&foreign_graph.face_handle("cut-wall").unwrap())
                .is_err()
        );

        let mut incomplete = named(&graph, &topology);
        incomplete.remove("cylinder");
        assert!(topology.canonical_geometry(&incomplete).is_err());

        let mut duplicate = named(&graph, &topology);
        let inlet = duplicate["inlet"][0].clone();
        duplicate.get_mut("walls").unwrap().push(inlet);
        assert!(topology.canonical_geometry(&duplicate).is_err());

        let mut mixed = named(&graph, &topology);
        let fluid = mixed["fluid"][0].clone();
        mixed.get_mut("walls").unwrap().push(fluid);
        assert!(topology.canonical_geometry(&mixed).is_err());

        let mut empty = named(&graph, &topology);
        empty.insert("empty".to_owned(), Vec::new());
        assert!(topology.canonical_geometry(&empty).is_err());
    }

    #[test]
    fn mutated_deleted_split_or_merged_lineage_rejects() {
        let graph = scaled_graph(1.0);
        let build = GeometryBuildReceipt::from_graph(&graph).unwrap();
        let baseline = LineageProjection {
            retained_unchanged: keys(build.retained_unchanged()),
            retained_modified: keys(build.retained_modified()),
            created: keys(build.created()),
            deleted: keys(build.deleted()),
            split: keys(build.split()),
            merged: keys(build.merged()),
        };
        for mutation in 0..5 {
            let mut lineage = baseline.clone();
            match mutation {
                0 => {
                    lineage.retained_unchanged.pop();
                }
                1 => lineage.created.clear(),
                2 => lineage.deleted.push(FaceKey::cut_wall()),
                3 => lineage.split.push(FaceKey::end_cap()),
                _ => lineage.merged.push(FaceKey::profile_x_lower()),
            };
            assert!(
                CadAuthoredResultTopology::admit(&graph, graph.digest_bytes(), lineage).is_err()
            );
        }
        assert!(CadAuthoredResultTopology::admit(&graph, [0x55; 32], baseline).is_err());
    }

    #[test]
    fn registered_planar_circular_hole_geometry_v2_evidence() {
        crate::circular_hole::tests::independent_ordinary_identity_witness_is_exact();
        crate::circular_hole::tests::retired_tolerance_bearing_wire_is_rejected();
        scale_family_projects_identical_typed_membership_and_replays();
        owner_dimension_and_complete_membership_fail_closed();
        mutated_deleted_split_or_merged_lineage_rejects();
        crate::circular_hole::tests::strict_scale_independent_geometry_replays_without_tolerance();
        crate::circular_hole::tests::finite_increasing_positive_and_strict_clearance_predicates_fail_closed();
        crate::circular_hole::tests::closed_v2_decoder_rejects_noncanonical_and_open_wire_mutants();
    }
}
