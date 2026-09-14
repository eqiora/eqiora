//! Private exact-area profile and one-way Stokes design binding.
//!
//! The complete private cell is consumed only by its precommitted `cfg(test)`
//! evidence until the accepted successor product path starts, so production
//! builds relax the unused-item lint exactly as the sibling geometry
//! realization module does. Under `cfg(test)` the lint stays denied.
#![cfg_attr(not(test), allow(dead_code))]

mod model_association;
mod topology_content;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod e1_evidence;
#[cfg(test)]
mod e1_sealed_input;

use super::api::SteadyIncompressibleStokesModel2d;
use super::geometry_realization::require_stokes_dissipation_mesh_predicates;
use super::recognize::lower_stokes_dissipation_profile_model_2d;
use crate::simplicial_ale_fsi::P1HarmonicMeshMotionAction;
use crate::simplicial_fsi::FixedReferenceFsiPartition;
use eqiora_artifact::{
    CanonicalModelArtifact, GeometryDefinitionV1, GeometryMeshCorrespondenceEnvelopeV1,
    ModelArtifactReference, ModelEnvelope, SimplicialMeshEnvelopeV1,
};
use eqiora_core::{Diagnostic, Id, RawId, entity::kinds};
use eqiora_meshing::{
    CellId, FixedTopologyGeometryState2d, MeshEntity, MeshQualityGate, SimplicialMesh,
};
use eqiora_realization::{
    AleGeometryQualityGate, ConformingTraceQuotient, P1HarmonicMeshMotionPolicy, TraceFieldEndpoint,
};
use eqiora_sem::KernelProgram;
use eqiora_solver::LinearSolveRequest;
use model_association::{require_complete_boundary_model, require_profile_parameters};
use std::collections::{BTreeMap, BTreeSet};
use topology_content::{
    chordal_geometry, mesh_facet_for_vertices, reference_vertices, require_topology_content,
    require_topology_indices, semantic_role,
};

#[cfg(test)]
pub(super) use e1_evidence::{E1ProfileTopologyEvidenceMutation2d, E1ProfileTopologyRejection2d};
#[cfg(test)]
pub(super) use e1_sealed_input::e1_stokes_dissipation_sealed_inputs_v1;

const PROFILE_FORMULA_VERSION: &str = "stokes-dissipation-two-mode-exact-area-v1";
const REFERENCE_TOPOLOGY_ID: &str = "stokes-square-ring-reference-n32-m4-v1";
const REFINED_TOPOLOGY_ID: &str = "stokes-square-ring-refined-n64-m8-v1";

/// The exact entity-set roles owned by one admitted chordal Geometry.
const ENTITY_SET_ROLES: [&str; 6] = [
    "fluid",
    "body",
    "outer_x_lower",
    "outer_x_upper",
    "outer_y_lower",
    "outer_y_upper",
];

#[derive(Debug, Clone, PartialEq)]
pub(super) struct StokesDissipationProfileGeometry2d {
    formula_version: &'static str,
    area_radius_parameter: RawId,
    a2_parameter: RawId,
    a4_parameter: RawId,
    area_radius_m: f64,
    a2: f64,
    a4: f64,
}
impl StokesDissipationProfileGeometry2d {
    pub(super) fn new(
        area_radius_parameter: RawId,
        a2_parameter: RawId,
        a4_parameter: RawId,
        area_radius_m: f64,
        a2: f64,
        a4: f64,
    ) -> Result<Self, Diagnostic> {
        if [area_radius_parameter, a2_parameter, a4_parameter]
            .into_iter()
            .collect::<BTreeSet<_>>()
            .len()
            != 3
        {
            return Err(invalid("profile Parameter identities must be distinct"));
        }
        if !area_radius_m.is_finite()
            || area_radius_m <= 0.0
            || [area_radius_m, a2, a4]
                .iter()
                .any(|value| !value.is_finite() || (*value == 0.0 && value.is_sign_negative()))
        {
            return Err(invalid(
                "profile values must be finite, canonical, and have positive r_A",
            ));
        }
        if a2.abs() + a4.abs() > 0.5 {
            return Err(invalid("profile coefficients violate |a_2| + |a_4| <= 1/2"));
        }
        Ok(Self {
            formula_version: PROFILE_FORMULA_VERSION,
            area_radius_parameter,
            a2_parameter,
            a4_parameter,
            area_radius_m,
            a2,
            a4,
        })
    }
    pub(super) const fn parameters(&self) -> [RawId; 3] {
        [
            self.area_radius_parameter,
            self.a2_parameter,
            self.a4_parameter,
        ]
    }
    pub(super) const fn values(&self) -> [f64; 3] {
        [self.area_radius_m, self.a2, self.a4]
    }
    /// Exact private analytic design identity: formula revision plus canonical
    /// coherent-SI `r_A` and dimensionless `(a_2, a_4)` bits.
    pub(super) fn identity(&self) -> (&'static str, [u64; 3]) {
        (self.formula_version, self.values().map(f64::to_bits))
    }
    /// Exact coherent-SI equal-area radius owned by this analytic profile.
    pub(super) const fn area_radius_m(&self) -> f64 {
        self.area_radius_m
    }
    /// Exact dimensionless two-mode coefficients in sealed order.
    pub(super) const fn coefficients(&self) -> [f64; 2] {
        [self.a2, self.a4]
    }
    /// The sole analytic radial formula `rho_a(theta)`.
    pub(super) fn radial_coordinate_m(&self, angle: f64) -> f64 {
        let denominator = (1.0 + 0.5 * (self.a2 * self.a2 + self.a4 * self.a4)).sqrt();
        self.area_radius_m * (1.0 + self.a2 * (2.0 * angle).cos() + self.a4 * (4.0 * angle).cos())
            / denominator
    }
    pub(super) fn radius(&self, angle: f64) -> Result<f64, Diagnostic> {
        if !angle.is_finite() {
            return Err(invalid("profile angle must be finite"));
        }
        let radius = self.radial_coordinate_m(angle);
        if !radius.is_finite() || radius <= 0.0 {
            return Err(invalid("profile evaluation is not finite and positive"));
        }
        Ok(radius)
    }
    /// Exact analytic area `pi r_A^2`, never the chordal polygon diagnostic.
    pub(super) fn analytic_area_m2(&self) -> f64 {
        std::f64::consts::PI * self.area_radius_m * self.area_radius_m
    }
    pub(super) fn bounds(&self) -> [[f64; 2]; 2] {
        let half_width = 10.0 * self.area_radius_m;
        [[-half_width, half_width], [-half_width, half_width]]
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StokesDissipationTopologyRole2d {
    Reference,
    Refined,
}
impl StokesDissipationTopologyRole2d {
    const fn contract(self) -> (&'static str, usize, usize) {
        match self {
            Self::Reference => (REFERENCE_TOPOLOGY_ID, 32, 4),
            Self::Refined => (REFINED_TOPOLOGY_ID, 64, 8),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct StokesDissipationVertexRecord2d {
    pub(super) id: usize,
    pub(super) ring_index: usize,
    pub(super) angle_index: usize,
    pub(super) ring_fraction: String,
    pub(super) angle_turns: String,
    pub(super) classification: String,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct StokesDissipationCellRecord2d {
    pub(super) id: usize,
    pub(super) vertices: [usize; 3],
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct StokesDissipationBoundaryFacetSource2d {
    pub(super) id: usize,
    pub(super) vertices: [usize; 2],
    pub(super) kind: String,
    pub(super) label: String,
    pub(super) orientation: String,
}
impl StokesDissipationBoundaryFacetSource2d {
    pub(super) const fn id(&self) -> usize {
        self.id
    }
    pub(super) const fn vertices(&self) -> [usize; 2] {
        self.vertices
    }
    pub(super) fn kind_name(&self) -> &str {
        &self.kind
    }
    pub(super) fn source_label(&self) -> &str {
        &self.label
    }
    pub(super) fn orientation_name(&self) -> &str {
        &self.orientation
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct StokesDissipationBodyCorrespondence2d {
    pub(super) angle_index: usize,
    pub(super) angle_turns: String,
    pub(super) body_vertex: usize,
    pub(super) body_facet: usize,
}
impl StokesDissipationBodyCorrespondence2d {
    pub(super) const fn angle_index(&self) -> usize {
        self.angle_index
    }
    pub(super) fn angle_turns(&self) -> &str {
        &self.angle_turns
    }
    pub(super) const fn body_vertex_id(&self) -> usize {
        self.body_vertex
    }
    pub(super) const fn body_facet_id(&self) -> usize {
        self.body_facet
    }
}
#[derive(Debug, Clone, PartialEq)]
pub(super) struct StokesDissipationTopologySource2d {
    pub(super) role: StokesDissipationTopologyRole2d,
    pub(super) content_identity: String,
    pub(super) sector_count: usize,
    pub(super) radial_interval_count: usize,
    pub(super) vertex_count: usize,
    pub(super) cell_count: usize,
    pub(super) facet_count: usize,
    pub(super) membership_counts: [usize; 3],
    pub(super) vertices: Vec<StokesDissipationVertexRecord2d>,
    pub(super) cells: Vec<StokesDissipationCellRecord2d>,
    pub(super) boundary_facets: Vec<StokesDissipationBoundaryFacetSource2d>,
    pub(super) ordered_body_angles: Vec<String>,
    pub(super) correspondence: Vec<StokesDissipationBodyCorrespondence2d>,
    pub(super) quality_gate: MeshQualityGate,
    pub(super) minimum_signed_area_m2: f64,
    pub(super) minimum_body_clearance_radius_multiple: f64,
    pub(super) coordinate_tolerance_m: f64,
}
#[derive(Debug, Clone, PartialEq)]
pub(super) struct StokesDissipationTopology2d {
    source: StokesDissipationTopologySource2d,
    role: StokesDissipationTopologyRole2d,
    content_identity: String,
    sector_count: usize,
    radial_interval_count: usize,
    body_vertex_ids: Vec<usize>,
    outer_vertex_ids: Vec<usize>,
    reference_mesh: SimplicialMesh,
    entity_sets: BTreeMap<String, Vec<MeshEntity>>,
}
impl StokesDissipationTopology2d {
    pub(super) fn admit(
        source: StokesDissipationTopologySource2d,
        area_radius_m: f64,
    ) -> Result<Self, Diagnostic> {
        let retained_source = source.clone();
        let (identity, sectors, intervals) = source.role.contract();
        if source.content_identity != identity
            || source.sector_count != sectors
            || source.radial_interval_count != intervals
        {
            return Err(invalid(
                "topology role, content identity, or exact N/M bounds differ",
            ));
        }
        if [
            source.minimum_signed_area_m2,
            source.minimum_body_clearance_radius_multiple,
            source.coordinate_tolerance_m,
        ]
        .into_iter()
        .any(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(invalid(
                "topology acceptance predicates must be finite and positive",
            ));
        }
        require_topology_indices(&source, sectors)?;
        require_topology_content(&source, sectors, intervals)?;
        let vertices = reference_vertices(area_radius_m, sectors, intervals)?;
        let body_vertex_ids = classified_vertex_ids(&source, "body_boundary");
        let outer_vertex_ids = classified_vertex_ids(&source, "outer_boundary");
        let mesh = SimplicialMesh::new(
            2,
            vertices,
            source
                .cells
                .iter()
                .map(|cell| Vec::from(cell.vertices))
                .collect(),
            source.quality_gate,
        )?;
        let mut entity_sets = BTreeMap::<String, Vec<MeshEntity>>::new();
        for facet in &source.boundary_facets {
            let mesh_facet = mesh_facet_for_vertices(&mesh, facet.vertices)?;
            let role = semantic_role(&facet.label)?;
            entity_sets
                .entry(role.to_owned())
                .or_default()
                .push(mesh_facet);
        }
        let required = ENTITY_SET_ROLES[1..]
            .iter()
            .map(|role| (*role).to_owned())
            .collect::<BTreeSet<_>>();
        if entity_sets.keys().cloned().collect::<BTreeSet<_>>() != required
            || entity_sets.values().any(Vec::is_empty)
        {
            return Err(invalid(
                "topology does not own the complete five-role boundary",
            ));
        }
        Ok(Self {
            source: retained_source,
            role: source.role,
            content_identity: source.content_identity,
            sector_count: sectors,
            radial_interval_count: intervals,
            body_vertex_ids,
            outer_vertex_ids,
            reference_mesh: mesh,
            entity_sets,
        })
    }
    pub(super) const fn entity_sets(&self) -> &BTreeMap<String, Vec<MeshEntity>> {
        &self.entity_sets
    }
    fn realize(
        &self,
        profile: &StokesDissipationProfileGeometry2d,
        solver: LinearSolveRequest<'_>,
    ) -> Result<RealizedStokesDissipationGeometry2d, Diagnostic> {
        let (augmented, partition, policy, original_vertices) =
            self.harmonic_auxiliary(solver.plan())?;
        let action = P1HarmonicMeshMotionAction::<2>::new(&augmented, &partition, policy, solver)?;
        debug_assert_eq!(original_vertices, self.reference_mesh.vertices().len());
        let state = self.harmonic_state(profile, &action)?;
        let mesh = state.reconstruct_mesh(&self.reference_mesh)?;
        require_stokes_dissipation_mesh_predicates(
            profile,
            &mesh,
            &self.reference_mesh,
            self.sector_count,
            self.source.minimum_signed_area_m2,
            self.source.minimum_body_clearance_radius_multiple,
            self.source.coordinate_tolerance_m,
        )?;
        let geometry = chordal_geometry(
            profile,
            self,
            &mesh,
            self.sector_count,
            self.source.coordinate_tolerance_m,
        )?;
        let mesh = SimplicialMeshEnvelopeV1::from_mesh(&mesh)?;
        let correspondence = GeometryMeshCorrespondenceEnvelopeV1::from_region(&geometry, &mesh)?;
        let mut entity_sets = BTreeMap::new();
        for role in ENTITY_SET_ROLES {
            entity_sets.insert(
                role.to_owned(),
                correspondence.region_entity_set_entities(&geometry, role)?,
            );
        }
        Ok(RealizedStokesDissipationGeometry2d {
            motion_action: action,
            state,
            geometry,
            mesh,
            correspondence,
            entity_sets,
        })
    }
    fn harmonic_state(
        &self,
        profile: &StokesDissipationProfileGeometry2d,
        action: &P1HarmonicMeshMotionAction<2>,
    ) -> Result<FixedTopologyGeometryState2d, Diagnostic> {
        let (augmented, _, _, original_vertices) =
            self.harmonic_auxiliary(action.policy().solver())?;
        let mut solid_displacement = (0..=self.sector_count)
            .map(|vertex| (eqiora_meshing::VertexId::new(vertex), [0.0; 2]))
            .collect::<BTreeMap<_, _>>();
        for angle_index in 0..self.sector_count {
            let angle = std::f64::consts::TAU * angle_index as f64 / self.sector_count as f64;
            let radius = profile.radius(angle)?;
            let start = &augmented.vertices()[angle_index];
            solid_displacement.insert(
                eqiora_meshing::VertexId::new(angle_index),
                [
                    radius * angle.cos() - start[0],
                    radius * angle.sin() - start[1],
                ],
            );
        }
        let displacement =
            action.apply(action.policy().solid_displacement(), &solid_displacement)?;
        let coordinates = augmented
            .vertices()
            .iter()
            .zip(displacement)
            .take(original_vertices)
            .map(|(reference, displacement)| {
                vec![
                    reference[0] + displacement[0],
                    reference[1] + displacement[1],
                ]
            })
            .collect();
        FixedTopologyGeometryState2d::new(&self.reference_mesh, coordinates)
    }
    fn harmonic_auxiliary(
        &self,
        solver: eqiora_solver::SolverPlan,
    ) -> Result<
        (
            SimplicialMesh,
            FixedReferenceFsiPartition<2>,
            P1HarmonicMeshMotionPolicy,
            usize,
        ),
        Diagnostic,
    > {
        let original_vertices = self.reference_mesh.vertices().len();
        let mut vertices = self.reference_mesh.vertices().to_vec();
        let center = vertices.len();
        vertices.push(vec![0.0, 0.0]);
        let mut cells = self.reference_mesh.cells().to_vec();
        let fluid_count = cells.len();
        for angle in 0..self.sector_count {
            cells.push(vec![center, angle, (angle + 1) % self.sector_count]);
        }
        let augmented =
            SimplicialMesh::new(2, vertices, cells, self.reference_mesh.quality_gate())?;
        let fluid_cells = (0..fluid_count).map(CellId::new).collect::<Vec<_>>();
        let solid_cells = (fluid_count..fluid_count + self.sector_count)
            .map(CellId::new)
            .collect::<Vec<_>>();
        let exact_id = |suffix| {
            let mut bytes = *b"eqiora-stokes-fi";
            bytes[15] = suffix;
            ulid::Ulid::from_bytes(bytes)
        };
        let fluid_domain = Id::<kinds::Domain>::from_ulid(exact_id(1));
        let solid_domain = Id::<kinds::Domain>::from_ulid(exact_id(2));
        let fluid_velocity = Id::<kinds::Field>::from_ulid(exact_id(3));
        let solid_velocity = Id::<kinds::Field>::from_ulid(exact_id(4));
        let solid_displacement = Id::<kinds::Field>::from_ulid(exact_id(5));
        let connection = Id::<kinds::Connection>::from_ulid(exact_id(6));
        let quotient = ConformingTraceQuotient::new(
            connection,
            TraceFieldEndpoint::new(fluid_domain, fluid_velocity),
            TraceFieldEndpoint::new(solid_domain, solid_velocity),
        )?;
        let partition = FixedReferenceFsiPartition::<2>::new(
            &augmented,
            [(fluid_domain, fluid_cells), (solid_domain, solid_cells)],
            &[quotient],
        )?;
        let policy = P1HarmonicMeshMotionPolicy::new(
            fluid_domain,
            solid_domain,
            solid_displacement,
            connection,
            AleGeometryQualityGate::new(1.0e-12)?,
            solver,
        )?;
        Ok((augmented, partition, policy, original_vertices))
    }
}
fn classified_vertex_ids(
    source: &StokesDissipationTopologySource2d,
    classification: &str,
) -> Vec<usize> {
    source
        .vertices
        .iter()
        .filter(|record| record.classification == classification)
        .map(|record| record.id)
        .collect()
}
#[derive(Debug, Clone, PartialEq)]
struct RealizedStokesDissipationGeometry2d {
    motion_action: P1HarmonicMeshMotionAction<2>,
    state: FixedTopologyGeometryState2d,
    geometry: GeometryDefinitionV1,
    mesh: SimplicialMeshEnvelopeV1,
    correspondence: GeometryMeshCorrespondenceEnvelopeV1,
    entity_sets: BTreeMap<String, Vec<MeshEntity>>,
}
#[derive(Debug, Clone, PartialEq)]
pub(super) struct StokesDissipationGeometryModelBinding2d {
    profile: StokesDissipationProfileGeometry2d,
    topology: StokesDissipationTopology2d,
    motion_action: P1HarmonicMeshMotionAction<2>,
    state: FixedTopologyGeometryState2d,
    geometry: GeometryDefinitionV1,
    mesh: SimplicialMeshEnvelopeV1,
    mesh_artifact: [u8; 32],
    correspondence: GeometryMeshCorrespondenceEnvelopeV1,
    entity_sets: BTreeMap<String, Vec<MeshEntity>>,
    program: KernelProgram,
    model: SteadyIncompressibleStokesModel2d,
    model_artifact: ModelArtifactReference,
    model_profile_values: [f64; 3],
    sealed_input: Option<[u8; 32]>,
}
impl StokesDissipationGeometryModelBinding2d {
    pub(super) fn new<F>(
        profile: StokesDissipationProfileGeometry2d,
        topology: &StokesDissipationTopology2d,
        harmonic_solver: LinearSolveRequest<'_>,
        build_model: F,
        sealed_input: Option<[u8; 32]>,
    ) -> Result<Self, Diagnostic>
    where
        F: FnOnce(&GeometryDefinitionV1) -> Result<KernelProgram, Diagnostic>,
    {
        let realized = topology.realize(&profile, harmonic_solver)?;
        let program = build_model(&realized.geometry)?;
        let model_profile_values = require_profile_parameters(&program, &profile)?;
        let model = lower_stokes_dissipation_profile_model_2d(
            &program,
            &realized.geometry,
            profile.bounds(),
            profile.parameters(),
        )?;
        require_complete_boundary_model(&program, &model, &profile)?;
        let model_artifact = ModelEnvelope::from_program(&program)?.artifact_reference()?;
        if model_artifact.semantic_revision().get() != program.revision().0
            || model_artifact.model() != program.model()
        {
            return Err(invalid(
                "Model artifact identity differs from the exact semantic revision",
            ));
        }
        Ok(Self {
            profile,
            topology: topology.clone(),
            motion_action: realized.motion_action,
            state: realized.state,
            mesh_artifact: realized.mesh.digest()?.sha256_bytes(),
            geometry: realized.geometry,
            mesh: realized.mesh,
            correspondence: realized.correspondence,
            entity_sets: realized.entity_sets,
            program,
            model,
            model_artifact,
            model_profile_values,
            sealed_input,
        })
    }
    pub(super) const fn mesh(&self) -> &SimplicialMeshEnvelopeV1 {
        &self.mesh
    }
    pub(super) const fn model(&self) -> &SteadyIncompressibleStokesModel2d {
        &self.model
    }
    /// The exact Model program this binding was admitted from.
    pub(super) const fn program(&self) -> &KernelProgram {
        &self.program
    }
    /// Exact analytic design owner retained by value.
    pub(super) const fn profile(&self) -> &StokesDissipationProfileGeometry2d {
        &self.profile
    }
    /// Exact analytic design identity retained by the binding.
    pub(super) fn profile_identity(&self) -> (&'static str, [u64; 3]) {
        self.profile.identity()
    }
    /// Exact `r_A`, `a_2`, and `a_4` values read from the admitted Model.
    pub(super) const fn model_profile_values(&self) -> [f64; 3] {
        self.model_profile_values
    }
    /// The same analytic identity, formed only from Model Parameter values.
    pub(super) fn model_profile_identity(&self) -> (&'static str, [u64; 3]) {
        (
            PROFILE_FORMULA_VERSION,
            self.model_profile_values.map(f64::to_bits),
        )
    }
    /// Digest of the derived straight-edged finite-element realization.
    pub(super) fn chordal_geometry_digest(&self) -> [u8; 32] {
        self.geometry.canonical().digest_bytes()
    }
    /// Digest the admitted Model's `GeometryRegion` Domain names.
    pub(super) fn model_geometry_region_digest(&self) -> [u8; 32] {
        self.model
            .geometry_source_digest()
            .expect("an admitted binding retains one exact GeometryRegion digest")
    }
    /// Exact Mesh artifact digest of the realized fixed-topology state.
    pub(super) const fn mesh_artifact_digest(&self) -> [u8; 32] {
        self.mesh_artifact
    }
    /// Exact Model artifact digest of the admitted trial Model.
    pub(super) fn model_artifact_digest(&self) -> [u8; 32] {
        self.model_artifact.artifact().sha256_bytes()
    }
    /// Complete exact entity-set inventory of the derived chordal Geometry.
    pub(super) fn entity_set_names(&self) -> Vec<&str> {
        self.entity_sets.keys().map(String::as_str).collect()
    }
    /// Explicit content-bound topology role; never inferred.
    pub(super) const fn topology_role(&self) -> StokesDissipationTopologyRole2d {
        self.topology.role
    }
    /// Exact sealed topology content identity.
    pub(super) fn topology_content_identity(&self) -> &str {
        &self.topology.content_identity
    }
    /// The unchanged start-design connectivity and reference coordinates.
    pub(super) const fn reference_topology(&self) -> &SimplicialMesh {
        &self.topology.reference_mesh
    }
    /// The accepted fixed-reference harmonic state for this design.
    pub(super) const fn fixed_topology_state(&self) -> &FixedTopologyGeometryState2d {
        &self.state
    }
    /// Ordered exact body-boundary vertex identities.
    pub(super) fn body_vertex_ids(&self) -> &[usize] {
        &self.topology.body_vertex_ids
    }
    /// Ordered exact outer-boundary vertex identities.
    pub(super) fn outer_vertex_ids(&self) -> &[usize] {
        &self.topology.outer_vertex_ids
    }
    /// Exact sealed ordered body angles as rational turn strings.
    pub(super) fn ordered_body_angle_turns(&self) -> &[String] {
        &self.topology.source.ordered_body_angles
    }
    /// Exact sealed same-index body correspondence.
    pub(super) fn correspondence(&self) -> &[StokesDissipationBodyCorrespondence2d] {
        &self.topology.source.correspondence
    }
    /// Exact sealed oriented boundary-facet inventory.
    pub(super) fn boundary_facets(&self) -> &[StokesDissipationBoundaryFacetSource2d] {
        &self.topology.source.boundary_facets
    }
    pub(super) fn entities(&self, role: &str) -> Result<&[MeshEntity], Diagnostic> {
        self.entity_sets
            .get(role)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid(format!("binding has no exact entity-set role `{role}`")))
    }
    pub(super) fn revalidate(&self) -> Result<(), Diagnostic> {
        let admitted = StokesDissipationTopology2d::admit(
            self.topology.source.clone(),
            self.profile.area_radius_m(),
        )?;
        if admitted != self.topology {
            return Err(invalid(
                "binding topology differs from complete exact-content replay",
            ));
        }
        if require_profile_parameters(&self.program, &self.profile)? != self.model_profile_values {
            return Err(invalid(
                "binding retained Model design values differ from exact Parameter replay",
            ));
        }
        if self.model.geometry_source_digest() != Some(self.geometry.canonical().digest_bytes()) {
            return Err(invalid(
                "binding Model GeometryRegion digest differs from the derived chordal Geometry",
            ));
        }
        if self
            .topology
            .harmonic_state(&self.profile, &self.motion_action)?
            != self.state
        {
            return Err(invalid(
                "binding harmonic state differs from exact fixed-reference replay",
            ));
        }
        let replayed_mesh = self.state.reconstruct_mesh(&self.topology.reference_mesh)?;
        let mesh = SimplicialMeshEnvelopeV1::from_mesh(&replayed_mesh)?;
        if mesh != self.mesh || mesh.digest()?.sha256_bytes() != self.mesh_artifact {
            return Err(invalid(
                "binding fixed-topology state and retained Mesh identity differ",
            ));
        }
        require_stokes_dissipation_mesh_predicates(
            &self.profile,
            &replayed_mesh,
            &self.topology.reference_mesh,
            self.topology.sector_count,
            self.topology.source.minimum_signed_area_m2,
            self.topology.source.minimum_body_clearance_radius_multiple,
            self.topology.source.coordinate_tolerance_m,
        )?;
        let geometry = chordal_geometry(
            &self.profile,
            &self.topology,
            &replayed_mesh,
            self.topology.sector_count,
            self.topology.source.coordinate_tolerance_m,
        )?;
        if geometry != self.geometry {
            return Err(invalid(
                "binding analytic profile/state and retained chordal Geometry differ",
            ));
        }
        let correspondence =
            GeometryMeshCorrespondenceEnvelopeV1::from_region(&geometry, &self.mesh)?;
        if correspondence != self.correspondence {
            return Err(invalid(
                "binding chordal Geometry and retained Mesh correspondence differ",
            ));
        }
        for role in ENTITY_SET_ROLES {
            if correspondence.region_entity_set_entities(&geometry, role)?
                != *self.entities(role)?
            {
                return Err(invalid(
                    "binding retained entity membership differs from exact correspondence replay",
                ));
            }
        }
        let model = lower_stokes_dissipation_profile_model_2d(
            &self.program,
            &geometry,
            self.profile.bounds(),
            self.profile.parameters(),
        )?;
        require_complete_boundary_model(&self.program, &model, &self.profile)?;
        let model_artifact = ModelEnvelope::from_program(&self.program)?.artifact_reference()?;
        if model != self.model || model_artifact != self.model_artifact {
            return Err(invalid(
                "binding retained Model meaning or exact Model artifact differs from replay",
            ));
        }
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(eqiora_core::diagnostic::codes::INVALID_REALIZATION, message)
}
