//! Exact-geometry binding for the common transient MINI/P1 path.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_artifact::{GeometryMeshCorrespondenceEnvelopeV1, SimplicialMeshEnvelopeV1};
use eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND;
use eqiora_core::{Diagnostic, RawId};
use eqiora_geometry::CanonicalGeometryV1;
use eqiora_meshing::{
    MeshEntity, MeshTopology, QuadratureRule, SimplicialMesh, simplex_duffy_gauss_legendre,
    triangle_duffy_gauss_legendre,
};
use eqiora_realization::{
    MeshArtifactReference, NonlinearSolvePlan, ResolvedTransientFieldwiseRealization,
    TransientFieldwiseRealizationPlan, TransientFieldwiseRealizationRequirements,
};
use eqiora_schema::kernel::{DomainKind, KernelNode};
use eqiora_sem::KernelProgram;
use eqiora_solver::{LinearSolverBackend, SolverPlan};

use super::IncompressibleFlowScaleProfile2d;
use super::geometry_realization::normalize_geometry_mesh;
use super::navier_stokes::{
    TransientIncompressibleNavierStokesModel2d,
    lower_transient_incompressible_navier_stokes_geometry_2d,
};
use super::navier_stokes_realization::{
    ResolvedTransientNavierStokesState2d, TransientNavierStokesInitialState2d,
    TransientNavierStokesRun2d, invalid_realization, normalize_state, pressure_id,
    reconstruct_state, require_exact_transient_plan,
    transient_navier_stokes_fieldwise_requirements_for_2d,
    transient_navier_stokes_mini_plan_for_2d, velocity_id,
};
use crate::canonical_boundary::{PhysicalBoundaryDisposition, PhysicalBoundaryQuantity};
use crate::discrete_block::DiscreteBlockSystem;
use crate::simplicial_navier_stokes::{
    MiniNavierStokesStepPlan2d, PreparedStepStructure,
    advance_simplicial_mini_navier_stokes_2d_with_prepared_structure, prepare_step_structure,
};
use crate::simplicial_stokes::{
    SimplicialMiniStokesBoundary2d, SimplicialMiniStokesBoundaryCondition2d,
    SimplicialMiniStokesBoundaryFacet2d,
};

const DIMENSION: usize = 2;
const DUFFY_POINTS_PER_AXIS: usize = 5;
const REQUIRED_BOUNDARY_SETS: [&str; 4] = ["cylinder", "inlet", "outlet", "walls"];

pub(super) struct TransientGeometryBoundary2d {
    pub(super) boundary: SimplicialMiniStokesBoundary2d,
    pub(super) fixed_velocity: Vec<Option<[f64; DIMENSION]>>,
}

/// Authenticated exact Geometry-to-Gmsh binding for transient flow.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TransientNavierStokesGeometryBinding2d {
    program: KernelProgram,
    model: TransientIncompressibleNavierStokesModel2d,
    source: CanonicalGeometryV1,
    mesh: SimplicialMeshEnvelopeV1,
    correspondence: GeometryMeshCorrespondenceEnvelopeV1,
    entity_sets: BTreeMap<String, Vec<MeshEntity>>,
}

impl TransientNavierStokesGeometryBinding2d {
    pub(crate) fn new_authenticated(
        program: &KernelProgram,
        source: &CanonicalGeometryV1,
        mesh: &SimplicialMeshEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        if unique_geometry_source_digest(program)
            .is_some_and(|digest| digest != source.digest_bytes())
        {
            return Err(invalid("Model belongs to another exact source revision"));
        }
        let mut entity_sets = BTreeMap::new();
        for name in REQUIRED_BOUNDARY_SETS.into_iter().chain(["fluid"]) {
            entity_sets.insert(
                name.to_owned(),
                correspondence.planar_circular_hole_v2_entity_set_entities(source, name)?,
            );
        }
        let expected_cells = (0..mesh.mesh().entity_count(DIMENSION).expect("2D mesh cells"))
            .map(|index| MeshEntity::new(DIMENSION, index))
            .collect::<Vec<_>>();
        if entity_sets["fluid"] != expected_cells {
            return Err(invalid(
                "the exact `fluid` entity set does not realize every mesh cell exactly once",
            ));
        }
        let model = lower_transient_incompressible_navier_stokes_geometry_2d(program, source)?;
        if model.geometry_source_digest() != Some(source.digest_bytes()) {
            return Err(invalid(
                "Model GeometryRegion digest differs from the accepted exact source revision",
            ));
        }
        Ok(Self {
            program: program.clone(),
            model,
            source: source.clone(),
            mesh: mesh.clone(),
            correspondence: correspondence.clone(),
            entity_sets,
        })
    }

    pub(crate) fn domain(&self) -> RawId {
        self.model.domain
    }

    pub(crate) fn velocity(&self) -> RawId {
        self.model.velocity
    }

    pub(crate) fn pressure(&self) -> RawId {
        self.model.pressure
    }

    pub(crate) fn fieldwise_requirements(&self) -> TransientFieldwiseRealizationRequirements {
        transient_navier_stokes_fieldwise_requirements_for_2d(&self.model)
    }

    pub(crate) fn mini_plan(
        &self,
        mesh: MeshArtifactReference,
        scales: IncompressibleFlowScaleProfile2d,
        time_step: eqiora_core::DynQuantity,
        nonlinear: NonlinearSolvePlan,
        solver: SolverPlan,
    ) -> Result<TransientFieldwiseRealizationPlan, Diagnostic> {
        transient_navier_stokes_mini_plan_for_2d(
            &self.model,
            mesh,
            scales,
            time_step,
            nonlinear,
            solver,
        )
    }

    pub(crate) fn model(&self) -> &TransientIncompressibleNavierStokesModel2d {
        &self.model
    }

    pub(crate) fn source(&self) -> &CanonicalGeometryV1 {
        &self.source
    }

    pub(crate) fn mesh(&self) -> &SimplicialMeshEnvelopeV1 {
        &self.mesh
    }

    pub(crate) fn entities(&self, name: &str) -> Result<&[MeshEntity], Diagnostic> {
        self.entity_sets
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid(format!("geometry binding has no entity set named `{name}`")))
    }
}

pub(super) fn geometry_boundary(
    binding: &TransientNavierStokesGeometryBinding2d,
    normalized: &eqiora_meshing::SimplicialMesh,
    scales: IncompressibleFlowScaleProfile2d,
) -> Result<TransientGeometryBoundary2d, Diagnostic> {
    let model = binding.model();
    let physical = binding.mesh.mesh();
    let mut facet_owners = BTreeMap::<MeshEntity, RawId>::new();
    let mut fixed_velocity = vec![None; physical.vertices().len()];
    for (name, boundary_id) in &model.named_boundary_ids {
        let facets = binding.entities(name)?;
        let disposition = *model
            .boundary_dispositions
            .get(boundary_id)
            .ok_or_else(|| invalid(format!("transient Model omits Boundary `{name}`")))?;
        for &facet in facets {
            if facet.dimension() != DIMENSION - 1
                || !physical
                    .is_boundary_entity(facet)
                    .is_some_and(|boundary| boundary)
                || facet_owners.insert(facet, *boundary_id).is_some()
            {
                return Err(invalid(
                    "transient geometry boundary membership overlaps or contains a non-boundary facet",
                ));
            }
        }
        let essential = disposition == PhysicalBoundaryDisposition::TraceZero
            || matches!(
                disposition,
                PhysicalBoundaryDisposition::Prescribed(law)
                    if law.quantity() == PhysicalBoundaryQuantity::Trace
            );
        if !essential {
            continue;
        }
        let outward = match disposition {
            PhysicalBoundaryDisposition::TraceZero => None,
            PhysicalBoundaryDisposition::Prescribed(_) => Some(
                binding
                    .source
                    .constant_parent_outward_normal(name)
                    .ok_or_else(|| {
                        invalid(format!(
                            "prescribed normal velocity on `{name}` has no exact fixed-side normal"
                        ))
                    })?,
            ),
            _ => unreachable!("essential predicate is exact"),
        };
        for &facet in facets {
            for vertex in physical
                .entity_vertices(facet)
                .expect("validated correspondence facet owns vertices")
            {
                let index = vertex.index();
                let value = if let Some(outward) = outward {
                    model
                        .prescribed_normal_velocity(
                            *boundary_id,
                            outward,
                            &physical.vertices()[index],
                        )?
                        .ok_or_else(|| {
                            invalid(format!(
                                "prescribed normal velocity on `{name}` has no retained expression"
                            ))
                        })?
                        .map(|component| component / scales.velocity_value())
                } else {
                    [0.0; DIMENSION]
                };
                if fixed_velocity[index].is_some_and(|existing| existing != value) {
                    return Err(invalid(format!(
                        "essential velocity prescriptions disagree at a vertex shared by `{name}`"
                    )));
                }
                fixed_velocity[index] = Some(value);
            }
        }
    }
    let all_boundary_facets = (0..physical.entity_count(DIMENSION - 1).expect("2D facets"))
        .map(|index| MeshEntity::new(DIMENSION - 1, index))
        .filter(|facet| {
            physical
                .is_boundary_entity(*facet)
                .expect("mesh owns every facet")
        })
        .collect::<BTreeSet<_>>();
    if facet_owners.keys().copied().collect::<BTreeSet<_>>() != all_boundary_facets {
        return Err(invalid(
            "transient geometry boundary roles do not exhaust the exact boundary exactly once",
        ));
    }
    let facets = facet_owners
        .into_iter()
        .map(|(facet, boundary_id)| {
            let disposition = model.boundary_dispositions[&boundary_id];
            let condition = match disposition {
                PhysicalBoundaryDisposition::TraceZero => {
                    SimplicialMiniStokesBoundaryCondition2d::EssentialVelocity
                }
                PhysicalBoundaryDisposition::Prescribed(law)
                    if law.quantity() == PhysicalBoundaryQuantity::Trace =>
                {
                    SimplicialMiniStokesBoundaryCondition2d::EssentialVelocity
                }
                PhysicalBoundaryDisposition::FluxZero => {
                    SimplicialMiniStokesBoundaryCondition2d::ConstantTraction {
                        value: [0.0; DIMENSION],
                    }
                }
                PhysicalBoundaryDisposition::Prescribed(law) => {
                    return Err(invalid(format!(
                        "transient prescribed traction Relation {} is not admitted by the Geometry slice",
                        law.relation()
                    )));
                }
                PhysicalBoundaryDisposition::PortBinding { connection, port } => {
                    return Err(invalid(format!(
                        "live transient PortBinding {connection} through Port {port} requires an explicit trace-space interface Realization"
                    )));
                }
            };
            Ok(SimplicialMiniStokesBoundaryFacet2d::new(facet, condition))
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    let boundary = SimplicialMiniStokesBoundary2d::new(normalized, facets)
        .and_then(|boundary| {
            boundary.with_named_reaction_surface(
                normalized,
                "cylinder",
                binding.entities("cylinder")?.iter().copied(),
            )
        })
        .map_err(|error| invalid(error.message()))?;
    Ok(TransientGeometryBoundary2d {
        boundary,
        fixed_velocity,
    })
}

/// Ephemeral invariant structure for one exact-Geometry MINI Run.
pub(crate) struct PreparedResolvedTransientGeometryMiniRun2d<'a> {
    binding: &'a TransientNavierStokesGeometryBinding2d,
    mesh_artifact: MeshArtifactReference,
    physical_mesh: &'a SimplicialMesh,
    normalized: SimplicialMesh,
    step_structure: PreparedStepStructure,
    scales: IncompressibleFlowScaleProfile2d,
    numerical_plan: MiniNavierStokesStepPlan2d,
    block_system: DiscreteBlockSystem,
    cell_quadrature: QuadratureRule,
    facet_quadrature: QuadratureRule,
    with_gauge: bool,
}

impl PreparedResolvedTransientGeometryMiniRun2d<'_> {
    pub(crate) fn advance(
        &self,
        initial: TransientNavierStokesInitialState2d,
        run: TransientNavierStokesRun2d,
        solver: &dyn LinearSolverBackend,
    ) -> Result<Vec<ResolvedTransientNavierStokesState2d>, Diagnostic> {
        let common = self.binding.model();
        if initial.mesh_artifact != self.mesh_artifact
            || initial.velocity_field != velocity_id(common)
            || initial.pressure_field != pressure_id(common)
        {
            return Err(invalid_realization(
                "transient initial state identity differs from the resolved Model or mesh revision",
            ));
        }
        if initial.velocity.mesh() != self.physical_mesh
            || initial.pressure.mesh() != self.physical_mesh
        {
            return Err(invalid_realization(
                "transient Navier--Stokes initial fields are stale for the selected mesh artifact",
            ));
        }
        let numerical_initial =
            normalize_state(&initial, &self.normalized, self.scales, self.with_gauge)?;
        let checked_assembly = self
            .block_system
            .checked_backend(&REFERENCE_ASSEMBLY_BACKEND);
        let lower = [common.bounds[0][0], common.bounds[1][0]];
        let length = self.scales.length_value();
        let pressure = self.scales.pressure_value();
        let body_force = |coordinate_hat: [f64; DIMENSION]| {
            let coordinate = [
                lower[0] + length * coordinate_hat[0],
                lower[1] + length * coordinate_hat[1],
            ];
            let force = common.conservative_body_force(&coordinate)?;
            Ok([length * force[0] / pressure, length * force[1] / pressure])
        };
        let numerical = advance_simplicial_mini_navier_stokes_2d_with_prepared_structure(
            &self.normalized,
            &self.step_structure,
            &body_force,
            numerical_initial,
            run.step_count(),
            self.numerical_plan,
            &self.cell_quadrature,
            &self.facet_quadrature,
            &checked_assembly,
            solver,
        )?;
        if checked_assembly.validated_materialization_count() == 0 {
            return Err(invalid_realization(
                "transient execution returned without a validated block materialization",
            ));
        }
        numerical
            .states()
            .iter()
            .enumerate()
            .map(|(index, state)| {
                reconstruct_state(
                    state,
                    self.physical_mesh,
                    common,
                    self.scales,
                    index.checked_sub(1).map(|step| &numerical.steps()[step]),
                )
            })
            .collect()
    }
}

pub(crate) fn prepare_resolved_transient_navier_stokes_geometry_mini_run_2d<'a>(
    program: &KernelProgram,
    resolved: &'a ResolvedTransientFieldwiseRealization,
    binding: &'a TransientNavierStokesGeometryBinding2d,
) -> Result<PreparedResolvedTransientGeometryMiniRun2d<'a>, Diagnostic> {
    let mesh = binding.mesh();
    let mesh_artifact = mesh.artifact_reference()?;
    if program.model() != resolved.model()
        || program.revision().0 != resolved.semantic_revision().get()
    {
        return Err(invalid_realization(
            "resolved transient realization does not reference this exact Semantic Model revision",
        ));
    }
    let replayed =
        lower_transient_incompressible_navier_stokes_geometry_2d(program, binding.source())?;
    if &replayed != binding.model() {
        return Err(invalid_realization(
            "transient Model meaning differs from the exact Geometry binding",
        ));
    }
    let common = binding.model();
    let with_gauge = super::navier_stokes_realization::boundary::pressure_uses_gauge(common)?;
    let realization_graph = resolved.portable_graph()?;
    let (scales, numerical_plan) =
        require_exact_transient_plan(common, resolved, &realization_graph, mesh_artifact)?;
    let normalized = normalize_geometry_mesh(&common.bounds, mesh.mesh(), scales.length_value())?;
    let geometry_boundary = geometry_boundary(binding, &normalized, scales)?;
    let fixed_velocity = normalized
        .vertices()
        .iter()
        .enumerate()
        .filter_map(|(index, coordinate)| {
            geometry_boundary.fixed_velocity[index]
                .map(|value| ((coordinate[0].to_bits(), coordinate[1].to_bits()), value))
        })
        .collect::<BTreeMap<_, _>>();
    let block_system = super::block::transient_navier_stokes_block_system(
        program,
        common,
        mesh_artifact,
        &normalized,
        &geometry_boundary.boundary,
        resolved,
        scales,
    )?;
    let essential_velocity = |coordinate: [f64; DIMENSION]| {
        fixed_velocity
            .get(&(coordinate[0].to_bits(), coordinate[1].to_bits()))
            .copied()
            .ok_or_else(|| {
                invalid_realization(
                    "an essential geometry vertex is absent from correspondence-derived trace data",
                )
            })
    };
    let cell_quadrature = triangle_duffy_gauss_legendre(DUFFY_POINTS_PER_AXIS)?;
    let facet_quadrature = simplex_duffy_gauss_legendre(DIMENSION - 1, 2)?;
    let step_structure = prepare_step_structure(
        &normalized,
        &geometry_boundary.boundary,
        &essential_velocity,
        &cell_quadrature,
        &facet_quadrature,
    )?;
    Ok(PreparedResolvedTransientGeometryMiniRun2d {
        binding,
        mesh_artifact,
        physical_mesh: mesh.mesh(),
        normalized,
        step_structure,
        scales,
        numerical_plan,
        block_system,
        cell_quadrature,
        facet_quadrature,
        with_gauge,
    })
}

fn unique_geometry_source_digest(program: &KernelProgram) -> Option<[u8; 32]> {
    let mut digests = program.nodes().filter_map(|node| match node {
        KernelNode::Domain(domain) => match domain.kind() {
            DomainKind::GeometryRegion { geometry, .. } => Some(geometry.bytes()),
            _ => None,
        },
        _ => None,
    });
    let digest = digests.next()?;
    digests.next().is_none().then_some(digest)
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(eqiora_core::diagnostic::codes::INVALID_REALIZATION, message)
}
