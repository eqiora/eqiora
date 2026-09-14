//! Exact-geometry binding for the shared steady MINI/P1 Stokes path.

use std::collections::{BTreeMap, BTreeSet};

use eqiora_artifact::{GeometryMeshCorrespondenceEnvelopeV1, SimplicialMeshEnvelopeV1};
use eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND;
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_geometry::CanonicalGeometryV1;
use eqiora_meshing::{MeshEntity, MeshTopology, SimplicialMesh};
use eqiora_realization::{
    FieldwiseRealizationPlan, FieldwiseRealizationRequirements, MeshArtifactReference,
    ResolvedFieldwiseRealization,
};
use eqiora_schema::kernel::{DomainKind, KernelNode};
use eqiora_sem::KernelProgram;
use eqiora_solver::{LinearSolverBackend, SolverPlan};

use super::api::{SteadyIncompressibleStokesModel2d, StokesBoundaryKey2d};
use super::dissipation_profile::{
    StokesDissipationGeometryModelBinding2d, StokesDissipationProfileGeometry2d,
};
use super::physical::FinalizedSteadyStokesMini2dProblem;
use super::physical::SteadyStokesMiniSolution2d;
use super::realization::{
    SteadyStokesScaleProfile2d, finalize_lowered_steady_stokes_mini_2d_with_assembly,
    resolved_stokes_scales, steady_stokes_fieldwise_requirements_for_model_2d,
    steady_stokes_mini_plan_for_model_2d, validate_model_owned_essential_transport,
};
use super::recognize::lower_steady_incompressible_stokes_geometry_2d;
use crate::canonical_boundary::{PhysicalBoundaryDisposition, PhysicalBoundaryQuantity};
use crate::simplicial_stokes::{
    SimplicialMiniStokesBoundary2d, SimplicialMiniStokesBoundaryCondition2d,
    SimplicialMiniStokesBoundaryFacet2d,
};

pub(crate) mod scaling;

const DIMENSION: usize = 2;
const REQUIRED_BOUNDARY_SETS: [&str; 4] = ["cylinder", "inlet", "outlet", "walls"];

struct GeometryBoundary2d {
    boundary: SimplicialMiniStokesBoundary2d,
    fixed_velocity: Vec<Option<[f64; DIMENSION]>>,
}

/// Authenticated exact-circle Geometry-to-Mesh binding for steady Stokes.
///
/// This value is deliberately narrower than a geometry framework. It accepts
/// the circular kind of the common exact owner, its ordinary region and mesh
/// artifacts, and the authored-region correspondence. Named sets are resolved
/// once through that correspondence and never recovered from coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct SteadyStokesGeometryBinding2d {
    program: KernelProgram,
    model: SteadyIncompressibleStokesModel2d,
    source: CanonicalGeometryV1,
    mesh: SimplicialMeshEnvelopeV1,
    correspondence: GeometryMeshCorrespondenceEnvelopeV1,
    entity_sets: BTreeMap<String, Vec<MeshEntity>>,
}

impl SteadyStokesGeometryBinding2d {
    pub(crate) const fn formulation_correspondence(
        &self,
    ) -> &crate::form_compiler::vocabulary::MixedGalerkinCorrespondence {
        self.model.correspondence()
    }

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
        let model = lower_steady_incompressible_stokes_geometry_2d(program, source)?;
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

    /// Exact field-wise requirements derived from the bound Model.
    #[must_use]
    pub fn fieldwise_requirements(&self) -> FieldwiseRealizationRequirements {
        steady_stokes_fieldwise_requirements_for_model_2d(&self.model)
    }

    pub(crate) fn algebraic_structure(
        &self,
    ) -> Result<eqiora_solver::AlgebraicStructure, Diagnostic> {
        super::realization::steady_stokes_algebraic_structure(&self.model)
    }

    /// Build the bounded MINI/P1 plan for this source-bound Model.
    ///
    /// # Errors
    /// Preserves exact scale, solver-tuple, and plan-construction diagnostics.
    pub fn mini_plan(
        &self,
        mesh: MeshArtifactReference,
        scales: SteadyStokesScaleProfile2d,
        solver: SolverPlan,
    ) -> Result<FieldwiseRealizationPlan, Diagnostic> {
        steady_stokes_mini_plan_for_model_2d(&self.model, mesh, scales, solver)
    }

    pub(crate) fn entities(&self, name: &str) -> Result<&[MeshEntity], Diagnostic> {
        self.entity_sets
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid(format!("geometry binding has no entity set named `{name}`")))
    }

    fn mesh_reference(&self) -> Result<MeshArtifactReference, Diagnostic> {
        self.mesh.artifact_reference()
    }
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

/// Solve the exact geometry-backed steady Stokes model through reference paths.
///
/// The Model source digest and accepted artifact resources are
/// all reaccepted before assembly. The returned named `cylinder` reaction uses
/// the existing constrained-residual convention: force on the fluid. Retained
/// inlet and outlet fluxes use the physical parent-outward convention.
///
/// # Errors
/// Preserves semantic replay, exact-source binding, field-wise Realization,
/// reference assembly, backend execution, coherent-SI reconstruction and
/// named-boundary flux reconstruction diagnostics.
pub fn solve_resolved_steady_stokes_geometry_mini_2d(
    program: &KernelProgram,
    resolved: &ResolvedFieldwiseRealization,
    binding: &SteadyStokesGeometryBinding2d,
    backend: &dyn LinearSolverBackend,
) -> Result<SteadyStokesMiniSolution2d, Diagnostic> {
    // Replay at the use site so a future decoded binding cannot bypass the
    // source/region/mesh relationship established by `new`.
    let model = lower_steady_incompressible_stokes_geometry_2d(program, &binding.source)?;
    if model != binding.model
        || model.geometry_source_digest() != Some(binding.source.digest_bytes())
    {
        return Err(invalid(
            "Model meaning or GeometryRegion digest differs from the source-bound Stokes binding",
        ));
    }
    let mesh_reference = binding.mesh_reference()?;
    let scales = resolved_stokes_scales(program, resolved, mesh_reference, &model)?;
    let normalized =
        normalize_geometry_mesh(model.bounds(), binding.mesh.mesh(), scales.length_value())?;
    let geometry_boundary = geometry_boundary(&model, binding, &normalized, scales)?;
    let lookup = normalized
        .vertices()
        .iter()
        .enumerate()
        .filter_map(|(index, coordinate)| {
            geometry_boundary.fixed_velocity[index]
                .map(|value| ((coordinate[0].to_bits(), coordinate[1].to_bits()), value))
        })
        .collect::<BTreeMap<_, _>>();
    let essential_velocity = |coordinate: [f64; DIMENSION]| {
        lookup
            .get(&(coordinate[0].to_bits(), coordinate[1].to_bits()))
            .copied()
            .ok_or_else(|| {
                invalid(
                    "an essential geometry vertex is absent from correspondence-derived trace data",
                )
            })
    };
    let finalized = finalize_lowered_steady_stokes_mini_2d_with_assembly(
        resolved,
        mesh_reference,
        &model,
        binding.mesh.mesh(),
        &normalized,
        &geometry_boundary.boundary,
        scales,
        &essential_velocity,
        &REFERENCE_ASSEMBLY_BACKEND,
    )?;
    let solved = backend.solve(&finalized.linear_problem()?, finalized.solver_plan())?;
    let solution = finalized.finish(solved)?;
    let named_boundary_fluxes = ["inlet", "outlet"]
        .into_iter()
        .map(|name| {
            boundary_flux(
                binding.mesh.mesh(),
                solution.velocity().vertex_values(),
                binding.entities(name)?,
            )
            .map(|flux| (name.to_owned(), flux))
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    solution.with_named_boundary_fluxes(named_boundary_fluxes)
}

/// Finalize the private profile-owned Stokes binding after exhaustive Model replay.
///
/// `transport` is the already existing MINI essential callback shape.  Its
/// values are compared exactly with every retained Model-owned boundary value
/// before any value is admitted to assembly.
// The accepted E1 contract owns this finalized boundary/system step, whose
// first consumer is the stopped successor optimization path. Remove this
// attribute when that path calls it.
#[allow(dead_code)]
pub(super) fn finalize_resolved_stokes_dissipation_profile_mini_2d_with_transport<B>(
    program: &KernelProgram,
    resolved: &ResolvedFieldwiseRealization,
    binding: &StokesDissipationGeometryModelBinding2d,
    transport: &B,
) -> Result<FinalizedSteadyStokesMini2dProblem, Diagnostic>
where
    B: Fn([f64; DIMENSION]) -> Result<[f64; DIMENSION], Diagnostic> + Sync,
{
    if program != binding.program() {
        return Err(invalid(
            "profile binding was admitted from another exact Model program",
        ));
    }
    binding.revalidate()?;
    let model = binding.model();
    let physical = binding.mesh().mesh();
    let mesh_reference = binding.mesh().artifact_reference()?;
    let scales = resolved_stokes_scales(program, resolved, mesh_reference, model)?;
    let normalized = normalize_geometry_mesh(model.bounds(), physical, scales.length_value())?;
    let geometry_boundary =
        dissipation_profile_boundary(model, binding, physical, &normalized, scales)?;
    let carried = validate_model_owned_essential_transport(
        &normalized,
        &geometry_boundary.fixed_velocity,
        transport,
    )?;
    let essential_velocity = |coordinate: [f64; DIMENSION]| {
        carried
            .get(&(coordinate[0].to_bits(), coordinate[1].to_bits()))
            .copied()
            .ok_or_else(|| {
                invalid("an essential profile vertex is absent from the validated Model replay")
            })
    };
    finalize_lowered_steady_stokes_mini_2d_with_assembly(
        resolved,
        mesh_reference,
        model,
        physical,
        &normalized,
        &geometry_boundary.boundary,
        scales,
        &essential_velocity,
        &REFERENCE_ASSEMBLY_BACKEND,
    )
}

fn dissipation_profile_boundary(
    model: &SteadyIncompressibleStokesModel2d,
    binding: &StokesDissipationGeometryModelBinding2d,
    physical: &SimplicialMesh,
    normalized: &SimplicialMesh,
    scales: SteadyStokesScaleProfile2d,
) -> Result<GeometryBoundary2d, Diagnostic> {
    let roles = [
        "body",
        "outer_x_lower",
        "outer_x_upper",
        "outer_y_lower",
        "outer_y_upper",
    ];
    let mut facet_owners = BTreeMap::new();
    let mut vertex_roles = BTreeMap::<usize, BTreeSet<&str>>::new();
    let mut fixed_velocity = vec![None; physical.vertices().len()];
    for role in roles {
        let key = StokesBoundaryKey2d::NamedEntitySet(role.to_owned());
        let entry = model
            .boundary_entry(&key)
            .ok_or_else(|| invalid(format!("profile Model omits exact Boundary `{role}`")))?;
        let body = role == "body";
        if (body && entry.disposition != PhysicalBoundaryDisposition::TraceZero)
            || (!body
                && !matches!(
                    entry.disposition,
                    PhysicalBoundaryDisposition::Prescribed(law)
                        if law.quantity() == PhysicalBoundaryQuantity::Trace
                ))
        {
            return Err(invalid(format!(
                "profile Boundary `{role}` does not retain its exact trace disposition"
            )));
        }
        for &facet in binding.entities(role)? {
            if facet.dimension() != DIMENSION - 1
                || !physical
                    .is_boundary_entity(facet)
                    .is_some_and(|boundary| boundary)
                || facet_owners.insert(facet, role).is_some()
            {
                return Err(invalid(
                    "profile boundary membership overlaps or contains a non-boundary facet",
                ));
            }
            let outward = facet_outward_unit_normal(physical, facet)?;
            for vertex in physical
                .entity_vertices(facet)
                .expect("accepted profile facet owns vertices")
            {
                let index = vertex.index();
                vertex_roles.entry(index).or_default().insert(role);
                let value = if body {
                    [0.0; DIMENSION]
                } else {
                    model
                        .prescribed_velocity(&key, Some(outward), &physical.vertices()[index])?
                        .ok_or_else(|| {
                            invalid(format!(
                                "profile Boundary `{role}` has no retained complete trace"
                            ))
                        })?
                }
                .map(|component| component / scales.velocity_value());
                if fixed_velocity[index].is_some_and(|existing| existing != value) {
                    return Err(invalid(
                        "Model-owned profile traces disagree at a shared essential vertex",
                    ));
                }
                fixed_velocity[index] = Some(value);
            }
        }
    }
    if vertex_roles
        .values()
        .any(|owned| owned.contains("body") && owned.iter().any(|role| role.starts_with("outer_")))
    {
        return Err(invalid(
            "profile body and outer roles share an essential mesh vertex",
        ));
    }
    let boundary_facets = (0..physical.entity_count(DIMENSION - 1).expect("2D facets"))
        .map(|index| MeshEntity::new(DIMENSION - 1, index))
        .filter(|facet| {
            physical
                .is_boundary_entity(*facet)
                .expect("mesh owns every facet")
        })
        .collect::<BTreeSet<_>>();
    if facet_owners.keys().copied().collect::<BTreeSet<_>>() != boundary_facets
        || (0..physical.vertices().len()).any(|index| {
            physical
                .is_boundary_entity(MeshEntity::new(0, index))
                .expect("mesh owns every vertex")
                != fixed_velocity[index].is_some()
        })
    {
        return Err(invalid(
            "profile boundary roles do not exhaust the exact boundary exactly once",
        ));
    }
    let facets = facet_owners
        .keys()
        .copied()
        .map(|facet| {
            SimplicialMiniStokesBoundaryFacet2d::new(
                facet,
                SimplicialMiniStokesBoundaryCondition2d::EssentialVelocity,
            )
        })
        .collect::<Vec<_>>();
    let boundary = SimplicialMiniStokesBoundary2d::new(normalized, facets)
        .map_err(|error| invalid(error.message()))?;
    Ok(GeometryBoundary2d {
        boundary,
        fixed_velocity,
    })
}

pub(super) fn require_stokes_dissipation_mesh_predicates(
    profile: &StokesDissipationProfileGeometry2d,
    mesh: &SimplicialMesh,
    reference: &SimplicialMesh,
    sectors: usize,
    minimum_signed_area_m2: f64,
    minimum_clearance_radius_multiple: f64,
    coordinate_tolerance_m: f64,
) -> Result<(), Diagnostic> {
    if mesh.vertices().len() != reference.vertices().len() {
        return Err(invalid(
            "realized topology has a different vertex inventory",
        ));
    }
    for angle_index in 0..sectors {
        let angle = std::f64::consts::TAU * angle_index as f64 / sectors as f64;
        let radius = profile.radius(angle)?;
        let expected_body = [radius * angle.cos(), radius * angle.sin()];
        if mesh.vertices()[angle_index]
            .iter()
            .zip(expected_body)
            .any(|(actual, expected)| (actual - expected).abs() > coordinate_tolerance_m)
            || mesh.vertices()[mesh.vertices().len() - sectors + angle_index]
                != reference.vertices()[reference.vertices().len() - sectors + angle_index]
        {
            return Err(invalid(
                "realized body samples or fixed outer vertices differ from exact ownership",
            ));
        }
    }
    let minimum_signed_area = mesh
        .cells()
        .iter()
        .map(|cell| {
            let (a, b, c) = (
                &mesh.vertices()[cell[0]],
                &mesh.vertices()[cell[1]],
                &mesh.vertices()[cell[2]],
            );
            0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        })
        .fold(f64::INFINITY, f64::min);
    let half_width = 10.0 * profile.area_radius_m();
    let clearance = mesh.vertices()[..sectors]
        .iter()
        .map(|point| half_width - point[0].abs().max(point[1].abs()))
        .fold(f64::INFINITY, f64::min);
    if !minimum_signed_area.is_finite()
        || minimum_signed_area <= minimum_signed_area_m2
        || !clearance.is_finite()
        || clearance <= minimum_clearance_radius_multiple * profile.area_radius_m()
    {
        return Err(invalid(
            "realized topology violates signed-area or body-clearance predicate",
        ));
    }
    Ok(())
}

fn boundary_flux(
    mesh: &SimplicialMesh,
    velocity: &[[f64; DIMENSION]],
    facets: &[MeshEntity],
) -> Result<f64, Diagnostic> {
    if velocity.len() != mesh.vertices().len() {
        return Err(invalid(
            "named Stokes boundary flux requires one velocity per mesh vertex",
        ));
    }
    if facets.is_empty() {
        return Err(invalid(
            "named Stokes boundary flux requires at least one facet",
        ));
    }
    let mut total = 0.0;
    for &facet in facets {
        let vertices = mesh
            .entity_vertices(facet)
            .filter(|vertices| {
                facet.dimension() == DIMENSION - 1
                    && vertices.len() == DIMENSION
                    && vertices.iter().all(|vertex| vertex.dimension() == 0)
            })
            .ok_or_else(|| invalid("named Stokes flux set contains a non-edge entity"))?;
        let [first, second] = [vertices[0].index(), vertices[1].index()];
        let first_coordinate = &mesh.vertices()[first];
        let second_coordinate = &mesh.vertices()[second];
        let length = (second_coordinate[0] - first_coordinate[0])
            .hypot(second_coordinate[1] - first_coordinate[1]);
        let normal = facet_outward_unit_normal(mesh, facet)?;
        let average_velocity = [
            0.5 * (velocity[first][0] + velocity[second][0]),
            0.5 * (velocity[first][1] + velocity[second][1]),
        ];
        total += length * (average_velocity[0] * normal[0] + average_velocity[1] * normal[1]);
    }
    if !total.is_finite() {
        return Err(invalid(
            "named Stokes boundary flux reconstruction is non-finite",
        ));
    }
    Ok(total)
}

pub(super) fn normalize_geometry_mesh(
    bounds: &[[f64; 2]; DIMENSION],
    mesh: &SimplicialMesh,
    length: f64,
) -> Result<SimplicialMesh, Diagnostic> {
    if mesh.topological_dimension() != DIMENSION {
        return Err(invalid(
            "geometry-backed coherent-SI MINI flow requires an intrinsic 2D mesh",
        ));
    }
    let vertices = mesh
        .vertices()
        .iter()
        .map(|coordinate| {
            if coordinate.len() != DIMENSION
                || coordinate
                    .iter()
                    .enumerate()
                    .any(|(axis, value)| *value < bounds[axis][0] || *value > bounds[axis][1])
            {
                return Err(invalid(
                    "geometry-backed flow mesh has a vertex outside the exact source bounds",
                ));
            }
            Ok(vec![
                (coordinate[0] - bounds[0][0]) / length,
                (coordinate[1] - bounds[1][0]) / length,
            ])
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    SimplicialMesh::new(
        DIMENSION,
        vertices,
        mesh.cells().to_vec(),
        mesh.quality_gate(),
    )
    .map_err(|error| invalid(error.message()))
}

fn geometry_boundary(
    model: &SteadyIncompressibleStokesModel2d,
    binding: &SteadyStokesGeometryBinding2d,
    normalized: &SimplicialMesh,
    scales: SteadyStokesScaleProfile2d,
) -> Result<GeometryBoundary2d, Diagnostic> {
    let mut facet_owners = BTreeMap::new();
    let mut fixed_velocity = vec![None; normalized.vertices().len()];
    let physical = binding.mesh.mesh();
    for (key, entry) in model.boundary_entries() {
        let StokesBoundaryKey2d::NamedEntitySet(name) = key else {
            return Err(invalid(
                "geometry-backed Stokes model contains a Cartesian boundary key",
            ));
        };
        let facets = binding.entities(name)?;
        for facet in facets {
            if facet.dimension() != DIMENSION - 1
                || facet_owners.insert(*facet, key.clone()).is_some()
            {
                return Err(invalid(
                    "correspondence-derived Stokes boundary partition overlaps or contains a non-facet",
                ));
            }
        }
        let essential = entry.disposition == PhysicalBoundaryDisposition::TraceZero
            || matches!(
                entry.disposition,
                PhysicalBoundaryDisposition::Prescribed(law)
                    if law.quantity() == PhysicalBoundaryQuantity::Trace
            );
        if !essential {
            continue;
        }
        let outward = match entry.disposition {
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
        for facet in facets {
            for vertex in physical
                .entity_vertices(*facet)
                .expect("validated correspondence facet owns vertices")
            {
                let index = vertex.index();
                let coordinate = &physical.vertices()[index];
                let value = if let Some(outward) = outward {
                    model
                        .prescribed_normal_velocity(key, outward, coordinate)?
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

    let facets = facet_owners
        .iter()
        .map(|(facet, key)| {
            let entry = model
                .boundary_entry(key)
                .expect("facet owner came from the exact model inventory");
            let condition = match entry.disposition {
                PhysicalBoundaryDisposition::TraceZero => {
                    SimplicialMiniStokesBoundaryCondition2d::EssentialVelocity
                }
                PhysicalBoundaryDisposition::Prescribed(law)
                    if law.quantity() == PhysicalBoundaryQuantity::Trace =>
                {
                    SimplicialMiniStokesBoundaryCondition2d::EssentialVelocity
                }
                PhysicalBoundaryDisposition::FluxZero
                | PhysicalBoundaryDisposition::Prescribed(_) => {
                    let pressure = model
                        .normal_pressure_for(key)
                        .ok_or_else(|| {
                            invalid(format!(
                                "traction boundary {key:?} has no retained normal-pressure law"
                            ))
                        })?
                        .expression()
                        .constant_value()
                        .ok_or_else(|| {
                            invalid(
                                "geometry-backed Stokes admits only constant normal pressure",
                            )
                        })?;
                    let outward = facet_outward_unit_normal(normalized, *facet)?;
                    SimplicialMiniStokesBoundaryCondition2d::ConstantTraction {
                        value: outward.map(|normal| -pressure * normal / scales.pressure_value()),
                    }
                }
                PhysicalBoundaryDisposition::PortBinding { connection, port } => {
                    return Err(invalid(format!(
                        "live Stokes PortBinding {connection} through Port {port} requires an explicit trace-space interface Realization"
                    )));
                }
            };
            Ok(SimplicialMiniStokesBoundaryFacet2d::new(
                *facet, condition,
            ))
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    let boundary = SimplicialMiniStokesBoundary2d::new(normalized, facets)
        .map_err(|error| invalid(error.message()))?
        .with_named_reaction_surface(
            normalized,
            "cylinder",
            binding.entities("cylinder")?.iter().copied(),
        )
        .map_err(|error| invalid(error.message()))?;
    Ok(GeometryBoundary2d {
        boundary,
        fixed_velocity,
    })
}

fn facet_outward_unit_normal(
    mesh: &SimplicialMesh,
    facet: MeshEntity,
) -> Result<[f64; DIMENSION], Diagnostic> {
    let vertices = mesh
        .entity_vertices(facet)
        .ok_or_else(|| invalid("correspondence facet is absent from the bound mesh"))?;
    let adjacent = mesh
        .incidence(facet, DIMENSION)
        .ok_or_else(|| invalid("correspondence facet has no cell incidence"))?;
    let ([left, right], [cell]) = (vertices.as_slice(), adjacent.as_slice()) else {
        return Err(invalid(
            "boundary facet requires two vertices and exactly one adjacent fluid cell",
        ));
    };
    let a = &mesh.vertices()[left.index()];
    let b = &mesh.vertices()[right.index()];
    let tangent = [b[0] - a[0], b[1] - a[1]];
    let length = tangent[0].hypot(tangent[1]);
    if !length.is_finite() || length <= 0.0 {
        return Err(invalid("boundary facet has non-positive finite length"));
    }
    let cell_vertices = mesh
        .entity_vertices(cell.entity)
        .expect("validated adjacent cell owns vertices");
    let mut centroid = [0.0; DIMENSION];
    for vertex in cell_vertices {
        centroid[0] += mesh.vertices()[vertex.index()][0] / 3.0;
        centroid[1] += mesh.vertices()[vertex.index()][1] / 3.0;
    }
    let midpoint = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
    let mut normal = [tangent[1] / length, -tangent[0] / length];
    if normal[0] * (midpoint[0] - centroid[0]) + normal[1] * (midpoint[1] - centroid[1]) < 0.0 {
        normal = [-normal[0], -normal[1]];
    }
    Ok(normal)
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}
