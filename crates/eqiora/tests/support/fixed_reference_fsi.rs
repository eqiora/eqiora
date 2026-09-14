use std::num::NonZeroUsize;

use eqiora::api::ModelDocument;
use eqiora::artifact::{
    ExecutionProvenanceV1, ExecutionTopologyV1, GeometryIdentityEnvelopeV1,
    GeometryMeshCorrespondenceEnvelopeV1, LayoutArtifacts, ModelEnvelope, RealizationEnvelopeV8,
    RunManifestV2, SimplicialMeshEnvelopeV1,
};
use eqiora::meshing::{CellId, FacetId, MeshQualityGate, SimplicialMesh};
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageDependencyV1, PackageManifestV1, PackageReleaseV1, PackageSourcesV1,
    PackagedModelDocument, QualifiedName, ResolutionRecordV1, SourceFileV1,
    prepare_package_release_v1,
};
use eqiora::realization::{
    CoupledFieldwiseRealizationRequest, MeshArtifactReference, RealizationCapabilities,
    RealizationRevision, ResolvedCoupledFieldwiseRealization, SemanticRevision,
    resolve_coupled_fieldwise,
};
use eqiora::solver::{
    CanonicalCsrAgreementFingerprintV1, LinearSolver, PreconditionerPolicy,
    REFERENCE_LINEAR_SOLVER, ReductionPolicy, SolverPlan,
};
use eqiora::{DimExponents, DynQuantity};
use eqiora_numerics::{
    fsi::FixedReferenceFsiCartesianModel2d, fsi::FixedReferenceFsiPartition,
    fsi::FixedReferenceFsiScaleProfile2d, fsi::FixedReferenceFsiState,
    fsi::ResolvedFixedReferenceFsiSolution2d, fsi::finalize_resolved_fixed_reference_fsi_step_2d,
    fsi::fixed_reference_fsi_plan_2d, fsi::fixed_reference_fsi_requirements_2d,
};

#[path = "embedded_package.rs"]
mod embedded_package;
mod operator;

pub(crate) const DIRECT: &str =
    include_str!("../../../../verify/fsi/fixed-reference-monolithic-step-2d/models/direct.eqi");
pub(crate) const PACKAGED: &str =
    include_str!("../../../../verify/fsi/fixed-reference-monolithic-step-2d/models/packaged.eqi");

const ROOT_PACKAGE: &str = "org.eqiora.verify.fixed_reference_monolithic_fsi_step_2d";
const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const TIME: DimExponents =
    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension");
const VELOCITY: DimExponents =
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
const PRESSURE: DimExponents =
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");

pub(crate) struct SpatialContext {
    pub(crate) model: ModelEnvelope,
    pub(crate) mesh: SimplicialMesh,
    pub(crate) mesh_artifact: SimplicialMeshEnvelopeV1,
    pub(crate) geometry: GeometryIdentityEnvelopeV1,
    pub(crate) correspondence: GeometryMeshCorrespondenceEnvelopeV1,
    pub(crate) partition: FixedReferenceFsiPartition<2>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SpatialObservation {
    pub(crate) fluid_cells: Vec<usize>,
    pub(crate) solid_cells: Vec<usize>,
    pub(crate) interface_facets: Vec<usize>,
}

impl SpatialContext {
    pub(crate) fn observation(&self) -> SpatialObservation {
        SpatialObservation {
            fluid_cells: self
                .partition
                .fluid_cells()
                .iter()
                .map(|cell| cell.index())
                .collect(),
            solid_cells: self
                .partition
                .solid_cells()
                .iter()
                .map(|cell| cell.index())
                .collect(),
            interface_facets: self
                .partition
                .interface_facets()
                .iter()
                .map(|facet| facet.index())
                .collect(),
        }
    }
}

pub(crate) struct ExecutionContext {
    pub(crate) mesh_reference: MeshArtifactReference,
    pub(crate) resolved: ResolvedCoupledFieldwiseRealization,
    pub(crate) realization: RealizationEnvelopeV8,
    pub(crate) run: RunManifestV2,
}

pub(crate) struct ExecutionWitness {
    pub(crate) physical_operator: operator::PhysicalOperator,
    pub(crate) operator: CanonicalCsrAgreementFingerprintV1,
    pub(crate) replayed_operator: CanonicalCsrAgreementFingerprintV1,
    pub(crate) solution: ResolvedFixedReferenceFsiSolution2d,
}

pub(crate) fn direct_document() -> ModelDocument {
    eqiora::api::ModelDocument::compile("direct.eqi", DIRECT)
        .expect("direct inertial-fluid/dynamic-solid Model compiles")
}

pub(crate) fn exact_spatial_witness(
    program: &eqiora::sem::KernelProgram,
    canonical: &FixedReferenceFsiCartesianModel2d,
) -> SpatialObservation {
    spatial_context(program, canonical).observation()
}

pub(crate) fn spatial_context(
    program: &eqiora::sem::KernelProgram,
    canonical: &FixedReferenceFsiCartesianModel2d,
) -> SpatialContext {
    let model = ModelEnvelope::from_program(program).expect("canonical current FSI Model");
    let mesh = physical_mesh();
    let mesh_artifact = SimplicialMeshEnvelopeV1::from_mesh(&mesh).expect("exact mesh artifact");
    let fluid = canonical
        .fluid()
        .domain()
        .downcast::<eqiora::kinds::Domain>()
        .expect("fluid Domain identity");
    let solid = canonical
        .solid()
        .continuum()
        .domain()
        .downcast::<eqiora::kinds::Domain>()
        .expect("solid Domain identity");
    let geometry = GeometryIdentityEnvelopeV1::new(&model, [solid, fluid], 1.0e-12)
        .expect("exact two-body geometry identity");
    let correspondence =
        GeometryMeshCorrespondenceEnvelopeV1::new(&geometry, &model, &mesh_artifact)
            .expect("exact body/facet correspondence");
    let connection = canonical
        .interface()
        .connection()
        .downcast::<eqiora::kinds::Connection>()
        .expect("conserving Connection identity");
    let interface = correspondence
        .derive_conserving_interface(&geometry, &model, &mesh_artifact, connection)
        .expect("content-bound exact interface witness");
    assert_eq!(interface.connection(), connection);
    assert_eq!(interface.facet_indices().len(), 2);
    assert_eq!(
        interface
            .parents()
            .into_iter()
            .map(|parent| parent.ulid())
            .collect::<std::collections::BTreeSet<_>>(),
        [fluid, solid]
            .into_iter()
            .map(|parent| parent.ulid())
            .collect()
    );
    assert_eq!(interface.model_artifact(), &model.digest().unwrap());
    assert_eq!(interface.geometry_artifact(), &geometry.digest().unwrap());
    assert_eq!(interface.mesh_artifact(), &mesh_artifact.digest().unwrap());
    assert_eq!(
        interface.correspondence_artifact(),
        &correspondence.digest().unwrap()
    );
    let fluid_cells = correspondence
        .body_cells(fluid)
        .expect("complete fluid cell set")
        .into_iter()
        .map(CellId::new)
        .collect::<Vec<_>>();
    let solid_cells = correspondence
        .body_cells(solid)
        .expect("complete solid cell set")
        .into_iter()
        .map(CellId::new)
        .collect::<Vec<_>>();
    let interface_facets = interface
        .facet_indices()
        .iter()
        .copied()
        .map(FacetId::new)
        .collect::<Vec<_>>();
    assert!(
        FixedReferenceFsiPartition::<2>::new(
            &mesh,
            fluid_cells.clone(),
            solid_cells.clone(),
            interface_facets[..1].to_vec(),
        )
        .is_err(),
        "a partial semantic interface must not reach assembly"
    );
    let partition =
        FixedReferenceFsiPartition::<2>::new(&mesh, fluid_cells, solid_cells, interface_facets)
            .expect("exact correspondence defines one complete FSI partition");
    SpatialContext {
        model,
        mesh,
        mesh_artifact,
        geometry,
        correspondence,
        partition,
    }
}

pub(crate) fn execution_context(
    program: &eqiora::sem::KernelProgram,
    canonical: &FixedReferenceFsiCartesianModel2d,
    spatial: &SpatialContext,
) -> ExecutionContext {
    let mesh_reference = MeshArtifactReference::from_sha256(
        spatial
            .mesh_artifact
            .digest()
            .expect("mesh digest")
            .sha256_bytes(),
    );
    let plan = fixed_reference_fsi_plan_2d(
        canonical,
        mesh_reference,
        DynQuantity::new(0.05, TIME),
        FixedReferenceFsiScaleProfile2d::new(
            DynQuantity::new(2.0, LENGTH),
            DynQuantity::new(0.5, VELOCITY),
            DynQuantity::new(4.0, PRESSURE),
        )
        .expect("positive coherent-SI FSI scales"),
        reference_solver(),
    )
    .expect("exact FSI plan");
    let resolved = resolve_coupled_fieldwise(
        &CoupledFieldwiseRealizationRequest::explicit(
            program.model(),
            SemanticRevision::new(canonical.semantic_revision()),
            RealizationRevision::new(1),
            plan,
        ),
        fixed_reference_fsi_requirements_2d(canonical),
        &RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
    )
    .expect("reference coupled capability resolves exact FSI plan");
    let realization = RealizationEnvelopeV8::from_resolved(
        &spatial.model,
        &resolved,
        LayoutArtifacts::Replicated,
    )
    .expect("content-bound multi-Domain Realization v3");
    realization
        .validate_model_artifact(&spatial.model)
        .expect("Realization replays the exact Model");
    realization
        .validate_mesh_artifact(&spatial.mesh_artifact)
        .expect("Realization replays the exact mesh");
    let run = RunManifestV2::new(
        &realization,
        ExecutionProvenanceV1::new(
            "eqiora.host.serial",
            env!("CARGO_PKG_VERSION"),
            "eqiora.reference",
            env!("CARGO_PKG_VERSION"),
            ExecutionTopologyV1::Host {
                workers: NonZeroUsize::MIN,
            },
            ReductionPolicy::Reproducible,
        )
        .expect("reference CPU execution provenance"),
    )
    .expect("Run v2 binds exact Model/Realization inputs");
    run.validate_against(&realization)
        .expect("Run replays against the exact Realization");
    assert_eq!(run.model(), spatial.model.digest().unwrap());
    assert_eq!(run.realization(), realization.digest().unwrap());
    assert_eq!(
        spatial.correspondence.geometry_artifact(),
        spatial.geometry.digest().unwrap()
    );
    assert_eq!(
        spatial.correspondence.mesh_artifact(),
        spatial.mesh_artifact.digest().unwrap()
    );
    ExecutionContext {
        mesh_reference,
        resolved,
        realization,
        run,
    }
}

pub(crate) fn prestrained_state(spatial: &SpatialContext) -> FixedReferenceFsiState<2> {
    let mut displacement = vec![[0.0; 2]; spatial.mesh.vertices().len()];
    let interface_midpoint = spatial
        .mesh
        .vertices()
        .iter()
        .position(|point| point.as_slice() == [1.0, 0.5])
        .expect("fixture owns one free interface midpoint");
    displacement[interface_midpoint] = [0.02, 0.0];
    FixedReferenceFsiState::<2>::new(
        &spatial.mesh,
        &spatial.partition,
        vec![[0.0; 2]; spatial.mesh.vertices().len()],
        spatial
            .partition
            .fluid_cells()
            .iter()
            .copied()
            .map(|cell| (cell, [0.0; 2]))
            .collect(),
        displacement,
    )
    .expect("finite prestrained previous state")
}

pub(crate) fn state_from_solution(
    spatial: &SpatialContext,
    solution: &ResolvedFixedReferenceFsiSolution2d,
) -> FixedReferenceFsiState<2> {
    FixedReferenceFsiState::<2>::new(
        &spatial.mesh,
        &spatial.partition,
        solution.vertex_velocity_coefficients().to_vec(),
        solution.fluid_velocity_bubble_coefficients().clone(),
        solution.solid_displacement_coefficients().to_vec(),
    )
    .expect("accepted solution re-enters the exact next-step state contract")
}

pub(crate) fn solve_step(
    canonical: &FixedReferenceFsiCartesianModel2d,
    spatial: &SpatialContext,
    execution: &ExecutionContext,
    previous: &FixedReferenceFsiState<2>,
) -> ExecutionWitness {
    let finalized = finalize_resolved_fixed_reference_fsi_step_2d(
        canonical,
        &execution.resolved,
        execution.mesh_reference,
        &spatial.mesh,
        &spatial.partition,
        previous,
    )
    .expect("exact content-bound FSI inputs finalize");
    let operator = finalized.linear_system().agreement_fingerprint();
    let physical_operator = operator::PhysicalOperator::capture(&finalized, spatial);
    let replayed = finalize_resolved_fixed_reference_fsi_step_2d(
        canonical,
        &execution.resolved,
        execution.mesh_reference,
        &spatial.mesh,
        &spatial.partition,
        previous,
    )
    .expect("identical finalization replays");
    let replayed_operator = replayed.linear_system().agreement_fingerprint();
    let solution = replayed
        .solve(&REFERENCE_LINEAR_SOLVER)
        .expect("reference MINRES solution passes independent FSI acceptance");
    assert_eq!(solution.model(), canonical.model());
    assert_eq!(
        solution.semantic_revision(),
        SemanticRevision::new(canonical.semantic_revision())
    );
    assert_eq!(solution.realization_revision(), RealizationRevision::new(1));
    assert_eq!(
        solution.fields().fluid_velocity(),
        canonical.fluid().velocity().downcast().unwrap()
    );
    assert_eq!(
        solution.fields().fluid_pressure(),
        canonical.fluid().pressure().downcast().unwrap()
    );
    assert_eq!(
        solution.fields().solid_velocity(),
        canonical.solid().velocity().downcast().unwrap()
    );
    assert_eq!(
        solution.fields().solid_displacement(),
        canonical
            .solid()
            .continuum()
            .displacement()
            .downcast()
            .unwrap()
    );
    ExecutionWitness {
        physical_operator,
        operator,
        replayed_operator,
        solution,
    }
}

pub(crate) fn execute_initial_step(
    program: &eqiora::sem::KernelProgram,
    canonical: &FixedReferenceFsiCartesianModel2d,
) -> ExecutionWitness {
    let spatial = spatial_context(program, canonical);
    let execution = execution_context(program, canonical, &spatial);
    solve_step(
        canonical,
        &spatial,
        &execution,
        &prestrained_state(&spatial),
    )
}

fn reference_solver() -> SolverPlan {
    SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(20_000).expect("20,000 is non-zero"),
    )
    .expect("MINRES plan")
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Reproducible)
}

fn physical_mesh() -> SimplicialMesh {
    SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 0.5],
            vec![1.0, 0.5],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 0.0],
            vec![2.0, 0.5],
            vec![2.0, 1.0],
        ],
        vec![
            vec![0, 1, 3],
            vec![0, 3, 2],
            vec![2, 3, 5],
            vec![2, 5, 4],
            vec![1, 6, 7],
            vec![1, 7, 3],
            vec![3, 7, 8],
            vec![3, 8, 5],
        ],
        MeshQualityGate::new(0.05).expect("quality gate"),
    )
    .expect("conforming two-body mesh")
}

pub(crate) fn packaged_document() -> PackagedModelDocument {
    let mechanics = public_release("Eqiora.Mechanics.Interfaces", &[]);
    let fluid = public_release(
        "Eqiora.Fluid.Incompressible",
        std::slice::from_ref(&mechanics),
    );
    let inertial = public_release("Eqiora.Fluid.InertialStokes", &[]);
    let solid = public_release(
        "Eqiora.Solid.LinearElasticity",
        std::slice::from_ref(&mechanics),
    );
    let dependencies = [
        ("inertial", &inertial),
        ("fluid", &fluid),
        ("solid", &solid),
        ("mechanics", &mechanics),
    ]
    .into_iter()
    .map(|(_, release)| {
        PackageDependencyV1::new(release.package_identity().expect("package identity"))
    })
    .collect();
    let source = format!(
        "import Eqiora.Fluid.InertialStokes.inertial_stokes as inertial;\n\
         import Eqiora.Fluid.Incompressible.incompressible as fluid;\n\
         import Eqiora.Solid.LinearElasticity.linear_elasticity as solid;\n\
         import Eqiora.Mechanics.Interfaces.interfaces as mechanics;\n{PACKAGED}"
    );
    let root = prepare_package_release_v1(
        inline_sources(ROOT_PACKAGE, "0.1.0", dependencies, &source),
        &[
            inertial.clone(),
            fluid.clone(),
            solid.clone(),
            mechanics.clone(),
        ],
    )
    .expect("prepare exact FSI verification root");
    let resolution = ResolutionRecordV1::from_exact_releases(
        &root,
        &[
            inertial.clone(),
            fluid.clone(),
            solid.clone(),
            mechanics.clone(),
        ],
    )
    .expect("resolve exact FSI package closure");
    let mut store = InMemoryPackageStore::default();
    for release in [&mechanics, &fluid, &inertial, &solid, &root] {
        store.insert(release).expect("install exact package");
    }
    PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .expect("compile exact FSI package graph offline")
}

fn public_release(package: &str, dependencies: &[PackageReleaseV1]) -> PackageReleaseV1 {
    let sources = embedded_package::public_sources(package);
    prepare_package_release_v1(sources, dependencies)
        .unwrap_or_else(|error| panic!("prepare public package {package}: {error:?}"))
}

fn inline_sources(
    name: &str,
    version: &str,
    dependencies: Vec<PackageDependencyV1>,
    model_source: &str,
) -> PackageSourcesV1 {
    let readme = NormalizedRelativePath::parse("README.md").expect("README path");
    let model = NormalizedRelativePath::parse("src/main.eqi").expect("model path");
    let manifest = PackageManifestV1::new(
        "main",
        QualifiedName::parse(name).expect("package name"),
        ExactVersion::parse(version).expect("exact version"),
        dependencies,
        vec![
            BundleEntryV1::new(readme.clone(), BundleRoleV1::Documentation),
            BundleEntryV1::new(model.clone(), BundleRoleV1::ModelSource),
        ],
    )
    .expect("package manifest");
    PackageSourcesV1::new(
        manifest,
        vec![
            SourceFileV1::new(
                readme,
                BundleRoleV1::Documentation,
                b"Exact fixed-reference FSI verification root.\n".to_vec(),
            ),
            SourceFileV1::new(
                model,
                BundleRoleV1::ModelSource,
                model_source.as_bytes().to_vec(),
            ),
        ],
    )
    .expect("closed author inventory")
}
