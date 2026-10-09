use std::num::NonZeroUsize;

use eqiora::artifact::{
    ArtifactDigest, ExecutionProvenanceV1, ExecutionTopologyV1, LayoutArtifacts, ModelEnvelope,
    RealizationEnvelopeV7, RunManifestV2, SimplicialMeshEnvelopeV1,
};
use eqiora::meshing::{MeshQualityGate, SimplicialMesh, triangle_duffy_gauss_legendre};
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageDependencyV1, PackageManifestV1, PackageReleaseV1, PackageSourcesV1,
    PackagedModelDocument, QualifiedName, ResolutionRecordV1, SourceFileV1,
    prepare_package_release_v1,
};
use eqiora::realization::{
    FieldwiseRealizationRequest, MeshArtifactReference, PlacementRequirementNode,
    RealizationCapabilities, RealizationRevision, ResolvedFieldwiseRealization, SemanticRevision,
    SolveRoot, resolve_fieldwise,
};
use eqiora::sem::KernelProgram;
use eqiora::solver::{
    CanonicalCsrSystemView, LinearSolverBackend, REFERENCE_LINEAR_SOLVER, ReductionPolicy,
    SolverPlan,
};
use eqiora::{Diagnostic, DimExponents, DynQuantity, Id, kinds};
use eqiora_numerics::{
    fluid::FinalizedSteadyStokesMini2dProblem, fluid::SteadyStokesMiniSolution2d,
    fluid::SteadyStokesScaleProfile2d, fluid::finalize_resolved_steady_stokes_mini_2d,
    fluid::finalize_simplicial_mini_stokes_2d,
    fluid::lower_steady_incompressible_stokes_cartesian_2d,
    fluid::steady_stokes_fieldwise_requirements_2d, fluid::steady_stokes_mini_plan_2d,
};

#[path = "support/embedded_package.rs"]
mod embedded_package;

const DIRECT: &str =
    include_str!("../../../verify/fluid/fieldwise-si-mini-stokes-2d/models/direct.eqi");
const PACKAGED: &str =
    include_str!("../../../verify/fluid/fieldwise-si-mini-stokes-2d/models/packaged.eqi");
const COMPONENT_README: &[u8] =
    include_bytes!("../../../verify/fluid/packaged-steady-stokes-2d/package-v0.1.0/README.md");
const COMPONENT_SOURCE: &[u8] =
    include_bytes!("../../../verify/fluid/packaged-steady-stokes-2d/models/component.eqi");
const ROOT_PACKAGE: &str = "org.eqiora.verify.fieldwise_si_mini_stokes_2d";
const VERSION: &str = "0.1.0";

const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const VELOCITY: DimExponents =
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
const PRESSURE: DimExponents =
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");

#[derive(Debug)]
struct Observation {
    realization_digest: ArtifactDigest,
    run_digest: ArtifactDigest,
    system: CanonicalCsrSystemView,
    solution: SteadyStokesMiniSolution2d,
}

#[test]
fn direct_and_packaged_models_share_one_fieldwise_si_mini_realization() {
    let direct = eqiora::api::ModelDocument::compile("direct.eqi", DIRECT)
        .expect("direct physical Stokes Model compiles");
    let packaged = packaged_document();
    let mesh =
        SimplicialMeshEnvelopeV1::from_mesh(&physical_mesh()).expect("physical mesh artifact");
    let mesh_bytes = mesh.canonical_json().expect("mesh bytes");
    let mesh = SimplicialMeshEnvelopeV1::from_json(&mesh_bytes, Default::default())
        .expect("mesh artifact replay");
    assert_eq!(mesh.canonical_json().unwrap(), mesh_bytes);

    let direct_a = observe(direct.program(), &mesh, profile_a(), 1);
    let direct_b = observe(direct.program(), &mesh, profile_b(), 2);
    let packaged_a = observe(packaged.model().program(), &mesh, profile_a(), 1);
    let packaged_b = observe(packaged.model().program(), &mesh, profile_b(), 2);

    assert_profile_pair(&direct_a, &direct_b);
    assert_profile_pair(&packaged_a, &packaged_b);
    assert_authoring_pair(&direct_a, &packaged_a);
    assert_authoring_pair(&direct_b, &packaged_b);
}

#[test]
fn equation_aware_adapter_rejects_generic_plan_artifact_and_mesh_drift() {
    let document = eqiora::api::ModelDocument::compile("direct.eqi", DIRECT).unwrap();
    let program = document.program();
    let mesh = SimplicialMeshEnvelopeV1::from_mesh(&physical_mesh()).unwrap();
    let mesh_reference = mesh.artifact_reference().unwrap();
    let (lowered, resolved) = resolve_exact(program, mesh_reference, profile_a(), 7);
    let model = ModelEnvelope::from_program(program).unwrap();
    let realization =
        RealizationEnvelopeV7::from_resolved(&model, &resolved, LayoutArtifacts::Replicated)
            .unwrap();

    for (length, velocity, pressure) in [
        (
            DynQuantity::new(0.0, LENGTH),
            DynQuantity::new(1.0, VELOCITY),
            DynQuantity::new(1.0, PRESSURE),
        ),
        (
            DynQuantity::new(f64::NAN, LENGTH),
            DynQuantity::new(1.0, VELOCITY),
            DynQuantity::new(1.0, PRESSURE),
        ),
        (
            DynQuantity::new(1.0, LENGTH),
            DynQuantity::new(-1.0, VELOCITY),
            DynQuantity::new(1.0, PRESSURE),
        ),
        (
            DynQuantity::new(1.0, LENGTH),
            DynQuantity::new(1.0, VELOCITY),
            DynQuantity::new(f64::INFINITY, PRESSURE),
        ),
        (
            DynQuantity::new(1.0, PRESSURE),
            DynQuantity::new(1.0, VELOCITY),
            DynQuantity::new(1.0, PRESSURE),
        ),
    ] {
        assert!(SteadyStokesScaleProfile2d::new(length, velocity, pressure).is_err());
    }
    let cg = SolverPlan::new(
        eqiora::solver::LinearSolver::ConjugateGradient,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap();
    assert!(steady_stokes_mini_plan_2d(&lowered, mesh_reference, profile_a(), cg).is_err());

    let bytes = realization.canonical_json().unwrap();
    let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

    let mut missing_binding = original.clone();
    missing_binding["plan"]["spatial"]["field_spaces"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(decode_and_resolve(&missing_binding).is_err());

    let mut duplicate_binding = original.clone();
    let duplicate = duplicate_binding["plan"]["spatial"]["field_spaces"][0].clone();
    duplicate_binding["plan"]["spatial"]["field_spaces"]
        .as_array_mut()
        .unwrap()
        .push(duplicate);
    assert!(decode_and_resolve(&duplicate_binding).is_err());

    let mut unknown_binding = original.clone();
    unknown_binding["plan"]["spatial"]["field_spaces"][0]["field_ulid"] =
        serde_json::json!(Id::<kinds::Field>::new().ulid().to_string());
    assert!(decode_and_resolve(&unknown_binding).is_err());

    let mut swapped_spaces = original.clone();
    let bindings = swapped_spaces["plan"]["spatial"]["field_spaces"]
        .as_array_mut()
        .unwrap();
    let first = bindings[0]["space"].clone();
    bindings[0]["space"] = bindings[1]["space"].clone();
    bindings[1]["space"] = first;
    let swapped = resolved_from_wire(&swapped_spaces);
    assert!(
        finalize_resolved_steady_stokes_mini_2d(program, &swapped, mesh_reference, mesh.mesh(),)
            .is_err()
    );

    let mut missing_constraint = original.clone();
    missing_constraint["plan"]["spatial"]["constraints"] = serde_json::json!([]);
    missing_constraint["plan"]["scaling"]["block_scales"]
        .as_array_mut()
        .unwrap()
        .retain(|entry| entry["block"]["kind"] != "constraint-multiplier");
    let missing_constraint = resolved_from_wire(&missing_constraint);
    assert_adapter_rejects(program, &missing_constraint, mesh_reference, mesh.mesh());

    let mut duplicate_constraint = original.clone();
    let duplicate = duplicate_constraint["plan"]["spatial"]["constraints"][0].clone();
    duplicate_constraint["plan"]["spatial"]["constraints"]
        .as_array_mut()
        .unwrap()
        .push(duplicate);
    assert!(decode_and_resolve(&duplicate_constraint).is_err());

    let mut force_as_unknown = original.clone();
    replace_string_value(
        &mut force_as_unknown,
        &lowered.pressure().ulid().to_string(),
        &lowered.force_potential().ulid().to_string(),
    );
    canonicalize_field_arrays(&mut force_as_unknown);
    let force_as_unknown = resolved_from_wire(&force_as_unknown);
    assert_adapter_rejects(program, &force_as_unknown, mesh_reference, mesh.mesh());

    let mut forged_gauge = original.clone();
    let blocks = forged_gauge["plan"]["scaling"]["block_scales"]
        .as_array_mut()
        .unwrap();
    let gauge = blocks
        .iter_mut()
        .find(|entry| entry["block"]["kind"] == "constraint-multiplier")
        .expect("one gauge block");
    gauge["scale"]["coherent_si_value"] = serde_json::json!(0.25);
    let forged = resolved_from_wire(&forged_gauge);
    assert!(
        finalize_resolved_steady_stokes_mini_2d(program, &forged, mesh_reference, mesh.mesh(),)
            .is_err()
    );

    let mut forged_functional = original.clone();
    forged_functional["plan"]["scaling"]["weak_functional_scale"]["coherent_si_value"] =
        serde_json::json!(3.0);
    let forged_functional = resolved_from_wire(&forged_functional);
    assert_adapter_rejects(program, &forged_functional, mesh_reference, mesh.mesh());

    let mut missing_scale = original.clone();
    missing_scale["plan"]["scaling"]["block_scales"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(decode_and_resolve(&missing_scale).is_err());

    let mut low_quadrature = original.clone();
    low_quadrature["plan"]["spatial"]["discretization"]["quadrature"]["points_per_axis"] =
        serde_json::json!(2);
    let low_quadrature = resolved_from_wire(&low_quadrature);
    assert_adapter_rejects(program, &low_quadrature, mesh_reference, mesh.mesh());

    let mut general = original.clone();
    general["plan"]["operator_properties"] = serde_json::json!("general");
    assert!(decode_and_resolve(&general).is_err());

    let mut jacobi = original.clone();
    jacobi["plan"]["solver"]["preconditioner"] = serde_json::json!("jacobi");
    assert!(decode_and_resolve(&jacobi).is_err());

    let mut alien_domain = original.clone();
    replace_string_value(
        &mut alien_domain,
        &lowered.domain().ulid().to_string(),
        &Id::<kinds::Domain>::new().ulid().to_string(),
    );
    let alien_domain = resolved_from_wire(&alien_domain);
    assert_adapter_rejects(program, &alien_domain, mesh_reference, mesh.mesh());

    let mut alien_model = original;
    alien_model["model_ulid"] = serde_json::json!(
        eqiora::ontology::OntologyId::<eqiora::ontology::Model>::new()
            .ulid()
            .to_string()
    );
    let alien_model = resolved_from_wire(&alien_model);
    assert_adapter_rejects(program, &alien_model, mesh_reference, mesh.mesh());

    let stale = resolve_fieldwise(
        &FieldwiseRealizationRequest::explicit(
            program.model(),
            SemanticRevision::new(program.revision().0 + 1),
            RealizationRevision::new(8),
            resolved.plan().clone(),
        ),
        steady_stokes_fieldwise_requirements_2d(&lowered),
        &RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
    )
    .unwrap();
    assert!(
        finalize_resolved_steady_stokes_mini_2d(program, &stale, mesh_reference, mesh.mesh(),)
            .is_err()
    );
    assert!(
        finalize_resolved_steady_stokes_mini_2d(
            program,
            &resolved,
            MeshArtifactReference::from_sha256([7; 32]),
            mesh.mesh(),
        )
        .is_err()
    );

    let drifted_mesh = SimplicialMesh::new(
        2,
        mesh.mesh()
            .vertices()
            .iter()
            .enumerate()
            .map(|(index, point)| {
                if index == 0 {
                    vec![-0.125, point[1]]
                } else {
                    point.clone()
                }
            })
            .collect(),
        mesh.mesh().cells().to_vec(),
        mesh.mesh().quality_gate(),
    )
    .unwrap();
    assert!(
        finalize_resolved_steady_stokes_mini_2d(program, &resolved, mesh_reference, &drifted_mesh,)
            .is_err()
    );
    let drifted_artifact = SimplicialMeshEnvelopeV1::from_mesh(&drifted_mesh).unwrap();
    assert!(
        realization
            .validate_mesh_artifact(&drifted_artifact)
            .is_err()
    );

    let disconnected = SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![4.0, 0.0],
            vec![4.0, 2.0],
            vec![0.0, 2.0],
            vec![0.0, 0.0],
            vec![4.0, 0.0],
            vec![4.0, 2.0],
            vec![0.0, 2.0],
        ],
        vec![vec![0, 1, 2], vec![0, 2, 3], vec![4, 5, 6], vec![4, 6, 7]],
        MeshQualityGate::new(0.4).unwrap(),
    )
    .unwrap();
    assert_adapter_rejects(program, &resolved, mesh_reference, &disconnected);

    let valid_run = RunManifestV2::new(&realization, execution_provenance(1, false)).unwrap();
    assert!(RunManifestV2::new(&realization, execution_provenance(2, false)).is_err());
    assert!(RunManifestV2::new(&realization, execution_provenance(1, true)).is_err());
    let (_, other_resolved) = resolve_exact(program, mesh_reference, profile_b(), 9);
    let other_realization =
        RealizationEnvelopeV7::from_resolved(&model, &other_resolved, LayoutArtifacts::Replicated)
            .unwrap();
    assert!(valid_run.validate_against(&other_realization).is_err());

    let wrong_shape = DIRECT.replace(
        "variable velocity: vector<m / s, 2> on body in h1;",
        "variable velocity: vector<1, 2> on body in h1;",
    );
    match eqiora::api::ModelDocument::compile("wrong-shape.eqi", &wrong_shape) {
        Err(diagnostics) => assert!(!diagnostics.is_empty()),
        Ok(document) => {
            assert!(lower_steady_incompressible_stokes_cartesian_2d(document.program()).is_err())
        }
    }
}

fn observe(
    program: &KernelProgram,
    mesh: &SimplicialMeshEnvelopeV1,
    scales: SteadyStokesScaleProfile2d,
    realization_revision: u64,
) -> Observation {
    let mesh_reference = mesh.artifact_reference().expect("mesh identity");
    let solver = reference_solver();
    let (lowered, resolved) = resolve_exact(program, mesh_reference, scales, realization_revision);

    let model = ModelEnvelope::from_program(program).expect("canonical current Model");
    let model_bytes = model.canonical_json().expect("Model bytes");
    let model_replay =
        ModelEnvelope::from_json(&model_bytes, Default::default()).expect("current Model replay");
    assert_eq!(model_replay.canonical_json().unwrap(), model_bytes);
    assert_eq!(model_replay.digest().unwrap(), model.digest().unwrap());
    let realization =
        RealizationEnvelopeV7::from_resolved(&model, &resolved, LayoutArtifacts::Replicated)
            .expect("field-wise Realization v2");
    let realization_bytes = realization.canonical_json().expect("Realization bytes");
    let graph = resolved
        .portable_graph()
        .expect("resolved field-wise plan has one lossless portable DAG");
    let SolveRoot::Linear(root) = graph.root() else {
        panic!("steady Stokes graph must have a linear root");
    };
    let linear = graph.linear_solve(root).expect("linear root exists");
    assert_eq!(graph.transformations(), []);
    assert_eq!(
        graph.placement(linear.placement()),
        Some(PlacementRequirementNode::HostWorkers {
            workers_per_partition: NonZeroUsize::MIN,
        })
    );
    let realization_replay =
        RealizationEnvelopeV7::from_json(&realization_bytes, Default::default())
            .expect("Realization v2 replay");
    assert_eq!(
        realization_replay.canonical_json().unwrap(),
        realization_bytes
    );
    assert_eq!(
        realization_replay.digest().unwrap(),
        realization.digest().unwrap()
    );
    realization
        .validate_model_artifact(&model)
        .expect("Realization retains the exact Model artifact");
    realization
        .validate_mesh_artifact(mesh)
        .expect("Realization retains the exact mesh artifact");

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
        .expect("typed execution provenance"),
    )
    .expect("run manifest matches the typed Realization");
    let run_bytes = run.canonical_json().expect("Run bytes");
    let run_replay =
        RunManifestV2::from_json(&run_bytes, Default::default()).expect("Run v2 replay");
    assert_eq!(run_replay.canonical_json().unwrap(), run_bytes);
    assert_eq!(run_replay.digest().unwrap(), run.digest().unwrap());
    run.validate_against(&realization)
        .expect("run/Realization linkage replays");

    let (_, finalized) =
        finalize_resolved_steady_stokes_mini_2d(program, &resolved, mesh_reference, mesh.mesh())
            .expect("validated artifact payload finalizes through the SI adapter");
    assert_exact_symmetry(finalized.canonical_csr_system_view());
    assert_congruence(&finalized, mesh.mesh(), scales);
    let system = finalized.canonical_csr_system_view().clone();
    let linear_solution = REFERENCE_LINEAR_SOLVER
        .solve(
            &finalized.linear_problem().expect("captured linear problem"),
            solver,
        )
        .expect("reference MINRES solves the dimensionless system");
    let solution = finalized
        .finish(linear_solution)
        .expect("exact solution is reaccepted and reconstructed");
    assert_physical_solution(&solution, &lowered);

    Observation {
        realization_digest: realization.digest().expect("Realization digest"),
        run_digest: run.digest().expect("Run digest"),
        system,
        solution,
    }
}

fn assert_congruence(
    dimensionless: &FinalizedSteadyStokesMini2dProblem,
    mesh: &SimplicialMesh,
    scales: SteadyStokesScaleProfile2d,
) {
    let quadrature = triangle_duffy_gauss_legendre(3).expect("degree-four positive rule");
    let physical = finalize_simplicial_mini_stokes_2d(
        mesh,
        6.0,
        &|_| Ok([0.75, 0.0]),
        &|_| Ok([0.0, 0.0]),
        &quadrature,
        reference_solver(),
        dimensionless.vector_layout(),
        eqiora::realization::Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
    )
    .expect("independent physical-system oracle");
    let physical = physical.canonical_csr_system_view();
    let normalized = dimensionless.canonical_csr_system_view();
    assert_eq!(normalized.row_offsets(), physical.row_offsets());
    assert_eq!(normalized.column_indices(), physical.column_indices());

    let pressure_start = normalized
        .rows()
        .checked_sub(mesh.vertices().len() + 1)
        .expect("mixed layout owns pressure and gauge blocks");
    let gauge = normalized.rows() - 1;
    let block_scale = |index: usize| {
        if index < pressure_start {
            scales.velocity().value()
        } else if index < gauge {
            scales.pressure().value()
        } else {
            scales.gauge().value()
        }
    };
    let theta = scales.weak_functional().value();
    for row in 0..normalized.rows() {
        for entry in normalized.row_offsets()[row]..normalized.row_offsets()[row + 1] {
            let column = normalized.column_indices()[entry];
            let expected =
                physical.values()[entry] * block_scale(row) * block_scale(column) / theta;
            assert_close(normalized.values()[entry], expected, 2.0e-13);
        }
        let expected_rhs = physical.right_hand_side()[row] * block_scale(row) / theta;
        assert_close(normalized.right_hand_side()[row], expected_rhs, 2.0e-13);
    }
}

fn assert_physical_solution(
    solution: &SteadyStokesMiniSolution2d,
    lowered: &eqiora_numerics::fluid::SteadyIncompressibleStokesCartesianModel2d,
) {
    assert_eq!(solution.velocity_field().ulid(), lowered.velocity().ulid());
    assert_eq!(solution.pressure_field().ulid(), lowered.pressure().ulid());
    assert_eq!(
        solution.force_potential_field().ulid(),
        lowered.force_potential().ulid()
    );
    for value in solution
        .velocity()
        .vertex_values()
        .iter()
        .chain(solution.velocity().cell_bubble_values())
        .flatten()
    {
        assert_close(*value, 0.0, 2.0e-10);
    }
    for (coordinates, pressure) in solution
        .pressure()
        .mesh()
        .vertices()
        .iter()
        .zip(solution.pressure().vertex_values())
    {
        assert_close(*pressure, 3.0 * (coordinates[0] / 4.0 - 0.5), 2.0e-10);
    }
    assert_close(
        solution
            .gauge_multiplier()
            .expect("all-essential pressure uses a zero-integral gauge"),
        0.0,
        2.0e-10,
    );
    assert_close(solution.pressure_integral(), 0.0, 2.0e-10);
    assert_close(solution.integrated_body_force()[0], 6.0, 2.0e-10);
    assert_close(solution.integrated_body_force()[1], 0.0, 2.0e-10);
    assert_close(solution.boundary_reaction()[0], -6.0, 2.0e-9);
    assert_close(solution.boundary_reaction()[1], 0.0, 2.0e-9);
    assert_eq!(solution.named_boundary_flux("inlet"), None);
    assert_eq!(solution.named_boundary_flux("outlet"), None);
    assert_eq!(solution.named_boundary_flux("cylinder"), None);
    assert_eq!(solution.named_boundary_flux("unknown"), None);
    assert!(
        solution
            .dimensionless_solution()
            .solve_report()
            .true_residual_norm()
            < 2.0e-10
    );
}

fn assert_profile_pair(a: &Observation, b: &Observation) {
    assert_ne!(a.realization_digest, b.realization_digest);
    assert_ne!(a.run_digest, b.run_digest);
    assert_eq!(a.system.row_offsets(), b.system.row_offsets());
    assert_eq!(a.system.column_indices(), b.system.column_indices());
    assert_eq!(a.system.values(), b.system.values());
    assert_ne!(a.system.right_hand_side(), b.system.right_hand_side());
    assert_physical_equivalence(&a.solution, &b.solution);
}

fn assert_authoring_pair(direct: &Observation, packaged: &Observation) {
    assert_eq!(direct.system, packaged.system);
    assert_physical_equivalence(&direct.solution, &packaged.solution);
}

fn assert_physical_equivalence(
    left: &SteadyStokesMiniSolution2d,
    right: &SteadyStokesMiniSolution2d,
) {
    assert_eq!(
        left.velocity().vertex_values(),
        right.velocity().vertex_values()
    );
    assert_eq!(
        left.velocity().cell_bubble_values(),
        right.velocity().cell_bubble_values()
    );
    assert_eq!(
        left.pressure().vertex_values(),
        right.pressure().vertex_values()
    );
    assert_eq!(left.gauge_multiplier(), right.gauge_multiplier());
    assert_eq!(left.boundary_reaction(), right.boundary_reaction());
    assert_eq!(left.integrated_body_force(), right.integrated_body_force());
    assert_eq!(left.pressure_integral(), right.pressure_integral());
}

fn resolve_exact(
    program: &KernelProgram,
    mesh: MeshArtifactReference,
    scales: SteadyStokesScaleProfile2d,
    realization_revision: u64,
) -> (
    eqiora_numerics::fluid::SteadyIncompressibleStokesCartesianModel2d,
    ResolvedFieldwiseRealization,
) {
    let lowered = lower_steady_incompressible_stokes_cartesian_2d(program)
        .expect("ordinary canonical Stokes lowerer accepts the authored Model");
    let plan = steady_stokes_mini_plan_2d(&lowered, mesh, scales, reference_solver())
        .expect("exact field-wise MINI plan");
    let request = FieldwiseRealizationRequest::explicit(
        program.model(),
        SemanticRevision::new(program.revision().0),
        RealizationRevision::new(realization_revision),
        plan,
    );
    let resolved = resolve_fieldwise(
        &request,
        steady_stokes_fieldwise_requirements_2d(&lowered),
        &RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
    )
    .expect("reference mixed capability resolves the exact plan");
    (lowered, resolved)
}

fn resolved_from_wire(value: &serde_json::Value) -> ResolvedFieldwiseRealization {
    decode_and_resolve(value)
        .unwrap_or_else(|error| panic!("generic mixed capability accepts the near-miss: {error}"))
}

fn decode_and_resolve(
    value: &serde_json::Value,
) -> Result<ResolvedFieldwiseRealization, Diagnostic> {
    let envelope =
        RealizationEnvelopeV7::from_json(&serde_json::to_vec(value).unwrap(), Default::default())?;
    resolve_fieldwise(
        &FieldwiseRealizationRequest::explicit(
            envelope.model()?,
            envelope.semantic_revision(),
            envelope.realization_revision(),
            envelope.plan()?,
        ),
        envelope.requirements()?,
        &RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
    )
}

fn assert_adapter_rejects(
    program: &KernelProgram,
    resolved: &ResolvedFieldwiseRealization,
    mesh_reference: MeshArtifactReference,
    mesh: &SimplicialMesh,
) {
    assert!(
        finalize_resolved_steady_stokes_mini_2d(program, resolved, mesh_reference, mesh).is_err()
    );
}

fn replace_string_value(value: &mut serde_json::Value, from: &str, to: &str) {
    match value {
        serde_json::Value::String(text) if text == from => *text = to.to_owned(),
        serde_json::Value::Array(values) => {
            for value in values {
                replace_string_value(value, from, to);
            }
        }
        serde_json::Value::Object(values) => {
            for value in values.values_mut() {
                replace_string_value(value, from, to);
            }
        }
        _ => {}
    }
}

fn canonicalize_field_arrays(value: &mut serde_json::Value) {
    value["requirements"]["unknown_field_ulids"]
        .as_array_mut()
        .unwrap()
        .sort_by_key(|entry| entry.as_str().unwrap().to_owned());
    for path in ["field_spaces", "constraints"] {
        value["plan"]["spatial"][path]
            .as_array_mut()
            .unwrap()
            .sort_by_key(|entry| entry["field_ulid"].as_str().unwrap().to_owned());
    }
    value["plan"]["scaling"]["block_scales"]
        .as_array_mut()
        .unwrap()
        .sort_by_key(|entry| {
            (
                usize::from(entry["block"]["kind"] == "constraint-multiplier"),
                entry["block"]["field_ulid"].as_str().unwrap().to_owned(),
            )
        });
}

fn execution_provenance(workers: usize, fast: bool) -> ExecutionProvenanceV1 {
    ExecutionProvenanceV1::new(
        "eqiora.host.serial",
        env!("CARGO_PKG_VERSION"),
        "eqiora.reference",
        env!("CARGO_PKG_VERSION"),
        ExecutionTopologyV1::Host {
            workers: NonZeroUsize::new(workers).unwrap(),
        },
        if fast {
            ReductionPolicy::Fast
        } else {
            ReductionPolicy::Reproducible
        },
    )
    .unwrap()
}

fn assert_exact_symmetry(system: &CanonicalCsrSystemView) {
    for row in 0..system.rows() {
        for entry in system.row_offsets()[row]..system.row_offsets()[row + 1] {
            let column = system.column_indices()[entry];
            let transpose = (system.row_offsets()[column]..system.row_offsets()[column + 1])
                .find(|candidate| system.column_indices()[*candidate] == row)
                .expect("symmetric sparsity owns the transpose entry");
            assert_eq!(
                system.values()[entry].to_bits(),
                system.values()[transpose].to_bits()
            );
        }
    }
}

fn profile_a() -> SteadyStokesScaleProfile2d {
    scale_profile(4.0, 0.5, 0.75)
}

fn profile_b() -> SteadyStokesScaleProfile2d {
    scale_profile(4.0, 1.0, 1.5)
}

fn scale_profile(length: f64, velocity: f64, pressure: f64) -> SteadyStokesScaleProfile2d {
    SteadyStokesScaleProfile2d::new(
        DynQuantity::new(length, LENGTH),
        DynQuantity::new(velocity, VELOCITY),
        DynQuantity::new(pressure, PRESSURE),
    )
    .expect("positive coherent-SI scale profile")
}

fn reference_solver() -> SolverPlan {
    SolverPlan::new(
        eqiora::solver::LinearSolver::MinimumResidual,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(10_000).expect("10,000 is non-zero"),
    )
    .expect("MINRES policy")
    .with_preconditioner(eqiora::solver::PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Reproducible)
}

fn physical_mesh() -> SimplicialMesh {
    let nx = 4;
    let ny = 2;
    let width = nx + 1;
    let vertices = (0..=ny)
        .flat_map(|j| (0..=nx).map(move |i| vec![i as f64, j as f64]))
        .collect::<Vec<_>>();
    let mut cells = Vec::with_capacity(2 * nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let lower_left = j * width + i;
            let lower_right = lower_left + 1;
            let upper_left = lower_left + width;
            let upper_right = upper_left + 1;
            cells.push(vec![lower_left, lower_right, upper_right]);
            cells.push(vec![lower_left, upper_right, upper_left]);
        }
    }
    SimplicialMesh::new(2, vertices, cells, MeshQualityGate::new(0.5).unwrap())
        .expect("connected 4 m by 2 m affine triangle mesh")
}

fn packaged_document() -> PackagedModelDocument {
    let component = component_release();
    let root = root_release(&component);
    let resolution =
        ResolutionRecordV1::from_exact_releases(&root, std::slice::from_ref(&component))
            .expect("exact package resolution");
    let mut store = InMemoryPackageStore::default();
    store.insert(&component).expect("install fluid package");
    store.insert(&root).expect("install verification root");
    PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .expect("exact package compiles offline")
}

fn component_release() -> PackageReleaseV1 {
    let sources = embedded_package::generated_sources(
        "Eqiora.Fluid.Incompressible",
        VERSION,
        &[
            ("README.md", BundleRoleV1::Documentation, COMPONENT_README),
            (
                "src/incompressible.eqi",
                BundleRoleV1::ModelSource,
                COMPONENT_SOURCE,
            ),
        ],
    );
    prepare_package_release_v1(sources, &[]).expect("prepare exact fluid release")
}

fn root_release(component: &PackageReleaseV1) -> PackageReleaseV1 {
    let readme = NormalizedRelativePath::parse("README.md").unwrap();
    let model = NormalizedRelativePath::parse("src/main.eqi").unwrap();
    let manifest = PackageManifestV1::new(
        "main",
        QualifiedName::parse(ROOT_PACKAGE).unwrap(),
        ExactVersion::parse(VERSION).unwrap(),
        vec![PackageDependencyV1::new(
            component.package_identity().unwrap(),
        )],
        vec![
            BundleEntryV1::new(readme.clone(), BundleRoleV1::Documentation),
            BundleEntryV1::new(model.clone(), BundleRoleV1::ModelSource),
        ],
    )
    .unwrap();
    let component_name = component
        .package_identity()
        .expect("fluid package identity")
        .name;
    let source = format!("import {component_name}.incompressible as fluid;\n{PACKAGED}");
    let sources = PackageSourcesV1::new(
        manifest,
        vec![
            SourceFileV1::new(
                readme,
                BundleRoleV1::Documentation,
                b"Field-wise SI MINI Stokes verification root.\n".to_vec(),
            ),
            SourceFileV1::new(model, BundleRoleV1::ModelSource, source.into_bytes()),
        ],
    )
    .unwrap();
    prepare_package_release_v1(sources, std::slice::from_ref(component))
        .expect("prepare exact verification root")
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected:.17e}, got {actual:.17e}, tolerance {tolerance:.3e}"
    );
}
