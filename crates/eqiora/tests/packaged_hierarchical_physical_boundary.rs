use std::num::NonZeroUsize;

use eqiora::Id;
use eqiora::artifact::{
    ArtifactDigest, PhysicalExposureCatalogEnvelopeV1, PhysicalExposureContractV1,
    PhysicalExposureDecoderLimits, PhysicalExposureObservationBindingV1,
    PhysicalExposureProjectionV1, PhysicalExposureQuantityV1, RunManifestV1,
};
use eqiora::compiler::identity::{
    DeclarationPath, ElaborationKey, IdentityNamespace, InstancePath,
};
use eqiora::diagnostic::codes;
use eqiora::entity::kinds;
use eqiora::kernel::KernelNode;
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, ModelPackageIdentityV1,
    NormalizedRelativePath, PackageDependencyV1, PackageManifestV1, PackageReleaseV1,
    PackageSourcesV1, PackagedModelDocument, QualifiedName, ResolutionRecordV1, SourceFileV1,
    prepare_package_release_v1,
};
use eqiora::sem::PhysicalUnknown;
use eqiora::solver::{
    LinearSolveRequest, LinearSolver, PreconditionerPolicy, ReductionPolicy, SolverPlan,
};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{
    scalar::ScalarPhysicalAffineProblem, scalar::ScalarPhysicalAffineSolution,
    scalar::lower_scalar_physical_affine, scalar::solve_scalar_physical_affine,
};

mod support;

use support::connection_set_conformance::{connection_containing, observe_connection_sets};
use support::exact_package;

const COMPONENTS: &str =
    include_str!("../../../verify/packages/hierarchical-physical-boundary/models/components.eqi");
const NARY: &str =
    include_str!("../../../verify/packages/hierarchical-physical-boundary/models/nary.eqi");
const PARTITIONED: &str =
    include_str!("../../../verify/packages/hierarchical-physical-boundary/models/partitioned.eqi");
const INVALID_UNCLOSED: &str = include_str!(
    "../../../verify/packages/hierarchical-physical-boundary/models/invalid-unclosed.eqi"
);
const VERSION: &str = "0.1.0";
const VALUE_TOLERANCE: f64 = 2.0e-11;
const RESIDUAL_TOLERANCE: f64 = 1.0e-11;

fn manifest(name: &str, dependencies: Vec<PackageDependencyV1>) -> PackageManifestV1 {
    PackageManifestV1::new(
        "model",
        QualifiedName::parse(name).expect("package name"),
        ExactVersion::parse(VERSION).expect("package version"),
        dependencies,
        vec![BundleEntryV1::new(
            NormalizedRelativePath::parse("src/model.eqi").expect("model path"),
            BundleRoleV1::ModelSource,
        )],
    )
    .expect("package manifest")
}

fn sources(manifest: PackageManifestV1, source: &str) -> PackageSourcesV1 {
    PackageSourcesV1::new(
        manifest,
        vec![SourceFileV1::new(
            NormalizedRelativePath::parse("src/model.eqi").expect("model path"),
            BundleRoleV1::ModelSource,
            source.as_bytes().to_vec(),
        )],
    )
    .expect("closed package sources")
}

fn components_release() -> PackageReleaseV1 {
    prepare_package_release_v1(
        sources(
            manifest("Eqiora.Verify.PhysicalComponents", vec![]),
            COMPONENTS,
        ),
        &[],
    )
    .expect("component package release")
}

fn root_sources(components: &PackageReleaseV1, source: &str) -> PackageSourcesV1 {
    let identity = components
        .package_identity()
        .expect("component package identity");
    let dependency = PackageDependencyV1::new(identity);
    let source = format!("import Eqiora.Verify.PhysicalComponents.model as components;\n{source}");
    sources(
        manifest(
            "org.eqiora.verify.hierarchical_physical_boundary",
            vec![dependency],
        ),
        &source,
    )
}

fn root_release(components: &PackageReleaseV1, source: &str) -> PackageReleaseV1 {
    prepare_package_release_v1(
        root_sources(components, source),
        std::slice::from_ref(components),
    )
    .expect("root package release")
}

fn compile_locked(
    components: &PackageReleaseV1,
    root: &PackageReleaseV1,
) -> (PackagedModelDocument, ResolutionRecordV1) {
    let (store, resolution) = exact_package::resolved_store(root, std::slice::from_ref(components));
    let packaged = PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .expect("compile exact package graph");
    packaged
        .compilation()
        .validate_against(&resolution)
        .expect("compilation matches exact resolution");
    (packaged, resolution)
}

fn selected_connection(
    packaged: &PackagedModelDocument,
    member: Id<kinds::Port>,
) -> Id<kinds::Connection> {
    connection_containing(packaged.model().program(), member.erase())
        .expect("selected Port belongs to one Connection")
        .connection
        .downcast()
        .expect("observed Connection has its declared entity kind")
}

fn lower_and_solve(
    packaged: &PackagedModelDocument,
) -> (ScalarPhysicalAffineProblem, ScalarPhysicalAffineSolution) {
    let positive = packaged.model().aliases()["load.element.positive"]
        .downcast()
        .expect("retained internal load Port");
    let problem = lower_scalar_physical_affine(
        packaged.model().program(),
        selected_connection(packaged, positive),
        None,
    )
    .expect("complete affine physical closure");
    assert_eq!(
        (
            problem.canonical_system().rows(),
            problem.canonical_system().columns()
        ),
        (10, 10)
    );
    // The only nonzero right-hand side is the 12 V source, so its norm is 12.
    // Request 1.2e-12 residual accuracy, strictly inside the independent 1e-11
    // semantic oracle; a 1e-12 relative request would instead permit 1.2e-11.
    let plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-13,
        1.0e-14,
        NonZeroUsize::new(100).expect("nonzero iterations"),
    )
    .expect("solver plan")
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast);
    let solution = solve_scalar_physical_affine(
        &problem,
        &vec![1.0; problem.canonical_system().columns()],
        LinearSolveRequest::new(&FaerLinearSolver, plan),
    )
    .expect("solve exact packaged physical model");
    (problem, solution)
}

fn physical_run_fixture(packaged: &PackagedModelDocument, output: ArtifactDigest) -> RunManifestV1 {
    RunManifestV1::new(
        ArtifactDigest::from_hex(packaged.model().digest().expect("Model digest"))
            .expect("canonical Artifact digest"),
        packaged.model().program().revision().0,
        "eqiora-backend-faer",
        env!("CARGO_PKG_VERSION"),
    )
    .expect("physical Run")
    .with_numerical_setting("solver.method", "bicgstab")
    .expect("solver method")
    .with_numerical_setting("solver.preconditioner", "identity")
    .expect("preconditioner")
    .with_numerical_setting("solver.reduction", "fast")
    .expect("reduction")
    .with_output(output)
}

fn value(
    packaged: &PackagedModelDocument,
    solution: &ScalarPhysicalAffineSolution,
    name: &str,
    unknown: fn(Id<kinds::Port>) -> PhysicalUnknown,
) -> f64 {
    let port = packaged.model().aliases()[name]
        .downcast()
        .expect("retained physical Port");
    solution.value(unknown(port)).expect("physical value")
}

fn assert_root_connection_identity(
    packaged: &PackagedModelDocument,
    root: &ModelPackageIdentityV1,
    connection: Id<kinds::Connection>,
    members: &[&str],
) {
    let digest = root.semantic_digest.to_hex();
    let namespace = IdentityNamespace::new([
        "resolved-package-v1",
        root.name.as_str(),
        root.version.as_str(),
        digest.as_str(),
        "module",
        "model",
    ])
    .expect("root identity namespace");
    let declaration = DeclarationPath::new([
        "package",
        root.name.as_str(),
        root.version.as_str(),
        digest.as_str(),
        "module",
        "model",
        "model",
        "Main",
        "net",
    ])
    .expect("root net declaration path");
    let member_identities = members.iter().map(|name| {
        let graph_id = packaged.model().aliases()[*name];
        packaged
            .provenance()
            .identity_for_graph_id(graph_id)
            .expect("retained member full identity")
    });
    let expected = ElaborationKey::anonymous_connection(
        namespace,
        InstancePath::new(["Main"]).expect("root instance path"),
        declaration,
        member_identities,
    )
    .expect("root-owned anonymous Connection")
    .full_identity()
    .expect("full Connection identity");
    assert_eq!(
        packaged
            .provenance()
            .identity_for_graph_id(connection.erase()),
        Some(expected),
        "the maximal net is owned by the exact root LCA, not a child fragment"
    );
}

fn assert_low_net_provenance(
    packaged: &PackagedModelDocument,
    connection: Id<kinds::Connection>,
    root_fragment_count: usize,
) {
    let provenance = packaged
        .provenance()
        .get_by_graph_id(connection.erase())
        .expect("low-net provenance");
    let origins = provenance.origins();
    assert_eq!(origins.len(), root_fragment_count + 1);
    assert_eq!(provenance.definition_span(), origins[0].definition_span());
    assert_eq!(provenance.instance_span(), origins[0].instance_span());
    assert_eq!(provenance.binding_spans(), origins[0].binding_spans());

    let dependency = origins
        .iter()
        .filter(|origin| {
            origin
                .definition_span()
                .file
                .contains("Eqiora.Verify.PhysicalComponents")
        })
        .collect::<Vec<_>>();
    assert_eq!(dependency.len(), 1);
    let dependency = dependency[0];
    assert!(
        dependency
            .instance_span()
            .file
            .contains("org.eqiora.verify.hierarchical_physical_boundary")
    );
    assert_eq!(dependency.binding_spans().len(), 1);
    assert!(
        dependency.binding_spans()[0]
            .file
            .contains("org.eqiora.verify.hierarchical_physical_boundary")
    );

    let root = origins
        .iter()
        .filter(|origin| {
            origin
                .definition_span()
                .file
                .contains("org.eqiora.verify.hierarchical_physical_boundary")
        })
        .collect::<Vec<_>>();
    assert_eq!(root.len(), root_fragment_count);
    assert!(root.iter().all(|origin| {
        origin
            .instance_span()
            .file
            .contains("org.eqiora.verify.hierarchical_physical_boundary")
            && origin.binding_spans().is_empty()
    }));
    assert!(
        root.windows(2)
            .all(|pair| pair[0].definition_span().start < pair[1].definition_span().start),
        "distinct root fragments remain in source order"
    );
}

#[test]
fn exact_package_boundary_partitions_have_one_model_and_one_solution() {
    let components = components_release();
    let nary_root = root_release(&components, NARY);
    let partitioned_root = root_release(&components, PARTITIONED);
    let root_identity = nary_root.package_identity().expect("root identity");

    assert_eq!(
        root_identity,
        partitioned_root
            .package_identity()
            .expect("partitioned identity"),
        "connection spelling must not enter package semantic identity"
    );
    assert_ne!(
        nary_root.source_digest().expect("N-ary source digest"),
        partitioned_root
            .source_digest()
            .expect("partitioned source digest"),
        "different author sources remain distinct provenance inputs"
    );

    let (nary, nary_resolution) = compile_locked(&components, &nary_root);
    let (partitioned, partitioned_resolution) = compile_locked(&components, &partitioned_root);
    assert_eq!(nary_resolution.nodes().len(), 2);
    assert_eq!(nary_resolution.edges().len(), 1);
    assert_ne!(nary_resolution, partitioned_resolution);
    assert_eq!(
        nary.model().canonical_json().expect("N-ary Model"),
        partitioned
            .model()
            .canonical_json()
            .expect("partitioned Model")
    );
    assert_eq!(
        nary.model().digest().expect("N-ary Model digest"),
        partitioned
            .model()
            .digest()
            .expect("partitioned Model digest")
    );

    for packaged in [&nary, &partitioned] {
        assert!(packaged.model().aliases().get("load.positive").is_none());
        assert!(packaged.model().aliases().get("load.negative").is_none());
        assert_eq!(
            packaged
                .model()
                .program()
                .nodes()
                .filter(|node| matches!(node, KernelNode::Port(_)))
                .count(),
            5,
            "forwarding exposures are not physical unknowns"
        );
        assert_eq!(observe_connection_sets(packaged.model().program()).len(), 2);
    }

    let negative = |packaged: &PackagedModelDocument| {
        packaged.model().aliases()["load.element.negative"]
            .downcast()
            .expect("retained negative load Port")
    };
    let nary_low = selected_connection(&nary, negative(&nary));
    let partitioned_low = selected_connection(&partitioned, negative(&partitioned));
    assert_eq!(nary_low, partitioned_low);
    for (packaged, connection) in [(&nary, nary_low), (&partitioned, partitioned_low)] {
        assert_root_connection_identity(
            packaged,
            &root_identity,
            connection,
            &[
                "source.negative",
                "load.element.negative",
                "ground.terminal",
            ],
        );
    }
    assert_low_net_provenance(&nary, nary_low, 1);
    assert_low_net_provenance(&partitioned, partitioned_low, 2);

    assert_ne!(
        nary.compilation()
            .digest()
            .expect("N-ary compilation digest"),
        partitioned
            .compilation()
            .digest()
            .expect("partitioned compilation digest"),
        "source lineage remains outside equal Model identity"
    );
    let changed_compilation_packages = nary
        .compilation()
        .packages()
        .iter()
        .zip(partitioned.compilation().packages())
        .filter_map(|(nary, partitioned)| {
            assert_eq!(nary.package(), partitioned.package());
            (nary.source_digest() != partitioned.source_digest())
                .then(|| nary.package().name.as_str())
        })
        .collect::<Vec<_>>();
    assert_eq!(
        changed_compilation_packages,
        ["org.eqiora.verify.hierarchical_physical_boundary"]
    );

    let (nary_problem, nary_solution) = lower_and_solve(&nary);
    let (partitioned_problem, partitioned_solution) = lower_and_solve(&partitioned);
    assert_eq!(
        nary_problem, partitioned_problem,
        "semantic composition, ordered DAGs, parameter binding, and CSR must all agree"
    );
    assert_eq!(
        nary_problem
            .reference_residuals(&vec![1.0; nary_problem.canonical_system().columns()])
            .expect("N-ary reference probe"),
        partitioned_problem
            .reference_residuals(&vec![1.0; partitioned_problem.canonical_system().columns()])
            .expect("partitioned reference probe")
    );
    for (name, across, through) in [
        ("source.positive", 12.0, -6.0),
        ("load.element.positive", 12.0, 6.0),
        ("source.negative", 0.0, 6.0),
        ("load.element.negative", 0.0, -6.0),
        ("ground.terminal", 0.0, 0.0),
    ] {
        for (packaged, solution) in [
            (&nary, &nary_solution),
            (&partitioned, &partitioned_solution),
        ] {
            assert!(
                (value(packaged, solution, name, PhysicalUnknown::Across) - across).abs()
                    <= VALUE_TOLERANCE
            );
            assert!(
                (value(packaged, solution, name, PhysicalUnknown::Through) - through).abs()
                    <= VALUE_TOLERANCE
            );
        }
    }
    assert!(
        nary_solution.reference_residual_norm() <= RESIDUAL_TOLERANCE,
        "N-ary reference residual {} exceeds {RESIDUAL_TOLERANCE}",
        nary_solution.reference_residual_norm()
    );
    assert!(
        partitioned_solution.reference_residual_norm() <= RESIDUAL_TOLERANCE,
        "partitioned reference residual {} exceeds {RESIDUAL_TOLERANCE}",
        partitioned_solution.reference_residual_norm()
    );
    assert_eq!(nary_solution.values(), partitioned_solution.values());
}

#[test]
fn physical_exposure_catalog_and_observation_replay_are_exact() {
    let components = components_release();
    let nary_root = root_release(&components, NARY);
    let partitioned_root = root_release(&components, PARTITIONED);
    let (nary, nary_resolution) = compile_locked(&components, &nary_root);
    let (partitioned, partitioned_resolution) = compile_locked(&components, &partitioned_root);

    let nary_catalog = nary
        .physical_exposure_catalog()
        .expect("N-ary exposure catalog");
    let partitioned_catalog = partitioned
        .physical_exposure_catalog()
        .expect("partitioned exposure catalog");
    let mut selectors = nary_catalog
        .projections()
        .iter()
        .map(|projection| projection.selector())
        .collect::<Vec<_>>();
    selectors.sort_unstable();
    assert_eq!(selectors, ["load.negative", "load.positive"]);
    assert_eq!(nary_catalog.projections().len(), 2);
    assert_eq!(partitioned_catalog.projections().len(), 2);
    for (nary_projection, partitioned_projection) in nary_catalog
        .projections()
        .iter()
        .zip(partitioned_catalog.projections())
    {
        assert_eq!(
            nary_projection.selector(),
            partitioned_projection.selector()
        );
        assert_eq!(
            nary_projection.projection_id(),
            partitioned_projection.projection_id(),
            "source fragment partition and source spans do not change projection meaning"
        );
        assert_eq!(
            nary_projection.interior_port_sha256(),
            partitioned_projection.interior_port_sha256()
        );
    }
    assert_ne!(
        nary_catalog.digest().expect("N-ary catalog digest"),
        partitioned_catalog
            .digest()
            .expect("partitioned catalog digest"),
        "the catalog retains exact package compilation and source provenance"
    );

    let bytes = nary_catalog.canonical_json().expect("catalog JSON");
    let decoded = PhysicalExposureCatalogEnvelopeV1::from_json(&bytes, Default::default())
        .expect("decode catalog");
    assert_eq!(decoded.canonical_json().expect("re-encoded catalog"), bytes);
    assert_eq!(decoded.digest(), nary_catalog.digest());
    nary.validate_physical_exposure_catalog(&decoded, &nary_resolution)
        .expect("exact catalog replay");
    assert!(
        PhysicalExposureCatalogEnvelopeV1::from_json(
            &bytes,
            PhysicalExposureDecoderLimits {
                max_physical_exposure_projections: 1,
                ..Default::default()
            },
        )
        .is_err(),
        "catalog projection count has an independent decoder budget"
    );
    let mut drifted_provenance: serde_json::Value =
        serde_json::from_slice(&bytes).expect("catalog JSON value");
    drifted_provenance["projections"][0]["origins"][0]["definition"]["file"] =
        serde_json::Value::String("moved/source/model.eqi".to_owned());
    let drifted_provenance = PhysicalExposureCatalogEnvelopeV1::from_json(
        &serde_json::to_vec(&drifted_provenance).expect("drifted catalog JSON"),
        Default::default(),
    )
    .expect("locally valid moved provenance");
    assert!(
        nary.validate_physical_exposure_catalog(&drifted_provenance, &nary_resolution)
            .is_err(),
        "exact package replay rejects source-provenance drift"
    );
    let target = nary_catalog
        .projections()
        .iter()
        .find(|projection| projection.selector() == "load.positive")
        .expect("positive exposure");
    let outside = nary
        .provenance()
        .identity_for_graph_id(nary.model().aliases()["source.positive"])
        .expect("outside retained Port identity");
    let PhysicalExposureContractV1::ScalarPhysical { connector_sha256 } = target.contract() else {
        panic!("positive exposure must be scalar physical");
    };
    let forged_cut = PhysicalExposureProjectionV1::scalar(
        target.selector(),
        target.exposure_sha256(),
        target.connection_sha256(),
        vec![*outside.as_bytes()],
        connector_sha256,
        target.origins().to_vec(),
    )
    .expect("a different proper subset is structurally admissible");
    let forged_entries = nary_catalog
        .projections()
        .iter()
        .map(|projection| {
            if projection.selector() == target.selector() {
                forged_cut.clone()
            } else {
                projection.clone()
            }
        })
        .collect();
    let forged_catalog = PhysicalExposureCatalogEnvelopeV1::new(
        nary_catalog.model_artifact(),
        nary.model().program(),
        nary_catalog.package_compilation(),
        forged_entries,
    )
    .expect("flat Model alone cannot distinguish a forged proper cut");
    assert!(
        nary.validate_physical_exposure_catalog(&forged_catalog, &nary_resolution)
            .is_err(),
        "exact compiler replay rejects a different proper subset of the same Connection"
    );
    let mut unknown_field: serde_json::Value =
        serde_json::from_slice(&bytes).expect("catalog JSON value");
    unknown_field["values"] = serde_json::json!([1.0]);
    assert!(
        PhysicalExposureCatalogEnvelopeV1::from_json(
            &serde_json::to_vec(&unknown_field).expect("unknown-field JSON"),
            Default::default(),
        )
        .is_err(),
        "the identity catalog cannot grow an untyped numerical payload"
    );
    assert!(
        nary.validate_physical_exposure_catalog(partitioned_catalog, &nary_resolution)
            .is_err(),
        "equal Model meaning cannot erase distinct exact source lineage"
    );
    assert!(
        partitioned
            .validate_physical_exposure_catalog(nary_catalog, &partitioned_resolution)
            .is_err()
    );

    let output = ArtifactDigest::from_sha256([0x5a; 32]);
    let run = physical_run_fixture(&nary, output.clone());
    nary.bind_run_v1(&run).expect("exact package Run binding");
    let projection = nary_catalog.projections()[0].projection_id();
    let common = nary
        .bind_physical_observation_v1(
            projection.clone(),
            PhysicalExposureQuantityV1::Common,
            &run,
            output.clone(),
        )
        .expect("common observation");
    let net_outward = nary
        .bind_physical_observation_v1(
            projection,
            PhysicalExposureQuantityV1::NetOutward,
            &run,
            output,
        )
        .expect("net-outward observation");
    assert_ne!(common.digest(), net_outward.digest());
    nary.validate_physical_observation_v1(&common, &run, &nary_resolution)
        .expect("observation replay");
    let common_bytes = common.canonical_json().expect("observation JSON");
    let decoded_common =
        PhysicalExposureObservationBindingV1::from_json(&common_bytes, Default::default())
            .expect("decode observation");
    assert_eq!(decoded_common.canonical_json().unwrap(), common_bytes);
    nary.validate_physical_observation_v1(&decoded_common, &run, &nary_resolution)
        .expect("decoded observation replay");
    let different_run = RunManifestV1::new(
        run.model(),
        run.semantic_revision(),
        "different-executor",
        env!("CARGO_PKG_VERSION"),
    )
    .expect("different Run")
    .with_output(decoded_common.result());
    assert!(
        nary.validate_physical_observation_v1(&decoded_common, &different_run, &nary_resolution,)
            .is_err(),
        "matching Model and output cannot substitute a different Run identity"
    );
    assert!(
        nary.validate_physical_observation_v1(&decoded_common, &run, &partitioned_resolution,)
            .is_err(),
        "an observation cannot substitute a source-distinct package resolution"
    );

    let absent = ArtifactDigest::from_sha256([0xa5; 32]);
    assert!(
        PhysicalExposureObservationBindingV1::new_v1(
            nary_catalog,
            nary_catalog.projections()[0].projection_id(),
            PhysicalExposureQuantityV1::Common,
            &run,
            absent,
        )
        .is_err(),
        "an output absent from the exact Run fails closed"
    );
}

#[test]
fn unclosed_imported_boundary_fails_before_model_exposure() {
    let components = components_release();
    let root = prepare_package_release_v1(
        root_sources(&components, INVALID_UNCLOSED),
        std::slice::from_ref(&components),
    )
    .expect("definition-valid package release");
    let resolution =
        ResolutionRecordV1::from_exact_releases(&root, std::slice::from_ref(&components))
            .expect("exact package resolution");
    let mut store = InMemoryPackageStore::default();
    store.insert(&components).expect("insert component package");
    store.insert(&root).expect("insert root package");
    let error = PackagedModelDocument::compile_locked(&store, &resolution, "Main")
        .expect_err("an unclosed imported boundary cannot expose a compiled Model");
    let diagnostics = match error {
        eqiora::package::PackageCompilationError::Diagnostics(diagnostics) => diagnostics,
        other => panic!("expected typed source diagnostics, received {other}"),
    };
    assert!(
        diagnostics.iter().any(|diagnostic| {
            diagnostic.code() == codes::LANGUAGE_TYPE_ERROR
                && (diagnostic
                    .message()
                    .contains("conserving Connection membership")
                    || diagnostic.message().contains("retains 1 members"))
        }),
        "{diagnostics:#?}"
    );
}
