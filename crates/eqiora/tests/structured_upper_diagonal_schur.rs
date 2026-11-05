//! Ordinary packaged Model -> Plan -> Run pilot for the declared MINI Stokes oracle.
use eqiora::artifact::{
    AffineTriangleMeshPolicy, GeometryMeshCorrespondenceEnvelopeV1,
    MeshProductionLineageEnvelopeV1, ModelEnvelope,
};
use eqiora::compiler::StaticBindingValue;
use eqiora::geometry::{GeometryGraph, PlanarTopologyHandle};
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageDependencyV1, PackageManifestV1, PackageSourcesV1, PackagedModelDocument, QualifiedName,
    ResolutionRecordV1, SourceFileV1, bundled_package_releases, prepare_package_release_v1,
};
use eqiora::solver::{
    AlgebraicBlock, FixedFactorizationChoice, LinearSolver, PreconditionerPolicy, ReductionPolicy,
    SolverPlan, UpperDiagonalSchurConfig,
};
use eqiora_backend_faer::{FAER_SOLVER_PROVIDER, FaerLinearSolver};
use eqiora_numerics::{
    AuthenticatedCommonMesh, CommonLinearRequest, CommonSolvePolicy, CommonSpatialPolicy,
    ResolvedCommonPlan,
};
use std::{collections::BTreeMap, num::NonZeroUsize};

const SOURCE: &str = r#"import Eqiora.Fluid.fluid as fluid;
public component Manufactured(
 support body: volume(ambient_dimension=2),
 support x_lower: boundary(parent=body), support x_upper: boundary(parent=body),
 support y_lower: boundary(parent=body), support y_upper: boundary(parent=body)
) {
 variable velocity: vector<m/s,2> on body in smooth;
 variable pressure: kg/(m*s^2) on body in h1;
 variable force_potential: kg/(m*s^2) on body;
 parameter length_scale: m = 1;
 parameter speed: m/s = 1;
 parameter pressure_scale: kg/(m*s^2) = 1;
 relation potential on body { force_potential = -pressure_scale*(coordinate(0)/length_scale - 0.5); }
 instance balance: fluid.ForcedSteadyStokesBalance2d(body=body, velocity=velocity, pressure=pressure, force_potential=force_potential, dynamic_viscosity=1[kg/(m*s)]);
 relation left on x_lower { trace(velocity) = speed*(coordinate(0)/length_scale)^2*tensor_value(frame=body,components=[1,0]) - 2*speed*coordinate(0)*coordinate(1)/length_scale^2*tensor_value(frame=body,components=[0,1]); }
 relation right on x_upper { trace(velocity) = speed*(coordinate(0)/length_scale)^2*tensor_value(frame=body,components=[1,0]) - 2*speed*coordinate(0)*coordinate(1)/length_scale^2*tensor_value(frame=body,components=[0,1]); }
 relation bottom on y_lower { trace(velocity) = speed*(coordinate(0)/length_scale)^2*tensor_value(frame=body,components=[1,0]) - 2*speed*coordinate(0)*coordinate(1)/length_scale^2*tensor_value(frame=body,components=[0,1]); }
 relation top on y_upper { trace(velocity) = speed*(coordinate(0)/length_scale)^2*tensor_value(frame=body,components=[1,0]) - 2*speed*coordinate(0)*coordinate(1)/length_scale^2*tensor_value(frame=body,components=[0,1]); }
}"#;

#[test]
fn ordinary_manual_manufactured_stokes_small() {
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([0., 1.], [0., 1.]).unwrap();
    let edges = rectangle.boundaries();
    let mut sets = BTreeMap::from([("body".to_owned(), vec![rectangle.region().into()])]);
    for (name, edge) in ["x_lower", "x_upper", "y_lower", "y_upper"]
        .into_iter()
        .zip(edges)
    {
        sets.insert(name.to_owned(), vec![PlanarTopologyHandle::from(edge)]);
    }
    let geometry = graph.build(&rectangle, &sets).unwrap();
    let bindings = ["body", "x_lower", "x_upper", "y_lower", "y_upper"].map(|name| {
        (
            name,
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set(name).unwrap(),
                parent: if name == "body" {
                    None
                } else {
                    Some(geometry.entity_set("body").unwrap())
                },
            },
        )
    });
    let releases = bundled_package_releases("Eqiora.Fluid").unwrap();
    let fluid = releases
        .iter()
        .find(|release| release.package_identity().unwrap().name.as_str() == "Eqiora.Fluid")
        .unwrap();
    let path = NormalizedRelativePath::parse("src/main.eqi").unwrap();
    let manifest = PackageManifestV1::new(
        "main",
        QualifiedName::parse("org.eqiora.verify.StructuredSchur").unwrap(),
        ExactVersion::parse("0.1.0").unwrap(),
        vec![PackageDependencyV1::new(fluid.package_identity().unwrap())],
        vec![BundleEntryV1::new(path.clone(), BundleRoleV1::ModelSource)],
    )
    .unwrap();
    let sources = PackageSourcesV1::new(
        manifest,
        vec![SourceFileV1::new(
            path,
            BundleRoleV1::ModelSource,
            SOURCE.as_bytes().to_vec(),
        )],
    )
    .unwrap();
    let root = prepare_package_release_v1(sources, &releases).unwrap();
    let resolution = ResolutionRecordV1::from_exact_releases(&root, &releases).unwrap();
    let mut store = InMemoryPackageStore::default();
    for release in &releases {
        store.insert(release).unwrap();
    }
    store.insert(&root).unwrap();
    let document =
        PackagedModelDocument::compile_selected(&store, &resolution, "Manufactured", &bindings)
            .unwrap();
    let velocity = document.model().aliases()["velocity"].downcast().unwrap();
    let pressure = document.model().aliases()["pressure"].downcast().unwrap();
    let policy = AffineTriangleMeshPolicy::new([2, 2]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_affine_triangles(&geometry, &policy).unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_affine_triangle_resources(
        &policy,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    let owner =
        AuthenticatedCommonMesh::affine_triangles(geometry, mesh, correspondence, production)
            .unwrap();
    let factor = FixedFactorizationChoice::new(LinearSolver::SparseLu, "eqiora.faer").unwrap();
    let composition = PreconditionerPolicy::upper_diagonal_schur(
        UpperDiagonalSchurConfig::new(
            [
                vec![AlgebraicBlock::Field(velocity)],
                vec![
                    AlgebraicBlock::Field(pressure),
                    AlgebraicBlock::ConstraintMultiplier { field: pressure },
                ],
            ],
            factor.clone(),
            factor,
            100_000,
            10_000_000,
        )
        .unwrap(),
    );
    let solver = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-11,
        1e-13,
        NonZeroUsize::new(10_000).unwrap(),
    )
    .unwrap()
    .with_preconditioner(composition.clone())
    .with_reduction(ReductionPolicy::Reproducible);
    let plan = ResolvedCommonPlan::resolve(
        &ModelEnvelope::from_program(document.model().program()).unwrap(),
        owner,
        CommonSpatialPolicy::MiniP1,
        CommonSolvePolicy::Linear(
            CommonLinearRequest::exact(solver, FAER_SOLVER_PROVIDER).unwrap(),
        ),
        None,
        None,
        &FaerLinearSolver,
        None,
        None,
    )
    .expect("ordinary manufactured Plan admission");
    assert_eq!(
        plan.effective_solver().unwrap().preconditioner(),
        &composition
    );
    assert_eq!(plan.linear_solver_provider(), Some(FAER_SOLVER_PROVIDER));
    let result = plan
        .as_steady_stokes()
        .unwrap()
        .run_result(&FaerLinearSolver)
        .expect("ordinary manufactured Run");
    assert_eq!(result.plan().identity(), plan.identity());
}
