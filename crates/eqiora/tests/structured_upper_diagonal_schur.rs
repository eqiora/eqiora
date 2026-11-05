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
    AuthenticatedCommonMesh, CommonLinearRequest, CommonScalingRequest, CommonSolvePolicy,
    CommonSpatialPolicy, ResolvedCommonPlan,
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

fn admit_small(
    source: &str,
    generic: bool,
) -> Result<(ResolvedCommonPlan, eqiora::meshing::SimplicialMesh), eqiora::Diagnostic> {
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
            source.as_bytes().to_vec(),
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
    let velocity = document.model().aliases()["definition.velocity"]
        .downcast()
        .unwrap();
    let pressure = document.model().aliases()["definition.pressure"]
        .downcast()
        .unwrap();
    let policy = AffineTriangleMeshPolicy::new([2, 2]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_affine_triangles(&geometry, &policy).unwrap();
    let observed_mesh = mesh.mesh().clone();
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
    let composition = if generic {
        PreconditionerPolicy::Identity
    } else {
        composition
    };
    let solver = SolverPlan::new(
        if generic {
            LinearSolver::SparseLu
        } else {
            LinearSolver::BiConjugateGradientStabilized
        },
        1e-11,
        1e-13,
        NonZeroUsize::new(10_000).unwrap(),
    )
    .unwrap()
    .with_preconditioner(composition.clone())
    .with_reduction(ReductionPolicy::Fast);
    let plan = ResolvedCommonPlan::resolve(
        &ModelEnvelope::from_program(document.model().program()).unwrap(),
        owner,
        CommonSpatialPolicy::MiniP1,
        CommonSolvePolicy::Linear(
            CommonLinearRequest::exact(solver, FAER_SOLVER_PROVIDER).unwrap(),
        ),
        (!generic).then_some(CommonScalingRequest::Incompressible(
            eqiora_numerics::IncompressibleScalingRequest2d::from_si(
                Some(1.0),
                Some(1.0),
                Some(1.0),
            )
            .unwrap(),
        )),
        None,
        &FaerLinearSolver,
        None,
        None,
    )?;
    assert_eq!(
        plan.effective_solver().unwrap().preconditioner(),
        &composition
    );
    assert_eq!(plan.linear_solver_provider(), Some(FAER_SOLVER_PROVIDER));
    Ok((plan, observed_mesh))
}

#[test]
fn ordinary_manual_manufactured_stokes_small() {
    let (plan, observed_mesh) =
        admit_small(SOURCE, false).expect("ordinary manufactured Plan admission");
    let result = plan
        .as_steady_stokes()
        .unwrap()
        .run_result(&FaerLinearSolver)
        .expect("ordinary manufactured Run");
    assert_eq!(result.plan().identity(), plan.identity());
    let errors = normalized_errors(&result, &observed_mesh, 1., 1.);
    assert!(errors.iter().all(|value| value.is_finite() && *value >= 0.));
    let residual = result.solve_true_residual_norm(None, None).unwrap();
    let target = result.solve_evidence(None, None).unwrap().residual_target();
    assert!(residual <= target);
}

const FIRST_COMPONENT: &str =
    "speed*(coordinate(0)/length_scale)^2*tensor_value(frame=body,components=[1,0])";
const SECOND_COMPONENT: &str =
    "2*speed*coordinate(0)*coordinate(1)/length_scale^2*tensor_value(frame=body,components=[0,1])";

fn rewritten_velocity_laws(law: &str) -> String {
    let datum = format!("trace(velocity) = {FIRST_COMPONENT} - {SECOND_COMPONENT}");
    assert_eq!(SOURCE.matches(&datum).count(), 4);
    SOURCE.replace(&datum, law)
}

#[test]
fn equivalent_grouped_and_signed_vector_boundary_residuals_admit() {
    for law in [
        format!("{SECOND_COMPONENT} - {FIRST_COMPONENT} + trace(velocity) = 0"),
        format!("-(trace(velocity) - ({FIRST_COMPONENT} - {SECOND_COMPONENT})) = 0"),
    ] {
        let (plan, _) = admit_small(&rewritten_velocity_laws(&law), false)
            .expect("equivalent vector boundary Plan");
        assert!(plan.as_steady_stokes().is_some());
    }
}

#[test]
fn duplicate_velocity_trace_is_rejected() {
    let source = rewritten_velocity_laws(&format!(
        "trace(velocity) + trace(velocity) = 2*({FIRST_COMPONENT} - {SECOND_COMPONENT})"
    ));
    assert!(admit_small(&source, false).is_err());
}

#[test]
fn tangential_traction_keeps_the_generic_mixed_projection() {
    let top = format!(
        "relation top on y_upper {{ trace(velocity) = {FIRST_COMPONENT} - {SECOND_COMPONENT}; }}"
    );
    assert!(SOURCE.contains(&top));
    // Exact manufactured stress on y=L: sigma*n=P*(-2, 1/2-5*x/L).
    let traction = "relation top on y_upper { normal(2*1[kg/(m*s)]*symmetric_part(grad(velocity))-isotropic_lift(pressure)) = pressure_scale*(-2*tensor_value(frame=body,components=[1,0]) + (0.5-5*coordinate(0)/length_scale)*tensor_value(frame=body,components=[0,1])); }";
    let source = SOURCE.replace(&top, traction);
    let (plan, _) = admit_small(&source, true).expect("generic mixed tangential-traction Plan");
    assert!(plan.as_spatial().is_some());
    assert!(plan.as_steady_stokes().is_none());
}

/// Integrate the declared analytic oracle from ordinary Result coefficient blocks.
/// Barycentric P1/bubble values and gradients are written from their definitions.
fn normalized_errors(
    result: &eqiora_numerics::CommonResult,
    mesh: &eqiora::meshing::SimplicialMesh,
    length: f64,
    pressure_scale: f64,
) -> [f64; 4] {
    use eqiora::meshing::{MeshEntity, MeshTopology};
    let plan = result.plan().as_steady_stokes().unwrap();
    let field_block = |id: &str, association: &str| {
        let field = (0..result.field_count())
            .find(|&index| result.field(index).unwrap().0 == id)
            .unwrap();
        (0..result.field_block_count(field))
            .find_map(|index| {
                let (kind, values, _) = result.field_block(field, index).unwrap();
                (kind == association).then_some(values)
            })
            .unwrap()
    };
    let velocity = field_block(plan.velocity_field_id(), "vertex");
    let bubbles = field_block(plan.velocity_field_id(), "cell-bubble");
    let pressure = field_block(plan.pressure_field_id(), "vertex");
    let quadrature = eqiora::meshing::simplex_duffy_gauss_legendre(2, 4).unwrap();
    let mut sums = [0.; 4];
    let mut pressure_integral = 0.;
    for cell in 0..mesh.entity_count(2).unwrap() {
        let ids = mesh.entity_vertices(MeshEntity::new(2, cell)).unwrap();
        let coordinates = ids
            .iter()
            .map(|id| &mesh.vertices()[id.index()])
            .collect::<Vec<_>>();
        let dx1 = coordinates[1][0] - coordinates[0][0];
        let dx2 = coordinates[2][0] - coordinates[0][0];
        let dy1 = coordinates[1][1] - coordinates[0][1];
        let dy2 = coordinates[2][1] - coordinates[0][1];
        let determinant = dx1 * dy2 - dx2 * dy1;
        let gradients = [
            [(dy1 - dy2) / determinant, (dx2 - dx1) / determinant],
            [dy2 / determinant, -dx2 / determinant],
            [-dy1 / determinant, dx1 / determinant],
        ];
        for point in quadrature.points() {
            let lambda = [
                1. - point.coordinates[0] - point.coordinates[1],
                point.coordinates[0],
                point.coordinates[1],
            ];
            let bubble = 27. * lambda[0] * lambda[1] * lambda[2];
            let bubble_gradient: [f64; 2] = std::array::from_fn(|axis| {
                27. * (gradients[0][axis] * lambda[1] * lambda[2]
                    + gradients[1][axis] * lambda[0] * lambda[2]
                    + gradients[2][axis] * lambda[0] * lambda[1])
            });
            let location: [f64; 2] = std::array::from_fn(|axis| {
                (0..3)
                    .map(|node| lambda[node] * coordinates[node][axis])
                    .sum::<f64>()
                    / length
            });
            let exact = [location[0] * location[0], -2. * location[0] * location[1]];
            let exact_gradient = [
                [2. * location[0] / length, 0.],
                [-2. * location[1] / length, -2. * location[0] / length],
            ];
            let mut value = [0.; 2];
            let mut gradient = [[0.; 2]; 2];
            for component in 0..2 {
                value[component] = (0..3)
                    .map(|node| lambda[node] * velocity[2 * ids[node].index() + component])
                    .sum::<f64>()
                    + bubble * bubbles[2 * cell + component];
                for axis in 0..2 {
                    gradient[component][axis] = (0..3)
                        .map(|node| {
                            gradients[node][axis] * velocity[2 * ids[node].index() + component]
                        })
                        .sum::<f64>()
                        + bubble_gradient[axis] * bubbles[2 * cell + component];
                }
            }
            let p = (0..3)
                .map(|node| lambda[node] * pressure[ids[node].index()])
                .sum::<f64>();
            let weight = determinant.abs() * point.weight;
            for component in 0..2 {
                sums[0] += weight * (value[component] - exact[component]).powi(2);
                for axis in 0..2 {
                    sums[1] += weight
                        * (gradient[component][axis] - exact_gradient[component][axis]).powi(2);
                }
            }
            sums[2] += weight * (p - pressure_scale * (location[0] - 0.5)).powi(2);
            sums[3] += weight * (gradient[0][0] + gradient[1][1]).powi(2);
            pressure_integral += weight * p;
        }
    }
    assert!((pressure_integral / (pressure_scale * length * length)).abs() <= 2e-10);
    [
        sums[0].sqrt() / length,
        sums[1].sqrt(),
        sums[2].sqrt() / (pressure_scale * length),
        sums[3].sqrt(),
    ]
}
