//! A dimensional cubic manufactured solution through imported expression contracts.
use eqiora::artifact::ModelEnvelope;
use eqiora::differentiation::{AcceptedLinearization, PrimalAcceptance, forward_sensitivity};
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::meshing::{MeshEntity, MeshTopology, QuadratureRule};
use eqiora::package::*;
use eqiora::realization::*;
use eqiora::sem::KernelProgram;
use eqiora::solver::LinearSolverBackend;
use eqiora::solver::{
    LinearOperatorProperties, LinearSolveRequest, LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan,
};
use eqiora_core::{Id, ScalarType, entity::kinds};
use eqiora_numerics::common::SpatialDesignCoordinate;
use eqiora_numerics::scalar::{
    ResolvedScalarEllipticCartesianSolution, finalize_scalar_elliptic_parameter_point,
    solve_and_linearize_resolved_scalar_elliptic_cartesian,
};
use std::num::{NonZeroU16, NonZeroUsize};

const LIBRARY: &str = r#"
public component Poisson(support body:volume(ambient_dimension=2),variable u:K on body,expression source:K/m^2 on body){relation balance on body{-div(grad(u))=source;}}
public component Fixed(support body:volume(ambient_dimension=2),support face:boundary(parent=body),variable u:K on body,expression value:K on face){relation boundary_value on face{trace(u)=value;}}
"#;
const DECLARATIONS: &str = r#"
domain body=box(0,1,0,1);
domain left=boundary(body,axis=0,side=lower);domain right=boundary(body,axis=0,side=upper);
domain bottom=boundary(body,axis=1,side=lower);domain top=boundary(body,axis=1,side=upper);
variable u:K on body in h1;parameter A:K=2;parameter L:m=1;
"#;
fn imported() -> String {
    format!(
        "import org.eqiora.tests.Expression.main as eq;model Main(){{{DECLARATIONS}\ninstance volume:eq.Poisson(body=body,u=u,source=-6*A*coordinate(0)*L^-3);\ninstance left_value:eq.Fixed(body=body,face=left,u=u,value=0[K]);\ninstance right_value:eq.Fixed(body=body,face=right,u=u,value=A);\ninstance bottom_value:eq.Fixed(body=body,face=bottom,u=u,value=A*(coordinate(0)*L^-1)^3);\ninstance top_value:eq.Fixed(body=body,face=top,u=u,value=A*(coordinate(0)*L^-1)^3);}}"
    )
}
fn handwritten() -> String {
    format!(
        "model Main(){{{DECLARATIONS}\nrelation balance on body{{-div(grad(u))=-6*A*coordinate(0)*L^-3;}}\nrelation left_value on left{{trace(u)=0[K];}}\nrelation right_value on right{{trace(u)=A;}}\nrelation bottom_value on bottom{{trace(u)=A*(coordinate(0)*L^-1)^3;}}\nrelation top_value on top{{trace(u)=A*(coordinate(0)*L^-1)^3;}}}}"
    )
}
fn release(name: &str, source: &str, dependencies: &[PackageReleaseV1]) -> PackageReleaseV1 {
    let path = NormalizedRelativePath::parse("src/main.eqi").unwrap();
    let manifest = PackageManifestV1::new(
        "main",
        QualifiedName::parse(name).unwrap(),
        ExactVersion::parse("0.1.0").unwrap(),
        dependencies
            .iter()
            .map(|dep| PackageDependencyV1::new(dep.package_identity().unwrap()))
            .collect(),
        vec![BundleEntryV1::new(path.clone(), BundleRoleV1::ModelSource)],
    )
    .unwrap();
    prepare_package_release_v1(
        PackageSourcesV1::new(
            manifest,
            vec![SourceFileV1::new(
                path,
                BundleRoleV1::ModelSource,
                source.as_bytes().to_vec(),
            )],
        )
        .unwrap(),
        dependencies,
    )
    .unwrap()
}
fn packaged_program() -> KernelProgram {
    let library = release("org.eqiora.tests.Expression", LIBRARY, &[]);
    let root = release(
        "org.eqiora.tests.ExpressionRoot",
        &imported(),
        std::slice::from_ref(&library),
    );
    let root_id = root.package_identity().unwrap();
    let library_id = library.package_identity().unwrap();
    let resolution = ResolutionRecordV1::new(
        root_id.clone(),
        vec![
            ResolutionNodeV1::new(root_id.clone(), root.source_digest().unwrap()),
            ResolutionNodeV1::new(library_id.clone(), library.source_digest().unwrap()),
        ],
        vec![ResolutionEdgeV1::new(root_id, library_id)],
    )
    .unwrap();
    let directory =
        std::env::temp_dir().join(format!("eqiora-expression-slots-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    for release in [&library, &root] {
        std::fs::write(
            directory.join(format!("{}.json", release.source_digest().unwrap())),
            release.canonical_json().unwrap(),
        )
        .unwrap();
    }
    let compiled = PackagedModelDocument::compile_locked(
        &DirectoryPackageStore::open_ambient(&directory).unwrap(),
        &resolution,
        "Main",
    )
    .unwrap();
    let first = ModelEnvelope::from_program(compiled.model().program()).unwrap();
    drop(compiled);
    // Reopen solely from exact serialized releases and a serialized resolution.
    let resolution = ResolutionRecordV1::from_json(&resolution.canonical_json().unwrap()).unwrap();
    let reopened = PackagedModelDocument::compile_locked(
        &DirectoryPackageStore::open_ambient(&directory).unwrap(),
        &resolution,
        "Main",
    )
    .unwrap();
    assert_eq!(
        first.canonical_json().unwrap(),
        ModelEnvelope::from_program(reopened.model().program())
            .unwrap()
            .canonical_json()
            .unwrap()
    );
    let program = ModelEnvelope::from_json(&first.canonical_json().unwrap(), Default::default())
        .unwrap()
        .to_program()
        .unwrap();
    std::fs::remove_dir_all(directory).unwrap();
    program
}
fn direct_program() -> KernelProgram {
    let (tx, id, _) = eqiora::compiler::compile("handwritten.eqi", &handwritten())
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(tx).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), id).unwrap()
}
fn resolved(program: &KernelProgram, cells: usize) -> ResolvedRealization {
    let plan = RealizationPlan::new(
        Space::continuous_lagrange(NonZeroU16::MIN),
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(cells).unwrap(),
            },
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(2).unwrap(),
            },
        ),
        solver(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap();
    resolve(
        &RealizationRequest::explicit(
            program.model(),
            SemanticRevision::new(program.revision().0),
            RealizationRevision::new(1),
            plan,
        ),
        RealizationRequirements::new(
            NonZeroUsize::new(2).unwrap(),
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        ),
        &RealizationCapabilities::scalar_elliptic_reference(),
    )
    .unwrap()
}
fn solver() -> SolverPlan {
    SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1e-12,
        1e-14,
        NonZeroUsize::new(10000).unwrap(),
    )
    .unwrap()
}
fn amplitude(program: &KernelProgram) -> Id<kinds::Parameter> {
    program
        .nodes()
        .find_map(|node| match node {
            eqiora::kernel::KernelNode::Parameter(p)
                if p.value_type().dimension()
                    == eqiora::DimExponents::from_integers([0, 0, 0, 0, 1, 0, 0]).unwrap() =>
            {
                Some(p.id())
            }
            _ => None,
        })
        .unwrap()
}
#[test]
fn imported_and_handwritten_cubic_solutions_rebind_and_differentiate_without_carriers() {
    for program in [packaged_program(), direct_program()] {
        assert_eq!(
            program
                .nodes()
                .filter(|node| matches!(node, eqiora::kernel::KernelNode::Field(_)))
                .count(),
            1
        );
        assert_eq!(
            program
                .nodes()
                .filter(|node| matches!(node, eqiora::kernel::KernelNode::Parameter(_)))
                .count(),
            2
        );
        assert_eq!(
            program
                .nodes()
                .filter(|node| matches!(node, eqiora::kernel::KernelNode::Relation(_)))
                .count(),
            5
        );
        let amplitude = amplitude(&program);
        let mut errors = Vec::new();
        let mut derivative_errors = Vec::new();
        for cells in [4, 8, 16] {
            let resolved = resolved(&program, cells);
            let (model, solution, linearization) =
                solve_and_linearize_resolved_scalar_elliptic_cartesian(
                    &program,
                    &resolved,
                    &REFERENCE_LINEAR_SOLVER,
                    &[SpatialDesignCoordinate::from(amplitude)],
                )
                .unwrap();
            let ResolvedScalarEllipticCartesianSolution::FiniteElement(solution) = solution else {
                panic!("Q1")
            };
            let quadrature = QuadratureRule::tensor_product_gauss_legendre(2, 4).unwrap();
            errors.push(
                solution
                    .field()
                    .l2_error(&|point: &[f64]| 2. * point[0].powi(3), &quadrature)
                    .unwrap(),
            );
            let accepted =
                AcceptedLinearization::new(&linearization, &PrimalAcceptance::Residual(2e-9))
                    .unwrap();
            let tangent = forward_sensitivity(
                &accepted,
                &[1.],
                LinearOperatorProperties::SymmetricPositiveDefinite,
                LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, solver()),
            )
            .unwrap();
            let mesh = solution.field().mesh();
            let mut values = vec![0.; mesh.entity_count(0).unwrap()];
            let mut free = 0;
            for (vertex, value) in values.iter_mut().enumerate() {
                let point = mesh.vertex_coordinates(MeshEntity::new(0, vertex)).unwrap();
                let x = point[0];
                if x == 0. || x == 1. || point[1] == 0. || point[1] == 1. {
                    // Analytic derivative of the prescribed cubic boundary data.
                    *value = x.powi(3);
                } else {
                    *value = tangent.values()[free];
                    free += 1;
                }
            }
            assert_eq!(free, tangent.values().len());
            // Q1 shape functions are all 1/4 at each rectangular cell centre.
            // Compare the reconstructed AD field there with du/dA=x^3.
            let mut squared_error = 0.;
            for cell in 0..mesh.entity_count(2).unwrap() {
                let vertices = mesh.entity_vertices(MeshEntity::new(2, cell)).unwrap();
                let x = vertices
                    .iter()
                    .map(|v| mesh.vertex_coordinates(*v).unwrap()[0])
                    .sum::<f64>()
                    / 4.;
                let tangent = vertices.iter().map(|v| values[v.index()]).sum::<f64>() / 4.;
                squared_error += (tangent - x.powi(3)).powi(2);
            }
            derivative_errors.push((squared_error / mesh.entity_count(2).unwrap() as f64).sqrt());
            if cells == 8 {
                let graph = model.portable_realization(&resolved).unwrap();
                let bound = model.bind_selected_parameters(&[amplitude], &[3.]).unwrap();
                let finalized = finalize_scalar_elliptic_parameter_point(
                    bound,
                    &graph,
                    Target::HostCpu {
                        threads: NonZeroUsize::MIN,
                    },
                )
                .unwrap();
                let solved = REFERENCE_LINEAR_SOLVER
                    .solve(
                        &finalized.linear_problem().unwrap(),
                        finalized.solver_plan(),
                    )
                    .unwrap();
                let accepted = finalized.finish(solved).unwrap();
                let ResolvedScalarEllipticCartesianSolution::FiniteElement(bound) =
                    accepted.solution()
                else {
                    panic!("Q1")
                };
                // For an x-only cubic on a uniform tensor grid, exact load quadrature
                // makes the Q1 nodal equations exact. The solution is its interpolant.
                // Error <= max|u''|h²/8 = (9/4)h² for A=3,L=1.
                assert!(
                    bound
                        .field()
                        .l2_error(&|point: &[f64]| 3. * point[0].powi(3), &quadrature)
                        .unwrap()
                        <= 2.25 / (cells * cells) as f64
                );
            }
        }
        // Same Q1 order >=1.9 criterion as compiled_package_poisson_form.rs.
        for errors in [errors, derivative_errors] {
            for pair in errors.windows(2) {
                assert!((pair[0] / pair[1]).log2() >= 1.9, "{errors:?}");
            }
        }
    }
}
