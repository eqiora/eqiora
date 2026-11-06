//! Divided scalar coefficients through the ordinary Q1 compiler and solve path.
use eqiora::differentiation::{AcceptedLinearization, PrimalAcceptance, forward_sensitivity};
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::meshing::{MeshEntity, MeshTopology, QuadratureRule};
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

const DECLARATIONS: &str = r#"
domain body=box(0,1,0,1);
domain left=boundary(body,axis=0,side=lower);domain right=boundary(body,axis=0,side=upper);
domain bottom=boundary(body,axis=1,side=lower);domain top=boundary(body,axis=1,side=upper);
variable u:K on body in h1;parameter A:K=2;parameter L:m=1;
"#;
fn handwritten() -> String {
    format!(
        "model Main(){{{DECLARATIONS}\nrelation balance on body{{-div(grad(u))=-6*A*coordinate(0)/L^3;}}\nrelation left_value on left{{trace(u)=0[K];}}\nrelation right_value on right{{trace(u)=A*(1[m]/L)^3;}}\nrelation bottom_value on bottom{{trace(u)=A*(coordinate(0)/L)^3;}}\nrelation top_value on top{{trace(u)=A*(coordinate(0)/L)^3;}}}}"
    )
}
fn program(source: &str) -> KernelProgram {
    let (tx, id, _) = eqiora::compiler::compile("handwritten.eqi", source)
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
// Pointwise error budget, derived from the uniform-square Q1 stencil rather
// than measured Eqiora output: K has diagonal 8c/3 and eight neighbours -c/3.
// These are SI-coordinate bounds: K for the primal, K/m for the L tangent.
// The full stencil on phi=x(1-x) gives 2c/N^2, with max(phi)=1/4.
// Since phi is nonnegative on the boundary, K_ff*phi_free >= 2c/N^2;
// the discrete maximum principle gives ||K_ff^-1||_infinity <= N^2/(8c).
// CG uses the Euclidean residual target
// max(1e-14, 1e-12*||b||_2); ||r||_infinity <= ||r||_2 transfers that target
// to a nodal error bound. The 2e-9 checks below exceed twice their respective
// exact-arithmetic bounds. This factor is an engineering allowance for assembly
// and solve rounding, not a rigorous floating-point error bound.
#[test]
fn divided_cubic_temperature_source_rebinds_and_differentiates() {
    let program = program(&handwritten());
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
    let length = program
        .nodes()
        .find_map(|node| match node {
            eqiora::kernel::KernelNode::Parameter(p)
                if p.value_type().dimension()
                    == eqiora::DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap() =>
            {
                Some(p.id())
            }
            _ => None,
        })
        .unwrap();
    let mut errors = Vec::new();
    let mut derivative_errors = Vec::new();
    for cells in [4, 8, 16] {
        let resolved = resolved(&program, cells);
        let (model, solution, linearization) =
            solve_and_linearize_resolved_scalar_elliptic_cartesian(
                &program,
                &resolved,
                &REFERENCE_LINEAR_SOLVER,
                &[
                    SpatialDesignCoordinate::from(amplitude),
                    SpatialDesignCoordinate::from(length),
                ],
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
            AcceptedLinearization::new(&linearization, &PrimalAcceptance::Residual(2e-9)).unwrap();
        let tangent = forward_sensitivity(
            &accepted,
            &[1., 0.],
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
        let length_tangent = forward_sensitivity(
            &accepted,
            &[0., 1.],
            LinearOperatorProperties::SymmetricPositiveDefinite,
            LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, solver()),
        )
        .unwrap();
        let mut free = 0;
        for vertex in 0..mesh.entity_count(0).unwrap() {
            let point = mesh.vertex_coordinates(MeshEntity::new(0, vertex)).unwrap();
            if point[0] != 0. && point[0] != 1. && point[1] != 0. && point[1] != 1. {
                // d(A*x^3/L^3)/dL = -3*A*x^3/L^4 at A=2,L=1.
                // forward_sensitivity solves K_ff*g = -R_L. Here neither geometry
                // nor K depends on L: R_L=K_fb*g_boundary - f_L, so its RHS is
                // exactly K_ff*g_free for g=-6*x^3 in exact arithmetic; no accepted
                // primal approximation enters this RHS. Independent rational stencil
                // sums give ||b||_2 < 9,17,27 at N=4,8,16 (squares 20171/256,
                // 2343691/8192,179085227/262144). Thus the worst nodal solve-error
                // bound is (16^2/8)*(27e-12)=8.64e-10, below half this threshold.
                assert!((length_tangent.values()[free] + 6. * point[0].powi(3)).abs() < 2e-9);
                free += 1;
            }
        }
        assert_eq!(free, length_tangent.values().len());
        if cells == 8 {
            let graph = model.portable_realization(&resolved).unwrap();
            let bound = model
                .bind_selected_parameters(&[amplitude, length], &[3., 2.])
                .unwrap();
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
                    finalized.solver_plan().clone(),
                )
                .unwrap();
            let accepted = finalized.finish(solved).unwrap();
            let ResolvedScalarEllipticCartesianSolution::FiniteElement(bound) = accepted.solution()
            else {
                panic!("Q1")
            };
            // For an x-only cubic on a uniform tensor grid, exact load quadrature
            // makes the Q1 nodal equations exact. The solution is its interpolant.
            // Error <= max|u''|h²/8 = (9/32)h² for A=3,L=2.
            assert!(
                bound
                    .field()
                    .l2_error(&|point: &[f64]| 3. * point[0].powi(3) / 8., &quadrature)
                    .unwrap()
                    <= (9. / 32.) / (cells * cells) as f64
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

#[test]
fn zero_coefficient_denominator_fails_before_primal_acceptance() {
    let source = handwritten().replace("parameter L:m=1", "parameter L:m=0");
    let program = program(&source);
    let error = solve_and_linearize_resolved_scalar_elliptic_cartesian(
        &program,
        &resolved(&program, 4),
        &REFERENCE_LINEAR_SOLVER,
        &[SpatialDesignCoordinate::from(amplitude(&program))],
    )
    .unwrap_err();
    assert!(error.message().contains("finite"), "{error:?}");
}

#[test]
fn trial_dependent_denominator_stays_outside_linear_scalar_forms() {
    let source = handwritten().replace("/L^3", "/(L^3*(1+u/1[K]))");
    let program = program(&source);
    let error = solve_and_linearize_resolved_scalar_elliptic_cartesian(
        &program,
        &resolved(&program, 4),
        &REFERENCE_LINEAR_SOLVER,
        &[SpatialDesignCoordinate::from(amplitude(&program))],
    )
    .unwrap_err();
    assert!(
        error.message().contains("linear trial dependence"),
        "{error:?}"
    );
}

#[test]
fn divided_diffusion_coefficient_uses_the_same_checked_q1_path() {
    let source = handwritten()
        .replace("parameter L:m=1;", "parameter L:m=1;parameter divisor:1=2;")
        .replace("-div(grad(u))", "-div((1/divisor)*grad(u))")
        .replace("=-6*A*coordinate(0)/L^3", "=-6*A*coordinate(0)/L^3/divisor");
    let program = program(&source);
    let (_, solution, _) = solve_and_linearize_resolved_scalar_elliptic_cartesian(
        &program,
        &resolved(&program, 8),
        &REFERENCE_LINEAR_SOLVER,
        &[SpatialDesignCoordinate::from(amplitude(&program))],
    )
    .unwrap();
    let ResolvedScalarEllipticCartesianSolution::FiniteElement(solution) = solution else {
        panic!("Q1")
    };
    for vertex in 0..solution.field().mesh().entity_count(0).unwrap() {
        let x = solution
            .field()
            .mesh()
            .vertex_coordinates(MeshEntity::new(0, vertex))
            .unwrap()[0];
        // At N=8,c=1/2,g=2*x^3, rational stencil summation gives
        // ||b||_2^2=2343691/294912 < 3^2. The barrier/CG bound above is
        // (8^2/(8*c))*(3e-12)=4.8e-11, below half this threshold.
        assert!((solution.field().vertex_values()[vertex] - 2. * x.powi(3)).abs() < 2e-9);
    }
}
