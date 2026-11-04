//! Imported fixed-unit scalar components against independent analytic solutions.
use eqiora::meshing::QuadratureRule;
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageDependencyV1, PackageManifestV1, PackageReleaseV1, PackageSourcesV1,
    PackagedModelDocument, QualifiedName, ResolutionRecordV1, SourceFileV1,
    prepare_package_release_v1,
};
use eqiora::realization::*;
use eqiora::solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};
use eqiora_core::ScalarType;
use eqiora_numerics::scalar::{
    ResolvedScalarEllipticCartesianSolution, lower_scalar_elliptic_cartesian,
    solve_resolved_scalar_elliptic_cartesian,
};
use std::f64::consts::PI;
use std::num::{NonZeroU16, NonZeroUsize};

#[path = "support/embedded_package.rs"]
mod embedded_package;

fn scalar_release() -> PackageReleaseV1 {
    prepare_package_release_v1(
        embedded_package::sources(
            include_bytes!("../../eqiora-api/packages/Eqiora.Scalar/package.json"),
            &[
                (
                    "README.md",
                    BundleRoleV1::Documentation,
                    include_bytes!("../../eqiora-api/packages/Eqiora.Scalar/README.md"),
                ),
                (
                    "src/scalar.eqi",
                    BundleRoleV1::ModelSource,
                    include_bytes!("../../eqiora-api/packages/Eqiora.Scalar/src/scalar.eqi"),
                ),
            ],
        ),
        &[],
    )
    .unwrap()
}

fn root_sources(release: &PackageReleaseV1, source: &str) -> PackageSourcesV1 {
    let path = NormalizedRelativePath::parse("src/main.eqi").unwrap();
    let manifest = PackageManifestV1::new(
        "main",
        QualifiedName::parse("org.example.StandardScalar").unwrap(),
        ExactVersion::parse("0.1.0").unwrap(),
        vec![PackageDependencyV1::new(
            release.package_identity().unwrap(),
        )],
        vec![BundleEntryV1::new(path.clone(), BundleRoleV1::ModelSource)],
    )
    .unwrap();
    PackageSourcesV1::new(
        manifest,
        vec![SourceFileV1::new(
            path,
            BundleRoleV1::ModelSource,
            format!("import Eqiora.Scalar.scalar as scalar;\n{source}").into_bytes(),
        )],
    )
    .unwrap()
}
fn compile(source: &str) -> PackagedModelDocument {
    let scalar = scalar_release();
    let root =
        prepare_package_release_v1(root_sources(&scalar, source), std::slice::from_ref(&scalar))
            .unwrap();
    let resolution =
        ResolutionRecordV1::from_exact_releases(&root, std::slice::from_ref(&scalar)).unwrap();
    let mut store = InMemoryPackageStore::default();
    store.insert(&scalar).unwrap();
    store.insert(&root).unwrap();
    let compiled = PackagedModelDocument::compile_locked(&store, &resolution, "Main").unwrap();
    compiled
        .compilation()
        .validate_against(&resolution)
        .unwrap();
    compiled
}

const DOMAINS: &str = r#"
 domain body = box(0, 1, 0, 1);
 domain left = boundary(body, axis = 0, side = lower);
 domain right = boundary(body, axis = 0, side = upper);
 domain bottom = boundary(body, axis = 1, side = lower);
 domain top = boundary(body, axis = 1, side = upper);
 variable u: 1 on body in h1;
"#;
fn fixed(face: &str, value: &str) -> String {
    format!(
        "instance {face}_value: scalar.FixedValue2d(body=body, face={face}, potential=u, value={value});"
    )
}
fn poisson(override_coefficient: bool) -> String {
    let coefficient = if override_coefficient {
        ", coefficient=2"
    } else {
        ""
    };
    let scale = if override_coefficient { "4" } else { "2" };
    format!(
        "model Main() {{ {DOMAINS}
 parameter wave: 1/m = 3.141592653589793;
 instance equation: scalar.Poisson2d(body=body, potential=u{coefficient},
 source={scale}*wave^2*math.sin(wave*coordinate(0))*math.sin(wave*coordinate(1)));
 {} }}",
        ["left", "right", "bottom", "top"]
            .map(|face| fixed(face, "0"))
            .join("\n")
    )
}
fn laplace(flux: &str) -> String {
    format!("model Main() {{ {}
 instance equation: scalar.Laplace2d(body=body, potential=u, coefficient=2);
 {}
 instance right_flux: scalar.NormalFlux2d(body=body, face=right, potential=u, coefficient=2, flux={flux});
 instance bottom_flux: scalar.NormalFlux2d(body=body, face=bottom, potential=u, coefficient=2, flux=0[1/m]);
 instance top_flux: scalar.NormalFlux2d(body=body, face=top, potential=u, coefficient=2, flux=0[1/m]);
 }}", DOMAINS.replace("in h1", "in smooth"), fixed("left", "0"))
}

// Same independent manufactured solution, convergence floor and balance budget as
// compiled_package_poisson_form and verify/numerics/cartesian-poisson-fem-fvm.
#[test]
fn imported_poisson_defaults_and_override_converge_to_the_sine_solution() {
    for override_coefficient in [false, true] {
        let document = compile(&poisson(override_coefficient));
        let mut errors = Vec::new();
        for cells in [4, 8, 16, 32] {
            let ResolvedScalarEllipticCartesianSolution::FiniteElement(solution) =
                solve(&document, cells)
            else {
                panic!("expected Q1 FEM");
            };
            let source = solution.integrated_source();
            let reaction = solution.boundary_reaction_sum();
            assert!((reaction + source).abs() / (reaction.abs() + source.abs()) <= 2e-11);
            errors.push(
                solution
                    .field()
                    .l2_error(
                        &|x: &[f64]| (PI * x[0]).sin() * (PI * x[1]).sin(),
                        &QuadratureRule::tensor_product_gauss_legendre(2, 4).unwrap(),
                    )
                    .unwrap(),
            );
        }
        for pair in errors.windows(2) {
            assert!((pair[0] / pair[1]).log2() >= 1.9, "{errors:?}");
        }
    }
}

#[test]
fn imported_laplace_uses_outward_constitutive_flux_with_the_authored_sign() {
    for (flux, slope) in [("2[1/m]", 1.0), ("-2[1/m]", -1.0)] {
        let document = compile(&laplace(flux));
        let ResolvedScalarEllipticCartesianSolution::FiniteElement(solution) = solve(&document, 8)
        else {
            panic!("expected Q1 FEM");
        };
        // The authored right normal flux +/-2/m, c=2, left value zero,
        // and zero top/bottom flux give the unique solution u=+/-x/m.
        // Certify against an independently assembled exact integer stencil,
        // including any error in the production assembly and solve.
        let mut values = [0.0; 81];
        for (vertex, value) in solution.field().vertex_values().iter().enumerate() {
            let index = solution
                .field()
                .mesh()
                .vertex_multi_index(eqiora::meshing::MeshEntity::new(0, vertex))
                .unwrap();
            values[index[0] + 9 * index[1]] = *value;
        }
        certify_linear_laplace(&values, slope);
    }
}

// Independent Q1 unit-square stencil: exact integration of bilinear basis
// gradients gives (c/6)*[[4,-1,-2,-1], ...] in cyclic corner order.
// Coordinates below mean x_physical/(1 m), h=1/8. Thus A=6K/c=3K is
// integer, and the prescribed scaled right-flux load is dyadic exact.
fn certify_linear_laplace(values: &[f64; 81], slope: f64) {
    assert!(
        values.iter().all(|value| value.is_finite()),
        "certificate requires finite nodal values"
    );
    assert_eq!(
        slope.abs(),
        1.0,
        "certificate covers the two authored flux signs"
    );
    let mut matrix = [[0_i32; 81]; 81];
    let local = [
        [4, -1, -2, -1],
        [-1, 4, -1, -2],
        [-2, -1, 4, -1],
        [-1, -2, -1, 4],
    ];
    for y in 0..8 {
        for x in 0..8 {
            let vertices = [
                x + 9 * y,
                x + 1 + 9 * y,
                x + 1 + 9 * (y + 1),
                x + 9 * (y + 1),
            ];
            for i in 0..4 {
                for j in 0..4 {
                    matrix[vertices[i]][vertices[j]] += local[i][j];
                }
            }
        }
    }
    let mut matrix_norm = 0.0_f64;
    let mut load_norm = 0.0_f64;
    let mut maximum_residual = 0.0_f64;
    let mut maximum_error = 0.0_f64;
    for y in 0..9 {
        assert_eq!(values[9 * y], 0.0, "essential left trace");
        for x in 1..9 {
            let row = x + 9 * y;
            let load = if x == 8 {
                slope * if y == 0 || y == 8 { 0.375 } else { 0.75 }
            } else {
                0.0
            };
            let mut residual = -load;
            let mut row_sum = 0_i32;
            let mut terms = 0;
            for column in 0..81 {
                if column % 9 == 0 {
                    continue;
                } // exactly eliminated zero trace
                let entry = matrix[row][column];
                if entry != 0 {
                    residual += f64::from(entry) * values[column];
                    row_sum += entry.abs();
                    terms += 1;
                }
            }
            assert!(terms <= 9);
            matrix_norm = matrix_norm.max(f64::from(row_sum));
            load_norm = load_norm.max(load.abs());
            assert!(
                residual.is_finite(),
                "independent residual must remain finite"
            );
            maximum_residual = maximum_residual.max(residual.abs());
            maximum_error = maximum_error.max((values[row] - slope * (x as f64) / 8.0).abs());
        }
    }
    // For K, nodal phi=x(2-x)/(2c) has K*phi equal to constant-load
    // masses (h² interior, h²/2 edge, h²/4 corner), max(phi)=1/(2c).
    // The M-matrix comparison bound is ||K^-1||inf <=2/(c*h²)=64.
    // Since A=3K, ||A^-1||inf <= B=64/3. The independent residual
    // check includes production assembly error rather than assuming it away.
    let inverse_bound = (64.0_f64 / 3.0).next_up();
    let residual_limit = 6e-12; // Three times the requested physical CG bound.
    assert!(
        maximum_residual <= residual_limit,
        "independent residual {maximum_residual}"
    );
    // At most 9 products and 9 additions, with exact integer coefficients
    // and dyadic load, are bounded by gamma20. With ||u_exact||inf=1,
    // e <= B[R + gamma20*(||A||inf*(1+e)+||b||inf)]. Rearranging gives
    // this output-independent error budget. All norm inputs come from the
    // independent stencil/load above, never the computed solution.
    // Each positive intermediate is explicitly rounded outward: numerator
    // factors up, subtractive denominators down. A final next_up alone
    // would not cover accumulated rounding in the bound evaluation.
    let roundoff = f64::EPSILON / 2.0; // Exact binary operation.
    let twenty_roundoff = 20.0 * roundoff; // Exact representable product.
    let gamma_denominator = (1.0 - twenty_roundoff).next_down();
    assert!(gamma_denominator > 0.0);
    let gamma = (twenty_roundoff / gamma_denominator).next_up();
    let norm_sum = (matrix_norm + load_norm).next_up();
    let residual_rounding = (gamma * norm_sum).next_up();
    let residual_bound = (residual_limit + residual_rounding).next_up();
    let numerator = (inverse_bound * residual_bound).next_up();
    let scaled_matrix_rounding = (gamma * matrix_norm).next_up();
    let feedback = (inverse_bound * scaled_matrix_rounding).next_up();
    let denominator = (1.0 - feedback).next_down();
    assert!(denominator > 0.0);
    let bound = (numerator / denominator).next_up();
    assert!(
        maximum_error <= bound,
        "nodal error {maximum_error} exceeds {bound}"
    );
    // Q1 shape functions are nonnegative and sum to one: this nodal bound
    // also bounds field error everywhere, without numerical L2 quadrature.
    // Opposite-sign solutions differ by 2 at x=1; no sign is rejected as
    // invalid merely because it describes a different well-posed problem.
    assert!((values[8] + slope).abs() > 1.0);
}

#[test]
fn independent_laplace_certificate_rejects_nonfinite_nodal_values() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut values = [0.0; 81];
        values[40] = invalid;
        assert!(std::panic::catch_unwind(|| certify_linear_laplace(&values, 1.0)).is_err());
    }
}

#[test]
fn units_and_foreign_parent_bindings_reject_before_numerical_lowering() {
    let scalar = scalar_release();
    for source in [
        poisson(false).replace("variable u: 1", "variable u: K"),
        poisson(true).replace("coefficient=2", "coefficient=2[K]"),
        poisson(false).replace("box(0, 1, 0, 1)", "box(0, 1, 0, 1, 0, 1)"),
        poisson(false).replace("source=2*wave^2", "source=2[K]*wave^2"),
        poisson(false).replace("value=0", "value=1[K]"),
        laplace("2[K/m]"),
        laplace("2[1/m]").replace("in smooth", "in h1"),
        laplace("2[1/m]").replace("variable u:", "domain foreign = box(0, 1, 0, 1); domain other = boundary(foreign, axis=0, side=upper); variable u:")
            .replace("face=right", "face=other"),
    ] {
        let error = prepare_package_release_v1(root_sources(&scalar, &source), std::slice::from_ref(&scalar))
            .expect_err("invalid typed binding must reject");
        assert!(matches!(error, eqiora::package::PackagePreparationError::Diagnostics(ref diagnostics)
            if !diagnostics.is_empty()), "unexpected rejection: {error}");
    }
}

#[test]
fn flux_coefficient_must_match_the_volume_constitutive_law() {
    let document = compile(&laplace("2[1/m]").replace(
        "face=right, potential=u, coefficient=2",
        "face=right, potential=u, coefficient=3",
    ));
    assert!(lower_scalar_elliptic_cartesian(document.model().program()).is_err());
}

fn solve(
    document: &PackagedModelDocument,
    cells: usize,
) -> ResolvedScalarEllipticCartesianSolution {
    let program = document.model().program();
    let resolved = resolved(program, cells);
    let (_, solution) =
        solve_resolved_scalar_elliptic_cartesian(program, &resolved, &REFERENCE_LINEAR_SOLVER)
            .unwrap();
    solution
}

fn resolved(program: &eqiora::sem::KernelProgram, cells: usize) -> ResolvedRealization {
    resolved_with_points(program, cells, 2)
}

fn resolved_with_points(
    program: &eqiora::sem::KernelProgram,
    cells: usize,
    points_per_axis: usize,
) -> ResolvedRealization {
    let plan = RealizationPlan::new(
        Space::continuous_lagrange(NonZeroU16::MIN),
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(cells).unwrap(),
            },
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(points_per_axis).unwrap(),
            },
        ),
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(cells * cells * 8).unwrap(),
        )
        .unwrap(),
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
