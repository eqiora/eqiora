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
    format!("model Main() {{ {DOMAINS}
 instance equation: scalar.Laplace2d(body=body, potential=u, coefficient=2);
 {}
 instance right_flux: scalar.NormalFlux2d(body=body, face=right, potential=u, coefficient=2, flux={flux});
 instance bottom_flux: scalar.NormalFlux2d(body=body, face=bottom, potential=u, coefficient=2, flux=0);
 instance top_flux: scalar.NormalFlux2d(body=body, face=top, potential=u, coefficient=2, flux=0);
 }}", fixed("left", "0"))
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
        // This linear function lies in Q1 exactly. For the uniform unit
        // square, write x for the numeric coordinate x_physical/(1 m)
        // and h=1/N in those normalized coordinates. The Q1 free-node
        // stiffness is an M-matrix. Applying it to nodal
        // phi=x*(2-x)/(2*c) gives the constant-load lumped masses:
        // h^2 inside, h^2/2 on a free edge, h^2/4 at a free corner.
        // phi has zero left value and zero right normal derivative.
        // Comparison with the minimum mass h^2/4 and max(phi)=1/(2*c)
        // gives ||K_ff^-1||_inf <= 2/(c*h^2) = 64 (c=2, h=1/8).
        // Right-flux loads are 2*h (h at its two corners), so ||b||_2<=2.
        // CG requests max(1e-14,1e-12*||b||_2), giving <=1.28e-10 nodal
        // error in exact arithmetic, also bounding L2 on a unit-area box.
        // As in scalar_coefficient_division, allow twice the analytic bound
        // for assembly/solve rounding, not as a floating-point proof.
        let error = solution
            .field()
            .l2_error(
                &|x: &[f64]| slope * x[0],
                &QuadratureRule::tensor_product_gauss_legendre(2, 4).unwrap(),
            )
            .unwrap();
        assert!(error < 2.56e-10, "linear Laplace error {error}");
        let wrong_sign = solution
            .field()
            .l2_error(
                &|x: &[f64]| -slope * x[0],
                &QuadratureRule::tensor_product_gauss_legendre(2, 4).unwrap(),
            )
            .unwrap();
        // ||x/m - (-x/m)||_L2 = 2/sqrt(3) > 1, independently of Eqiora.
        assert!(wrong_sign > 1.0, "opposite sign must be distinguishable");
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
