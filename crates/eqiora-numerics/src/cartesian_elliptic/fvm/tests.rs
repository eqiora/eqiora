use eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND;

use super::*;
use eqiora_solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};
use std::num::NonZeroUsize;

#[test]
fn complex_reaction_diffusion_tpfa_reproduces_affine_values_with_trace_or_flux() {
    use num_complex::Complex64 as C;

    let mesh = CartesianMesh::from_axes(vec![vec![0., 1., 3.]]).unwrap();
    let cell = QuadratureRule::gauss_legendre(1).unwrap();
    let facet = QuadratureRule::point();
    let exact = |x: f64| C::new(1. + 2. * x, 1. - x);
    for natural_right in [false, true] {
        let boundary = |_: usize, side, x: &[f64]| {
            Ok(if natural_right && side == BoundarySide::Upper {
                // a*u' = (2+3i)(2-i) = 7+4i at the outward upper face.
                CartesianBoundaryValue::Natural(C::new(7., 4.))
            } else {
                CartesianBoundaryValue::Essential(exact(x[0]))
            })
        };
        let assemble = |properties| {
            finalize_elliptic_cartesian_fvm(
                &mesh,
                1,
                &|_| Ok(C::new(2., 3.)),
                &|_, _, _| Ok(C::new(3., -2.)),
                // -a*u'' + q*u = q*u for this affine u.
                &|_, x| Ok(C::new(3., -2.) * exact(x[0])),
                &|_, axis, side, x| boundary(axis, side, x),
                &cell,
                &facet,
                &REFERENCE_ASSEMBLY_BACKEND,
                properties,
            )
        };
        let system = assemble(LinearOperatorProperties::General)
            .unwrap()
            .canonical_system()
            .unwrap();
        let solver = SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1e-13,
            1e-14,
            NonZeroUsize::new(32).unwrap(),
        )
        .unwrap();
        let result = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, solver)
            .solve(&system.linear_problem().unwrap())
            .unwrap();
        for (&value, coordinate) in result.values().iter().zip([0.5, 2.]) {
            assert!((value - exact(coordinate)).norm() < 1e-12);
        }
        assert!(
            assemble(LinearOperatorProperties::HermitianPositiveDefinite)
                .unwrap_err()
                .message()
                .contains("positive-definite")
        );
    }
}

#[test]
fn local_packets_preserve_complex_balance_and_physical_measure() {
    use num_complex::Complex64 as C;

    let mesh = CartesianMesh::from_axes(vec![vec![2., 5.]]).unwrap();
    let geometry = mesh.geometry_map(MeshEntity::new(1, 0)).unwrap();
    let source = |_: usize, x: &[f64]| Ok(C::new(x[0] * x[0], 2. * x[0] - 1.));
    let reaction = |_: usize, _: usize, x: &[f64]| Ok(C::new(2. * x[0], -x[0] * x[0]));
    let local = CartesianVolumeCell {
        components: 1,
        reaction: &reaction,
        source: &source,
    }
    .evaluate(&geometry, &QuadratureRule::gauss_legendre(2).unwrap())
    .unwrap();
    // Integral from 2 to 5: x^2 gives 39, and 2x-1 gives 18.
    assert!((local.rhs()[0] - C::new(39., 18.)).norm() < 1e-12);
    // Integral of 2x - i*x² on the same physical cell.
    assert!((local.entry(0, 0).unwrap() - C::new(21., -39.)).norm() < 1e-12);

    let transmissibility = C::new(2., -3.);
    let values = [C::new(4., 1.), C::new(1., -2.)];
    let interior = CartesianInteriorFlux
        .evaluate(&transmissibility, &QuadratureRule::point())
        .unwrap();
    let action = |local: &LocalContribution<C>, row| {
        local.entry(row, 0).unwrap() * values[0] + local.entry(row, 1).unwrap() * values[1]
    };
    // (2-3i) [(4+i)-(1-2i)] = 15-3i, without conjugating the flux law.
    assert_eq!(action(&interior, 0), C::new(15., -3.));
    assert_eq!(action(&interior, 0) + action(&interior, 1), C::new(0., 0.));
    let boundary = CartesianBoundaryFlux
        .evaluate(&transmissibility, &QuadratureRule::point())
        .unwrap();
    assert_eq!(action(&boundary, 0), C::new(15., -3.));

    let invalid_source = |_: usize, _: &[f64]| Ok(C::new(1., f64::NAN));
    assert!(
        CartesianVolumeCell {
            components: 1,
            reaction: &reaction,
            source: &invalid_source
        }
        .evaluate(&geometry, &QuadratureRule::gauss_legendre(2).unwrap())
        .is_err()
    );
}

#[test]
fn coupled_volume_blocks_preserve_nonsymmetric_real_and_complex_component_maps() {
    use num_complex::Complex64 as C;

    // u = [1+2x, 3-x], Q = [[2,3],[5,7]], so Q*u = [11+x, 26+3x].
    coupled_volume_case(
        [[2., 3.], [5., 7.]],
        2.,
        |component, x| [1. + 2. * x, 3. - x][component],
        |component, x| [11. + x, 26. + 3. * x][component],
        &REFERENCE_LINEAR_SOLVER,
    );
    // Direct complex multiplication gives the two affine sources below.
    // Unequal off-diagonal phases distinguish transpose and conjugation errors.
    coupled_volume_case(
        [
            [C::new(2., 1.), C::new(3., -2.)],
            [C::new(5., 4.), C::new(7., -3.)],
        ],
        C::new(2., 3.),
        |component, x| [C::new(1. + 2. * x, 2. - x), C::new(3. - x, 1. + 3. * x)][component],
        |component, x| {
            [
                C::new(11. + 8. * x, 2. + 11. * x),
                C::new(21. + 16. * x, 12. + 27. * x),
            ][component]
        },
        &REFERENCE_LINEAR_SOLVER,
    );
}

fn coupled_volume_case<S>(
    reaction: [[S; 2]; 2],
    diffusion: S,
    exact: impl Fn(usize, f64) -> S,
    source: impl Fn(usize, f64) -> S + Sync,
    backend: &dyn eqiora_solver::LinearSolverBackend<S>,
) where
    S: eqiora_core::Scalar + num_complex::ComplexFloat<Real = f64> + From<f64> + Send + Sync,
{
    let mesh = CartesianMesh::from_axes(vec![vec![2., 3., 5.]]).unwrap();
    let assembly = finalize_elliptic_cartesian_fvm(
        &mesh,
        2,
        &|_| Ok(diffusion),
        &|row, column, _| Ok(reaction[row][column]),
        &|component, x| Ok(source(component, x[0])),
        &|component, _, _, x| Ok(CartesianBoundaryValue::Essential(exact(component, x[0]))),
        &QuadratureRule::gauss_legendre(1).unwrap(),
        &QuadratureRule::point(),
        &REFERENCE_ASSEMBLY_BACKEND,
        LinearOperatorProperties::General,
    )
    .unwrap();
    // Each off-diagonal cell block entry is its physical measure times Q_ij.
    // Diffusion acts within a component and cannot contribute to these entries.
    for (cell, measure) in [1., 2.].into_iter().enumerate() {
        for (row, column) in [(0, 1), (1, 0)] {
            let actual = assembly
                .linear_system
                .matrix()
                .entry(2 * cell + row, 2 * cell + column)
                .unwrap();
            assert!(
                (actual - <S as From<f64>>::from(measure) * reaction[row][column]).abs() < 1e-12
            );
        }
    }
    let system = assembly.canonical_system().unwrap();
    let solver = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-13,
        1e-13,
        NonZeroUsize::new(128).unwrap(),
    )
    .unwrap();
    let solved = LinearSolveRequest::new(backend, solver)
        .solve(&system.linear_problem().unwrap())
        .unwrap();
    for (cell, center) in [2.5, 4.].into_iter().enumerate() {
        for component in 0..2 {
            assert!(
                (solved.values()[2 * cell + component] - exact(component, center)).abs() < 1e-10
            );
        }
    }
}

#[test]
fn volume_component_dimensions_fail_before_callbacks() {
    let mesh = CartesianMesh::from_axes(vec![vec![2., 5.]]).unwrap();
    let geometry = mesh.geometry_map(MeshEntity::new(1, 0)).unwrap();
    for components in [0, usize::MAX] {
        let result: Result<LocalContribution<f64>, _> = CartesianVolumeCell {
            components,
            reaction: &|_, _, _: &[f64]| panic!("invalid dimensions must fail before evaluation"),
            source: &|_, _: &[f64]| panic!("invalid dimensions must fail before evaluation"),
        }
        .evaluate(&geometry, &QuadratureRule::gauss_legendre(1).unwrap());
        assert!(result.unwrap_err().message().contains("invalid dimensions"));
    }
}

#[test]
fn tpfa_preserves_coefficient_and_boundary_evaluation_diagnostics() {
    let mesh = CartesianMesh::from_axes(vec![vec![0., 1.]]).unwrap();
    let cell = QuadratureRule::gauss_legendre(1).unwrap();
    let facet = QuadratureRule::point();
    let problem = |failure: usize| {
        finalize_elliptic_cartesian_fvm(
            &mesh,
            1,
            &|_| {
                if failure == 0 {
                    Err(invalid("diffusion evaluation failed"))
                } else {
                    Ok(1.)
                }
            },
            &|_, _, _| {
                if failure == 1 {
                    Err(invalid("reaction evaluation failed"))
                } else {
                    Ok(0.)
                }
            },
            &|_, _| {
                if failure == 2 {
                    Err(invalid("source evaluation failed"))
                } else {
                    Ok(0.)
                }
            },
            &|_, _, _, _| {
                if failure == 3 {
                    Err(invalid("boundary evaluation failed"))
                } else {
                    Ok(CartesianBoundaryValue::Essential(0.))
                }
            },
            &cell,
            &facet,
            &REFERENCE_ASSEMBLY_BACKEND,
            LinearOperatorProperties::General,
        )
    };
    for (index, owner) in ["diffusion", "reaction", "source", "boundary"]
        .iter()
        .enumerate()
    {
        assert!(
            problem(index)
                .unwrap_err()
                .message()
                .contains(&format!("{owner} evaluation failed"))
        );
    }
}

fn solve(
    axes: Vec<f64>,
    source: f64,
    fluxes: [f64; 2],
    mean: Option<f64>,
    algorithm: LinearSolver,
) -> Result<ScalarEllipticCartesianFvmSolution, Diagnostic> {
    let mesh = CartesianMesh::from_axes(vec![axes])?;
    let assembly = finalize_elliptic_cartesian_fvm(
        &mesh,
        1,
        &|_| Ok(1.),
        &|_, _, _| Ok(0.),
        &|_, _| Ok(source),
        &|_, _, side, _| {
            Ok(CartesianBoundaryValue::Natural(match side {
                BoundarySide::Lower => fluxes[0],
                BoundarySide::Upper => fluxes[1],
            }))
        },
        &QuadratureRule::gauss_legendre(1)?,
        &QuadratureRule::point(),
        &REFERENCE_ASSEMBLY_BACKEND,
        eqiora_solver::LinearOperatorProperties::SymmetricPositiveDefinite,
    )?;
    let (system, state) = assembly.into_canonical(mean)?;
    assert_eq!(system.properties(), LinearOperatorProperties::Symmetric);
    let plan = SolverPlan::new(algorithm, 1e-12, 1e-12, NonZeroUsize::new(128).unwrap())?;
    state.solve(
        LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan),
        system,
    )
}

#[test]
fn affine_neumann_tpfa_uses_cell_measure_reference_and_preserves_gradient() {
    // u=2x-1+c, integral(u)=c, outward grad(u)=[-2,2].
    let axes = vec![0., 0.125, 0.25, 0.5, 1.];
    for mean in [0., 3.] {
        let solved = solve(
            axes.clone(),
            0.,
            [-2., 2.],
            Some(mean),
            LinearSolver::MinimumResidual,
        )
        .unwrap();
        for (center, value) in solved.cell_centers().iter().zip(solved.cell_values()) {
            assert!((value - (2. * center[0] - 1. + mean)).abs() < 1e-10);
        }
        assert!(
            solved
                .reconstruction()
                .l2_error(
                    &|x| 2. * x[0] - 1. + mean,
                    &QuadratureRule::gauss_legendre(2).unwrap()
                )
                .unwrap()
                < 1e-10
        );
        assert!(solved.original_residual_norm() < 1e-10);
        assert_eq!(solved.compatibility_residual(), Some(0.));
        assert!(solved.gauge_residual().unwrap().abs() < 1e-10);
        assert!(solved.gauge_multiplier().unwrap().abs() < 1e-10);
    }
}

#[test]
fn polynomial_neumann_tpfa_has_independently_derived_discrete_mean() {
    // u=x²-x+1/6 has -u''=-2, conormal endpoint loads [1,1].
    // Midpoint quadrature underestimates its mean by h²/12, so the
    // zero-integral cell-constant solution adds h²/12 to each midpoint value.
    for cells in [2, 4, 8] {
        let h = 1. / cells as f64;
        let solved = solve(
            (0..=cells).map(|i| i as f64 * h).collect(),
            -2.,
            [1., 1.],
            Some(0.),
            LinearSolver::MinimumResidual,
        )
        .unwrap();
        for (center, value) in solved.cell_centers().iter().zip(solved.cell_values()) {
            let x = center[0];
            assert!((value - (x * x - x + 1. / 6. + h * h / 12.)).abs() < 1e-10);
        }
        assert!((solved.integrated_source() + solved.boundary_flux_sum()).abs() < 1e-12);
        assert!(solved.cell_values().iter().sum::<f64>().abs() * h < 1e-10);
    }
}

#[test]
fn neumann_tpfa_never_repairs_missing_reference_incompatible_load_or_solver() {
    let axes = vec![0., 0.25, 0.5, 0.75, 1.];
    for (source, flux, mean, solver) in [
        (0., [-2., 2.], None, LinearSolver::MinimumResidual),
        (1., [-2., 2.], Some(0.), LinearSolver::MinimumResidual),
        (-2., [-1., 1.], Some(0.), LinearSolver::MinimumResidual),
        (0., [-2., 2.], Some(0.), LinearSolver::ConjugateGradient),
        (0., [-2., 2.], Some(f64::NAN), LinearSolver::MinimumResidual),
    ] {
        assert!(solve(axes.clone(), source, flux, mean, solver).is_err());
    }
}
