use eqiora_core::ScalarType;
use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_meshing::QuadratureRule;
use eqiora_numerics::{
    scalar::ResolvedScalarEllipticCartesianSolution,
    scalar::solve_resolved_scalar_elliptic_cartesian,
};
use eqiora_realization::{
    Discretization, DiscretizationMethod, ExecutionSchedule, MeshPolicy, QuadraturePolicy,
    RealizationCapabilities, RealizationPlan, RealizationRequest, RealizationRequirements,
    RealizationRevision, SemanticRevision, Space, Target, VectorLayoutKind, resolve,
};
use eqiora_sem::KernelProgram;
use eqiora_solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};

pub struct ManufacturedCase<'a> {
    pub file: &'a str,
    pub source: &'a str,
    pub expected: &'a str,
    pub dimension: usize,
    pub source_at_center: f64,
    pub maximum_relative_balance: f64,
}

pub fn verify<E>(case: ManufacturedCase<'_>, exact: &E)
where
    E: Fn(&[f64]) -> f64 + ?Sized,
{
    let program = compile_program(case.file, case.source);
    let error_quadrature =
        QuadratureRule::tensor_product_gauss_legendre(case.dimension, 4).unwrap();
    let capabilities = RealizationCapabilities::scalar_elliptic_reference();
    let requirements = RealizationRequirements::new(
        NonZeroUsize::new(case.dimension).unwrap(),
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    );
    let semantic_revision = SemanticRevision::new(program.revision().0);
    let expected = case
        .expected
        .lines()
        .skip(1)
        .map(ExpectedRow::parse)
        .collect::<Vec<_>>();
    assert!(
        expected.len() >= 2,
        "convergence evidence requires two levels"
    );
    assert!(
        expected[0].fem_order.is_none() && expected[0].fvm_order.is_none(),
        "the coarsest evidence row has no observed order"
    );
    assert!(
        expected
            .iter()
            .skip(1)
            .all(|row| row.fem_order.is_some() && row.fvm_order.is_some()),
        "every refined evidence row requires both observed orders"
    );
    let center = vec![0.5; case.dimension];
    let mut rows = Vec::new();

    for (level, expected) in expected.iter().enumerate() {
        let cells = expected.cells;
        let fem = resolve(
            &RealizationRequest::explicit(
                program.model(),
                semantic_revision,
                RealizationRevision::new((2 * level) as u64),
                plan(
                    cells,
                    case.dimension,
                    DiscretizationMethod::ContinuousGalerkin,
                ),
            ),
            requirements,
            &capabilities,
        )
        .unwrap();
        let fvm = resolve(
            &RealizationRequest::explicit(
                program.model(),
                semantic_revision,
                RealizationRevision::new((2 * level + 1) as u64),
                plan(
                    cells,
                    case.dimension,
                    DiscretizationMethod::CellCenteredFiniteVolume,
                ),
            ),
            requirements,
            &capabilities,
        )
        .unwrap();
        let (fem_model, fem_solution) =
            solve_resolved_scalar_elliptic_cartesian(&program, &fem, &REFERENCE_LINEAR_SOLVER)
                .unwrap();
        let (fvm_model, fvm_solution) =
            solve_resolved_scalar_elliptic_cartesian(&program, &fvm, &REFERENCE_LINEAR_SOLVER)
                .unwrap();
        assert_eq!(fem_model, fvm_model);
        assert_eq!(fem_model.dimension(), case.dimension);
        assert_eq!(fem_model.source().coordinate_dimension(), case.dimension);
        assert!(
            relative_difference(
                fem_model.source().evaluate(&center).unwrap(),
                case.source_at_center,
            ) < 2.0e-14
        );
        let ResolvedScalarEllipticCartesianSolution::FiniteElement(fem_solution) = fem_solution
        else {
            panic!("FEM plan returned a different method")
        };
        let ResolvedScalarEllipticCartesianSolution::FiniteVolume(fvm_solution) = fvm_solution
        else {
            panic!("FVM plan returned a different method")
        };
        let fem_error = fem_solution
            .field()
            .l2_error(exact, &error_quadrature)
            .unwrap();
        let fvm_error = fvm_solution
            .reconstruction()
            .l2_error(exact, &error_quadrature)
            .unwrap();
        let fem_balance = relative_balance(
            fem_solution.boundary_reaction_sum(),
            fem_solution.integrated_source(),
        );
        let fvm_balance = relative_balance(
            fvm_solution.boundary_flux_sum(),
            fvm_solution.integrated_source(),
        );
        rows.push((cells, fem_error, fvm_error, fem_balance, fvm_balance));
    }

    for (row, expected) in rows.iter().zip(&expected) {
        assert!(relative_difference(1.0 / row.0 as f64, expected.h_max) < 1.0e-14);
        assert!(
            relative_difference(row.1, expected.fem_l2) < 1.0e-12,
            "FEM L2 mismatch: actual={}, expected={}",
            row.1,
            expected.fem_l2
        );
        assert!(
            relative_difference(row.2, expected.fvm_l2) < 1.0e-12,
            "FVM L2 mismatch: actual={}, expected={}",
            row.2,
            expected.fvm_l2
        );
        assert!((row.3 - expected.fem_balance).abs() < 1.0e-13);
        assert!((row.4 - expected.fvm_balance).abs() < 1.0e-13);
    }
    for (pair, expected) in rows.windows(2).zip(expected.iter().skip(1)) {
        let fem_order = (pair[0].1 / pair[1].1).log2();
        let fvm_order = (pair[0].2 / pair[1].2).log2();
        assert!((fem_order - expected.fem_order.unwrap()).abs() < 1.0e-10);
        assert!((fvm_order - expected.fvm_order.unwrap()).abs() < 1.0e-10);
        assert!(fem_order > 1.9, "FEM rows: {pair:?}, order={fem_order}");
        assert!(fvm_order > 1.9, "FVM rows: {pair:?}, order={fvm_order}");
    }
    assert!(
        rows.windows(2)
            .all(|pair| pair[1].1 < pair[0].1 && pair[1].2 < pair[0].2)
    );
    assert!(
        rows.iter().all(|row| {
            row.3 < case.maximum_relative_balance && row.4 < case.maximum_relative_balance
        }),
        "balance rows: {rows:?}"
    );
}

fn plan(cells: usize, dimension: usize, method: DiscretizationMethod) -> RealizationPlan {
    let (space, quadrature) = match method {
        DiscretizationMethod::ContinuousGalerkin => (
            Space::continuous_lagrange(NonZeroU16::MIN),
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(2).unwrap(),
            },
        ),
        DiscretizationMethod::CellCenteredFiniteVolume => {
            (Space::cell_constant(), QuadraturePolicy::CellCentroid)
        }
    };
    let exponent = u32::try_from(dimension).unwrap();
    let unknown_scale = cells.checked_pow(exponent).unwrap();
    RealizationPlan::new(
        space,
        Discretization::new(
            method,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(cells).unwrap(),
            },
            quadrature,
        ),
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(unknown_scale.checked_mul(8).unwrap()).unwrap(),
        )
        .unwrap(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap()
}

fn compile_program(file: &str, source: &str) -> KernelProgram {
    let mut compiled = compile(file, source).unwrap();
    let (transaction, model, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    KernelProgram::from_snapshot(&store.snapshot(), model).unwrap()
}

fn relative_balance(boundary: f64, source: f64) -> f64 {
    (boundary + source).abs() / (boundary.abs() + source.abs()).max(f64::MIN_POSITIVE)
}

fn relative_difference(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE)
}

struct ExpectedRow {
    cells: usize,
    h_max: f64,
    fem_l2: f64,
    fem_order: Option<f64>,
    fvm_l2: f64,
    fvm_order: Option<f64>,
    fem_balance: f64,
    fvm_balance: f64,
}

impl ExpectedRow {
    fn parse(line: &str) -> Self {
        let columns = line.split(',').collect::<Vec<_>>();
        assert_eq!(columns.len(), 8, "unexpected evidence row: {line}");
        Self {
            cells: columns[0].parse().unwrap(),
            h_max: columns[1].parse().unwrap(),
            fem_l2: columns[2].parse().unwrap(),
            fem_order: parse_optional(columns[3]),
            fvm_l2: columns[4].parse().unwrap(),
            fvm_order: parse_optional(columns[5]),
            fem_balance: columns[6].parse().unwrap(),
            fvm_balance: columns[7].parse().unwrap(),
        }
    }
}

fn parse_optional(value: &str) -> Option<f64> {
    (!value.is_empty()).then(|| value.parse().unwrap())
}
