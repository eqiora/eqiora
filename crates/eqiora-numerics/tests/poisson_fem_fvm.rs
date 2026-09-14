use eqiora_core::ScalarType;
use std::f64::consts::PI;
use std::num::NonZeroUsize;

use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_numerics::{
    scalar::ResolvedScalarEllipticSolution1d, scalar::compare_canonical_scalar_elliptic_1d,
    scalar::solve_resolved_scalar_elliptic_1d,
};
use eqiora_realization::{
    DefaultPolicyVersion, Discretization, DiscretizationMethod, ExecutionSchedule, MeshPolicy,
    QuadraturePolicy, RealizationCapabilities, RealizationPlan, RealizationRequest,
    RealizationRequirements, RealizationRevision, SemanticRevision, Space, Target,
    VectorLayoutKind, default_plan_v0, resolve,
};
use eqiora_sem::KernelProgram;
use eqiora_solver::REFERENCE_LINEAR_SOLVER;

const EXPECTED: &str =
    include_str!("../../../verify/numerics/poisson-fem-fvm/expected/convergence.csv");
const SOURCE: &str = include_str!("../../../verify/numerics/poisson-fem-fvm/models/poisson.eqi");

#[test]
fn canonical_poisson_meets_fem_fvm_convergence_and_balance_contracts() {
    let program = compile_program("verify/numerics/poisson-fem-fvm/models/poisson.eqi", SOURCE);

    let expected = EXPECTED
        .lines()
        .skip(1)
        .map(ExpectedRow::parse)
        .collect::<Vec<_>>();
    let levels = expected.iter().map(|row| row.cells).collect::<Vec<_>>();
    let (model, report) = compare_canonical_scalar_elliptic_1d(&program, &levels, &|coordinate| {
        (PI * coordinate).sin()
    })
    .expect("one canonical model realizes through both methods");

    assert!(model.source().is_coordinate_dependent());
    assert!(
        (model.source().evaluate(&[0.5]).unwrap() - PI.powi(2)).abs() < 2.0e-15,
        "lowered source must retain the canonical coordinate expression"
    );

    assert_eq!(report.len(), expected.len());
    for (row, expected) in report.iter().zip(expected) {
        assert!(relative_difference(row.max_cell_measure, expected.h_max) < 1.0e-12);
        assert!(relative_difference(row.fem_l2_error, expected.fem_l2) < 1.0e-8);
        assert!(relative_difference(row.fvm_l2_error, expected.fvm_l2) < 1.0e-8);
        assert_optional_close(row.fem_order, expected.fem_order, 1.0e-6);
        assert_optional_close(row.fvm_order, expected.fvm_order, 1.0e-6);
        if let Some(order) = row.fem_order {
            assert!(order > 1.9, "FEM row: {row:?}");
        }
        if let Some(order) = row.fvm_order {
            assert!(order > 1.9, "FVM row: {row:?}");
        }
        assert!(row.fem_relative_balance_error < 2.0e-12, "{row:?}");
        assert!(row.fvm_relative_balance_error < 2.0e-12, "{row:?}");
        assert!(expected.fem_balance < 2.0e-12);
        assert!(expected.fvm_balance < 2.0e-12);
    }
    assert!(report.windows(2).all(|pair| {
        pair[1].fem_l2_error < pair[0].fem_l2_error && pair[1].fvm_l2_error < pair[0].fvm_l2_error
    }));
}

#[test]
fn one_poisson_revision_executes_through_two_realization_plans() {
    let program = compile_program("verify/numerics/poisson-fem-fvm/models/poisson.eqi", SOURCE);
    let semantic_revision = SemanticRevision::new(program.revision().0);
    let capabilities = RealizationCapabilities::scalar_elliptic_reference();
    let requirements = RealizationRequirements::new(
        NonZeroUsize::MIN,
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    );
    let default = resolve(
        &RealizationRequest::default(program.model(), semantic_revision, DefaultPolicyVersion::V0),
        requirements,
        &capabilities,
    )
    .expect("default FEM realization resolves");

    let default_plan = default_plan_v0().expect("frozen default plan is internally valid");
    let fvm_plan = RealizationPlan::new(
        Space::cell_constant(),
        Discretization::new(
            DiscretizationMethod::CellCenteredFiniteVolume,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(16).unwrap(),
            },
            QuadraturePolicy::CellCentroid,
        ),
        default_plan.solver(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .expect("explicit FVM plan is consistent");
    let fvm = resolve(
        &RealizationRequest::explicit(
            program.model(),
            semantic_revision,
            RealizationRevision::new(1),
            fvm_plan,
        ),
        requirements,
        &capabilities,
    )
    .expect("explicit FVM realization resolves");

    let (fem_model, fem_solution) =
        solve_resolved_scalar_elliptic_1d(&program, &default, &REFERENCE_LINEAR_SOLVER)
            .expect("resolved FEM executes");
    let (fvm_model, fvm_solution) =
        solve_resolved_scalar_elliptic_1d(&program, &fvm, &REFERENCE_LINEAR_SOLVER)
            .expect("resolved FVM executes");

    assert_eq!(fem_model, fvm_model);
    assert_eq!(default.model(), fvm.model());
    assert_eq!(default.semantic_revision(), fvm.semantic_revision());
    assert!(matches!(
        fem_solution,
        ResolvedScalarEllipticSolution1d::FiniteElement(_)
    ));
    assert!(matches!(
        fvm_solution,
        ResolvedScalarEllipticSolution1d::FiniteVolume(_)
    ));
}

#[test]
fn coordinate_axis_is_checked_against_the_relation_domain() {
    let source = r#"
model invalid_axis() {
  domain interval = box(0, 1);

  variable length: m on interval;
  relation identity on interval { length - coordinate(1) = 0; }
}
"#;
    let diagnostics = compile("invalid-axis.eqi", source)
        .expect_err("axis one is outside a one-dimensional Domain");
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(
        diagnostics[0].code(),
        eqiora_core::diagnostic::codes::LANGUAGE_TYPE_ERROR
    );
    assert_eq!(
        diagnostics[0].message(),
        "coordinate axis 1 is outside Domain dimension 1"
    );
    let span = diagnostics[0]
        .source_span()
        .expect("source-local axis error");
    assert_eq!(
        &source[span.start as usize..span.end as usize],
        "coordinate(1)"
    );
}

#[test]
fn canonical_coordinate_axis_is_runtime_dimensional() {
    let source = r#"
model coordinate_plane() {
  domain plane = box(0, 1, 0, 2);

  variable ordinate: m on plane;
  relation identity on plane { ordinate - coordinate(1) = 0; }
}
"#;

    let _program = compile_program("coordinate-plane.eqi", source);
}

fn compile_program(file: &str, source: &str) -> KernelProgram {
    let mut compiled = compile(file, source).expect("the canonical manufactured problem compiles");
    assert_eq!(compiled.len(), 1);
    let (transaction, model_id, _) = compiled.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store
        .commit(transaction)
        .expect("canonical graph transaction commits atomically");
    KernelProgram::from_snapshot(&store.snapshot(), model_id)
        .expect("coordinate-dependent spatial model validates")
}

fn relative_difference(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE)
}

fn assert_optional_close(actual: Option<f64>, expected: Option<f64>, tolerance: f64) {
    match (actual, expected) {
        (Some(actual), Some(expected)) => assert!((actual - expected).abs() < tolerance),
        (None, None) => {}
        mismatch => panic!("optional evidence mismatch: {mismatch:?}"),
    }
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
