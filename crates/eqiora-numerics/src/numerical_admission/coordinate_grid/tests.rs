use super::*;
use crate::numerical_admission::{AuthenticatedCommonMesh, NativeMeshResources};
use eqiora_artifact::ModelEnvelope;
use eqiora_compiler::{CompiledModel, StaticBindingValue};
use eqiora_graph::{GraphStore, InMemoryGraphStore};

fn model() -> (ModelEnvelope, Id<kinds::Domain>) {
    let source =
        "model Distribution(support position:interval(m), support velocity:interval(m/s)) {
        support phase:product(position,velocity);
        variable f:s/m^2 on phase;
        relation density on phase {f=2[s/m^2];}
        observable number:1/m on position=integral(f,measure(velocity));
    }";
    compile(source)
}

fn compile(source: &str) -> (ModelEnvelope, Id<kinds::Domain>) {
    let interval = |lower, upper, time| {
        let unit = DimExponents::from_integers([0, 1, time, 0, 0, 0, 0]).unwrap();
        StaticBindingValue::CoordinateInterval(
            AxisBounds::new(DynQuantity::new(lower, unit), DynQuantity::new(upper, unit)).unwrap(),
        )
    };
    let compiled = CompiledModel::compile_selected(
        "distribution.eqi",
        source,
        "Distribution",
        &[
            ("position", interval(0.0, 2.0, 0)),
            ("velocity", interval(-2.0, 4.0, -1)),
        ],
    )
    .unwrap();
    let phase = compiled.symbols().get("phase").unwrap().downcast().unwrap();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    (ModelEnvelope::from_program(&program).unwrap(), phase)
}

#[test]
fn coordinate_grid_replays_exact_factor_units_and_axes_without_physical_geometry() {
    let (model, phase) = model();
    let owner = AuthenticatedCommonMesh::coordinate_factors(&model, phase, &[2, 3]).unwrap();
    assert!(owner.geometry().is_none());
    assert!(owner.correspondence().is_none());
    assert!(owner.production().is_none());
    let mesh = owner.cartesian_mesh().unwrap().mesh();
    assert_eq!(mesh.axis_coordinates(0).unwrap(), [0.0, 1.0, 2.0]);
    assert_eq!(mesh.axis_coordinates(1).unwrap(), [-2.0, 0.0, 2.0, 4.0]);
    let bytes = owner.to_bytes().unwrap();
    let replay = AuthenticatedCommonMesh::from_bytes(&bytes).unwrap();
    assert_eq!(replay, owner);
    assert_eq!(replay.digest().unwrap(), owner.digest().unwrap());
    let NativeMeshResources::Coordinates(grid) = &replay.resources else {
        panic!("coordinate grid")
    };
    grid.source
        .require_program(&model.to_program().unwrap())
        .unwrap();
    assert_eq!(
        grid.source.factors[0].dimension,
        DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0])
            .unwrap()
            .exponents()
    );
    assert_eq!(
        grid.source.factors[1].dimension,
        DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0])
            .unwrap()
            .exponents()
    );
}

#[test]
fn coordinate_grid_rejects_factor_unit_bound_and_mesh_substitution() {
    let (model, phase) = model();
    let program = model.to_program().unwrap();
    let grid = CoordinateGrid::new(&program, phase, &[2, 3]).unwrap();
    for change in [0, 1, 2] {
        let mut source = grid.source.clone();
        match change {
            0 => source.factors[1].domain = Id::<kinds::Domain>::new().ulid().to_string(),
            1 => source.factors[1].dimension = source.factors[0].dimension,
            _ => source.factors[1].upper = 5.0,
        }
        source.validate().unwrap();
        assert!(source.require_program(&program).is_err());
    }
    let foreign = CartesianMesh::uniform(&[[0.0, 2.0], [-2.0, 5.0]], &[2, 3]).unwrap();
    assert!(
        CoordinateGrid::from_parts(
            grid.source.clone(),
            CartesianMeshEnvelopeV1::from_mesh(&foreign).unwrap()
        )
        .is_err()
    );
    for cells in [&[2][..], &[2, 0][..], &[usize::MAX, 2][..]] {
        assert!(CoordinateGrid::new(&program, phase, cells).is_err());
    }
    assert!(CoordinateGrid::new(&program, Id::new(), &[2, 3]).is_err());
}

const POLYNOMIAL: &str =
    "model Distribution(support position:interval(m), support velocity:interval(m/s)) {
    support phase:product(position,velocity);
    coordinate x:m on phase from position;
    coordinate v:m/s on phase from velocity;
    variable f:s/m^2 on phase;
    relation density on phase { f=3[s/m^2]*(1+x/2[m])*(1+v*v/16[m^2/s^2]); }
    observable number:1/m on position=integral(f,measure(velocity));
    observable first:1/s on position=integral(v*f,measure(velocity));
    observable second:m/s^2 on position=integral(v*v*f,measure(velocity));
    observable mass:1=integral(f,measure(phase));
    observable radial:s*m on velocity=integral(f,spherical_measure(position));
}";

#[test]
fn coordinate_grid_projection_uses_cell_averages_not_midpoint_samples() {
    use super::projection::CellProjection;
    let (model, phase) = compile(POLYNOMIAL);
    let program = model.to_program().unwrap();
    for velocity_cells in [1, 3, 6] {
        let grid = CoordinateGrid::new(&program, phase, &[2, velocity_cells]).unwrap();
        let projection = CellProjection::lower(&program, &grid).unwrap();
        let values = projection.cell_values(&grid).unwrap();
        let h = 6.0 / velocity_cells as f64;
        let mut mass = 0.0;
        for (index, value) in values.iter().enumerate() {
            let ij = grid
                .mesh
                .mesh()
                .cell_multi_index(eqiora_meshing::MeshEntity::new(2, index))
                .unwrap();
            let x = ij[0] as f64 + 0.5;
            let v = -2.0 + (ij[1] as f64 + 0.5) * h;
            // Integral of v^2 over a cell divided by its width is v_mid^2 + h^2/12.
            let expected = 3.0 * (1.0 + x / 2.0) * (1.0 + (v * v + h * h / 12.0) / 16.0);
            assert!((value - expected).abs() < 1e-12, "{value} != {expected}");
            mass += value * h;
        }
        // Integral in x is 3; in v it is 15/2; amplitude is 3.
        assert!((mass - 67.5).abs() < 1e-11);
    }
}

#[test]
fn coordinate_grid_projection_rejects_unsupported_equations() {
    use super::projection::CellProjection;
    for source in [
        POLYNOMIAL.replace("v*v/16[m^2/s^2]", "v^4/16[m^4/s^4]"),
        POLYNOMIAL.replace("v*v/16[m^2/s^2]", "math.sin(v/4[m/s])"),
        POLYNOMIAL.replace("f=3[s/m^2]", "f=f+3[s/m^2]"),
        POLYNOMIAL.replace("f=3[s/m^2]", "2*f=3[s/m^2]"),
    ] {
        let (model, phase) = compile(&source);
        let program = model.to_program().unwrap();
        let grid = CoordinateGrid::new(&program, phase, &[2, 3]).unwrap();
        assert!(CellProjection::lower(&program, &grid).is_err(), "{source}");
    }
}

#[test]
fn coordinate_grid_cell_lookup_preserves_units_and_half_open_boundary_ownership() {
    let (model, phase) = model();
    let grid = CoordinateGrid::new(&model.to_program().unwrap(), phase, &[2, 3]).unwrap();
    let unit =
        |axis: usize| DimExponents::from_rationals(grid.source.factors[axis].dimension).unwrap();
    for (x, v, i, j) in [(0.0, -2.0, 0, 0), (1.0, 0.0, 1, 1), (2.0, 4.0, 1, 2)] {
        let cell = grid
            .cell_at(&[DynQuantity::new(x, unit(0)), DynQuantity::new(v, unit(1))])
            .unwrap();
        assert_eq!(cell, grid.mesh.mesh().cell_at(&[i, j]).unwrap().index());
        let axes = grid.cell_axes(cell).unwrap();
        assert_eq!(
            axes[0].0,
            (
                parse_domain(&grid.source.factors[0].domain)
                    .unwrap()
                    .erase(),
                0
            )
        );
        assert_eq!(axes[1].1.lower().dim(), unit(1));
    }
    assert!(
        grid.cell_at(&[
            DynQuantity::new(1.0, unit(1)),
            DynQuantity::new(0.0, unit(0))
        ])
        .is_err()
    );
    assert!(grid.cell_at(&[DynQuantity::new(1.0, unit(0))]).is_err());
    for v in [-2.0001, 4.0001, f64::NAN, f64::INFINITY] {
        assert!(
            grid.cell_at(&[DynQuantity::new(1.0, unit(0)), DynQuantity::new(v, unit(1))])
                .is_err()
        );
    }
    assert!(grid.cell_axes(6).is_err());
}

#[test]
fn coordinate_grid_projected_velocity_moments_have_analytic_refinement_error() {
    use super::projection::CellProjection;
    use crate::factor_measure::mapped_sample;
    use eqiora_meshing::QuadratureRule;
    use eqiora_schema::kernel::ObservableMeasure;
    let (model, phase) = compile(POLYNOMIAL);
    let program = model.to_program().unwrap();
    let rule = QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap();
    for n in [1, 3, 6] {
        let grid = CoordinateGrid::new(&program, phase, &[2, n]).unwrap();
        let values = CellProjection::lower(&program, &grid)
            .unwrap()
            .cell_values(&grid)
            .unwrap();
        for i in 0..2 {
            let mut moments = [0.0; 3];
            for j in 0..n {
                let cell = grid.mesh.mesh().cell_at(&[i, j]).unwrap().index();
                let axes = grid.cell_axes(cell).unwrap();
                // Split at each velocity cell, then integrate the represented constant Field.
                for sample in rule.points() {
                    let (v, weight) =
                        mapped_sample(&axes[1..], sample, ObservableMeasure::Volume).unwrap();
                    for (order, sum) in moments.iter_mut().enumerate() {
                        *sum += weight.value() * values[cell] * v[0].value().powi(order as i32);
                    }
                }
            }
            let a = 3.0 * (1.0 + (i as f64 + 0.5) / 2.0);
            let n2 = (n * n) as f64;
            // Integrating each constant projection against v and v^2 loses within-cell
            // covariance. Direct polynomial antiderivatives give these error terms.
            let expected = [
                a * 7.5,
                a * (39.0 / 4.0 - 9.0 / (4.0 * n2)),
                a * (186.0 / 5.0 - 18.0 / n2 + 54.0 / (5.0 * n2 * n2)),
            ];
            for (observed, expected) in moments.into_iter().zip(expected) {
                assert!(
                    (observed - expected).abs() < 1e-10,
                    "n={n}: {observed} != {expected}"
                );
            }
        }
    }
}

#[test]
fn coordinate_grid_plan_lineage_binds_units_without_fabricating_geometry() {
    use crate::numerical_admission::native::resource_digests;
    let (model, phase) = model();
    let grid = CoordinateGrid::new(&model.to_program().unwrap(), phase, &[2, 3]).unwrap();
    let original = resource_digests(&NativeMeshResources::Coordinates(grid.clone())).unwrap();
    assert!(original.geometry().is_none());
    assert!(original.correspondence().is_none());
    assert!(original.production().is_none());
    let mut source = grid.source.clone();
    source.factors[1].dimension = source.factors[0].dimension;
    let substituted = CoordinateGrid::from_parts(source, grid.mesh.clone()).unwrap();
    let changed = resource_digests(&NativeMeshResources::Coordinates(substituted)).unwrap();
    assert_eq!(original.mesh(), changed.mesh());
    let mut expected = Vec::new();
    let mut actual = Vec::new();
    original.bind(&mut expected);
    changed.bind(&mut actual);
    assert_ne!(expected, actual);
}

#[test]
fn coordinate_grid_common_plan_solves_and_replays_complete_cell_field() {
    use crate::numerical_admission::{
        CommonLinearRequest, CommonSolvePolicy, CommonSpatialPolicy, ResolvedCommonPlan,
        resolve_common_plan,
    };
    use eqiora_solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};
    let (model, phase) = compile(POLYNOMIAL);
    let mesh = AuthenticatedCommonMesh::coordinate_factors(&model, phase, &[2, 3]).unwrap();
    let linear = CommonLinearRequest::exact(
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1e-12,
            1e-12,
            std::num::NonZeroUsize::new(100).unwrap(),
        )
        .unwrap(),
        REFERENCE_LINEAR_SOLVER.provider(),
    )
    .unwrap();
    let plan = resolve_common_plan(
        &model,
        mesh,
        CommonSpatialPolicy::CellCentered,
        CommonSolvePolicy::Linear(linear),
        None,
        None,
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap();
    assert!(plan.geometry_digest().is_none());
    assert!(plan.correspondence_digest().is_none());
    assert!(plan.production_digest().is_none());
    let bytes = plan.to_bytes().unwrap();
    let replay = ResolvedCommonPlan::from_bytes(
        &bytes,
        &REFERENCE_LINEAR_SOLVER,
        eqiora_time::TimeBackendCapabilities::new(
            eqiora_time::TimeBackendIdentity::new("eqiora.test.time", "1"),
            &[
                eqiora_core::ScalarDomain::Real,
                eqiora_core::ScalarDomain::Complex,
            ],
            &[eqiora_core::ScalarType::F64],
        ),
    )
    .unwrap();
    assert_eq!(plan, replay);
    let result = replay
        .as_scalar()
        .unwrap()
        .run_result(&REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let (association, values, shape) = result.field_block(0, 0).unwrap();
    assert_eq!(association, "cell");
    assert_eq!(shape, [2, 3]);
    let mesh = replay.authenticated_mesh().unwrap();
    let mesh = mesh.cartesian_mesh().unwrap().mesh();
    for (index, actual) in values.iter().enumerate() {
        let ij = mesh
            .cell_multi_index(eqiora_meshing::MeshEntity::new(2, index))
            .unwrap();
        let x = ij[0] as f64 + 0.5;
        let v = -1.0 + 2.0 * ij[1] as f64;
        let expected = 3.0 * (1.0 + x / 2.0) * (1.0 + (v * v + 1.0 / 3.0) / 16.0);
        assert!((actual - expected).abs() < 1e-10);
    }
    assert_factor_observations(&model, &result);
    let result_bytes = result.to_bytes().unwrap();
    assert_factor_observations(
        &model,
        &crate::CommonResult::from_bytes(&result_bytes, &replay).unwrap(),
    );
    assert_eq!(
        crate::CommonResult::from_bytes(&result_bytes, &replay)
            .unwrap()
            .to_bytes()
            .unwrap(),
        result_bytes
    );
}

fn assert_factor_observations(model: &ModelEnvelope, result: &crate::CommonResult) {
    use eqiora_meshing::QuadratureRule;
    use eqiora_schema::kernel::ObservableReduction;
    use std::collections::HashMap;
    let program = model.to_program().unwrap();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let speed = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
    for node in program.nodes() {
        let KernelNode::Observable(observable) = node else {
            continue;
        };
        let ObservableReduction::SpatialIntegral { domain, .. } = observable.reduction() else {
            continue;
        };
        let dimension = crate::factor_measure::axes(&program, domain).unwrap().len();
        let rules = HashMap::from([(
            domain,
            QuadratureRule::tensor_product_gauss_legendre(dimension, 2).unwrap(),
        )]);
        let unit = observable.value_type().dimension();
        if unit == DimExponents::DIMENSIONLESS {
            let value = result.observe(model, observable.id(), &rules).unwrap();
            assert!((value.value().real_scalar_value().unwrap().value() - 67.5).abs() < 1e-10);
        } else if unit == DimExponents::from_integers([0, 1, 1, 0, 0, 0, 0]).unwrap() {
            let value = result
                .observe_at(
                    model,
                    observable.id(),
                    &[DynQuantity::new(-1.0, speed)],
                    &rules,
                )
                .unwrap();
            // On the first velocity cell, g=13/12. The two radial cell averages are
            // 3*g*5/4 and 3*g*7/4; spherical cell weights are 4*pi/3 and 28*pi/3.
            assert!(
                (value.value().real_scalar_value().unwrap().value() - 58.5 * std::f64::consts::PI)
                    .abs()
                    < 1e-10
            );
        } else {
            let factors = [
                7.5,
                39.0 / 4.0 - 9.0 / 36.0,
                186.0 / 5.0 - 2.0 + 54.0 / 405.0,
            ];
            let order = (0..3)
                .find(|order| {
                    unit == DimExponents::from_integers([0, order - 1, -order, 0, 0, 0, 0]).unwrap()
                })
                .unwrap();
            // A cell face belongs to its upper cell; the final endpoint remains in the last.
            for (x, average_x) in [(0.0, 0.5), (0.5, 0.5), (1.0, 1.5), (2.0, 1.5)] {
                let value = result
                    .observe_at(
                        model,
                        observable.id(),
                        &[DynQuantity::new(x, length)],
                        &rules,
                    )
                    .unwrap();
                let expected = 3.0 * (1.0 + average_x / 2.0) * factors[order as usize];
                assert!(
                    (value.value().real_scalar_value().unwrap().value() - expected).abs() < 1e-10
                );
            }
            assert!(
                result
                    .observe_at(
                        model,
                        observable.id(),
                        &[DynQuantity::new(0.5, speed)],
                        &rules
                    )
                    .is_err()
            );
        }
    }
}
