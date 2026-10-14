mod geometry;

use super::structure::build_structural_jacobian_pattern;

use eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND;
use eqiora_ir::{LinearizedRelation, RelationTangent};
use eqiora_meshing::simplex_duffy_gauss_legendre;

use super::*;
use crate::simplicial_ale_fsi::test_support::{
    harmonic_solver, linearization_step_plan, material, material_for, motion_for_plan,
    partition_for_plan, tetrahedral_problem, two_domain_inventories, two_domain_mesh,
};
use crate::simplicial_ale_fsi::{AleFsiState, AleFsiStepPlan, P1HarmonicMeshMotionAction};
use crate::simplicial_fsi::{
    FixedReferenceFsiBoundary, FixedReferenceFsiPartition, FixedReferenceFsiScale,
};

struct Fixture {
    mesh: SimplicialMesh,
    partition: FixedReferenceFsiPartition<2>,
    layout: crate::simplicial_fsi::layout::FsiLayout<2>,
    boundary: FixedReferenceFsiBoundary<2>,
    motion: P1HarmonicMeshMotionAction<2>,
    previous: AleFsiState<2>,
    plan: AleFsiStepPlan<2>,
}

struct Fixture3d {
    mesh: SimplicialMesh,
    partition: FixedReferenceFsiPartition<3>,
    layout: crate::simplicial_fsi::layout::FsiLayout<3>,
    boundary: FixedReferenceFsiBoundary<3>,
    motion: P1HarmonicMeshMotionAction<3>,
    previous: AleFsiState<3>,
    plan: AleFsiStepPlan<3>,
}

#[test]
fn analytic_global_jvp_matches_centered_full_reassembly() {
    let fixture = fixture();
    let quadrature = simplex_duffy_gauss_legendre(2, 5).unwrap();
    let mut point = initial_point(
        &fixture.mesh,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        &fixture.plan,
        &quadrature,
        &fixture.layout,
    )
    .unwrap();
    for (index, value) in point.iter_mut().enumerate() {
        *value = 2.0e-3 * ((index % 7) as f64 - 3.0);
    }
    let direction = (0..point.len())
        .map(|index| 0.1 * ((index % 5) as f64 - 2.0))
        .collect::<Vec<_>>();
    let assembled = assemble(&fixture, &point, &quadrature);
    let residual_only = residual(&fixture, &point, &quadrature);
    assert_eq!(
        residual_only
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        assembled
            .residual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    let mut captured_primal = vec![0.0; point.len()];
    assembled.relation.primal(&mut captured_primal).unwrap();
    let primal_defect = captured_primal
        .iter()
        .zip(&assembled.residual)
        .map(|(captured, direct)| (captured - direct).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(primal_defect < 1.0e-12, "{primal_defect:e}");
    let mut analytic = vec![0.0; point.len()];
    assembled
        .relation
        .jvp(RelationTangent::Unknown(&direction), &mut analytic)
        .unwrap();

    let epsilon = f64::EPSILON.cbrt();
    let shifted = |sign: f64| {
        point
            .iter()
            .zip(&direction)
            .map(|(point, direction)| point + sign * epsilon * direction)
            .collect::<Vec<_>>()
    };
    let plus = residual(&fixture, &shifted(1.0), &quadrature);
    let minus = residual(&fixture, &shifted(-1.0), &quadrature);
    let centered = plus
        .iter()
        .zip(minus)
        .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
        .collect::<Vec<_>>();
    let error = centered
        .iter()
        .zip(&analytic)
        .map(|(centered, analytic)| (centered - analytic).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale = analytic
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    assert!(error < 2.0e-6 * (1.0 + scale), "{error:e} versus {scale:e}");
    assert_eq!(
        assembled.assembly_report.packet_count(),
        fixture.partition.cell_count()
    );
    assert_eq!(assembled.assembly_report.target_count(), 1);
}

#[test]
fn sealed_harmonic_driver_columns_are_singletons_in_real_ale_patterns() {
    let fixture = fixture();
    let quadrature = simplex_duffy_gauss_legendre(2, 5).unwrap();
    let point = initial_point(
        &fixture.mesh,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        &fixture.plan,
        &quadrature,
        &fixture.layout,
    )
    .unwrap();
    let assembled = assemble(&fixture, &point, &quadrature);
    let pattern = build_structural_jacobian_pattern(
        &fixture.mesh,
        &fixture.partition,
        &fixture.motion,
        &assembled.layout,
    )
    .unwrap();
    assert_harmonic_driver_singletons(&fixture.motion, &assembled.layout, &pattern);

    let fixture = fixture_3d();
    let quadrature = simplex_duffy_gauss_legendre(3, 7).unwrap();
    let point = initial_point(
        &fixture.mesh,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        &fixture.plan,
        &quadrature,
        &fixture.layout,
    )
    .unwrap();
    let assembled = assemble_step_linearization(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        &point,
        &fixture.plan,
        &quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        &fixture.layout,
        &crate::simplicial_ale_fsi::test_support::geometry_graph(
            &fixture.layout,
            &fixture.motion,
            &fixture.plan,
        ),
    )
    .unwrap();
    let pattern = build_structural_jacobian_pattern(
        &fixture.mesh,
        &fixture.partition,
        &fixture.motion,
        &assembled.layout,
    )
    .unwrap();
    assert_harmonic_driver_singletons(&fixture.motion, &assembled.layout, &pattern);
}

#[test]
fn zero_solid_update_produces_an_exact_static_geometry_action() {
    let fixture = fixture();
    let quadrature = simplex_duffy_gauss_legendre(2, 5).unwrap();
    let point = initial_point(
        &fixture.mesh,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        &fixture.plan,
        &quadrature,
        &fixture.layout,
    )
    .unwrap();
    let assembled = assemble(&fixture, &point, &quadrature);
    assert_eq!(
        assembled.current_state().geometry(),
        fixture.previous.geometry()
    );
    assert!(
        assembled
            .geometry_action()
            .vertex_velocities()
            .iter()
            .flatten()
            .all(|value| *value == 0.0)
    );
    for cell in assembled.geometry_action().cells() {
        assert_eq!(cell.previous_map(), cell.current_map());
        assert_eq!(cell.current_velocity_divergence(), 0.0);
        assert_eq!(cell.skew_gcl_correction(), 0.0);
        assert_eq!(cell.endpoint_metric_rate(), 0.0);
    }
}

#[test]
fn degree_six_rule_is_rejected_before_ale_assembly() {
    let fixture = fixture();
    assert!(
        initial_point(
            &fixture.mesh,
            &fixture.boundary,
            &fixture.motion,
            &fixture.previous,
            &fixture.plan,
            &simplex_duffy_gauss_legendre(2, 4).unwrap(),
            &fixture.layout,
        )
        .is_err()
    );
}

#[test]
fn residual_only_rejects_nonfinite_and_wrong_shape_candidates() {
    let fixture = fixture();
    let quadrature = simplex_duffy_gauss_legendre(2, 5).unwrap();
    let point = initial_point(
        &fixture.mesh,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        &fixture.plan,
        &quadrature,
        &fixture.layout,
    )
    .unwrap();
    let mut short = point.clone();
    short.pop();
    let shape_error = assemble_step_residual(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        &short,
        &fixture.plan,
        &quadrature,
        &fixture.layout,
        &crate::simplicial_ale_fsi::test_support::geometry_graph(
            &fixture.layout,
            &fixture.motion,
            &fixture.plan,
        ),
    )
    .unwrap_err();
    assert!(
        shape_error
            .message()
            .contains("exact reduced quotient layout")
    );

    let mut nonfinite = point;
    nonfinite[0] = f64::NAN;
    let finite_error = assemble_step_residual(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        &nonfinite,
        &fixture.plan,
        &quadrature,
        &fixture.layout,
        &crate::simplicial_ale_fsi::test_support::geometry_graph(
            &fixture.layout,
            &fixture.motion,
            &fixture.plan,
        ),
    )
    .unwrap_err();
    assert!(finite_error.message().contains("finite"));
}

#[test]
fn tetrahedral_assembly_has_typed_power_exactness_and_centered_jvp() {
    let fixture = fixture_3d();
    let degree_nine = simplex_duffy_gauss_legendre(3, 6).unwrap();
    let rejected = initial_point(
        &fixture.mesh,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        &fixture.plan,
        &degree_nine,
        &fixture.layout,
    )
    .unwrap_err();
    assert!(rejected.message().contains("at least 11"));

    let quadrature = simplex_duffy_gauss_legendre(3, 7).unwrap();
    let mut point = initial_point(
        &fixture.mesh,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        &fixture.plan,
        &quadrature,
        &fixture.layout,
    )
    .unwrap();
    for (index, value) in point.iter_mut().enumerate() {
        *value = 2.0e-4 * ((index % 7) as f64 - 3.0);
    }
    let assembled = assemble_step_linearization(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        &point,
        &fixture.plan,
        &quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        &fixture.layout,
        &crate::simplicial_ale_fsi::test_support::geometry_graph(
            &fixture.layout,
            &fixture.motion,
            &fixture.plan,
        ),
    )
    .unwrap();
    let residual_only = assemble_step_residual(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        &point,
        &fixture.plan,
        &quadrature,
        &fixture.layout,
        &crate::simplicial_ale_fsi::test_support::geometry_graph(
            &fixture.layout,
            &fixture.motion,
            &fixture.plan,
        ),
    )
    .unwrap();
    assert_eq!(
        residual_only
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        assembled
            .residual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    let direction = (0..point.len())
        .map(|index| 0.01 * ((index % 5) as f64 - 2.0))
        .collect::<Vec<_>>();
    let mut jvp = vec![0.0; point.len()];
    assembled
        .relation
        .jvp(RelationTangent::Unknown(&direction), &mut jvp)
        .unwrap();
    assert!(jvp.iter().all(|value| value.is_finite()));
    assert!(jvp.iter().any(|value| *value != 0.0));
    assert!(
        assembled
            .geometry_action()
            .vertex_velocities()
            .iter()
            .flatten()
            .any(|value| *value != 0.0)
    );

    let epsilon = f64::EPSILON.cbrt();
    let shifted_residual = |sign: f64| {
        let shifted = point
            .iter()
            .zip(&direction)
            .map(|(point, direction)| point + sign * epsilon * direction)
            .collect::<Vec<_>>();
        assemble_step_residual(
            &fixture.mesh,
            &fixture.partition,
            &fixture.boundary,
            &fixture.motion,
            &fixture.previous,
            &shifted,
            &fixture.plan,
            &quadrature,
            &fixture.layout,
            &crate::simplicial_ale_fsi::test_support::geometry_graph(
                &fixture.layout,
                &fixture.motion,
                &fixture.plan,
            ),
        )
        .unwrap()
    };
    let plus = shifted_residual(1.0);
    let minus = shifted_residual(-1.0);
    let centered = plus
        .iter()
        .zip(minus)
        .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
        .collect::<Vec<_>>();
    let error = centered
        .iter()
        .zip(&jvp)
        .map(|(centered, analytic)| (centered - analytic).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale = jvp.iter().map(|value| value * value).sum::<f64>().sqrt();
    assert!(error < 5.0e-6 * (1.0 + scale), "{error:e} versus {scale:e}");
    assert_eq!(
        assembled.assembly_report.packet_count(),
        fixture.partition.cell_count()
    );
    assert_eq!(
        assembled.geometry_action().cells().len(),
        fixture.mesh.cells().len()
    );

    assert_eq!(fixture.plan.scale().power(), 60.0);
    let domain = fixture.motion.policy().harmonic_domains()[0].erase();
    let form = fixture.layout.region_form(domain).unwrap();
    assert_eq!(
        form.fields()
            .iter()
            .map(|field| field.range.len())
            .sum::<usize>(),
        19
    );
    for field in form.fields() {
        let multiplier = form
            .physical_dual_scale(field.field)
            .unwrap()
            .value()
            .recip();
        let expected = if field.components == 3 {
            5.0 / 60.0
        } else {
            -3.0 / 60.0
        };
        assert_eq!(multiplier, expected);
    }
}

fn assemble(fixture: &Fixture, point: &[f64], quadrature: &QuadratureRule) -> StepAssembly<2> {
    assemble_step_linearization(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        point,
        &fixture.plan,
        quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        &fixture.layout,
        &crate::simplicial_ale_fsi::test_support::geometry_graph(
            &fixture.layout,
            &fixture.motion,
            &fixture.plan,
        ),
    )
    .unwrap()
}

fn residual(fixture: &Fixture, point: &[f64], quadrature: &QuadratureRule) -> Vec<f64> {
    assemble_step_residual(
        &fixture.mesh,
        &fixture.partition,
        &fixture.boundary,
        &fixture.motion,
        &fixture.previous,
        point,
        &fixture.plan,
        quadrature,
        &fixture.layout,
        &crate::simplicial_ale_fsi::test_support::geometry_graph(
            &fixture.layout,
            &fixture.motion,
            &fixture.plan,
        ),
    )
    .unwrap()
}

fn assert_harmonic_driver_singletons<const D: usize>(
    motion: &P1HarmonicMeshMotionAction<D>,
    layout: &FsiLayout<D>,
    pattern: &StructuralJacobianPattern,
) {
    let velocity = layout
        .state_rate(motion.policy().drivers()[0].1.erase())
        .expect("driver State owns an exact rate Field");
    let mut represented_driver_columns = 0;
    for driver in motion.driver_vertices() {
        for component in 0..D {
            if let Some(dof) = layout.reduced_vertex_velocity(velocity, driver.index(), component) {
                represented_driver_columns += 1;
                assert!(
                    pattern.is_singleton(dof.index()),
                    "harmonic driver vertex {} component {component} column {} is not singleton",
                    driver.index(),
                    dof.index()
                );
            }
        }
    }
    assert!(represented_driver_columns > 0);
}

fn fixture() -> Fixture {
    let mesh = two_domain_mesh();
    let (fluid, solid, _) = two_domain_inventories(&mesh);
    let seed = linearization_step_plan(
        material(),
        FixedReferenceFsiScale::<2>::new(2.0, 1.0, 1.0).unwrap(),
    );
    let model = crate::simplicial_fsi::test_model::planar_model(
        &crate::simplicial_fsi::test_model::adjacent_rectangles(),
        &mesh,
        seed.fixed_reference_config().clone(),
        seed.linear_solver(),
        true,
    );
    let fields = crate::simplicial_fsi::test_model::exact_fields(&model.plan);
    let partition = partition_for_plan(&mesh, fluid, solid, &model.plan);
    let boundary = FixedReferenceFsiBoundary::<2>::homogeneous_exterior(&mesh).unwrap();
    let motion = motion_for_plan(&mesh, &partition, &model.plan, harmonic_solver());
    let physical = crate::simplicial_fsi::test_model::exact_state(
        &model.program,
        &model.plan,
        &mesh,
        &partition,
        |_field, _entity, _component| 0.0,
    );
    let previous = AleFsiState::<2>::new(0.0, &mesh, &partition, &motion, physical).unwrap();
    let plan = linearization_step_plan(
        material_for(fields),
        FixedReferenceFsiScale::new(2.0, 1.0, 1.0).unwrap(),
    );
    let layout = crate::simplicial_fsi::layout::FsiLayout::bind(
        &model.program,
        &crate::simplicial_fsi::test_model::realization_graph(&model.plan),
        &mesh,
        &partition,
        &boundary,
        &crate::simplicial_fsi::test_model::test_orientations(&model.plan),
    )
    .unwrap();
    Fixture {
        layout,
        mesh,
        partition,
        boundary,
        motion,
        previous,
        plan,
    }
}

fn fixture_3d() -> Fixture3d {
    let (mesh, fluid, solid, _) = tetrahedral_problem();
    let seed = linearization_step_plan(
        material(),
        FixedReferenceFsiScale::<3>::new(2.0, 5.0, 3.0).unwrap(),
    );
    let geometry = crate::simplicial_fsi::test_model::polyhedra::tetrahedral_geometry();
    let model = crate::simplicial_fsi::test_model::polyhedra::polyhedral_model(
        &geometry,
        &mesh,
        seed.fixed_reference_config().clone(),
        seed.linear_solver(),
        true,
    );
    let fields = crate::simplicial_fsi::test_model::exact_fields(&model.plan);
    let partition = partition_for_plan(&mesh, fluid, solid, &model.plan);
    let boundary = FixedReferenceFsiBoundary::<3>::homogeneous_exterior(&mesh).unwrap();
    let motion = motion_for_plan(&mesh, &partition, &model.plan, harmonic_solver());
    let physical = crate::simplicial_fsi::test_model::exact_state(
        &model.program,
        &model.plan,
        &mesh,
        &partition,
        |_field, _entity, _component| 0.0,
    );
    let previous = AleFsiState::<3>::new(0.0, &mesh, &partition, &motion, physical).unwrap();
    let plan = linearization_step_plan(
        material_for(fields),
        FixedReferenceFsiScale::<3>::new(2.0, 5.0, 3.0).unwrap(),
    );
    let layout = crate::simplicial_fsi::layout::FsiLayout::bind(
        &model.program,
        &crate::simplicial_fsi::test_model::realization_graph(&model.plan),
        &mesh,
        &partition,
        &boundary,
        &crate::simplicial_fsi::test_model::test_orientations(&model.plan),
    )
    .unwrap();
    Fixture3d {
        layout,
        mesh,
        partition,
        boundary,
        motion,
        previous,
        plan,
    }
}
