use super::*;
use crate::simplicial_ale_fsi::test_support::geometry_graph;
use eqiora_realization::{PortableRealizationGraph, TransformationNode};

#[test]
fn graph_geometry_requires_exact_model_rows_and_sealed_duration() {
    let fixture = fixture();
    let graph = geometry_graph(&fixture.layout, &fixture.motion, &fixture.plan);
    let bind = |graph: &PortableRealizationGraph, plan: &AleFsiStepPlan<2>| {
        configuration::current_pullbacks(graph, &fixture.layout, &fixture.motion, plan)
    };
    assert_eq!(bind(&graph, &fixture.plan).unwrap().len(), 1);

    let text = String::from_utf8(graph.to_bytes().unwrap()).unwrap();
    let domain = graph.domains()[0].domain();
    let replacement = eqiora_core::Id::<eqiora_core::entity::kinds::Domain>::new();
    let foreign = text.replace(&domain.to_string(), &replacement.to_string());
    assert_ne!(text, foreign);
    let foreign = PortableRealizationGraph::from_bytes(foreign.as_bytes()).unwrap();
    assert!(
        bind(&foreign, &fixture.plan)
            .unwrap_err()
            .message()
            .contains("Domains")
    );

    let relation = graph
        .transformations()
        .iter()
        .find_map(|node| match node {
            TransformationNode::GclCompatibleAlePullback { relation, .. } => Some(*relation),
            _ => None,
        })
        .unwrap();
    let replacement = eqiora_core::Id::<eqiora_core::entity::kinds::Relation>::new();
    // Keep the BE/GCL pair internally consistent. Only the compiled Model can
    // distinguish this well-formed graph from the actual equation inventory.
    let changed = text.replace(&relation.to_string(), &replacement.to_string());
    assert_ne!(text, changed);
    let changed = PortableRealizationGraph::from_bytes(changed.as_bytes()).unwrap();
    assert!(
        bind(&changed, &fixture.plan)
            .unwrap_err()
            .message()
            .contains("Relation and Field")
    );

    let plan = &fixture.plan;
    let changed = AleFsiStepPlan::new(
        2. * plan.time_step(),
        plan.material().clone(),
        plan.scale(),
        plan.load(),
        plan.nonlinear(),
        plan.linear_solver(),
        plan.target(),
    )
    .unwrap();
    assert!(
        bind(&graph, &changed)
            .unwrap_err()
            .message()
            .contains("duration")
    );
}

#[test]
fn driven_domain_current_geometry_obeys_the_plan_and_full_shape_jacobian() {
    let fixture = fixture();
    let graph = geometry_graph(&fixture.layout, &fixture.motion, &fixture.plan);
    let text = String::from_utf8(graph.to_bytes().unwrap()).unwrap();
    let changed = text.replace(
        "\"kind\":\"reference-configuration\"",
        "\"kind\":\"current-ale-geometry\",\"geometry_action\":0",
    );
    assert_ne!(changed, text);
    let current = PortableRealizationGraph::from_bytes(changed.as_bytes()).unwrap();
    let configurations =
        configuration::current_pullbacks(&current, &fixture.layout, &fixture.motion, &fixture.plan)
            .unwrap();
    let driver_domain = fixture.motion.policy().drivers()[0].0.erase();
    assert!(configurations[&driver_domain].is_empty());
    assert_eq!(configurations.len(), 2);

    let quadrature = triangle_duffy_gauss_legendre(5).unwrap();
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
        .map(|i| 0.1 * ((i % 5) as f64 - 2.))
        .collect::<Vec<_>>();
    let evaluate = |point: &[f64], graph: &PortableRealizationGraph| {
        assemble_step_residual(
            &fixture.mesh,
            &fixture.partition,
            &fixture.boundary,
            &fixture.motion,
            &fixture.previous,
            point,
            &fixture.plan,
            &quadrature,
            &fixture.layout,
            graph,
        )
        .unwrap()
    };
    let action = assemble_step_linearization(
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
        &current,
    )
    .unwrap();
    assert_eq!(action.residual, evaluate(&point, &current));
    // Both plans use the same Fields and pressure layout; only the explicit
    // configuration changes. A role-based selector would leave these equal.
    assert!(
        action
            .residual
            .iter()
            .zip(evaluate(&point, &graph))
            .any(|(a, b)| (a - b).abs() > 1.0e-8)
    );
    let mut analytic = vec![0.; point.len()];
    action
        .relation
        .jvp(RelationTangent::Unknown(&direction), &mut analytic)
        .unwrap();
    let epsilon = f64::EPSILON.cbrt();
    let shifted = |sign: f64| {
        point
            .iter()
            .zip(&direction)
            .map(|(x, d)| x + sign * epsilon * d)
            .collect::<Vec<_>>()
    };
    let plus = evaluate(&shifted(1.), &current);
    let minus = evaluate(&shifted(-1.), &current);
    let error = plus
        .iter()
        .zip(minus)
        .zip(&analytic)
        .map(|((p, m), a)| ((p - m) / (2. * epsilon) - a).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale = analytic.iter().map(|a| a * a).sum::<f64>().sqrt();
    // Same centered-difference error bound as the established global JVP test.
    assert!(error < 2.0e-6 * (1. + scale), "{error:e} versus {scale:e}");
}
