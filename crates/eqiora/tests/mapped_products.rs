mod support;
use support::derivative_products::FixtureCoordinates;

use std::sync::Arc;

use eqiora::api::{
    DifferentiableProgram, EvaluationMapJvp, EvaluationMapPlan, EvaluationMapProducts,
    EvaluationMapVjp,
};
use support::common_scalar_plan::document_and_plans;

#[test]
fn public_q1_and_tpfa_map_products_use_retained_evaluations() {
    let (document, q1, tpfa) = document_and_plans();
    for plan in [q1, tpfa] {
        let inputs = ["source_scale", "diffusion", "boundary_offset"]
            .map(|name| document.parameter_ref(name).unwrap());
        let field = document
            .field_ref(&plan.fields().next().unwrap().0.ulid().to_string())
            .unwrap();
        let program = Arc::new(
            DifferentiableProgram::compile(
                eqiora_numerics::ResolvedCommonPlan::Spatial(Box::new(plan)),
                &inputs,
                &[field.erase()],
                None,
                &eqiora_solver::REFERENCE_LINEAR_SOLVER,
            )
            .unwrap(),
        );
        let shared = eqiora::api::DifferentiableValues::from_components(
            &program.identity().inputs()[..1],
            &[&[(2.0, 0.0)]],
        )
        .unwrap();
        let mapped = [[1.0, 0.0], [2.0, 0.5], [1.0, 0.0]].map(|point| {
            eqiora::api::DifferentiableValues::from_components(
                &program.identity().inputs()[1..],
                &[&[(point[0], 0.0)], &[(point[1], 0.0)]],
            )
            .unwrap()
        });
        let map = EvaluationMapPlan::from_partition(
            program.clone(),
            &shared,
            &mapped,
            eqiora::api::EvaluationMapExecutionPolicy::retained(1 << 30),
        )
        .unwrap()
        .execute()
        .unwrap();
        let products = EvaluationMapProducts::new(&map, &[3], &[2], &[1], 1 << 30).unwrap();
        assert_eq!(products.shape(), [2, 3]);
        assert!(std::ptr::eq(products.map(), &map));
        let jvp: EvaluationMapJvp = products.jvp(&[1.0, 2.0], &[0.0; 12]).unwrap();
        let width = program.identity().output_dimension();
        let vjp: EvaluationMapVjp = products.vjp(&vec![1.0; 6 * width]).unwrap();
        assert_eq!(jvp.shape(), [2, 3, width]);
        assert_eq!(vjp.shared_shape(), [2]);
        assert_eq!(vjp.mapped_shape(), [2, 3]);
        for seed in 0..2 {
            let mut expected_shared = 0.0;
            for (point, evaluation) in map.members().unwrap().iter().enumerate() {
                assert_eq!(
                    jvp.products()[seed * 3 + point],
                    evaluation
                        .jvp(
                            &support::derivative_products::tangent(
                                evaluation.identity(),
                                &[(seed + 1) as f64, 0.0, 0.0]
                            )
                            .unwrap()
                        )
                        .unwrap()
                );
                let expected = evaluation
                    .vjp(
                        &support::derivative_products::cotangent(
                            evaluation.identity(),
                            &vec![1.0; width],
                        )
                        .unwrap(),
                    )
                    .unwrap();
                expected_shared += expected.derivative().fixture_coordinates().as_slice()[0];
                assert_eq!(vjp.products()[seed * 3 + point], expected);
            }
            assert_eq!(
                vjp.shared_cotangents()[seed]
                    .components(0)
                    .unwrap()
                    .collect::<Vec<_>>(),
                [(expected_shared, 0.0)]
            );
        }
    }
}
