mod support;

use std::sync::Arc;

use eqiora::api::{DifferentiableProgram, EvaluationMapPlan};
use support::common_scalar_plan::document_and_plans;

#[test]
fn partition_expansion_matches_complete_ordered_points_for_both_methods() {
    let (document, q1, tpfa) = document_and_plans();
    for plan in [q1, tpfa] {
        let inputs = ["source_scale", "diffusion", "boundary_offset"]
            .map(|name| document.parameter_ref(name).unwrap());
        let field = document
            .field_ref(&plan.fields().next().unwrap().0.ulid().to_string())
            .unwrap();
        let program = Arc::new(
            DifferentiableProgram::compile(
                eqiora_numerics::ResolvedCommonPlan::Linear(Box::new(plan)),
                &inputs,
                &field,
                None,
                &eqiora_solver::REFERENCE_LINEAR_SOLVER,
            )
            .unwrap(),
        );
        let partition = EvaluationMapPlan::from_partition(
            program.clone(),
            &[inputs[1].id()],
            &[2.0],
            &[3.0, 0.5, 1.0, -0.25, 3.0, 0.5],
            &[1, 3],
            eqiora::api::EvaluationMapExecutionPolicy::retained(1 << 30),
        )
        .unwrap();
        let exact = EvaluationMapPlan::new(
            program.clone(),
            &[&[3.0, 2.0, 0.5], &[1.0, 2.0, -0.25], &[3.0, 2.0, 0.5]],
            eqiora::api::EvaluationMapExecutionPolicy::retained(1 << 30),
        )
        .unwrap();
        assert_eq!(partition, exact);
        let empty = EvaluationMapPlan::from_partition(
            program.clone(),
            &[inputs[1].id()],
            &[2.0],
            &[],
            &[2, 0],
            eqiora::api::EvaluationMapExecutionPolicy::retained(0),
        )
        .unwrap();
        assert!(empty.points().is_empty());
        let all_shared = EvaluationMapPlan::from_partition(
            program.clone(),
            &[inputs[2].id(), inputs[0].id(), inputs[1].id()],
            &[0.5, 3.0, 2.0],
            &[],
            &[3],
            eqiora::api::EvaluationMapExecutionPolicy::retained(1 << 30),
        )
        .unwrap();
        for point in all_shared.points() {
            assert_eq!(point.values(), [3.0, 2.0, 0.5]);
        }
        for shape in [&[0, usize::MAX, 2][..], &[usize::MAX][..], &[1; 33][..]] {
            assert!(
                EvaluationMapPlan::from_partition(
                    program.clone(),
                    &[],
                    &[],
                    &[],
                    shape,
                    eqiora::api::EvaluationMapExecutionPolicy::retained(usize::MAX)
                )
                .is_err()
            );
        }
        assert!(
            EvaluationMapPlan::from_partition(
                program.clone(),
                &[inputs[0].id()],
                &[f64::NAN],
                &[],
                &[0],
                eqiora::api::EvaluationMapExecutionPolicy::retained(0)
            )
            .is_err()
        );
        assert!(
            EvaluationMapPlan::from_partition(
                program.clone(),
                &[inputs[0].id(), inputs[0].id()],
                &[1.0, 1.0],
                &[],
                &[0],
                eqiora::api::EvaluationMapExecutionPolicy::retained(0)
            )
            .is_err()
        );
        assert!(
            EvaluationMapPlan::from_partition(
                program.clone(),
                &[],
                &[],
                &[1.0, 2.0, 0.0],
                &[],
                eqiora::api::EvaluationMapExecutionPolicy::retained(0)
            )
            .is_err()
        );
        let singleton = EvaluationMapPlan::from_partition(
            program,
            &[],
            &[],
            &[1.0, 2.0, 0.0],
            &[],
            eqiora::api::EvaluationMapExecutionPolicy::retained(1 << 30),
        )
        .unwrap();
        assert_eq!(singleton.points().len(), 1);
    }
}
