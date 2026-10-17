mod support;
use support::derivative_products::FixtureCoordinates;

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
                eqiora_numerics::ResolvedCommonPlan::Spatial(Box::new(plan)),
                &inputs,
                &[field.erase()],
                None,
                &eqiora_solver::REFERENCE_LINEAR_SOLVER,
            )
            .unwrap(),
        );
        let inputs = program.identity().inputs();
        let subset = |indices: &[usize], values: &[f64]| {
            let signatures = indices
                .iter()
                .map(|&index| inputs[index].clone())
                .collect::<Vec<_>>();
            let components = values
                .iter()
                .map(|&value| [(value, 0.0)])
                .collect::<Vec<_>>();
            eqiora::api::DifferentiableValues::from_components(
                &signatures,
                &components
                    .iter()
                    .map(|value| value.as_slice())
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let policy = eqiora::api::EvaluationMapExecutionPolicy::retained(1 << 30);
        let shared = subset(&[1], &[2.0]);
        let mapped = [
            subset(&[0, 2], &[3.0, 0.5]),
            subset(&[0, 2], &[1.0, -0.25]),
            subset(&[0, 2], &[3.0, 0.5]),
        ];
        let partition =
            EvaluationMapPlan::from_partition(program.clone(), &shared, &mapped, policy).unwrap();
        let exact = EvaluationMapPlan::new(
            program.clone(),
            &support::real_points(
                &program,
                &[&[3.0, 2.0, 0.5], &[1.0, 2.0, -0.25], &[3.0, 2.0, 0.5]],
            ),
            &[document.parameter_ref("diffusion").unwrap().id()],
            policy,
        )
        .unwrap();
        assert_eq!(partition, exact);
        // An empty partition retains one shared Parameter ID and the
        // execution's 128-bit occurrence namespace, but no member buffers.
        let empty_storage = size_of::<eqiora::Id<eqiora::entity::kinds::Parameter>>() + 128 / 8;
        assert!(
            EvaluationMapPlan::from_partition(
                program.clone(),
                &shared,
                &[],
                eqiora::api::EvaluationMapExecutionPolicy::retained(empty_storage - 1),
            )
            .is_err()
        );
        let empty = EvaluationMapPlan::from_partition(
            program.clone(),
            &shared,
            &[],
            eqiora::api::EvaluationMapExecutionPolicy::retained(empty_storage),
        )
        .unwrap();
        assert_eq!(empty.estimated_storage_bytes(), empty_storage);
        assert!(empty.points().unwrap().is_empty());
        let all_shared = EvaluationMapPlan::from_partition(
            program.clone(),
            &subset(&[2, 0, 1], &[0.5, 3.0, 2.0]),
            &vec![subset(&[], &[]); 3],
            policy,
        )
        .unwrap();
        for point in all_shared.points().unwrap() {
            assert_eq!(point.fixture_coordinates(), [3.0, 2.0, 0.5]);
        }
        // Valid products with the wrong order, missing leaves or overlapping
        // shared leaves are rejected at composition, before numerical execution.
        for invalid in [
            subset(&[2, 0], &[0.5, 3.0]),
            subset(&[0], &[3.0]),
            subset(&[0, 1], &[3.0, 2.0]),
        ] {
            assert!(
                EvaluationMapPlan::from_partition(program.clone(), &shared, &[invalid], policy)
                    .is_err()
            );
        }
        assert!(
            EvaluationMapPlan::from_partition(
                program.clone(),
                &subset(&[], &[]),
                &[subset(&[0, 1, 2], &[1.0, 2.0, 0.0])],
                eqiora::api::EvaluationMapExecutionPolicy::retained(0)
            )
            .is_err()
        );
        let singleton = EvaluationMapPlan::from_partition(
            program.clone(),
            &subset(&[], &[]),
            &[subset(&[0, 1, 2], &[1.0, 2.0, 0.0])],
            policy,
        )
        .unwrap();
        assert_eq!(singleton.points().unwrap().len(), 1);
    }
}
