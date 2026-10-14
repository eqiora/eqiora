use std::sync::Arc;

use eqiora_numerics::CommonSpatialPolicy;
use eqiora_realization::RealizationRevision;

use super::*;
use crate::evaluation_map::EvaluationMapPlan;
use crate::evaluation_map::tests::{fixture, program_for};

const LIMIT: usize = 1 << 30;
const POINTS: [[f64; 3]; 3] = [[2.0, 1.0, 0.0], [2.0, 3.0, 0.5], [2.0, 1.0, 0.0]];

#[test]
fn registered_mapped_product_contract() {
    analytic_chain_rule_shared_sum_and_duality_are_hand_derived();
    q1_and_tpfa_products_preserve_point_seed_axes_and_lineage();
    shared_selection_shape_profile_and_foreign_members_fail_closed();
    empty_scalar_singleton_and_zero_seed_axes_remain_typed();
    numerical_limits_inputs_and_accumulation_fail_closed();
    failed_actions_preserve_the_original_diagnostic_and_point_seed_occurrence();
    recomputed_products_keep_shared_sums_and_original_point_receipts();
}

#[test]
fn recomputed_products_keep_shared_sums_and_original_point_receipts() {
    let (_, program) = fixture();
    let program = Arc::new(program);
    let expected = POINTS.map(|point| program.evaluate(&point).unwrap());
    let map = EvaluationMapPlan::new(
        program.clone(),
        &POINTS
            .iter()
            .map(|point| point.as_slice())
            .collect::<Vec<_>>(),
        crate::EvaluationMapExecutionPolicy::new(
            2,
            eqiora_realization::Target::HostCpu {
                threads: std::num::NonZeroUsize::new(2).unwrap(),
            },
            crate::EvaluationMapRetention::Recompute,
            LIMIT,
        )
        .unwrap(),
    )
    .unwrap()
    .execute()
    .unwrap();
    let shared = [program.identity().inputs()[0].parameter()];
    let products = EvaluationMapProducts::new(&map, &shared, &[3], &[2], &[1], LIMIT).unwrap();
    let jvp = products.jvp(&[0.5, -1.0], &[0.25; 12]).unwrap();
    let cotangent = vec![1.0; program.identity().output_dimension()];
    let vjp = products.vjp(&cotangent.repeat(6)).unwrap();
    let mut sum = 0.0;
    for (point, member) in expected.iter().enumerate() {
        let ordinary = member.vjp(&cotangent).unwrap();
        sum += ordinary.input_cotangent()[0];
        for (seed, shared_tangent) in [0.5, -1.0].into_iter().enumerate() {
            assert_eq!(
                jvp.products()[seed * 3 + point],
                member.jvp(&[shared_tangent, 0.25, 0.25]).unwrap()
            );
            assert_eq!(vjp.products()[seed * 3 + point], ordinary);
        }
    }
    assert_eq!(vjp.shared_cotangents(), [sum, sum]);
    assert!(map.members().is_none());
}

#[test]
fn failed_actions_preserve_the_original_diagnostic_and_point_seed_occurrence() {
    let (_, program) = fixture();
    let map = EvaluationMapPlan::new(
        Arc::new(program),
        &[&POINTS[0], &POINTS[1], &POINTS[0]],
        crate::EvaluationMapExecutionPolicy::retained(LIMIT),
    )
    .unwrap()
    .execute()
    .unwrap();
    let products = EvaluationMapProducts::new(&map, &[], &[3], &[2], &[1], LIMIT).unwrap();
    // Seed-major grid [2,3]: point 1 / seed 1 is flat grid occurrence 4.
    // The positive elliptic response and constant-boundary derivative sum to
    // more than f64::MAX for this finite direction. The VJP boundary component
    // likewise sums more than one MAX cotangent. Neither result is representable.
    let bad_direction = [f64::MAX, 0.0, f64::MAX];
    let mut directions = vec![0.0; 6 * 3];
    directions[4 * 3..5 * 3].copy_from_slice(&bad_direction);
    let original = map.members().unwrap()[1].jvp(&bad_direction).unwrap_err();
    let error = products.jvp(&[], &directions).unwrap_err();
    assert_eq!(error.code(), original.code());
    assert_eq!(error.severity(), original.severity());
    assert!(error.message().ends_with(original.message()));
    assert!(
        error
            .message()
            .contains("mapped JVP point occurrence 1 [1], seed occurrence 1 [1], grid offset 4")
    );
    let width = map.plan().program_identity().output_dimension();
    let bad_cotangent = vec![f64::MAX; width];
    let mut cotangents = vec![0.0; 6 * width];
    cotangents[4 * width..5 * width].copy_from_slice(&bad_cotangent);
    let original = map.members().unwrap()[1].vjp(&bad_cotangent).unwrap_err();
    let error = products.vjp(&cotangents).unwrap_err();
    assert_eq!(error.code(), original.code());
    assert_eq!(error.severity(), original.severity());
    assert!(error.message().ends_with(original.message()));
    assert!(
        error
            .message()
            .contains("mapped VJP point occurrence 1 [1], seed occurrence 1 [1], grid offset 4")
    );
}

#[test]
fn analytic_chain_rule_shared_sum_and_duality_are_hand_derived() {
    // f(s,x) = (s*x+x*x, 2*s-3*x), s=2. The independently differentiated
    // columns are Ds=(x,2), Dx=(2+2*x,-3). No solve output determines them.
    let xs = [1.0, 3.0, 1.0];
    let dx = [1.0, -2.0, 0.25];
    let bars = [[1.0, 2.0], [-1.0, 1.0], [2.0, -1.0]];
    let mut shared = [0.0];
    let mut mapped = [0.0; 3];
    let mut jvps = Vec::new();
    for (index, x) in xs.into_iter().enumerate() {
        let tangent = assemble_tangent(2, &[0], &[1], &[0.5], &[dx[index]]);
        let jacobian = [[x, 2.0 + 2.0 * x], [2.0, -3.0]];
        let jvp = jacobian.map(|row| row[0] * tangent[0] + row[1] * tangent[1]);
        jvps.push(jvp);
        let gradient = [
            jacobian[0][0] * bars[index][0] + jacobian[1][0] * bars[index][1],
            jacobian[0][1] * bars[index][0] + jacobian[1][1] * bars[index][1],
        ];
        accumulate(
            &gradient,
            &[0],
            &[1],
            &mut shared,
            &mut mapped[index..index + 1],
        )
        .unwrap();
    }
    assert_eq!(shared, [4.0]);
    assert_eq!(mapped, [-2.0, -11.0, 11.0]);
    assert_eq!(jvps, [[4.5, -2.0], [-14.5, 7.0], [1.5, 0.25]]);
    let primal_pairing: f64 = jvps
        .iter()
        .zip(bars)
        .map(|(a, b)| a[0] * b[0] + a[1] * b[1])
        .sum();
    let dual_pairing = shared[0] * 0.5 + mapped.iter().zip(dx).map(|(a, b)| a * b).sum::<f64>();
    assert_eq!(primal_pairing, 24.75);
    assert_eq!(dual_pairing, 24.75);
    assert_ne!(shared[0] / 3.0, 4.0, "averaging is not the shared VJP");
    let mut drop_repeat = [0.0];
    for gradient in [[5.0, -2.0], [-1.0, -11.0]] {
        accumulate(&gradient, &[0], &[1], &mut drop_repeat, &mut [0.0]).unwrap();
    }
    // The frozen bars happen to give the repeated final member zero shared
    // contribution, but its nonzero mapped contribution must still be retained.
    assert_eq!(drop_repeat, [4.0]);
    assert_eq!(mapped.len(), 3);
    assert_ne!(mapped[2], 0.0);
}

#[test]
fn q1_and_tpfa_products_preserve_point_seed_axes_and_lineage() {
    let (document, _) = fixture();
    for spatial in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
        let program = Arc::new(program_for(
            &document,
            spatial,
            RealizationRevision::new(21),
            &["source_scale", "diffusion", "boundary_offset"],
        ));
        let points = [
            POINTS[0], POINTS[1], POINTS[2], POINTS[1], POINTS[0], POINTS[1],
        ];
        let map = EvaluationMapPlan::new(
            program.clone(),
            &points
                .iter()
                .map(|point| point.as_slice())
                .collect::<Vec<_>>(),
            crate::EvaluationMapExecutionPolicy::retained(LIMIT),
        )
        .unwrap()
        .execute()
        .unwrap();
        let width = program.identity().output_dimension();
        let mut expected_shared = [0.0; 4];
        // Each seed has a different input/output direction; expected products
        // are ordinary calls outside mapped-product composition.
        let mut expected = Vec::new();
        for (point, evaluation) in map.members().unwrap().iter().enumerate() {
            for (seed, shared_sum) in expected_shared.iter_mut().enumerate() {
                let tangent = direction(point, seed);
                let bar = cotangent(point, seed, width);
                let jvp = evaluation.jvp(&tangent).unwrap();
                let vjp = evaluation.vjp(&bar).unwrap();
                *shared_sum += vjp.input_cotangent()[0];
                expected.push((jvp, vjp));
            }
        }
        // Nontrivial nested point [2,3] and seed [2,2] axes, including interleaving.
        for positions in [[0, 1], [2, 3], [1, 3]] {
            let products = EvaluationMapProducts::new(
                &map,
                &program.identity().inputs()[..1]
                    .iter()
                    .map(crate::DifferentiableValueSignature::parameter)
                    .collect::<Vec<_>>(),
                &[2, 3],
                &[2, 2],
                &positions,
                LIMIT,
            )
            .unwrap();
            let mut tangents = vec![0.0; 48];
            let mut bars = vec![0.0; 24 * width];
            for point in 0..6 {
                for seed in 0..4 {
                    let position = independent_position(point, seed, positions);
                    assert_eq!(products.position(point, seed).unwrap(), position);
                    row_mut(&mut tangents, position, 2)
                        .copy_from_slice(&direction(point, seed)[1..]);
                    row_mut(&mut bars, position, width)
                        .copy_from_slice(&cotangent(point, seed, width));
                }
            }
            let shared = [0.25, 0.5, 0.75, 1.0];
            let jvp = products.jvp(&shared, &tangents).unwrap();
            let vjp = products.vjp(&bars).unwrap();
            assert_eq!(vjp.shared_shape(), [2, 2, 1]);
            assert_eq!(
                vjp.shared_inputs(),
                &program.identity().inputs()[..1]
                    .iter()
                    .map(crate::DifferentiableValueSignature::parameter)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                vjp.mapped_inputs(),
                &program.identity().inputs()[1..]
                    .iter()
                    .map(crate::DifferentiableValueSignature::parameter)
                    .collect::<Vec<_>>()
            );
            assert_eq!(vjp.shared_cotangents(), expected_shared);
            assert_eq!(jvp.products().len(), 24);
            for point in 0..6 {
                for seed in 0..4 {
                    let position = independent_position(point, seed, positions);
                    let (expected_jvp, expected_vjp) = &expected[point * 4 + seed];
                    assert_eq!(&jvp.products()[position], expected_jvp);
                    assert_eq!(&vjp.products()[position], expected_vjp);
                    assert_eq!(
                        row(vjp.mapped_cotangents(), position, 2),
                        &expected_vjp.input_cotangent()[1..]
                    );
                    assert_eq!(
                        jvp.products()[position].evidence().point().coordinates(),
                        points[point]
                    );
                    assert_eq!(
                        jvp.products()[position].evidence().linearization_state(),
                        LinearizationState::Reused
                    );
                }
            }
            // Retained derivatives still refer to their original primal after
            // another point is independently accepted by the same Program.
            program.evaluate(&[5.0, 2.0, 1.0]).unwrap();
            let repeated = products.vjp(&bars).unwrap();
            assert_eq!(repeated.products(), vjp.products());
            assert_eq!(repeated.shared_cotangents(), vjp.shared_cotangents());
        }
    }
}

fn direction(point: usize, seed: usize) -> [f64; 3] {
    [
        (seed + 1) as f64 * 0.25,
        (point + 1) as f64 * -0.125,
        seed as f64 * 0.25,
    ]
}
fn cotangent(point: usize, seed: usize, width: usize) -> Vec<f64> {
    (0..width)
        .map(|index| {
            if (index + point + seed).is_multiple_of(2) {
                0.25 * (seed + 1) as f64
            } else {
                -0.125 * (point + 1) as f64
            }
        })
        .collect()
}
fn independent_position(point: usize, seed: usize, positions: [usize; 2]) -> usize {
    match positions {
        [0, 1] => point * 4 + seed,
        [2, 3] => seed * 6 + point,
        [1, 3] => (seed / 2) * 12 + (point / 3) * 6 + (seed % 2) * 3 + point % 3,
        _ => panic!("unplanned test layout"),
    }
}

#[test]
fn shared_selection_shape_profile_and_foreign_members_fail_closed() {
    let (document, program) = fixture();
    let program = Arc::new(program);
    let plan = EvaluationMapPlan::new(
        program.clone(),
        &[&POINTS[0], &POINTS[1], &POINTS[2]],
        crate::EvaluationMapExecutionPolicy::retained(LIMIT),
    )
    .unwrap();
    let map = plan.execute().unwrap();
    let id = program.identity().inputs()[0].parameter();
    for shared in [
        vec![id, id],
        vec![program.identity().inputs()[1].parameter()],
        vec![Id::new()],
    ] {
        assert!(EvaluationMapProducts::new(&map, &shared, &[3], &[2], &[0], LIMIT).is_err());
    }
    for (point, seed, axes) in [
        (vec![2], vec![2], vec![0]),
        (vec![3], vec![2], vec![2]),
        (vec![1, 3], vec![2], vec![0, 0]),
        (vec![3], vec![usize::MAX, 2], vec![0]),
        (vec![3], vec![1; 32], vec![0]),
    ] {
        assert!(EvaluationMapProducts::new(&map, &[id], &point, &seed, &axes, LIMIT).is_err());
    }
    // No public terminal-to-product conversion exists; a private partial-map
    // mutation also fails the complete point inventory before derivative work.
    let mut partial = map.clone();
    partial.retained_members_mut().pop();
    assert!(EvaluationMapProducts::new(&partial, &[id], &[3], &[1], &[0], LIMIT).is_err());
    assert!(EvaluationMapProducts::new(&partial, &[id], &[2], &[1], &[0], LIMIT).is_err());
    let foreign = program_for(
        &document,
        CommonSpatialPolicy::CellCenteredTpfa,
        RealizationRevision::new(21),
        &["source_scale", "diffusion", "boundary_offset"],
    );
    let mut substituted = map.clone();
    substituted.retained_members_mut()[0] = foreign.evaluate(&POINTS[0]).unwrap();
    assert!(EvaluationMapProducts::new(&substituted, &[id], &[3], &[1], &[0], LIMIT).is_err());
    let expected = &map.members().unwrap()[0];
    let foreign_action = foreign
        .evaluate(&POINTS[0])
        .unwrap()
        .jvp(&[1.0, 0.0, 0.0])
        .unwrap();
    assert!(
        validate_evidence(
            expected,
            foreign_action.evidence(),
            DifferentiationMode::Jvp
        )
        .is_err()
    );
    let stale_action = map.members().unwrap()[1].jvp(&[1.0, 0.0, 0.0]).unwrap();
    assert!(
        validate_evidence(expected, stale_action.evidence(), DifferentiationMode::Jvp).is_err()
    );
    assert!(
        validate_evidence(
            expected,
            expected.primal().evidence(),
            DifferentiationMode::Jvp
        )
        .is_err()
    );
    // Shared selections preserve their explicit order, not a hidden sorting by
    // Parameter ID or the order of the surrounding Program coordinates.
    let pair = EvaluationMapPlan::new(
        program.clone(),
        &[&POINTS[0], &[2.0, 2.0, 0.0]],
        crate::EvaluationMapExecutionPolicy::retained(LIMIT),
    )
    .unwrap()
    .execute()
    .unwrap();
    let selected = [program.identity().inputs()[2].parameter(), id];
    let reordered = EvaluationMapProducts::new(&pair, &selected, &[2], &[], &[0], LIMIT).unwrap();
    let jvp = reordered.jvp(&[0.5, 0.25], &[-0.125, 0.25]).unwrap();
    for (index, value) in [-0.125, 0.25].into_iter().enumerate() {
        assert_eq!(
            jvp.products()[index],
            pair.members().unwrap()[index]
                .jvp(&[0.25, value, 0.5])
                .unwrap()
        );
    }
    let width = program.identity().output_dimension();
    let reverse = reordered.vjp(&vec![1.0; 2 * width]).unwrap();
    let references = pair
        .members()
        .unwrap()
        .iter()
        .map(|evaluation| evaluation.vjp(&vec![1.0; width]).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(reverse.shared_inputs(), selected);
    assert_eq!(
        reverse.shared_cotangents(),
        [
            references[0].input_cotangent()[2] + references[1].input_cotangent()[2],
            references[0].input_cotangent()[0] + references[1].input_cotangent()[0]
        ]
    );
    let invalid = [2.0, -1.0, 0.0];
    assert!(
        EvaluationMapPlan::new(
            program,
            &[&POINTS[0], &invalid],
            crate::EvaluationMapExecutionPolicy::retained(LIMIT)
        )
        .unwrap()
        .execute()
        .is_err()
    );
}

#[test]
fn empty_scalar_singleton_and_zero_seed_axes_remain_typed() {
    let (_, program) = fixture();
    let program = Arc::new(program);
    let id = program.identity().inputs()[0].parameter();
    let empty = EvaluationMapPlan::new(
        program.clone(),
        &[],
        crate::EvaluationMapExecutionPolicy::retained(0),
    )
    .unwrap()
    .execute()
    .unwrap();
    let products = EvaluationMapProducts::new(&empty, &[id], &[0], &[2], &[0], LIMIT).unwrap();
    assert!(
        products
            .jvp(&[0.0, 1.0], &[])
            .unwrap()
            .products()
            .is_empty()
    );
    assert_eq!(products.vjp(&[]).unwrap().shared_cotangents(), [0.0, 0.0]);
    let output_width = program.identity().output_dimension();
    let overflow =
        EvaluationMapProducts::new(&empty, &[], &[0], &[usize::MAX / output_width + 1], &[0], 0)
            .unwrap_err();
    assert_eq!(overflow.message(), "invalid mapped product axes: Overflow");
    let map = EvaluationMapPlan::new(
        program,
        &[&POINTS[0]],
        crate::EvaluationMapExecutionPolicy::retained(LIMIT),
    )
    .unwrap()
    .execute()
    .unwrap();
    let scalar = EvaluationMapProducts::new(&map, &[id], &[], &[], &[], LIMIT).unwrap();
    assert!(scalar.shape().is_empty());
    assert_eq!(scalar.jvp(&[1.0], &[0.0, 0.0]).unwrap().products().len(), 1);
    let no_seeds = EvaluationMapProducts::new(&map, &[id], &[1], &[0], &[1], 0).unwrap();
    assert!(no_seeds.jvp(&[], &[]).unwrap().products().is_empty());
    assert!(no_seeds.vjp(&[]).unwrap().products().is_empty());
    let all_shared = EvaluationMapProducts::new(
        &map,
        &map.plan()
            .program_identity()
            .inputs()
            .iter()
            .map(crate::DifferentiableValueSignature::parameter)
            .collect::<Vec<_>>(),
        &[1],
        &[1],
        &[0],
        LIMIT,
    )
    .unwrap();
    assert_eq!(
        all_shared
            .jvp(&[1.0, 0.0, 0.0], &[])
            .unwrap()
            .products()
            .len(),
        1
    );
    let all_mapped = EvaluationMapProducts::new(&map, &[], &[1], &[1], &[0], LIMIT).unwrap();
    assert_eq!(
        all_mapped
            .jvp(&[], &[1.0, 0.0, 0.0])
            .unwrap()
            .products()
            .len(),
        1
    );
}

#[test]
fn numerical_limits_inputs_and_accumulation_fail_closed() {
    let (_, program) = fixture();
    let program = Arc::new(program);
    let map = EvaluationMapPlan::new(
        program,
        &[&POINTS[0]],
        crate::EvaluationMapExecutionPolicy::retained(LIMIT),
    )
    .unwrap()
    .execute()
    .unwrap();
    let id = map.plan().program_identity().inputs()[0].parameter();
    let products = EvaluationMapProducts::new(&map, &[id], &[1], &[2], &[0], LIMIT).unwrap();
    let required = products.estimated_numerical_bytes();
    // Independently inventory one retained JVP and VJP: each action clones
    // its accepted point; JVP retains primal+tangent, VJP primal+gradient.
    // The reverse aggregate additionally retains mapped values/shared sums.
    let outputs = map.plan().program_identity().output_dimension();
    let inputs = map.plan().program_identity().input_dimension();
    let jvp_values = 2 * (2 * outputs + inputs);
    let vjp_values = 2 * (outputs + inputs + inputs);
    let aggregate_values = 2 * 2 + 2;
    assert_eq!(required, (jvp_values + vjp_values + aggregate_values) * 8);
    assert!(EvaluationMapProducts::new(&map, &[id], &[1], &[2], &[0], required).is_ok());
    assert!(EvaluationMapProducts::new(&map, &[id], &[1], &[2], &[0], required - 1).is_err());
    assert!(numerical_bytes(usize::MAX, 2, 2, 2, 1, 1).is_err());
    assert!(addressable::<DifferentiableJvp>(usize::MAX).is_err());
    assert!(products.jvp(&[1.0], &[0.0; 4]).is_err());
    assert!(products.jvp(&[1.0, f64::NAN], &[0.0; 4]).is_err());
    assert!(products.jvp(&[1.0, 0.0], &[f64::INFINITY; 4]).is_err());
    assert!(products.vjp(&[]).is_err());
    let width = map.plan().program_identity().output_dimension();
    assert!(products.vjp(&vec![f64::NAN; 2 * width]).is_err());
    assert!(finite(&[f64::INFINITY]).is_err());
    assert!(accumulate(&[f64::MAX], &[0], &[], &mut [f64::MAX], &mut []).is_err());
}
