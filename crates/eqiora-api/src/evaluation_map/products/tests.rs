use std::sync::Arc;

use eqiora_numerics::CommonSpatialPolicy;
use eqiora_realization::RealizationRevision;

use super::*;
use crate::evaluation_map::tests::{fixture, program_for};
use crate::evaluation_map::{EvaluationMapPlan, Parameter};

const LIMIT: usize = 1 << 30;
const POINTS: [[f64; 3]; 3] = [[2.0, 1.0, 0.0], [2.0, 3.0, 0.5], [2.0, 1.0, 0.0]];

// Preserve independent numeric fixture data while binding it to native derivative roles.
pub(in crate::evaluation_map) trait FixtureProducts {
    fn fixture_jvp(
        &self,
        shared: &[f64],
        mapped: &[f64],
    ) -> Result<EvaluationMapResult<DifferentiableTangent>, Diagnostic>;
    fn fixture_vjp(
        &self,
        values: &[f64],
    ) -> Result<EvaluationMapResult<DifferentiableCotangent>, Diagnostic>;
}

impl FixtureProducts for EvaluationMapProducts<'_> {
    fn fixture_jvp(
        &self,
        shared: &[f64],
        mapped: &[f64],
    ) -> Result<EvaluationMapResult<DifferentiableTangent>, Diagnostic> {
        let identity = self.map.plan().program_identity();
        let (shared_leaves, mapped_leaves) =
            super::super::partition::leaf_indices(identity, self.map.plan().shared_inputs())?;
        let select = |indices: Vec<usize>| {
            indices
                .into_iter()
                .map(|i| identity.inputs()[i].clone())
                .collect::<Arc<[_]>>()
        };
        self.jvp(
            &fixture_rows(
                select(shared_leaves),
                shared,
                self.seeds,
                DifferentiableTangent::from_coordinates,
            )?,
            &fixture_rows(
                select(mapped_leaves),
                mapped,
                self.cells,
                DifferentiableTangent::from_coordinates,
            )?,
        )
    }
    fn fixture_vjp(
        &self,
        values: &[f64],
    ) -> Result<EvaluationMapResult<DifferentiableCotangent>, Diagnostic> {
        self.vjp(&fixture_rows(
            self.map.plan().program_identity().outputs().to_vec().into(),
            values,
            self.cells,
            DifferentiableCotangent::from_coordinates,
        )?)
    }
}

fn fixture_rows<T>(
    signatures: Arc<[DifferentiableValueSignature]>,
    values: &[f64],
    count: usize,
    bind: impl Fn(Arc<[DifferentiableValueSignature]>, Vec<f64>) -> Result<T, Diagnostic>,
) -> Result<Vec<T>, Diagnostic> {
    let width = crate::differentiation::signature::dimension(&signatures)?;
    admit(values, multiply(count, width)?)?;
    (0..count)
        .map(|i| bind(signatures.clone(), row(values, i, width).to_vec()))
        .collect()
}

// Select shared leaves once when constructing this three-scalar-input fixture.
fn partitioned_plan(
    program: Arc<crate::DifferentiableProgram>,
    shared: &[usize],
    points: &[&[f64]],
    policy: crate::EvaluationMapExecutionPolicy,
) -> Result<EvaluationMapPlan, Diagnostic> {
    let shared = shared
        .iter()
        .map(|&index| program.identity().inputs()[index].parameter())
        .collect::<Vec<_>>();
    let points = crate::evaluation_map::tests::typed_points(&program, points);
    EvaluationMapPlan::new(program, &points, &shared, policy)
}

#[test]
fn registered_mapped_product_contract() {
    analytic_chain_rule_shared_sum_and_duality_are_hand_derived();
    q1_and_tpfa_products_preserve_point_seed_axes_and_lineage();
    shared_selection_shape_profile_and_foreign_members_fail_closed();
    empty_scalar_singleton_and_zero_seed_axes_remain_typed();
    numerical_limits_inputs_and_accumulation_fail_closed();
    failed_actions_preserve_the_original_diagnostic_and_point_seed_occurrence();
    recomputed_products_keep_shared_sums_and_original_point_receipts();
    derivative_inputs_require_exact_ordered_base_spaces_and_grid_counts();
}

#[test]
fn derivative_inputs_require_exact_ordered_base_spaces_and_grid_counts() {
    let (_, program) = fixture();
    let program = Arc::new(program);
    let map = partitioned_plan(
        program.clone(),
        &[0],
        &[&POINTS[0], &POINTS[1]],
        crate::EvaluationMapExecutionPolicy::retained(LIMIT),
    )
    .unwrap()
    .execute()
    .unwrap();
    let products = EvaluationMapProducts::new(&map, &[2], &[], &[0], LIMIT).unwrap();
    let inputs = program.identity().inputs();
    let shared = DifferentiableTangent::from_components(&inputs[..1], &[&[(1.0, 0.0)]]).unwrap();
    let mapped =
        DifferentiableTangent::from_components(&inputs[1..], &[&[(0.5, 0.0)], &[(-0.25, 0.0)]])
            .unwrap();
    let output = DifferentiableCotangent::from_coordinates(
        program.identity().outputs().to_vec().into(),
        vec![1.0; program.identity().output_dimension()],
    )
    .unwrap();
    // Ordinary positive actions reach both native gates before rejection probes.
    products
        .jvp(
            std::slice::from_ref(&shared),
            &[mapped.clone(), mapped.clone()],
        )
        .unwrap();
    products.vjp(&[output.clone(), output.clone()]).unwrap();
    for error in [
        products
            .jvp(&[], &[mapped.clone(), mapped.clone()])
            .unwrap_err(),
        products
            .jvp(std::slice::from_ref(&shared), std::slice::from_ref(&mapped))
            .unwrap_err(),
        products.vjp(std::slice::from_ref(&output)).unwrap_err(),
    ] {
        assert!(error.message().contains("product count differs"));
    }
    let wrong_shared =
        DifferentiableTangent::from_components(&inputs[1..2], &[&[(1.0, 0.0)]]).unwrap();
    let reordered = DifferentiableTangent::from_components(
        &[inputs[2].clone(), inputs[1].clone()],
        &[&[(0.5, 0.0)], &[(-0.25, 0.0)]],
    )
    .unwrap();
    // Equal numeric widths cannot substitute a different input or leaf order;
    // the last occurrence is checked as well as the first.
    for error in [
        products
            .jvp(&[wrong_shared], &[mapped.clone(), mapped.clone()])
            .unwrap_err(),
        products.jvp(&[shared], &[mapped, reordered]).unwrap_err(),
    ] {
        assert!(error.message().contains("exact ordered signatures"));
    }
    let foreign = crate::evaluation_map::tests::foreign_fixture();
    assert_eq!(
        foreign.identity().output_dimension(),
        program.identity().output_dimension()
    );
    let foreign_output = DifferentiableCotangent::from_coordinates(
        foreign.identity().outputs().to_vec().into(),
        vec![1.0; foreign.identity().output_dimension()],
    )
    .unwrap();
    assert!(
        products
            .vjp(&[output, foreign_output])
            .unwrap_err()
            .message()
            .contains("exact ordered signatures")
    );
}

#[test]
fn recomputed_products_keep_shared_sums_and_original_point_receipts() {
    let (_, program) = fixture();
    let program = Arc::new(program);
    let expected = POINTS.map(|point| {
        program
            .evaluate(
                &program
                    .point_from_coordinates(&point)
                    .expect("validated native point"),
            )
            .unwrap()
    });
    let map = partitioned_plan(
        program.clone(),
        &[0],
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
    let products = EvaluationMapProducts::new(&map, &[3], &[2], &[1], LIMIT).unwrap();
    let jvp = products.fixture_jvp(&[0.5, -1.0], &[0.25; 12]).unwrap();
    let cotangent = vec![1.0; program.identity().output_dimension()];
    let vjp = products.fixture_vjp(&cotangent.repeat(6)).unwrap();
    let mut sum = 0.0;
    for (point, member) in expected.iter().enumerate() {
        let ordinary = member
            .vjp(
                &member
                    .identity()
                    .cotangent_from_coordinates(&cotangent)
                    .unwrap(),
            )
            .unwrap();
        sum += ordinary.derivative().coordinates()[0];
        for (seed, shared_tangent) in [0.5, -1.0].into_iter().enumerate() {
            assert_eq!(
                jvp.products()[seed * 3 + point],
                member
                    .jvp(
                        &member
                            .identity()
                            .tangent_from_coordinates(&[shared_tangent, 0.25, 0.25])
                            .unwrap()
                    )
                    .unwrap()
            );
            assert_eq!(vjp.products()[seed * 3 + point], ordinary);
        }
    }
    assert_eq!(
        vjp.shared_cotangents()
            .iter()
            .map(|value| value.coordinates()[0])
            .collect::<Vec<_>>(),
        [sum, sum]
    );
    assert!(map.members().is_none());
}

#[test]
fn failed_actions_preserve_the_original_diagnostic_and_point_seed_occurrence() {
    let (_, program) = fixture();
    let map = EvaluationMapPlan::new(
        Arc::new(program.clone()),
        &crate::evaluation_map::tests::typed_points(
            &program,
            &[&POINTS[0], &POINTS[1], &POINTS[0]],
        ),
        &[],
        crate::EvaluationMapExecutionPolicy::retained(LIMIT),
    )
    .unwrap()
    .execute()
    .unwrap();
    let products = EvaluationMapProducts::new(&map, &[3], &[2], &[1], LIMIT).unwrap();
    // Seed-major grid [2,3]: point 1 / seed 1 is flat grid occurrence 4.
    // The positive elliptic response and constant-boundary derivative sum to
    // more than f64::MAX for this finite direction. The VJP boundary component
    // likewise sums more than one MAX cotangent. Neither result is representable.
    let bad_direction = [f64::MAX, 0.0, f64::MAX];
    let mut directions = vec![0.0; 6 * 3];
    directions[4 * 3..5 * 3].copy_from_slice(&bad_direction);
    let original = map.members().unwrap()[1]
        .jvp(
            &map.members().unwrap()[1]
                .identity()
                .tangent_from_coordinates(&bad_direction)
                .unwrap(),
        )
        .unwrap_err();
    let error = products.fixture_jvp(&[], &directions).unwrap_err();
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
    let original = map.members().unwrap()[1]
        .vjp(
            &map.members().unwrap()[1]
                .identity()
                .cotangent_from_coordinates(&bad_cotangent)
                .unwrap(),
        )
        .unwrap_err();
    let error = products.fixture_vjp(&cotangents).unwrap_err();
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
        let tangent = crate::evaluation_map::partition::assemble_coordinates(
            2,
            &[0],
            &[1],
            &[0.5],
            &[dx[index]],
        );
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
        let map = partitioned_plan(
            program.clone(),
            &[0],
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
                let jvp = evaluation
                    .jvp(
                        &evaluation
                            .identity()
                            .tangent_from_coordinates(&tangent)
                            .unwrap(),
                    )
                    .unwrap();
                let vjp = evaluation
                    .vjp(
                        &evaluation
                            .identity()
                            .cotangent_from_coordinates(&bar)
                            .unwrap(),
                    )
                    .unwrap();
                *shared_sum += vjp.derivative().coordinates()[0];
                expected.push((jvp, vjp));
            }
        }
        // Nontrivial nested point [2,3] and seed [2,2] axes, including interleaving.
        for positions in [[0, 1], [2, 3], [1, 3]] {
            let products =
                EvaluationMapProducts::new(&map, &[2, 3], &[2, 2], &positions, LIMIT).unwrap();
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
            let jvp = products.fixture_jvp(&shared, &tangents).unwrap();
            let vjp = products.fixture_vjp(&bars).unwrap();
            assert_eq!(vjp.shared_shape(), [2, 2]);
            assert_eq!(vjp.shared_signatures(), &program.identity().inputs()[..1]);
            assert_eq!(vjp.mapped_signatures(), &program.identity().inputs()[1..]);
            assert_eq!(
                vjp.shared_cotangents()
                    .iter()
                    .map(|value| value.coordinates()[0])
                    .collect::<Vec<_>>(),
                expected_shared
            );
            assert_eq!(jvp.products().len(), 24);
            for point in 0..6 {
                for seed in 0..4 {
                    let position = independent_position(point, seed, positions);
                    let (expected_jvp, expected_vjp) = &expected[point * 4 + seed];
                    assert_eq!(&jvp.products()[position], expected_jvp);
                    assert_eq!(&vjp.products()[position], expected_vjp);
                    assert_eq!(
                        vjp.mapped_cotangents()[position].coordinates(),
                        &expected_vjp.derivative().coordinates()[1..]
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
            program
                .evaluate(
                    &program
                        .point_from_coordinates(&[5.0, 2.0, 1.0])
                        .expect("validated native point"),
                )
                .unwrap();
            let repeated = products.fixture_vjp(&bars).unwrap();
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
    let plan = partitioned_plan(
        program.clone(),
        &[0],
        &[&POINTS[0], &POINTS[1], &POINTS[2]],
        crate::EvaluationMapExecutionPolicy::retained(LIMIT),
    )
    .unwrap();
    let map = plan.execute().unwrap();
    let id = program.identity().inputs()[0].parameter();
    let points = crate::evaluation_map::tests::typed_points(&program, &[&POINTS[0], &POINTS[1]]);
    for (shared, expected_gate) in [
        (vec![id, id], "repeats a Parameter"),
        (
            vec![program.identity().inputs()[1].parameter()],
            "differs between planned points",
        ),
        (vec![eqiora_core::Id::new()], "foreign to the exact Program"),
    ] {
        let error = EvaluationMapPlan::new(
            program.clone(),
            &points,
            &shared,
            crate::EvaluationMapExecutionPolicy::retained(LIMIT),
        )
        .unwrap_err();
        assert!(error.message().contains(expected_gate));
    }
    let duplicate = crate::DifferentiableValues::from_components(
        &[
            program.identity().inputs()[0].clone(),
            program.identity().inputs()[0].clone(),
        ],
        &[&[(2.0, 0.0)], &[(2.0, 0.0)]],
    )
    .unwrap_err();
    assert!(duplicate.message().contains("distinct entities"));

    for (point, seed, axes) in [
        (vec![2], vec![2], vec![0]),
        (vec![3], vec![2], vec![2]),
        (vec![1, 3], vec![2], vec![0, 0]),
        (vec![3], vec![usize::MAX, 2], vec![0]),
        (vec![3], vec![1; 32], vec![0]),
    ] {
        assert!(EvaluationMapProducts::new(&map, &point, &seed, &axes, LIMIT).is_err());
    }
    // No public terminal-to-product conversion exists; a private partial-map
    // mutation also fails the complete point inventory before derivative work.
    let mut partial = map.clone();
    partial.retained_members_mut().pop();
    let missing = crate::EvaluationMapEstimator::evaluate(
        &partial,
        crate::EnsembleReduction::Sum,
        &[],
        &[1.0; 3],
    )
    .unwrap_err();
    assert!(missing.message().contains("missing an accepted occurrence"));
    assert!(EvaluationMapProducts::new(&partial, &[3], &[1], &[0], LIMIT).is_err());
    assert!(EvaluationMapProducts::new(&partial, &[2], &[1], &[0], LIMIT).is_err());
    let foreign = program_for(
        &document,
        CommonSpatialPolicy::CellCenteredTpfa,
        RealizationRevision::new(21),
        &["source_scale", "diffusion", "boundary_offset"],
    );
    let mut substituted = map.clone();
    substituted.retained_members_mut()[0] = foreign
        .evaluate(
            &foreign
                .point_from_coordinates(&POINTS[0])
                .expect("validated native point"),
        )
        .unwrap();
    assert!(EvaluationMapProducts::new(&substituted, &[3], &[1], &[0], LIMIT).is_err());
    let expected = &map.members().unwrap()[0];
    let foreign_action = foreign
        .evaluate(
            &foreign
                .point_from_coordinates(&POINTS[0])
                .expect("validated native point"),
        )
        .unwrap()
        .jvp(
            &foreign
                .identity()
                .tangent_from_coordinates(&[1.0, 0.0, 0.0])
                .unwrap(),
        )
        .unwrap();
    assert!(
        validate_evidence(
            expected,
            foreign_action.evidence(),
            DifferentiationMode::Jvp
        )
        .is_err()
    );
    let stale_action = map.members().unwrap()[1]
        .jvp(
            &map.members().unwrap()[1]
                .identity()
                .tangent_from_coordinates(&[1.0, 0.0, 0.0])
                .unwrap(),
        )
        .unwrap();
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
    let selected = [program.identity().inputs()[2].parameter(), id];
    let pair = partitioned_plan(
        program.clone(),
        &[2, 0],
        &[&POINTS[0], &[2.0, 2.0, 0.0]],
        crate::EvaluationMapExecutionPolicy::retained(LIMIT),
    )
    .unwrap()
    .execute()
    .unwrap();
    assert_eq!(pair.plan().shared_inputs(), selected);
    let reordered = EvaluationMapProducts::new(&pair, &[2], &[], &[0], LIMIT).unwrap();
    let jvp = reordered
        .fixture_jvp(&[0.5, 0.25], &[-0.125, 0.25])
        .unwrap();
    for (index, value) in [-0.125, 0.25].into_iter().enumerate() {
        assert_eq!(
            jvp.products()[index],
            pair.members().unwrap()[index]
                .jvp(
                    &pair.members().unwrap()[index]
                        .identity()
                        .tangent_from_coordinates(&[0.25, value, 0.5])
                        .unwrap()
                )
                .unwrap()
        );
    }
    let width = program.identity().output_dimension();
    let reverse = reordered.fixture_vjp(&vec![1.0; 2 * width]).unwrap();
    let references = pair
        .members()
        .unwrap()
        .iter()
        .map(|evaluation| {
            evaluation
                .vjp(
                    &evaluation
                        .identity()
                        .cotangent_from_coordinates(&vec![1.0; width])
                        .unwrap(),
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        reverse.shared_signatures(),
        &[
            program.identity().inputs()[2].clone(),
            program.identity().inputs()[0].clone()
        ]
    );
    assert_eq!(
        reverse.shared_cotangents()[0].coordinates(),
        [
            references[0].derivative().coordinates()[2]
                + references[1].derivative().coordinates()[2],
            references[0].derivative().coordinates()[0]
                + references[1].derivative().coordinates()[0]
        ]
    );
    let invalid = [2.0, -1.0, 0.0];
    assert!(
        EvaluationMapPlan::new(
            program.clone(),
            &crate::evaluation_map::tests::typed_points(&program, &[&POINTS[0], &invalid]),
            &[],
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
    let shared_bytes = size_of::<Parameter>();
    assert!(
        partitioned_plan(
            program.clone(),
            &[0],
            &[],
            crate::EvaluationMapExecutionPolicy::retained(shared_bytes - 1)
        )
        .is_err()
    );
    let empty_plan = partitioned_plan(
        program.clone(),
        &[0],
        &[],
        crate::EvaluationMapExecutionPolicy::retained(shared_bytes),
    )
    .unwrap();
    assert_eq!(empty_plan.estimated_storage_bytes(), shared_bytes);
    let empty = empty_plan.execute().unwrap();
    let products = EvaluationMapProducts::new(&empty, &[0], &[2], &[0], LIMIT).unwrap();
    assert!(
        products
            .fixture_jvp(&[0.0, 1.0], &[])
            .unwrap()
            .products()
            .is_empty()
    );
    let reverse = products.fixture_vjp(&[]).unwrap();
    assert_eq!(
        reverse.shared_signatures(),
        &program.identity().inputs()[..1]
    );
    assert_eq!(
        reverse.mapped_signatures(),
        &program.identity().inputs()[1..]
    );
    assert_eq!(reverse.shared_cotangents().len(), 2);
    assert!(
        reverse
            .shared_cotangents()
            .iter()
            .all(|value| value.coordinates() == [0.0])
    );
    assert!(reverse.mapped_cotangents().is_empty());
    let output_width = program.identity().output_dimension();
    // No shared result allocation may mask the logical output-stride overflow.
    let empty_mapped = EvaluationMapPlan::new(
        program.clone(),
        &[],
        &[],
        crate::EvaluationMapExecutionPolicy::retained(0),
    )
    .unwrap()
    .execute()
    .unwrap();
    let empty_records = 2 * size_of::<DifferentiableCotangent>();
    assert!(
        EvaluationMapProducts::new(&empty_mapped, &[0], &[2], &[0], empty_records - 1).is_err()
    );
    let empty_products =
        EvaluationMapProducts::new(&empty_mapped, &[0], &[2], &[0], empty_records).unwrap();
    let empty_reverse = empty_products.fixture_vjp(&[]).unwrap();
    assert_eq!(empty_reverse.shared_cotangents().len(), 2);
    assert!(
        empty_reverse
            .shared_cotangents()
            .iter()
            .all(|value| value.base_signatures().is_empty())
    );
    assert_eq!(
        empty_reverse.mapped_signatures(),
        program.identity().inputs()
    );
    let overflow = EvaluationMapProducts::new(
        &empty_mapped,
        &[0],
        &[usize::MAX / output_width + 1],
        &[0],
        0,
    )
    .unwrap_err();
    assert_eq!(overflow.message(), "invalid mapped product axes: Overflow");
    let map = partitioned_plan(
        program.clone(),
        &[0],
        &[&POINTS[0]],
        crate::EvaluationMapExecutionPolicy::retained(LIMIT),
    )
    .unwrap()
    .execute()
    .unwrap();
    let scalar = EvaluationMapProducts::new(&map, &[], &[], &[], LIMIT).unwrap();
    assert!(scalar.shape().is_empty());
    assert_eq!(
        scalar
            .fixture_jvp(&[1.0], &[0.0, 0.0])
            .unwrap()
            .products()
            .len(),
        1
    );
    let no_seeds = EvaluationMapProducts::new(&map, &[1], &[0], &[1], 0).unwrap();
    assert!(
        no_seeds
            .fixture_jvp(&[], &[])
            .unwrap()
            .products()
            .is_empty()
    );
    let no_seed_reverse = no_seeds.fixture_vjp(&[]).unwrap();
    assert!(no_seed_reverse.products().is_empty());
    assert!(no_seed_reverse.shared_cotangents().is_empty());
    assert_eq!(
        no_seed_reverse.shared_signatures(),
        &program.identity().inputs()[..1]
    );
    let shared_map = partitioned_plan(
        program.clone(),
        &[0, 1, 2],
        &[&POINTS[0]],
        crate::EvaluationMapExecutionPolicy::retained(LIMIT),
    )
    .unwrap()
    .execute()
    .unwrap();
    let all_shared = EvaluationMapProducts::new(&shared_map, &[1], &[1], &[0], LIMIT).unwrap();
    assert_eq!(
        all_shared
            .fixture_jvp(&[1.0, 0.0, 0.0], &[])
            .unwrap()
            .products()
            .len(),
        1
    );
    let mapped_map = EvaluationMapPlan::new(
        program.clone(),
        &crate::evaluation_map::tests::typed_points(&program, &[&POINTS[0]]),
        &[],
        crate::EvaluationMapExecutionPolicy::retained(LIMIT),
    )
    .unwrap()
    .execute()
    .unwrap();
    assert_eq!(mapped_map.plan().points(), shared_map.plan().points());
    assert_ne!(mapped_map.plan(), shared_map.plan());
    assert!(mapped_map.plan().shared_inputs().is_empty());
    let all_mapped = EvaluationMapProducts::new(&mapped_map, &[1], &[1], &[0], LIMIT).unwrap();
    assert_eq!(
        all_mapped
            .fixture_jvp(&[], &[1.0, 0.0, 0.0])
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
    let map = partitioned_plan(
        program.clone(),
        &[0],
        &[&POINTS[0]],
        crate::EvaluationMapExecutionPolicy::retained(LIMIT),
    )
    .unwrap()
    .execute()
    .unwrap();
    let products = EvaluationMapProducts::new(&map, &[1], &[2], &[0], LIMIT).unwrap();
    let required = products.estimated_numerical_bytes();
    // Independently inventory one retained JVP and VJP: each action clones
    // its accepted point; JVP retains primal+tangent, VJP primal+gradient.
    // The reverse aggregate additionally retains mapped values/shared sums.
    let outputs = map.plan().program_identity().output_dimension();
    let inputs = map.plan().program_identity().input_dimension();
    let jvp_values = 2 * (2 * outputs + inputs);
    let vjp_values = 2 * (outputs + inputs + inputs);
    let aggregate_values = 2 * 2 + 2;
    let aggregate_records = 4 * size_of::<DifferentiableCotangent>();
    assert_eq!(
        required,
        (jvp_values + vjp_values + aggregate_values) * 8 + aggregate_records
    );
    assert!(EvaluationMapProducts::new(&map, &[1], &[2], &[0], required).is_ok());
    assert!(EvaluationMapProducts::new(&map, &[1], &[2], &[0], required - 1).is_err());
    assert!(numerical_bytes(usize::MAX, 2, 2, 2, 1, 1).is_err());
    assert!(addressable::<DifferentiableResult<DifferentiableTangent>>(usize::MAX).is_err());
    assert!(products.fixture_jvp(&[1.0], &[0.0; 4]).is_err());
    assert!(products.fixture_jvp(&[1.0, f64::NAN], &[0.0; 4]).is_err());
    assert!(
        products
            .fixture_jvp(&[1.0, 0.0], &[f64::INFINITY; 4])
            .is_err()
    );
    assert!(products.fixture_vjp(&[]).is_err());
    let width = map.plan().program_identity().output_dimension();
    assert!(products.fixture_vjp(&vec![f64::NAN; 2 * width]).is_err());
    assert!(accumulate(&[f64::MAX], &[0], &[], &mut [f64::MAX], &mut []).is_err());
}
