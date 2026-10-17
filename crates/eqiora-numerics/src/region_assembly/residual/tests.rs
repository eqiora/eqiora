use super::*;
use eqiora_assembly::{
    AssemblyPacket, AssemblyTarget, DofId, IndexedAssemblyWork, TargetAssemblyMap,
};
use num_complex::Complex64 as C;

#[test]
fn complex_affine_actions_keep_phase_fixed_traces_and_zero_rows() {
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(3).unwrap()]).unwrap();
    let target = plan.target_id(0).unwrap();
    let map = AssemblyMap::new(
        vec![Some(DofId::new(1)), None, Some(DofId::new(0))],
        vec![
            LocalUnknown::Free(DofId::new(0)),
            LocalUnknown::Fixed(C::new(3., -2.)),
        ],
    )
    .unwrap();
    // At x=(1+2i, 3-2i), (2+i)x0+(-1+3i)x1-(4-i) = -1+17i.
    // The skipped row and zero row must not create spurious target equations.
    let local = LocalContribution::new(
        3,
        2,
        vec![
            C::new(2., 1.),
            C::new(-1., 3.),
            C::new(7., 0.),
            C::new(0., 5.),
            C::new(0., 0.),
            C::new(0., 0.),
        ],
        vec![C::new(4., -1.), C::new(0., 0.), C::new(0., 0.)],
    )
    .unwrap();
    let work = IndexedAssemblyWork::new(2, |_| {
        AssemblyPacket::new(
            local.clone(),
            vec![TargetAssemblyMap::new(target, map.clone())],
        )
    });
    let point = [C::new(1., 2.), C::new(-7., 8.), C::new(9., -10.)];
    assert_eq!(
        evaluate_affine_work(&plan, target, &work, &point).unwrap(),
        [C::new(0., 0.), C::new(-2., 34.), C::new(0., 0.)]
    );
    assert!(evaluate_affine_work(&plan, target, &work, &point[..2]).is_err());
    let foreign = AssemblyPlan::new(vec![
        AssemblyTarget::new(3).unwrap(),
        AssemblyTarget::new(3).unwrap(),
    ])
    .unwrap();
    assert!(evaluate_affine_work(&plan, foreign.target_id(1).unwrap(), &work, &point).is_err());
}

#[test]
fn affine_action_rejects_nonfinite_accumulation_and_missing_target() {
    let plan = AssemblyPlan::new(vec![
        AssemblyTarget::new(1).unwrap(),
        AssemblyTarget::new(1).unwrap(),
    ])
    .unwrap();
    let target = plan.target_id(0).unwrap();
    let map = AssemblyMap::new(
        vec![Some(DofId::new(0))],
        vec![LocalUnknown::Free(DofId::new(0))],
    )
    .unwrap();
    let work = IndexedAssemblyWork::new(2, |_| {
        AssemblyPacket::new(
            LocalContribution::new(1, 1, vec![0.], vec![-f64::MAX]).unwrap(),
            vec![TargetAssemblyMap::new(target, map.clone())],
        )
    });
    assert!(
        evaluate_affine_work(&plan, target, &work, &[0.])
            .unwrap_err()
            .message()
            .contains("accumulation is non-finite")
    );
    assert!(evaluate_affine_work(&plan, target, &work, &[f64::NAN]).is_err());
    assert!(
        evaluate_affine_work(&plan, plan.target_id(1).unwrap(), &work, &[0.])
            .unwrap_err()
            .message()
            .contains("omits its residual target")
    );
}
