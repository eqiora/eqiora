use super::*;
use eqiora_assembly::DofId;
use num_complex::Complex64 as C;

#[test]
fn complex_affine_actions_keep_phase_fixed_traces_and_zero_rows() {
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
    let point = [C::new(1., 2.), C::new(-7., 8.), C::new(9., -10.)];
    let action = evaluate_linear_residual(&local, &map, &point).unwrap();
    let mut residual = vec![C::new(0., 0.); 3];
    scatter_residual(&mut residual, &map, &action).unwrap();
    scatter_residual(&mut residual, &map, &action).unwrap();
    assert_eq!(residual, [C::new(0., 0.), C::new(-2., 34.), C::new(0., 0.)]);
    assert!(evaluate_linear_residual(&local, &map, &[]).is_err());
    assert!(evaluate_linear_residual(&local, &map, &[C::new(f64::NAN, 0.)]).is_err());
    assert!(scatter_residual(&mut [], &map, &action).is_err());
}
