//! Complete real/complex channel seeds retain type, words and exact ownership.
use super::*;
use num_complex::Complex64 as C;

pub(super) fn check(model: &ModelEnvelope, plan: &ResolvedCommonPlan, complex: bool) {
    let spatial = plan.as_spatial().unwrap();
    let (field, ty) = spatial.fields().next().unwrap();
    // Independently assigned two-channel affine trace, with an intentionally
    // nonequilibrium interior seed. Imaginary parts are physical scalar parts.
    let mut values = (0..5)
        .flat_map(|index| {
            let x = index as f64 / 4.;
            [
                C::new(1. + 2. * x, if complex { 1. - x } else { 0. }),
                C::new(-2. + x, if complex { 3. + 2. * x } else { 0. }),
            ]
        })
        .collect::<Vec<_>>();
    values[4] += C::new(7., if complex { -5. } else { 0. });
    let assignment = |values: Vec<C>, shape| {
        CommonInitialField::new(
            model.digest().unwrap(),
            field,
            Some(CommonInitialValues::new(shape, values).unwrap()),
            None,
        )
        .unwrap()
    };
    let state = spatial
        .initial_state(None, vec![assignment(values.clone(), ty.shape().clone())])
        .unwrap();
    assert_eq!(state.time_s(), None);
    let words = values
        .iter()
        .flat_map(|v| [v.re, v.im].into_iter().take(if complex { 2 } else { 1 }))
        .collect::<Vec<_>>();
    assert_eq!(state.spatial_values().unwrap(), words);
    let fields = spatial.state_fields(&state).unwrap().collect::<Vec<_>>();
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].0, field);
    assert_eq!(fields[0].1, ty);
    assert_eq!(fields[0].2.len(), 5);
    assert_eq!(fields[0].3, words);
    let bytes = state.to_bytes().unwrap();
    assert_eq!(CommonState::from_bytes(&bytes, plan).unwrap(), state);
    assert_eq!(state.to_bytes().unwrap(), bytes);
    let mut stale = values.clone();
    stale[0] += C::new(0., 1.);
    assert!(
        spatial
            .initial_state(None, vec![assignment(stale, ty.shape().clone())])
            .is_err()
    );
    assert!(
        spatial
            .initial_state(
                None,
                vec![assignment(values[..8].to_vec(), ty.shape().clone())]
            )
            .is_err()
    );
    assert!(
        spatial
            .initial_state(
                None,
                vec![assignment(
                    values.clone(),
                    eqiora_core::ValueShape::scalar()
                )]
            )
            .is_err()
    );
    assert!(
        spatial
            .initial_state(
                Some(0.),
                vec![assignment(values.clone(), ty.shape().clone())]
            )
            .is_err()
    );
    if !complex {
        values[4].im = 1.;
        assert!(
            spatial
                .initial_state(None, vec![assignment(values, ty.shape().clone())])
                .is_err()
        );
    }
}
