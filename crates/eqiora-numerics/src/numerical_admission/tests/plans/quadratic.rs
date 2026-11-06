//! Ordinary quadratic admission/topology; manufactured solves have separate oracles.
use super::*;

#[test]
fn quadratic_plan_keeps_shared_edges_faces_and_cell_nodes_through_replay() {
    let geometry = cartesian_box_3d();
    let names = [
        "x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper",
    ];
    let model = scalar_box_model(&geometry, POISSON_BOX, "PoissonBox", &names);
    let plan = resolve_scalar_box(
        &model,
        cartesian_box_resources(&geometry, &[2, 3, 2]),
        CommonSpatialPolicy::Q2,
    );
    let replayed = replay_plan(
        ResolvedCommonPlan::Spatial(Box::new(plan)),
        &ResolveOnlyBackend,
    )
    .as_spatial()
    .cloned()
    .unwrap();
    let field = replayed.fields().next().unwrap().0;
    let entities = replayed.field_coefficient_entities(field).unwrap();
    // For nx=2,ny=3,nz=2: V=3*4*3, E=2*4*3+3*3*3+3*4*2,
    // F=2*3*3+2*4*2+3*3*2, C=2*3*2. Shared entities occur once.
    assert_eq!(
        (0..=3)
            .map(|d| entities
                .iter()
                .filter(|entity| entity.dimension() == d)
                .count())
            .collect::<Vec<_>>(),
        [36, 75, 52, 12]
    );
    assert_eq!(entities.len(), 5 * 7 * 5);
    assert!(entities.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(
        replayed.field_space(field).unwrap(),
        Space::continuous_lagrange(std::num::NonZeroU16::new(2).unwrap())
    );
}
