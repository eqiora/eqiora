use eqiora_geometry::{
    CanonicalGeometryV1, GeometryGraph, GeometryOperation, GeometryTopologyHandle,
};
use std::collections::BTreeMap;

fn names(source: &GeometryOperation) -> BTreeMap<String, Vec<GeometryTopologyHandle>> {
    let mut names = BTreeMap::from([("body".to_owned(), vec![source.region().into()])]);
    for (name, handle) in [
        "x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper",
    ]
    .into_iter()
    .zip(source.boundaries())
    {
        names.insert(name.to_owned(), vec![handle.into()]);
    }
    names
}

#[test]
fn cuboid_publishes_owned_volume_and_six_axis_ordered_faces() {
    let graph = GeometryGraph::new();
    let source = graph.cuboid([-1., 2.], [3., 5.], [7., 11.]).unwrap();
    assert_eq!(source.region().dimension(), 3);
    assert_eq!(source.boundaries().len(), 6);
    assert!(
        source
            .boundaries()
            .iter()
            .all(|handle| handle.dimension() == 2)
    );
    let geometry = graph.build(&source, &names(&source)).unwrap();
    assert_eq!(
        geometry.cartesian_box_bounds(),
        Some([[-1., 2.], [3., 5.], [7., 11.]].as_slice())
    );
    assert_eq!(geometry.entity_set("body").unwrap().dimension(), 3);
    for (member, name) in [
        "x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper",
    ]
    .into_iter()
    .enumerate()
    {
        let selected = geometry.entity_set(name).unwrap();
        assert_eq!(selected.dimension(), 2);
        assert_eq!(selected.members(), [member]);
    }
    let replay = CanonicalGeometryV1::decode_cartesian_box_v1_canonical(
        geometry.canonical_bytes(),
        Default::default(),
    )
    .unwrap();
    assert_eq!(replay, geometry);
}

#[test]
fn cuboid_rejects_foreign_stale_mixed_and_incomplete_topology() {
    let graph = GeometryGraph::new();
    let source = graph.cuboid([0., 1.], [0., 1.], [0., 1.]).unwrap();
    let stale = graph.cuboid([0., 2.], [0., 1.], [0., 1.]).unwrap();
    let foreign = GeometryGraph::new()
        .cuboid([0., 1.], [0., 1.], [0., 1.])
        .unwrap();
    let rectangle = graph.rectangle([0., 1.], [0., 1.]).unwrap();
    for handle in [
        stale.boundaries()[0].into(),
        foreign.boundaries()[0].into(),
        rectangle.region().into(),
    ] {
        let mut mapping = names(&source);
        mapping.insert("x_lower".to_owned(), vec![handle]);
        assert!(graph.build(&source, &mapping).is_err());
    }
    let mut mixed = names(&source);
    mixed.insert(
        "body".to_owned(),
        vec![source.region().into(), source.boundaries()[0].into()],
    );
    assert!(graph.build(&source, &mixed).is_err());
    let mut incomplete = names(&source);
    incomplete.remove("z_upper");
    assert!(graph.build(&source, &incomplete).is_err());
    let mut duplicate = names(&source);
    duplicate.insert("z_upper".to_owned(), vec![source.boundaries()[4].into()]);
    assert!(graph.build(&source, &duplicate).is_err());
}

#[test]
fn cuboid_uses_canonical_strict_finite_bounds_and_normalizes_signed_zero() {
    let graph = GeometryGraph::new();
    for bad in [
        [1., 1.],
        [2., 1.],
        [f64::NAN, 1.],
        [0., f64::INFINITY],
        [f64::NEG_INFINITY, 1.],
    ] {
        assert!(graph.cuboid(bad, [0., 1.], [0., 1.]).is_err());
        assert!(graph.cuboid([0., 1.], bad, [0., 1.]).is_err());
        assert!(graph.cuboid([0., 1.], [0., 1.], bad).is_err());
    }
    let negative = graph.cuboid([-0., 1.], [0., 1.], [0., 1.]).unwrap();
    let positive = graph.cuboid([0., 1.], [0., 1.], [0., 1.]).unwrap();
    assert_eq!(
        graph
            .build(&negative, &names(&negative))
            .unwrap()
            .canonical_bytes(),
        graph
            .build(&positive, &names(&positive))
            .unwrap()
            .canonical_bytes()
    );
}
