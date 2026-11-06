use super::*;

#[test]
fn p2_standalone_basis_products_remain_strict_while_raw_graph_is_structural() {
    let p2 = Space::continuous_lagrange(NonZeroU16::new(2).unwrap());
    let defaults = crate::default_plan_v0().unwrap();
    for dimension in [2, 3] {
        let minimum = if dimension == 2 { 3 } else { 4 };
        assert_eq!(
            p2.minimum_simplex_duffy_points(NonZeroUsize::new(dimension).unwrap())
                .unwrap()
                .get(),
            minimum
        );
        for points in [3, 4] {
            let discretization = Discretization::new(
                DiscretizationMethod::ContinuousGalerkin,
                MeshPolicy::ImportedSimplicial {
                    artifact: MeshArtifactReference::from_sha256([7; 32]),
                },
                QuadraturePolicy::SimplexDuffyGaussLegendre {
                    spatial_dimension: NonZeroUsize::new(dimension).unwrap(),
                    points_per_axis: NonZeroUsize::new(points).unwrap(),
                },
            );
            let standalone = crate::RealizationPlan::new(
                p2,
                discretization,
                defaults.solver().clone(),
                defaults.target(),
                defaults.schedule(),
            );
            assert_eq!(standalone.is_ok(), points >= minimum);
            // This graph has no equations and cannot certify form exactness.
            let graph = super::moments::graph(p2, dimension, points, false).unwrap();
            let bytes = graph.to_bytes().unwrap();
            assert_eq!(
                PortableRealizationGraph::from_bytes(&bytes)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
        }
    }
}
