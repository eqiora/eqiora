use super::*;

#[test]
fn remesh_policy_compares_common_geometry_and_solve_choices_except_mesh_identity() {
    let fixture = Fixture::new();
    let graph = fixture.resolved.portable_graph().unwrap();
    let text = String::from_utf8(graph.to_bytes().unwrap()).unwrap();
    let wire: serde_json::Value = serde_json::from_str(&text).unwrap();
    let mesh = wire["domains"][0]["discretization"]["mesh"]["artifact_sha256"]
        .as_str()
        .unwrap();
    let replacement_mesh = "ab".repeat(32);
    assert_ne!(mesh, replacement_mesh);
    let remeshed_text = text.replace(mesh, &replacement_mesh);
    let remeshed = PortableRealizationGraph::from_bytes(remeshed_text.as_bytes()).unwrap();
    assert_ne!(graph, remeshed);
    assert!(same_spatial_policy_except_imported_mesh(&graph, &remeshed));
    assert!(same_spatial_policy_except_imported_mesh(&remeshed, &graph));

    // Independently valid numerical alternatives must reject at policy equality,
    // rather than being vacuous malformed-graph probes.
    for (before, after) in [
        ("\"minimum_mean_ratio\":0.3", "\"minimum_mean_ratio\":0.4"),
        ("\"points_per_axis\":5", "\"points_per_axis\":6"),
    ] {
        assert!(remeshed_text.contains(before));
        let changed = remeshed_text.replace(before, after);
        let changed = PortableRealizationGraph::from_bytes(changed.as_bytes()).unwrap();
        assert!(!same_spatial_policy_except_imported_mesh(&graph, &changed));
    }
    for arena in ["linear_solves", "nonlinear_solves"] {
        let original = wire[arena][0]["plan"]["maximum_iterations"]
            .as_u64()
            .unwrap();
        let before = format!("\"maximum_iterations\":{original}");
        let after = format!("\"maximum_iterations\":{}", original + 1);
        // Change only this solve arena; harmonic and nonlinear policies may
        // happen to share the same original iteration limit.
        let offset = remeshed_text.find(&format!("\"{arena}\":[")).unwrap();
        let (prefix, selected) = remeshed_text.split_at(offset);
        let changed = format!("{prefix}{}", selected.replacen(&before, &after, 1));
        let changed = PortableRealizationGraph::from_bytes(changed.as_bytes()).unwrap();
        assert!(!same_spatial_policy_except_imported_mesh(&graph, &changed));
    }
}
