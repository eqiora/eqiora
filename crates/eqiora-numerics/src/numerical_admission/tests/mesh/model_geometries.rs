use super::*;

#[test]
fn authenticated_mesh_retains_canonical_model_geometry_dependencies() {
    let primary = rectangle();
    let a = interval_geometry(0., 1.);
    let b = interval_geometry(-2., 3.);
    let owner = resources(&primary)
        .with_model_geometries(vec![a.clone(), b.clone()])
        .unwrap();
    let reversed = resources(&primary)
        .with_model_geometries(vec![b.clone(), a.clone()])
        .unwrap();
    assert_eq!(owner, reversed);
    let bytes = owner.to_bytes().unwrap();
    let replay = AuthenticatedCommonMesh::from_bytes(&bytes).unwrap();
    assert_eq!(replay, owner);
    assert_eq!(replay.to_bytes().unwrap(), bytes);
    assert_eq!(replay.digest().unwrap(), owner.digest().unwrap());
    assert_ne!(
        owner.digest().unwrap(),
        resources(&primary).digest().unwrap()
    );
    assert_eq!(
        owner.source_digest().unwrap(),
        resources(&primary).source_digest().unwrap()
    );
    assert!(
        resources(&primary)
            .with_model_geometries(vec![a.clone(), a.clone()])
            .is_err()
    );
    assert!(
        resources(&primary)
            .with_model_geometries(vec![primary.clone()])
            .is_err()
    );

    let mut wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(wire["schema"], "eqiora.authenticated-common-mesh/v3");
    wire["schema"] = serde_json::json!("eqiora.authenticated-common-mesh/v2");
    let error =
        AuthenticatedCommonMesh::from_bytes(&serde_json::to_vec(&wire).unwrap()).unwrap_err();
    assert!(error.message().contains("unknown schema"));
    wire["schema"] = serde_json::json!("eqiora.authenticated-common-mesh/v3");
    let duplicate = wire["model_geometries_base64"][0].clone();
    wire["model_geometries_base64"]
        .as_array_mut()
        .unwrap()
        .push(duplicate);
    let error =
        AuthenticatedCommonMesh::from_bytes(&serde_json::to_vec(&wire).unwrap()).unwrap_err();
    assert!(error.message().contains("unique and distinct"), "{error:?}");

    // A valid artifact is not authority to append unrelated Model dependencies.
    let model = model(&primary);
    let error = replay_program(&model, &primary, &[a]).unwrap_err();
    assert!(
        error.message().contains("unreferenced canonical geometry"),
        "{error:?}"
    );
}
