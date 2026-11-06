#[test]
#[ignore = "read-only recovery requires review of its distinct harness and original artifact bindings"]
fn recover_original_native_result_without_running() {
    use sha2::{Digest, Sha256};
    let hash = |bytes: &[u8]| {
        Sha256::digest(bytes)
            .iter()
            .map(|value| format!("{value:02x}"))
            .collect::<String>()
    };
    let receipt_path = std::env::var_os("EQIORA_HEX_NATIVE_RECOVERY_RECEIPT")
        .expect("reviewed recovery receipt required");
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(receipt_path).unwrap()).unwrap();
    assert_eq!(receipt["recovery_review"], "accepted");
    let binary = std::env::current_exe().unwrap();
    assert_eq!(
        receipt["recovery_binary_sha256"],
        hash(&std::fs::read(&binary).unwrap())
    );
    assert_eq!(receipt["recovery_harness_sha256"], harness_source_sha256());
    for (argument, key) in [
        ("HEAD", "recovery_source_commit"),
        ("HEAD^{tree}", "recovery_source_tree"),
    ] {
        let actual = std::process::Command::new("git")
            .args(["rev-parse", argument])
            .output()
            .unwrap();
        assert_eq!(
            receipt[key],
            String::from_utf8(actual.stdout).unwrap().trim()
        );
    }
    assert!(
        std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .output()
            .unwrap()
            .stdout
            .is_empty()
    );
    assert_eq!(
        receipt["original_source_commit"],
        "6826671d537882b3daded212866e76250b9c036f"
    );
    assert_eq!(
        receipt["original_runtime_binary_sha256"],
        "a4fc848c5540fbbab3637f1f2f86216015b25340100c9fed6198a8e3ad35a986"
    );
    assert_eq!(
        receipt["original_files_sha256"]["result.json"],
        "d31edd4e04c651d5a8fc4586d17d1c8e09aa67e2a3a7c82f2e68431373d70517"
    );
    let original_binary = PathBuf::from(receipt["original_runtime_binary_copy"].as_str().unwrap());
    assert_eq!(
        receipt["original_runtime_binary_sha256"],
        hash(&std::fs::read(original_binary).unwrap())
    );
    let original = PathBuf::from(receipt["original_attempt_directory"].as_str().unwrap());
    let original_artifacts =
        PathBuf::from(receipt["original_artifact_directory"].as_str().unwrap());
    for (name, directory) in [
        ("launch.json", &original),
        ("preparation.json", &original),
        ("run-attempt.json", &original),
        ("model.eqi", &original_artifacts),
        ("model.json", &original_artifacts),
        ("mesh.json", &original_artifacts),
        ("plan.json", &original_artifacts),
        ("result.json", &original_artifacts),
    ] {
        assert_eq!(
            receipt["original_files_sha256"][name],
            hash(&std::fs::read(directory.join(name)).unwrap())
        );
    }
    let original_launch: serde_json::Value =
        serde_json::from_slice(&std::fs::read(original.join("launch.json")).unwrap()).unwrap();
    let original_preparation: serde_json::Value =
        serde_json::from_slice(&std::fs::read(original.join("preparation.json")).unwrap()).unwrap();
    assert_eq!(
        original_launch["source_commit"],
        receipt["original_source_commit"]
    );
    assert_eq!(
        original_launch["binary_sha256"],
        receipt["original_runtime_binary_sha256"]
    );
    assert_eq!(
        original_launch["review_receipt"]["producer_review"],
        "accepted"
    );
    assert_eq!(original_launch["pid"], 1431892);
    assert_eq!(
        original_preparation["artifact_directory"].as_str().unwrap(),
        original_artifacts.to_str().unwrap()
    );
    assert_eq!(
        std::fs::read(original_artifacts.join("model.eqi")).unwrap(),
        SOURCE.as_bytes()
    );
    let reference = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../verify/numerics/cartesian-hex-elasticity/reference");
    for name in ["reference.py", "protocol.py"] {
        assert_eq!(
            receipt["reference_sha256"][name],
            hash(&std::fs::read(reference.join(name)).unwrap())
        );
    }
    let output = PathBuf::from(
        std::env::var_os("EQIORA_HEX_NATIVE_RECOVERY_OUTPUT")
            .expect("new recovery directory required"),
    );
    assert!(
        output.is_absolute() && !output.exists(),
        "preserve original and earlier recovery attempts"
    );
    assert_eq!(
        std::env::var_os("EQIORA_HEX_PREPARATION_DIR").map(PathBuf::from),
        Some(output.clone())
    );
    std::fs::create_dir(&output).unwrap();
    let root = Some(output.clone());
    persist(
        &root,
        "recovery-launch.json",
        &serde_json::to_vec_pretty(&serde_json::json!({
        "recovery_pid":std::process::id(),"recovery_binary":binary,"review_receipt":receipt,
        "original_attempt_status":"export-failed-after-one-completed-run","new_pde_solves":0}))
        .unwrap(),
    );
    // Compile/resolve only: bind fresh source selections to the exact archived artifacts.
    let mut prepared = prepare_source(SOURCE).unwrap();
    for name in ["model.eqi", "model.json", "mesh.json", "plan.json"] {
        assert_eq!(
            std::fs::read(prepared.evidence.as_ref().unwrap().join(name)).unwrap(),
            std::fs::read(original_artifacts.join(name)).unwrap(),
            "fresh no-solve binding differs for {name}"
        );
    }
    let plan_bytes = std::fs::read(original_artifacts.join("plan.json")).unwrap();
    prepared.plan = eqiora_numerics::ResolvedCommonPlan::from_bytes(
        &plan_bytes,
        &eqiora::solver::REFERENCE_LINEAR_SOLVER,
        &[],
    )
    .unwrap();
    assert_eq!(prepared.plan.to_bytes().unwrap(), plan_bytes);
    let plan_wire: serde_json::Value = serde_json::from_slice(&plan_bytes).unwrap();
    assert_eq!(plan_wire["identity"], original_preparation["plan_identity"]);
    let model_bytes = std::fs::read(original_artifacts.join("model.json")).unwrap();
    prepared.model =
        eqiora::artifact::ModelEnvelope::from_json(&model_bytes, Default::default()).unwrap();
    assert_eq!(prepared.model.canonical_json().unwrap(), model_bytes);
    assert_eq!(
        prepared.model.digest().unwrap().to_string(),
        original_preparation["model_digest"].as_str().unwrap()
    );
    assert_eq!(
        selected_field(&prepared).ulid().to_string(),
        original_preparation["field"].as_str().unwrap()
    );
    for (index, name) in ["energy", "left_x", "left_y", "left_z"]
        .into_iter()
        .enumerate()
    {
        assert_eq!(
            selected_observable(&prepared, name).ulid().to_string(),
            original_preparation["observables"][index].as_str().unwrap()
        );
    }
    let result_bytes = std::fs::read(original_artifacts.join("result.json")).unwrap();
    let result = eqiora_numerics::CommonResult::from_bytes(&result_bytes, &prepared.plan).unwrap();
    assert_eq!(result.to_bytes().unwrap(), result_bytes);
    persist(&root, "original-result.json", &result_bytes);
    let replay =
        eqiora_numerics::CommonResult::from_bytes(&result.to_bytes().unwrap(), &prepared.plan)
            .unwrap();
    assert_eq!(replay.to_bytes().unwrap(), result_bytes);
    for (name, lane) in [("primary.json", &result), ("replay.json", &replay)] {
        persist(
            &root,
            name,
            &serde_json::to_vec_pretty(&native_export(&prepared, lane)).unwrap(),
        );
    }
    compare_native_exports(&output);
    persist(&root,"recovery-qualification.json",&serde_json::to_vec_pretty(&serde_json::json!({
        "status":"read-only-recovery-passed","new_pde_solves":0,"original_completed_pde_solves":1,
        "original_attempt_status":"export-failed-after-result-save","original_source_commit":receipt["original_source_commit"],
        "original_runtime_binary_sha256":receipt["original_runtime_binary_sha256"],
        "original_result_sha256":receipt["original_files_sha256"]["result.json"],
        "recovery_harness_commit":receipt["recovery_source_commit"],"result_identity":result.identity()})).unwrap());
}
