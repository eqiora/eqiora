#[test]
#[ignore = "one scientific Run requires a reviewed native producer/build receipt"]
fn native_ordinary_hex_result_observation_and_canonical_replay() {
    use sha2::{Digest, Sha256};
    let hash = |bytes: &[u8]| {
        Sha256::digest(bytes)
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect::<String>()
    };
    let receipt_path =
        std::env::var("EQIORA_HEX_NATIVE_REVIEW_RECEIPT").expect("reviewed receipt required");
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(receipt_path).unwrap()).unwrap();
    assert_eq!(receipt["producer_review"], "accepted");
    let binary = std::env::current_exe().unwrap();
    assert_eq!(
        receipt["binary_sha256"],
        hash(&std::fs::read(&binary).unwrap())
    );
    assert_eq!(receipt["native_producer_sha256"], harness_source_sha256());
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    assert_eq!(
        receipt["source_commit"],
        String::from_utf8(commit.stdout).unwrap().trim()
    );
    let tree = std::process::Command::new("git")
        .args(["rev-parse", "HEAD^{tree}"])
        .output()
        .unwrap();
    assert_eq!(
        receipt["source_tree"],
        String::from_utf8(tree.stdout).unwrap().trim()
    );
    assert_eq!(receipt["model_source_sha256"], hash(SOURCE.as_bytes()));
    let reference = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../verify/numerics/cartesian-hex-elasticity/reference");
    for name in ["reference.py", "protocol.py"] {
        assert_eq!(
            receipt["reference_sha256"][name],
            hash(&std::fs::read(reference.join(name)).unwrap())
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
    assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok(""));
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ] {
        assert_eq!(std::env::var(name).as_deref(), Ok("1"));
    }
    assert!(
        std::env::var("LD_LIBRARY_PATH")
            .unwrap_or_default()
            .is_empty()
    );
    let cpu = std::env::var("EQIORA_HEX_NATIVE_CPU").expect("declared CPU required");
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    assert_eq!(
        status
            .lines()
            .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
            .unwrap()
            .trim(),
        cpu
    );
    let output =
        PathBuf::from(std::env::var_os("EQIORA_HEX_NATIVE_OUTPUT").expect("new output required"));
    assert!(output.is_absolute(), "absolute evidence directory required");
    assert!(!output.exists(), "preserve earlier attempts");
    assert_eq!(
        std::env::var_os("EQIORA_HEX_PREPARATION_DIR").map(PathBuf::from),
        Some(output.clone())
    );
    std::fs::create_dir(&output).unwrap();
    let root = Some(output.clone());
    persist(&root, "launch.json", &serde_json::to_vec_pretty(&serde_json::json!({"pid":std::process::id(),
        "binary":binary,"binary_sha256":receipt["binary_sha256"],"source_commit":receipt["source_commit"],
        "review_receipt":receipt,"cpu":cpu,"ordinary_run_attempted":0,"pde_solves":0,"preparation_manifest":"preparation.json",
        "environment":(["CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","RAYON_NUM_THREADS","LD_LIBRARY_PATH"].into_iter()
            .map(|name| (name, std::env::var(name).ok())).collect::<BTreeMap<_,_>>())})).unwrap());
    let prepared = prepare_source(SOURCE).unwrap();
    assert!(prepared.evidence.is_some());
    let plan_wire: serde_json::Value =
        serde_json::from_slice(&prepared.plan.to_bytes().unwrap()).unwrap();
    persist(&root, "preparation.json", &serde_json::to_vec_pretty(&serde_json::json!({
        "artifact_directory":prepared.evidence,"result_path":prepared.evidence.as_ref().unwrap().join("result.json"),
        "model_digest":prepared.model.digest().unwrap().to_string(),"mesh_digest":prepared.plan.as_spatial().unwrap().mesh_digest(),
        "plan_identity":plan_wire["identity"],"field":selected_field(&prepared).ulid().to_string(),
        "observables":(["energy","left_x","left_y","left_z"].map(|name| selected_observable(&prepared,name).ulid().to_string())),
        "status":"prepared-no-run"})).unwrap());
    persist(
        &root,
        "run-attempt.json",
        b"{\"ordinary_run_attempted\":1,\"declared_pde_solves\":1}\n",
    );
    let result = match prepared
        .plan
        .as_spatial()
        .unwrap()
        .run_result(None, &eqiora::solver::REFERENCE_LINEAR_SOLVER)
    {
        Ok(result) => result,
        Err(error) => {
            persist(
                &root,
                "failure.json",
                &serde_json::to_vec_pretty(
                    &serde_json::json!({"diagnostic":error.to_string(),"ordinary_run_attempted":1}),
                )
                .unwrap(),
            );
            panic!("native ordinary Run rejected: {error}");
        }
    };
    let result_bytes = result.to_bytes().unwrap();
    // Retain the actual accepted Result before replay, observations or comparison.
    persist(&prepared.evidence, "result.json", &result_bytes);
    let replay = eqiora_numerics::CommonResult::from_bytes(&result_bytes, &prepared.plan).unwrap();
    assert_eq!(replay.to_bytes().unwrap(), result_bytes);
    for (name, lane) in [("primary.json", &result), ("replay.json", &replay)] {
        persist(
            &root,
            name,
            &serde_json::to_vec_pretty(&native_export(&prepared, lane)).unwrap(),
        );
    }
    compare_native_exports(&output);
    persist(
        &root,
        "qualification.json",
        b"{\"status\":\"passed\",\"pde_solves\":1,\"profile_calls\":0,\"result_reopens\":1}\n",
    );
}

fn compare_native_exports(output: &std::path::Path) {
    let root = Some(output.to_path_buf());
    let reference = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../verify/numerics/cartesian-hex-elasticity/reference");
    let protocol_python = std::env::var_os("EQIORA_HEX_PROTOCOL_PYTHON")
        .expect("independent protocol interpreter required");
    let checked = std::process::Command::new(protocol_python).current_dir(reference).args(["-c",
        "import pathlib,protocol,sys; q=protocol.r.budgets(); protocol.check(pathlib.Path(sys.argv[1]).read_text(),q); protocol.check(pathlib.Path(sys.argv[2]).read_text(),q)"])
        .arg(output.join("primary.json")).arg(output.join("replay.json")).output().unwrap();
    persist(&root, "protocol-stdout.txt", &checked.stdout);
    persist(&root, "protocol-stderr.txt", &checked.stderr);
    assert!(
        checked.status.success(),
        "independent primary/replay protocol rejected; inspect retained stderr"
    );
}

#[test]
fn actual_hex_corner_mapping_rejects_axis_and_local_node_permutations() {
    let mesh = CartesianMesh::uniform(&[[0., 2.], [0., 3.], [0., 4.]], &[2; 3]).unwrap();
    let corners = mesh
        .entity_vertices(MeshEntity::new(3, 0))
        .unwrap()
        .into_iter()
        .map(|vertex| mesh.vertex_coordinates(vertex).unwrap())
        .collect::<Vec<_>>();
    assert!(physical_corner_mapping_valid(&corners));
    let mut wrong_local_nodes = corners.clone();
    wrong_local_nodes.swap(1, 2);
    assert!(!physical_corner_mapping_valid(&wrong_local_nodes));
    let mut wrong_axes = corners;
    for point in &mut wrong_axes {
        point.swap(0, 1);
    }
    assert!(!physical_corner_mapping_valid(&wrong_axes));
}
