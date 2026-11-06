//! Ordinary imported public observations; scientific expectations live in the reference owner.
use super::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, path::Path, process::Command};

const IDENTITIES: [(&str, &str, &str, &str); 6] = [
    ("left_heat", "left", "outward_conductive_heat", "W/m"),
    ("right_heat", "right", "outward_conductive_heat", "W/m"),
    ("bottom_heat", "bottom", "outward_conductive_heat", "W/m"),
    ("top_heat", "top", "outward_conductive_heat", "W/m"),
    ("right_temperature", "right", "temperature_trace", "K*m"),
    (
        "right_convective_heat",
        "right",
        "outward_convective_heat",
        "W/m",
    ),
];

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn write(path: &Path, bytes: &[u8]) -> String {
    std::fs::write(path, bytes).unwrap();
    digest(bytes)
}
fn hex(value: f64) -> String {
    assert!(value.is_finite());
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0x0.0p+0"
        } else {
            "0x0.0p+0"
        }
        .to_owned();
    }
    let bits = value.to_bits();
    let sign = if bits >> 63 == 0 { "" } else { "-" };
    let exponent = (bits >> 52) & 0x7ff;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent == 0 {
        format!("{sign}0x0.{fraction:013x}p-1022")
    } else {
        format!("{sign}0x1.{fraction:013x}p{:+}", exponent as i32 - 1023)
    }
}

#[test]
fn canonical_binary64_hex_covers_zero_subnormal_and_exponent_boundaries() {
    // Canonical strings follow Python float.hex(), the reference input contract.
    for (value, expected) in [
        (0.0, "0x0.0p+0"),
        (-0.0, "-0x0.0p+0"),
        (1.0, "0x1.0000000000000p+0"),
        (-1.5, "-0x1.8000000000000p+0"),
        (0.5, "0x1.0000000000000p-1"),
        (f64::MIN_POSITIVE, "0x1.0000000000000p-1022"),
        (f64::from_bits(1), "0x0.0000000000001p-1022"),
        (f64::from_bits((1_u64 << 52) - 1), "0x0.fffffffffffffp-1022"),
        (f64::MAX, "0x1.fffffffffffffp+1023"),
    ] {
        assert_eq!(hex(value), expected);
    }
}

fn coefficients(result: &CommonResult, plan: &ResolvedCommonPlan, field: &str) -> (Value, Value) {
    assert_eq!(result.field_count(), 1);
    let (actual, value_type, _) = result.field(0).unwrap();
    assert_eq!(actual, field);
    assert_eq!(
        value_type,
        &eqiora_core::ValueType::scalar(
            eqiora_core::ScalarDomain::Real,
            eqiora::DimExponents::from_integers([0, 0, 0, 0, 1, 0, 0]).unwrap()
        )
        .unwrap()
    );
    let (association, values, _) = result.field_block(0, 0).unwrap();
    assert_eq!(association, "vertex");
    assert_eq!(values.len(), 9);
    let authenticated = plan.authenticated_mesh().unwrap();
    let mesh = authenticated.cartesian_mesh().unwrap().mesh();
    let mut ordered = values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let entity = eqiora::meshing::MeshEntity::new(0, index);
            (mesh.vertex_coordinates(entity).unwrap(), *value, index)
        })
        .collect::<Vec<_>>();
    ordered.sort_by(|a, b| a.0[0].total_cmp(&b.0[0]).then(a.0[1].total_cmp(&b.0[1])));
    let records = ordered
        .iter()
        .map(|(point, value, _)| {
            json!({
        "coordinates_m":point.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "value_binary64_hex":hex(*value)})
        })
        .collect::<Vec<_>>();
    let entities = ordered.iter().map(|(point,_,index)| json!({"entity_dimension":0,"entity_index":index,"coordinates_m":point})).collect::<Vec<_>>();
    (json!(records), json!(entities))
}

#[test]
fn imported_convection_public_observations_and_artifact_replay_match_reference() {
    let root = std::path::PathBuf::from(
        std::env::var_os("EQIORA_ROBIN_PACKAGE_EVIDENCE").unwrap_or_else(|| {
            let home = std::env::var_os("HOME").expect("home-backed evidence directory");
            Path::new(&home)
                .join(".cache/eqiora/packaged-robin")
                .join(format!(
                    "{}-{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                ))
                .into_os_string()
        }),
    );
    assert!(root.is_absolute() && !root.starts_with("/tmp"));
    std::fs::create_dir_all(&root).unwrap();
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let oracle = repository.join("verify/numerics/cartesian-robin-q1/reference/robin.py");
    assert_eq!(
        digest(&std::fs::read(&oracle).unwrap()),
        "200cc1ff5de8327d438506a75e4af59e03b188e730d4fb2b2b653ea06caa507d"
    );
    let head = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&repository)
        .output()
        .unwrap();
    assert!(head.status.success());
    let executable = digest(&std::fs::read(std::env::current_exe().unwrap()).unwrap());
    for (case, (length, height, k, h, fixed, ambient)) in [
        "insulated",
        "robin",
        "changed_h",
        "changed_geometry_material_cooling",
    ]
    .into_iter()
    .zip(CASES)
    {
        let directory = root.join(case);
        std::fs::create_dir(&directory).unwrap();
        let authored = source(k, h, fixed, ambient);
        let source_hash = write(&directory.join("model.eqi"), authored.as_bytes());
        write(
            &directory.join("case-attempt.json"),
            &serde_json::to_vec_pretty(&json!({
                "schema":1,"case":case,"status":"attempt-started",
                "source_head":String::from_utf8(head.stdout.clone()).unwrap().trim(),
                "authored_source_sha256":source_hash,"native_executable_sha256":executable,
                "producer_source_sha256":digest(include_bytes!("observations.rs")),
                "oracle_sha256":digest(&std::fs::read(&oracle).unwrap())
            }))
            .unwrap(),
        );
        let (document, owner) =
            fixture_rectangle(&authored, [f64::from(length), f64::from(height)]).unwrap();
        let SourceCompilation::Packaged(packaged) = &document else {
            panic!("ordinary imported package closure required")
        };
        write(
            &directory.join("package-compilation.json"),
            &packaged.compilation().canonical_json().unwrap(),
        );
        for (name, relative) in [
            ("scalar.eqi", "src/scalar.eqi"),
            ("scalar-package.json", "package.json"),
            ("scalar-README.md", "README.md"),
        ] {
            let bytes = std::fs::read(
                repository
                    .join("crates/eqiora-api/packages/Eqiora.Scalar")
                    .join(relative),
            )
            .unwrap();
            write(&directory.join(name), &bytes);
        }
        let model = ModelEnvelope::from_program(document.model().program()).unwrap();
        let model_bytes = model.canonical_json().unwrap();
        let model_hash = write(&directory.join("model.json"), &model_bytes);
        let restored_model = ModelEnvelope::from_json(
            &model_bytes,
            eqiora::artifact::ModelDecoderLimits::default(),
        )
        .unwrap();
        assert_eq!(restored_model.canonical_json().unwrap(), model_bytes);
        assert_eq!(restored_model, model);
        let plan = resolve_fixture(&document, owner, None).unwrap();
        assert_eq!(plan.model_artifact(), &model);
        let plan_bytes = plan.to_bytes().unwrap();
        let plan_hash = write(&directory.join("plan.eqplan"), &plan_bytes);
        let mesh_bytes = plan.authenticated_mesh().unwrap().to_bytes().unwrap();
        let mesh_hash = write(&directory.join("mesh.eqmesh"), &mesh_bytes);
        let reopened =
            ResolvedCommonPlan::from_bytes(&plan_bytes, &REFERENCE_LINEAR_SOLVER, &[]).unwrap();
        assert_eq!(reopened.to_bytes().unwrap(), plan_bytes);
        assert_eq!(reopened.model_artifact(), &model);
        let result = reopened
            .as_spatial()
            .unwrap()
            .run_result(None, &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        let result_bytes = result.to_bytes().unwrap();
        let result_hash = write(&directory.join("result.eqresult"), &result_bytes);
        let replay = CommonResult::from_bytes(&result_bytes, &reopened).unwrap();
        assert_eq!(replay.to_bytes().unwrap(), result_bytes);
        let aliases = document.model().aliases();
        let field = aliases["definition.temperature"].ulid().to_string();
        let (records, entities) = coefficients(&result, &reopened, &field);
        assert_eq!(coefficients(&replay, &reopened, &field).0, records);
        let mut observations = Vec::new();
        let mut bindings = Vec::new();
        for (name, boundary, quantity, unit) in IDENTITIES {
            let observable = aliases[&format!("definition.{name}")].downcast().unwrap();
            let domain = aliases[boundary].downcast().unwrap();
            let rules = HashMap::from([(
                domain,
                eqiora::meshing::QuadratureRule::gauss_legendre(2).unwrap(),
            )]);
            assert_eq!(
                reopened.observation_quadratures(&[observable], 2).unwrap(),
                rules
            );
            let primary = result
                .observe(&restored_model, observable, &rules, None)
                .unwrap();
            let secondary = replay
                .observe(&restored_model, observable, &rules, None)
                .unwrap();
            let expected_dim = if unit == "K*m" {
                [0, 1, 0, 0, 1, 0, 0]
            } else {
                [1, 1, -3, 0, 0, 0, 0]
            };
            for (sample, accepted) in [(&primary, &result), (&secondary, &replay)] {
                assert_eq!(sample.observable(), observable);
                assert_eq!(sample.result_identity(), accepted.identity());
                assert_eq!(sample.quadratures(), &rules);
                assert_eq!(
                    sample.value().real_scalar_value().unwrap().dim(),
                    eqiora::DimExponents::from_integers(expected_dim).unwrap()
                );
            }
            observations.push(json!({"name":name,"boundary":boundary,"quantity":quantity,"unit":unit,
                "quadrature":{"family":"gauss-legendre","points_per_segment":2},
                "value_binary64_hex":hex(primary.value().real_scalar_value().unwrap().value()),
                "replay_value_binary64_hex":hex(secondary.value().real_scalar_value().unwrap().value())}));
            bindings.push(json!({"name":name,"observable_id":observable.ulid().to_string(),"boundary_domain_id":domain.ulid().to_string(),
                "result_identity":primary.result_identity(),"replay_result_identity":secondary.result_identity(),"unit":unit,"quadrature":format!("{:?}",primary.quadratures())}));
        }
        let order = records
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["coordinates_m"].clone())
            .collect::<Vec<_>>();
        let input = json!({"schema":1,"kind":"public-observations","case":case,"order":order,
            "units":{"coordinates":"m","coefficients":"K"},"coefficients":records,"observations":observations});
        let input_hash = write(
            &directory.join("observations.json"),
            &serde_json::to_vec_pretty(&input).unwrap(),
        );
        let package_hash =
            digest(&std::fs::read(directory.join("package-compilation.json")).unwrap());
        let receipt = json!({"schema":1,"source_head":String::from_utf8(head.stdout.clone()).unwrap().trim(),
            "producer_source_sha256":digest(include_bytes!("observations.rs")),"native_executable_sha256":executable,
            "authored_source_sha256":source_hash,"model_sha256":model_hash,"plan_sha256":plan_hash,"result_sha256":result_hash,
            "mesh_sha256":mesh_hash,"package_compilation_sha256":package_hash,"input_sha256":input_hash,"oracle_sha256":digest(&std::fs::read(&oracle).unwrap()),
            "field_id":field,"field_alias":"definition.temperature","coefficient_entities":entities,
            "plan_identity":reopened.identity(),"result_identity":result.identity(),"observables":bindings,
            "canonical_model_reopen":true,"canonical_plan_reopen":true,"canonical_result_reopen":true,"replay_coefficients_equal":true,
            "ordinary_public_runs":1,"scope":"public coefficients and observations; no full A/b or private weak reactions"});
        write(
            &directory.join("producer-receipt.json"),
            &serde_json::to_vec_pretty(&receipt).unwrap(),
        );
        let report_path = directory.join("comparison.json");
        let status = Command::new("python3")
            .arg(&oracle)
            .arg("compare-observations")
            .arg(directory.join("observations.json"))
            .arg("--output")
            .arg(&report_path)
            .status()
            .unwrap();
        let report: Value = serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
        assert!(
            status.success() && report["accepted"] == true,
            "{case}: {report}"
        );
        assert_eq!(report["input_sha256"], input_hash);
        assert_eq!(
            report["counts"],
            json!({"coefficients":9,"free":6,"observations":6,"observation_evaluations":12})
        );
        assert!(
            report["checks"]
                .as_object()
                .unwrap()
                .values()
                .all(|v| v == true)
        );
        assert!(
            report["observations"]
                .as_array()
                .unwrap()
                .iter()
                .all(|r| r["checks"].as_object().unwrap().values().all(|v| v == true))
        );
    }
}
