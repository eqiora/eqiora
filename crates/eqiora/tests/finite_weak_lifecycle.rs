//! Authored finite forms use the ordinary Model/Plan/Result and replay path.
use eqiora::api::ModelDocument;
use eqiora_artifact::ModelEnvelope;
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{CommonEigenPlan, CommonEigenRequest, CommonResult, ResolvedCommonPlan};
use std::num::NonZeroUsize;

const SOURCE: &str = r#"
space Spin=orthonormal(up,down);
public component Wave() {
 parameter h:map<complex<1>,Spin,Spin>=linear_map(Spin,Spin,[[2,math.complex(0,-1)],[math.complex(0,1),2]]);
 variable u:coordinates<complex<1>,Spin>;
 variable lambda:1;
 relation states {apply(h,u)=lambda*u;}
 form weak for states {test eta:1 for u; inner(eta,apply(h,u))=inner(eta,lambda*u);}
}
"#;
fn request() -> CommonEigenRequest {
    CommonEigenRequest::dense(NonZeroUsize::new(2).unwrap(), 1e-12, 1e-12).unwrap()
}
fn resolve(source: &str) -> Result<CommonEigenPlan, eqiora_core::Diagnostic> {
    let doc = ModelDocument::compile_selected("finite.eqi", source, "Wave", &[]).unwrap();
    CommonEigenPlan::resolve(
        &ModelEnvelope::from_program(doc.program()).unwrap(),
        request(),
        &FaerLinearSolver,
        doc.authored_formulation_projection().unwrap(),
    )
}
fn replay(bytes: &[u8]) -> Result<ResolvedCommonPlan, eqiora_core::Diagnostic> {
    ResolvedCommonPlan::from_bytes(
        bytes,
        &FaerLinearSolver,
        eqiora_time::TimeBackendCapabilities::new(
            eqiora_time::TimeBackendIdentity::new("eqiora.test.time", "1"),
            &[
                eqiora_core::ScalarDomain::Real,
                eqiora_core::ScalarDomain::Complex,
            ],
            &[eqiora_core::ScalarType::F64],
        ),
    )
}
#[test]
fn real_and_complex_weak_forms_execute_and_replay_with_exact_authored_identity() {
    let physical = SOURCE
        .replace("map<complex<1>", "map<complex<J>")
        .replace(
            "[[2,math.complex(0,-1)],[math.complex(0,1),2]]",
            "[[2[J],math.complex(0[J],-1[J])],[math.complex(0[J],1[J]),2[J]]]",
        )
        .replace("lambda:1", "lambda:J");
    let rectangular = SOURCE.replace("public component Wave", "space Other=orthonormal(first,second,third); public component Wave")
        .replace("parameter h:map<complex<1>,Spin,Spin>=linear_map(Spin,Spin,[[2,math.complex(0,-1)],[math.complex(0,1),2]]);",
        "parameter a:map<complex<1>,Other,Spin>=linear_map(Other,Spin,[[2,math.complex(0,-1),7],[math.complex(0,1),2,9]]); parameter b:map<complex<1>,Spin,Other>=linear_map(Spin,Other,[[1,0],[0,1],[0,0]]);")
        .replace("apply(h,u)", "apply(a,apply(b,u))");
    for source in [
        rectangular,
        physical,
        SOURCE.replace("test eta:1", "test eta:m"),
        SOURCE.to_owned(),
        SOURCE
            .replace("complex<1>", "1")
            .replace("math.complex(0,-1)", "1")
            .replace("math.complex(0,1)", "1"),
    ] {
        let plan = resolve(&source).unwrap();
        let automatic =
            CommonEigenPlan::resolve(plan.model_artifact(), request(), &FaerLinearSolver, None)
                .unwrap();
        assert_eq!(plan.operator(), automatic.operator());
        assert_eq!(plan.metric(), automatic.metric());
        assert_ne!(plan.identity(), automatic.identity());
        let resolved = ResolvedCommonPlan::Eigen(Box::new(plan));
        let form = resolved.formulation().unwrap();
        assert_eq!(
            form.requested(),
            eqiora_numerics::FormulationSelectionMode::Authored
        );
        assert!(form.requested_source_identity().is_some());
        let bytes = resolved.to_bytes().unwrap();
        let restored = replay(&bytes).unwrap();
        assert_eq!(resolved, restored);
        assert_eq!(bytes, restored.to_bytes().unwrap());
        let result = restored
            .as_eigen()
            .unwrap()
            .run_result(&FaerLinearSolver)
            .unwrap();
        // Both characteristic polynomials are (2-lambda)^2 - 1.
        for (i, expected) in [1., 3.].into_iter().enumerate() {
            let (value, _, residual, normalization) = result.eigenpair(i).unwrap();
            assert!((value.component(0).unwrap().0 - expected).abs() < 1e-12);
            assert!(residual < 1e-12 && normalization < 1e-12);
        }
        assert_eq!(
            CommonResult::from_bytes(&result.to_bytes().unwrap(), &restored).unwrap(),
            result
        );
        assert!(
            CommonResult::from_bytes(
                &result.to_bytes().unwrap(),
                &ResolvedCommonPlan::Eigen(Box::new(automatic))
            )
            .is_err()
        );
        let mut wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        wire["authored_formulation_base64"] = serde_json::Value::Null;
        assert!(replay(&serde_json::to_vec(&wire).unwrap()).is_err());
        let mut retired: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        retired["schema"] = serde_json::json!("eqiora.resolved-common-plan/v11");
        assert!(
            replay(&serde_json::to_vec(&retired).unwrap())
                .unwrap_err()
                .message()
                .contains("unknown schema")
        );
    }
}
#[test]
fn correspondence_does_not_admit_changed_coefficients_or_infer_hermitian_structure() {
    for wrong in [
        SOURCE.replace("inner(eta,apply(h,u))", "inner(eta,2*apply(h,u))"),
        SOURCE.replace(
            "inner(eta,apply(h,u))",
            "inner(eta,math.complex(0,1)*apply(h,u))",
        ),
        SOURCE.replace("[math.complex(0,1),2]", "[math.complex(0,-1),2]"),
    ] {
        assert!(resolve(&wrong).is_err(), "{wrong}");
    }
}

#[test]
fn finite_correspondence_rejects_a_unit_only_coefficient_change() {
    let wrong = SOURCE
        .replace("parameter h:", "parameter scale:m=1; parameter h:")
        .replace("inner(eta,apply(h,u))", "inner(eta,scale*apply(h,u))")
        .replace("inner(eta,lambda*u)", "inner(eta,scale*lambda*u)");
    assert!(
        resolve(&wrong).is_err(),
        "unitful weak scaling must not match the source residual"
    );
}

#[test]
fn replay_rejects_foreign_live_map_basis_even_inside_a_zero_term() {
    use eqiora::compiler::{AuthoredFormExpressionV1 as E, AuthoredFormulationProjection};
    let authored = resolve(SOURCE).unwrap();
    let native_source = r#"
space Spin=orthonormal(up,down); space Other=orthonormal(first,second);
model Native() {
 parameter h:map<complex<1>,Spin,Spin>=linear_map(Spin,Spin,[[2,math.complex(0,-1)],[math.complex(0,1),2]]);
 parameter foreign:map<complex<1>,Other,Other>=linear_map(Other,Other,[[1,0],[0,1]]);
 variable u:coordinates<complex<1>,Spin>; variable lambda:1;
 relation states {apply(h,u)=lambda*u;}
}
"#;
    let doc = ModelDocument::compile("native.eqi", native_source).unwrap();
    let model = ModelEnvelope::from_program(doc.program()).unwrap();
    let native = CommonEigenPlan::resolve(&model, request(), &FaerLinearSolver, None).unwrap();
    let text = std::str::from_utf8(authored.authored_formulation_bytes().unwrap())
        .unwrap()
        .replace(
            &authored.mode_field().ulid().to_string(),
            &native.mode_field().ulid().to_string(),
        )
        .replace(
            &authored.eigenvalue_field().ulid().to_string(),
            &native.eigenvalue_field().ulid().to_string(),
        )
        .replace(
            &authored.relation().ulid().to_string(),
            &native.relation().ulid().to_string(),
        );
    let projection = AuthoredFormulationProjection::decode(text.as_bytes()).unwrap();
    let E::Inner { right, .. } = &projection.equations()[0].1 else {
        panic!("weak inner");
    };
    let E::Apply { left, .. } = right.as_ref() else {
        panic!("map application");
    };
    let text = text.replace(
        &serde_json::to_string(left).unwrap(),
        &serde_json::to_string(&E::Parameter {
            ulid: doc.aliases()["h"].ulid().to_string(),
        })
        .unwrap(),
    );
    let valid = AuthoredFormulationProjection::decode(text.as_bytes()).unwrap();
    CommonEigenPlan::resolve(&model, request(), &FaerLinearSolver, Some(&valid)).unwrap();
    let field = native.mode_field().ulid().to_string();
    let invalid_pairing = E::Inner {
        left: Box::new(E::Test {
            field_ulid: field.clone(),
        }),
        right: Box::new(E::Apply {
            left: Box::new(E::Parameter {
                ulid: doc.aliases()["foreign"].ulid().to_string(),
            }),
            right: Box::new(E::Field { ulid: field }),
        }),
    };
    let forged_left = E::Add {
        left: Box::new(valid.equations()[0].1.clone()),
        right: Box::new(E::Mul {
            left: Box::new(E::Number { value: 0. }),
            right: Box::new(invalid_pairing),
        }),
    };
    let forged = text.replacen(
        &serde_json::to_string(&valid.equations()[0].1).unwrap(),
        &serde_json::to_string(&forged_left).unwrap(),
        1,
    );
    let forged = AuthoredFormulationProjection::decode(forged.as_bytes()).unwrap();
    assert!(
        CommonEigenPlan::resolve(&model, request(), &FaerLinearSolver, Some(&forged)).is_err(),
        "numeric cancellation must not erase an invalid nominal endpoint"
    );
}
