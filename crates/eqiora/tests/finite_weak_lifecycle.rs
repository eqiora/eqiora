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
    for source in [
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
