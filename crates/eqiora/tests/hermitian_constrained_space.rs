//! Source-owned coordinate embeddings must preserve the original eigenproblem.
use eqiora::api::ModelDocument;
use eqiora_artifact::ModelEnvelope;
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{CommonEigenPlan, CommonEigenRequest};
use std::num::NonZeroUsize;

const QUOTIENT: &str = r#"
space Full=orthonormal(first,second);
space Reduced=orthonormal(relative);
model Floating() {
 parameter a:map<1,Full,Full>=linear_map(Full,Full,[[1,-1],[-1,1]]);
 parameter b:map<1,Full,Full>=linear_map(Full,Full,[[1,-1],[-1,1]]);
 parameter p:map<1,Reduced,Full>=linear_map(Reduced,Full,[[1],[-1]]);
 variable u:coordinates<1,Full>;
 variable q:coordinates<1,Reduced>;
 variable lambda:1;
 relation pencil {apply(a,u)=lambda*apply(b,u);}
 relation coordinates {u=apply(p,q);}
}
"#;

fn request() -> CommonEigenRequest {
    CommonEigenRequest::dense(NonZeroUsize::new(1).unwrap(), 1e-12, 1e-12).unwrap()
}

#[test]
fn declared_embedding_removes_shared_nullspace_without_regularizing_metric() {
    // A=B=[[1,-1],[-1,1]] share the null vector (1,1). P=(1,-1)
    // gives P^H A P=P^H B P=4, hence lambda=1 and u=+/-(1/2,-1/2).
    // The original metric is singular; only its declared restriction is positive.
    let document = ModelDocument::compile("quotient.eqi", QUOTIENT).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let plan = CommonEigenPlan::resolve(&model, request(), &FaerLinearSolver).unwrap();
    assert_eq!(plan.mode_field().erase(), document.aliases()["u"]);
    let result = plan.run_result(&FaerLinearSolver).unwrap();
    assert_eq!(result.eigen_convergence(), Some("converged"));
    assert_eq!(result.eigenpair_count(), 1);
    let (lambda, mode, residual, normalization) = result.eigenpair(0).unwrap();
    assert!((lambda.component(0).unwrap().0 - 1.).abs() < 1e-12);
    assert_eq!(mode.value_type().shape().component_count(), Some(2));
    let first = mode.component(0).unwrap().0;
    let second = mode.component(1).unwrap().0;
    assert!((first.abs() - 0.5).abs() < 1e-12);
    assert!((first + second).abs() < 1e-12);
    assert!(residual < 1e-12 && normalization < 1e-12);
    let embeddings = plan.coordinate_embeddings().collect::<Vec<_>>();
    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0].0.erase(), document.aliases()["coordinates"]);
    assert_eq!(embeddings[0].1.erase(), document.aliases()["u"]);
    assert_eq!(embeddings[0].2.erase(), document.aliases()["q"]);
    let resolved = eqiora_numerics::ResolvedCommonPlan::Eigen(Box::new(plan));
    let restored = eqiora_numerics::ResolvedCommonPlan::from_bytes(
        &resolved.to_bytes().unwrap(),
        &FaerLinearSolver,
        eqiora_time::TimeBackendIdentity::new("eqiora.test.time", "1"),
    )
    .unwrap();
    assert_eq!(restored, resolved);
    let replay =
        eqiora_numerics::CommonResult::from_bytes(&result.to_bytes().unwrap(), &restored).unwrap();
    assert_eq!(replay, result);
    assert_eq!(
        replay
            .eigenpair(0)
            .unwrap()
            .1
            .value_type()
            .shape()
            .component_count(),
        Some(2)
    );
}

#[test]
fn a_projected_residual_cannot_hide_an_unsatisfied_original_equation() {
    // P=e1 gives a reduced eigenvalue 2, but A e1-2 B e1=(0,1).
    // No reaction variable exists in this Model: manufacturing one changes it.
    let source = QUOTIENT
        .replace(
            "[[1,-1],[-1,1]]);\n parameter b",
            "[[2,1],[1,2]]);\n parameter b",
        )
        .replace(
            "b:map<1,Full,Full>=linear_map(Full,Full,[[1,-1],[-1,1]])",
            "b:map<1,Full,Full>=linear_map(Full,Full,[[1,0],[0,1]])",
        )
        .replace("[[1],[-1]]", "[[1],[0]]");
    let document = ModelDocument::compile("incompatible-subspace.eqi", &source).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let plan = CommonEigenPlan::resolve(&model, request(), &FaerLinearSolver).unwrap();
    let result = plan.run_result(&FaerLinearSolver).unwrap();
    assert_eq!(result.eigenpair_count(), 0);
    assert_eq!(result.eigen_convergence(), Some("not-converged"));
}

#[test]
fn chained_coordinate_equalities_preserve_all_source_roles() {
    let source = QUOTIENT
        .replace(
            " variable lambda:1;",
            " variable r:coordinates<1,Reduced>;\n variable lambda:1;",
        )
        .replace(
            " relation coordinates",
            " relation second { q=2*r; }\n relation coordinates",
        );
    let document = ModelDocument::compile("chain.eqi", &source).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let plan = CommonEigenPlan::resolve(&model, request(), &FaerLinearSolver).unwrap();
    let chain = plan.coordinate_embeddings().collect::<Vec<_>>();
    assert_eq!(chain.len(), 2);
    assert_eq!(chain[0].1.erase(), document.aliases()["u"]);
    assert_eq!(chain[0].2, chain[1].1);
    assert_eq!(chain[1].2.erase(), document.aliases()["r"]);
    let result = plan.run_result(&FaerLinearSolver).unwrap();
    let (_, mode, residual, normalization) = result.eigenpair(0).unwrap();
    assert!((mode.component(0).unwrap().0.abs() - 0.5).abs() < 1e-12);
    assert!(residual < 1e-12 && normalization < 1e-12);
    let fields = result.eigenmode_fields(0).unwrap();
    assert_eq!(fields.len(), 3);
    assert_eq!(fields[1].0.erase(), document.aliases()["q"]);
    assert_eq!(fields[2].0.erase(), document.aliases()["r"]);
    assert_eq!(
        fields[1].1.component(0).unwrap().0,
        2. * fields[2].1.component(0).unwrap().0
    );
    assert_eq!(fields[0].1.component(0), fields[1].1.component(0));
}

#[test]
fn complex_source_embedding_keeps_five_modes_and_the_full_metric_projector() {
    let matrix = format!(
        "[{}]",
        (0..6)
            .map(|i| format!(
                "[{}]",
                (0..6)
                    .map(|j| if i == j {
                        (i * i).to_string()
                    } else {
                        "0".into()
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            ))
            .collect::<Vec<_>>()
            .join(",")
    );
    let embedding = format!(
        "[{}]",
        (0..6)
            .map(|i| format!(
                "[{}]",
                (0..5)
                    .map(|j| if i == j + 1 { "math.complex(0,1)" } else { "0" })
                    .collect::<Vec<_>>()
                    .join(",")
            ))
            .collect::<Vec<_>>()
            .join(",")
    );
    let source = QUOTIENT
        .replace("first,second", "a,b,c,d,e,f")
        .replace("orthonormal(relative)", "orthonormal(a,b,c,d,e)")
        .replace("[[1,-1],[-1,1]]", &matrix)
        .replace("[[1],[-1]]", &embedding)
        .replace("map<1,", "map<complex<1>,")
        .replace("coordinates<1,", "coordinates<complex<1>,");
    for source in [
        source.clone(),
        source
            .replace(
                "apply(a,u)=lambda*apply(b,u)",
                "lambda*apply(b,u)=apply(a,u)",
            )
            .replace("u=apply(p,q)", "apply(p,q)=u"),
    ] {
        let document = ModelDocument::compile("complex-quotient.eqi", &source).unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let request =
            CommonEigenRequest::dense(NonZeroUsize::new(5).unwrap(), 1e-12, 1e-12).unwrap();
        let plan = CommonEigenPlan::resolve(&model, request, &FaerLinearSolver).unwrap();
        let result = plan.run_result(&FaerLinearSolver).unwrap();
        assert_eq!(result.eigen_convergence(), Some("converged"));
        assert_eq!(result.eigenpair_count(), 5);
        for i in 0..5 {
            let (lambda, mode, residual, normalization) = result.eigenpair(i).unwrap();
            assert!((lambda.component(0).unwrap().0 - 1.).abs() < 1e-12);
            assert_eq!(mode.value_type().shape().component_count(), Some(6));
            assert_eq!(mode.component(0), Some((0., 0.)));
            assert!(residual < 1e-12 && normalization < 1e-12);
        }
        let projector = result.eigenprojector(&[0, 1, 2, 3, 4]).unwrap();
        for i in 0..6 {
            for j in 0..6 {
                let (re, im) = projector.component(i * 6 + j).unwrap();
                assert!((re - if i == j && i != 0 { 1. } else { 0. }).abs() < 1e-12);
                assert!(im.abs() < 1e-12);
            }
        }
        let replay =
            eqiora_numerics::CommonResult::from_bytes(&result.to_bytes().unwrap(), result.plan())
                .unwrap();
        assert_eq!(replay, result);
    }
}

#[test]
fn source_embeddings_reject_offsets_nonlinearity_and_unresolved_fields() {
    for source in [
        QUOTIENT.replace("[[1],[-1]]", "[[1],[1]]"),
        QUOTIENT
            .replace(
                " variable u:",
                " parameter offset:coordinates<1,Full>=coordinates(Full,[1,0]);\n variable u:",
            )
            .replace("u=apply(p,q)", "u=apply(p,q)+offset"),
        QUOTIENT.replace("u=apply(p,q)", "u=lambda*apply(p,q)"),
        QUOTIENT.replace(
            " variable lambda:1;",
            " variable extra:coordinates<1,Reduced>; variable lambda:1;",
        ),
    ] {
        let document = ModelDocument::compile("invalid-embedding.eqi", &source).unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        assert!(CommonEigenPlan::resolve(&model, request(), &FaerLinearSolver).is_err());
    }
}

#[test]
fn dimensionful_maps_and_excluded_indefinite_directions_preserve_the_source_problem() {
    let dimensionful = QUOTIENT
        .replace(
            "p:map<1,Reduced,Full>=linear_map(Reduced,Full,[[1],[-1]])",
            "p:map<m,Reduced,Full>=linear_map(Reduced,Full,[[1[m]],[-1[m]]])",
        )
        .replace("q:coordinates<1,Reduced>", "q:coordinates<1/m,Reduced>");
    let indefinite = QUOTIENT
        .replace("[[1,-1],[-1,1]]", "[[-1,0],[0,4]]")
        .replace("[[1],[-1]]", "[[0],[1]]");
    for source in [dimensionful, indefinite] {
        let document = ModelDocument::compile("typed-restriction.eqi", &source).unwrap();
        let model = ModelEnvelope::from_program(document.program()).unwrap();
        let plan = CommonEigenPlan::resolve(&model, request(), &FaerLinearSolver).unwrap();
        let result = plan.run_result(&FaerLinearSolver).unwrap();
        assert_eq!(result.eigen_convergence(), Some("converged"));
        let (lambda, mode, residual, normalization) = result.eigenpair(0).unwrap();
        assert!((lambda.component(0).unwrap().0 - 1.).abs() < 1e-12);
        assert!((mode.component(1).unwrap().0.abs() - 0.5).abs() < 1e-12);
        assert!(residual < 1e-12 && normalization < 1e-12);
    }
}
