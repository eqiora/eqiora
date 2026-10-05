//! Structural candidates are distinct from numerical DAE admission.
use eqiora::api::ModelDocument;
use eqiora::sem::{Interpreter, ReferenceConfig};

#[test]
fn derivative_incidence_distinguishes_candidates_without_changing_balance() {
    for (equations, rank) in [
        ("derivative(x)=-rate*z; z=x*x;", 2),
        ("derivative(x)=rate*z; x=0;", 1),
    ] {
        let model = ModelDocument::compile("candidate.eqi", &format!(
            "model M() {{ parameter rate: 1/s=1; state x: 1; variable z: 1; relation r {{ {equations} }} }}"
        )).unwrap();
        let report = Interpreter::new()
            .analyze_equations(model.program())
            .unwrap();
        assert_eq!(report.equations().len(), 2);
        assert_eq!(report.balance().rank(), 2);
        assert_eq!(report.declared_rate_partition().rank(), rank);
        assert!(
            report
                .equations()
                .all(|(owner, _, _)| owner == model.aliases()["r"])
        );
        assert_eq!(
            report
                .equations()
                .map(|(_, ordinal, _)| ordinal)
                .collect::<Vec<_>>(),
            [0, 1]
        );
    }
}

#[test]
fn deficient_rate_candidate_does_not_reject_a_regular_coupled_descriptor() {
    let model = ModelDocument::compile("descriptor.eqi", "model M() { parameter rate: 1/s=1; state x: 1; state y: 1; initial { x=1; } relation r { derivative(x)+derivative(y)+2*rate*x=0; x-y=0; } }").unwrap();
    let report = Interpreter::new()
        .analyze_equations(model.program())
        .unwrap();
    assert_eq!(report.balance().rank(), 2);
    assert_eq!(report.declared_rate_partition().rank(), 1);
    // x=y and x'+y'=-2*x imply x=y=1, x'=y'=-1 initially.
    let initial = Interpreter::new()
        .initialize(
            model.program(),
            0.0,
            ReferenceConfig::new(0.0, 0.1).unwrap(),
        )
        .unwrap();
    for alias in ["x", "y"] {
        assert!(
            (initial.derivatives()[&(model.aliases()[alias], std::num::NonZeroU32::MIN)] + 1.0)
                .abs()
                < 1e-9
        );
    }
}

#[test]
fn an_unbalanced_system_can_be_analyzed_before_execution_rejects_it() {
    let model = ModelDocument::compile(
        "unbalanced.eqi",
        "model M() { variable x: 1; variable y: 1; variable z: 1; relation r { x=0; x=0; y+z=0; } }",
    )
    .unwrap();
    let report = Interpreter::new()
        .analyze_equations(model.program())
        .unwrap();
    assert_eq!(report.balance().rank(), 2);
    assert_eq!(report.balance().overdetermined_equations().count(), 2);
    assert_eq!(report.balance().underdetermined_coordinates().count(), 2);
    assert!(
        Interpreter::new()
            .initialize(
                model.program(),
                0.0,
                ReferenceConfig::new(0.0, 0.1).unwrap()
            )
            .is_err()
    );
}
