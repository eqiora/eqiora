use super::*;
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{
    CommonAlgebraicPlan, CommonInitialField, CommonLinearRequest, CommonSolvePolicy,
};
use eqiora_realization::NonlinearSolvePlan;
use eqiora_solver::{LinearSolver, ReductionPolicy};
use std::num::NonZeroUsize;

fn fixture(
    nonlinear: bool,
) -> (
    DifferentiableProgram,
    ResolvedCommonPlan,
    Option<CommonAlgebraicState>,
) {
    let equation = if nonlinear { "w*w=p" } else { "w=p" };
    let document = ModelDocument::compile("record.eqi", &format!(
        "model Record() {{parameter p:1=4; variable w:1; relation balance {{{equation};}} observable output:1=w+p;}}"
    )).unwrap();
    fixture_for_document(document, nonlinear)
}

fn fixture_for_document(
    document: ModelDocument,
    nonlinear: bool,
) -> (
    DifferentiableProgram,
    ResolvedCommonPlan,
    Option<CommonAlgebraicState>,
) {
    let model = eqiora_artifact::ModelEnvelope::from_program(document.program()).unwrap();
    let solver = SolverPlan::new(
        LinearSolver::SparseLu,
        1e-13,
        1e-15,
        NonZeroUsize::new(16).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast);
    let linear = CommonLinearRequest::exact(solver, FaerLinearSolver.provider()).unwrap();
    let policy = if nonlinear {
        CommonSolvePolicy::Newton {
            nonlinear: NonlinearSolvePlan::new(0., 1e-12, NonZeroUsize::new(32).unwrap(), 16)
                .unwrap(),
            linear,
        }
    } else {
        CommonSolvePolicy::Linear(linear)
    };
    let plan =
        CommonAlgebraicPlan::resolve(&model, policy, None, &[], None, &FaerLinearSolver).unwrap();
    let initial = nonlinear.then(|| {
        plan.initial_state(&[CommonInitialField::finite(
            model.artifact_reference().unwrap().artifact().clone(),
            document.field_ref("w").unwrap().id(),
            eqiora_core::ValueShape::scalar(),
            vec![(2., 0.)],
        )
        .unwrap()])
            .unwrap()
    });
    let plan = ResolvedCommonPlan::Algebraic(Box::new(plan));
    let program = DifferentiableProgram::compile(
        plan.clone(),
        &[document.parameter_ref("p").unwrap()],
        &[document.observable_ref("output").unwrap().erase()],
        Default::default(),
        initial.clone(),
        &FaerLinearSolver,
    )
    .unwrap();
    (program, plan, initial)
}

#[test]
fn evaluation_record_restores_typed_primal_and_provenance_after_live_owner_is_dropped() {
    for nonlinear in [false, true] {
        let (program, plan, initial) = fixture(nonlinear);
        for p in [4., 9.] {
            let point =
                DifferentiableValues::from_coordinates(program.identity.inputs.clone(), vec![p])
                    .unwrap();
            let live = program.evaluate(&point).unwrap();
            let record = live.record().unwrap();
            assert_eq!(record, live.record().unwrap());
            let bytes = record.to_bytes().unwrap();
            drop(live);
            let reopened =
                RecordedDifferentiableEvaluation::from_bytes(&bytes, &plan, initial.as_ref())
                    .unwrap();
            assert_eq!(reopened, record);
            let expected = if nonlinear { p.sqrt() + p } else { 2. * p };
            assert!((reopened.result().output().coordinates()[0] - expected).abs() < 1e-12);
            assert_eq!(reopened.to_bytes().unwrap(), bytes);
            if nonlinear {
                assert_eq!(
                    reopened
                        .result()
                        .evidence()
                        .nonlinear_initial_residual_norm(),
                    Some((4. - p).abs())
                );
                assert_eq!(
                    reopened.result().evidence().nonlinear_iterations().unwrap() == 0,
                    p == 4.
                );
            }
        }
        let record = program
            .evaluate(program.default_point())
            .unwrap()
            .record()
            .unwrap();
        drop(program);
        assert_eq!(
            RecordedDifferentiableEvaluation::from_bytes(
                &record.to_bytes().unwrap(),
                &plan,
                initial.as_ref()
            )
            .unwrap(),
            record
        );
    }
}

#[test]
fn evaluation_record_rejects_changed_candidates_outputs_point_and_provenance() {
    for nonlinear in [false, true] {
        let (program, plan, initial) = fixture(nonlinear);
        let bytes = program
            .evaluate(program.default_point())
            .unwrap()
            .record()
            .unwrap()
            .to_bytes()
            .unwrap();
        for field in ["state", "output", "point"] {
            let mut wire: WireRecord = serde_json::from_slice(&bytes).unwrap();
            match field {
                "state" => wire.state[0] += 1.,
                "output" => wire.output[0] += 1.,
                _ => wire.point[0] += 1.,
            }
            let error = RecordedDifferentiableEvaluation::from_bytes(
                &serde_json::to_vec(&wire).unwrap(),
                &plan,
                initial.as_ref(),
            )
            .unwrap_err();
            assert!(!error.is_empty());
        }
        let mut wire: WireRecord = serde_json::from_slice(&bytes).unwrap();
        if let PrimalEvidence::Nonlinear { iterations, .. } = &mut wire.primal {
            *iterations = 1;
            let error = RecordedDifferentiableEvaluation::from_bytes(
                &serde_json::to_vec(&wire).unwrap(),
                &plan,
                initial.as_ref(),
            )
            .unwrap_err();
            assert!(error[0].message().contains("zero-update"));
        } else {
            let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            value["primal"]["record"]["solver"]["implementation_version"] =
                serde_json::json!("foreign");
            let altered: WireRecord = serde_json::from_value(value).unwrap();
            let errors = RecordedDifferentiableEvaluation::from_bytes(
                &serde_json::to_vec(&altered).unwrap(),
                &plan,
                None,
            )
            .unwrap_err();
            assert!(errors[0].message().contains("differs from the exact Plan"));
        }
        if nonlinear {
            assert!(RecordedDifferentiableEvaluation::from_bytes(&bytes, &plan, None).is_err());
            let mut changed: WireRecord = serde_json::from_slice(&bytes).unwrap();
            if let PrimalEvidence::Nonlinear {
                initial_residual_norm,
                ..
            } = &mut changed.primal
            {
                *initial_residual_norm = 1.;
            }
            let errors = RecordedDifferentiableEvaluation::from_bytes(
                &serde_json::to_vec(&changed).unwrap(),
                &plan,
                initial.as_ref(),
            )
            .unwrap_err();
            assert!(
                errors[0]
                    .message()
                    .contains("differs from its initial State")
            );
        }
        let (_, other, _) = fixture(!nonlinear);
        assert!(
            RecordedDifferentiableEvaluation::from_bytes(&bytes, &other, initial.as_ref()).is_err()
        );
        let mut whitespace = bytes.clone();
        whitespace.push(b' ');
        for bad in [
            whitespace,
            bytes[..bytes.len() - 1].to_vec(),
            vec![b'['; 65],
        ] {
            assert!(
                RecordedDifferentiableEvaluation::from_bytes(&bad, &plan, initial.as_ref())
                    .is_err()
            );
        }
    }
}

#[test]
fn evaluation_record_preserves_signed_zero_input_bits() {
    let (program, plan, _) = fixture(false);
    let point =
        DifferentiableValues::from_coordinates(program.identity.inputs.clone(), vec![-0.]).unwrap();
    let record = program.evaluate(&point).unwrap().record().unwrap();
    let reopened =
        RecordedDifferentiableEvaluation::from_bytes(&record.to_bytes().unwrap(), &plan, None)
            .unwrap();
    assert_eq!(
        reopened.result().evidence().point().coordinates()[0].to_bits(),
        (-0_f64).to_bits()
    );
    assert_eq!(reopened, record);
}

#[test]
fn evaluation_records_rederive_each_parameter_points_numerical_domain() {
    // For z=x+iy, z+p*conj(z)=1+2i gives x=1/(1+p), y=2/(1-p).
    for canonical in [0., 3.] {
        let document = ModelDocument::compile("point-domain.eqi", &format!(
            "model Record(){{parameter p:1={canonical};variable z:complex<1>;relation balance{{z+p*math.conj(z)=math.complex(1,2);}} observable output:complex<1>=z;}}"
        )).unwrap();
        let (program, plan, initial) = fixture_for_document(document, false);
        for p in [0., 3.] {
            let point =
                DifferentiableValues::from_coordinates(program.identity.inputs.clone(), vec![p])
                    .unwrap();
            let live = program.evaluate(&point).unwrap();
            let record = live.record().unwrap();
            let bytes = record.to_bytes().unwrap();
            drop(live);
            let reopened =
                RecordedDifferentiableEvaluation::from_bytes(&bytes, &plan, initial.as_ref())
                    .unwrap();
            assert_eq!(reopened, record);
            assert_eq!(reopened.to_bytes().unwrap(), bytes);
            for (actual, expected) in reopened
                .result()
                .output()
                .coordinates()
                .iter()
                .zip([1. / (1. + p), 2. / (1. - p)])
            {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
    }
}
