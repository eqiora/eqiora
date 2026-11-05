use super::*;
use crate::common_result::evidence::{CommonSolveEvidence, CommonSolveHistory};
use eqiora_solver::REFERENCE_LINEAR_SOLVER;

// Called with independently specified Poisson fixtures; no production mutation hooks.
pub(in crate::numerical_admission) fn check(plan: &CommonSpatialPlan) {
    let count = || STATIC_FINALIZATIONS.with(std::cell::Cell::get);
    let before = count();
    let result = plan.run_result(None, &REFERENCE_LINEAR_SOLVER).unwrap();
    assert_eq!(count() - before, 1, "live execution must finalize once");
    let before = count();
    assert_eq!(
        crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), result.plan()).unwrap(),
        result
    );
    assert_eq!(
        count() - before,
        1,
        "artifact replay must independently finalize"
    );

    for mutate in [
        |out: &mut CommonLinearRunOutput<f64>| out.fields[0].2[1] += 1.,
        |out: &mut CommonLinearRunOutput<f64>| out.fields[0].2[0] = 1.,
        |out: &mut CommonLinearRunOutput<f64>| out.fields.clear(),
        |out: &mut CommonLinearRunOutput<f64>| out.fields[0].0 = eqiora_core::Id::new(),
        |out: &mut CommonLinearRunOutput<f64>| out.fields[0].3 = Space::tetrahedral_edge(),
        |out: &mut CommonLinearRunOutput<f64>| out.fields[0].2[1] = f64::NAN,
        |out: &mut CommonLinearRunOutput<f64>| {
            out.fields[0].2.pop();
        },
    ] {
        let mut live = LiveStaticExecution::execute(plan, None, &REFERENCE_LINEAR_SOLVER).unwrap();
        mutate(live.output.as_mut().unwrap());
        assert!(live.publish().is_err(), "corrupted live output must reject");
    }

    let live = LiveStaticExecution::execute(plan, None, &REFERENCE_LINEAR_SOLVER).unwrap();
    let output = live.output.as_ref().unwrap();
    let values = output
        .fields
        .iter()
        .map(|(id, _, values, _)| (id.erase(), values.clone()))
        .collect();
    let mut solve = CommonSolveEvidence::from_report(&output.solve_report);
    solve.initial_residual_norm = f64::from_bits(solve.initial_residual_norm.to_bits() + 1);
    let execution = CommonSpatialExecution {
        solve: CommonSolveHistory::Linear(Box::new(solve)),
        assembly: crate::common_result::evidence::CommonAssemblyEvidence::from_report(
            &output.assembly_report,
        ),
    };
    assert!(
        live.validate(&values, &execution, None)
            .unwrap_err()
            .message()
            .contains("initial residual")
    );
    // Failure retains the owned equation and output for a subsequent honest validation.
    let mut execution = execution;
    execution.solve = CommonSolveHistory::Linear(Box::new(CommonSolveEvidence::from_report(
        &output.solve_report,
    )));
    live.validate(&values, &execution, None).unwrap();
}

pub(in crate::numerical_admission) fn check_initial(
    plan: &CommonSpatialPlan,
    initial: &CommonState,
    changed: &CommonState,
    foreign: &CommonState,
) {
    let result = plan
        .run_result(Some(initial), &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    assert_eq!(
        crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), result.plan()).unwrap(),
        result
    );
    let mut live =
        LiveStaticExecution::execute(plan, Some(initial), &REFERENCE_LINEAR_SOLVER).unwrap();
    live.initial = Some(changed);
    assert!(
        live.publish()
            .err()
            .unwrap()
            .message()
            .contains("initial residual")
    );
    assert!(LiveStaticExecution::execute(plan, Some(foreign), &REFERENCE_LINEAR_SOLVER).is_err());
}
