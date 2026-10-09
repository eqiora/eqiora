use super::*;
use crate::{
    CommonAlgebraicPlan, CommonLinearRequest, CommonResult, CommonSolvePolicy, ResolvedCommonPlan,
};
use eqiora_solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, ReductionPolicy, SolverPlan};
use std::num::NonZeroUsize;

mod conventions;
mod transient;

pub(crate) fn check_finite_profiles() {
    rc_response_uses_the_original_equations_and_shared_complex_solver();
    rlc_response_and_real_conjugation_share_the_same_reduction();
    nonlinear_time_varying_and_dc_inputs_reject_before_solve();
    transient::rc_harmonic_response_matches_settled_time_solution_without_claiming_initial_equivalence();
    conventions::cyclic_frequency_conversion_and_peak_power_are_explicit();
}

fn source() -> &'static str {
    include_str!("../../../../../docs/language/harmonic-rc.md")
        .split_once("```eqiora\n")
        .unwrap()
        .1
        .split_once("```")
        .unwrap()
        .0
}

fn compile(source: &str) -> (KernelProgram, AuthoredFormulationProjection) {
    let model = eqiora_compiler::CompiledModel::compile_selected("harmonic.eqi", source, "RC", &[])
        .unwrap();
    let form = model
        .authored_formulations()
        .next()
        .unwrap()
        .projection()
        .clone();
    let (transaction, model, _) = model.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        form,
    )
}

fn resolve(source: &str) -> Result<CommonAlgebraicPlan, Diagnostic> {
    let (program, form) = compile(source);
    let model = ModelEnvelope::from_program(&program).unwrap();
    let solver = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-14,
        1e-16,
        NonZeroUsize::new(32).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Reproducible);
    let request = CommonLinearRequest::exact(solver, REFERENCE_LINEAR_SOLVER.provider()).unwrap();
    CommonAlgebraicPlan::resolve(
        &model,
        CommonSolvePolicy::Linear(request),
        None,
        &[],
        Some(&form),
        &REFERENCE_LINEAR_SOLVER,
    )
}

fn rc_response_uses_the_original_equations_and_shared_complex_solver() {
    let plan = resolve(source()).unwrap();
    let original = plan.harmonic_original_model().unwrap();
    assert_ne!(
        original.model().unwrap().ulid().to_string(),
        plan.model_id()
    );
    assert_eq!(
        original
            .to_program()
            .unwrap()
            .nodes()
            .filter(|node| matches!(node, KernelNode::Relation(relation) if relation.is_initial()))
            .count(),
        1
    );
    assert!(
        plan.kernel()
            .nodes()
            .all(|node| !matches!(node, KernelNode::Relation(relation) if relation.is_initial()))
    );
    assert_eq!(plan.harmonic_angular_frequency(), Some(1000.0));
    let initial = plan.initial_state(&[]).unwrap();
    let result = plan.run_result(&initial, &REFERENCE_LINEAR_SOLVER).unwrap();
    for (time, voltage) in [
        (0.0, 0.5),
        (std::f64::consts::FRAC_PI_2 / 1000.0, 0.5),
        (std::f64::consts::PI / 1000.0, -0.5),
    ] {
        let reconstructed = plan
            .reconstruct_harmonic_fields(
                &result,
                eqiora_core::DynQuantity::new(
                    time,
                    eqiora_core::DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap(),
                ),
            )
            .unwrap();
        assert_eq!(reconstructed.len(), 2);
        let original = plan
            .harmonic_amplitudes()
            .find(|(name, _, _)| *name == "voltage_hat")
            .unwrap()
            .1;
        let actual = reconstructed
            .iter()
            .find(|(id, _)| *id == original)
            .unwrap()
            .1
            .real_scalar_value()
            .unwrap()
            .value();
        assert!((actual - voltage).abs() < 1e-10);
    }
    let fields = plan
        .field_values(result.finite_values().unwrap())
        .unwrap()
        .into_iter()
        .map(|(id, value)| (id.erase(), value))
        .collect::<BTreeMap<_, _>>();
    // Independent circuit algebra: V=1/(1-i)=0.5+0.5i;
    // I=-i*omega*C*V=0.0005-0.0005i. In particular the original V(0)=0 is not this response.
    for (name, original, amplitude) in plan.harmonic_amplitudes() {
        assert_ne!(original, amplitude);
        let expected = match name {
            "voltage_hat" => (0.5, 0.5),
            "current_hat" => (0.0005, -0.0005),
            _ => panic!("unexpected amplitude"),
        };
        let actual = fields[&amplitude.erase()].component(0).unwrap();
        assert!(
            (actual.0 - expected.0).abs() < 1e-10 && (actual.1 - expected.1).abs() < 1e-10,
            "{name}: {actual:?}"
        );
    }
    let resolved = ResolvedCommonPlan::Algebraic(Box::new(plan));
    assert_eq!(
        resolved.formulation().unwrap().effective(),
        crate::FormulationKind::HarmonicResponse
    );
    assert_eq!(
        resolved.formulation().unwrap().requested(),
        crate::FormulationSelectionMode::Authored
    );
    let replayed = ResolvedCommonPlan::from_bytes(
        &resolved.to_bytes().unwrap(),
        &REFERENCE_LINEAR_SOLVER,
        eqiora_time::TimeBackendCapabilities::new(
            eqiora_time::TimeBackendIdentity::new("harmonic.test", "1"),
            &[ScalarDomain::Real, ScalarDomain::Complex],
            &[eqiora_core::ScalarType::F64],
        ),
    )
    .unwrap();
    assert_eq!(resolved, replayed);
    assert_eq!(
        CommonResult::from_bytes(&result.to_bytes().unwrap(), &replayed).unwrap(),
        result
    );
}

fn rlc_response_and_real_conjugation_share_the_same_reduction() {
    let source = source()
        .replace(
            "parameter resistance:",
            "parameter inductance: H = 1[H], parameter resistance:",
        )
        .replace("variable current: A;", "state current: A;")
        .replace(
            "voltage = initial_voltage;",
            "voltage = initial_voltage; current = 0[A];",
        )
        .replace(
            "resistance * current;",
            "resistance * current + inductance * derivative(current);",
        );
    for source in [
        source.clone(),
        source.replace("derivative(voltage)", "math.conj(derivative(voltage))"),
    ] {
        let plan = resolve(&source).unwrap();
        let result = plan
            .run_result(&plan.initial_state(&[]).unwrap(), &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        let values = plan.field_values(result.finite_values().unwrap()).unwrap();
        assert_eq!(plan.harmonic_amplitudes().count(), 2);
        // omega^2*L*C=1 and omega*R*C=1, so V=1/(-i)=i, I=0.001 A.
        // The unscaled SI matrix inverse has infinity norm 1+1000*sqrt(2).
        // A 1e-14 residual target leaves margin inside the 1e-10 value tolerance.
        for (name, _, amplitude) in plan.harmonic_amplitudes() {
            let expected = if name == "voltage_hat" {
                (0., 1.)
            } else {
                (0.001, 0.)
            };
            let actual = values
                .iter()
                .find(|(id, _)| *id == amplitude)
                .unwrap()
                .1
                .component(0)
                .unwrap();
            assert!(
                (actual.0 - expected.0).abs() < 1e-10 && (actual.1 - expected.1).abs() < 1e-10,
                "{actual:?}"
            );
        }
    }
}

fn nonlinear_time_varying_and_dc_inputs_reject_before_solve() {
    resolve(source()).unwrap();
    for (old, new, diagnostic) in [
        (
            "resistance * current",
            "resistance * current * voltage / 1[V]",
            "nonlinear harmonic product",
        ),
        (
            "resistance * current",
            "(1 + time() / 1[s]) * resistance * current",
            "time-dependent",
        ),
        ("source - voltage", "source + 1[V] - voltage", "mixed DC"),
        (
            "angular_frequency = omega",
            "angular_frequency = 0[1/s]",
            "positive finite angular frequency",
        ),
        (
            "angular_frequency = omega",
            "angular_frequency = -1000[1/s]",
            "positive finite angular frequency",
        ),
    ] {
        assert!(source().contains(old));
        let error = resolve(&source().replace(old, new)).unwrap_err();
        assert!(error.message().contains(diagnostic), "{error:?}");
    }
}

#[test]
fn harmonic_reduction_preserves_the_original_spatial_trace_hypothesis() {
    for (syntax, expected) in [
        ("h1", SpatialRegularity::H1),
        ("smooth", SpatialRegularity::Smooth),
    ] {
        let source = format!(
            "model RC() {{ domain body=box(0,1); domain wall=boundary(body,axis=0,side=lower); \
             state u:1 on body in {syntax}; \
             relation evolution on body {{ derivative(u)=0[1/s]; }} \
             relation boundary_value on wall {{ trace(u)=0; }} \
             form response for evolution,boundary_value {{ \
             harmonic(angular_frequency=1[1/s],convention=negative_exponential,normalization=peak); \
             amplitude u_hat:complex<1> on body for u; }} }}"
        );
        let (program, form) = compile(&source);
        let reduction = HarmonicReduction::derive(&program, &form, None).unwrap();
        let reduced = reduction.reduced.to_program().unwrap();
        let fields = reduced
            .nodes()
            .filter_map(|node| match node {
                KernelNode::Field(field) => Some(field.spatial_regularity()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(fields, [expected]);
    }
}
