//! Independently integrated discrete Fields, direct output terms and implicit products.
use super::*;
use eqiora::differentiation::{
    AcceptedOutputLinearization, adjoint_output_gradient, forward_output_sensitivity,
};
use eqiora::solver::LinearSolveRequest;

#[path = "observables/activity.rs"]
mod activity;

fn observation_oracle(
    method: CommonSpatialPolicy,
    point: &[C; 5],
    direction: &[C; 5],
) -> (Vec<f64>, Vec<f64>) {
    // Integrate the represented Q1 interpolant or TPFA constants on four cells
    // of [0,3]. These are discrete moments, not the continuous quadratic energy.
    let (ia, iff, iaa, iaf, ifff) = if method == CommonSpatialPolicy::Q1 {
        (4.5, 135. / 64., 7., 405. / 128., 1863. / 1024.)
    } else {
        (4.5, 81. / 32., 447. / 64., 243. / 64., 1215. / 512.)
    };
    let a = point[0];
    let da = direction[0];
    let mut total = [C::new(0., 0.); 2];
    let mut dt = total;
    let mut energy = 0.;
    let mut de = 0.;
    for i in 0..2 {
        let p = point[1 + i];
        let dp = direction[1 + i];
        let q = point[3 + i];
        let dq = direction[3 + i];
        let c = q / a.conj();
        let dc = dq / a.conj() - q * da.conj() / (a.conj() * a.conj());
        total[i] = ia * p + iff * c;
        dt[i] = ia * dp + iff * dc;
        let w = (i + 1) as f64;
        energy += w * (iaa * p.norm_sqr() + 2. * iaf * (p.conj() * c).re + ifff * c.norm_sqr());
        de += w
            * (2. * iaa * (p.conj() * dp).re
                + 2. * iaf * (dp.conj() * c + p.conj() * dc).re
                + 2. * ifff * (c.conj() * dc).re);
    }
    let first = (a * total[1]).conj();
    let df = (da * total[1] + a * dt[1]).conj();
    let second = total[0] + 3. * point[2];
    let ds = dt[0] + 3. * direction[2];
    (
        vec![energy + first.re, first.re, first.im, second.re, second.im],
        vec![de + df.re, df.re, df.im, ds.re, ds.im],
    )
}

#[test]
fn spatial_observables_share_lifts_quadrature_and_paired_implicit_products() {
    let (prefix, _) = SOURCE.rsplit_once('}').unwrap();
    let source = format!(
        "{prefix}
        observable totals:array<complex<m>,2> = integral(u, measure(body));
        observable energy:m = integral(math.abs2(u[0])+2*math.abs2(u[1]), measure(body));
        observable combined:array<complex<m>,2> = [math.conj(a*totals[1]), totals[0]+3[m]*p[1]];
        observable loss:m = energy+math.real(combined[0]);
    }}"
    );
    let direction = [
        C::new(0.5, -0.25),
        C::new(2., -3.),
        C::new(-4., 1.),
        C::new(5., 6.),
        C::new(-2., 7.),
    ];
    for method in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
        let (document, plan) = fixture_source(method, &source);
        let inputs = ["a", "p", "q"].map(|name| document.parameter_ref(name).unwrap());
        let selected = inputs.each_ref().map(|input| input.id());
        let field = document.field_ref("definition.u").unwrap().id().erase();
        let loss = document.observable_ref("definition.loss").unwrap().id();
        let combined = document.observable_ref("definition.combined").unwrap().id();
        let rules = plan.observation_quadratures(&[loss, combined], 2).unwrap();
        for a in [C::new(2., 3.), C::new(3., -2.)] {
            let point = [
                a,
                C::new(5., 7.),
                C::new(-1., 4.),
                C::new(11., -2.),
                C::new(3., 8.),
            ];
            let values = point.iter().flat_map(|v| [v.re, v.im]).collect::<Vec<_>>();
            let tangent = direction
                .iter()
                .flat_map(|v| [v.re, v.im])
                .collect::<Vec<_>>();
            for include_field in [true, false] {
                let outputs = if include_field {
                    vec![loss.erase(), field, combined.erase()]
                } else {
                    vec![combined.erase(), loss.erase()]
                };
                let expected = |direction: &[C; 5]| {
                    let (value, tangent) = observation_oracle(method, &point, direction);
                    if include_field {
                        let (field, df) = oracle(method, &point, direction);
                        (
                            [value[..1].to_vec(), field, value[1..].to_vec()].concat(),
                            [tangent[..1].to_vec(), df, tangent[1..].to_vec()].concat(),
                        )
                    } else {
                        (
                            [value[1..].to_vec(), value[..1].to_vec()].concat(),
                            [tangent[1..].to_vec(), tangent[..1].to_vec()].concat(),
                        )
                    }
                };
                let native = plan
                    .as_spatial()
                    .unwrap()
                    .differentiate(
                        &selected,
                        Some(&values),
                        &outputs,
                        &rules,
                        &REFERENCE_LINEAR_SOLVER,
                    )
                    .unwrap();
                let linearization = native.linearization();
                let (primal, derivative) = expected(&direction);
                near(linearization.output_values(), &primal);
                let accepted = AcceptedOutputLinearization::new_with_canonical_state_jacobian(
                    linearization.relation(),
                    linearization,
                    linearization.relation().state_jacobian(),
                    linearization.residual_target(),
                )
                .unwrap();
                let properties = linearization.relation().state_jacobian().properties();
                let forward = forward_output_sensitivity(
                    &accepted,
                    &tangent,
                    properties,
                    LinearSolveRequest::new(
                        &REFERENCE_LINEAR_SOLVER,
                        plan.effective_solver().unwrap(),
                    ),
                )
                .unwrap();
                near(forward.output_tangent(), &derivative);
                let weights = (0..primal.len())
                    .map(|i| (i + 1) as f64 * if i % 2 == 0 { 0.25 } else { -0.5 })
                    .collect::<Vec<_>>();
                let reverse = adjoint_output_gradient(
                    &accepted,
                    &weights,
                    properties,
                    LinearSolveRequest::new(
                        &REFERENCE_LINEAR_SOLVER,
                        plan.effective_solver().unwrap(),
                    ),
                )
                .unwrap();
                let gradient = (0..10)
                    .map(|i| {
                        let mut seed = [C::new(0., 0.); 5];
                        seed[i / 2] = if i % 2 == 0 {
                            C::new(1., 0.)
                        } else {
                            C::new(0., 1.)
                        };
                        expected(&seed)
                            .1
                            .iter()
                            .zip(&weights)
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                near(reverse.gradient(), &gradient);
                let selections = if include_field {
                    vec![
                        document.observable_ref("definition.loss").unwrap().erase(),
                        document.field_ref("definition.u").unwrap().erase(),
                        document
                            .observable_ref("definition.combined")
                            .unwrap()
                            .erase(),
                    ]
                } else {
                    vec![
                        document
                            .observable_ref("definition.combined")
                            .unwrap()
                            .erase(),
                        document.observable_ref("definition.loss").unwrap().erase(),
                    ]
                };
                let program = DifferentiableProgram::compile(
                    plan.clone(),
                    &inputs,
                    &selections,
                    rules.clone(),
                    None,
                    &REFERENCE_LINEAR_SOLVER,
                )
                .unwrap();
                assert_eq!(program.identity().observation_quadratures(), &rules);
                let parts = point.map(|value| (value.re, value.im));
                let input = DifferentiableValues::from_components(
                    program.identity().inputs(),
                    &[&parts[..1], &parts[1..3], &parts[3..]],
                )
                .unwrap();
                let evaluation = program.evaluate(&input).unwrap();
                near(&evaluation.primal().output().fixture_coordinates(), &primal);
                let seed =
                    support::derivative_products::tangent(program.identity(), &tangent).unwrap();
                near(
                    &evaluation
                        .jvp(&seed)
                        .unwrap()
                        .derivative()
                        .fixture_coordinates(),
                    &derivative,
                );
                let dual =
                    support::derivative_products::cotangent(program.identity(), &weights).unwrap();
                near(
                    &evaluation
                        .vjp(&dual)
                        .unwrap()
                        .derivative()
                        .fixture_coordinates(),
                    &gradient,
                );
                let identity = program.identity().to_bytes().unwrap();
                assert_eq!(
                    eqiora::api::DifferentiableProgramIdentity::from_bytes(&identity, &plan, None)
                        .unwrap(),
                    *program.identity()
                );
                let record = evaluation.record().unwrap();
                assert_eq!(
                    RecordedDifferentiableEvaluation::from_bytes(
                        &record.to_bytes().unwrap(),
                        &plan,
                        None
                    )
                    .unwrap(),
                    record
                );
                if method == CommonSpatialPolicy::Q1 && include_field && a == C::new(2., 3.) {
                    assert!(
                        DifferentiableProgram::compile(
                            plan.clone(),
                            &inputs,
                            &selections,
                            Default::default(),
                            None,
                            &REFERENCE_LINEAR_SOLVER
                        )
                        .is_err()
                    );
                    let changed = DifferentiableProgram::compile(
                        plan.clone(),
                        &inputs,
                        &selections,
                        plan.observation_quadratures(&[loss, combined], 1).unwrap(),
                        None,
                        &REFERENCE_LINEAR_SOLVER,
                    )
                    .unwrap();
                    assert_ne!(program.identity(), changed.identity());
                    // Replace the stored rule with another valid canonical Program identity:
                    // the old output must fail mathematical replay, not just JSON canonicality.
                    let mut wire: serde_json::Value =
                        serde_json::from_slice(&record.to_bytes().unwrap()).unwrap();
                    let replacement = changed.evaluate(&input).unwrap().record().unwrap();
                    let replacement: serde_json::Value =
                        serde_json::from_slice(&replacement.to_bytes().unwrap()).unwrap();
                    wire["program"] = replacement["program"].clone();
                    let rejected = RecordedDifferentiableEvaluation::from_bytes(
                        &serde_json::to_vec(&wire).unwrap(),
                        &plan,
                        None,
                    )
                    .unwrap_err();
                    assert!(rejected.iter().any(|diagnostic| {
                        diagnostic.message().contains("accepted point or outputs")
                    }));
                    let mut old: serde_json::Value = serde_json::from_slice(&identity).unwrap();
                    old["schema"] = "eqiora.differentiable-program-identity/v1".into();
                    assert!(
                        eqiora::api::DifferentiableProgramIdentity::from_bytes(
                            &serde_json::to_vec(&old).unwrap(),
                            &plan,
                            None
                        )
                        .is_err()
                    );
                }

                let reopened = plan
                    .as_spatial()
                    .unwrap()
                    .validate_differentiated_values(
                        &selected,
                        Some(&values),
                        &outputs,
                        &rules,
                        linearization.relation().accepted_unknowns(),
                    )
                    .unwrap();
                assert_eq!(&reopened, linearization);
                assert!(
                    plan.as_spatial()
                        .unwrap()
                        .differentiate(
                            &selected,
                            Some(&values),
                            &outputs,
                            &Default::default(),
                            &REFERENCE_LINEAR_SOLVER
                        )
                        .is_err()
                );
            }
        }
    }
}
