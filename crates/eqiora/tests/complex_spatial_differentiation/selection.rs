use super::*;

const COUPLED: &str = r#"
public component ComplexPoisson(
 support body:volume(ambient_dimension=1),
 support left:boundary(parent=body), support right:boundary(parent=body),
 parameter a:complex<1>, parameter p:array<complex<1>,2>,
 parameter q:array<complex<1>,2>
) {
 variable u:array<complex<1>,2> on body in h1;
 variable v:array<complex<1>,2> on body in h1;
 relation first on body {-div(math.conj(a)*grad(u)+grad(v))=q/1[m^2];}
 relation second on body {-div(grad(v))=2*q/1[m^2];}
 relation left_u on left {trace(u)=p;}
 relation right_u on right {trace(u)=2*p;}
 relation left_v on left {trace(v)=3*p;}
 relation right_v on right {trace(v)=6*p;}
}
"#;

#[test]
fn coupled_spatial_output_selection_preserves_requested_fields_and_order() {
    for method in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
        for mixed in [false, true] {
            let source = if mixed {
                COUPLED
                    .replace("on body in h1", "on body in smooth")
                    .replace("-div(grad(v))", "-div(a*grad(v))")
                    .replace("trace(v)=6*p;", "normal(a*grad(v))=a*p/1[m]-3*q/1[m];")
            } else {
                COUPLED.to_owned()
            };
            let (document, plan) = fixture_source(
                method,
                &source,
                &REFERENCE_LINEAR_SOLVER,
                solver_plan(
                    LinearSolver::BiConjugateGradientStabilized,
                    ReductionPolicy::Reproducible,
                ),
            );
            let plan = ResolvedCommonPlan::from_bytes(
                &plan.to_bytes().unwrap(),
                &REFERENCE_LINEAR_SOLVER,
                &[eqiora::time::ImplicitMidpointTimeBackend::CAPABILITIES],
            )
            .unwrap();
            // Establish that the ordinary coupled solve is admitted before testing selection.
            plan.as_spatial()
                .unwrap()
                .run_result(None, &REFERENCE_LINEAR_SOLVER)
                .unwrap();
            let inputs = ["a", "p", "q"].map(|name| {
                let authored = document.parameter_ref(name).unwrap();
                let selected = eqiora::api::ModelEntityRef::from_model_artifact(
                    plan.model_artifact(),
                    authored.id(),
                )
                .unwrap();
                assert_eq!(selected, authored);
                selected
            });
            assert!(
                eqiora::api::ModelEntityRef::from_model_artifact(
                    plan.model_artifact(),
                    eqiora::Id::<eqiora::kinds::Field>::new(),
                )
                .is_err()
            );
            for names in [vec!["u"], vec!["v"], vec!["v", "u"], vec!["u", "v"]] {
                let outputs = names
                    .iter()
                    .map(|name| {
                        let authored = document.field_ref(&format!("definition.{name}")).unwrap();
                        eqiora::api::ModelEntityRef::from_model_artifact(
                            plan.model_artifact(),
                            authored.id(),
                        )
                        .unwrap()
                        .erase()
                    })
                    .collect::<Vec<_>>();
                let program = DifferentiableProgram::compile(
                    plan.clone(),
                    &inputs,
                    &outputs,
                    Default::default(),
                    None,
                    &REFERENCE_LINEAR_SOLVER,
                )
                .unwrap();
                assert_eq!(program.identity().outputs().len(), names.len());
                let direction = [
                    C::new(0.5, -0.25),
                    C::new(2., -3.),
                    C::new(-4., 1.),
                    C::new(5., 6.),
                    C::new(-2., 7.),
                ];
                for a in [C::new(2., 3.), C::new(3., -2.)] {
                    let point = [
                        a,
                        C::new(5., 7.),
                        C::new(-1., 4.),
                        C::new(11., -2.),
                        C::new(3., 8.),
                    ];
                    let parts = point.map(|value| (value.re, value.im));
                    let values = DifferentiableValues::from_components(
                        program.identity().inputs(),
                        &[&parts[..1], &parts[1..3], &parts[3..]],
                    )
                    .unwrap();
                    let evaluation = program.evaluate(&values).unwrap();
                    let flat_direction = direction
                        .iter()
                        .flat_map(|value| [value.re, value.im])
                        .collect::<Vec<_>>();
                    let tangent =
                        support::derivative_products::tangent(program.identity(), &flat_direction)
                            .unwrap();
                    let (primal, expected_jvp) =
                        selected_oracle(method, &point, &direction, &names, mixed);
                    near(&evaluation.primal().output().fixture_coordinates(), &primal);
                    near(
                        &evaluation
                            .jvp(&tangent)
                            .unwrap()
                            .derivative()
                            .fixture_coordinates(),
                        &expected_jvp,
                    );
                    let weights = (0..primal.len())
                        .map(|i| (i as f64 + 1.) * if i % 2 == 0 { 0.75 } else { -0.5 })
                        .collect::<Vec<_>>();
                    let cotangent =
                        support::derivative_products::cotangent(program.identity(), &weights)
                            .unwrap();
                    let expected_vjp = (0..10)
                        .map(|coordinate| {
                            let mut seed = [C::new(0., 0.); 5];
                            seed[coordinate / 2] = if coordinate % 2 == 0 {
                                C::new(1., 0.)
                            } else {
                                C::new(0., 1.)
                            };
                            selected_oracle(method, &point, &seed, &names, mixed)
                                .1
                                .iter()
                                .zip(&weights)
                                .map(|(a, b)| a * b)
                                .sum::<f64>()
                        })
                        .collect::<Vec<_>>();
                    near(
                        &evaluation
                            .vjp(&cotangent)
                            .unwrap_or_else(|error| panic!("{method:?}, {names:?}, a={a}: {error}"))
                            .derivative()
                            .fixture_coordinates(),
                        &expected_vjp,
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
                }
                let selected = inputs.each_ref().map(|input| input.id());
                for invalid in [
                    vec![],
                    vec![outputs[0].id(), outputs[0].id()],
                    vec![selected[0].erase()],
                ] {
                    assert!(
                        plan.as_spatial()
                            .unwrap()
                            .differentiate(
                                None,
                                &selected,
                                None,
                                &invalid,
                                &std::collections::HashMap::new(),
                                &REFERENCE_LINEAR_SOLVER
                            )
                            .is_err()
                    );
                }
            }
        }
    }
}

// Integrate the second balance before the first: v''=-2q, u''=q/conj(a).
// Reuse the independently derived scalar quadratic with its exact Q1/TPFA shape.
fn selected_oracle(
    method: CommonSpatialPolicy,
    point: &[C; 5],
    direction: &[C; 5],
    names: &[&str],
    mixed: bool,
) -> (Vec<f64>, Vec<f64>) {
    let mut values = Vec::new();
    let mut tangents = Vec::new();
    for name in names {
        let mut point = *point;
        let mut direction = *direction;
        match *name {
            "u" => {
                for i in 3..5 {
                    if mixed {
                        // v''=-2q/a, so -conj(a)u''=q-2q/a.
                        let factor = C::new(1., 0.) - 2. / point[0];
                        direction[i] = direction[i] * factor
                            + 2. * point[i] * direction[0] / (point[0] * point[0]);
                        point[i] *= factor;
                    } else {
                        point[i] = -point[i];
                        direction[i] = -direction[i];
                    }
                }
            }
            "v" => {
                if mixed {
                    point[0] = point[0].conj();
                    direction[0] = direction[0].conj();
                } else {
                    point[0] = C::new(1., 0.);
                    direction[0] = C::new(0., 0.);
                }
                for i in 1..5 {
                    let scale = if i < 3 { 3. } else { 2. };
                    point[i] *= scale;
                    direction[i] *= scale;
                }
            }
            _ => unreachable!("exact fixture Field"),
        }
        let (value, tangent) = oracle(method, &point, &direction);
        values.extend(value);
        tangents.extend(tangent);
    }
    (values, tangents)
}
