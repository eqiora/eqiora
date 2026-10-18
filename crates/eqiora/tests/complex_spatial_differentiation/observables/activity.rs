//! Prescribed inputs retain their structural activity through observation pullbacks.
use super::*;

#[test]
fn frozen_prescribed_observables_do_not_require_an_unrelated_derivative() {
    let (prefix, _) = SOURCE.rsplit_once('}').unwrap();
    for active in [false, true] {
        let prescribed = if active {
            "math.real(p[0])-5"
        } else {
            "math.real(q[0])-11"
        };
        let source = format!(
            "{prefix}
            variable fixed:1 on body;
            relation prescribed on body {{ fixed={prescribed}; }}
            observable output:m = integral(math.sqrt(fixed)+math.real(u[0]), measure(body));
        }}"
        );
        for method in [
            CommonSpatialPolicy::Q1,
            CommonSpatialPolicy::CellCenteredTpfa,
        ] {
            let (document, plan) = fixture_source(
                method,
                &source,
                &REFERENCE_LINEAR_SOLVER,
                solver_plan(
                    LinearSolver::BiConjugateGradientStabilized,
                    ReductionPolicy::Reproducible,
                ),
            );
            let inputs = ["a", "p"].map(|name| document.parameter_ref(name).unwrap());
            let output = document.observable_ref("definition.output").unwrap();
            let quadratures = plan.observation_quadratures(&[output.id()], 2).unwrap();
            let program = DifferentiableProgram::compile(
                plan.clone(),
                &inputs,
                &[output.erase()],
                quadratures,
                None,
                &REFERENCE_LINEAR_SOLVER,
            );
            if active {
                let errors = program.unwrap_err();
                assert!(
                    errors
                        .iter()
                        .any(|error| error.message().contains("derivative"))
                );
                continue;
            }
            let program = program.unwrap();
            let a = C::new(2., 3.);
            let load = C::new(11., -2.);
            let moment = if method == CommonSpatialPolicy::Q1 {
                135. / 64.
            } else {
                81. / 32.
            };
            let point = DifferentiableValues::from_components(
                program.identity().inputs(),
                &[&[(a.re, a.im)], &[(5., 7.), (-1., 4.)]],
            )
            .unwrap();
            let evaluation = program.evaluate(&point).unwrap();
            near(
                &evaluation.primal().output().fixture_coordinates(),
                &[4.5 * 5. + moment * (load / a.conj()).re],
            );
            let direction = support::derivative_products::tangent(
                program.identity(),
                &[0.5, -0.25, 2., -3., -4., 1.],
            )
            .unwrap();
            let derivative =
                4.5 * 2. - moment * (load * C::new(0.5, 0.25) / (a.conj() * a.conj())).re;
            near(
                &evaluation
                    .jvp(&direction)
                    .unwrap()
                    .derivative()
                    .fixture_coordinates(),
                &[derivative],
            );
            let weight =
                support::derivative_products::cotangent(program.identity(), &[1.]).unwrap();
            let ga = -moment * load / (a.conj() * a.conj());
            near(
                &evaluation
                    .vjp(&weight)
                    .unwrap()
                    .derivative()
                    .fixture_coordinates(),
                &[ga.re, ga.im, 4.5, 0., 0., 0.],
            );
        }
    }
}
