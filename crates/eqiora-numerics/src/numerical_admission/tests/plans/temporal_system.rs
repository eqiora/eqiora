//! Independent semidiscrete residuals through the existing time-system contract.
mod linearization;
mod midpoint;
mod policy;

use super::*;
use crate::numerical_admission::scalar::implicit::SpatialImplicitSystem;
use eqiora_time::ImplicitTimeSystem;
use num_complex::Complex64 as C;

fn resolve(
    complex: bool,
    spatial: CommonSpatialPolicy,
    step: f64,
    balance: &str,
    boundary: Option<&str>,
) -> ResolvedCommonPlan {
    let geometry = cartesian_interval();
    let body = geometry.entity_set("body").unwrap();
    let scalar = if complex { "complex<1>" } else { "1" };
    let boundary = boundary.map_or(String::new(), |value| {
        format!(
            "relation left_value on left {{trace(u)={value};}}
         relation right_value on right {{trace(u)={value};}}"
        )
    });
    let source = format!(
        "public component Heat(
        support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body)) {{
        state u:{scalar} on body in smooth;
        relation balance on body {{{balance}}}
        {boundary}
    }}"
    );
    let model = compile_model(
        "temporal-system.eqi",
        &source,
        &geometry,
        "Heat",
        &[
            ("body", body, None),
            (
                "left",
                geometry.entity_set("left").unwrap(),
                Some(("body", body)),
            ),
            (
                "right",
                geometry.entity_set("right").unwrap(),
                Some(("body", body)),
            ),
        ],
        &[],
    );
    ResolvedCommonPlan::resolve(
        &model,
        cartesian_box_resources(&geometry, &[2]),
        spatial,
        CommonSolvePolicy::Newton {
            linear: exact_reference_linear(
                LinearSolver::BiConjugateGradientStabilized,
                1e-12,
                1e-14,
                NonZeroUsize::new(100).unwrap(),
            ),
            nonlinear: NonlinearSolvePlan::new(0., 1e-12, NonZeroUsize::new(16).unwrap(), 16)
                .unwrap(),
        },
        None,
        Some(CommonBackwardEuler::from_seconds(step).unwrap().into()),
        &REFERENCE_LINEAR_SOLVER,
        None,
    )
    .unwrap()
}

fn system(plan: &CommonSpatialPlan) -> Box<dyn ImplicitTimeSystem + '_> {
    match plan.admission.recognized_model() {
        RecognizedNativeModel::Spatial(equations) => {
            Box::new(SpatialImplicitSystem::new(plan, equations, 0.).unwrap())
        }
        RecognizedNativeModel::ComplexSpatial(equations) => {
            Box::new(SpatialImplicitSystem::new(plan, equations, 0.).unwrap())
        }
        _ => unreachable!(),
    }
}

fn coordinates(values: &[C], complex: bool) -> Vec<f64> {
    values
        .iter()
        .flat_map(|value| {
            [value.re, value.im]
                .into_iter()
                .take(if complex { 2 } else { 1 })
        })
        .collect()
}

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-11, "{actual} != {expected}");
    }
}

#[test]
fn spatial_time_residual_rebinds_analytic_traces_without_a_discrete_step() {
    for complex in [false, true] {
        for conjugate in [false, true].into_iter().take(if complex { 2 } else { 1 }) {
            for spatial in [
                CommonSpatialPolicy::Q1,
                CommonSpatialPolicy::CellCenteredTpfa,
            ] {
                for step in [0.25, 2.] {
                    let datum = if complex { "math.complex(1,2)" } else { "1" };
                    let stored = if conjugate { "math.conj(u)" } else { "u" };
                    let balance = format!(
                        "1[s/m^2]*derivative({stored})-div(grad(u))=2*{datum}*time()/1[s*m^2];"
                    );
                    let resolved = resolve(
                        complex,
                        spatial,
                        step,
                        &balance,
                        Some(&format!("{datum}*(time()/1[s])^2")),
                    );
                    let system = system(resolved.as_spatial().unwrap());
                    let q1 = matches!(spatial, CommonSpatialPolicy::Q1);
                    let count = if q1 { 1 } else { 2 };
                    let value = |re, im| C::new(re, if complex { im } else { 0. });
                    let y = [value(3., 4.), value(-1., 2.)];
                    let v = [value(-2., 1.), value(1., -3.)];
                    let dy = [value(-1., 2.), value(2., 0.5)];
                    let dv = [value(3., -2.), value(-2., 4.)];
                    let source = value(1., 2.);
                    let rate = |v: C| if conjugate { v.conj() } else { v };
                    let y_values = coordinates(&y[..count], complex);
                    let v_values = coordinates(&v[..count], complex);
                    assert_eq!(system.dimension(), y_values.len());
                    for t in [0., 0.5, 2.] {
                        // Q1: M_ii=1/3, sum M_if=1/6, K_ii=4, sum K_if=-4.
                        // TPFA: M=I/2, K=[[6,-2],[-2,6]], each boundary weight -4.
                        // The continuous fixed rate is g_dot=2t*source; no secant
                        // or arbitrary Plan step enters either original residual.
                        let expected = (0..count)
                            .map(|i| {
                                if q1 {
                                    rate(v[i]) / 3. + 4. * y[i] - 4. * t * t * source
                                        + rate(source) * t / 3.
                                        - t * source
                                } else {
                                    rate(v[i]) / 2. + 6. * y[i]
                                        - 2. * y[1 - i]
                                        - (4. * t * t + t) * source
                                }
                            })
                            .collect::<Vec<_>>();
                        let tangent = (0..count)
                            .map(|i| {
                                if q1 {
                                    4. * dy[i] + rate(dv[i]) / 3.
                                } else {
                                    6. * dy[i] - 2. * dy[1 - i] + rate(dv[i]) / 2.
                                }
                            })
                            .collect::<Vec<_>>();
                        let mut output = vec![f64::NAN; system.dimension()];
                        system
                            .residual(t, &y_values, &v_values, &mut output)
                            .unwrap();
                        close(&output, &coordinates(&expected, complex));
                        system
                            .residual_jvp(
                                t,
                                &y_values,
                                &v_values,
                                &coordinates(&dy[..count], complex),
                                &coordinates(&dv[..count], complex),
                                &mut output,
                            )
                            .unwrap();
                        close(&output, &coordinates(&tangent, complex));
                    }
                    let mut output = vec![123.; system.dimension()];
                    assert!(
                        system
                            .residual(f64::NAN, &y_values, &v_values, &mut output)
                            .is_err()
                    );
                    assert_eq!(output, vec![123.; system.dimension()]);
                    assert!(
                        system
                            .residual_jvp(
                                0.,
                                &y_values,
                                &v_values,
                                &vec![f64::NAN; system.dimension()],
                                &v_values,
                                &mut output
                            )
                            .is_err()
                    );
                    assert_eq!(output, vec![123.; system.dimension()]);
                }
            }
        }
    }
}

#[test]
fn spatial_time_actions_keep_state_dependent_storage_and_complex_coordinates() {
    for complex in [false, true] {
        for spatial in [
            CommonSpatialPolicy::Q1,
            CommonSpatialPolicy::CellCenteredTpfa,
        ] {
            let resolved = resolve(complex, spatial, 0.25, "1[s]*derivative(u*u)+u*u=0;", None);
            let system = system(resolved.as_spatial().unwrap());
            let weights = if matches!(spatial, CommonSpatialPolicy::Q1) {
                vec![0.25, 0.5, 0.25]
            } else {
                vec![0.5, 0.5]
            };
            let u = C::new(2., if complex { 1. } else { 0. });
            let v = C::new(-1., if complex { 2. } else { 0. });
            let du = C::new(1., if complex { -2. } else { 0. });
            let dv = C::new(2., if complex { 0.5 } else { 0. });
            let repeated = |value| coordinates(&vec![value; weights.len()], complex);
            let mut output = vec![f64::NAN; system.dimension()];
            system
                .residual(0., &repeated(u), &repeated(v), &mut output)
                .unwrap();
            let expected = weights
                .iter()
                .map(|w| w * (2. * u * v + u * u))
                .collect::<Vec<_>>();
            close(&output, &coordinates(&expected, complex));
            system
                .residual_jvp(
                    0.,
                    &repeated(u),
                    &repeated(v),
                    &repeated(du),
                    &repeated(dv),
                    &mut output,
                )
                .unwrap();
            // F_u=2v+2u and F_v=2u; constant fields integrate against the
            // exact Q1 hat or cell-constant test function measures above.
            let expected = weights
                .iter()
                .map(|w| w * (2. * (v + u) * du + 2. * u * dv))
                .collect::<Vec<_>>();
            close(&output, &coordinates(&expected, complex));
        }
    }
}

#[test]
fn spatial_time_actions_do_not_differentiate_frozen_volume_time_inputs() {
    let resolved = resolve(
        false,
        CommonSpatialPolicy::Q1,
        0.25,
        "1[s]*derivative(u*u)=math.sqrt(time()/1[s]);",
        None,
    );
    let system = system(resolved.as_spatial().unwrap());
    let mut output = vec![f64::NAN; system.dimension()];
    system
        .residual(0., &[1.; 3], &[0.; 3], &mut output)
        .unwrap();
    close(&output, &[0.; 3]);
    system
        .residual_jvp(0., &[1.; 3], &[0.; 3], &[0.; 3], &[1.; 3], &mut output)
        .unwrap();
    close(&output, &[0.5, 1., 0.5]);
}
