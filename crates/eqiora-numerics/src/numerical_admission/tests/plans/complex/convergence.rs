//! Analytic quadratic fields distinguish coefficient and physical-field convergence.
use super::*;

#[test]
fn quadratic_real_and_complex_fields_refine_with_their_selected_space() {
    let geometry = cartesian_interval();
    for (complex, channels) in [(false, false), (true, false), (false, true), (true, true)] {
        let model = scalar_box_model(
            &geometry,
            &source(complex, channels),
            "PoissonInterval",
            &["left", "right"],
        );
        let polynomials = if complex {
            vec![
                [C::new(1., 1.), C::new(3., 1.), C::new(-1., -2.)],
                [C::new(-2., 3.), C::new(5., -2.), C::new(3., 4.)],
            ]
        } else {
            vec![
                [C::new(1., 0.), C::new(3., 0.), C::new(-1., 0.)],
                [C::new(-2., 0.), C::new(5., 0.), C::new(3., 0.)],
            ]
        };
        let count = if channels { 2 } else { 1 };
        for method in [
            CommonSpatialPolicy::Q1,
            CommonSpatialPolicy::CellCenteredTpfa,
        ] {
            let nodal = method == CommonSpatialPolicy::Q1;
            let mut errors = vec![Vec::new(); count];
            for cells in [8, 16, 32] {
                let h = 1. / cells as f64;
                let plan = ResolvedCommonPlan::resolve(
                    &model,
                    cartesian_box_resources(&geometry, &[cells]),
                    method,
                    CommonSolvePolicy::Linear(exact_reference_linear(
                        LinearSolver::BiConjugateGradientStabilized,
                        1e-12,
                        1e-14,
                        NonZeroUsize::new(512).unwrap(),
                    )),
                    None,
                    None,
                    &REFERENCE_LINEAR_SOLVER,
                    None,
                    None,
                )
                .unwrap();
                let result = plan
                    .as_spatial()
                    .unwrap()
                    .run_result(None, &REFERENCE_LINEAR_SOLVER)
                    .unwrap();
                let (association, values, shape) = result.field_block(0, 0).unwrap();
                assert_eq!(association, if nodal { "vertex" } else { "cell" });
                let mut expected_shape = vec![cells + usize::from(nodal)];
                if channels {
                    expected_shape.push(count);
                }
                assert_eq!(shape, expected_shape);
                let width = if complex { 2 } else { 1 };
                assert_eq!(values.len(), (cells + usize::from(nodal)) * count * width);
                for component in 0..count {
                    let values = values
                        .chunks_exact(count * width)
                        .map(|value| {
                            C::new(
                                value[component * width],
                                if complex {
                                    value[component * width + 1]
                                } else {
                                    0.
                                },
                            )
                        })
                        .collect::<Vec<_>>();
                    let [constant, slope, curvature] = polynomials[component];
                    let exact = |x: f64| constant + slope * x + curvature * x * x;
                    let coefficient_error = values
                        .iter()
                        .enumerate()
                        .map(|(index, value)| {
                            let x = (index as f64 + if nodal { 0. } else { 0.5 }) * h;
                            (*value - exact(x)).norm()
                        })
                        .fold(0., f64::max);
                    if nodal {
                        assert!((values[0] - exact(0.)).norm() < 1e-12);
                        assert!((values[cells] - exact(1.)).norm() < 1e-12);
                    }
                    let mut squared_error = 0.;
                    for cell in 0..cells {
                        let midpoint = (cell as f64 + 0.5) * h;
                        let (constant, linear) = if nodal {
                            (
                                (values[cell] + values[cell + 1]) / 2.,
                                (values[cell + 1] - values[cell]) / h,
                            )
                        } else {
                            (values[cell], C::new(0., 0.))
                        };
                        let d0 = constant - exact(midpoint);
                        let d1 = linear - (slope + 2. * curvature * midpoint);
                        let d2 = -curvature;
                        // Exact integral of |d0+d1*t+d2*t²|², t in [-h/2,h/2].
                        // Odd moments vanish; even moments are h, h³/12, h⁵/80.
                        // No implementation quadrature or reconstruction is used.
                        squared_error += h * d0.norm_sqr()
                            + h.powi(3) / 12. * (d1.norm_sqr() + 2. * (d0.conj() * d2).re)
                            + h.powi(5) / 80. * d2.norm_sqr();
                    }
                    assert!(squared_error > 0. && coefficient_error > 0.);
                    errors[component].push((coefficient_error, squared_error.sqrt()));
                }
                assert_eq!(
                    crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
                    result
                );
            }
            for (component, errors) in errors.iter().enumerate() {
                for pair in errors.windows(2) {
                    let coefficient_order = (pair[0].0 / pair[1].0).log2();
                    let field_order = (pair[0].1 / pair[1].1).log2();
                    // Both coefficient vectors are second order. The actual P0
                    // Field is first order; Q1 interpolation is second order.
                    assert!(
                        (1.8..2.2).contains(&coefficient_order),
                        "{method:?}, complex={complex}, component={component}: coefficient order {coefficient_order}"
                    );
                    let expected = if nodal { 2. } else { 1. };
                    assert!(
                        (field_order - expected).abs() < 0.2,
                        "{method:?}, complex={complex}, component={component}: Field order {field_order}"
                    );
                }
            }
        }
    }
}

fn source(complex: bool, channels: bool) -> String {
    // u0=1+3x-x², u1=-2+5x+3x² (real).
    // u0=(1+i)+(3+i)x-(1+2i)x², u1=(-2+3i)+(5-2i)x+(3+4i)x² (complex).
    // Independently expand -a*u''+q*u with a=2, q=3 or a=2+3i, q=3-2i.
    let x = "coordinate(0)/1[m]";
    let (scalar, diffusion, reaction, forcing, left, right) = if complex {
        (
            "complex<1>",
            "math.complex(2,3)",
            "math.complex(3,-2)",
            [
                format!("math.complex(-3+11*({x})-7*({x})^2,15-3*({x})-4*({x})^2)"),
                format!("math.complex(12+11*({x})+17*({x})^2,-21-16*({x})+6*({x})^2)"),
            ],
            ["math.complex(1,1)", "math.complex(-2,3)"],
            ["3", "math.complex(6,5)"],
        )
    } else {
        (
            "1",
            "2",
            "3",
            [
                format!("7+9*({x})-3*({x})^2"),
                format!("-18+15*({x})+9*({x})^2"),
            ],
            ["1", "-2"],
            ["3", "6"],
        )
    };
    let shape = |values: [&str; 2]| {
        if channels {
            format!("[{},{}]", values[0], values[1])
        } else {
            values[0].to_owned()
        }
    };
    let ty = if channels {
        format!("array<{scalar},2>")
    } else {
        scalar.to_owned()
    };
    POISSON_INTERVAL
        .replace("variable potential: 1", &format!("variable potential: {ty}"))
        .replace(" on body in h1;", " on body in smooth;")
        .replace("-div(grad(potential)) - source_scale = 0",
            &format!("-div({diffusion}*grad(potential))+{reaction}*source_scale*potential-({})*source_scale=0", shape([&forcing[0], &forcing[1]])))
        .replace("relation left_value on left { trace(potential) = 0; }", &format!("relation left_value on left {{ trace(potential)={}; }}", shape(left)))
        .replace("relation right_value on right { trace(potential) = 0; }", &format!("relation right_value on right {{ trace(potential)={}; }}", shape(right)))
}
