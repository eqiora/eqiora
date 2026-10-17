//! Analytic quadratic fields distinguish coefficient and physical-field convergence.
use super::*;

#[test]
fn quadratic_real_and_complex_fields_refine_with_their_selected_space() {
    let geometry = cartesian_interval();
    for complex in [false, true] {
        let balance = if complex {
            "-div(math.complex(2, 3)*grad(potential)) + math.complex(3, -2)*source_scale*potential - math.complex(-3 + 11*coordinate(0)/1[m] - 7*(coordinate(0)/1[m])^2, 15 - 3*coordinate(0)/1[m] - 4*(coordinate(0)/1[m])^2)*source_scale = 0"
        } else {
            "-div(2*grad(potential)) + 3*source_scale*potential - (7 + 9*coordinate(0)/1[m] - 3*(coordinate(0)/1[m])^2)*source_scale = 0"
        };
        let mut source = POISSON_INTERVAL
            .replace(" on body in h1;", " on body in smooth;")
            .replace("-div(grad(potential)) - source_scale = 0", balance)
            .replace(
                "relation left_value on left { trace(potential) = 0; }",
                if complex {
                    "relation left_value on left { trace(potential) = math.complex(1, 1); }"
                } else {
                    "relation left_value on left { trace(potential) = 1; }"
                },
            )
            .replace(
                "relation right_value on right { trace(potential) = 0; }",
                "relation right_value on right { trace(potential) = 3; }",
            );
        if complex {
            source = source.replace("variable potential: 1", "variable potential: complex<1>");
        }
        let model = scalar_box_model(&geometry, &source, "PoissonInterval", &["left", "right"]);
        let curvature = C::new(1., if complex { 2. } else { 0. });
        let slope = C::new(2., if complex { -1. } else { 0. });
        let exact = |x: f64| {
            C::new(1., if complex { 1. } else { 0. }) + slope * x + curvature * (x * (1. - x))
        };
        // The complex polynomial is u=1+3x-x²+i(1+x-2x²).
        // For a=2+3i, q=3-2i, -a*u''+q*u=(-3+11x-7x²)+i(15-3x-4x²).
        for method in [
            CommonSpatialPolicy::Q1,
            CommonSpatialPolicy::CellCenteredTpfa,
        ] {
            let nodal = method == CommonSpatialPolicy::Q1;
            let mut errors = Vec::new();
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
                )
                .unwrap();
                let result = plan
                    .as_spatial()
                    .unwrap()
                    .run_result(None, &REFERENCE_LINEAR_SOLVER)
                    .unwrap();
                let (association, values, shape) = result.field_block(0, 0).unwrap();
                assert_eq!(association, if nodal { "vertex" } else { "cell" });
                assert_eq!(shape, &[cells + usize::from(nodal)]);
                let width = if complex { 2 } else { 1 };
                let values = values
                    .chunks_exact(width)
                    .map(|value| C::new(value[0], if complex { value[1] } else { 0. }))
                    .collect::<Vec<_>>();
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
                    let d1 = linear - (slope + curvature * (1. - 2. * midpoint));
                    let d2 = curvature;
                    // Exact integral of |d0+d1*t+d2*t²|² for t in [-h/2,h/2].
                    // Odd moments vanish; the even moments are h, h³/12, h⁵/80.
                    // This uses no implementation quadrature or reconstruction.
                    squared_error += h * d0.norm_sqr()
                        + h.powi(3) / 12. * (d1.norm_sqr() + 2. * (d0.conj() * d2).re)
                        + h.powi(5) / 80. * d2.norm_sqr();
                }
                assert!(squared_error > 0. && coefficient_error > 0.);
                errors.push((coefficient_error, squared_error.sqrt()));
                assert_eq!(
                    crate::CommonResult::from_bytes(&result.to_bytes().unwrap(), &plan).unwrap(),
                    result
                );
            }
            for pair in errors.windows(2) {
                let coefficient_order = (pair[0].0 / pair[1].0).log2();
                let field_order = (pair[0].1 / pair[1].1).log2();
                // Both coefficient vectors are second order. The actual P0
                // Field is first order; Q1 interpolation is second order.
                assert!(
                    (1.8..2.2).contains(&coefficient_order),
                    "{method:?}, complex={complex}: coefficient order {coefficient_order}"
                );
                let expected = if nodal { 2. } else { 1. };
                assert!(
                    (field_order - expected).abs() < 0.2,
                    "{method:?}, complex={complex}: Field order {field_order}"
                );
            }
        }
    }
}
