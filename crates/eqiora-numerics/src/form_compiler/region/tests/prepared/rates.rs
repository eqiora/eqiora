//! Independent local residual and chain-rule checks before global State acceptance.
use super::*;
use crate::form_compiler::region::coordinates::Real;
use crate::form_compiler::region::field_point::{FieldPullback, FieldValues};
use eqiora_meshing::QuadratureRule;
use num_complex::Complex64 as C;

#[test]
fn nonlinear_rate_action_uses_physical_history_and_normalized_candidate() {
    for balance in [
        "relation r on body {derivative(u*u)-div(1[m^2/s]*grad(u))+u*u/1[s]=0;}",
        "law r on body {storage u*u; flux -1[m^2/s]*grad(u); source -u*u/1[s];}",
    ] {
        check_nonlinear_rate_action(balance);
    }
}

fn check_nonlinear_rate_action(balance: &str) {
    let source = format!(
        "model Storage() {{
        domain body=box(0,1,0,1); state u:1 on body;
        {balance}
    }}"
    );
    let form = derive(&source).unwrap();
    let (mut fields, mut rows) = form.si_bindings(|_| Ok(p1())).unwrap();
    fields[0].scale = DynQuantity::new(3., fields[0].scale.dim());
    for scale in rows.values_mut() {
        *scale = DynQuantity::new(7., scale.dim());
    }
    let reference = ReferenceCell::simplex(2).unwrap();
    assert!(form.bind(reference, &fields, &rows, None).is_err());
    for step in [0.25, 0.5] {
        let bound = form
            .bind(
                reference,
                &fields,
                &rows,
                Some(&RegionTimeBinding {
                    step: DynQuantity::new(step, dim([0, 0, 1, 0, 0, 0, 0])),
                    states: Vec::new(),
                }),
            )
            .unwrap();
        let rule = simplex_duffy_gauss_legendre(2, 3).unwrap();
        let prepared = bound.prepare_cell(&geometry(), &rule).unwrap();
        let previous = BTreeMap::from([(fields[0].field, vec![5.; 3])]);
        let candidate = [2.; 3]; // u=3*2=6, previous physical u=5.
        let action = prepared.linearize(&previous, &candidate).unwrap();
        // Supply an independent physical rate, rather than manufacturing a previous State.
        let independent = FieldValues {
            current: &candidate,
            rates: Some(&[4.; 3]),
        };
        for (pullback, expected) in [
            (
                FieldPullback {
                    current: Some(1.),
                    rate: None,
                },
                20.,
            ),
            (
                FieldPullback {
                    current: None,
                    rate: Some(1.),
                },
                12.,
            ),
            (
                FieldPullback {
                    current: Some(2.),
                    rate: Some(-3.),
                },
                4.,
            ),
        ] {
            let partial = nonlinear_action(&prepared, &independent, Some(pullback), 3).unwrap();
            for i in 0..3 {
                close(partial.residual[i], 7. * 84. / 6.);
                for j in 0..3 {
                    close(partial.jacobian[i * 3 + j], 7. * 3. * expected * mass(i, j));
                }
            }
        }
        let rate = 1. / step;
        let residual = 12. * rate + 36.;
        // R_u+R_v/dt = 2v+2u+2u/dt; the candidate scale adds a factor three.
        let jacobian = 2. * rate + 12. + 12. / step;
        for (i, gradient_i) in GRADIENT.iter().enumerate() {
            close(action.residual[i], 7. * residual / 6.);
            for (j, gradient_j) in GRADIENT.iter().enumerate() {
                close(
                    action.jacobian[i * 3 + j],
                    7. * 3.
                        * (jacobian * mass(i, j)
                            + 0.5
                                * (gradient_i[0] * gradient_j[0] + gradient_i[1] * gradient_j[1])),
                );
            }
        }
        assert_eq!(
            prepared.residual(&previous, &candidate).unwrap(),
            action.residual
        );
        assert!(prepared.residual(&BTreeMap::new(), &candidate).is_err());
        for values in [vec![5.; 2], vec![f64::NAN; 3]] {
            assert!(
                prepared
                    .residual(&BTreeMap::from([(fields[0].field, values)]), &candidate)
                    .is_err()
            );
        }
    }
}

#[test]
fn unchanged_physical_state_has_zero_rate_even_for_small_steps() {
    let form = derive(
        "model Storage() {
        domain body=box(0,1,0,1); state u:1 on body;
        relation r on body {derivative(u*u)=0;}
    }",
    )
    .unwrap();
    let (mut fields, rows) = form.si_bindings(|_| Ok(p1())).unwrap();
    fields[0].scale = DynQuantity::new(0.1, fields[0].scale.dim());
    let bound = form
        .bind(
            ReferenceCell::simplex(2).unwrap(),
            &fields,
            &rows,
            Some(&RegionTimeBinding {
                step: DynQuantity::new(1e-15, dim([0, 0, 1, 0, 0, 0, 0])),
                states: Vec::new(),
            }),
        )
        .unwrap();
    let rule = simplex_duffy_gauss_legendre(2, bound.minimum_gauss_points()).unwrap();
    let prepared = bound.prepare_cell(&geometry(), &rule).unwrap();
    let candidate = [3.; 3];
    let physical = 3. * 0.1;
    let previous = BTreeMap::from([(fields[0].field, vec![physical; 3])]);
    // Dividing history by 0.1 and subtracting in normalized coordinates would
    // give a spurious physical rate of about -0.0444 at this tiny step.
    assert_eq!(
        prepared.residual(&previous, &candidate).unwrap(),
        vec![0.; 3]
    );
}

#[test]
fn complex_nonlinear_rate_action_pulls_back_real_and_imaginary_history() {
    let source = "model Storage() {
        domain body=box(0,2,0,3); state u:complex<1> on body;
        relation r on body {derivative(math.abs2(u))*u=0;}
    }";
    let (transaction, model, symbols) = compile("complex-storage-rate.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let form = CompiledRegionForm::<C>::derive(&program, symbols.get("body").unwrap(), 2).unwrap();
    let (mut fields, rows) = form.si_bindings(|_| Ok(Space::cell_constant())).unwrap();
    fields[0].scale = DynQuantity::new(3., fields[0].scale.dim());
    let reference = ReferenceCell::hypercube(2).unwrap();
    let bound = form
        .bind(
            reference,
            &fields,
            &rows,
            Some(&RegionTimeBinding {
                step: DynQuantity::new(0.25, dim([0, 0, 1, 0, 0, 0, 0])),
                states: Vec::new(),
            }),
        )
        .unwrap();
    let map = AffineGeometryMap::new(reference, 2, vec![1., 1.5], vec![1., 0., 0., 1.5]).unwrap();
    let rule =
        QuadratureRule::tensor_product_gauss_legendre(2, bound.minimum_gauss_points()).unwrap();
    let (prepared, _) = bound
        .prepare_projected_cell_actions::<Real>(&map, &map, None, &rule)
        .unwrap();
    let previous = BTreeMap::from([(fields[0].field, vec![1., 4.])]);
    let action = prepared.linearize(&previous, &[2. / 3., 1.]).unwrap();
    let independent = FieldValues {
        current: &[C::new(2. / 3., 1.)],
        rates: Some(&[C::new(4., -4.)]),
    };
    // At fixed v: J_u=[[8,-16],[24,-32]]. At fixed u: J_v=[[8,12],[12,18]].
    for (pullback, matrix) in [
        (
            FieldPullback {
                current: Some(1.),
                rate: None,
            },
            [8., -16., 24., -32.],
        ),
        (
            FieldPullback {
                current: None,
                rate: Some(1.),
            },
            [8., 12., 12., 18.],
        ),
        (
            FieldPullback {
                current: Some(2.),
                rate: Some(-3.),
            },
            [-8., -68., 12., -118.],
        ),
    ] {
        let partial = nonlinear_action(&prepared, &independent, Some(pullback), 2).unwrap();
        for (actual, expected) in partial.residual.iter().zip([-96., -144.]) {
            close(*actual, expected);
        }
        for (actual, expected) in partial.jacobian.iter().zip(matrix) {
            close(*actual, 6. * 3. * expected);
        }
    }
    // u=2+3i, v=(u-(1+4i))/.25=4-4i. R=2 Re(conj(u)v)u=-16-24i.
    // At fixed previous u, the real-coordinate Jacobian is [[40,32],[72,40]].
    // Multiply by area six and candidate normalization three.
    for (actual, expected) in action.residual.iter().zip([-96., -144.]) {
        close(*actual, expected);
    }
    for (actual, expected) in action.jacobian.iter().zip([720., 576., 1296., 720.]) {
        close(*actual, expected);
    }
}

#[test]
fn assembled_rate_partial_leaves_singular_state_partial_undemanded() {
    let form = derive(
        "model Storage() {
        domain body=box(0,1,0,1); state u:1 on body;
        relation r on body {math.sqrt(u) + 1[s]*derivative(u*u)=0;}
    }",
    )
    .unwrap();
    let (fields, rows) = form.si_bindings(|_| Ok(p1())).unwrap();
    let bound = form
        .bind(
            ReferenceCell::simplex(2).unwrap(),
            &fields,
            &rows,
            Some(&RegionTimeBinding {
                step: DynQuantity::new(0.25, dim([0, 0, 1, 0, 0, 0, 0])),
                states: Vec::new(),
            }),
        )
        .unwrap();
    let rule = simplex_duffy_gauss_legendre(2, bound.minimum_gauss_points()).unwrap();
    let prepared = bound.prepare_cell(&geometry(), &rule).unwrap();
    let point = FieldValues {
        current: &[0.; 3],
        rates: Some(&[3.; 3]),
    };
    let rate = nonlinear_action(
        &prepared,
        &point,
        Some(FieldPullback {
            current: None,
            rate: Some(1.),
        }),
        3,
    )
    .unwrap();
    assert_eq!(rate.residual, vec![0.; 3]);
    assert_eq!(rate.jacobian, vec![0.; 9]);
    for weight in [0., 1.] {
        // An explicit zero direction still demands the State partial at sqrt(0).
        assert!(
            nonlinear_action(
                &prepared,
                &point,
                Some(FieldPullback {
                    current: Some(weight),
                    rate: None
                }),
                3,
            )
            .is_err()
        );
    }
    let invalid = FieldValues {
        current: &[-1.; 3],
        rates: Some(&[3.; 3]),
    };
    assert!(
        nonlinear_action(
            &prepared,
            &invalid,
            Some(FieldPullback {
                current: None,
                rate: Some(1.)
            }),
            3,
        )
        .is_err()
    );
    let incomplete = FieldValues {
        current: &[0.; 2],
        rates: Some(&[3.; 3]),
    };
    assert!(nonlinear_action(&prepared, &incomplete, None, 3).is_err());
}

fn nonlinear_action<
    S: crate::spatial_expression::Coefficient,
    P: crate::form_compiler::region::coordinates::Projection<S, Scalar = f64>,
>(
    prepared: &crate::form_compiler::region::prepared::PreparedRegionCell<S, P>,
    point: &FieldValues<'_, S>,
    pullback: Option<FieldPullback>,
    count: usize,
) -> Result<RegionLinearization, Diagnostic> {
    let mut action = RegionLinearization {
        residual: vec![0.; count],
        jacobian: if pullback.is_some() {
            vec![0.; count * count]
        } else {
            Vec::new()
        },
    };
    prepared.add_nonlinear_action(point, pullback, &mut action)?;
    Ok(action)
}
