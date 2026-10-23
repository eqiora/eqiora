//! Independently integrated R(u,v), without recovering a mass matrix from J/dt.
use super::*;
use crate::form_compiler::region::coordinates::Real;
use crate::form_compiler::region::field_point::FieldPullback;
use eqiora_meshing::QuadratureRule;
use num_complex::Complex64 as C;

#[test]
fn affine_state_and_rate_actions_are_independent_of_the_bound_step() {
    let form = derive(
        "model Storage() {
        domain body=box(0,1,0,1); state u:1 on body;
        relation r on body {2[s]*derivative(u)-div(3[m^2]*grad(u))+5*u=7;}
    }",
    )
    .unwrap();
    let (mut fields, mut rows) = form.si_bindings(|_| Ok(p1())).unwrap();
    fields[0].scale = DynQuantity::new(3., fields[0].scale.dim());
    for scale in rows.values_mut() {
        *scale = DynQuantity::new(7., scale.dim());
    }
    for step in [None, Some(0.25), Some(2.)] {
        let time = step.map(|step| RegionTimeBinding {
            step: DynQuantity::new(step, dim([0, 0, 1, 0, 0, 0, 0])),
            states: Vec::new(),
        });
        let bound = form
            .bind(
                ReferenceCell::simplex(2).unwrap(),
                &fields,
                &rows,
                time.as_ref(),
            )
            .unwrap();
        let rule = simplex_duffy_gauss_legendre(2, bound.minimum_gauss_points()).unwrap();
        let prepared = bound.prepare_cell(&geometry(), &rule).unwrap();
        let weights = [[Some(2.), None], [None, Some(-3.)], [Some(-1.), Some(4.)]];
        for pullback in [
            FieldPullback::Uniform {
                current: Some(1.),
                rate: None,
            },
            FieldPullback::Uniform {
                current: None,
                rate: Some(1.),
            },
            FieldPullback::Uniform {
                current: Some(2.),
                rate: Some(-3.),
            },
            FieldPullback::Coordinates(&weights),
        ] {
            let action = prepared
                .implicit_action(&[2.; 3], Some(&[4.; 3]), Some(pullback))
                .unwrap();
            // u=6 and v=4 give R=2*4+5*6-7=31; integral of each basis is 1/6.
            for (i, gi) in GRADIENT.iter().enumerate() {
                close(action.residual[i], 7. * 31. / 6.);
                for (j, gj) in GRADIENT.iter().enumerate() {
                    let state = 5. * mass(i, j) + 1.5 * (gi[0] * gj[0] + gi[1] * gj[1]);
                    let rate_entry = 2. * mass(i, j);
                    close(
                        action.jacobian[i * 3 + j],
                        7. * 3.
                            * (pullback.weight(j, false).unwrap_or(0.) * state
                                + pullback.weight(j, true).unwrap_or(0.) * rate_entry),
                    );
                }
            }
        }
        if step.is_none() {
            assert!(bound.previous_fields().is_empty());
            assert!(prepared.evaluate(&BTreeMap::new()).is_err());
            assert!(prepared.linearize(&BTreeMap::new(), &[2.; 3]).is_err());
        }
        assert!(prepared.implicit_action(&[2.; 3], None, None).is_err());
        assert!(
            prepared
                .implicit_action(&[2.; 3], Some(&[4.; 2]), None)
                .is_err()
        );
        assert!(
            prepared
                .implicit_action(&[2.; 3], Some(&[f64::NAN; 3]), None)
                .is_err()
        );
    }
}

#[test]
fn affine_unchanged_physical_state_does_not_acquire_a_roundoff_rate() {
    let form = derive(
        "model Storage() {
        domain body=box(0,1,0,1); state u:1 on body;
        relation r on body {derivative(u)=0;}
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
    let previous = BTreeMap::from([(fields[0].field, vec![3. * 0.1; 3])]);
    assert_eq!(prepared.residual(&previous, &[3.; 3]).unwrap(), vec![0.; 3]);
}

#[test]
fn complex_affine_rates_keep_real_linear_columns_before_time_pullback() {
    for conjugate in [false, true] {
        let rate = if conjugate {
            "derivative(math.conj(u))"
        } else {
            "derivative(u)"
        };
        let source = format!("model Storage() {{
            domain body=box(0,2,0,3); state u:complex<1> on body;
            relation r on body {{math.complex(2,3)*1[s]*{rate}+math.complex(5,-7)*u=math.complex(11,13);}}
        }}");
        let (transaction, model, symbols) = compile("complex-affine-rate.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let form =
            CompiledRegionForm::<C>::derive(&program, symbols.get("body").unwrap(), 2).unwrap();
        let (mut fields, rows) = form.si_bindings(|_| Ok(Space::cell_constant())).unwrap();
        fields[0].scale = DynQuantity::new(3., fields[0].scale.dim());
        let reference = ReferenceCell::hypercube(2).unwrap();
        let map =
            AffineGeometryMap::new(reference, 2, vec![1., 1.5], vec![1., 0., 0., 1.5]).unwrap();
        for step in [None, Some(0.25), Some(2.)] {
            let time = step.map(|step| RegionTimeBinding {
                step: DynQuantity::new(step, dim([0, 0, 1, 0, 0, 0, 0])),
                states: Vec::new(),
            });
            let bound = form.bind(reference, &fields, &rows, time.as_ref()).unwrap();
            let rule =
                QuadratureRule::tensor_product_gauss_legendre(2, bound.minimum_gauss_points())
                    .unwrap();
            let (prepared, _) = bound
                .prepare_projected_cell_actions::<Real>(&map, &map, None, &rule)
                .unwrap();
            let expected_residual = if conjugate { [96., 48.] } else { [240., -48.] };
            let rate_matrix = if conjugate {
                [2., 3., 3., -2.]
            } else {
                [2., -3., 3., 2.]
            };
            let weights = [[Some(2.), None], [None, Some(-3.)]];
            for pullback in [
                FieldPullback::Uniform {
                    current: Some(1.),
                    rate: None,
                },
                FieldPullback::Uniform {
                    current: None,
                    rate: Some(1.),
                },
                FieldPullback::Uniform {
                    current: Some(2.),
                    rate: Some(-3.),
                },
                FieldPullback::Coordinates(&weights),
            ] {
                let action = prepared
                    .implicit_action(&[2. / 3., 1.], Some(&[4., -4.]), Some(pullback))
                    .unwrap();
                for (actual, expected) in action.residual.iter().zip(expected_residual) {
                    close(*actual, expected);
                }
                for (index, ((actual, state), rate_entry)) in action
                    .jacobian
                    .iter()
                    .zip([5., 7., -7., 5.])
                    .zip(rate_matrix)
                    .enumerate()
                {
                    close(
                        *actual,
                        6. * 3.
                            * (pullback.weight(index % 2, false).unwrap_or(0.) * state
                                + pullback.weight(index % 2, true).unwrap_or(0.) * rate_entry),
                    );
                }
            }
        }
    }
}
