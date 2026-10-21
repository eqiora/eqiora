use super::*;
use crate::parameter_point::{ParameterPoint, coordinates};
use eqiora_compiler::compile;
use eqiora_core::{Id, entity::kinds};
use eqiora_graph::{GraphStore, InMemoryGraphStore};

fn component(
    expression: &str,
) -> (
    Arc<FieldComponent<f64>>,
    KernelProgram,
    [Id<kinds::Parameter>; 2],
) {
    let source = format!(
        "model Products() {{
            domain body=box(0,1);
            parameter p:1=0; parameter q:1=0;
            variable u:1 on body;
            relation r on body {{u={expression};}}
        }}"
    );
    component_from_source(&source, "r", None)
}

fn component_from_source<S: Coefficient>(
    source: &str,
    relation: &str,
    prescribed: Option<(&str, &str)>,
) -> (
    Arc<FieldComponent<S>>,
    KernelProgram,
    [Id<kinds::Parameter>; 2],
) {
    let (transaction, model, symbols) = compile("field-products.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let mut known = BTreeMap::new();
    if let Some((field, definition)) = prescribed {
        let field = symbols.get(field).unwrap();
        let owner = symbols.get(definition).unwrap();
        let typed = crate::form_compiler::scalar::typed_relation(&program, owner).unwrap();
        let dag = typed.expression();
        let Some(ExprNode::Sub(_, rhs)) = dag.node(dag.roots()[0]) else {
            panic!("coefficient equation");
        };
        let KernelNode::Field(definition) = program.node(field).unwrap() else {
            panic!("coefficient Field");
        };
        let context = Context::<S> {
            program: &program,
            dag,
            owner,
            dimension: 1,
            coefficients: &known,
            time_s: None,
        };
        let data = Components::prescribed(&context, definition.value_type(), Some(*rhs)).unwrap();
        known.insert(field, data);
    }
    let owner = symbols.get(relation).unwrap();
    let typed = crate::form_compiler::scalar::typed_relation(&program, owner).unwrap();
    let dag = typed.expression();
    let Some(ExprNode::Sub(_, rhs)) = dag.node(dag.roots()[0]) else {
        panic!("test equation");
    };
    let context = Context::<S> {
        program: &program,
        dag,
        owner,
        dimension: 1,
        coefficients: &known,
        time_s: None,
    };
    let value = context
        .field_components(*rhs, 0)
        .unwrap()
        .unwrap()
        .remove(0);
    let parameters = ["p", "q"].map(|name| symbols.get(name).unwrap().downcast().unwrap());
    (value, program, parameters)
}

#[test]
fn rate_partials_do_not_demand_undefined_state_partials() {
    let source = "model Storage() {
        domain body=box(0,1); parameter p:1=2; parameter q:1=0;
        state u:1 on body;
        relation r on body {u=math.sqrt(u)+p*derivative(u)*1[s];}
    }";
    let (component, program, parameters) = component_from_source::<f64>(source, "r", None);
    let rate = Some(FieldDerivatives(&[[false; 2], [true; 2]]));
    let current = Some(FieldDerivatives(&[[true; 2], [false; 2]]));
    // R(u,v,p)=sqrt(u)+pv is defined at u=0. R_v=p and R_pv=1 exist
    // there, although R_u does not. Every query must still demand the primal.
    assert_eq!(
        component.action(&[0.25], &[0., 3.], rate).unwrap(),
        (6., vec![[0., 0.], [2., 0.]])
    );
    assert!(component.action(&[0.25], &[0., 3.], current).is_err());
    assert!(
        component
            .action(
                &[0.25],
                &[0., 3.],
                Some(FieldDerivatives(&vec![[true; 2]; component.columns.len()]))
            )
            .is_err()
    );
    assert!(component.action(&[0.25], &[-1., 3.], rate).is_err());
    let parameter = coordinates(&program, &parameters).unwrap().remove(0);
    assert_eq!(
        component
            .parameter_action(&[0.25], &[0., 3.], &parameter, rate)
            .unwrap(),
        (3., vec![[0., 0.], [1., 0.]])
    );
}

#[test]
fn state_partials_do_not_demand_undefined_rate_partials() {
    let source = "model Storage() {
        domain body=box(0,1); parameter p:1=0; parameter q:1=0;
        state u:1 on body;
        relation r on body {u=u+math.sqrt(derivative(u)*1[s]);}
    }";
    let (component, _, _) = component_from_source::<f64>(source, "r", None);
    let current = Some(FieldDerivatives(&[[true; 2], [false; 2]]));
    let rate = Some(FieldDerivatives(&[[false; 2], [true; 2]]));
    assert_eq!(
        component.action(&[0.25], &[5., 0.], current).unwrap(),
        (5., vec![[1., 0.], [0., 0.]])
    );
    assert!(component.action(&[0.25], &[5., 0.], rate).is_err());
    assert!(component.action(&[0.25], &[5., -1.], current).is_err());
}

#[test]
fn state_and_rate_are_independent_component_coordinates() {
    let source = "model Storage() {
        domain body=box(0,1); parameter p:1=0; parameter q:1=0;
        state u:1 on body;
        relation r on body {u/1[s]=derivative(u*u)+u*u/1[s];}
    }";
    let (component, _, _) = component_from_source::<f64>(source, "r", None);
    let columns = component.columns().collect::<Vec<_>>();
    assert_eq!(columns.len(), 2);
    assert_eq!(columns[0].0, columns[1].0);
    assert_eq!(columns[0].1, BasisComponent::Value(0));
    assert_eq!(columns[1].1, BasisComponent::Value(0));
    assert!(!columns[0].2);
    assert!(columns[1].2);
    assert!(!component.is_affine());
    // R(u,v)=2uv+u² has independent partials R_u=2v+2u, R_v=2u.
    // The integrator, not the expression, supplies v and dv/du.
    for (u, rate) in [(3., -2.), (-1., 4.), (0., 5.)] {
        let (value, gradient) = component
            .action(
                &[0.25],
                &[u, rate],
                Some(FieldDerivatives(&vec![[true; 2]; component.columns.len()])),
            )
            .unwrap();
        assert_eq!(value, 2. * u * rate + u * u);
        assert_eq!(gradient, [[2. * rate + 2. * u, 0.], [2. * u, 0.]]);
        assert_eq!(
            component.action(&[0.25], &[u, rate], None).unwrap(),
            (value, vec![])
        );
    }
    assert!(
        component
            .action(
                &[0.25],
                &[3.],
                Some(FieldDerivatives(&vec![[true; 2]; component.columns.len()]))
            )
            .is_err()
    );
}

#[test]
fn affine_projection_keeps_state_and_rate_coefficients_separate() {
    let source = "model Storage() {
        domain body=box(0,1); parameter p:1=0; parameter q:1=0;
        state u:1 on body;
        relation r on body {u/1[s]=derivative(u)+3*u/1[s];}
    }";
    let (component, _, _) = component_from_source::<f64>(source, "r", None);
    assert!(component.is_affine());
    let row = super::super::affine::project(component);
    assert_eq!(row.offset.evaluate(&[0.25]).unwrap(), 0.);
    assert_eq!(row.terms.len(), 2);
    for (_, basis, derivative, coefficient) in row.terms {
        assert_eq!(basis, BasisComponent::Value(0));
        assert_eq!(
            coefficient.evaluate(&[0.25]).unwrap(),
            if derivative { 1. } else { 3. }
        );
    }
}

#[test]
fn complex_storage_rates_keep_all_four_real_coordinate_partials() {
    use num_complex::Complex64 as C;
    let source = "model Storage() {
        domain body=box(0,1); parameter p:1=0; parameter q:1=0;
        state u:complex<1> on body;
        relation r on body {u/1[s]=derivative(math.abs2(u))*u;}
    }";
    let (component, _, _) = component_from_source::<C>(source, "r", None);
    // For u=a+ib and v=c+id, R=2(ac+bd)(a+ib).
    // At (a,b,c,d)=(2,3,-1,4), these four real partials follow directly.
    let (value, gradient) = component
        .action(
            &[0.25],
            &[C::new(2., 3.), C::new(-1., 4.)],
            Some(FieldDerivatives(&vec![[true; 2]; component.columns.len()])),
        )
        .unwrap();
    assert_eq!(value, C::new(40., 60.));
    assert_eq!(
        gradient,
        [
            [C::new(16., -6.), C::new(16., 44.)],
            [C::new(8., 12.), C::new(12., 18.)],
        ]
    );
}

#[test]
fn field_parameter_products_hold_fields_fixed_and_rebind_parameters() {
    let (component, program, parameters) = component("(1+p*u)*u");
    let parameter = coordinates(&program, &parameters).unwrap().remove(0);
    for p in [0., 3.] {
        let point = ParameterPoint::bind(&program, &parameters, &[p, 0.]).unwrap();
        let bound = component.bind_parameter_point(&point).unwrap();
        for u in [-1., 2.] {
            let (value, gradient) = bound
                .action(
                    &[0.25],
                    &[u],
                    Some(FieldDerivatives(&vec![[true; 2]; component.columns.len()])),
                )
                .unwrap();
            assert_eq!(value, u + p * u * u);
            assert_eq!(gradient, [[1. + 2. * p * u, 0.]]);
            assert_eq!(
                bound
                    .parameter_action(&[0.25], &[u], &parameter, None)
                    .map(|(value, _)| value)
                    .unwrap(),
                u * u
            );
        }
    }
    assert!(
        component
            .bind_parameter_point(&ParameterPoint::default())
            .is_err()
    );
    assert!(
        component
            .parameter_action(&[0.25], &[], &parameter, None)
            .map(|(value, _)| value)
            .is_err()
    );
    assert!(
        component
            .parameter_action(&[0.25], &[f64::NAN], &parameter, None)
            .map(|(value, _)| value)
            .is_err()
    );
}

#[test]
fn parameter_products_do_not_request_unneeded_field_derivatives() {
    let (component, program, parameters) = component("math.sqrt(u)+p*u");
    let parameter = coordinates(&program, &parameters).unwrap().remove(0);
    // At fixed u=0 the map p -> sqrt(u)+p*u is constant and differentiable.
    // Its Field differential is undefined, and the primal at u<0 is invalid.
    assert_eq!(
        component
            .parameter_action(&[0.5], &[0.], &parameter, None)
            .map(|(value, _)| value)
            .unwrap(),
        0.
    );
    assert!(
        component
            .action(
                &[0.5],
                &[0.],
                Some(FieldDerivatives(&vec![[true; 2]; component.columns.len()]))
            )
            .is_err()
    );
    assert!(
        component
            .parameter_action(&[0.5], &[-1.], &parameter, None)
            .map(|(value, _)| value)
            .is_err()
    );
}

#[test]
fn parameter_products_keep_structural_activity_at_zero_inner_tangents() {
    let (component, program, parameters) = component("math.sqrt(p*p)+u");
    let parameter = coordinates(&program, &parameters).unwrap().remove(0);
    // sqrt(p²)=|p| is not differentiable at zero despite d(p²)/dp=0.
    assert!(
        component
            .parameter_action(&[0.5], &[2.], &parameter, None)
            .map(|(value, _)| value)
            .is_err()
    );
    for p in [-2_f64, 2.] {
        let point = ParameterPoint::bind(&program, &parameters, &[p, 0.]).unwrap();
        let bound = component.bind_parameter_point(&point).unwrap();
        assert_eq!(
            bound
                .parameter_action(&[0.5], &[2.], &parameter, None)
                .map(|(value, _)| value)
                .unwrap(),
            p.signum()
        );
    }
}

#[test]
fn an_unconsumed_parameter_keeps_the_original_primal_domain() {
    let (component, program, parameters) = component("math.sqrt(p)+u");
    let parameter = coordinates(&program, &parameters).unwrap().remove(1);
    assert_eq!(
        component
            .parameter_action(&[0.5], &[2.], &parameter, None)
            .map(|(value, _)| value)
            .unwrap(),
        0.
    );
    let invalid = ParameterPoint::bind(&program, &parameters, &[-1., 0.]).unwrap();
    assert!(
        component
            .bind_parameter_point(&invalid)
            .unwrap()
            .parameter_action(&[0.5], &[2.], &parameter, None)
            .map(|(value, _)| value)
            .is_err()
    );
}

#[test]
fn prescribed_field_bindings_keep_the_parameter_chain_and_activity() {
    for expression in ["(1+k)*u", "math.sqrt(k)+u"] {
        let source = format!(
            "model Products() {{
                domain body=box(0,1); coordinate x:m on body from body[0];
                parameter p:1=0; parameter q:1=0;
                variable u:1 on body; variable k:1 on body;
                relation c on body {{k=p*p*x/1[m];}}
                relation r on body {{u={expression};}}
            }}"
        );
        let (component, program, parameters) =
            component_from_source(&source, "r", Some(("k", "c")));
        let parameter = coordinates(&program, &parameters).unwrap().remove(0);
        for p in [-2_f64, 0., 2.] {
            let point = ParameterPoint::bind(&program, &parameters, &[p, 0.]).unwrap();
            let bound = component.bind_parameter_point(&point).unwrap();
            let partial = bound
                .parameter_action(&[0.25], &[3.], &parameter, None)
                .map(|(value, _)| value);
            if expression.starts_with("math.sqrt") {
                // sqrt(p²*x)=|p|*sqrt(x) retains the cusp through a prescribed Field.
                if p == 0. {
                    assert!(partial.is_err());
                } else {
                    assert_eq!(partial.unwrap(), p.signum() * 0.5);
                }
            } else {
                assert_eq!(partial.unwrap(), 2. * p * 0.25 * 3.);
            }
        }
    }
}

#[test]
fn material_parameter_products_obey_only_the_required_first_partial_contract() {
    for profile in ["first_partials", "value_only"] {
        let source = format!(
            "property contract Response(input value:1):1 {{derivatives {profile};}}
            property release Material:Response {{
                analytic {{value=1+value^2;source_unit:1=1;}}
                validity value>=-4 and value<=4;outside reject;branch single;
                citation org.example.manufactured;license spdx.CC0_1_0;
            }}
            component Equation(
                support body:volume(ambient_dimension=1),property response:Response,
                parameter p:1,parameter q:1
            ) {{
                variable u:1 on body;
                relation r on body {{u=response(value=p*u)+q*u;}}
            }}
            model Products() {{
                domain body=box(0,1);parameter p:1=0;parameter q:1=0;
                instance physics:Equation(body=body,response=Material,p=p,q=q);
            }}"
        );
        let (component, program, parameters) = component_from_source(&source, "physics.r", None);
        let selected = coordinates(&program, &parameters).unwrap();
        for p in [0., 1.] {
            let point = ParameterPoint::bind(&program, &parameters, &[p, 0.]).unwrap();
            let bound = component.bind_parameter_point(&point).unwrap();
            let partial = bound
                .parameter_action(&[0.5], &[2.], &selected[0], None)
                .map(|(value, _)| value);
            if profile == "first_partials" {
                assert_eq!(partial.unwrap(), 2. * p * 4.);
            } else {
                assert!(
                    partial
                        .unwrap_err()
                        .message()
                        .contains("value-only contract")
                );
            }
            // q does not enter the property; its value-only contract suffices.
            assert_eq!(
                bound
                    .parameter_action(&[0.5], &[2.], &selected[1], None)
                    .map(|(value, _)| value)
                    .unwrap(),
                2.
            );
        }
        let point = ParameterPoint::bind(&program, &parameters, &[2., 0.]).unwrap();
        let bound = component.bind_parameter_point(&point).unwrap();
        // At p*u=4 the active p direction leaves the declared domain.
        // q holds that input fixed and still has its ordinary derivative.
        assert!(
            bound
                .parameter_action(&[0.5], &[2.], &selected[0], None)
                .map(|(value, _)| value)
                .is_err()
        );
        assert_eq!(
            bound
                .parameter_action(&[0.5], &[2.], &selected[1], None)
                .map(|(value, _)| value)
                .unwrap(),
            2.
        );
        let point = ParameterPoint::bind(&program, &parameters, &[3., 0.]).unwrap();
        let bound = component.bind_parameter_point(&point).unwrap();
        // A frozen property still checks its original validity domain.
        assert!(
            bound
                .parameter_action(&[0.5], &[2.], &selected[1], None)
                .map(|(value, _)| value)
                .is_err()
        );
    }
}

#[test]
fn complex_component_products_retain_independent_real_directions() {
    use num_complex::Complex64 as C;
    let source = "model Products() {
        domain body=box(0,1);parameter p:1=0;parameter q:1=0;
        variable u:complex<1> on body;
        relation r on body {u=(1+p*math.real(u))*u+q*math.conj(u);}
    }";
    let (component, program, parameters) = component_from_source::<C>(source, "r", None);
    let coordinates = coordinates(&program, &parameters).unwrap();
    for p in [0., 3.] {
        let q = -2.;
        let point = ParameterPoint::bind(&program, &parameters, &[p, q]).unwrap();
        let bound = component.bind_parameter_point(&point).unwrap();
        for u in [C::new(-1., 2.), C::new(2., -1.)] {
            let (a, b) = (u.re, u.im);
            // f=(1+p*a)*(a+i*b)+q*(a-i*b). These are independent
            // derivatives with respect to a and b, not a complex derivative.
            let (value, gradient) = bound
                .action(
                    &[0.25],
                    &[u],
                    Some(FieldDerivatives(&vec![[true; 2]; component.columns.len()])),
                )
                .unwrap();
            assert_eq!(value, C::new((1. + p * a + q) * a, (1. + p * a - q) * b));
            assert_eq!(
                gradient,
                [[
                    C::new(1. + 2. * p * a + q, p * b),
                    C::new(0., 1. + p * a - q)
                ]]
            );
            let (value, gradient) = bound
                .parameter_action(
                    &[0.25],
                    &[u],
                    &coordinates[0],
                    Some(FieldDerivatives(&vec![[true; 2]; component.columns.len()])),
                )
                .unwrap();
            assert_eq!(value, C::new(a * a, a * b));
            assert_eq!(gradient, [[C::new(2. * a, b), C::new(0., a)]]);
            let (value, gradient) = bound
                .parameter_action(
                    &[0.25],
                    &[u],
                    &coordinates[1],
                    Some(FieldDerivatives(&vec![[true; 2]; component.columns.len()])),
                )
                .unwrap();
            assert_eq!(value, u.conj());
            assert_eq!(gradient, [[C::new(1., 0.), C::new(0., -1.)]]);
        }
    }
}

#[test]
fn real_outputs_and_imaginary_only_inputs_keep_their_coordinate_roles() {
    use num_complex::Complex64 as C;
    for (expression, expected, gradient) in [
        (
            "math.abs2(u)",
            C::new(13., 0.),
            [C::new(4., 0.), C::new(-6., 0.)],
        ),
        (
            "math.complex(0,math.imag(u)^2)",
            C::new(0., 9.),
            [C::new(0., 0.), C::new(0., -6.)],
        ),
    ] {
        let source = format!(
            "model Products() {{
            domain body=box(0,1);parameter p:1=0;parameter q:1=0;
            variable u:complex<1> on body;relation r on body {{u={expression};}}
        }}"
        );
        let (component, _, _) = component_from_source::<C>(&source, "r", None);
        let (value, actual) = component
            .action(
                &[0.25],
                &[C::new(2., -3.)],
                Some(FieldDerivatives(&vec![[true; 2]; component.columns.len()])),
            )
            .unwrap();
        assert_eq!(value, expected);
        assert_eq!(actual, [gradient]);
    }
}

#[test]
fn mixed_field_domains_preserve_real_inputs_and_complex_output_parts() {
    use num_complex::Complex64 as C;
    let source = "model Products() {
        domain body=box(0,1);parameter p:1=0;parameter q:1=0;
        variable u:complex<1> on body;variable v:1 on body;
        relation r on body {u=(1+v)*u;}
    }";
    let (component, _, _) = component_from_source::<C>(source, "r", None);
    let values = component
        .columns
        .iter()
        .map(|column| {
            if column.scalar_domain == ScalarDomain::Real {
                C::new(2., 0.)
            } else {
                C::new(3., -4.)
            }
        })
        .collect::<Vec<_>>();
    let (value, gradient) = component
        .action(
            &[0.25],
            &values,
            Some(FieldDerivatives(&vec![[true; 2]; component.columns.len()])),
        )
        .unwrap();
    assert_eq!(value, C::new(9., -12.));
    for (column, gradient) in component.columns.iter().zip(gradient) {
        let expected = if column.scalar_domain == ScalarDomain::Real {
            [C::new(3., -4.), C::new(0., 0.)]
        } else {
            [C::new(3., 0.), C::new(0., 3.)]
        };
        assert_eq!(gradient, expected);
    }
    let mut invalid = values;
    let real = component
        .columns
        .iter()
        .position(|column| column.scalar_domain == ScalarDomain::Real)
        .unwrap();
    invalid[real].im = 1.;
    assert!(component.action(&[0.25], &invalid, None).is_err());

    let source = "model Products() {
        domain body=box(0,1);parameter p:1=0;parameter q:1=0;
        variable u:complex<1> on body;
        relation r on body {u=math.complex(math.real(u),math.imag(u));}
    }";
    let (component, _, _) = component_from_source::<C>(source, "r", None);
    let (value, _) = component
        .action(&[0.25], &[C::new(-0., -0.)], None)
        .unwrap();
    assert_eq!(value.re.to_bits(), (-0_f64).to_bits());
    assert_eq!(value.im.to_bits(), (-0_f64).to_bits());
}

#[test]
fn affine_real_projections_of_complex_trials_require_real_coordinates() {
    use num_complex::Complex64 as C;
    for (expression, expected) in [
        ("math.real(u)", [C::new(1., 0.), C::new(0., 0.)]),
        ("math.imag(u)", [C::new(0., 0.), C::new(1., 0.)]),
    ] {
        let source = format!(
            "model Products() {{
                domain body=box(0,1);parameter p:1=0;parameter q:1=0;
                variable u:complex<1> on body;relation r on body {{u={expression};}}
            }}"
        );
        let (component, _, _) = component_from_source::<C>(&source, "r", None);
        assert!(component.is_affine());
        let affine = crate::form_compiler::spatial::data::affine::project(component);
        assert_eq!(affine.terms.len(), 1);
        let coefficient = &affine.terms[0].3;
        assert!(coefficient.evaluate(&[0.25]).is_err());
        let images = coefficient.coordinate_images(1).unwrap();
        for (image, expected) in images.iter().zip(expected) {
            assert_eq!(image.evaluate(&[0.25]).unwrap(), expected);
        }
    }
}

#[test]
fn complex_partial_selection_precedes_scalar_differentiation() {
    let source = "model Material() {
        domain body=box(0,1); parameter p:1=2; parameter q:1=0;
        state u:complex<1> on body;
        relation r on body {u=math.complex(math.sqrt(math.real(u)),p*math.imag(u)*math.imag(u));}
    }";
    let (component, program, parameters) =
        component_from_source::<num_complex::Complex64>(source, "r", None);
    let value = num_complex::Complex64::new(0., 3.);
    let selection = Some(FieldDerivatives(&[[false, true]]));
    let (primal, gradient) = component.action(&[0.25], &[value], selection).unwrap();
    assert_eq!(primal, num_complex::Complex64::new(0., 18.));
    assert_eq!(
        gradient,
        vec![[
            num_complex::Complex64::new(0., 0.),
            num_complex::Complex64::new(0., 12.)
        ]]
    );
    assert!(
        component
            .action(&[0.25], &[value], Some(FieldDerivatives(&[[true, false]])))
            .is_err()
    );
    let parameter = coordinates(&program, &parameters).unwrap().remove(0);
    let (primal, gradient) = component
        .parameter_action(&[0.25], &[value], &parameter, selection)
        .unwrap();
    assert_eq!(primal, num_complex::Complex64::new(0., 9.));
    assert_eq!(gradient[0][1], num_complex::Complex64::new(0., 6.));
    assert!(
        component
            .action(&[0.25], &[num_complex::Complex64::new(-1., 3.)], selection)
            .is_err()
    );
    assert!(
        component
            .action(&[0.25], &[value], Some(FieldDerivatives(&[])))
            .is_err()
    );
}
