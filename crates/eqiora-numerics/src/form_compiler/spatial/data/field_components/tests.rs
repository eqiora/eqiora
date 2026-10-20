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
fn field_parameter_products_hold_fields_fixed_and_rebind_parameters() {
    let (component, program, parameters) = component("(1+p*u)*u");
    let parameter = coordinates(&program, &parameters).unwrap().remove(0);
    for p in [0., 3.] {
        let point = ParameterPoint::bind(&program, &parameters, &[p, 0.]).unwrap();
        let bound = component.bind_parameter_point(&point).unwrap();
        for u in [-1., 2.] {
            let (value, gradient) = bound.action(&[0.25], &[u], true).unwrap();
            assert_eq!(value, u + p * u * u);
            assert_eq!(gradient, [[1. + 2. * p * u, 0.]]);
            assert_eq!(
                bound
                    .parameter_action(&[0.25], &[u], &parameter, false)
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
            .parameter_action(&[0.25], &[], &parameter, false)
            .map(|(value, _)| value)
            .is_err()
    );
    assert!(
        component
            .parameter_action(&[0.25], &[f64::NAN], &parameter, false)
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
            .parameter_action(&[0.5], &[0.], &parameter, false)
            .map(|(value, _)| value)
            .unwrap(),
        0.
    );
    assert!(component.action(&[0.5], &[0.], true).is_err());
    assert!(
        component
            .parameter_action(&[0.5], &[-1.], &parameter, false)
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
            .parameter_action(&[0.5], &[2.], &parameter, false)
            .map(|(value, _)| value)
            .is_err()
    );
    for p in [-2_f64, 2.] {
        let point = ParameterPoint::bind(&program, &parameters, &[p, 0.]).unwrap();
        let bound = component.bind_parameter_point(&point).unwrap();
        assert_eq!(
            bound
                .parameter_action(&[0.5], &[2.], &parameter, false)
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
            .parameter_action(&[0.5], &[2.], &parameter, false)
            .map(|(value, _)| value)
            .unwrap(),
        0.
    );
    let invalid = ParameterPoint::bind(&program, &parameters, &[-1., 0.]).unwrap();
    assert!(
        component
            .bind_parameter_point(&invalid)
            .unwrap()
            .parameter_action(&[0.5], &[2.], &parameter, false)
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
                .parameter_action(&[0.25], &[3.], &parameter, false)
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
                .parameter_action(&[0.5], &[2.], &selected[0], false)
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
                    .parameter_action(&[0.5], &[2.], &selected[1], false)
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
                .parameter_action(&[0.5], &[2.], &selected[0], false)
                .map(|(value, _)| value)
                .is_err()
        );
        assert_eq!(
            bound
                .parameter_action(&[0.5], &[2.], &selected[1], false)
                .map(|(value, _)| value)
                .unwrap(),
            2.
        );
        let point = ParameterPoint::bind(&program, &parameters, &[3., 0.]).unwrap();
        let bound = component.bind_parameter_point(&point).unwrap();
        // A frozen property still checks its original validity domain.
        assert!(
            bound
                .parameter_action(&[0.5], &[2.], &selected[1], false)
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
            let (value, gradient) = bound.action(&[0.25], &[u], true).unwrap();
            assert_eq!(value, C::new((1. + p * a + q) * a, (1. + p * a - q) * b));
            assert_eq!(
                gradient,
                [[
                    C::new(1. + 2. * p * a + q, p * b),
                    C::new(0., 1. + p * a - q)
                ]]
            );
            let (value, gradient) = bound
                .parameter_action(&[0.25], &[u], &coordinates[0], true)
                .unwrap();
            assert_eq!(value, C::new(a * a, a * b));
            assert_eq!(gradient, [[C::new(2. * a, b), C::new(0., a)]]);
            let (value, gradient) = bound
                .parameter_action(&[0.25], &[u], &coordinates[1], true)
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
        let (value, actual) = component.action(&[0.25], &[C::new(2., -3.)], true).unwrap();
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
    let (value, gradient) = component.action(&[0.25], &values, true).unwrap();
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
    assert!(component.action(&[0.25], &invalid, false).is_err());

    let source = "model Products() {
        domain body=box(0,1);parameter p:1=0;parameter q:1=0;
        variable u:complex<1> on body;
        relation r on body {u=math.complex(math.real(u),math.imag(u));}
    }";
    let (component, _, _) = component_from_source::<C>(source, "r", None);
    let (value, _) = component
        .action(&[0.25], &[C::new(-0., -0.)], false)
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
        let coefficient = &affine.terms[0].2;
        assert!(coefficient.evaluate(&[0.25]).is_err());
        let images = coefficient.coordinate_images(1).unwrap();
        for (image, expected) in images.iter().zip(expected) {
            assert_eq!(image.evaluate(&[0.25]).unwrap(), expected);
        }
    }
}
