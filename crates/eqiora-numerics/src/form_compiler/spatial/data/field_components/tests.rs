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

fn component_from_source(
    source: &str,
    relation: &str,
    prescribed: Option<(&str, &str)>,
) -> (
    Arc<FieldComponent<f64>>,
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
        let context = Context::<f64> {
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
    let context = Context::<f64> {
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
            assert_eq!(gradient, [1. + 2. * p * u]);
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
