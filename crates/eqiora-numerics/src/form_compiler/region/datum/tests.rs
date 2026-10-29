//! Primal operation order and fixed-geometry coefficient derivatives.
use super::*;
use eqiora_core::{Id, entity::kinds};
use eqiora_graph::{GraphStore, InMemoryGraphStore};

fn model(
    rhs: &str,
    p: f64,
    r: f64,
) -> (
    KernelProgram,
    RawId,
    Id<kinds::Parameter>,
    Id<kinds::Parameter>,
) {
    let source = format!(
        "model M(){{domain body=box(0,3,0,2);domain wall=boundary(body,axis=1,side=upper);variable u:vector<1,2> on body in smooth;variable q:1 on body;coordinate x:m on body from body[0];parameter p:1={p:e};parameter r:1={r:e};relation bc on wall{{trace(u)={rhs};}}}}"
    );
    let (transaction, id, symbols) = eqiora_compiler::compile("datum.eqi", &source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), id).unwrap(),
        symbols.get("bc").unwrap(),
        symbols.get("p").unwrap().downcast().unwrap(),
        symbols.get("r").unwrap().downcast().unwrap(),
    )
}

fn datum(program: &KernelProgram, relation: RawId) -> Result<PrescribedDatum<f64>, Diagnostic> {
    datum_with_coefficients(program, relation, &BTreeMap::new())
}

fn datum_with_coefficients(
    program: &KernelProgram,
    relation: RawId,
    coefficients: &BTreeMap<RawId, crate::form_compiler::spatial::data::Components<f64>>,
) -> Result<PrescribedDatum<f64>, Diagnostic> {
    let typed = super::super::super::scalar::typed_relation(program, relation)?;
    let [root] = typed.expression().roots() else {
        unreachable!()
    };
    let Some(ExprNode::Sub(_, rhs)) = typed.expression().node(*root) else {
        unreachable!()
    };
    let context = Context {
        time_s: None,
        program,
        dag: typed.expression(),
        owner: relation,
        dimension: 2,
        coefficients,
    };
    PrescribedDatum::derive(
        &context,
        &typed,
        &typed.node_type(*rhs).unwrap().value_type,
        Some(*rhs),
    )
}

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }
}

#[test]
fn outer_normal_division_and_multiplication_are_not_precombined() {
    // p=2^-1024 and r=2^-1023: 1/p overflows, while each n/p is finite,
    // followed by multiplication by r yielding exactly 2*n.
    let (program, relation, _, _) = model(
        "(normal_vector()/p)*r",
        f64::MIN_POSITIVE / 4.,
        f64::MIN_POSITIVE / 2.,
    );
    let value = datum(&program, relation).unwrap();
    close(
        &value.evaluate(&[2., 2.], &[0.6, 0.8]).unwrap(),
        &[1.2, 1.6],
    );
    close(
        &value.evaluate(&[2., 2.], &[-0.6, -0.8]).unwrap(),
        &[-1.2, -1.6],
    );
}

#[test]
fn parameter_in_numerator_and_denominator_has_the_analytic_derivative() {
    let (program, relation, p, _) = model("p*normal_vector()/(p+1)", 1., 7.);
    let value = datum(&program, relation).unwrap();
    close(
        &value.evaluate(&[2., 2.], &[0.6, 0.8]).unwrap(),
        &[0.3, 0.4],
    );
    let coordinate = crate::parameter_point::coordinates(&program, &[p])
        .unwrap()
        .remove(0);
    let mut derivative = value.clone();
    derivative.input_partial(&coordinate, 2).unwrap();
    // d[p/(p+1)]/dp=1/(p+1)^2; p=1 gives n/4.
    close(
        &derivative.evaluate(&[2., 2.], &[0.6, 0.8]).unwrap(),
        &[0.15, 0.2],
    );
    let point = crate::parameter_point::ParameterPoint::bind(&program, &[p], &[3.]).unwrap();
    derivative.bind_inputs(&point).unwrap();
    close(
        &derivative.evaluate(&[2., 2.], &[0.6, 0.8]).unwrap(),
        &[0.0375, 0.05],
    );
}

#[test]
fn invalid_primal_denominator_survives_even_an_unrelated_zero_tangent() {
    let (program, relation, p, r) = model("p*normal_vector()/(p+1)", -1., 7.);
    let value = datum(&program, relation).unwrap();
    assert!(
        value
            .evaluate(&[2., 2.], &[0.6, 0.8])
            .unwrap_err()
            .message()
            .contains("non-finite vector boundary datum")
    );
    for parameter in [p, r] {
        let coordinate = crate::parameter_point::coordinates(&program, &[parameter])
            .unwrap()
            .remove(0);
        let mut derivative = value.clone();
        derivative.input_partial(&coordinate, 2).unwrap();
        assert!(
            derivative
                .evaluate(&[2., 2.], &[0.6, 0.8])
                .unwrap_err()
                .message()
                .contains("non-finite vector boundary datum")
        );
    }
}

#[test]
fn unknown_scalar_factor_is_not_reinterpreted_as_prescribed_data() {
    let (program, relation, _, _) = model("normal_vector()*q", 1., 7.);
    let error = datum(&program, relation).unwrap_err();
    assert!(error.message().contains("unknown-dependent coefficient"));
}

#[test]
fn scalar_times_gradient_is_distinct_from_gradient_of_the_product() {
    let (program, relation, _, _) = model("trace((x/1[m])*grad(x*x/(2*1[m])))", 1., 7.);
    let outside = datum(&program, relation).unwrap();
    let (program, relation, _, _) = model("trace(grad((x/1[m])*(x*x/(2*1[m]))))", 1., 7.);
    let inside = datum(&program, relation).unwrap();
    // At x=2m, (x/m)*grad(x^2/(2m))=4 e_x; grad(x^3/(2m^2))=6 e_x.
    close(&outside.evaluate(&[2., 2.], &[0., 1.]).unwrap(), &[4., 0.]);
    close(&inside.evaluate(&[2., 2.], &[0., 1.]).unwrap(), &[6., 0.]);
}

#[test]
fn authenticated_carrier_and_direct_profile_share_values_and_parameter_derivative() {
    let source = "model M(){domain body=box(0,3,0,2);domain wall=boundary(body,axis=1,side=upper);variable u:vector<1,2> on body in smooth;variable profile:1 on body in h1;parameter p:1=1;coordinate x:m on body from body[0];relation definition on body{profile-p*x/1[m]=0;}relation carrier on wall{trace(u)=normal(isotropic_lift(profile));}relation direct on wall{trace(u)=normal_vector()*p*x/1[m];}}";
    let (transaction, id, symbols) = eqiora_compiler::compile("carrier.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), id).unwrap();
    let definition = symbols.get("definition").unwrap();
    let typed = super::super::super::scalar::typed_relation(&program, definition).unwrap();
    let Some(ExprNode::Sub(_, rhs)) = typed.expression().node(typed.expression().roots()[0]) else {
        unreachable!()
    };
    let profile = symbols.get("profile").unwrap();
    let Some(eqiora_schema::kernel::KernelNode::Field(field)) = program.node(profile) else {
        unreachable!()
    };
    let context = Context {
        time_s: None,
        program: &program,
        dag: typed.expression(),
        owner: definition,
        dimension: 2,
        coefficients: &BTreeMap::new(),
    };
    let coefficient = crate::form_compiler::spatial::data::Components::prescribed(
        &context,
        field.value_type(),
        Some(*rhs),
    )
    .unwrap();
    let coefficients = BTreeMap::from([(profile, coefficient)]);
    let carrier_relation = symbols.get("carrier").unwrap();
    assert!(
        datum(&program, carrier_relation)
            .unwrap_err()
            .message()
            .contains("unknown-dependent coefficient")
    );
    let carrier = datum_with_coefficients(&program, carrier_relation, &coefficients).unwrap();
    let direct = datum(&program, symbols.get("direct").unwrap()).unwrap();
    let p = symbols.get("p").unwrap().downcast().unwrap();
    let coordinate = crate::parameter_point::coordinates(&program, &[p])
        .unwrap()
        .remove(0);
    for mut value in [carrier, direct] {
        close(
            &value.evaluate(&[2., 2.], &[0.6, 0.8]).unwrap(),
            &[1.2, 1.6],
        );
        value.input_partial(&coordinate, 2).unwrap();
        close(
            &value.evaluate(&[2., 2.], &[0.6, 0.8]).unwrap(),
            &[1.2, 1.6],
        );
    }
}

#[test]
fn independent_quotient_branches_do_not_create_invalid_zero_tangents() {
    let denominator = f64::MIN_POSITIVE / 4.;
    for rhs in [
        "(normal_vector()/p)*r",
        "r*(normal_vector()/p)",
        "(normal_vector()*r)/p",
    ] {
        let (program, relation, _, r) = model(rhs, denominator, 1.);
        let mut value = datum(&program, relation).unwrap();
        assert!(
            value
                .evaluate(&[2., 2.], &[0.6, 0.8])
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
        let coordinate = crate::parameter_point::coordinates(&program, &[r])
            .unwrap()
            .remove(0);
        value.input_partial(&coordinate, 2).unwrap();
        // The exact derivative wrt r is n/p, whose two binary64 components are finite.
        assert_eq!(
            value
                .evaluate(&[2., 2.], &[0.6, 0.8])
                .expect("finite analytic derivative"),
            vec![0.6 / denominator, 0.8 / denominator]
        );
    }
}
