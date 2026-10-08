use super::*;
use crate::form_compiler::vocabulary::{CONJUGATED_TEST_PAIRING, VALUE_PAIRING, WeakSign};

const SOURCE: &str = r#"public component Wave(
  support body:volume(ambient_dimension=1),
  support left:boundary(parent=body),
  support right:boundary(parent=body)
) {
  parameter a: complex<m^2> = math.complex(-6[m^2], 6[m^2]);
  parameter q: complex<1> = math.complex(3, -1);
  parameter f: complex<1> = math.complex(1, 3);
  parameter g: complex<m> = math.complex(2[m], -4[m]);
  variable u: complex<1> on body;
  relation balance on body { -div(a*grad(u)) + q*u = f; }
  relation fixed on left { trace(u) = math.complex(1, 2); }
  relation flux on right { normal(a*grad(u)) = g; }
  form weak for balance {
    test eta:1 for u zero_on left;
    integrate(body,inner(grad(eta),a*grad(u)) + inner(eta,q*u)) =
      integrate(body,inner(eta,f)) + integrate(right,inner(trace(eta),g));
  }
}"#;

fn admit(source: &str) -> Result<DerivedScalarGalerkinForm, Diagnostic> {
    use eqiora_compiler::{CompiledModel, StaticBindingValue};
    let graph = eqiora_geometry::GeometryGraph::new();
    let interval = graph.interval([0.0, 6.0]).unwrap();
    let geometry = graph
        .build(
            &interval,
            &std::collections::BTreeMap::from([
                ("body".to_owned(), vec![interval.region().into()]),
                ("left".to_owned(), vec![interval.boundaries()[0].into()]),
                ("right".to_owned(), vec![interval.boundaries()[1].into()]),
            ]),
        )
        .unwrap();
    let bindings = ["body", "left", "right"].map(|name| {
        (
            name,
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set(name).unwrap(),
                parent: (name != "body").then(|| geometry.entity_set("body").unwrap()),
            },
        )
    });
    let compiled =
        CompiledModel::compile_selected("complex-correspondence.eqi", source, "Wave", &bindings)
            .map_err(|errors| {
                errors
                    .into_iter()
                    .next()
                    .expect("failed compilation has a diagnostic")
            })?;
    let projection = compiled
        .authored_formulations()
        .next()
        .unwrap()
        .projection()
        .clone();
    let (transaction, model, symbols) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry]).unwrap();
    let derived =
        derive_candidate_with_dimension(&program, symbols.get("body").unwrap(), 1)?.unwrap();
    derived.validate_certificate()?;
    super::super::authored::admit(&projection, &program, &derived)?;
    Ok(derived)
}

#[test]
fn complex_helmholtz_correspondence_binds_reaction_conjugation_phase_and_boundary() {
    let form = admit(SOURCE).unwrap();
    assert_eq!(
        form.certificate.formulation.rules[0].id(),
        CONJUGATED_TEST_PAIRING
    );
    assert!(
        form.certificate
            .formulation
            .rules
            .iter()
            .any(|rule| rule.id() == VALUE_PAIRING)
    );
    let reaction = form
        .certificate
        .entries
        .iter()
        .position(|entry| entry.rule_id == VALUE_PAIRING)
        .unwrap();
    assert_eq!(form.certificate.entries[reaction].sign, WeakSign::Positive);
    let mut forged = form.clone();
    forged.certificate.entries[reaction].sign = WeakSign::Negative;
    assert!(forged.validate_certificate().is_err());
    let mut forged = form.clone();
    forged.certificate.formulation.conjugate_test = false;
    assert!(forged.validate_certificate().is_err());
    assert!(
        form.admit_quadrature(&QuadratureRule::gauss_legendre(2).unwrap())
            .is_err()
    );
    admit(&SOURCE.replace("-div(a*grad(u)) + q*u = f", "div(a*grad(u)) - q*u = -f")).unwrap();
    for (from, to) in [
        ("inner(grad(eta),a*grad(u))", "inner(a*grad(u),grad(eta))"),
        ("inner(eta,q*u)", "inner(eta,-q*u)"),
        ("inner(eta,q*u)", "inner(eta,math.conj(q)*u)"),
        ("inner(eta,f)", "inner(f,eta)"),
        ("inner(eta,f)", "inner(eta,math.conj(f))"),
        ("inner(trace(eta),g)", "inner(trace(eta),math.conj(g))"),
        ("inner(trace(eta),g)", "inner(trace(eta),-g)"),
        ("q*u = f", "q*u*u = f"),
        ("q*u = f", "q*(u+f) = f"),
        ("q*u = f", "q*math.conj(u) = f"),
        ("-div(a*grad(u))", "-div(a*u*grad(u))"),
    ] {
        assert!(
            admit(&SOURCE.replace(from, to)).is_err(),
            "accepted mutation {to}"
        );
    }
}

#[test]
fn real_reaction_uses_the_same_inventory_and_conjugation_is_identity() {
    let source = real_source();
    let form = admit(&source).unwrap();
    assert!(!form.conjugate_test);
    assert!(
        form.certificate
            .entries
            .iter()
            .any(|entry| entry.rule_id == VALUE_PAIRING)
    );
    admit(&source.replace("q*u = f", "q*math.conj(u) = f")).unwrap();
}

fn real_source() -> String {
    SOURCE
        .replace("complex<m^2>", "m^2")
        .replace("complex<1>", "1")
        .replace("complex<m>", "m")
        .replace("math.complex(-6[m^2], 6[m^2])", "6[m^2]")
        .replace("math.complex(3, -1)", "3")
        .replace("math.complex(1, 3)", "1")
        .replace("math.complex(1, 2)", "1")
        .replace("math.complex(2[m], -4[m])", "2[m]")
}

#[test]
fn real_nonpolynomial_forcing_retains_its_source_grouping() {
    let source = real_source()
        .replace(" + q*u", "")
        .replace(" + inner(eta,q*u)", "")
        .replace("inner(grad(eta),a*grad(u))", "dot(grad(eta),a*grad(u))")
        .replace("inner(eta,f)", "eta*f")
        .replace("inner(trace(eta),g)", "trace(eta)*g");
    for forcing in ["math.sin(f)+f", "math.sin(f)-f", "-(math.sin(f)+f)"] {
        let source = source
            .replace("= f;", &format!("= {forcing};"))
            .replace("eta*f)", &format!("eta*({forcing}))"));
        let form = admit(&source).unwrap();
        assert_eq!(form.volume_nodes.values.len(), 1);
        assert!(!form.volume_nodes.values[0].trial_dependent);
        assert!(admit(&source.replace("trace(eta)*g", "-trace(eta)*g")).is_err());
    }
}
