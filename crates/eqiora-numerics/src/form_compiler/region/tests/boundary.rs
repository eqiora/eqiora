use super::*;
use crate::canonical_boundary::PhysicalBoundaryQuantity;

fn boundary_fixture(
    source: &str,
    addition: &str,
) -> (
    KernelProgram,
    CompiledRegionForm<f64>,
    BTreeMap<String, RawId>,
) {
    boundary_fixture_dimension(source, addition, 2)
}

fn boundary_fixture_dimension(
    source: &str,
    addition: &str,
    dimension: usize,
) -> (
    KernelProgram,
    CompiledRegionForm<f64>,
    BTreeMap<String, RawId>,
) {
    let (prefix, _) = source.rsplit_once('}').unwrap();
    let source = format!("{prefix} {addition} }}");
    let (transaction, model, symbols) = compile("boundary.eqi", &source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let ids = ["body", "wall", "law", "v", "p", "d", "p0"]
        .into_iter()
        .filter_map(|name| symbols.get(name).map(|id| (name.to_owned(), id)))
        .collect::<BTreeMap<_, _>>();
    let form = CompiledRegionForm::<f64>::derive(&program, ids["body"], dimension).unwrap();
    (program, form, ids)
}

#[test]
fn complete_mixed_stress_selects_velocity_row_without_pressure_boundary() {
    let addition = "domain wall = boundary(body, axis=0, side=lower);
        relation law on wall { normal(2*viscosity*symmetric_part(grad(v))-isotropic_lift(p)) = 0; }";
    let (program, form, ids) = boundary_fixture(MIXED, addition);
    let law = form
        .boundary_laws(&program, ids["wall"], ids["law"], None)
        .unwrap()
        .remove(0);
    assert_eq!(law.tested, ids["v"]);
    assert_eq!(law.quantity, PhysicalBoundaryQuantity::Flux);
    assert_eq!(law.evaluate(&[0.0, 0.3], &[-1.0, 0.0]).unwrap(), [0.0, 0.0]);
    assert!(
        form.rows
            .iter()
            .find(|row| row.tested == ids["p"])
            .unwrap()
            .flux
            .is_empty()
    );
    for law in [
        "trace(p) = 0",
        "normal(2*viscosity*symmetric_part(grad(v))) = 0",
    ] {
        let (program, form, ids) = boundary_fixture(
            MIXED,
            &format!(
                "domain wall = boundary(body,axis=0,side=lower); relation law on wall {{ {law}; }}"
            ),
        );
        assert!(
            form.boundary_laws(&program, ids["wall"], ids["law"], None)
                .is_err()
        );
    }
}

#[test]
fn eliminated_state_trace_keeps_original_state_and_tested_rate() {
    let (program, form, ids) = boundary_fixture(
        ELIMINATED,
        "domain wall = boundary(body,axis=0,side=lower); relation law on wall { trace(d) = 0; }",
    );
    let law = form
        .boundary_laws(&program, ids["wall"], ids["law"], None)
        .unwrap()
        .remove(0);
    assert_eq!(law.tested, ids["v"]);
    assert_eq!(law.trace_field, Some(ids["d"]));
    assert_eq!(law.quantity, PhysicalBoundaryQuantity::Trace);
}

#[test]
fn vector_potential_trace_and_normal_lift_retain_mathematical_data() {
    let potential = "domain wall = boundary(body,axis=0,side=lower);
        variable phi: m^2/s on body in smooth;
        parameter a: 1/s = 3;
        parameter b: 1/s = 5;
        relation potential on body { phi = a*coordinate(0)^2 + b*coordinate(1)^2; }
        relation law on wall { trace(v) = trace(grad(phi)); }";
    let (program, form, ids) = boundary_fixture(MIXED, potential);
    let law = form
        .boundary_laws(&program, ids["wall"], ids["law"], None)
        .unwrap()
        .remove(0);
    assert_eq!(
        law.evaluate(&[0.25, 0.5], &[-1.0, 0.0]).unwrap(),
        [1.5, 5.0]
    );
    let normal = "domain wall = boundary(body,axis=0,side=lower);
        parameter p0: kg/(m*s^2) = 3;
        variable pressure_data: kg/(m*s^2) on body in h1;
        relation data on body { pressure_data = p0; }
        relation law on wall { normal(2*viscosity*symmetric_part(grad(v))-isotropic_lift(p)) = normal(isotropic_lift(pressure_data)); }";
    let (program, form, ids) = boundary_fixture(MIXED, normal);
    let mut law = form
        .boundary_laws(&program, ids["wall"], ids["law"], None)
        .unwrap()
        .remove(0);
    assert_eq!(
        law.evaluate(&[0.0, 0.5], &[-1.0, 0.0]).unwrap(),
        [-3.0, 0.0]
    );
    law.bind_parameter_point(&[ids["p0"].downcast().unwrap()], &[7.0])
        .unwrap();
    assert_eq!(
        law.evaluate(&[0.0, 0.5], &[-1.0, 0.0]).unwrap(),
        [-7.0, 0.0]
    );
}

#[test]
fn one_dimensional_vector_keeps_its_gradient_datum_shape() {
    let source = "model One() { domain body = box(0, 1);
        variable v: vector<m, 1> on body in h1;
        variable phi: m^2 on body in smooth;
        relation potential on body { phi = coordinate(0)^2; }
        relation balance on body { -div(grad(v)) = 0; }
    }";
    let (program, form, ids) = boundary_fixture_dimension(
        source,
        "domain wall = boundary(body,axis=0,side=lower); relation law on wall { trace(v) = trace(grad(phi)); }",
        1,
    );
    let law = form
        .boundary_laws(&program, ids["wall"], ids["law"], None)
        .unwrap()
        .remove(0);
    assert_eq!(law.evaluate(&[0.25], &[-1.0]).unwrap(), [0.5]);
}

#[test]
fn additive_normal_data_keeps_relative_sign_under_whole_equation_reversal() {
    for (equation, expected) in [
        (
            "normal(2*viscosity*symmetric_part(grad(v))-isotropic_lift(p)) + normal(isotropic_lift(pressure_data))",
            [3.0, 0.0],
        ),
        (
            "-normal(2*viscosity*symmetric_part(grad(v))-isotropic_lift(p)) - normal(isotropic_lift(pressure_data))",
            [3.0, 0.0],
        ),
        (
            "normal(2*viscosity*symmetric_part(grad(v))-isotropic_lift(p)) - normal(isotropic_lift(pressure_data))",
            [-3.0, 0.0],
        ),
    ] {
        let addition = format!(
            "domain wall = boundary(body,axis=0,side=lower);
            parameter p0: kg/(m*s^2) = 3;
            variable pressure_data: kg/(m*s^2) on body in h1;
            relation data on body {{ pressure_data = p0; }}
            relation law on wall {{ {equation} = 0; }}"
        );
        let (program, form, ids) = boundary_fixture(MIXED, &addition);
        let law = form
            .boundary_laws(&program, ids["wall"], ids["law"], None)
            .unwrap()
            .remove(0);
        assert_eq!(law.evaluate(&[0.0, 0.5], &[-1.0, 0.0]).unwrap(), expected);
    }
}
