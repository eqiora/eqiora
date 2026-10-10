mod geometry;
use super::*;
use crate::canonical_boundary::PhysicalBoundaryQuantity;
use crate::form_compiler::linear::CompiledLinearBlockForm;
use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore};

const SOURCE: &str = r#"
model Boundaries() {
 domain body = box(0, 1);
 domain left = boundary(body, axis = 0, side = lower);
 domain right = boundary(body, axis = 0, side = upper);

 variable u: 1 on body in smooth;
 variable v: 1 on body in smooth;
 parameter k: 1 = 2;
 parameter other: 1 = 2;
 parameter q: 1 / m = 3;
 relation first on body { -div(k * grad(u)) = 0; }
 relation second on body { -div(grad(v)) = 0; }
 relation ul on left { trace(u) = 2; }
 relation vl on left { trace(v) = 4; }
 relation ur on right { normal(k * grad(u)) = q; }
 relation vr on right { normal(grad(v)) = 0; }
}
"#;

fn derive(source: &str) -> Result<CompiledLinearBlockForm<f64>, Diagnostic> {
    derive_with_interface(source, None)
}

fn derive_with_interface(
    source: &str,
    interface_field: Option<&str>,
) -> Result<CompiledLinearBlockForm<f64>, Diagnostic> {
    let (transaction, model, symbols) = compile("boundary.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    CompiledLinearBlockForm::<f64>::derive(
        &program,
        symbols.get("body").unwrap(),
        1,
        &interface_field
            .into_iter()
            .map(|field| InterfaceBoundary {
                boundary: symbols.get("left").unwrap(),
                field: symbols.get(field).unwrap(),
                carrier: None,
            })
            .collect(),
    )
}

#[test]
fn mixed_laws_preserve_data_and_complete_dependencies() {
    let form = derive(SOURCE).unwrap();
    assert_eq!(form.boundary_laws().len(), 2);
    let laws = form
        .boundary_laws()
        .values()
        .flat_map(|laws| laws.values())
        .collect::<Vec<_>>();
    let mut traces = laws
        .iter()
        .filter(|law| law.quantity == PhysicalBoundaryQuantity::Trace)
        .map(|law| law.evaluate(&[0.0], &[]).unwrap()[0])
        .collect::<Vec<_>>();
    traces.sort_by(f64::total_cmp);
    assert_eq!(traces, vec![2.0, 4.0]);
    assert!(
        laws.iter()
            .any(|law| law.quantity == PhysicalBoundaryQuantity::Flux
                && law.evaluate(&[1.0], &[]).unwrap() == vec![3.0])
    );
    assert!(
        laws.iter()
            .any(|law| law.quantity == PhysicalBoundaryQuantity::Flux
                && law.evaluate(&[1.0], &[]).unwrap() == vec![0.0])
    );
    assert_eq!(form.dependencies.len(), 6);
}

#[test]
fn volume_and_boundary_reversal_keep_outward_flux_orientation() {
    for reverse_volume in [false, true] {
        for reverse_boundary in [false, true] {
            let mut source = SOURCE.to_owned();
            if reverse_volume {
                source = source.replace("-div(", "div(");
            }
            if reverse_boundary {
                source = source.replace("normal(k * grad(u)) = q", "q - normal(k * grad(u)) = 0");
            }
            let form = derive(&source).unwrap();
            assert!(
                form.boundary_laws()
                    .values()
                    .flat_map(|laws| laws.values())
                    .any(|law| {
                        law.quantity == PhysicalBoundaryQuantity::Flux
                            && law.evaluate(&[1.0], &[]).unwrap() == vec![3.0]
                    })
            );
        }
    }
}

#[test]
fn flux_preserves_parameter_identity_not_just_its_value() {
    assert!(derive(&SOURCE.replace("normal(k * grad(u))", "normal(other * grad(u))")).is_err());
    assert!(derive(&SOURCE.replace("normal(k * grad(u))", "normal((2 * k) * grad(u))")).is_err());
    derive(&SOURCE.replace("normal(k * grad(u))", "normal(grad(u) * k)")).unwrap();
    derive(
        &SOURCE
            .replace("-div(k * grad(u))", "-div((k * 2) * grad(u))")
            .replace("normal(k * grad(u))", "normal((2 * k) * grad(u))"),
    )
    .unwrap();
    derive(&SOURCE.replace("parameter k: 1 = 2;", "parameter k: 1 = 2; variable a: 1 on body in smooth; relation coefficient on body { a - k = 0; }")
        .replace("k * grad(u)", "a * grad(u)")).unwrap();
}

#[test]
fn duplicate_missing_and_unknown_dependent_boundaries_reject() {
    assert!(derive(&SOURCE.replace("trace(v) = 4", "trace(u) = 4")).is_err());
    assert!(derive(&SOURCE.replace("trace(u) = 2", "trace(u) = trace(v)")).is_err());
    assert!(derive(&SOURCE.replace("relation vl on left { trace(v) = 4; }", "")).is_err());
}

#[test]
fn interface_coverage_preserves_other_fields_exterior_laws() {
    let source = SOURCE.replace("relation ul on left { trace(u) = 2; }", "");
    let form = derive_with_interface(&source, Some("u")).unwrap();
    let mut counts = form
        .boundary_laws()
        .values()
        .map(BTreeMap::len)
        .collect::<Vec<_>>();
    counts.sort_unstable();
    assert_eq!(counts, [1, 2]);
    let missing = source.replace("relation vl on left { trace(v) = 4; }", "");
    let error = derive_with_interface(&missing, Some("u")).unwrap_err();
    assert!(error.message().contains("complete boundary law coverage"));
    let duplicate = derive_with_interface(SOURCE, Some("u")).unwrap_err();
    assert!(duplicate.message().contains("duplicate Field boundary law"));
    let foreign = derive_with_interface(&source, Some("k")).unwrap_err();
    assert!(foreign.message().contains("exact local Field endpoint"));
}

#[test]
fn prescribed_spatial_factor_stays_outside_vector_potential_gradient() {
    let source = r#"
model ScaledDatum() {
    domain body=box(0,1,0,1);
    domain left=boundary(body,axis=0,side=lower);
    domain right=boundary(body,axis=0,side=upper);
    domain bottom=boundary(body,axis=1,side=lower);
    domain top=boundary(body,axis=1,side=upper);
    variable g:m on body in smooth;
    relation potential on body { g=(coordinate(0)^2+coordinate(1)^2)/1[m]; }
    variable u:vector<1,2> on body in smooth;
    relation balance on body { -div(grad(u))=0; }
    relation l on left { trace(u)=coordinate(0)/1[m]*trace(grad(g)); }
    relation r on right { trace(u)=coordinate(0)/1[m]*trace(grad(g)); }
    relation b on bottom { trace(u)=coordinate(0)/1[m]*trace(grad(g)); }
    relation t on top { trace(u)=coordinate(0)/1[m]*trace(grad(g)); }
}
"#;
    let (transaction, model, symbols) = compile("scaled-datum.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let form = CompiledLinearBlockForm::<f64>::derive(
        &program,
        symbols.get("body").unwrap(),
        2,
        &BTreeSet::new(),
    )
    .unwrap();
    let law = &form.boundary_laws()[&symbols.get("u").unwrap()][&symbols.get("top").unwrap()];
    // At x=1/2,y=1: (x/m) grad((x²+y²)/m)=(1/2,1).
    // Differentiating the product instead would incorrectly give (7/4,1).
    assert_eq!(
        law.evaluate(&[0.5, 1.0], &[0.0, 1.0]).unwrap(),
        vec![0.5, 1.0]
    );
}

#[test]
fn simplified_zero_coefficient_is_not_an_unknown_boundary_field() {
    for equation in ["f=0", "-f=0"] {
        let source = SOURCE.replace("parameter k: 1 = 2;", &format!("variable f:1 on body in smooth; relation zero_data on body {{{equation};}} parameter k: 1 = 2;"))
            .replace("relation second on body { -div(grad(v)) = 0; }", "relation second on body { -div(grad(v)) = f*1[1/m^2]; }");
        let form = derive(&source).unwrap();
        assert_eq!(form.fields().len(), 2);
        assert_eq!(form.boundary_laws().len(), 2);
        form.volume().unwrap();
    }
}
