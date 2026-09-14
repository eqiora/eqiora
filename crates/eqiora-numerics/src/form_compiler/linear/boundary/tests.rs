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

 variable u: 1 on body;
 variable v: 1 on body;
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

fn derive(source: &str) -> Result<CompiledLinearBlockForm, Diagnostic> {
    let (transaction, model, symbols) = compile("boundary.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    CompiledLinearBlockForm::derive(
        &program,
        symbols.get("body").unwrap(),
        1,
        &std::collections::BTreeSet::new(),
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
    derive(&SOURCE.replace("parameter k: 1 = 2;", "parameter k: 1 = 2; variable a: 1 on body; relation coefficient on body { a - k = 0; }")
        .replace("k * grad(u)", "a * grad(u)")).unwrap();
}

#[test]
fn duplicate_missing_and_unknown_dependent_boundaries_reject() {
    assert!(derive(&SOURCE.replace("trace(v) = 4", "trace(u) = 4")).is_err());
    assert!(derive(&SOURCE.replace("trace(u) = 2", "trace(u) = trace(v)")).is_err());
    assert!(derive(&SOURCE.replace("relation vl on left { trace(v) = 4; }", "")).is_err());
}
