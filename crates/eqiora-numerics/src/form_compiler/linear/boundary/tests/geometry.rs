use super::*;
use eqiora_compiler::{CompiledModel, StaticBindingValue};
use eqiora_geometry::{CanonicalGeometryV1, NamedEntitySet};

fn derive_geometry(
    groups: &[Vec<usize>],
    face: bool,
    omit: bool,
    duplicate: bool,
) -> Result<CompiledLinearBlockForm<f64>, Diagnostic> {
    let mut sets = vec![NamedEntitySet::new("body", 3, vec![0])];
    sets.extend(
        groups
            .iter()
            .enumerate()
            .map(|(i, facets)| NamedEntitySet::new(format!("side{i}"), 2, facets.clone())),
    );
    let geometry = CanonicalGeometryV1::from_convex_polyhedra(
        vec![[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.]],
        vec![vec![
            vec![0, 2, 1],
            vec![0, 1, 3],
            vec![1, 2, 3],
            vec![2, 0, 3],
        ]],
        sets,
        1e-12,
    )
    .unwrap();
    let mut source =
        String::from("public component VectorBoundary(support body: volume(ambient_dimension = 3)");
    for i in 0..groups.len() {
        source += &format!(", support side{i}: boundary(parent = body)");
    }
    let (operator, natural) = if face {
        ("-grad(div(u))", "normal(isotropic_lift(div(u))) = 0")
    } else {
        ("curl(curl(u))", "tangential_trace(-curl(u)) = 0")
    };
    source += &format!(
        ") {{ variable u: vector<1,3> on body; relation volume on body {{ {operator} = 0; }}"
    );
    for i in 0..groups.len() {
        if omit && i == 0 {
            continue;
        }
        source += &format!("relation law{i} on side{i} {{ {natural}; }}");
    }
    if duplicate {
        source += &format!("relation repeated on side0 {{ {natural}; }}");
    }
    source += "}";
    let parent = geometry.entity_set("body").unwrap();
    let names = (0..groups.len())
        .map(|i| format!("side{i}"))
        .collect::<Vec<_>>();
    let mut bindings = vec![(
        "body",
        StaticBindingValue::GeometrySupport {
            geometry: &geometry,
            selection: parent,
            parent: None,
        },
    )];
    bindings.extend(names.iter().map(|name| {
        (
            name.as_str(),
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set(name).unwrap(),
                parent: Some(parent),
            },
        )
    }));
    let compiled = CompiledModel::compile_selected(
        "geometry-boundary.eqi",
        &source,
        "VectorBoundary",
        &bindings,
    )
    .unwrap();
    let (transaction, model, symbols) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry]).unwrap();
    CompiledLinearBlockForm::derive(&program, symbols.get("body").unwrap(), 3, &BTreeSet::new())
}

#[test]
fn polyhedral_laws_follow_declared_supports_instead_of_cartesian_side_counts() {
    // A tetrahedron has four facets. One, two, or four named selections can
    // describe its complete frontier; none should need six box-side declarations.
    for groups in [
        vec![vec![0, 1, 2, 3]],
        vec![vec![0, 2], vec![1, 3]],
        vec![vec![0], vec![1], vec![2], vec![3]],
    ] {
        for face in [false, true] {
            let form = derive_geometry(&groups, face, false, false).unwrap();
            assert_eq!(form.boundary_laws().len(), 1);
            assert_eq!(
                form.boundary_laws().values().next().unwrap().len(),
                groups.len()
            );
            let omitted = derive_geometry(&groups, face, true, false).unwrap_err();
            assert!(omitted.message().contains("complete boundary law coverage"));
            let duplicate = derive_geometry(&groups, face, false, true).unwrap_err();
            assert!(duplicate.message().contains("duplicate Field boundary law"));
        }
    }
    let absent = derive_geometry(&[], false, false, false).unwrap_err();
    assert!(
        absent
            .message()
            .contains("declared Geometry boundary supports")
    );
}
