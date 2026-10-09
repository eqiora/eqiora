use super::*;
use eqiora_compiler::{CompiledModel, StaticBindingValue};
use eqiora_geometry::NamedEntitySet;

fn fixture(
    groups: &[Vec<usize>],
    duplicate_region: bool,
    permuted: bool,
    extent: f64,
) -> (ModelEnvelope, KernelProgram, AuthenticatedCommonMesh) {
    fixture_typed(groups, duplicate_region, permuted, extent, false)
}

fn fixture_typed(
    groups: &[Vec<usize>],
    duplicate_region: bool,
    permuted: bool,
    extent: f64,
    complex: bool,
) -> (ModelEnvelope, KernelProgram, AuthenticatedCommonMesh) {
    fixture_source(
        groups,
        duplicate_region,
        permuted,
        extent,
        complex,
        |source| source,
    )
}

fn fixture_source(
    groups: &[Vec<usize>],
    duplicate_region: bool,
    permuted: bool,
    extent: f64,
    complex: bool,
    transform: impl FnOnce(String) -> String,
) -> (ModelEnvelope, KernelProgram, AuthenticatedCommonMesh) {
    let mut sets = vec![NamedEntitySet::new("body", 3, vec![0])];
    if duplicate_region {
        sets.push(NamedEntitySet::new("other_body", 3, vec![0]));
    }
    sets.extend(
        groups
            .iter()
            .enumerate()
            .map(|(i, faces)| NamedEntitySet::new(format!("side{i}"), 2, faces.clone())),
    );
    let geometry = CanonicalGeometryV1::from_convex_polyhedra(
        vec![[0., 0., 0.], [extent, 0., 0.], [0., 3., 0.], [0., 0., 4.]],
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
        String::from("public component Flux(support body: volume(ambient_dimension = 3)");
    for i in 0..groups.len() {
        source += &format!(", support side{i}: boundary(parent = body)");
    }
    if duplicate_region {
        source += ", support other: volume(ambient_dimension = 3)";
    }
    source +=
        ") { variable u: vector<1,3> on body; relation balance on body { curl(curl(u)) = 0; }";
    for i in 0..groups.len() {
        source += &format!("relation law{i} on side{i} {{ tangential_trace(-curl(u)) = 0; }}");
    }
    if duplicate_region {
        source += "variable v: vector<1,3> on other; relation other_balance on other { curl(curl(v)) = 0; }";
    }
    source += "}";
    let body = geometry.entity_set("body").unwrap();
    let names = (0..groups.len())
        .map(|i| format!("side{i}"))
        .collect::<Vec<_>>();
    let mut bindings = vec![(
        "body",
        StaticBindingValue::GeometrySupport {
            geometry: &geometry,
            selection: body,
            parent: None,
        },
    )];
    if duplicate_region {
        bindings.push((
            "other",
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set("other_body").unwrap(),
                parent: None,
            },
        ));
    }
    bindings.extend(names.iter().map(|name| {
        (
            name.as_str(),
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set(name).unwrap(),
                parent: Some(body),
            },
        )
    }));
    if complex {
        source = source.replace("vector<1,3>", "vector<complex<1>,3>");
    }
    let source = transform(source);
    let compiled =
        CompiledModel::compile_selected("polyhedral-support.eqi", &source, "Flux", &bindings)
            .unwrap();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry]).unwrap();
    assert_eq!(program.nodes().filter(|node| matches!(node, KernelNode::Domain(domain) if matches!(domain.kind(), DomainKind::GeometryRegion { .. }))).count(), if duplicate_region { 2 } else { 1 });
    let model = ModelEnvelope::from_program(&program).unwrap();
    // Synthetic bounded MSH data exercises import/replay, not provider execution.
    let cell = if permuted { "2 1 4 3" } else { "1 2 3 4" };
    let observation = format!(
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 4 1 4\n3 1 0 4\n1\n2\n3\n4\n0 0 0\n{extent} 0 0\n0 3 0\n0 0 4\n$EndNodes\n$Elements\n1 1 1 1\n3 1 4 1\n1 {cell}\n$EndElements\n"
    );
    let policy = eqiora_artifact::GmshMeshPolicyV1::explicit(1e-12, 0.01, 8, 5.).unwrap();
    let owner =
        AuthenticatedCommonMesh::gmsh_4152(geometry, policy, observation.into_bytes()).unwrap();
    (model, program, owner)
}

#[test]
fn complete_polyhedral_selections_bind_before_numerical_recognition() {
    for groups in [
        vec![vec![0, 1, 2, 3]],
        vec![vec![0, 2], vec![1, 3]],
        vec![vec![0], vec![1], vec![2], vec![3]],
    ] {
        for permuted in [false, true] {
            let (model, program, owner) = fixture(&groups, false, permuted, 2.);
            assert!(
                owner
                    .resources
                    .cartesian_cells()
                    .unwrap_err()
                    .message()
                    .contains("axis cell counts")
            );
            bind_model_support(&program, &owner.resources).unwrap();
            let replay = replay_program(&model, owner.geometry().unwrap()).unwrap();
            bind_model_support(&replay, &owner.resources).unwrap();
            let domain = program
                .nodes()
                .find_map(|node| match node {
                    KernelNode::Domain(domain)
                        if matches!(domain.kind(), DomainKind::GeometryRegion { .. }) =>
                    {
                        Some(domain.id().erase())
                    }
                    _ => None,
                })
                .unwrap();
            let form = crate::form_compiler::linear::CompiledLinearBlockForm::<f64>::derive(
                &program,
                domain,
                3,
                &BTreeSet::new(),
            )
            .unwrap();
            assert_eq!(
                form.boundary_laws().values().next().unwrap().len(),
                groups.len()
            );
        }
    }
}

#[test]
fn normal_model_admission_rejects_missing_overlapping_and_duplicated_supports() {
    for (groups, duplicate, diagnostic) in [
        (
            vec![vec![0, 1, 2]],
            false,
            "completely cover the physical frontier",
        ),
        (
            vec![vec![0, 1, 2], vec![2, 3]],
            false,
            "selections overlap on a mesh facet",
        ),
        (
            vec![vec![0, 1, 2, 3]],
            true,
            "assigns a mesh cell more than once",
        ),
    ] {
        let (model, program, owner) = fixture(&groups, duplicate, false, 2.);
        assert!(
            bind_model_support(&program, &owner.resources)
                .unwrap_err()
                .message()
                .contains(diagnostic)
        );
        let error = RecognizedNativeAdmission::recognize(&model, owner).unwrap_err();
        assert!(error.message().contains(diagnostic), "{}", error.message());
    }
    let (_, program, _) = fixture(&[vec![0, 1, 2, 3]], false, false, 2.);
    let (_, _, foreign) = fixture(&[vec![0, 1, 2, 3]], false, false, 5.);
    let error = bind_model_support(&program, &foreign.resources).unwrap_err();
    assert!(error.message().contains("foreign Geometry"));
}

#[test]
fn native_recognition_retains_real_and_complex_polyhedral_linear_equations() {
    for complex in [false, true] {
        let (model, _, owner) = fixture_typed(&[vec![0, 1, 2, 3]], false, true, 2., complex);
        let recognized = RecognizedNativeAdmission::recognize(&model, owner).unwrap();
        for spatial in [
            NativeSpatialPolicy::LinearFiniteElement(Space::continuous_lagrange(
                std::num::NonZeroU16::MIN,
            )),
            NativeSpatialPolicy::ScalarTpfa(None),
        ] {
            let error = validate_resources(spatial, &recognized.resources).unwrap_err();
            assert!(error.message().contains("authenticated common Mesh kind"));
        }
        let fields = match recognized.recognized {
            RecognizedNativeModel::Scalar(equations) if !complex => {
                assert!(equations.single().unwrap().cartesian().is_err());
                equations.fields()
            }
            RecognizedNativeModel::ComplexScalar(equations) if complex => {
                assert!(equations.single().unwrap().cartesian().is_err());
                equations.fields()
            }
            _ => panic!("wrong scalar domain or equation family"),
        };
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].1.shape().component_count(), Some(3));
        assert_eq!(
            fields[0].1.scalar_domain(),
            if complex {
                eqiora_core::ScalarDomain::Complex
            } else {
                eqiora_core::ScalarDomain::Real
            }
        );
    }
}

mod execution;
