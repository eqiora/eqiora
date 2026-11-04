//! Exact selected Geometry partitions drive ordinary compiler family expansion.
use eqiora_compiler::{CompiledModel, StaticBindingValue};
use eqiora_geometry::{CanonicalGeometryV1, NamedEntitySet};

fn cylinder() -> CanonicalGeometryV1 {
    CanonicalGeometryV1::from_circular_hole(
        [[0.0, 2.2], [0.0, 0.41]],
        [0.2, 0.2],
        0.05,
        vec![
            NamedEntitySet::new("body", 2, vec![0]),
            NamedEntitySet::new("inlet", 1, vec![0]),
            NamedEntitySet::new("outlet", 1, vec![1]),
            NamedEntitySet::new("walls", 1, vec![2, 3]),
            NamedEntitySet::new("hole", 1, vec![4]),
        ],
    )
    .unwrap()
}
const SOURCE: &str = "model Selected(support body:volume(ambient_dimension=2),support exterior:complete_exterior(parent=body)){variable value:m on body;relation interior on body{value-value=0;} relation law[face in exterior] on face{trace(value)-trace(value)=0;}}";
fn region(geometry: &CanonicalGeometryV1) -> StaticBindingValue<'_> {
    StaticBindingValue::GeometrySupport {
        geometry,
        selection: geometry.entity_set("body").unwrap(),
        parent: None,
    }
}
#[test]
fn curved_and_grouped_exterior_parts_expand_without_fabricated_axes() {
    let geometry = cylinder();
    let mut prior = None;
    for names in [
        ["inlet", "outlet", "walls", "hole"],
        ["hole", "walls", "outlet", "inlet"],
    ] {
        let members = names.map(|name| geometry.entity_set(name).unwrap());
        let bindings = [
            ("body", region(&geometry)),
            (
                "exterior",
                StaticBindingValue::CompleteExterior {
                    geometry: &geometry,
                    parent: geometry.entity_set("body").unwrap(),
                    members: &members,
                },
            ),
        ];
        let direct =
            CompiledModel::compile_selected("partition.eqi", SOURCE, "Selected", &bindings)
                .unwrap_or_else(|errors| panic!("{errors:?}"));
        let native = eqiora_lang::Module::from_document(
            eqiora_lang::parse("partition.eqi", SOURCE)
                .into_document()
                .unwrap(),
        );
        let replay = eqiora_compiler::lower_module(&native, Some("Selected"), &bindings)
            .unwrap_or_else(|errors| panic!("{errors:?}"));
        assert_eq!(direct.transaction().ops(), replay.transaction().ops());
        assert_eq!(direct.symbols(), replay.symbols());
        assert!(direct.symbols().get("law[selection=\"hole\"]").is_some());
        assert!(direct.symbols().get("law[selection=\"walls\"]").is_some());
        assert!(direct.symbols().get("law[selection=\"inlet\"]").is_some());
        assert!(direct.symbols().get("law[selection=\"outlet\"]").is_some());
        if let Some(prior) = prior {
            assert_eq!(direct.transaction().ops(), prior);
        }
        prior = Some(direct.transaction().ops().to_vec());
    }
}
#[test]
fn coverage_and_same_bytes_foreign_catalogs_fail_before_a_witness() {
    let geometry = cylinder();
    let foreign = geometry.clone();
    for members in [
        vec![
            geometry.entity_set("inlet").unwrap(),
            geometry.entity_set("outlet").unwrap(),
            geometry.entity_set("walls").unwrap(),
        ],
        vec![
            geometry.entity_set("hole").unwrap(),
            geometry.entity_set("hole").unwrap(),
        ],
        vec![
            geometry.entity_set("inlet").unwrap(),
            geometry.entity_set("outlet").unwrap(),
            geometry.entity_set("walls").unwrap(),
            foreign.entity_set("hole").unwrap(),
        ],
    ] {
        let bindings = [
            ("body", region(&geometry)),
            (
                "exterior",
                StaticBindingValue::CompleteExterior {
                    geometry: &geometry,
                    parent: geometry.entity_set("body").unwrap(),
                    members: &members,
                },
            ),
        ];
        assert!(
            CompiledModel::compile_selected("partition.eqi", SOURCE, "Selected", &bindings)
                .is_err()
        );
    }
}
#[test]
fn singular_geometry_selections_compose_and_forward_a_proved_partition() {
    let geometry = cylinder();
    let source = "component Law(support body:volume(ambient_dimension=2),support exterior:complete_exterior(parent=body)){variable value:m on body;relation law[face in exterior] on face{trace(value)-trace(value)=0;}} component Wrapper(support body:volume(ambient_dimension=2),support exterior:complete_exterior(parent=body)){instance child:Law(body=body,exterior=exterior);} model Selected(support body:volume(ambient_dimension=2),support inlet:boundary(parent=body),support outlet:boundary(parent=body),support walls:boundary(parent=body),support hole:boundary(parent=body)){instance wrapped:Wrapper(body=body,exterior=boundaries(inlet,outlet,walls,hole));}";
    let bindings = ["body", "inlet", "outlet", "walls", "hole"].map(|name| {
        (
            name,
            StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set(name).unwrap(),
                parent: (name != "body").then(|| geometry.entity_set("body").unwrap()),
            },
        )
    });
    let compiled = CompiledModel::compile_selected("composed.eqi", source, "Selected", &bindings)
        .unwrap_or_else(|errors| panic!("{errors:?}"));
    assert!(
        compiled
            .symbols()
            .get("wrapped.child.law[selection=\"hole\"]")
            .is_some()
    );
    let incomplete = source.replace(
        "boundaries(inlet,outlet,walls,hole)",
        "boundaries(inlet,outlet,walls)",
    );
    assert!(
        CompiledModel::compile_selected("composed.eqi", &incomplete, "Selected", &bindings)
            .is_err()
    );
    // A same-bytes foreign singular selection must not be laundered through
    // snapshot cloning or name lookup before original-reference validation.
    let foreign = geometry.clone();
    let mut invalid = bindings;
    invalid[4].1 = StaticBindingValue::GeometrySupport {
        geometry: &geometry,
        selection: foreign.entity_set("hole").unwrap(),
        parent: Some(geometry.entity_set("body").unwrap()),
    };
    assert!(CompiledModel::compile_selected("composed.eqi", source, "Selected", &invalid).is_err());
}
#[test]
fn component_owned_cartesian_domains_remain_unsupported_in_source_and_native() {
    let source = "component Law(support body:volume(ambient_dimension=2),support exterior:complete_exterior(parent=body)){variable value:m on body;relation law[b in exterior] on b{trace(value)-trace(value)=0;}} component Wrapper(){domain body=box(0,1,0,1);domain left=boundary(body,axis=0,side=lower);domain right=boundary(body,axis=0,side=upper);domain bottom=boundary(body,axis=1,side=lower);domain top=boundary(body,axis=1,side=upper);instance child:Law(body=body,exterior=boundaries(left,right,bottom,top));} model Main(){instance first:Wrapper();instance second:Wrapper();}";
    let errors = eqiora_compiler::compile("local-domains.eqi", source).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("expected parameter, port"))
    );
    let model = eqiora_lang::parse("local.eqi", "model Local(){domain body=box(0,1,0,1);}")
        .into_document()
        .unwrap();
    let eqiora_lang::Item::Domain(domain) = &model.models()[0].items()[0] else {
        unreachable!()
    };
    let error = eqiora_lang::SourceAstFactory::component(
        eqiora_lang::VisibilitySyntax::Private,
        "Wrapper",
        vec![],
        vec![eqiora_lang::ComponentItem::Domain(domain.clone())],
        eqiora_lang::TextRange::default(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("exact coordinate product"));
}

#[test]
fn selected_partition_failures_keep_language_type_code_and_support_source_location() {
    let geometry = cylinder();
    for (names, message) in [
        (vec!["inlet", "outlet", "walls"], "omits an entity"),
        (
            vec!["inlet", "outlet", "walls", "hole", "hole"],
            "repeats an entity",
        ),
    ] {
        let members = names
            .into_iter()
            .map(|name| geometry.entity_set(name).unwrap())
            .collect::<Vec<_>>();
        let bindings = [
            ("body", region(&geometry)),
            (
                "exterior",
                StaticBindingValue::CompleteExterior {
                    geometry: &geometry,
                    parent: geometry.entity_set("body").unwrap(),
                    members: &members,
                },
            ),
        ];
        let errors = CompiledModel::compile_selected("located.eqi", SOURCE, "Selected", &bindings)
            .unwrap_err();
        let error = errors
            .iter()
            .find(|error| error.message().contains(message))
            .expect("precise owner cause retained");
        assert_eq!(
            error.code(),
            eqiora_core::diagnostic::codes::LANGUAGE_TYPE_ERROR
        );
        let span = error
            .source_span()
            .expect("selected support declaration location");
        assert_eq!(span.file, "located.eqi");
        assert_eq!(
            span.start as usize,
            SOURCE.find("support exterior").unwrap()
        );
        assert!(span.end > span.start);
    }
}
