//! Formal boundary constructors defer topology and discharge it at the selected occurrence.
use eqiora_compiler::{
    CompilationNamespaceId, ResolvedDependency, ResolvedHierarchyInput, ResolvedSourceUnit,
    StaticBindingValue, analyze_resolved_hierarchy,
};
use eqiora_core::{DimExponents, ScalarDomain, ValueLiteral, ValueType};
use eqiora_geometry::{CanonicalGeometryV1, NamedEntitySet};

fn cylinder() -> CanonicalGeometryV1 {
    CanonicalGeometryV1::from_circular_hole(
        [[0., 2.2], [0., 0.41]],
        [0.2, 0.2],
        0.05,
        vec![
            NamedEntitySet::new("fluid", 2, vec![0]),
            NamedEntitySet::new("inlet", 1, vec![0]),
            NamedEntitySet::new("outlet", 1, vec![1]),
            NamedEntitySet::new("walls", 1, vec![2, 3]),
            NamedEntitySet::new("cylinder", 1, vec![4]),
        ],
    )
    .unwrap()
}
fn real_import(native: bool) -> eqiora_compiler::ValidatedResolvedHierarchy {
    let root = CompilationNamespaceId::new(["formal_root", "1", "root-source"]).unwrap();
    let fluid = CompilationNamespaceId::new(["Eqiora.Fluid", "0.5.0", "fluid-source"]).unwrap();
    let mechanics =
        CompilationNamespaceId::new(["Eqiora.Mechanics.Interfaces", "0.3.0", "mechanics-source"])
            .unwrap();
    let unit = |owner, path, source| {
        if native {
            ResolvedSourceUnit::from_module(
                owner,
                path,
                eqiora_lang::Module::from_document(
                    eqiora_lang::parse(path, source).into_document().unwrap(),
                ),
            )
        } else {
            ResolvedSourceUnit::new(owner, path, source)
        }
        .unwrap()
    };
    let units = vec![
        unit(
            root.clone(),
            "src/main.eqi",
            include_str!("../../../examples/steady-flow-past-cylinder.eqi"),
        ),
        unit(
            fluid.clone(),
            "src/fluid.eqi",
            include_str!("../../eqiora-api/packages/Eqiora.Fluid/src/fluid.eqi"),
        ),
        unit(
            mechanics.clone(),
            "src/interfaces.eqi",
            include_str!(
                "../../eqiora-api/packages/Eqiora.Mechanics.Interfaces/src/interfaces.eqi"
            ),
        ),
    ];
    analyze_resolved_hierarchy(ResolvedHierarchyInput::new(
        root.clone(),
        units,
        vec![
            ResolvedDependency::new(root, fluid.clone()),
            ResolvedDependency::new(fluid, mechanics),
        ],
    ))
    .unwrap()
    .validate_definitions()
    .unwrap_or_else(|errors| panic!("{errors:?}"))
}
fn value(exponents: [i32; 7], value: f64) -> ValueLiteral {
    ValueLiteral::from_real(
        ValueType::scalar(
            ScalarDomain::Real,
            DimExponents::from_integers(exponents).unwrap(),
        )
        .unwrap(),
        value,
    )
    .unwrap()
}
#[test]
fn real_imported_selected_component_composes_formal_boundaries_in_source_and_native() {
    let geometry = cylinder();
    let viscosity = value([1, -1, -1, 0, 0, 0, 0], 0.001);
    let speed = value([0, 1, -1, 0, 0, 0, 0], 0.3);
    let height = value([0, 1, 0, 0, 0, 0, 0], 0.41);
    let mut bindings = ["fluid", "inlet", "outlet", "walls", "cylinder"]
        .map(|name| {
            (
                name,
                StaticBindingValue::GeometrySupport {
                    geometry: &geometry,
                    selection: geometry.entity_set(name).unwrap(),
                    parent: (name != "fluid").then(|| geometry.entity_set("fluid").unwrap()),
                },
            )
        })
        .to_vec();
    bindings.extend([
        ("dynamic_viscosity", StaticBindingValue::Value(&viscosity)),
        ("inlet_speed", StaticBindingValue::Value(&speed)),
        ("channel_height", StaticBindingValue::Value(&height)),
    ]);
    for native in [false, true] {
        let compiled = real_import(native)
            .compile_selected("SteadyFlowPastCylinder", &bindings)
            .unwrap_or_else(|errors| panic!("native={native}: {errors:?}"));
        assert!(
            compiled
                .symbols()
                .iter()
                .any(|(name, _)| name.ends_with(".mechanical[selection=\"cylinder\"]")),
            "{:?}",
            compiled.symbols()
        );
    }
}

fn miniature(members: &str) -> String {
    format!(
        "public component Leaf(support body:volume(ambient_dimension=2),support exterior:complete_exterior(parent=body)){{variable value:m on body;relation law[b in exterior] on b{{trace(value)-trace(value)=0;}}}} public component Inner(support body:volume(ambient_dimension=2),support exterior:complete_exterior(parent=body)){{instance leaf:Leaf(body=body,exterior=exterior);}} public component Root(support body:volume(ambient_dimension=2),support inlet:boundary(parent=body),support outlet:boundary(parent=body),support walls:boundary(parent=body),support cylinder:boundary(parent=body)){{instance child:Inner(body=body,exterior=boundaries({members}));}}"
    )
}
fn checked(
    source: &str,
    native: bool,
) -> Result<eqiora_compiler::ValidatedResolvedHierarchy, Vec<eqiora_core::Diagnostic>> {
    let root = CompilationNamespaceId::new(["formal_root", "1", "root-source"]).unwrap();
    let unit = if native {
        ResolvedSourceUnit::from_module(
            root.clone(),
            "src/main.eqi",
            eqiora_lang::Module::from_document(
                eqiora_lang::parse("formal.eqi", source)
                    .into_document()
                    .unwrap(),
            ),
        )
    } else {
        ResolvedSourceUnit::new(root.clone(), "src/main.eqi", source)
    }
    .unwrap();
    analyze_resolved_hierarchy(ResolvedHierarchyInput::new(root, vec![unit], vec![]))?
        .validate_definitions()
}
fn geometry_bindings(geometry: &CanonicalGeometryV1) -> Vec<(&str, StaticBindingValue<'_>)> {
    ["body", "inlet", "outlet", "walls", "cylinder"]
        .map(|slot| {
            (
                slot,
                StaticBindingValue::GeometrySupport {
                    geometry,
                    selection: geometry
                        .entity_set(if slot == "body" { "fluid" } else { slot })
                        .unwrap(),
                    parent: (slot != "body").then(|| geometry.entity_set("fluid").unwrap()),
                },
            )
        })
        .to_vec()
}
#[test]
fn nested_formal_constructor_is_deferred_then_checked_against_actual_geometry() {
    let geometry = cylinder();
    for native in [false, true] {
        let complete = checked(&miniature("inlet,outlet,walls,cylinder"), native)
            .unwrap_or_else(|errors| panic!("{errors:?}"));
        let model = complete
            .compile_selected("Root", &geometry_bindings(&geometry))
            .unwrap_or_else(|errors| panic!("{errors:?}"));
        // Four authored parts, even though grouped walls contain two entities.
        assert_eq!(
            model
                .symbols()
                .iter()
                .filter(|(name, _)| name.contains(".leaf.law[selection="))
                .count(),
            4
        );
        let incomplete = checked(&miniature("inlet,outlet,walls"), native)
            .expect("definition checking cannot infer the actual frontier");
        let errors = incomplete
            .compile_selected("Root", &geometry_bindings(&geometry))
            .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("omits an entity")),
            "{errors:?}"
        );
    }
}
#[test]
fn abstract_formal_constructor_retains_duplicate_parent_dimension_and_empty_checks() {
    let complete = miniature("inlet,outlet,walls,cylinder");
    let wrong_parent = complete
        .replace(
            "public component Root(support body:",
            "public component Root(support other:volume(ambient_dimension=2),support body:",
        )
        .replace(
            "support inlet:boundary(parent=body)",
            "support inlet:boundary(parent=other)",
        );
    let wrong_dimension = complete.replacen(
        "public component Leaf(support body:volume(ambient_dimension=2)",
        "public component Leaf(support body:volume(ambient_dimension=3)",
        1,
    );
    for native in [false, true] {
        for (source, expected) in [
            (miniature("inlet,inlet,walls,cylinder"), "more than once"),
            (miniature(""), "nonempty"),
            (wrong_parent.clone(), "parent"),
            (wrong_dimension.clone(), "ambient dimension"),
        ] {
            let errors = checked(&source, native).unwrap_err();
            assert!(
                errors
                    .iter()
                    .any(|error| error.message().contains(expected)),
                "{expected}: {errors:?}"
            );
        }
    }
}
