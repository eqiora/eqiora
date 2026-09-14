//! Admission of selected Model or Component signature bindings.

use eqiora_compiler::{CompiledModel, StaticBindingValue};
use eqiora_core::Diagnostic;

use crate::ModelDocument;

impl ModelDocument {
    /// Compile one selected Model or Component with explicit static bindings.
    ///
    /// The target signature determines each binding's kind. Geometry selections
    /// retain their exact canonical authority; nominal clocks retain their identity.
    /// All required static inputs must be satisfied before a Model is returned.
    ///
    /// # Errors
    /// Returns source, binding, semantic-admission, or artifact diagnostics.
    pub fn compile_selected(
        filename: &str,
        source: &str,
        entry: &str,
        bindings: &[(&str, StaticBindingValue<'_>)],
    ) -> Result<Self, Vec<Diagnostic>> {
        let compiled = CompiledModel::compile_selected(filename, source, entry, bindings)?;
        Self::accept_bound_compiled(compiled, bindings)
    }

    pub(crate) fn accept_bound_compiled(
        compiled: CompiledModel,
        bindings: &[(&str, StaticBindingValue<'_>)],
    ) -> Result<Self, Vec<Diagnostic>> {
        let mut geometries = Vec::new();
        for (_, binding) in bindings {
            if let StaticBindingValue::GeometrySupport { geometry, .. }
            | StaticBindingValue::CompleteExterior { geometry, .. } = binding
                && !geometries.contains(geometry)
            {
                geometries.push(*geometry);
            }
        }
        Self::accept_compiled_with_geometry(compiled, &geometries)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use eqiora_core::{DimExponents, DynQuantity};
    use eqiora_geometry::{
        EDGE_DIMENSION, FACE_DIMENSION, GeometryGraph, PlanarFace, PlanarRegion,
    };
    use eqiora_schema::kernel::{ExprNode, KernelNode, SymbolRef};

    use eqiora_core::ValueLiteral;
    use eqiora_geometry::{CanonicalGeometryV1, NamedEntitySet};

    fn compile_geometry_fixture(
        filename: &str,
        source: &str,
        geometry: &CanonicalGeometryV1,
        entry: &str,
        parameters: &[(&str, eqiora_lang::Expr)],
    ) -> Result<ModelDocument, Vec<Diagnostic>> {
        let fluid = geometry.entity_set("fluid").unwrap();
        let mut bindings = vec![(
            "fluid",
            StaticBindingValue::GeometrySupport {
                geometry,
                selection: fluid,
                parent: None,
            },
        )];
        let boundaries: &[&str] = if entry == "ScalarDiffusion" {
            &["inlet"]
        } else {
            &["inlet", "outlet", "walls", "cylinder"]
        };
        for &name in boundaries {
            bindings.push((
                name,
                StaticBindingValue::GeometrySupport {
                    geometry,
                    selection: geometry.entity_set(name).unwrap(),
                    parent: Some(fluid),
                },
            ));
        }
        bindings.extend(
            parameters
                .iter()
                .map(|(name, value)| (*name, StaticBindingValue::Expression(value))),
        );
        ModelDocument::compile_selected(filename, source, entry, &bindings)
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn compile_bound_fixture(
        filename: &str,
        source: &str,
        geometry: &CanonicalGeometryV1,
        entry: &str,
        supports: &[(&str, &NamedEntitySet, Option<(&str, &NamedEntitySet)>)],
        parameters: &[(&str, ValueLiteral)],
    ) -> Result<ModelDocument, Vec<Diagnostic>> {
        let mut bindings = supports
            .iter()
            .map(|(name, selection, parent)| {
                (
                    *name,
                    StaticBindingValue::GeometrySupport {
                        geometry,
                        selection,
                        parent: parent.map(|(_, selection)| selection),
                    },
                )
            })
            .collect::<Vec<_>>();
        bindings.extend(
            parameters
                .iter()
                .map(|(name, value)| (*name, StaticBindingValue::Value(value))),
        );
        ModelDocument::compile_selected(filename, source, entry, &bindings)
    }

    const SOURCE: &str = r#"
public component FluidBoundaryLaw(
  support fluid: volume(ambient_dimension = 2), support inlet: boundary(parent = fluid), support outlet: boundary(parent = fluid), support walls: boundary(parent = fluid), support cylinder: boundary(parent = fluid),
  parameter value: 1
) {
  variable state: 1 on fluid;
  relation volume_law on fluid { state - value = 0; }
  relation inlet_law on inlet { trace(state) = 0; }
  relation outlet_law on outlet { trace(state) = 0; }
  relation walls_law on walls { trace(state) = 0; }
  relation cylinder_law on cylinder { trace(state) = 0; }
}
"#;

    const SCALAR_PRIMAL_SOURCE: &str = r#"
public component ScalarDiffusion(
  support fluid: volume(ambient_dimension = 2),
  support inlet: boundary(parent = fluid),
  parameter diffusion: 1,
  parameter wave_number: 1 / m,
  parameter source_scale: 1 / m ^ 2
) {
  variable potential: 1 on fluid;
  relation balance on fluid {
    -div(diffusion * grad(potential))
      = source_scale * math.sin(wave_number * coordinate(0));
  }
  form weak for balance { test w: 1 for potential zero_on inlet;
    integrate(fluid, dot(grad(w), diffusion * grad(potential)))
      = integrate(
          fluid,
          w * source_scale * math.sin(wave_number * coordinate(0))
        );
  }
}
"#;

    const STEADY_FLOW_PAST_CYLINDER_COMPONENT: &str = r#"
public component SteadyFlowPastCylinder(
  support fluid: volume(ambient_dimension = 2), support inlet: boundary(parent = fluid), support outlet: boundary(parent = fluid), support walls: boundary(parent = fluid), support cylinder: boundary(parent = fluid),
  parameter dynamic_viscosity: kg / (m * s),
  parameter zero_pressure: kg / (m * s ^ 2),
  parameter inlet_speed: m / s,
  parameter channel_height: m
) {

  variable velocity: vector<m / s, 2> on fluid;
  variable pressure: kg / (m * s ^ 2) on fluid;
  variable force_potential: kg / (m * s ^ 2) on fluid;
  variable inlet_profile: m / s on fluid;

  relation force_definition on fluid {
    force_potential - zero_pressure = 0;
  }
  relation inlet_profile_definition on fluid {
    inlet_profile
      - 4 * inlet_speed * coordinate(1) * (channel_height - coordinate(1))
        / channel_height ^ 2 = 0;
  }
  relation momentum on fluid {
    -div(
      2 * dynamic_viscosity * symmetric_part(grad(velocity))
      - isotropic_lift(pressure)
    ) - grad(force_potential) = 0;
  }
  relation incompressibility on fluid {
    div(velocity) = 0;
  }

  relation inlet_velocity on inlet {
    trace(velocity) + normal(isotropic_lift(inlet_profile)) = 0;
  }
  relation outlet_traction on outlet {
    normal(
      2 * dynamic_viscosity * symmetric_part(grad(velocity))
      - isotropic_lift(pressure)
    ) = 0;
  }
  relation wall_velocity on walls { trace(velocity) = 0; }
  relation cylinder_velocity on cylinder { trace(velocity) = 0; }
}
"#;

    fn fixture_geometry() -> CanonicalGeometryV1 {
        CanonicalGeometryV1::from_circular_hole_named_roles(
            [[0.0, 2.2], [0.0, 0.41]],
            [0.2, 0.2],
            0.05,
            1.0e-12,
            "fluid",
            "inlet",
            "outlet",
            "walls",
            "walls",
            "cylinder",
        )
        .expect("bounded common Geometry")
    }

    fn fixture_geometry_v2() -> CanonicalGeometryV1 {
        let owner = GeometryGraph::new();
        let predecessor = owner
            .rectangle_extrusion((0.0, 2.2), (0.0, 0.41), 0.0, 1.0, 1.0e-10)
            .unwrap();
        let graph = owner
            .circular_through_cut(&predecessor, [0.2, 0.2], 0.05, 1.0e-10)
            .unwrap();
        let end_cap = graph.face_handle("end-cap").unwrap();
        let x_lower = graph.face_handle("profile-x-lower").unwrap();
        let x_upper = graph.face_handle("profile-x-upper").unwrap();
        let y_lower = graph.face_handle("profile-y-lower").unwrap();
        let y_upper = graph.face_handle("profile-y-upper").unwrap();
        let cut_wall = graph.face_handle("cut-wall").unwrap();
        owner
            .build_solid_geometry(
                &graph,
                &BTreeMap::from([
                    ("fluid".to_owned(), vec![end_cap]),
                    ("inlet".to_owned(), vec![x_lower]),
                    ("outlet".to_owned(), vec![x_upper]),
                    ("walls".to_owned(), vec![y_lower, y_upper]),
                    ("cylinder".to_owned(), vec![cut_wall]),
                ]),
            )
            .unwrap()
    }

    #[test]
    fn fresh_geometry_compile_retains_typed_form_without_changing_model_artifact() {
        let geometry = fixture_geometry();
        let parameters = [
            (
                "diffusion",
                eqiora_lang::DraftExpression::constant(
                    eqiora_lang::DecimalLiteral::from_f64(1.0).unwrap(),
                )
                .source_ast(|_| None, |_| None)
                .expect("numeric fixture"),
            ),
            (
                "wave_number",
                eqiora_lang::DraftExpression::constant(
                    eqiora_lang::DecimalLiteral::from_f64(2.0).unwrap(),
                )
                .source_ast(|_| None, |_| None)
                .expect("numeric fixture"),
            ),
            (
                "source_scale",
                eqiora_lang::DraftExpression::constant(
                    eqiora_lang::DecimalLiteral::from_f64(2.0).unwrap(),
                )
                .source_ast(|_| None, |_| None)
                .expect("numeric fixture"),
            ),
        ];
        let with_form = compile_geometry_fixture(
            "scalar-primal.eqi",
            SCALAR_PRIMAL_SOURCE,
            &geometry,
            "ScalarDiffusion",
            &parameters,
        )
        .unwrap();
        let without_form_source = format!(
            "{}}}\n",
            SCALAR_PRIMAL_SOURCE.split_once("  form weak").unwrap().0
        );
        let without_form = compile_geometry_fixture(
            "scalar-primal.eqi",
            &without_form_source,
            &geometry,
            "ScalarDiffusion",
            &parameters,
        )
        .unwrap();

        let forms = with_form.authored_formulations().collect::<Vec<_>>();
        let [form] = forms.as_slice() else {
            panic!("fresh compilation must retain exactly one typed form")
        };
        assert_eq!(form.domain(), with_form.domain_ref("fluid").unwrap().id());
        let local_alias = |name: &str| {
            let suffix = format!(".{name}");
            with_form
                .aliases()
                .iter()
                .find_map(|(candidate, id)| candidate.ends_with(&suffix).then_some(*id))
                .unwrap()
        };
        assert_eq!(
            form.trial(),
            local_alias("potential")
                .downcast::<eqiora_core::entity::kinds::Field>()
                .unwrap()
        );
        assert_eq!(
            form.relation(),
            local_alias("balance")
                .downcast::<eqiora_core::entity::kinds::Relation>()
                .unwrap()
        );
        assert_eq!(without_form.authored_formulations().len(), 0);
        assert_eq!(
            with_form.canonical_json().unwrap(),
            without_form.canonical_json().unwrap()
        );
        assert_eq!(with_form.digest().unwrap(), without_form.digest().unwrap());
        assert_eq!(
            with_form.structural_fingerprint().unwrap(),
            without_form.structural_fingerprint().unwrap()
        );
    }

    #[test]
    fn scalar_primal_form_dimension_mismatch_fails_closed() {
        let geometry = fixture_geometry();
        let invalid = SCALAR_PRIMAL_SOURCE.replace("w * source_scale", "w * diffusion");
        let diagnostics = compile_geometry_fixture(
            "invalid-primal.eqi",
            &invalid,
            &geometry,
            "ScalarDiffusion",
            &[
                (
                    "diffusion",
                    eqiora_lang::DraftExpression::constant(
                        eqiora_lang::DecimalLiteral::from_f64(1.0).unwrap(),
                    )
                    .source_ast(|_| None, |_| None)
                    .expect("numeric fixture"),
                ),
                (
                    "wave_number",
                    eqiora_lang::DraftExpression::constant(
                        eqiora_lang::DecimalLiteral::from_f64(2.0).unwrap(),
                    )
                    .source_ast(|_| None, |_| None)
                    .expect("numeric fixture"),
                ),
                (
                    "source_scale",
                    eqiora_lang::DraftExpression::constant(
                        eqiora_lang::DecimalLiteral::from_f64(2.0).unwrap(),
                    )
                    .source_ast(|_| None, |_| None)
                    .expect("numeric fixture"),
                ),
            ],
        )
        .unwrap_err();
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic
                .message()
                .contains("equality sides must have identical dimension")
        }));
    }

    #[test]
    fn scalar_primal_form_role_contraction_and_operator_errors_fail_closed() {
        let geometry = fixture_geometry();
        let parameters = [
            (
                "diffusion",
                eqiora_lang::DraftExpression::constant(
                    eqiora_lang::DecimalLiteral::from_f64(1.0).unwrap(),
                )
                .source_ast(|_| None, |_| None)
                .expect("numeric fixture"),
            ),
            (
                "wave_number",
                eqiora_lang::DraftExpression::constant(
                    eqiora_lang::DecimalLiteral::from_f64(2.0).unwrap(),
                )
                .source_ast(|_| None, |_| None)
                .expect("numeric fixture"),
            ),
            (
                "source_scale",
                eqiora_lang::DraftExpression::constant(
                    eqiora_lang::DecimalLiteral::from_f64(2.0).unwrap(),
                )
                .source_ast(|_| None, |_| None)
                .expect("numeric fixture"),
            ),
        ];
        let invalid = [
            (
                SCALAR_PRIMAL_SOURCE.replace("zero_on inlet", "zero_on fluid"),
                "zero_on requires a boundary",
            ),
            (
                SCALAR_PRIMAL_SOURCE.replace("zero_on inlet", "zero_on inlet, inlet"),
                "repeats an exact boundary",
            ),
            (
                SCALAR_PRIMAL_SOURCE.replace("test w:", "test potential:"),
                "must not shadow",
            ),
            (
                SCALAR_PRIMAL_SOURCE.replace("for balance", "for missing"),
                "unknown Formulation symbol `missing`",
            ),
            (
                SCALAR_PRIMAL_SOURCE.replace("for potential zero_on", "for diffusion zero_on"),
                "test argument is not a Field",
            ),
            (
                SCALAR_PRIMAL_SOURCE.replace(
                    "dot(grad(w), diffusion * grad(potential))",
                    "dot(w, diffusion * grad(potential))",
                ),
                "dot requires equal non-scalar vector shapes",
            ),
            (
                SCALAR_PRIMAL_SOURCE.replacen("integrate(fluid", "integrate(missing", 1),
                "unknown Formulation symbol `missing`",
            ),
            (
                SCALAR_PRIMAL_SOURCE.replace(
                    "w * source_scale * math.sin",
                    "div(w) * source_scale * math.sin",
                ),
                "unsupported scalar-primal operator `div`",
            ),
        ];
        for (source, expected) in invalid {
            let diagnostics = compile_geometry_fixture(
                "invalid-primal.eqi",
                &source,
                &geometry,
                "ScalarDiffusion",
                &parameters,
            )
            .unwrap_err();
            assert!(
                diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.message().contains(expected)),
                "missing {expected:?} in {diagnostics:?}",
            );
        }

        let ordinary_source = format!(
            "{SCALAR_PRIMAL_SOURCE}\nmodel root() {{ variable x: 1; relation hold {{ x = 0; }} }}\n"
        );
        let diagnostics = ModelDocument::compile("unsupported.eqi", &ordinary_source)
            .expect_err("ordinary Model compilation cannot discard authored forms");
        assert!(
            diagnostics.iter().any(|diagnostic| {
                diagnostic
                    .message()
                    .contains("require fresh external-Geometry component compilation")
            }),
            "unexpected diagnostics: {diagnostics:?}",
        );
    }

    fn cylinder_parameters() -> [(&'static str, ValueLiteral); 4] {
        [
            (
                "dynamic_viscosity",
                DynQuantity::new(
                    1.0e-3,
                    DimExponents::from_integers([1, -1, -1, 0, 0, 0, 0])
                        .expect("bounded dimension"),
                ),
            ),
            (
                "zero_pressure",
                DynQuantity::new(
                    0.0,
                    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0])
                        .expect("bounded dimension"),
                ),
            ),
            (
                "inlet_speed",
                DynQuantity::new(
                    0.3,
                    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension"),
                ),
            ),
            (
                "channel_height",
                DynQuantity::new(
                    0.41,
                    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension"),
                ),
            ),
        ]
        .map(|(name, value)| (name, ValueLiteral::try_from(value).unwrap()))
    }

    fn compile_cylinder(
        geometry: &CanonicalGeometryV1,
        parameters: &[(&str, ValueLiteral)],
    ) -> ModelDocument {
        let fluid = geometry.entity_set("fluid").unwrap();
        let supports = [
            ("fluid", fluid, None),
            (
                "inlet",
                geometry.entity_set("inlet").unwrap(),
                Some(("fluid", fluid)),
            ),
            (
                "outlet",
                geometry.entity_set("outlet").unwrap(),
                Some(("fluid", fluid)),
            ),
            (
                "walls",
                geometry.entity_set("walls").unwrap(),
                Some(("fluid", fluid)),
            ),
            (
                "cylinder",
                geometry.entity_set("cylinder").unwrap(),
                Some(("fluid", fluid)),
            ),
        ];
        compile_bound_fixture(
            "steady-flow-past-cylinder.eqi",
            STEADY_FLOW_PAST_CYLINDER_COMPONENT,
            geometry,
            "SteadyFlowPastCylinder",
            &supports,
            parameters,
        )
        .expect("Geometry v2 and actual Component close one common Model")
    }

    #[test]
    fn construction_named_geometry_v2_compiles_the_actual_cylinder_component() {
        let geometry = fixture_geometry_v2();
        let parameters = cylinder_parameters();
        let document = compile_cylinder(&geometry, &parameters);

        assert_eq!(
            document.geometry_authority.as_slice(),
            std::slice::from_ref(&geometry)
        );
        let bytes = document.canonical_json().expect("authority-backed replay");
        assert!(!bytes.is_empty());
        assert_eq!(
            document.replay_with_retained_geometry().unwrap(),
            *document.program()
        );
        let mut parameter_ids = BTreeSet::new();
        for (name, expected) in parameters {
            let parameter = document
                .aliases()
                .get(name)
                .copied()
                .unwrap_or_else(|| panic!("missing root Parameter `{name}`"));
            assert_eq!(
                document.aliases().get(&format!("definition.{name}")),
                Some(&parameter),
                "Component slot must retain the root Parameter identity",
            );
            let Some(KernelNode::Parameter(definition)) = document.program().node(parameter) else {
                panic!("`{name}` does not resolve to a Parameter")
            };
            assert_eq!(definition.value(), &expected);
            parameter_ids.insert(parameter);
        }
        assert_eq!(parameter_ids.len(), 4);
        let parameter_references = document
            .program()
            .nodes()
            .filter_map(|node| match node {
                KernelNode::Relation(relation) => Some(relation),
                _ => None,
            })
            .flat_map(|relation| relation.expression().nodes())
            .filter_map(|node| match node {
                ExprNode::Symbol(SymbolRef::Parameter(parameter)) => Some(parameter.erase()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(parameter_references.len(), 6);
        assert!(
            parameter_references
                .iter()
                .all(|parameter| parameter_ids.contains(parameter))
        );

        let mut reordered_parameters = cylinder_parameters();
        reordered_parameters.reverse();
        let reordered = compile_cylinder(&geometry, &reordered_parameters);
        assert!(document.structurally_equivalent(&reordered).unwrap());
        let mut changed_parameters = cylinder_parameters();
        changed_parameters[0].1 =
            ValueLiteral::from_real(changed_parameters[0].1.value_type().clone(), 2.0e-3).unwrap();
        let changed = compile_cylinder(&geometry, &changed_parameters);
        assert!(!document.structurally_equivalent(&changed).unwrap());

        assert!(
            ModelDocument::replay(&bytes)
                .unwrap_err()
                .iter()
                .any(|diagnostic| diagnostic.message().contains("requires artifact admission")),
            "resource-free replay must not fabricate Geometry authority",
        );
        for name in [
            "fluid",
            "inlet",
            "outlet",
            "walls",
            "cylinder",
            "definition.velocity",
            "definition.pressure",
        ] {
            assert!(document.aliases().contains_key(name), "missing `{name}`");
        }
    }

    #[test]
    fn typed_external_occurrence_returns_the_common_immutable_model_document() {
        let geometry = fixture_geometry();
        let fluid = geometry.entity_set("fluid").unwrap();
        let supports = [
            ("fluid", fluid, None),
            (
                "inlet",
                geometry.entity_set("inlet").unwrap(),
                Some(("fluid", fluid)),
            ),
            (
                "outlet",
                geometry.entity_set("outlet").unwrap(),
                Some(("fluid", fluid)),
            ),
            (
                "walls",
                geometry.entity_set("walls").unwrap(),
                Some(("fluid", fluid)),
            ),
            (
                "cylinder",
                geometry.entity_set("cylinder").unwrap(),
                Some(("fluid", fluid)),
            ),
        ];
        let document = compile_bound_fixture(
            "fluid-boundary.eqi",
            SOURCE,
            &geometry,
            "FluidBoundaryLaw",
            &supports,
            &[(
                "value",
                ValueLiteral::try_from(DynQuantity::new(2.0, DimExponents::DIMENSIONLESS)).unwrap(),
            )],
        )
        .expect("external occurrence reaches common Model admission");

        assert!(
            !document
                .canonical_json()
                .expect("canonical Model")
                .is_empty()
        );
        for name in ["fluid", "inlet", "outlet", "walls", "cylinder"] {
            assert!(document.aliases().contains_key(name), "missing `{name}`");
        }
    }

    #[test]
    fn explicit_signature_bindings_preserve_diagnostic_filename_independence() {
        let geometry = fixture_geometry();
        let automatic = compile_geometry_fixture(
            "fluid-boundary.eqi",
            SOURCE,
            &geometry,
            "FluidBoundaryLaw",
            &[(
                "value",
                eqiora_lang::DraftExpression::constant(
                    eqiora_lang::DecimalLiteral::from_f64(2.0).unwrap(),
                )
                .source_ast(|_| None, |_| None)
                .expect("numeric fixture"),
            )],
        )
        .expect("explicit public Component closes");
        let explicit = compile_geometry_fixture(
            "renamed-for-diagnostics.eqi",
            SOURCE,
            &geometry,
            "FluidBoundaryLaw",
            &[(
                "value",
                eqiora_lang::DraftExpression::constant(
                    eqiora_lang::DecimalLiteral::from_f64(2.0).unwrap(),
                )
                .source_ast(|_| None, |_| None)
                .expect("numeric fixture"),
            )],
        )
        .expect("explicit public Component closes identically");
        assert_eq!(automatic.digest().unwrap(), explicit.digest().unwrap());
        assert!(automatic.structurally_equivalent(&explicit).unwrap());
        compile_geometry_fixture(
            "negative-is-not-a-compiler-policy.eqi",
            SOURCE,
            &geometry,
            "FluidBoundaryLaw",
            &[(
                "value",
                eqiora_lang::DraftExpression::constant(
                    eqiora_lang::DecimalLiteral::from_f64(-2.0).unwrap(),
                )
                .source_ast(|_| None, |_| None)
                .expect("numeric fixture"),
            )],
        )
        .expect("compiler checks type and finiteness, not application positivity");

        let missing =
            compile_geometry_fixture("missing.eqi", SOURCE, &geometry, "FluidBoundaryLaw", &[])
                .unwrap_err();
        assert!(
            missing
                .iter()
                .any(|error| error.message().contains("value"))
        );
        let extra = compile_geometry_fixture(
            "extra.eqi",
            SOURCE,
            &geometry,
            "FluidBoundaryLaw",
            &[
                (
                    "value",
                    eqiora_lang::DraftExpression::constant(
                        eqiora_lang::DecimalLiteral::from_f64(2.0).unwrap(),
                    )
                    .source_ast(|_| None, |_| None)
                    .expect("numeric fixture"),
                ),
                (
                    "extra",
                    eqiora_lang::DraftExpression::constant(
                        eqiora_lang::DecimalLiteral::from_f64(1.0).unwrap(),
                    )
                    .source_ast(|_| None, |_| None)
                    .expect("numeric fixture"),
                ),
            ],
        )
        .unwrap_err();
        assert!(extra.iter().any(|error| error.message().contains("extra")));

        let ambiguous = format!(
            "{SOURCE}\n{}",
            SOURCE.replace("FluidBoundaryLaw", "OtherLaw")
        );
        let errors = compile_geometry_fixture(
            "ambiguous.eqi",
            &ambiguous,
            &geometry,
            "MissingLaw",
            &[(
                "value",
                eqiora_lang::DraftExpression::constant(
                    eqiora_lang::DecimalLiteral::from_f64(2.0).unwrap(),
                )
                .source_ast(|_| None, |_| None)
                .expect("numeric fixture"),
            )],
        )
        .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("MissingLaw"))
        );
    }

    #[test]
    fn common_geometry_rejects_foreign_kind_and_wrong_parent_bindings() {
        const LAW: &str = "public component Law(support fluid: volume(ambient_dimension = 2)) { variable x: 1 on fluid; relation value on fluid { x = 0; } }";
        let geometry = fixture_geometry();
        let foreign = fixture_geometry();
        let stale = [("fluid", foreign.entity_set("fluid").unwrap(), None)];
        assert!(
            compile_bound_fixture("foreign.eqi", LAW, &geometry, "Law", &stale, &[]).unwrap_err()
                [0]
            .message()
            .contains("foreign or stale")
        );

        let wrong_kind = [("fluid", geometry.entity_set("walls").unwrap(), None)];
        assert!(
            compile_bound_fixture("wrong-kind.eqi", LAW, &geometry, "Law", &wrong_kind, &[])
                .unwrap_err()[0]
                .message()
                .contains("selection dimension 1")
        );

        let topology = PlanarRegion::new(
            vec![
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [3.0, 0.0],
                [3.0, 1.0],
            ],
            vec![
                PlanarFace::new(vec![0, 2, 3, 1], Vec::new()),
                PlanarFace::new(vec![4, 6, 7, 5], Vec::new()),
            ],
            vec![
                NamedEntitySet::new("edge-a", EDGE_DIMENSION, vec![0]),
                NamedEntitySet::new("body-a", FACE_DIMENSION, vec![0]),
                NamedEntitySet::new("body-b", FACE_DIMENSION, vec![1]),
            ],
            1.0e-12,
        )
        .unwrap();
        let geometry = CanonicalGeometryV1::from_region(&topology).unwrap();
        let body = geometry.entity_set("body-b").unwrap();
        let wrong_parent = [
            ("body", body, None),
            (
                "wall",
                geometry.entity_set("edge-a").unwrap(),
                Some(("body", body)),
            ),
        ];
        assert!(
            compile_bound_fixture(
                "wrong-parent.eqi",
                "public component Law(support body: volume(ambient_dimension = 2), support wall: boundary(parent = body)) { variable x: 1 on body; relation value on body { x = 0; } relation boundary on wall { trace(x) = 0; } }",
                &geometry,
                "Law",
                &wrong_parent,
                &[]
            )
            .unwrap_err()[0]
                .message()
                .contains("does not bind exact parent")
        );
    }
}
