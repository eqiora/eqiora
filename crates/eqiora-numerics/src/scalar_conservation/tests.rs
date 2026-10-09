use eqiora_compiler::compile;
use eqiora_graph::{GraphStore, InMemoryGraphStore};

use super::*;

const TRANSIENT: &str = r#"
model ScalarBalance() {
  domain body = box(0, 1);
  domain lower_face = boundary(body, axis = 0, side = lower);
  domain upper_face = boundary(body, axis = 0, side = upper);

  state state: 1 on body;
  parameter capacity: s / m ^ 2 = 2;
  parameter conductivity: 1 = 3;
  parameter source_density: 1 / m ^ 2 = 0.5;
  parameter transfer: 1 / m = 4;
  parameter imposed_flux: 1 / m = 5;

  relation balance on body {
    capacity * derivative(state)
      - div(conductivity * grad(state))
      - source_density = 0;
  }
  relation lower_value on lower_face {
    trace(state) - 1 = 0;
  }
  relation upper_robin on upper_face {
    normal(conductivity * grad(state))
      + transfer * trace(state)
      - imposed_flux = 0;
  }
}
"#;

const COMPOSITE: &str = r#"
public connector ScalarBoundary {
  trace value: 1;
  flux outward_flux: 1 / m;
  shape [];
  frame invariant;
  pairing euclidean_boundary_duality;
  orientation parent_outward;
}

public component ScalarInterface1d(
  support body: volume(ambient_dimension = 1),
  support face: boundary(parent = body),
  variable state: 1 on body,
  parameter coefficient: 1,
  port interface: ScalarBoundary over face
) {

  relation carrier on face {
    trace(state) - interface.value = 0;
    normal(coefficient * grad(state)) - interface.outward_flux = 0;
  }
}

model CompositeBalance() {
  domain left = box(0, 0.5);
  domain left_lower = boundary(left, axis = 0, side = lower);
  domain left_upper = boundary(left, axis = 0, side = upper);
  domain right = box(0.5, 1);
  domain right_lower = boundary(right, axis = 0, side = lower);
  domain right_upper = boundary(right, axis = 0, side = upper);

  variable left_state: 1 on left;
  variable right_state: 1 on right;
  parameter left_coefficient: 1 = 2;
  parameter right_coefficient: 1 = 7;

  relation left_balance on left {
    -div(left_coefficient * grad(left_state)) = 0;
  }
  relation right_balance on right {
    -div(right_coefficient * grad(right_state)) = 0;
  }
  relation left_value on left_lower {
    trace(left_state) = 0;
  }
  relation right_symmetry on right_upper {
    normal(right_coefficient * grad(right_state)) = 0;
  }
  instance left_carrier: ScalarInterface1d(
    body = left,
    face = left_upper,
    state = left_state,
    coefficient = left_coefficient
  );
  instance right_carrier: ScalarInterface1d(
    body = right,
    face = right_lower,
    state = right_state,
    coefficient = right_coefficient
  );

  connect left_carrier.interface, right_carrier.interface;
}
"#;

#[test]
fn projects_transient_storage_source_and_robin_without_physics_names() {
    let program = program(TRANSIENT);
    let descriptor = recognize_scalar_conservation(&program).unwrap();
    assert_eq!(descriptor.regions.len(), 1);
    assert!(descriptor.interfaces.is_empty());
    let region = &descriptor.regions[0];
    assert_eq!(region.dimensions, 1);
    assert_eq!(
        region
            .storage
            .as_ref()
            .unwrap()
            .coefficient
            .constant_value(),
        Some(2.0)
    );
    assert_eq!(region.flux.coefficient.constant_value(), Some(3.0));
    let source = region.source.as_ref().unwrap();
    assert_eq!(source.expression.constant_value(), Some(0.5));
    assert_eq!(
        source.integrated_dimension.exponents()[1],
        (
            source.dimension.exponents()[1].0 + source.dimension.exponents()[1].1,
            source.dimension.exponents()[1].1
        )
    );
    assert!(matches!(
        &region.exterior[&(0, BoundarySide::Lower)].law,
        ScalarExteriorLaw::PrescribedTrace { value, .. }
            if value.constant_value() == Some(1.0)
    ));
    assert!(matches!(
        &region.exterior[&(0, BoundarySide::Upper)].law,
        ScalarExteriorLaw::Robin { trace_coefficient, value, .. }
            if trace_coefficient.constant_value() == Some(4.0)
                && value.constant_value() == Some(5.0)
    ));
    assert_eq!(descriptor.parameters.len(), 5);

    let upper = &region.exterior[&(0, BoundarySide::Upper)];
    let lineage = upper.law.lineage();
    let typed = typed_relation(&program, lineage.relation()).unwrap();
    let expression = typed.expression();
    assert!(matches!(
        expression.node(lineage.operator_expression()),
        Some(ExprNode::NormalComponent { .. })
    ));
    assert!(matches!(
        expression.node(lineage.datum_expression().unwrap()),
        Some(ExprNode::Symbol(SymbolRef::Parameter(_)))
    ));
    assert!(matches!(
        expression.node(lineage.robin_coefficient_expression().unwrap()),
        Some(ExprNode::Symbol(SymbolRef::Parameter(_)))
    ));
    assert!(matches!(
        expression.node(lineage.robin_trace_expression().unwrap()),
        Some(ExprNode::Trace { .. })
    ));
}

#[test]
fn projects_two_materials_through_one_oriented_conserving_interface() {
    let program = program(COMPOSITE);
    let descriptor = recognize_scalar_conservation(&program).unwrap();
    assert_eq!(descriptor.regions.len(), 2);
    assert_eq!(descriptor.interfaces.len(), 1);
    let interface = &descriptor.interfaces[0];
    assert_ne!(interface.sides[0].domain, interface.sides[1].domain);
    assert_eq!(interface.sides[0].axis, interface.sides[1].axis);
    assert_ne!(interface.sides[0].side, interface.sides[1].side);
    for side in &interface.sides {
        let expression = &relation_expression(&program, side.relation).unwrap();
        assert!(expression.roots().contains(&side.trace_relation_root));
        assert!(expression.roots().contains(&side.flux_relation_root));
    }
    let laws = descriptor
        .regions
        .iter()
        .flat_map(|region| region.exterior.values())
        .map(|entry| &entry.law)
        .collect::<Vec<_>>();
    assert!(
        laws.iter()
            .any(|law| matches!(law, ScalarExteriorLaw::PrescribedTrace { .. }))
    );
    assert!(
        laws.iter()
            .any(|law| matches!(law, ScalarExteriorLaw::ZeroOutwardFlux { .. }))
    );
}

#[test]
fn admits_positive_affine_isotropic_flux_and_nonzero_neumann_data() {
    let source = TRANSIENT
            .replace(
                "  parameter conductivity: 1 = 3;",
                "  parameter conductivity: 1 = 3;\n  parameter conductivity_gradient: 1 / m = 1;",
            )
            .replace(
                "conductivity * grad(state)",
                "(conductivity + conductivity_gradient * coordinate(0)) * grad(state)",
            )
            .replace(
                "normal((conductivity + conductivity_gradient * coordinate(0)) * grad(state))\n      + transfer * trace(state)\n      - imposed_flux = 0;",
                "normal((conductivity + conductivity_gradient * coordinate(0)) * grad(state))\n      - imposed_flux = 0;",
            );
    let descriptor = recognize_scalar_conservation(&program(&source)).unwrap();
    let region = &descriptor.regions[0];
    assert_eq!(region.flux.coefficient.evaluate(&[0.0]).unwrap(), 3.0);
    assert_eq!(region.flux.coefficient.evaluate(&[1.0]).unwrap(), 4.0);
    assert!(matches!(
        &region.exterior[&(0, BoundarySide::Upper)].law,
        ScalarExteriorLaw::PrescribedOutwardFlux { value, .. }
            if value.constant_value() == Some(5.0)
    ));
}

#[test]
fn whole_equation_reversal_preserves_the_descriptor_meaning() {
    let reversed = TRANSIENT
            .replace(
                "capacity * derivative(state)\n      - div(conductivity * grad(state))\n      - source_density = 0;",
                "div(conductivity * grad(state)) + source_density\n      - capacity * derivative(state) = 0;",
            )
            .replace("trace(state) - 1 = 0;", "1 - trace(state) = 0;")
            .replace(
                "normal(conductivity * grad(state))\n      + transfer * trace(state)\n      - imposed_flux = 0;",
                "imposed_flux - transfer * trace(state)\n      - normal(conductivity * grad(state)) = 0;",
            );
    let direct = recognize_scalar_conservation(&program(TRANSIENT)).unwrap();
    let reversed = recognize_scalar_conservation(&program(&reversed)).unwrap();
    assert_eq!(
        direct.regions[0].field_dimension,
        reversed.regions[0].field_dimension
    );
    assert_eq!(
        direct.regions[0].balance_dimension,
        reversed.regions[0].balance_dimension
    );
    assert_eq!(
        direct.regions[0]
            .storage
            .as_ref()
            .unwrap()
            .coefficient
            .constant_value(),
        reversed.regions[0]
            .storage
            .as_ref()
            .unwrap()
            .coefficient
            .constant_value()
    );
    assert_eq!(
        direct.regions[0].flux.coefficient.constant_value(),
        reversed.regions[0].flux.coefficient.constant_value()
    );
    assert_eq!(
        direct.regions[0]
            .source
            .as_ref()
            .unwrap()
            .expression
            .constant_value(),
        reversed.regions[0]
            .source
            .as_ref()
            .unwrap()
            .expression
            .constant_value()
    );
}

#[test]
fn rejects_nonpositive_coefficients_and_state_dependent_sources() {
    let negative_storage = TRANSIENT.replace("capacity: s / m ^ 2 = 2", "capacity: s / m ^ 2 = -2");
    assert!(recognize_scalar_conservation(&program(&negative_storage)).is_err());
    let negative_flux = TRANSIENT.replace("conductivity: 1 = 3", "conductivity: 1 = -3");
    assert!(recognize_scalar_conservation(&program(&negative_flux)).is_err());
    let state_source = TRANSIENT.replace("- source_density = 0;", "- source_density * state = 0;");
    assert!(recognize_scalar_conservation(&program(&state_source)).is_err());
}

#[test]
fn robin_admission_is_closed_to_positive_finite_constants() {
    for value in ["0", "-1"] {
        let source = TRANSIENT.replace(
            "parameter transfer: 1 / m = 4",
            &format!("parameter transfer: 1 / m = {value}"),
        );
        assert!(recognize_scalar_conservation(&program(&source)).is_err());
    }
    let spatial = TRANSIENT
        .replace(
            "  parameter transfer: 1 / m = 4;",
            "  parameter transfer: 1 / m = 4;\n  parameter transfer_gradient: 1 / m ^ 2 = -8;",
        )
        .replace(
            "transfer * trace(state)",
            "(transfer + transfer_gradient * coordinate(0)) * trace(state)",
        );
    assert!(recognize_scalar_conservation(&program(&spatial)).is_err());

    let nonfinite = TRANSIENT
        .replace(
            "  parameter transfer: 1 / m = 4;",
            "  parameter transfer: 1 / m = 4;\n  parameter zero: 1 = 0;",
        )
        .replace(
            "transfer * trace(state)",
            "(transfer / zero) * trace(state)",
        );
    assert!(recognize_scalar_conservation(&program(&nonfinite)).is_err());
}

#[test]
fn admits_complete_two_and_three_dimensional_closure() {
    for dimensions in [2, 3] {
        let descriptor =
            recognize_scalar_conservation(&program(&cartesian_regions(&[dimensions]))).unwrap();
        let region = &descriptor.regions[0];
        assert_eq!(region.dimensions, dimensions);
        assert_eq!(region.exterior.len(), 2 * dimensions);
    }
}

#[test]
fn exact_external_region_support_reuses_the_same_descriptor_and_fails_closed() {
    let program = program(TRANSIENT);
    let direct = recognize_scalar_conservation(&program).unwrap();
    let region = &direct.regions[0];
    let boundaries = exact_boundaries(&program, region.domain, region.dimensions).unwrap();
    let support =
        ScalarRegionSupport::new(region.domain, region.bounds.clone(), boundaries.clone());
    assert_eq!(
        recognize_scalar_conservation_on_supports(&program, vec![support]).unwrap(),
        direct
    );

    let mut incomplete = boundaries;
    incomplete.remove(&(0, BoundarySide::Lower));
    assert!(
        recognize_scalar_conservation_on_supports(
            &program,
            vec![ScalarRegionSupport::new(
                region.domain,
                region.bounds.clone(),
                incomplete,
            )],
        )
        .is_err()
    );
}

#[test]
fn rejects_unsupported_cartesian_domain_in_mixed_model() {
    let source = cartesian_regions(&[1, 4]);
    assert!(recognize_scalar_conservation(&program(&source)).is_err());
}

#[test]
fn interface_rejects_wrong_constitutive_lineage_and_incomplete_carrier() {
    let wrong_coefficient = COMPOSITE.replace(
        "coefficient = right_coefficient\n  );",
        "coefficient = left_coefficient\n  );",
    );
    assert!(recognize_scalar_conservation(&program(&wrong_coefficient)).is_err());

    let missing_flux = COMPOSITE.replace(
        "    normal(coefficient * grad(state)) - interface.outward_flux = 0;\n",
        "",
    );
    assert!(recognize_scalar_conservation(&program(&missing_flux)).is_err());

    let wrong_carrier_sign = COMPOSITE.replace(
        "normal(coefficient * grad(state)) - interface.outward_flux",
        "normal(coefficient * grad(state)) + interface.outward_flux",
    );
    assert!(recognize_scalar_conservation(&program(&wrong_carrier_sign)).is_err());
}

#[test]
fn rejects_incomplete_or_overlapping_boundary_meaning() {
    let missing = TRANSIENT.replace(
        "  relation lower_value on lower_face {\n    trace(state) - 1 = 0;\n  }\n",
        "",
    );
    assert!(recognize_scalar_conservation(&program(&missing)).is_err());
    let overlap = TRANSIENT.replace(
            "  relation lower_value on lower_face {",
            "  relation duplicate_value on lower_face { trace(state) = 0; }\n  relation lower_value on lower_face {",
        );
    assert!(recognize_scalar_conservation(&program(&overlap)).is_err());
}

#[test]
fn axial_scalar_balance_is_math_equivalent_not_a_thermal_claim() {
    let source = include_str!("../../../../verify/solid/axial-bar/models/axial-bar.eqi");
    let descriptor = recognize_scalar_conservation(&program(source)).unwrap();
    assert_eq!(descriptor.regions.len(), 1);
    assert_eq!(
        descriptor.regions[0].field_dimension,
        DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension")
    );
    assert!(descriptor.regions[0].storage.is_none());
}

fn program(source: &str) -> KernelProgram {
    let mut documents = compile("scalar-conservation.eqi", source).expect("fixture compiles");
    assert_eq!(documents.len(), 1);
    let (transaction, model, _) = documents.remove(0).into_parts();
    let mut store = InMemoryGraphStore::new();
    store
        .commit(transaction)
        .expect("fixture transaction commits");
    KernelProgram::from_snapshot(&store.snapshot(), model).expect("fixture Model validates")
}

fn cartesian_regions(dimensions: &[usize]) -> String {
    let mut source =
        String::from("model CartesianScalar() {\n  \n  parameter coefficient: 1 = 2;\n");
    for (region, dimensions) in dimensions.iter().copied().enumerate() {
        let bounds = (0..dimensions)
            .flat_map(|_| ["0", "1"])
            .collect::<Vec<_>>()
            .join(", ");
        source.push_str(&format!("  domain body_{region} = box({bounds});\n"));
        for axis in 0..dimensions {
            for side in ["lower", "upper"] {
                source.push_str(&format!(
                    "  domain face_{region}_{axis}_{side} = boundary(body_{region}, axis = {axis}, side = {side});\n"
                ));
            }
        }
        source.push_str(&format!(
            "  variable state_{region}: 1 on body_{region};\n  relation balance_{region} on body_{region} {{ -div(coefficient * grad(state_{region})) = 0; }}\n"
        ));
        for axis in 0..dimensions {
            for side in ["lower", "upper"] {
                source.push_str(&format!(
                    "  relation closure_{region}_{axis}_{side} on face_{region}_{axis}_{side} {{ trace(state_{region}) = 0; }}\n"
                ));
            }
        }
    }
    source.push_str("}\n");
    source
}
