fn binary64_hex(value: f64) -> String {
    assert!(value.is_finite(), "nonfinite exported value");
    assert!(
        value != 0. || !value.is_sign_negative(),
        "negative zero exported value"
    );
    if value == 0. {
        return "0x0.0p+0".to_owned();
    }
    let bits = value.to_bits();
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent == 0 {
        format!("{sign}0x0.{fraction:013x}p-1022")
    } else {
        format!("{sign}0x1.{fraction:013x}p{:+}", exponent - 1023)
    }
}

#[test]
fn native_binary64_projection_preserves_ieee_components() {
    for (value, expected) in [
        (0., "0x0.0p+0"),
        (1., "0x1.0000000000000p+0"),
        (1.5, "0x1.8000000000000p+0"),
        (-2., "-0x1.0000000000000p+1"),
        (f64::from_bits(1), "0x0.0000000000001p-1022"),
        (f64::MIN_POSITIVE, "0x1.0000000000000p-1022"),
    ] {
        assert_eq!(binary64_hex(value), expected);
    }
}

fn selected_field(prepared: &Prepared) -> eqiora::Id<eqiora::entity::kinds::Field> {
    let selected = prepared
        .document
        .aliases()
        .get("definition.displacement")
        .or_else(|| prepared.document.aliases().get("displacement"))
        .unwrap()
        .downcast()
        .unwrap();
    assert!(prepared.document.program().nodes().any(
        |node| matches!(node, eqiora::kernel::KernelNode::Field(value) if value.id() == selected)
    ));
    selected
}

fn selected_observable(
    prepared: &Prepared,
    name: &str,
) -> eqiora::Id<eqiora::entity::kinds::Observable> {
    let selected = prepared
        .document
        .aliases()
        .get(&format!("definition.{name}"))
        .or_else(|| prepared.document.aliases().get(name))
        .unwrap()
        .downcast()
        .unwrap();
    assert!(prepared.document.program().nodes().any(|node|
        matches!(node, eqiora::kernel::KernelNode::Observable(value) if value.id() == selected)));
    selected
}

fn cartesian_block_shape(mesh: &CartesianMesh, value_type: &eqiora::ValueType) -> Vec<usize> {
    let mut shape = (0..mesh.topological_dimension())
        .map(|axis| mesh.axis_coordinates(axis).unwrap().len())
        .collect::<Vec<_>>();
    shape.extend(
        value_type
            .shape()
            .extents()
            .iter()
            .map(|extent| extent.get() as usize),
    );
    shape
}

#[test]
fn cartesian_vector_block_retains_axes_before_components() {
    let mesh = CartesianMesh::uniform(&[[0., 2.], [0., 3.], [0., 4.]], &[2; 3]).unwrap();
    let ty = eqiora::ValueType::shaped(
        eqiora::ScalarDomain::Real,
        eqiora::DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
        eqiora::ValueShape::new([3]).unwrap(),
        eqiora::ValueFrame::SpatialCartesian,
    )
    .unwrap();
    assert_eq!(cartesian_block_shape(&mesh, &ty), [3, 3, 3, 3]);
}

fn native_export(prepared: &Prepared, result: &eqiora_numerics::CommonResult) -> serde_json::Value {
    use eqiora::kernel::KernelNode;
    use std::collections::{BTreeSet, HashMap};
    let spatial = prepared.plan.as_spatial().unwrap();
    assert_eq!(
        result.plan().to_bytes().unwrap(),
        prepared.plan.to_bytes().unwrap()
    );
    assert_eq!(result.field_count(), 1);
    let field = selected_field(prepared);
    let (actual_field, value_type, space) = result.field(0).unwrap();
    assert_eq!(actual_field, field.ulid().to_string());
    assert_eq!(space, spatial.field_space(field).unwrap());
    assert_eq!(value_type.scalar_domain(), eqiora::ScalarDomain::Real);
    assert_eq!(
        value_type
            .shape()
            .extents()
            .iter()
            .map(|v| v.get())
            .collect::<Vec<_>>(),
        [3]
    );
    assert_eq!(value_type.array_rank(), 0);
    assert_eq!(value_type.frame(), eqiora::ValueFrame::SpatialCartesian);
    assert_eq!(
        value_type.dimension(),
        eqiora::DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap()
    );
    assert_eq!(result.field_block_count(0), 1);
    let (association, coefficients, shape) = result.field_block(0, 0).unwrap();
    assert_eq!(association, "vertex");
    let expected_shape = cartesian_block_shape(prepared.mesh.mesh(), value_type);
    assert_eq!(shape, expected_shape);
    assert_eq!(coefficients.len(), 81);
    let entities = spatial.field_coefficient_entities(field).unwrap();
    assert_eq!(entities.len(), 27);
    let mut records = Vec::new();
    let mut retained = BTreeSet::new();
    for (entity, values) in entities.into_iter().zip(coefficients.chunks_exact(3)) {
        assert_eq!(entity.dimension(), 0);
        let closure = prepared.mesh.mesh().entity_vertices(entity).unwrap();
        assert_eq!(closure, [entity]);
        let point = prepared.mesh.mesh().vertex_coordinates(entity).unwrap();
        for (component, &value) in ["x", "y", "z"].into_iter().zip(values) {
            assert!(retained.insert((
                point.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                component
            )));
            records.push(serde_json::json!({"coordinate_m": point.iter().copied().map(binary64_hex).collect::<Vec<_>>(),
                "component": component, "value_m": binary64_hex(value)}));
        }
    }
    let mut expected = BTreeSet::new();
    for x in [0_f64, 1., 2.] {
        for y in [0_f64, 1.5, 3.] {
            for z in [0_f64, 2., 4.] {
                for component in ["x", "y", "z"] {
                    expected.insert((vec![x.to_bits(), y.to_bits(), z.to_bits()], component));
                }
            }
        }
    }
    assert_eq!(retained, expected);
    let mut observed = Vec::new();
    for (index, name) in ["energy", "left_x", "left_y", "left_z"]
        .into_iter()
        .enumerate()
    {
        let observable = selected_observable(prepared, name);
        let domain = prepared
            .document
            .program()
            .nodes()
            .find_map(|node| match node {
                KernelNode::Observable(value) if value.id() == observable => {
                    value.reduction().domain()
                }
                _ => None,
            })
            .unwrap();
        let dimension = if index == 0 { 3 } else { 2 };
        let rule =
            eqiora::meshing::QuadratureRule::tensor_product_gauss_legendre(dimension, 2).unwrap();
        let rules = HashMap::from([(domain, rule)]);
        let observation = result
            .observe(&prepared.model, observable, &rules, None)
            .unwrap();
        assert_eq!(observation.observable(), observable);
        assert_eq!(observation.result_identity(), result.identity());
        assert_eq!(observation.quadratures().len(), 1);
        let actual = observation.quadratures().get(&domain).unwrap();
        assert_eq!(actual.reference_cell().dimension(), dimension);
        assert_eq!(actual.points().len(), 1 << dimension);
        let value = observation.value();
        assert!(value.value_type().shape().is_scalar());
        let units = if index == 0 {
            [1, 2, -2, 0, 0, 0, 0]
        } else {
            [1, 1, -2, 0, 0, 0, 0]
        };
        assert_eq!(
            value.value_type().dimension(),
            eqiora::DimExponents::from_integers(units).unwrap()
        );
        observed.push(binary64_hex(value.real_scalar_value().unwrap().value()));
    }
    serde_json::json!({"identity": {"dimension":3,"family":"Hex","order":"Q1","geometry":"affine-cartesian",
        "cells":[2,2,2],"quadrature":[2,2,2],"units":["m","J","N"]},
        "lineage":{"mesh":spatial.mesh_digest(),"model":prepared.model.digest().unwrap().to_string(),"result":result.identity()},
        "coefficients":records,"energy_J":observed[0],"left_traction_N":&observed[1..]})
}
