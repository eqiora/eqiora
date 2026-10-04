//! Integrals and point probes use the same accepted coefficient reconstruction.
use super::*;
use crate::affine_fem::physical_gradient;
use crate::common_result::CommonResultFieldBlock;
use crate::discrete_space::BasisTabulation;
use eqiora_core::{DynQuantity, ValueType};
use eqiora_sem::EvaluationPoint;

pub(super) fn q1_field(
    value_type: &ValueType,
    owned: &[usize],
    block: &CommonResultFieldBlock,
    vertices: &[MeshEntity],
    basis: &BasisTabulation,
    inverse: &[f64],
    directions: [Option<&[f64]>; 2],
) -> Result<PointField, Diagnostic> {
    let dimension = basis.reference_dimension();
    let components = value_type
        .shape()
        .component_count()
        .ok_or_else(|| invalid("Observable Field component count is unavailable"))?;
    if block.association != CommonFieldAssociation::Vertex
        || block.values.len() != owned.len() * components
        || value_type.array_rank() != 0
        || value_type.scalar_domain() != eqiora_core::ScalarDomain::Real
        || vertices.len() != basis.values().len()
        || inverse.len() != dimension * dimension
        || directions
            .iter()
            .flatten()
            .any(|direction| direction.len() != block.values.len())
    {
        return Err(invalid(
            "Observable Q1 reconstruction requires complete real vertex components",
        ));
    }
    let mut value = vec![0.0; components];
    let mut direction = [vec![0.0; components], vec![0.0; components]];
    let mut gradient_tangent = [
        vec![0.0; components * dimension],
        vec![0.0; components * dimension],
    ];
    let mut gradient = vec![0.0; components * dimension];
    for (local, vertex) in vertices.iter().enumerate() {
        let owned_index = owned
            .binary_search(&vertex.index())
            .map_err(|_| invalid("point lies outside the exact Field cell closure"))?;
        let derivative = physical_gradient(
            basis.gradient(local).expect("Q1 basis gradient exists"),
            inverse,
            dimension,
        );
        for component in 0..components {
            let index = owned_index * components + component;
            let coefficient = block.values[index];
            value[component] += coefficient * basis.values()[local];
            for (axis, derivative) in derivative.iter().enumerate() {
                gradient[component * dimension + axis] += coefficient * derivative;
            }
            for (order, values) in directions.iter().enumerate() {
                let delta = values.map_or(0.0, |values| values[index]);
                direction[order][component] += delta * basis.values()[local];
                for (axis, derivative) in derivative.iter().enumerate() {
                    gradient_tangent[order][component * dimension + axis] += delta * derivative;
                }
            }
        }
    }
    Ok(PointField {
        value: ValueLiteral::new(
            value_type.clone(),
            value.into_iter().map(|value| (value, 0.0)),
        )
        .map_err(|error| invalid(error.to_string()))?,
        gradient,
        tangent: direction,
        gradient_tangent,
    })
}

pub(in crate::common_result::observe) fn sample(
    result: &CommonResult,
    field: Id<kinds::Field>,
    point: &EvaluationPoint,
) -> Result<ValueLiteral, Diagnostic> {
    Ok(reconstruct(result, field, point, false)?.value)
}

pub(in crate::common_result::observe) fn sample_partial(
    result: &CommonResult,
    field: Id<kinds::Field>,
    factor: Id<kinds::Domain>,
    axis: usize,
    point: &EvaluationPoint,
) -> Result<ValueLiteral, Diagnostic> {
    if factor != point.domain() {
        return Err(invalid(
            "point gradient requires the exact physical coordinate factor",
        ));
    }
    let sample = reconstruct(result, field, point, true)?;
    if !sample.value.value_type().shape().is_scalar() {
        return Err(invalid("point gradient requires an admitted scalar Field"));
    }
    let coordinate = point
        .coordinates()
        .find(|(key, _)| *key == (factor.erase(), axis))
        .ok_or_else(|| invalid("point gradient axis is outside the exact support"))?
        .1;
    let dimension = sample
        .value
        .value_type()
        .dimension()
        .div(coordinate.dim())
        .ok_or_else(|| invalid("point gradient units exceed the exact exponent bounds"))?;
    let value = sample
        .gradient
        .get(axis)
        .ok_or_else(|| invalid("point gradient axis is unavailable"))?;
    ValueLiteral::try_from(DynQuantity::new(*value, dimension))
        .map_err(|error| invalid(error.to_string()))
}

fn reconstruct(
    result: &CommonResult,
    field: Id<kinds::Field>,
    point: &EvaluationPoint,
    derivative: bool,
) -> Result<PointField, Diagnostic> {
    let plan = result
        .plan()
        .as_scalar()
        .ok_or_else(|| invalid("point Field reconstruction requires an admitted scalar Plan"))?;
    if plan.spatial() != CommonSpatialPolicy::Q1 {
        return Err(invalid(
            "point Field reconstruction requires the admitted Q1 space",
        ));
    }
    let (bounds, boundary) = plan.observation_support(point.domain().erase())?;
    if boundary.is_some() {
        return Err(invalid(
            "point evaluation requires its exact volume support",
        ));
    }
    let (_, owned) = plan.field_support(field.erase())?;
    let value_type = plan
        .fields()
        .find(|(id, _)| *id == field)
        .map(|(_, ty)| ty)
        .ok_or_else(|| invalid("point Field is outside the Plan"))?;
    let CommonResultPayload::Static(payload) = &result.payload else {
        return Err(invalid(
            "point Field requires an instantaneous accepted Result",
        ));
    };
    let accepted = payload
        .fields
        .iter()
        .find(|value| value.field_id == field.ulid().to_string())
        .ok_or_else(|| invalid("point Field is absent from the accepted Result"))?;
    let [block] = accepted.blocks.as_slice() else {
        return Err(invalid(
            "point Field requires one accepted coefficient block",
        ));
    };
    let owner = result
        .plan()
        .authenticated_mesh()
        .ok_or_else(|| invalid("point Field has no authenticated Mesh"))?;
    let artifact = owner
        .cartesian_mesh()
        .ok_or_else(|| invalid("point Field requires its Cartesian Mesh"))?;
    let mesh = artifact.mesh();
    let dimension = mesh.topological_dimension();
    let points = point.coordinates().collect::<BTreeMap<_, _>>();
    let mut indices = Vec::with_capacity(dimension);
    let mut reference = Vec::with_capacity(dimension);
    for (axis, bounds) in bounds.iter().enumerate() {
        let value = points
            .get(&(point.domain().erase(), axis))
            .ok_or_else(|| invalid("point omits an exact physical axis"))?
            .value();
        if value < bounds[0] || value > bounds[1] {
            return Err(invalid("point is outside the exact Field support"));
        }
        let coordinates = mesh.axis_coordinates(axis).expect("authenticated axis");
        let mut index = coordinates
            .partition_point(|edge| *edge <= value)
            .saturating_sub(1)
            .min(coordinates.len() - 2);
        if derivative && point.side().is_none() && coordinates.contains(&value) {
            return Err(invalid(
                "point gradient on a cell boundary requires an explicit side",
            ));
        }
        if index > 0
            && coordinates[index] == value
            && (value == bounds[1] || point.side() == Some(BoundarySide::Lower))
        {
            index -= 1;
        }
        reference.push(
            2.0 * (value - coordinates[index]) / (coordinates[index + 1] - coordinates[index])
                - 1.0,
        );
        indices.push(index);
    }
    let cell = mesh
        .cell_at(&indices)
        .ok_or_else(|| invalid("point does not resolve to a retained cell"))?;
    let vertices = mesh.entity_vertices(cell).expect("authenticated vertices");
    let geometry = mesh.geometry_map(cell).expect("authenticated cell");
    let basis = HypercubeQ1Space::new(dimension)?.tabulate(&reference)?;
    q1_field(
        value_type,
        &owned,
        block,
        &vertices,
        &basis,
        &geometry.inverse_jacobian()?,
        [None, None],
    )
}
