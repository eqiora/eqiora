//! Integrals and point probes use the same accepted coefficient reconstruction.
use super::*;
use crate::affine_fem::physical_gradient;
use crate::discrete_space::{BasisTabulation, DiscreteSpace};
use crate::field_sampling::interpolate_nodal_components;
use eqiora_core::{ScalarDomain, ValueType};
use eqiora_sem::EvaluationPoint;

fn scalar_width(value_type: &ValueType) -> Result<usize, Diagnostic> {
    match value_type.scalar_domain() {
        ScalarDomain::Real => Ok(1),
        ScalarDomain::Complex => Ok(2),
        _ => Err(invalid(
            "Field reconstruction requires real or complex coordinates",
        )),
    }
}

fn coordinate_count(value_type: &ValueType) -> Result<usize, Diagnostic> {
    value_type
        .shape()
        .component_count()
        .and_then(|count| count.checked_mul(scalar_width(value_type).ok()?))
        .ok_or_else(|| invalid("Field reconstruction coordinate count is unavailable"))
}

fn literal(value_type: &ValueType, coordinates: &[f64]) -> Result<ValueLiteral, Diagnostic> {
    let width = scalar_width(value_type)?;
    if coordinates.len() != coordinate_count(value_type)? {
        return Err(invalid(
            "Field reconstruction differs from its complete value type",
        ));
    }
    ValueLiteral::new(
        value_type.clone(),
        coordinates
            .chunks_exact(width)
            .map(|part| (part[0], part.get(1).copied().unwrap_or(0.))),
    )
    .map_err(|error| invalid(error.to_string()))
}

pub(super) fn nodal_field(
    value_type: &ValueType,
    owned: &[usize],
    block: FieldCoefficients<'_>,
    vertices: &[MeshEntity],
    basis: &BasisTabulation,
    inverse: &[f64],
    directions: [Option<&[f64]>; 2],
) -> Result<PointField, Diagnostic> {
    let dimension = basis.reference_dimension();
    let coordinates = coordinate_count(value_type)?;
    if block.association != CommonFieldAssociation::Vertex
        || block.values.len() != owned.len() * coordinates
        || vertices.len() != basis.values().len()
        || inverse.len() != dimension * dimension
        || directions
            .iter()
            .flatten()
            .any(|direction| direction.len() != block.values.len())
    {
        return Err(invalid(
            "Observable nodal reconstruction requires complete vertex coordinates",
        ));
    }
    let indices = vertices
        .iter()
        .map(|vertex| vertex.index())
        .collect::<Vec<_>>();
    let value =
        interpolate_nodal_components(owned, block.values, coordinates, &indices, basis.values())?;
    let [first, second] = directions.map(|values| {
        values.map_or_else(
            || Ok(vec![0.0; coordinates]),
            |values| {
                interpolate_nodal_components(owned, values, coordinates, &indices, basis.values())
            },
        )
    });
    let direction = [first?, second?];
    let mut gradient_tangent = [
        vec![0.0; coordinates * dimension],
        vec![0.0; coordinates * dimension],
    ];
    let mut gradient = vec![0.0; coordinates * dimension];
    for (local, vertex) in vertices.iter().enumerate() {
        let owned_index = owned
            .binary_search(&vertex.index())
            .map_err(|_| invalid("point lies outside the exact Field cell closure"))?;
        let derivative = physical_gradient(
            basis.gradient(local).expect("nodal basis gradient exists"),
            inverse,
            dimension,
        );
        for component in 0..coordinates {
            let index = owned_index * coordinates + component;
            let coefficient = block.values[index];
            for (axis, derivative) in derivative.iter().enumerate() {
                gradient[component * dimension + axis] += coefficient * derivative;
            }
            for (order, values) in directions.iter().enumerate() {
                let delta = values.map_or(0.0, |values| values[index]);
                for (axis, derivative) in derivative.iter().enumerate() {
                    gradient_tangent[order][component * dimension + axis] += delta * derivative;
                }
            }
        }
    }
    Ok(PointField {
        value: literal(value_type, &value)?,
        gradient,
        tangent: direction,
        gradient_tangent,
    })
}

pub(in crate::common_result::observe) fn sample(
    result: &Occurrence<'_>,
    field: Id<kinds::Field>,
    point: &EvaluationPoint,
) -> Result<ValueLiteral, Diagnostic> {
    let (plan, value_type, block) = point_source(result, field)?;
    if block.association == CommonFieldAssociation::Cell {
        let coordinates = point
            .coordinates()
            .map(|(axis, value)| (axis, value.value()))
            .collect();
        if let Some(value) = plan.factor_field_value(field, block.values, &coordinates)? {
            if point.side().is_some() {
                return Err(invalid(
                    "cell-constant point sampling does not admit coordinate-side traces",
                ));
            }
            return ValueLiteral::from_real(value_type.clone(), value)
                .map_err(|error| invalid(error.to_string()));
        }
    }
    if let Some(owner) = result.plan().authenticated_mesh()
        && let Some(mesh) = owner.simplicial_mesh()
    {
        return sample_simplicial(result, field, point, mesh.mesh());
    }
    Ok(reconstruct(result, field, point, false)?.value)
}

fn point_source<'a>(
    result: &'a Occurrence<'_>,
    field: Id<kinds::Field>,
) -> Result<
    (
        &'a crate::CommonSpatialPlan,
        &'a ValueType,
        FieldCoefficients<'a>,
    ),
    Diagnostic,
> {
    let plan = result
        .plan()
        .as_spatial()
        .ok_or_else(|| invalid("point Field reconstruction requires an admitted spatial Plan"))?;
    let value_type = plan
        .fields()
        .find(|(id, _)| *id == field)
        .map(|(_, ty)| ty)
        .ok_or_else(|| invalid("point Field is outside the Plan"))?;
    let block = result.field(field)?;
    coordinate_count(value_type)?;
    Ok((plan, value_type, block))
}

fn sample_simplicial(
    result: &Occurrence<'_>,
    field: Id<kinds::Field>,
    point: &EvaluationPoint,
    mesh: &eqiora_meshing::SimplicialMesh,
) -> Result<ValueLiteral, Diagnostic> {
    // Cartesian lower/upper sides do not identify a simplex facet or incident cell.
    if point.side().is_some() {
        return Err(invalid(
            "simplicial point sampling does not admit coordinate-side traces",
        ));
    }
    let (plan, value_type, block) = point_source(result, field)?;
    if block.association != CommonFieldAssociation::Vertex {
        return Err(invalid(
            "simplicial point Field requires vertex coefficients",
        ));
    }
    let (owned, cells) = plan.simplicial_point_support(field, point.domain())?;
    let coordinates = point.coordinates().collect::<BTreeMap<_, _>>();
    if coordinates.len() != 2 {
        return Err(invalid(
            "simplicial point sampling requires two physical coordinates",
        ));
    }
    let mut position = [0.0; 2];
    for (axis, value) in position.iter_mut().enumerate() {
        *value = coordinates
            .get(&(point.domain().erase(), axis))
            .ok_or_else(|| invalid("point omits an exact physical axis"))?
            .value();
    }
    let components = coordinate_count(value_type)?;
    let values = crate::field_sampling::ContinuousP1Sampler2d::new(
        mesh,
        &cells,
        &owned,
        block.values,
        components,
    )?
    .at(position)?;
    literal(value_type, &values)
}

pub(in crate::common_result::observe) fn sample_partial(
    result: &Occurrence<'_>,
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
    let value_type = sample
        .value
        .value_type()
        .clone()
        .with_dimension(dimension)
        .map_err(|error| invalid(error.to_string()))?;
    let spatial_dimension = point.coordinates().count();
    if axis >= spatial_dimension {
        return Err(invalid("point gradient axis is unavailable"));
    }
    let values = sample
        .gradient
        .chunks_exact(spatial_dimension)
        .map(|gradient| gradient[axis])
        .collect::<Vec<_>>();
    literal(&value_type, &values)
}

fn reconstruct(
    result: &Occurrence<'_>,
    field: Id<kinds::Field>,
    point: &EvaluationPoint,
    derivative: bool,
) -> Result<PointField, Diagnostic> {
    let (plan, value_type, block) = point_source(result, field)?;
    let cell_constant = plan.field_space(field)? == eqiora_realization::Space::cell_constant();
    if (!cell_constant
        && plan.field_space(field)?
            != eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN))
        || result
            .plan()
            .authenticated_mesh()
            .and_then(|owner| owner.cartesian_mesh().cloned())
            .is_none()
    {
        return Err(invalid(
            "point Field reconstruction requires an admitted Q1 or cell-constant space",
        ));
    }
    let (bounds, boundary) = plan.observation_support(point.domain().erase())?;
    if boundary.is_some() {
        return Err(invalid(
            "point evaluation requires its exact volume support",
        ));
    }
    let (_, owned) = plan.field_support(field.erase())?;
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
        if (derivative || cell_constant) && point.side().is_none() && coordinates.contains(&value) {
            return Err(invalid(
                "discontinuous point value or gradient on a cell boundary requires an explicit side",
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
    if cell_constant {
        let count = coordinate_count(value_type)?;
        if block.association != CommonFieldAssociation::Cell
            || block.values.len() != owned.len() * count
        {
            return Err(invalid(
                "cell-constant observation requires complete cell coordinates",
            ));
        }
        let index = owned
            .binary_search(&cell.index())
            .map_err(|_| invalid("point lies outside the exact Field cell support"))?;
        return Ok(PointField {
            value: literal(
                value_type,
                &block.values[index * count..(index + 1) * count],
            )?,
            gradient: vec![0.; count * dimension],
            tangent: [vec![0.; count], vec![0.; count]],
            gradient_tangent: [vec![0.; count * dimension], vec![0.; count * dimension]],
        });
    }
    let vertices = mesh.entity_vertices(cell).expect("authenticated vertices");
    let geometry = mesh.geometry_map(cell).expect("authenticated cell");
    let basis = DiscreteSpace::new(
        eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
        eqiora_meshing::ReferenceCell::hypercube(dimension)?,
    )?
    .tabulate(&reference)?;
    nodal_field(
        value_type,
        &owned,
        block,
        &vertices,
        &basis,
        &geometry.inverse_jacobian()?,
        [None, None],
    )
}
