//! Integrals and point probes use the same accepted coefficient reconstruction.
use super::super::projection::{coordinate_count, from_coordinates};
use super::*;
use crate::affine_fem::physical_gradient;
use crate::discrete_space::{BasisTabulation, DiscreteSpace};
use crate::field_sampling::CoefficientStencil;
use eqiora_core::ValueType;
use eqiora_sem::EvaluationPoint;

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
    let gradients = (0..vertices.len())
        .map(|local| {
            physical_gradient(
                basis.gradient(local).expect("nodal basis gradient exists"),
                inverse,
                dimension,
            )
        })
        .collect();
    reconstructed_field(
        value_type,
        block,
        directions,
        CoefficientStencil::new(owned, coordinates, &indices, basis.values(), gradients)?,
    )
}

pub(super) fn cell_field(
    value_type: &ValueType,
    owned: &[usize],
    block: FieldCoefficients<'_>,
    cell: MeshEntity,
    dimension: usize,
    directions: [Option<&[f64]>; 2],
) -> Result<PointField, Diagnostic> {
    let count = coordinate_count(value_type)?;
    if block.association != CommonFieldAssociation::Cell {
        return Err(invalid(
            "cell-constant observation requires cell coefficients",
        ));
    }
    reconstructed_field(
        value_type,
        block,
        directions,
        CoefficientStencil::new(
            owned,
            count,
            &[cell.index()],
            &[1.],
            vec![vec![0.; dimension]],
        )?,
    )
}

fn reconstructed_field(
    value_type: &ValueType,
    block: FieldCoefficients<'_>,
    directions: [Option<&[f64]>; 2],
    stencil: CoefficientStencil,
) -> Result<PointField, Diagnostic> {
    let (value, gradient) = stencil.sample(block.values)?;
    let [first, second] = directions.map(|direction| {
        direction.map_or_else(
            || Ok((vec![0.; value.len()], vec![0.; gradient.len()])),
            |direction| stencil.sample(direction),
        )
    });
    let (first, first_gradient) = first?;
    let (second, second_gradient) = second?;
    Ok(PointField {
        value: from_coordinates(value_type, &value)?,
        gradient,
        tangent: [first, second],
        gradient_tangent: [first_gradient, second_gradient],
        reconstruction: Some(stencil),
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
    if let Some(owner) = result.authenticated_mesh()
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
        .spatial_plan()
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
    let dimension = mesh.topological_dimension();
    if coordinates.len() != dimension {
        return Err(invalid(
            "simplicial point sampling requires every physical coordinate",
        ));
    }
    let mut position = vec![0.0; dimension];
    for (axis, value) in position.iter_mut().enumerate() {
        *value = coordinates
            .get(&(point.domain().erase(), axis))
            .ok_or_else(|| invalid("point omits an exact physical axis"))?
            .value();
    }
    let components = coordinate_count(value_type)?;
    let values = crate::field_sampling::ContinuousP1Sampler::new(
        mesh,
        &cells,
        &owned,
        block.values,
        components,
    )?
    .at(&position)?;
    from_coordinates(value_type, &values)
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
    from_coordinates(&value_type, &values)
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
        return cell_field(value_type, &owned, block, cell, dimension, [None, None]);
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
