use std::collections::BTreeMap;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id, RawId, ValueLiteral};
use eqiora_meshing::{
    GeometryMap, MeshEntity, MeshGeometry, MeshTopology, QuadratureRule, ReferenceCell,
};
use eqiora_schema::kernel::typing::TypedResidual;
use eqiora_schema::kernel::{BoundarySide, ObservableDef};
use eqiora_sem::KernelProgram;

use super::super::{CommonFieldAssociation, CommonResult, CommonResultPayload, invalid};
use crate::CommonSpatialPolicy;
use crate::affine_fem::physical_gradient;
use crate::discrete_space::{DiscreteSpace, HypercubeQ1Space};

mod projection;

pub(super) struct PointField {
    value: ValueLiteral,
    gradient: Vec<f64>,
    tangent: f64,
    gradient_tangent: Vec<f64>,
}

pub(super) fn integrate(
    result: &CommonResult,
    program: &KernelProgram,
    observable: &ObservableDef,
    typed: &TypedResidual<RawId>,
    domain: Id<kinds::Domain>,
    quadrature: &QuadratureRule,
    tangent: Option<&BTreeMap<RawId, Vec<f64>>>,
) -> Result<ValueLiteral, Diagnostic> {
    let plan = result.plan().as_scalar().ok_or_else(|| {
        invalid("spatial Observable requires an admitted scalar Cartesian Result")
    })?;
    if plan.spatial() != CommonSpatialPolicy::Q1 {
        return Err(invalid(
            "spatial Observable reconstruction currently requires the accepted Q1 field space",
        ));
    }
    let CommonResultPayload::Static(payload) = &result.payload else {
        return Err(invalid(
            "spatial Observable requires an instantaneous accepted State",
        ));
    };
    let mesh = result
        .plan()
        .authenticated_mesh()
        .ok_or_else(|| invalid("spatial Observable Result has no authenticated mesh"))?;
    let mesh = mesh
        .cartesian_mesh()
        .ok_or_else(|| invalid("spatial Observable requires the admitted Cartesian mesh profile"))?
        .mesh();
    let dimension = mesh.topological_dimension();
    let (bounds, boundary) = plan.observation_support(domain.erase())?;
    if bounds.len() != dimension {
        return Err(invalid(
            "Observable support dimension differs from Result mesh",
        ));
    }
    let support = plan
        .fields()
        .map(|(id, _)| Ok((id.erase(), plan.field_support(id.erase())?.1)))
        .collect::<Result<BTreeMap<_, _>, Diagnostic>>()?;
    let measure_dimension = dimension - usize::from(boundary.is_some());
    let expected_cell = if measure_dimension == 0 {
        ReferenceCell::point()
    } else {
        ReferenceCell::hypercube(measure_dimension)?
    };
    if quadrature.reference_cell() != expected_cell {
        return Err(invalid(
            "Observable quadrature reference cell differs from its exact measure",
        ));
    }
    let space = HypercubeQ1Space::new(dimension)?;
    let mut total = 0.0;
    for cell_index in 0..mesh
        .entity_count(dimension)
        .expect("mesh top stratum exists")
    {
        let cell = MeshEntity::new(dimension, cell_index);
        let geometry = mesh.geometry_map(cell).expect("mesh cell geometry exists");
        let cell_vertices = mesh.entity_vertices(cell).expect("cell vertices");
        if !cell_vertices.iter().all(|vertex| {
            let point = mesh.vertex_coordinates(*vertex).expect("vertex");
            bounds
                .iter()
                .enumerate()
                .all(|(axis, b)| point[axis] >= b[0] && point[axis] <= b[1])
        }) {
            continue;
        }
        if let Some((axis, side)) = boundary {
            let coordinate = bounds[axis][usize::from(side == BoundarySide::Upper)];
            if !cell_vertices
                .iter()
                .any(|vertex| mesh.vertex_coordinates(*vertex).expect("vertex")[axis] == coordinate)
            {
                continue;
            }
        }
        let inverse = geometry.inverse_jacobian()?;
        let vertices = mesh
            .entity_vertices(cell)
            .expect("mesh cell vertices exist");
        for point in quadrature.points() {
            let mut reference = point.coordinates.clone();
            let mut measure = geometry.measure_scale();
            let normal = boundary.map(|(axis, side)| {
                let sign = match side {
                    BoundarySide::Lower => -1.0,
                    BoundarySide::Upper => 1.0,
                };
                reference.insert(axis, sign);
                measure *= inverse[axis * dimension + axis].abs();
                (axis, sign)
            });
            let basis = space.tabulate(&reference)?;
            let mut coordinates = vec![0.0; dimension];
            geometry.map_point(&reference, &mut coordinates)?;
            let mut fields = BTreeMap::new();
            for (id, value_type) in plan.fields() {
                let owned = &support[&id.erase()];
                if !vertices
                    .iter()
                    .all(|vertex| owned.binary_search(&vertex.index()).is_ok())
                {
                    continue;
                }
                let accepted = payload
                    .fields
                    .iter()
                    .find(|field| field.field_id == id.ulid().to_string())
                    .ok_or_else(|| {
                        invalid("Observable required Field is absent from accepted Result")
                    })?;
                let [block] = accepted.blocks.as_slice() else {
                    return Err(invalid(
                        "Observable requires exactly one Q1 coefficient block",
                    ));
                };
                if block.association != CommonFieldAssociation::Vertex
                    || !accepted.value_shape.is_empty()
                {
                    return Err(invalid(
                        "Observable Q1 reconstruction requires a real scalar vertex Field",
                    ));
                }
                let mut value = 0.0;
                let mut direction = 0.0;
                let mut gradient_tangent = vec![0.0; dimension];
                let mut gradient = vec![0.0; dimension];
                for (local, vertex) in vertices.iter().enumerate() {
                    let owned_index = owned
                        .binary_search(&vertex.index())
                        .expect("checked Field cell closure");
                    let coefficient = block.values[owned_index];
                    let delta = tangent
                        .and_then(|fields| fields.get(&id.erase()))
                        .map_or(0.0, |values| values[owned_index]);
                    direction += delta * basis.values()[local];
                    value += coefficient * basis.values()[local];
                    let derivative = physical_gradient(
                        basis.gradient(local).expect("Q1 basis gradient exists"),
                        &inverse,
                        dimension,
                    );
                    for ((entry, tangent_entry), derivative) in gradient
                        .iter_mut()
                        .zip(&mut gradient_tangent)
                        .zip(derivative)
                    {
                        *entry += coefficient * derivative;
                        *tangent_entry += delta * derivative;
                    }
                }
                fields.insert(
                    id.erase(),
                    PointField {
                        value: ValueLiteral::from_real(value_type.clone(), value)
                            .map_err(|error| invalid(error.to_string()))?,
                        gradient,
                        tangent: direction,
                        gradient_tangent,
                    },
                );
            }
            let value = projection::evaluate(
                program,
                observable,
                typed,
                &coordinates,
                normal,
                &fields,
                tangent.is_some(),
            )?;
            total += point.weight * measure * value;
        }
    }
    ValueLiteral::from_real(observable.value_type().clone(), total)
        .map_err(|error| invalid(error.to_string()))
}
