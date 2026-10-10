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
use crate::discrete_space::DiscreteSpace;

mod interface;
mod projection;
mod sampling;
pub(super) use sampling::{sample, sample_partial};
mod variation;
use super::tangent::StateDerivative;

pub(super) struct PointField {
    value: ValueLiteral,
    gradient: Vec<f64>,
    tangent: [Vec<f64>; 2],
    gradient_tangent: [Vec<f64>; 2],
}

pub(super) fn integrate(
    result: &CommonResult,
    program: &KernelProgram,
    observable: &ObservableDef,
    typed: &TypedResidual<RawId>,
    domain: Id<kinds::Domain>,
    quadrature: &QuadratureRule,
    tangent: Option<&StateDerivative<'_>>,
) -> Result<ValueLiteral, Diagnostic> {
    let CommonResultPayload::Static(payload) = &result.payload else {
        return Err(invalid(
            "spatial Observable requires an instantaneous accepted State",
        ));
    };
    let owner = result
        .plan()
        .authenticated_mesh()
        .ok_or_else(|| invalid("spatial Observable Result has no authenticated mesh"))?;
    let artifact = owner.cartesian_mesh().ok_or_else(|| {
        invalid("spatial Observable requires the admitted Cartesian mesh profile")
    })?;
    let mesh = artifact.mesh();
    let dimension = mesh.topological_dimension();
    let physical_interface = matches!(program.node(domain.erase()),
        Some(eqiora_schema::kernel::KernelNode::Domain(definition))
        if matches!(definition.kind(), eqiora_schema::kernel::DomainKind::PhysicalInterface { .. }));
    let (bounds, boundary, field_types, support) = if let Some(plan) = result.plan().as_linear() {
        if plan.spatial() != CommonSpatialPolicy::Q1 {
            return Err(invalid(
                "spatial Observable reconstruction requires the accepted Q1 field space",
            ));
        }
        let (bounds, boundary) = plan.observation_support(domain.erase())?;
        let fields = plan
            .fields()
            .map(|(id, ty)| (id, ty.clone()))
            .collect::<Vec<_>>();
        let support = fields
            .iter()
            .map(|(id, _)| Ok((id.erase(), plan.field_support(id.erase())?.1)))
            .collect::<Result<BTreeMap<_, _>, Diagnostic>>()?;
        (bounds, boundary, fields, support)
    } else {
        return Err(invalid(
            "spatial Observable requires an admitted Cartesian Q1 Result",
        ));
    };
    if bounds.len() != dimension {
        return Err(invalid(
            "Observable support dimension differs from Result mesh",
        ));
    }
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
    let requested = typed
        .expression()
        .nodes()
        .iter()
        .filter_map(|node| match node {
            eqiora_schema::kernel::ExprNode::Symbol(eqiora_schema::kernel::SymbolRef::Field(
                id,
            )) => Some(id.erase()),
            _ => None,
        })
        .collect();
    let prescribed = crate::form_compiler::linear::observation::PrescribedFieldData::derive(
        program, &requested, dimension,
    )?;
    let local_variation = match tangent {
        Some(StateDerivative::Second { wrt, .. }) => Some(
            eqiora_compiler::AuthoredFormExpressionV1::derive_spatial_variation(
                observable,
                typed,
                *wrt,
                &["first".into(), "second".into()],
            )?,
        ),
        _ => None,
    };
    let directions = tangent.map_or([None, None], StateDerivative::directions);
    let space = DiscreteSpace::new(
        eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
        eqiora_meshing::ReferenceCell::hypercube(dimension)?,
    )?;
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
        let adjacent = if physical_interface {
            let (axis, side) =
                boundary.ok_or_else(|| invalid("Interface lacks admitted boundary support"))?;
            Some(interface::adjacent_cells(
                mesh,
                cell,
                axis,
                bounds[axis][usize::from(side == BoundarySide::Upper)],
            )?)
        } else {
            None
        };
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
            for (id, value_type) in &field_types {
                let owned = &support[&id.erase()];
                let field_cell = if let Some(adjacent) = &adjacent {
                    let Some(cell) = interface::owned_cell(mesh, adjacent, owned)? else {
                        continue;
                    };
                    cell
                } else if !vertices
                    .iter()
                    .all(|vertex| owned.binary_search(&vertex.index()).is_ok())
                {
                    continue;
                } else {
                    cell
                };
                let alternate = if field_cell != cell {
                    Some((
                        mesh.entity_vertices(field_cell).expect("adjacent vertices"),
                        interface::tabulate(mesh, field_cell, &coordinates, &space)?,
                    ))
                } else {
                    None
                };
                let (vertices, basis, inverse) = alternate.as_ref().map_or(
                    (&vertices, &basis, &inverse),
                    |(vertices, (basis, inverse))| (vertices, basis, inverse),
                );
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
                fields.insert(
                    id.erase(),
                    sampling::q1_field(
                        value_type,
                        owned,
                        block,
                        vertices,
                        basis,
                        inverse,
                        directions.map(|fields| {
                            fields
                                .and_then(|fields| fields.get(&id.erase()))
                                .map(Vec::as_slice)
                        }),
                    )?,
                );
            }
            prescribed.sample(&coordinates, |field, value, gradient| {
                fields.insert(
                    field,
                    PointField {
                        value,
                        gradient,
                        tangent: [vec![0.0], vec![0.0]],
                        gradient_tangent: [vec![0.0; dimension], vec![0.0; dimension]],
                    },
                );
            })?;
            let value = if let Some(local) = &local_variation {
                variation::evaluate(local, program, &coordinates, &fields)?
            } else {
                projection::evaluate(
                    program,
                    observable,
                    typed,
                    &coordinates,
                    normal,
                    &fields,
                    tangent.is_some(),
                )?
            };
            total += point.weight * measure * value;
        }
    }
    ValueLiteral::from_real(observable.value_type().clone(), total)
        .map_err(|error| invalid(error.to_string()))
}
