use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id, RawId, ValueLiteral};
use eqiora_meshing::{GeometryMap, MeshEntity, MeshGeometry, MeshTopology, QuadratureRule};
use eqiora_schema::kernel::typing::TypedResidual;
use eqiora_schema::kernel::{BoundarySide, ObservableDef};
use eqiora_sem::KernelProgram;

use super::super::{CommonFieldAssociation, invalid};
use super::occurrence::{FieldCoefficients, Occurrence};

use crate::discrete_space::DiscreteSpace;

mod interface;
mod measure;
mod projection;
mod sampling;
pub(super) use sampling::{sample, sample_partial};
mod variation;
use super::tangent::StateDerivative;

pub(super) struct PointField {
    value: ValueLiteral,
    // Real coordinates (component, real/imaginary part), then physical direction.
    // Real Fields have one coordinate per component; complex Fields have two.
    gradient: Vec<f64>,
    tangent: [Vec<f64>; 2],
    gradient_tangent: [Vec<f64>; 2],
}

pub(super) fn integrate(
    result: &Occurrence<'_>,
    program: &KernelProgram,
    observable: &ObservableDef,
    typed: &TypedResidual<RawId>,
    domain: Id<kinds::Domain>,
    quadrature: &QuadratureRule,
    tangent: Option<&StateDerivative<'_>>,
) -> Result<ValueLiteral, Diagnostic> {
    let plan = result
        .plan()
        .as_spatial()
        .ok_or_else(|| invalid("spatial Observable requires an admitted spatial Plan"))?;
    let owner = result
        .plan()
        .authenticated_mesh()
        .ok_or_else(|| invalid("spatial Observable Result has no authenticated mesh"))?;
    let physical_interface = matches!(program.node(domain.erase()),
        Some(eqiora_schema::kernel::KernelNode::Domain(definition))
        if matches!(definition.kind(), eqiora_schema::kernel::DomainKind::PhysicalInterface { .. }));
    let integrand = Integrand {
        result,
        program,
        observable,
        typed,
        quadrature,
        tangent,
    };
    if let Some(artifact) = owner.cartesian_mesh() {
        let mesh = artifact.mesh();
        return integrand.on_mesh(
            mesh,
            measure::cartesian(plan, mesh, domain.erase())?,
            physical_interface,
        );
    }
    if let Some(artifact) = owner.simplicial_mesh() {
        let mesh = artifact.mesh();
        let (cells, facets) = plan.simplicial_observation_support(domain.erase())?;
        let selected = if let Some(facets) = facets {
            measure::boundary(mesh, &cells, facets)?
        } else {
            cells
                .into_iter()
                .map(|cell| {
                    measure::MeasureCell::Volume(MeshEntity::new(
                        mesh.topological_dimension(),
                        cell,
                    ))
                })
                .collect()
        };
        return integrand.on_mesh(mesh, selected, physical_interface);
    }
    Err(invalid(
        "spatial Observable requires an admitted affine Mesh",
    ))
}

struct Integrand<'a, 'state> {
    result: &'a Occurrence<'a>,
    program: &'a KernelProgram,
    observable: &'a ObservableDef,
    typed: &'a TypedResidual<RawId>,
    quadrature: &'a QuadratureRule,
    tangent: Option<&'a StateDerivative<'state>>,
}

impl Integrand<'_, '_> {
    fn on_mesh<'mesh>(
        self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = eqiora_meshing::AffineGeometryMap>,
        selected: Vec<measure::MeasureCell>,
        physical_interface: bool,
    ) -> Result<ValueLiteral, Diagnostic> {
        let Self {
            result,
            program,
            observable,
            typed,
            quadrature,
            tangent,
        } = self;
        let plan = result.plan().as_spatial().expect("admitted spatial Plan");
        let requested: BTreeSet<_> = typed
            .expression()
            .nodes()
            .iter()
            .filter_map(|node| match node {
                eqiora_schema::kernel::ExprNode::Symbol(
                    eqiora_schema::kernel::SymbolRef::Field(id),
                ) => Some(id.erase()),
                _ => None,
            })
            .collect();
        let field_types = plan
            .fields()
            .filter(|(id, _)| requested.contains(&id.erase()))
            .map(|(id, ty)| (id, ty.clone()))
            .collect::<Vec<_>>();
        if field_types.iter().any(|(field, _)| {
            plan.field_space(*field).map_or(true, |space| {
                space != eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN)
                    && space != eqiora_realization::Space::cell_constant()
            })
        }) {
            return Err(invalid(
                "spatial Observable reconstruction requires a linear nodal or cell-constant Field space",
            ));
        }
        if selected.is_empty() {
            return Err(invalid("spatial Observable has an empty exact measure"));
        }
        let dimension = mesh.topological_dimension();
        let support = field_types
            .iter()
            .map(|(id, _)| Ok((id.erase(), plan.field_support(id.erase())?.1)))
            .collect::<Result<BTreeMap<_, _>, Diagnostic>>()?;
        let prescribed = crate::form_compiler::spatial::observation::PrescribedFieldData::derive(
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
        let mut total = vec![0.; sampling::coordinate_count(observable.value_type())?];
        for selected in selected {
            let cell = selected.cell();
            let geometry = mesh
                .geometry_map(cell)
                .ok_or_else(|| invalid("observation cell has no geometry"))?;
            let vertices = measure::vertices(mesh, cell)?;
            let inverse = geometry.inverse_jacobian()?;
            let space = DiscreteSpace::new(
                eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
                geometry.reference_cell(),
            )?;
            let boundary = if let measure::MeasureCell::Boundary { facet, parent } = selected {
                let map = mesh
                    .geometry_map(facet)
                    .ok_or_else(|| invalid("observation facet has no geometry"))?;
                let positions = measure::vertices(mesh, facet)?
                    .iter()
                    .map(|vertex| {
                        vertices
                            .iter()
                            .position(|candidate| candidate == vertex)
                            .ok_or_else(|| invalid("facet vertex is absent from its exact parent"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let facet_space = DiscreteSpace::new(
                    if map.reference_cell().dimension() == 0 {
                        eqiora_realization::Space::cell_constant()
                    } else {
                        eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN)
                    },
                    map.reference_cell(),
                )?;
                Some((
                    facet,
                    map,
                    positions,
                    facet_space,
                    crate::affine_fem::parent_outward_normal(&geometry, parent)?,
                ))
            } else {
                None
            };
            let reference_cell = boundary
                .as_ref()
                .map_or(geometry.reference_cell(), |(_, map, _, _, _)| {
                    map.reference_cell()
                });
            if quadrature.reference_cell() != reference_cell {
                return Err(invalid(
                    "Observable quadrature reference cell differs from its exact measure",
                ));
            }
            let adjacent = if physical_interface {
                let facet = boundary
                    .as_ref()
                    .ok_or_else(|| invalid("Interface lacks exact boundary support"))?
                    .0;
                let adjacent = mesh
                    .incidence(facet, dimension)
                    .ok_or_else(|| invalid("Interface facet has no parent cells"))?;
                if adjacent.len() != 2 {
                    return Err(invalid(
                        "Interface observation requires exactly two adjacent cells",
                    ));
                }
                Some(
                    adjacent
                        .into_iter()
                        .map(|parent| parent.entity)
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            let projection = projection::Projection::new(
                typed,
                boundary
                    .as_ref()
                    .map(|(_, _, _, _, normal)| normal.as_slice()),
            )?;
            for point in quadrature.points() {
                let (reference, map) = if let Some((_, map, positions, facet_space, _)) = &boundary
                {
                    let facet_basis = facet_space.tabulate(&point.coordinates)?;
                    (
                        crate::affine_fem::facet_reference_point(
                            geometry.reference_cell(),
                            positions,
                            facet_basis.values(),
                        )?,
                        map,
                    )
                } else {
                    (point.coordinates.clone(), &geometry)
                };
                let basis = space.tabulate(&reference)?;
                let mut coordinates = vec![0.0; dimension];
                map.map_point(&point.coordinates, &mut coordinates)?;
                let measure = map.measure_scale();
                let mut fields = BTreeMap::new();
                for (id, value_type) in &field_types {
                    let owned = &support[&id.erase()];
                    let block = result.field(*id)?;
                    let direction = directions.map(|fields| {
                        fields
                            .and_then(|fields| fields.get(&id.erase()))
                            .map(Vec::as_slice)
                    });
                    let candidates = adjacent.as_deref().unwrap_or(std::slice::from_ref(&cell));
                    let Some(field_cell) =
                        interface::owned_cell(mesh, candidates, owned, block.association)?
                    else {
                        continue;
                    };
                    if block.association == CommonFieldAssociation::Cell {
                        fields.insert(
                            id.erase(),
                            sampling::cell_field(
                                value_type, owned, block, field_cell, dimension, direction,
                            )?,
                        );
                        continue;
                    }
                    let alternate = if field_cell != cell {
                        Some((
                            measure::vertices(mesh, field_cell)?,
                            interface::tabulate(mesh, field_cell, &coordinates, &space)?,
                        ))
                    } else {
                        None
                    };
                    let (vertices, basis, inverse) = alternate.as_ref().map_or(
                        (&vertices, &basis, &inverse),
                        |(vertices, (basis, inverse))| (vertices, basis, inverse),
                    );
                    fields.insert(
                        id.erase(),
                        sampling::nodal_field(
                            value_type, owned, block, vertices, basis, inverse, direction,
                        )?,
                    );
                }
                prescribed.sample(&coordinates, |field, value, gradient| {
                    let count = gradient.len() / dimension;
                    fields.insert(
                        field,
                        PointField {
                            value,
                            gradient,
                            tangent: [vec![0.0; count], vec![0.0; count]],
                            gradient_tangent: [
                                vec![0.0; count * dimension],
                                vec![0.0; count * dimension],
                            ],
                        },
                    );
                })?;
                let value = if let Some(local) = &local_variation {
                    if total.len() != 1
                        || fields.values().any(|field| {
                            field.value.value_type().scalar_domain()
                                != eqiora_core::ScalarDomain::Real
                                || field.value.value_type().array_rank() != 0
                        })
                    {
                        return Err(invalid(
                            "structured second State variations require component variation lowering",
                        ));
                    }
                    vec![variation::evaluate(local, program, &coordinates, &fields)?]
                } else {
                    projection.evaluate(result, &coordinates, &fields, tangent.is_some())?
                };
                if value.len() != total.len() {
                    return Err(invalid(
                        "spatial integrand coordinates differ from its complete output type",
                    ));
                }
                for (total, value) in total.iter_mut().zip(value) {
                    *total += point.weight * measure * value;
                }
            }
        }
        sampling::literal(observable.value_type(), &total)
    }
}
