use super::*;
use crate::canonical_boundary::PhysicalBoundaryQuantity;
use crate::region_assembly::mapping::{FieldDof, RegionDofMap, bind_region_topology};
use eqiora_meshing::{CartesianMesh, MeshEntity, MeshGeometry, MeshTopology};

impl<S: Coefficient + crate::finalized_spatial::ResidualScalar + Send>
    ExecutableScalarEquations<S>
{
    pub(in crate::numerical_admission) fn execute(
        &self,
        workers: NonZeroUsize,
        request: LinearSolveRequest<'_, S>,
        mesh: &CartesianMesh,
        complete: impl FnOnce(
            &crate::region_assembly::InterfaceReactions<S>,
            &[S],
        ) -> Result<
            crate::region_assembly::RecoveredInterfaceReactions<S>,
            Diagnostic,
        >,
    ) -> Result<CommonScalarRunOutput<S>, Diagnostic> {
        let dimension = mesh.topological_dimension();
        let domains = self.cell_domains(mesh)?;
        let layouts = self
            .regions
            .iter()
            .map(|region| {
                Ok((
                    region.form.domain(),
                    region.form.volume()?.fields().to_vec(),
                ))
            })
            .collect::<Result<BTreeMap<_, _>, Diagnostic>>()?;
        let reference = self.regions[0].form.volume()?.reference_cell();
        let (domains, traces) = bind_region_topology(
            mesh,
            domains
                .into_iter()
                .enumerate()
                .map(|(cell, domain)| (eqiora_meshing::CellId::new(cell), domain)),
            &self.quotients()?,
        )?;
        let mut prescribed = BTreeMap::new();
        let mut natural = Vec::new();
        let facet_rule = if dimension == 1 {
            QuadratureRule::point()
        } else {
            QuadratureRule::tensor_product_gauss_legendre(dimension - 1, 2)?
        };
        for region in &self.regions {
            for (field, _) in region.form.fields() {
                for (&(axis, side), boundary) in &region.boundaries {
                    let Some(law) = region.form.boundary_laws()[field].get(boundary) else {
                        continue;
                    };
                    let coordinate = region.bounds[axis][usize::from(side == BoundarySide::Upper)];
                    for index in 0..mesh.entity_count(dimension - 1).expect("facets") {
                        let facet = MeshEntity::new(dimension - 1, index);
                        let vertices = mesh.entity_vertices(facet).expect("facet vertices");
                        if !vertices.iter().all(|vertex| {
                            let point = mesh.vertex_coordinates(*vertex).expect("vertex");
                            point[axis] == coordinate
                                && region
                                    .bounds
                                    .iter()
                                    .enumerate()
                                    .all(|(a, b)| point[a] >= b[0] && point[a] <= b[1])
                        }) {
                            continue;
                        }
                        let parents = mesh
                            .incidence(facet, dimension)
                            .expect("facet parents")
                            .into_iter()
                            .filter(|parent| domains[parent.entity.index()] == region.form.domain())
                            .collect::<Vec<_>>();
                        let [parent] = parents.as_slice() else {
                            return Err(invalid(
                                "Region boundary needs exactly one owned parent cell",
                            ));
                        };
                        if law.quantity == PhysicalBoundaryQuantity::Trace {
                            for vertex in vertices {
                                let value = law.evaluate(
                                    &mesh.vertex_coordinates(vertex).expect("vertex"),
                                    &[],
                                )?[0];
                                let key = FieldDof {
                                    field: *field,
                                    entity: vertex,
                                    slot: 0,
                                    component: 0,
                                };
                                let value =
                                    crate::cartesian_elliptic::support::require_compatible_boundary_value(
                                        prescribed.get(&key).copied(),
                                        value,
                                    )?
                                    .expect("finite candidate");
                                prescribed.insert(key, value);
                            }
                        } else {
                            let geometry = mesh.geometry_map(parent.entity).expect("cell geometry");
                            let facet_geometry = mesh.geometry_map(facet).expect("facet geometry");
                            let cell_vertices =
                                mesh.entity_vertices(parent.entity).expect("cell vertices");
                            let positions = vertices
                                .iter()
                                .map(|vertex| {
                                    cell_vertices
                                        .iter()
                                        .position(|candidate| candidate == vertex)
                                        .expect("incidence closure")
                                })
                                .collect::<Vec<_>>();
                            let local = region.form.volume()?.evaluate_natural_facet(
                                *field,
                                &geometry,
                                (&facet_geometry, *parent, &positions),
                                &facet_rule,
                                |point, _| law.evaluate(point, &[]),
                            )?;
                            natural.push((parent.entity.index(), local));
                        }
                    }
                }
            }
        }
        let mapping = RegionDofMap::new(mesh, &layouts, reference, &domains, &traces, &prescribed)?;
        let quadrature = QuadratureRule::tensor_product_gauss_legendre(dimension, 2)?;
        let forms = self
            .regions
            .iter()
            .map(|region| Ok((region.form.volume()?, quadrature.clone())))
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let output = mapping.solve(mesh, forms, natural, workers, request, complete)?;
        let fields = output
            .fields
            .into_iter()
            .map(|(field, recovered)| {
                (
                    field.downcast().expect("Field"),
                    recovered.value_type,
                    recovered.coefficients.into_values().collect(),
                    recovered.space,
                )
            })
            .collect();
        Ok(CommonScalarRunOutput {
            nullspace: None,
            fields,
            solve_report: output.solve_report,
            assembly_report: output.assembly_report,
        })
    }
}
