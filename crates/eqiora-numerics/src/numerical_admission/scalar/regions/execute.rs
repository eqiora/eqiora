use super::*;
use crate::canonical_boundary::PhysicalBoundaryQuantity;
use crate::region_assembly::mapping::{FieldDof, RegionDofMap, bind_region_topology};
use eqiora_meshing::{CartesianMesh, MeshEntity, MeshGeometry, MeshTopology};

impl<S: Coefficient + crate::finalized_spatial::ResidualScalar + Send>
    ExecutableLinearEquations<S>
{
    pub(in crate::numerical_admission) fn execute(
        &self,
        workers: NonZeroUsize,
        request: LinearSolveRequest<'_, S>,
        resources: &NativeMeshResources,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        operator_properties: LinearOperatorProperties,
        complete: impl FnOnce(
            &crate::region_assembly::InterfaceReactions<S>,
            &[S],
        ) -> Result<
            crate::region_assembly::RecoveredInterfaceReactions<S>,
            Diagnostic,
        >,
    ) -> Result<CommonLinearRunOutput<S>, Diagnostic> {
        let spaces = selected
            .iter()
            .flat_map(|domain| domain.field_spaces())
            .map(|field| field.space())
            .collect::<Vec<_>>();
        match resources {
            NativeMeshResources::Cartesian { mesh, .. }
                if spaces.iter().all(|space| {
                    *space == Space::continuous_lagrange(std::num::NonZeroU16::MIN)
                }) =>
            {
                self.execute_cartesian(
                    workers,
                    request,
                    mesh.mesh(),
                    selected,
                    operator_properties,
                    complete,
                )
            }
            NativeMeshResources::GmshSimplicial { mesh, .. }
                if spaces.iter().all(|space| {
                    matches!(space.family(),
                    SpaceFamily::ContinuousLagrange { order } if order == std::num::NonZeroU16::MIN)
                        || matches!(
                            space.family(),
                            SpaceFamily::SimplexP1Bubble
                                | SpaceFamily::TetrahedralEdge
                                | SpaceFamily::TetrahedralFace
                        )
                }) =>
            {
                self.execute_simplicial(
                    workers,
                    request,
                    mesh,
                    selected,
                    operator_properties,
                    complete,
                )
            }
            _ => Err(invalid(
                "linear execution requires a matching authenticated Mesh and Space",
            )),
        }
    }

    pub(in crate::numerical_admission) fn execute_cartesian(
        &self,
        workers: NonZeroUsize,
        request: LinearSolveRequest<'_, S>,
        mesh: &CartesianMesh,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        operator_properties: LinearOperatorProperties,
        complete: impl FnOnce(
            &crate::region_assembly::InterfaceReactions<S>,
            &[S],
        ) -> Result<
            crate::region_assembly::RecoveredInterfaceReactions<S>,
            Diagnostic,
        >,
    ) -> Result<CommonLinearRunOutput<S>, Diagnostic> {
        let (mapping, mut input) = self.cartesian_assembly(mesh, selected)?;
        input.operator_properties = operator_properties;
        let output = mapping.solve(mesh, input, workers, request, complete)?;
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
        Ok(CommonLinearRunOutput {
            reactions: Some(output.reactions),
            nullspace: None,
            fields,
            solve_report: output.solve_report,
            assembly_report: output.assembly_report,
        })
    }

    pub(in crate::numerical_admission) fn cartesian_assembly(
        &self,
        mesh: &CartesianMesh,
        selected: &[eqiora_realization::DomainFieldDiscretization],
    ) -> Result<
        (
            RegionDofMap<S>,
            crate::region_assembly::mapping::RegionSolveInput<S>,
        ),
        Diagnostic,
    > {
        self.cartesian_assembly_with_boundary(mesh, selected, |_, law, value| {
            if law.trace_field.is_some_and(|field| field != law.tested) {
                return Err(invalid("eliminated-state boundary data requires an explicit history-dependent rate constraint"));
            }
            Ok(Some(value))
        })
    }

    pub(in crate::numerical_admission) fn cartesian_assembly_with_boundary(
        &self,
        mesh: &CartesianMesh,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        mut boundary_value: impl FnMut(
            FieldDof,
            &crate::form_compiler::region::RegionBoundaryLaw<S>,
            S,
        ) -> Result<Option<S>, Diagnostic>,
    ) -> Result<
        (
            RegionDofMap<S>,
            crate::region_assembly::mapping::RegionSolveInput<S>,
        ),
        Diagnostic,
    > {
        let dimension = mesh.topological_dimension();
        let domains = self.cell_domains(mesh)?;
        let reference = eqiora_meshing::ReferenceCell::hypercube(dimension)?;
        let bound = self.bind_spaces(reference, selected)?;
        let layouts = self
            .regions
            .iter()
            .zip(&bound)
            .map(|(region, form)| (region.form.domain(), form.fields().to_vec()))
            .collect();
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
        for (region, form) in self.regions.iter().zip(&bound) {
            let support = region.cartesian()?;
            for (field, _) in region.form.fields() {
                for (&(axis, side), boundary) in &support.boundaries {
                    let Some(law) = region.form.boundary_laws()[field].get(boundary) else {
                        continue;
                    };
                    let coordinate = support.bounds[axis][usize::from(side == BoundarySide::Upper)];
                    for index in 0..mesh.entity_count(dimension - 1).expect("facets") {
                        let facet = MeshEntity::new(dimension - 1, index);
                        let vertices = mesh.entity_vertices(facet).expect("facet vertices");
                        if !vertices.iter().all(|vertex| {
                            let point = mesh.vertex_coordinates(*vertex).expect("vertex");
                            point[axis] == coordinate
                                && support
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
                                for (component, value) in law
                                    .evaluate(
                                        &mesh.vertex_coordinates(vertex).expect("vertex"),
                                        &[],
                                    )?
                                    .into_iter()
                                    .enumerate()
                                {
                                    let key = FieldDof {
                                        field: *field,
                                        entity: vertex,
                                        slot: 0,
                                        component,
                                    };
                                    let Some(value) = boundary_value(key, law, value)? else {
                                        continue;
                                    };
                                    let value =
                                        crate::cartesian_elliptic::support::require_compatible_boundary_value(
                                            prescribed.get(&key).copied(),
                                            value,
                                        )?
                                        .expect("finite candidate");
                                    prescribed.insert(key, value);
                                }
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
                            let local = form.evaluate_natural_facet(
                                *field,
                                &geometry,
                                (&facet_geometry, *parent, &positions),
                                &facet_rule,
                                |point, normal| law.evaluate(point, normal),
                            )?;
                            natural.push((parent.entity.index(), local));
                        }
                    }
                }
            }
        }
        let mapping = RegionDofMap::new(mesh, &layouts, reference, &domains, &traces, &prescribed)?;
        let quadrature = QuadratureRule::tensor_product_gauss_legendre(dimension, 2)?;
        let forms = bound
            .into_iter()
            .map(|form| (form, quadrature.clone()))
            .collect();
        Ok((
            mapping,
            crate::region_assembly::mapping::RegionSolveInput {
                operator_properties: eqiora_solver::LinearOperatorProperties::General,
                geometry_action: None,
                forms,
                natural,
                previous: None,
                prescribed_states: BTreeMap::new(),
            },
        ))
    }
}

mod moments;
