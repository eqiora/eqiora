use super::*;
use crate::canonical_boundary::PhysicalBoundaryQuantity;
use crate::region_assembly::mapping::{FieldDof, RegionDofMap, bind_region_topology};
use crate::region_assembly::{PreparedRegionAssembly, RegionAssemblyCell};
use eqiora_assembly::{
    AssemblyBackend, AssemblyPacket, AssemblyPacketSetIdentityV1, AssemblyPlan, AssemblyTarget,
    TargetAssemblyMap,
};
use eqiora_meshing::{CartesianMesh, MeshEntity, MeshGeometry, MeshTopology};

impl ExecutableScalarEquations {
    pub(in crate::numerical_admission) fn execute(
        &self,
        admission: &NativeNumericalAdmission,
        request: LinearSolveRequest<'_>,
        mesh: &CartesianMesh,
    ) -> Result<CommonScalarRunOutput, Diagnostic> {
        let dimension = mesh.topological_dimension();
        let domains = self.cell_domains(mesh)?;
        let layouts = self
            .regions
            .iter()
            .map(|region| (region.form.domain(), region.form.volume().fields().to_vec()))
            .collect();
        let reference = self.regions[0].form.volume().reference_cell();
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
                            let local = region.form.volume().evaluate_natural_facet(
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
        let plan = AssemblyPlan::new(vec![AssemblyTarget::new(mapping.free_count())?])?;
        let maps = |index| {
            Ok::<_, Diagnostic>(vec![TargetAssemblyMap::new(
                plan.target_id(0).expect("target"),
                mapping.cell_map(index, true)?,
            )])
        };
        let cells = domains
            .iter()
            .enumerate()
            .map(|(index, _)| {
                Ok(RegionAssemblyCell {
                    index,
                    geometry: mesh
                        .geometry_map(MeshEntity::new(dimension, index))
                        .expect("cell geometry"),
                    mappings: maps(index)?,
                    previous: BTreeMap::new(),
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let packets = natural
            .into_iter()
            .map(|(index, local)| AssemblyPacket::new(local, maps(index)?))
            .collect::<Result<Vec<_>, _>>()?;
        let quadrature = QuadratureRule::tensor_product_gauss_legendre(dimension, 2)?;
        let forms = self
            .regions
            .iter()
            .map(|region| (region.form.volume().clone(), quadrature.clone()))
            .collect();
        let work = PreparedRegionAssembly::new(
            AssemblyPacketSetIdentityV1::Unbound,
            &plan,
            forms,
            &domains,
            cells,
            packets,
        )?;
        let (systems, assembly_report) = REFERENCE_ASSEMBLY_BACKEND
            .assemble(&plan, &work)?
            .into_parts();
        let canonical = Arc::new(eqiora_solver::CanonicalCsrSystemView::new(
            &systems[0],
            eqiora_solver::LinearOperatorProperties::General,
        )?);
        let core = crate::finalized_spatial::FinalizedLinearCore::new(
            request.plan(),
            VectorLayoutKind::Replicated,
            Target::HostCpu {
                threads: admission.linear.workers,
            },
            canonical,
        );
        let solution = request.solve(&core.linear_problem()?)?;
        core.validate_solution(&solution)?;
        let (values, solve_report) = solution.into_parts();
        let expected =
            self.regions
                .iter()
                .flat_map(|region| {
                    region.form.fields().iter().map(move |(field, value_type)| {
                        (*field, (region.form.domain(), value_type))
                    })
                })
                .collect::<BTreeMap<_, _>>();
        let inventory = expected.keys().copied().collect::<Vec<_>>();
        let fields = mapping
            .recover(&values, &inventory)?
            .into_iter()
            .map(|(field, recovered)| {
                let (domain, value_type) = expected[&field];
                if recovered.domain != domain || &recovered.value_type != value_type {
                    return Err(invalid(
                        "Run recovery differs from exact Field type or Domain",
                    ));
                }
                Ok((
                    field.downcast().expect("Field"),
                    recovered.value_type,
                    recovered.coefficients.into_values().collect(),
                ))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        Ok(CommonScalarRunOutput {
            fields,
            solve_report,
            assembly_report,
        })
    }
}
