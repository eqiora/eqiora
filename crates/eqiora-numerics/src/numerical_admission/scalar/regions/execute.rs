use super::*;
use crate::canonical_boundary::PhysicalBoundaryQuantity;
use crate::region_assembly::mapping::{FieldDof, RegionDofMap, bind_region_topology};
use eqiora_meshing::{CartesianMesh, MeshEntity, MeshTopology};

impl<S: Coefficient + crate::finalized_spatial::ResidualScalar + Send>
    ExecutableSpatialEquations<S>
{
    pub(in crate::numerical_admission) fn finalize_static(
        &self,
        workers: NonZeroUsize,
        solver: eqiora_solver::SolverPlan,
        resources: &NativeMeshResources,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        scaling: &eqiora_realization::SystemScaling,
        operator_properties: LinearOperatorProperties,
    ) -> Result<StaticRegionSystem<S>, Diagnostic> {
        if selected
            .iter()
            .flat_map(|domain| domain.field_spaces())
            .any(|field| field.space() == Space::cell_constant())
        {
            return self.finalize_tpfa(
                workers,
                solver,
                resources,
                selected,
                scaling,
                operator_properties,
            );
        }
        let (mapping, input) =
            self.static_assembly(resources, selected, scaling, operator_properties)?;
        let (core, assembly_report) = match resources {
            NativeMeshResources::Cartesian { mesh, .. } => {
                mapping.finalize_linear(mesh.mesh(), &input, workers, solver)?
            }
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => {
                mapping.finalize_linear(mesh.mesh(), &input, workers, solver)?
            }
            _ => return Err(invalid("static system requires its admitted Mesh")),
        };
        Ok(StaticRegionSystem::new(mapping, core, assembly_report))
    }

    /// One physical coordinate map for static seeds, execution and replay.
    pub(in crate::numerical_admission) fn static_assembly(
        &self,
        resources: &NativeMeshResources,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        scaling: &eqiora_realization::SystemScaling,
        operator_properties: LinearOperatorProperties,
    ) -> Result<
        (
            RegionDofMap<S>,
            crate::region_assembly::mapping::RegionSolveInput<S>,
        ),
        Diagnostic,
    > {
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
                let (mapping, mut input) =
                    self.cartesian_assembly(mesh.mesh(), selected, scaling)?;
                input.operator_properties = operator_properties;
                Ok((mapping, input))
            }
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. }
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
                let (mapping, forms, natural) =
                    self.simplicial_assembly(mesh, selected, scaling)?;
                let input = crate::region_assembly::mapping::RegionSolveInput {
                    operator_properties,
                    geometry_action: None,
                    forms,
                    natural,
                    previous: None,
                    prescribed_states: BTreeMap::new(),
                };
                Ok((mapping, input))
            }
            _ => Err(invalid(
                "linear execution requires a matching authenticated Mesh and Space",
            )),
        }
    }

    pub(in crate::numerical_admission) fn cartesian_assembly(
        &self,
        mesh: &CartesianMesh,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        scaling: &eqiora_realization::SystemScaling,
    ) -> Result<
        (
            RegionDofMap<S>,
            crate::region_assembly::mapping::RegionSolveInput<S>,
        ),
        Diagnostic,
    > {
        self.cartesian_assembly_with_boundary(mesh, selected, scaling, |_, law, value| {
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
        scaling: &eqiora_realization::SystemScaling,
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
        let bound = self.bind_spaces(reference, selected, scaling)?;
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
                            let local = crate::region_assembly::mapping::prepare_mesh_facet(
                                mesh,
                                form,
                                *field,
                                facet,
                                *parent,
                                |point, normal| law.evaluate(point, normal),
                            )?;
                            natural.push((parent.entity.index(), local));
                        }
                    }
                }
            }
        }
        let mapping = RegionDofMap::new(mesh, &layouts, reference, &domains, &traces, &prescribed)?;
        natural.extend(crate::region_assembly::mapping::prepare_interface_facets(
            mesh, &bound, &traces,
        )?);
        let points = super::quadrature_points(reference, selected, &bound)?;
        let quadrature = QuadratureRule::tensor_product_gauss_legendre(dimension, points)?;
        let forms = bound
            .into_iter()
            .map(|form| (form, quadrature.clone()))
            .collect();
        Ok((
            mapping,
            crate::region_assembly::mapping::RegionSolveInput {
                operator_properties: self.finite_element_operator_properties(),
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

/// Static execution and artifact replay share this exact assembled problem.
pub(in crate::numerical_admission) struct StaticRegionSystem<S: Coefficient> {
    mapping: RegionDofMap<S>,
    core: crate::finalized_spatial::FinalizedLinearCore<S>,
    assembly_report: eqiora_assembly::AssemblyReport,
}

impl<S: Coefficient + crate::finalized_spatial::ResidualScalar + Send> StaticRegionSystem<S> {
    pub(super) fn new(
        mapping: RegionDofMap<S>,
        core: crate::finalized_spatial::FinalizedLinearCore<S>,
        assembly_report: eqiora_assembly::AssemblyReport,
    ) -> Self {
        Self {
            mapping,
            core,
            assembly_report,
        }
    }

    pub(in crate::numerical_admission) fn solve(
        self,
        request: LinearSolveRequest<'_, S>,
        initial: Option<&BTreeMap<RawId, Vec<S>>>,
    ) -> Result<CommonLinearRunOutput<S>, Diagnostic> {
        let initial = initial
            .map(|values| {
                let fields = self.mapping.physical_fields_from_coefficients(values)?;
                self.mapping.validated_algebraic_point(&fields)
            })
            .transpose()?;
        let mut problem = self.core.linear_problem()?;
        if let Some(values) = &initial {
            problem = problem.with_initial_guess(values)?;
        }
        let solution = request.solve(&problem)?;
        self.core.validate_solution(&solution)?;
        let (values, solve_report) = solution.into_parts();
        let ids = self
            .mapping
            .keys()
            .map(|key| key.field)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let fields = self
            .mapping
            .recover(&values, &ids)?
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
            fields,
            solve_report,
            assembly_report: self.assembly_report,
            nullspace: None,
        })
    }

    pub(in crate::numerical_admission) fn validate_initial_residual(
        &self,
        initial: Option<&BTreeMap<RawId, Vec<S>>>,
        reported: f64,
    ) -> Result<(), Diagnostic> {
        let reduced = match initial {
            Some(values) => {
                let fields = self.mapping.physical_fields_from_coefficients(values)?;
                self.mapping.validated_algebraic_point(&fields)?
            }
            None => vec![S::zero(); self.mapping.free_count()],
        };
        if self.core.residual_norm(&reduced)? != reported {
            return Err(invalid(
                "static Result initial residual differs from its retained State",
            ));
        }
        Ok(())
    }

    pub(in crate::numerical_admission) fn validate_coefficients(
        &self,
        values: &BTreeMap<RawId, Vec<S>>,
    ) -> Result<(), Diagnostic> {
        let fields = self.mapping.physical_fields_from_coefficients(values)?;
        let reduced = self.mapping.validated_algebraic_point(&fields)?;
        self.core.validate_coefficients(&reduced)?;
        Ok(())
    }
}
