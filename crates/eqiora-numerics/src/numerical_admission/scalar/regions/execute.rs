use super::*;
use crate::canonical_boundary::PhysicalBoundaryQuantity;
use crate::region_assembly::mapping::RegionAlgebraicCoordinates;
use crate::region_assembly::mapping::{FieldDof, RegionDofMap, bind_region_topology};
use eqiora_meshing::{CartesianMesh, MeshEntity, MeshTopology};

impl<S: Coefficient + Send + Sync> ExecutableSpatialEquations<S> {
    pub(in crate::numerical_admission) fn finalize_static(
        &self,
        workers: NonZeroUsize,
        solver: eqiora_solver::SolverPlan,
        resources: &NativeMeshResources,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        scaling: &eqiora_realization::SystemScaling,
        operator_properties: LinearOperatorProperties,
    ) -> Result<StaticRegionSystem<S>, Diagnostic> {
        let q1_spd = operator_properties == LinearOperatorProperties::SymmetricPositiveDefinite
            && self.is_real_interval_q1(resources, selected);
        let (mapping, mut input) = self.static_assembly(
            resources,
            selected,
            scaling,
            if q1_spd {
                LinearOperatorProperties::General
            } else {
                operator_properties
            },
        )?;
        if q1_spd {
            let fields = self
                .q1_diffusion_fields(resources, selected, scaling)?
                .ok_or_else(|| {
                    invalid(
                        "bound Parameter point no longer proves the admitted Q1 diffusion property",
                    )
                })?;
            if !self.q1_diffusion_is_anchored(&mapping, &input, &fields)? {
                return Err(invalid(
                    "bound Q1 diffusion operator has an unanchored nodal component",
                ));
            }
            // finalize_linear prepares every actual quadrature coefficient through
            // Term::require_coefficient before constructing a canonical system.
            // Together with rechecked shape/anchoring, this authenticates the
            // bound point's discrete energy instead of copying a stale assertion.
            input.operator_properties = LinearOperatorProperties::SymmetricPositiveDefinite;
        }
        let (core, assembly_report, full) = match resources {
            NativeMeshResources::Cartesian { mesh, .. } => {
                mapping.finalize_linear(mesh.mesh(), &input, workers, solver)?
            }
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => {
                mapping.finalize_linear(mesh.mesh(), &input, workers, solver)?
            }
            _ => return Err(invalid("static system requires its admitted Mesh")),
        };
        Ok(StaticRegionSystem::new(
            mapping,
            core,
            assembly_report,
            StaticAssembly { full, input },
        ))
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
                        || *space == Space::cell_constant()
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
                    conservative: None,
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
        boundary_value: impl FnMut(
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
        let reference = eqiora_meshing::ReferenceCell::hypercube(mesh.topological_dimension())?;
        let bound = self.bind_spaces(reference, selected, scaling)?;
        self.cartesian_assembly_bound(mesh, selected, bound, boundary_value)
    }

    /// Assembly and partial actions retain the same basis bindings and boundary owner.
    pub(in crate::numerical_admission) fn cartesian_assembly_bound(
        &self,
        mesh: &CartesianMesh,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        bound: Vec<crate::form_compiler::region::BoundRegionForm<S>>,
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
        if bound.len() != self.regions.len()
            || self
                .regions
                .iter()
                .zip(&bound)
                .any(|(region, form)| region.form.domain() != form.domain())
        {
            return Err(invalid(
                "bound assembly forms differ from their exact Region inventory",
            ));
        }
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
        let conservative = if selected
            .iter()
            .flat_map(|domain| domain.field_spaces())
            .any(|field| field.space() == Space::cell_constant())
        {
            let region = self.single()?;
            if !traces.is_empty() || !self.interfaces.is_empty() {
                return Err(invalid(
                    "TPFA requires its exact single Region without interface quotients",
                ));
            }
            let mut boundaries = BTreeMap::new();
            for (field, _) in region.form.fields() {
                if !region.form.requires_boundary(*field) {
                    continue;
                }
                for (&(axis, side), boundary) in &region.cartesian()?.boundaries {
                    let law = region
                        .form
                        .boundary_laws()
                        .get(field)
                        .and_then(|laws| laws.get(boundary))
                        .ok_or_else(|| invalid("TPFA requires complete exact boundary laws"))?;
                    boundaries.insert((*field, axis, side), law.clone());
                }
            }
            Some(Box::new(crate::region_assembly::ConservativeFacets {
                mesh: mesh.clone(),
                boundaries,
                primal: None,
            }))
        } else {
            None
        };
        let mut prescribed = BTreeMap::new();
        let mut natural = Vec::new();
        for (region, form) in self.regions.iter().zip(&bound) {
            if conservative.is_some() {
                break;
            }
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
                conservative,
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
mod real;
pub(in crate::numerical_admission) use real::StaticSpatialSystem;

/// Static execution and artifact replay share this exact assembled problem.
pub(in crate::numerical_admission) struct StaticRegionSystem<S: Coefficient, P: Coefficient = S> {
    pub(in crate::numerical_admission) mapping: RegionDofMap<P>,
    pub(in crate::numerical_admission) core: crate::finalized_spatial::FinalizedLinearCore<S>,
    assembly_report: eqiora_assembly::AssemblyReport,
    pub(in crate::numerical_admission) assembly: StaticAssembly<S, P>,
}

pub(in crate::numerical_admission) struct StaticAssembly<S: Coefficient, P: Coefficient = S> {
    pub(in crate::numerical_admission) full: eqiora_assembly::LinearSystem<S>,
    pub(in crate::numerical_admission) input: crate::region_assembly::mapping::RegionSolveInput<P>,
}

impl<S: Coefficient + Send + Sync, P: Coefficient + Send + Sync> StaticRegionSystem<S, P>
where
    RegionDofMap<P>: RegionAlgebraicCoordinates<S, P>,
{
    pub(super) fn new(
        mapping: RegionDofMap<P>,
        core: crate::finalized_spatial::FinalizedLinearCore<S>,
        assembly_report: eqiora_assembly::AssemblyReport,
        assembly: StaticAssembly<S, P>,
    ) -> Self {
        Self {
            mapping,
            core,
            assembly_report,
            assembly,
        }
    }

    pub(in crate::numerical_admission) fn solve(
        self,
        request: LinearSolveRequest<'_, S>,
        initial: Option<&BTreeMap<RawId, Vec<P>>>,
    ) -> Result<CommonLinearRunOutput<P>, Diagnostic> {
        // Ordinary execution needs only the canonical solve and recovery map.
        let initial = initial
            .map(|values| {
                let fields = self.mapping.physical_fields_from_coefficients(values)?;
                self.mapping.validated_coordinates(&fields)
            })
            .transpose()?;
        let mut problem = self.core.linear_problem()?;
        if let Some(values) = &initial {
            problem = problem.with_initial_guess(values)?;
        }
        let solution = request.solve(&problem)?;
        self.core.validate_solution(&solution)?;
        let (values, solve_report) = solution.into_parts();
        self.finish(values, solve_report, None)
    }

    fn finish(
        self,
        values: Vec<S>,
        solve_report: eqiora_solver::SolveReport,
        nullspace: Option<crate::nullspace::NullspaceEvidence>,
    ) -> Result<CommonLinearRunOutput<P>, Diagnostic> {
        let fields = self
            .mapping
            .recover_coordinates(&values)?
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
            nullspace,
        })
    }

    pub(in crate::numerical_admission) fn validate_initial_residual(
        &self,
        initial: Option<&BTreeMap<RawId, Vec<P>>>,
        reported: f64,
    ) -> Result<(), Diagnostic> {
        let reduced = match initial {
            Some(values) => {
                let fields = self.mapping.physical_fields_from_coefficients(values)?;
                self.mapping.validated_coordinates(&fields)?
            }
            None => vec![S::zero(); self.core.canonical_csr_system_view().rows()],
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
        values: &BTreeMap<RawId, Vec<P>>,
    ) -> Result<(), Diagnostic> {
        let fields = self.mapping.physical_fields_from_coefficients(values)?;
        let reduced = self.mapping.validated_coordinates(&fields)?;
        self.core.validate_coefficients(&reduced)?;
        Ok(())
    }
}

impl StaticRegionSystem<f64> {
    pub(in crate::numerical_admission) fn solve_with_reference(
        self,
        request: LinearSolveRequest<'_>,
        reference: &crate::nullspace::NullspaceConstraint,
    ) -> Result<CommonLinearRunOutput<f64>, Diagnostic> {
        let solved = crate::nullspace::solve_canonical_with_nullspace(
            request,
            self.core.canonical_csr_system_view(),
            reference,
        )?;
        self.finish(solved.values, solved.report, Some(solved.evidence))
    }
}
