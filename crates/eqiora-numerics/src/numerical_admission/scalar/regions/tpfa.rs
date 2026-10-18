//! One assembly owner for TPFA execution and retained-Result checks.
use super::*;
use crate::canonical_boundary::PhysicalBoundaryQuantity;
use crate::form_compiler::region::coordinates::{Native, Projection};
use crate::region_assembly::mapping::{FieldDof, RegionDofMap};
use crate::spatial_expression::Coefficient;
use eqiora_meshing::{MeshEntity, MeshTopology};

impl<S: Coefficient> ExecutableSpatialEquations<S> {
    pub(in crate::numerical_admission) fn tpfa_operator_properties(
        &self,
        constraint: Option<eqiora_solver::AlgebraicConstraint>,
    ) -> Result<LinearOperatorProperties, Diagnostic> {
        let region = self.single()?;
        region.cartesian()?;
        if !self.interfaces.is_empty() {
            return Err(invalid(
                "TPFA requires one Region without interface quotients",
            ));
        }
        let row = region.form.conservative_system()?;
        let mut essential = false;
        for (field, _) in row.fields() {
            let laws = region
                .form
                .boundary_laws()
                .get(&field)
                .ok_or_else(|| invalid("TPFA requires complete boundary laws"))?;
            let mut field_essential = false;
            for law in laws.values() {
                match law.quantity {
                    PhysicalBoundaryQuantity::Trace if law.trace_field == Some(field) => {
                        field_essential = true
                    }
                    PhysicalBoundaryQuantity::Flux if law.trace_field.is_none() => {}
                    _ => {
                        return Err(invalid(
                            "TPFA boundary law differs from its exact conservative row",
                        ));
                    }
                }
            }
            essential |= field_essential;
            if constraint.is_none() && !field_essential && !row.has_reaction() {
                return Err(invalid(
                    "Cartesian TPFA system requires an essential boundary or explicit spatial reference",
                ));
            }
        }
        if constraint.is_some() {
            if row.components() != 1
                || S::DOMAIN != eqiora_core::ScalarDomain::Real
                || region.form.dimension() != 1
                || essential
                || row.has_reaction()
                || !row.has_positive_diffusion()
            {
                return Err(invalid(
                    "TPFA spatial reference requires one real pure-Neumann diffusion row",
                ));
            }
            return Ok(LinearOperatorProperties::SymmetricIndefinite);
        }
        if !essential && !row.has_reaction() {
            return Err(invalid(
                "Cartesian TPFA system requires an essential boundary or explicit spatial reference",
            ));
        }
        Ok(if !row.has_reaction() && row.has_positive_diffusion() {
            LinearOperatorProperties::SymmetricPositiveDefinite
        } else {
            LinearOperatorProperties::General
        })
    }
}

impl<S: Coefficient + Send + Sync> ExecutableSpatialEquations<S> {
    fn tpfa_mapping(&self, resources: &NativeMeshResources) -> Result<RegionDofMap<S>, Diagnostic> {
        let NativeMeshResources::Cartesian { mesh, .. } = resources else {
            return Err(invalid("TPFA requires Cartesian resources"));
        };
        let region = self.single()?;
        let system = region.form.conservative_system()?;
        let count = mesh
            .mesh()
            .entity_count(mesh.dimension())
            .ok_or_else(|| invalid("TPFA mesh has no cells"))?;
        let mut offset = 0usize;
        let layouts = system
            .fields()
            .map(|(field, components)| {
                let value_type = &region
                    .form
                    .fields()
                    .iter()
                    .find(|(id, _)| *id == field)
                    .ok_or_else(|| invalid("TPFA Field is absent from its compiled inventory"))?
                    .1;
                let end = offset
                    .checked_add(components)
                    .ok_or_else(|| invalid("TPFA local Field layout overflow"))?;
                let layout = crate::form_compiler::region::RegionFieldLayout {
                    field,
                    value_type: value_type.clone(),
                    space: Space::cell_constant(),
                    range: offset..end,
                    components,
                    scale: 1.,
                };
                offset = end;
                Ok(layout)
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        RegionDofMap::new(
            mesh.mesh(),
            &BTreeMap::from([(region.form.domain(), layouts)]),
            eqiora_meshing::ReferenceCell::hypercube(mesh.dimension())?,
            &vec![region.form.domain(); count],
            &[],
            &BTreeMap::new(),
        )
    }

    pub(super) fn static_tpfa_mapping(
        &self,
        resources: &NativeMeshResources,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        scaling: &eqiora_realization::SystemScaling,
    ) -> Result<RegionDofMap<S>, Diagnostic> {
        if !matches!(scaling, eqiora_realization::SystemScaling::Dimensional)
            || selected != self.discretizations(Space::cell_constant(), None)?
        {
            return Err(invalid(
                "TPFA requires its exact unconstrained cell-constant Field layout",
            ));
        }
        self.tpfa_mapping(resources)
    }

    fn tpfa_boundary(
        &self,
        field: RawId,
        component: usize,
        axis: usize,
        side: BoundarySide,
        point: &[f64],
    ) -> Result<CartesianBoundaryValue<S>, Diagnostic> {
        let region = self.single()?;
        let row = region.form.conservative_system()?;
        let boundary = region
            .cartesian()?
            .boundaries
            .get(&(axis, side))
            .ok_or_else(|| invalid("TPFA facet has no exact Boundary support"))?;
        let law = region
            .form
            .boundary_laws()
            .get(&field)
            .and_then(|laws| laws.get(boundary))
            .ok_or_else(|| invalid("TPFA facet has no compiled boundary law"))?;
        let mut normal = vec![0.; region.form.dimension()];
        normal[axis] = match side {
            BoundarySide::Lower => -1.,
            BoundarySide::Upper => 1.,
        };
        let values = law.evaluate(point, &normal)?;
        if values.len()
            != row
                .fields()
                .find(|(id, _)| *id == field)
                .expect("bound conservative Field")
                .1
        {
            return Err(invalid(
                "TPFA boundary datum differs from its exact Field components",
            ));
        }
        let value = values
            .get(component)
            .ok_or_else(|| invalid("TPFA boundary component is out of bounds"))?;
        match law.quantity {
            PhysicalBoundaryQuantity::Trace if law.trace_field == Some(field) => {
                Ok(CartesianBoundaryValue::Essential(*value))
            }
            PhysicalBoundaryQuantity::Flux if law.trace_field.is_none() => {
                Ok(CartesianBoundaryValue::Natural(*value))
            }
            _ => Err(invalid(
                "TPFA boundary law differs from its exact conservative row",
            )),
        }
    }

    pub(in crate::numerical_admission) fn assemble_tpfa(
        &self,
        resources: &NativeMeshResources,
        properties: LinearOperatorProperties,
    ) -> Result<crate::cartesian_elliptic::FinalizedCartesianFvmAssembly<S>, Diagnostic> {
        let mapping = self.tpfa_mapping(resources)?;
        let (prepared, result) = self.with_tpfa_work::<Native, _>(
            resources,
            &mapping,
            properties,
            None,
            |plan, work| {
                eqiora_assembly::AssemblyBackend::assemble(&REFERENCE_ASSEMBLY_BACKEND, plan, work)
            },
        )?;
        Ok(prepared.finish(result))
    }

    pub(in crate::numerical_admission) fn with_tpfa_work<P: Projection<S>, T>(
        &self,
        resources: &NativeMeshResources,
        mapping: &RegionDofMap<S>,
        properties: LinearOperatorProperties,
        partial: Option<&Self>,
        consume: impl FnOnce(
            &eqiora_assembly::AssemblyPlan,
            &dyn eqiora_assembly::AssemblyWork<P::Scalar>,
        ) -> Result<T, Diagnostic>,
    ) -> Result<
        (
            crate::cartesian_elliptic::PreparedCartesianFvmAssembly<P::Scalar>,
            T,
        ),
        Diagnostic,
    >
    where
        P::Scalar: Send + Sync,
    {
        let NativeMeshResources::Cartesian { mesh, .. } = resources else {
            return Err(invalid("TPFA requires Cartesian resources"));
        };
        let region = self.single()?;
        let row = region.form.conservative_system()?.project::<P>()?;
        let partial_row = partial
            .map(|equations| {
                equations
                    .single()?
                    .form
                    .conservative_system()?
                    .project::<P>()
            })
            .transpose()?;
        let volume = partial_row.as_ref().unwrap_or(&row);
        let source = |component: usize, point: &[f64]| volume.source(component, point);
        let coefficient = |test, trial, point: &[f64]| row.diffusion(test, trial, point);
        let reaction =
            |test: usize, trial: usize, point: &[f64]| volume.reaction(test, trial, point);
        let boundary_value = |equations: &Self, component, axis, side, point: &[f64]| {
            let (field, physical, part) = row.coordinate(component)?;
            Ok(
                match equations.tpfa_boundary(field, physical, axis, side, point)? {
                    CartesianBoundaryValue::Essential(value) => {
                        CartesianBoundaryValue::Essential(P::component(value, part))
                    }
                    CartesianBoundaryValue::Natural(value) => {
                        CartesianBoundaryValue::Natural(P::component(value, part))
                    }
                },
            )
        };
        let boundary = |component, axis, side, point: &[f64]| {
            boundary_value(self, component, axis, side, point)
        };
        let coefficient_partial = |test, trial, point: &[f64]| {
            partial_row
                .as_ref()
                .expect("selected partial")
                .diffusion(test, trial, point)
        };
        let boundary_partial = |component, axis, side, point: &[f64]| {
            boundary_value(
                partial.expect("selected partial"),
                component,
                axis,
                side,
                point,
            )
        };
        let mut flux = crate::cartesian_elliptic::FvmFlux::new(&coefficient, &boundary);
        if partial.is_some() {
            flux = flux.with_direction(&coefficient_partial, &boundary_partial);
        }
        let cell = QuadratureRule::tensor_product_gauss_legendre(mesh.dimension(), 1)?;
        let facet = if mesh.dimension() == 1 {
            QuadratureRule::point()
        } else {
            QuadratureRule::tensor_product_gauss_legendre(mesh.dimension() - 1, 1)?
        };
        let dimension = mesh.dimension();
        let count = mesh
            .mesh()
            .entity_count(dimension)
            .ok_or_else(|| invalid("TPFA mesh has no cells"))?;
        let cell_dofs = (0..count)
            .map(|cell| {
                (0..row.components())
                    .map(|component| {
                        let (field, component, part) = row.coordinate(component)?;
                        let width = P::width(
                            &mapping
                                .field_layout(field)
                                .expect("mapped Field")
                                .1
                                .value_type,
                        )?;
                        mapping
                            .free_dof(FieldDof {
                                field,
                                entity: MeshEntity::new(dimension, cell),
                                slot: 0,
                                component,
                            })
                            // Conservative rows share one physical scalar domain;
                            // expand storage within each physical coefficient only.
                            .and_then(|physical| {
                                physical.index().checked_mul(width)?.checked_add(part)
                            })
                            .map(eqiora_assembly::DofId::new)
                            .ok_or_else(|| {
                                invalid(
                                    "TPFA coordinate is absent from its exact free Field layout",
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, Diagnostic>>()
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        crate::cartesian_elliptic::with_elliptic_cartesian_fvm(
            mesh.mesh(),
            row.components(),
            &cell_dofs,
            row.has_diffusion().then_some(flux),
            &reaction,
            &source,
            &cell,
            &facet,
            consume,
            properties,
        )
    }
}

impl<S: Coefficient + Send + Sync> ExecutableSpatialEquations<S> {
    pub(super) fn finalize_tpfa(
        &self,
        workers: NonZeroUsize,
        solver: eqiora_solver::SolverPlan,
        resources: &NativeMeshResources,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        scaling: &eqiora_realization::SystemScaling,
        operator_properties: LinearOperatorProperties,
    ) -> Result<super::execute::StaticRegionSystem<S>, Diagnostic> {
        let mapping = self.static_tpfa_mapping(resources, selected, scaling)?;
        let (prepared, result) = self.with_tpfa_work::<Native, _>(
            resources,
            &mapping,
            operator_properties,
            None,
            |plan, work| {
                eqiora_assembly::AssemblyBackend::assemble(&REFERENCE_ASSEMBLY_BACKEND, plan, work)
            },
        )?;
        let assembly = prepared.finish(result);
        let core = crate::finalized_spatial::FinalizedLinearCore::new(
            solver,
            eqiora_realization::VectorLayoutKind::Replicated,
            eqiora_realization::Target::HostCpu { threads: workers },
            assembly.canonical_system()?,
        );
        Ok(super::execute::StaticRegionSystem::new(
            mapping,
            core,
            assembly.assembly_report().clone(),
            super::execute::StaticAssembly::FiniteVolume,
        ))
    }
}
