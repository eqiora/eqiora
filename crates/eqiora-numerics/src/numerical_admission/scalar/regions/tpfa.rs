//! One assembly owner for TPFA execution and retained-Result checks.
use super::*;
use crate::canonical_boundary::PhysicalBoundaryQuantity;
use crate::spatial_expression::Coefficient;

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
        let row = region.form.conservative_row()?;
        let laws = region
            .form
            .boundary_laws()
            .get(&row.field())
            .ok_or_else(|| invalid("TPFA requires complete boundary laws"))?;
        let mut essential = false;
        for law in laws.values() {
            match law.quantity {
                PhysicalBoundaryQuantity::Trace if law.trace_field == Some(row.field()) => {
                    essential = true;
                }
                PhysicalBoundaryQuantity::Flux if law.trace_field.is_none() => {}
                _ => {
                    return Err(invalid(
                        "TPFA boundary law differs from its exact conservative row",
                    ));
                }
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
    pub(in crate::numerical_admission) fn tpfa_boundary(
        &self,
        component: usize,
        axis: usize,
        side: BoundarySide,
        point: &[f64],
    ) -> Result<CartesianBoundaryValue<S>, Diagnostic> {
        let region = self.single()?;
        let row = region.form.conservative_row()?;
        let boundary = region
            .cartesian()?
            .boundaries
            .get(&(axis, side))
            .ok_or_else(|| invalid("TPFA facet has no exact Boundary support"))?;
        let law = region
            .form
            .boundary_laws()
            .get(&row.field())
            .and_then(|laws| laws.get(boundary))
            .ok_or_else(|| invalid("TPFA facet has no compiled boundary law"))?;
        let mut normal = vec![0.; region.form.dimension()];
        normal[axis] = match side {
            BoundarySide::Lower => -1.,
            BoundarySide::Upper => 1.,
        };
        let values = law.evaluate(point, &normal)?;
        if values.len() != row.components() {
            return Err(invalid(
                "TPFA boundary datum differs from its exact Field components",
            ));
        }
        let value = values
            .get(component)
            .ok_or_else(|| invalid("TPFA boundary component is out of bounds"))?;
        match law.quantity {
            PhysicalBoundaryQuantity::Trace if law.trace_field == Some(row.field()) => {
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
        let (prepared, result) = self.with_tpfa_work(resources, properties, |plan, work| {
            eqiora_assembly::AssemblyBackend::assemble(&REFERENCE_ASSEMBLY_BACKEND, plan, work)
        })?;
        Ok(prepared.finish(result))
    }

    pub(in crate::numerical_admission) fn with_tpfa_work<T>(
        &self,
        resources: &NativeMeshResources,
        properties: LinearOperatorProperties,
        consume: impl FnOnce(
            &eqiora_assembly::AssemblyPlan,
            &dyn eqiora_assembly::AssemblyWork<S>,
        ) -> Result<T, Diagnostic>,
    ) -> Result<
        (
            crate::cartesian_elliptic::PreparedCartesianFvmAssembly<S>,
            T,
        ),
        Diagnostic,
    > {
        let NativeMeshResources::Cartesian { mesh, .. } = resources else {
            return Err(invalid("TPFA requires Cartesian resources"));
        };
        let region = self.single()?;
        let row = region.form.conservative_row()?;
        let source = |component: usize, point: &[f64]| row.source(component, point);
        let coefficient = |point: &[f64]| row.diffusion(point);
        let reaction = |test: usize, trial: usize, point: &[f64]| row.reaction(test, trial, point);
        let boundary =
            |component, axis, side, point: &[f64]| self.tpfa_boundary(component, axis, side, point);
        let cell = QuadratureRule::tensor_product_gauss_legendre(mesh.dimension(), 1)?;
        let facet = if mesh.dimension() == 1 {
            QuadratureRule::point()
        } else {
            QuadratureRule::tensor_product_gauss_legendre(mesh.dimension() - 1, 1)?
        };
        crate::cartesian_elliptic::with_elliptic_cartesian_fvm(
            mesh.mesh(),
            row.components(),
            row.has_diffusion().then_some((&coefficient, &boundary)),
            &reaction,
            &source,
            &cell,
            &facet,
            consume,
            properties,
        )
    }
}

impl<S: Coefficient + crate::finalized_spatial::ResidualScalar + Send>
    ExecutableSpatialEquations<S>
{
    pub(super) fn finalize_tpfa(
        &self,
        workers: NonZeroUsize,
        solver: eqiora_solver::SolverPlan,
        resources: &NativeMeshResources,
        selected: &[eqiora_realization::DomainFieldDiscretization],
        scaling: &eqiora_realization::SystemScaling,
        operator_properties: LinearOperatorProperties,
    ) -> Result<super::execute::StaticRegionSystem<S>, Diagnostic> {
        use crate::region_assembly::mapping::RegionDofMap;
        use eqiora_meshing::{MeshEntity, MeshTopology};
        if !matches!(scaling, eqiora_realization::SystemScaling::Dimensional)
            || selected != self.discretizations(Space::cell_constant(), None)?
        {
            return Err(invalid(
                "TPFA requires its exact unconstrained cell-constant Field layout",
            ));
        }
        let NativeMeshResources::Cartesian { mesh, .. } = resources else {
            return Err(invalid("TPFA requires Cartesian resources"));
        };
        let region = self.single()?;
        let [(field, value_type)] = region.form.fields() else {
            return Err(invalid("TPFA requires one admitted Field"));
        };
        let dimension = mesh.dimension();
        let count = mesh
            .mesh()
            .entity_count(dimension)
            .ok_or_else(|| invalid("TPFA mesh has no cell stratum"))?;
        let components = region.form.conservative_row()?.components();
        let dof_count = count
            .checked_mul(components)
            .ok_or_else(|| invalid("TPFA Field DOF count overflow"))?;
        let layout = crate::form_compiler::region::RegionFieldLayout {
            field: *field,
            value_type: value_type.clone(),
            space: Space::cell_constant(),
            range: 0..components,
            components,
            scale: 1.,
        };
        let mapping = RegionDofMap::new(
            mesh.mesh(),
            &BTreeMap::from([(region.form.domain(), vec![layout])]),
            eqiora_meshing::ReferenceCell::hypercube(dimension)?,
            &vec![region.form.domain(); count],
            &[],
            &BTreeMap::new(),
        )?;
        // The cell-local matrix and the shared exact Field map must use
        // the same coefficient order before sharing algebraic acceptance.
        if mapping.free_count() != dof_count
            || !mapping.keys().enumerate().all(|(index, key)| {
                key.field == *field
                    && key.entity == MeshEntity::new(dimension, index / components)
                    && key.slot == 0
                    && key.component == index % components
            })
        {
            return Err(invalid(
                "TPFA cell order differs from its exact Field mapping",
            ));
        }
        let assembly = self.assemble_tpfa(resources, operator_properties)?;
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
            super::execute::StaticAssembly::FiniteVolume(assembly),
        ))
    }
}
