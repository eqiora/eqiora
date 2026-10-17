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
        let row = region.form.scalar_conservative_row()?;
        let laws = region
            .form
            .boundary_laws()
            .get(&row.field())
            .ok_or_else(|| invalid("TPFA requires complete scalar boundary laws"))?;
        let mut essential = false;
        for law in laws.values() {
            match law.quantity {
                PhysicalBoundaryQuantity::Trace if law.trace_field == Some(row.field()) => {
                    essential = true;
                }
                PhysicalBoundaryQuantity::Flux if law.trace_field.is_none() => {}
                _ => {
                    return Err(invalid(
                        "TPFA boundary law differs from its exact scalar row",
                    ));
                }
            }
        }
        if constraint.is_some() {
            if S::DOMAIN != eqiora_core::ScalarDomain::Real
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
    pub(in crate::numerical_admission) fn assemble_scalar_tpfa(
        &self,
        resources: &NativeMeshResources,
        properties: LinearOperatorProperties,
    ) -> Result<crate::cartesian_elliptic::FinalizedCartesianFvmAssembly<S>, Diagnostic> {
        let NativeMeshResources::Cartesian { mesh, .. } = resources else {
            return Err(invalid("TPFA requires Cartesian resources"));
        };
        let region = self.single()?;
        let support = region.cartesian()?;
        let row = region.form.scalar_conservative_row()?;
        let source = |point: &[f64]| row.source(point);
        let coefficient = |point: &[f64]| row.diffusion(point);
        let reaction = |point: &[f64]| row.reaction(point);
        let boundary = |axis: usize, side: BoundarySide, point: &[f64]| {
            let boundary = support
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
            let [value] = values.as_slice() else {
                return Err(invalid("TPFA boundary datum is not scalar"));
            };
            match law.quantity {
                PhysicalBoundaryQuantity::Trace if law.trace_field == Some(row.field()) => {
                    Ok(CartesianBoundaryValue::Essential(*value))
                }
                PhysicalBoundaryQuantity::Flux if law.trace_field.is_none() => {
                    Ok(CartesianBoundaryValue::Natural(*value))
                }
                _ => Err(invalid(
                    "TPFA boundary law differs from its exact scalar row",
                )),
            }
        };
        let cell = QuadratureRule::tensor_product_gauss_legendre(mesh.dimension(), 1)?;
        let facet = if mesh.dimension() == 1 {
            QuadratureRule::point()
        } else {
            QuadratureRule::tensor_product_gauss_legendre(mesh.dimension() - 1, 1)?
        };
        finalize_scalar_elliptic_cartesian_fvm(
            mesh.mesh(),
            &coefficient,
            &reaction,
            &source,
            &boundary,
            &cell,
            &facet,
            &REFERENCE_ASSEMBLY_BACKEND,
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
            return Err(invalid("TPFA requires one admitted scalar Field"));
        };
        let dimension = mesh.dimension();
        let count = mesh
            .mesh()
            .entity_count(dimension)
            .ok_or_else(|| invalid("TPFA mesh has no cell stratum"))?;
        let layout = crate::form_compiler::region::RegionFieldLayout {
            field: *field,
            value_type: value_type.clone(),
            space: Space::cell_constant(),
            range: 0..1,
            components: 1,
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
        if mapping.free_count() != count
            || !mapping.keys().enumerate().all(|(index, key)| {
                key.field == *field
                    && key.entity == MeshEntity::new(dimension, index)
                    && key.slot == 0
                    && key.component == 0
            })
        {
            return Err(invalid(
                "TPFA cell order differs from its exact Field mapping",
            ));
        }
        let assembly = self.assemble_scalar_tpfa(resources, operator_properties)?;
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
        ))
    }
}
