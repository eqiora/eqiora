//! Bind the existing FSI projection from already authenticated Model/Plan support.

use eqiora_meshing::ReferenceCell;
use eqiora_realization::CoupledFieldwiseRealizationPlan;
use eqiora_sem::KernelProgram;
use eqiora_solver::AlgebraicBlock;

use super::*;
use crate::region_assembly::mapping::{bind_region_topology, field_layouts};

impl<const D: usize> FsiLayout<D> {
    pub(crate) fn from_mapping(
        equations: &crate::form_compiler::equation_roles::EquationRoles,
        plan: &CoupledFieldwiseRealizationPlan,
        mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        boundary: &FixedReferenceFsiBoundary<D>,
        mapping: &RegionDofMap,
    ) -> Result<Self, Diagnostic> {
        Self::new(
            mesh,
            partition,
            boundary,
            mapping,
            FsiRoles::derive(equations, plan)?,
        )
    }

    pub(crate) fn bind(
        program: &KernelProgram,
        plan: &CoupledFieldwiseRealizationPlan,
        mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        boundary: &FixedReferenceFsiBoundary<D>,
    ) -> Result<Self, Diagnostic> {
        let equations = crate::form_compiler::equation_roles::EquationRoles::derive(
            program,
            plan.spatial()
                .domains()
                .iter()
                .map(|domain| domain.domain().erase()),
        )?;
        let roles = FsiRoles::derive(&equations, plan)?;
        let scales = plan
            .scaling()
            .block_scales()
            .iter()
            .filter_map(|scale| match scale.block() {
                AlgebraicBlock::Field(field) => Some((field.erase(), scale.scale().quantity())),
                AlgebraicBlock::ConstraintMultiplier { .. } => None,
            })
            .collect();
        let reference = ReferenceCell::simplex(D)?;
        let layouts = field_layouts(program, plan.spatial().domains(), reference, &scales)?;
        let (domains, traces) = bind_region_topology(
            mesh,
            [
                (
                    roles.bindings[&roles.fluid_velocity].0,
                    partition.fluid_cells(),
                ),
                (
                    roles.bindings[&roles.solid_velocity].0,
                    partition.solid_cells(),
                ),
            ]
            .into_iter()
            .flat_map(|(domain, cells)| cells.iter().map(move |&cell| (cell, domain))),
            plan.spatial().trace_quotients(),
        )?;
        let mapping = RegionDofMap::new(
            mesh,
            &layouts,
            reference,
            &domains,
            &traces,
            &BTreeMap::new(),
        )?;
        Self::new(mesh, partition, boundary, &mapping, roles)
    }
}
