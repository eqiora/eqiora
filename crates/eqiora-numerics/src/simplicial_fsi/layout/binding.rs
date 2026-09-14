//! Bind the existing FSI projection from already authenticated Model/Plan support.

use eqiora_meshing::ReferenceCell;
use eqiora_realization::CoupledFieldwiseRealizationPlan;
use eqiora_sem::KernelProgram;
use eqiora_solver::AlgebraicBlock;

use super::*;
use crate::region_assembly::mapping::{bind_region_topology, field_layouts};

impl<const D: usize> FsiLayout<D> {
    pub(crate) fn bind(
        program: &KernelProgram,
        plan: &CoupledFieldwiseRealizationPlan,
        mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        boundary: &FixedReferenceFsiBoundary<D>,
        fields: [RawId; 3],
    ) -> Result<Self, Diagnostic> {
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
        let domain = |field| {
            plan.spatial()
                .domains()
                .iter()
                .find(|domain| {
                    domain
                        .field_spaces()
                        .iter()
                        .any(|binding| binding.field().erase() == field)
                })
                .map(|domain| domain.domain().erase())
                .ok_or_else(|| invalid("FSI role has no exact Plan Domain"))
        };
        let (domains, traces) = bind_region_topology(
            mesh,
            [
                (domain(fields[0])?, partition.fluid_cells()),
                (domain(fields[2])?, partition.solid_cells()),
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
        Self::new(mesh, partition, boundary, &mapping, fields)
    }
}
