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
        mapping: &RegionDofMap<f64>,
        forms: &BTreeMap<RawId, crate::form_compiler::region::BoundRegionForm<f64>>,
    ) -> Result<Self, Diagnostic> {
        Self::new(
            mesh,
            partition,
            boundary,
            mapping,
            FsiRoles::derive(equations, plan)?,
            plan.time_step().clone(),
            forms
                .iter()
                .filter(|(_, form)| {
                    plan.time_step().eliminated_states().iter().any(|state| {
                        form.fields()
                            .iter()
                            .any(|field| field.field == state.pair().rate().erase())
                    })
                })
                .map(|(&domain, form)| (domain, form.clone()))
                .collect(),
        )
    }

    pub(crate) fn bind(
        program: &KernelProgram,
        plan: &CoupledFieldwiseRealizationPlan,
        mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        boundary: &FixedReferenceFsiBoundary<D>,
        reference_orientations: &BTreeMap<RawId, f64>,
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
            partition.domains().flat_map(|domain| {
                partition
                    .domain_cells(domain)
                    .expect("exact Domain")
                    .iter()
                    .map(move |&cell| (cell, domain.erase()))
            }),
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
        let mut reference_forms = BTreeMap::new();
        for domain in plan.spatial().domains() {
            if !plan.time_step().eliminated_states().iter().any(|state| {
                domain
                    .field_spaces()
                    .iter()
                    .any(|field| field.field() == state.pair().rate())
            }) {
                continue;
            }
            let form = crate::form_compiler::region::CompiledRegionForm::<f64>::derive(
                program,
                domain.domain().erase(),
                D,
            )?;
            reference_forms.insert(
                domain.domain().erase(),
                form.bind_coupled_plan(reference, plan, reference_orientations)?,
            );
        }
        Self::new(
            mesh,
            partition,
            boundary,
            &mapping,
            roles,
            plan.time_step().clone(),
            reference_forms,
        )
    }
}
