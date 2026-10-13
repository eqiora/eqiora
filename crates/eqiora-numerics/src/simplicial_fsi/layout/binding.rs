//! Bind the existing FSI projection from already authenticated Model/Plan support.

use crate::form_compiler::region::RegionGraph;
use eqiora_meshing::ReferenceCell;
use eqiora_realization::PortableRealizationGraph;
use eqiora_sem::KernelProgram;

use super::*;
use crate::region_assembly::mapping::bind_region_topology;

impl<const D: usize> FsiLayout<D> {
    pub(crate) fn from_mapping(
        equations: &crate::form_compiler::equation_roles::EquationRoles,
        plan: &RegionGraph<'_>,
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
            plan.kinematic_step()?,
            forms.clone(),
        )
    }

    pub(crate) fn bind(
        program: &KernelProgram,
        graph: &PortableRealizationGraph,
        mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        boundary: &FixedReferenceFsiBoundary<D>,
        orientations: &BTreeMap<RawId, f64>,
    ) -> Result<Self, Diagnostic> {
        let plan = RegionGraph::new(graph)?;
        let equations =
            crate::form_compiler::equation_roles::EquationRoles::derive(program, plan.domains())?;
        let reference = ReferenceCell::simplex(D)?;
        let mut region_forms = BTreeMap::new();
        for domain in plan.domains() {
            let form = crate::form_compiler::region::CompiledRegionForm::<f64>::derive(
                program, domain, D,
            )?;
            region_forms.insert(domain, plan.bind(&form, reference, orientations)?);
        }
        let layouts = region_forms
            .iter()
            .map(|(&domain, form)| (domain, form.fields().to_vec()))
            .collect();
        let (domains, traces) = bind_region_topology(
            mesh,
            partition.domains().flat_map(|domain| {
                partition
                    .domain_cells(domain)
                    .expect("exact Domain")
                    .iter()
                    .map(move |&cell| (cell, domain.erase()))
            }),
            &plan.quotients,
        )?;
        let mapping = RegionDofMap::new(
            mesh,
            &layouts,
            reference,
            &domains,
            &traces,
            &BTreeMap::new(),
        )?;
        Self::from_mapping(
            &equations,
            &plan,
            mesh,
            partition,
            boundary,
            &mapping,
            &region_forms,
        )
    }
}
