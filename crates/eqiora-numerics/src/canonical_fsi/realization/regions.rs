//! Bind Model-derived region equations to the exact resolved numerical choices.

use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::ReferenceCell;
use eqiora_realization::CoupledFieldwiseRealizationPlan;

use crate::canonical_fsi::FixedReferenceFsiCartesianModel2d;
use crate::form_compiler::region::BoundRegionForm;

pub(super) fn bind(
    model: &FixedReferenceFsiCartesianModel2d,
    plan: &CoupledFieldwiseRealizationPlan,
) -> Result<BTreeMap<RawId, BoundRegionForm<f64>>, Diagnostic> {
    let reference = ReferenceCell::simplex(2)?;
    model
        .region_forms
        .iter()
        .map(|(&domain, form)| {
            form.bind_coupled_plan(reference, plan, &model.test_orientations)
                .map(|bound| (domain, bound))
        })
        .collect()
}

/// Project authenticated Model/Plan Fields and geometric partition into the
/// same global map used by every region packet consumer.
pub(super) fn layout(
    model: &FixedReferenceFsiCartesianModel2d,
    forms: &BTreeMap<RawId, BoundRegionForm<f64>>,
    plan: &CoupledFieldwiseRealizationPlan,
    mesh: &eqiora_meshing::SimplicialMesh,
    partition: &crate::simplicial_fsi::FixedReferenceFsiPartition<2>,
    boundary: &crate::simplicial_fsi::FixedReferenceFsiBoundary<2>,
) -> Result<crate::simplicial_fsi::layout::FsiLayout<2>, Diagnostic> {
    use crate::region_assembly::mapping::{RegionDofMap, bind_region_topology};
    let layouts = forms
        .iter()
        .map(|(domain, form)| (*domain, form.fields().to_vec()))
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
        plan.spatial().trace_quotients(),
    )?;
    let mapping = RegionDofMap::<f64>::new(
        mesh,
        &layouts,
        ReferenceCell::simplex(2)?,
        &domains,
        &traces,
        &BTreeMap::new(),
    )?;
    crate::simplicial_fsi::layout::FsiLayout::from_mapping(
        &model.equation_roles,
        plan,
        mesh,
        partition,
        boundary,
        &mapping,
        forms,
    )
}
