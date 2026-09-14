//! Bind Model-derived region equations to the exact resolved numerical choices.

use eqiora_solver::AlgebraicBlock;
use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, DynQuantity, RawId};
use eqiora_meshing::ReferenceCell;
use eqiora_realization::CoupledFieldwiseRealizationPlan;

use crate::canonical_fsi::FixedReferenceFsiCartesianModel2d;
use crate::form_compiler::region::{BoundRegionForm, RegionFieldBinding, RegionTimeBinding};

use super::validate::invalid_realization;

mod cells;
pub(super) use cells::prepare_cells;

pub(super) fn bind(
    model: &FixedReferenceFsiCartesianModel2d,
    plan: &CoupledFieldwiseRealizationPlan,
) -> Result<BTreeMap<RawId, BoundRegionForm>, Diagnostic> {
    let reference = ReferenceCell::simplex(2)?;
    let functional = plan.scaling().weak_functional_scale().quantity();
    model
        .region_forms
        .iter()
        .map(|(&domain, form)| {
            let spatial = plan
                .spatial()
                .domains()
                .iter()
                .find(|entry| entry.domain().erase() == domain)
                .ok_or_else(|| invalid_realization("region has no exact Plan Domain binding"))?;
            let fields = form
                .fields()
                .map(|(field, _)| {
                    let binding = spatial
                        .field_spaces()
                        .iter()
                        .find(|binding| binding.field().erase() == field)
                        .ok_or_else(|| invalid_realization("region Field has no Plan space"))?;
                    let scale = plan
                        .scaling()
                        .block_scales()
                        .iter()
                        .find(|scale| scale.block() == AlgebraicBlock::Field(binding.field()))
                        .ok_or_else(|| invalid_realization("region Field has no Plan scale"))?;
                    Ok(RegionFieldBinding {
                        field,
                        space: binding.space(),
                        scale: scale.scale().quantity(),
                    })
                })
                .collect::<Result<Vec<_>, Diagnostic>>()?;
            let rows = form
                .rows()
                .map(|(relation, tested, _)| {
                    let scale = fields
                        .iter()
                        .find(|binding| binding.field == tested)
                        .expect("every residual owns its exact test Field")
                        .scale
                        .try_div(functional)?;
                    // The admitted symmetric mixed formulation tests div(v)=0
                    // with -p, while momentum uses the positive velocity test.
                    // Preserve the recognizer's orientation of the entire
                    // momentum residual, including its forcing and history.
                    let sign = *model.test_orientations.get(&tested).ok_or_else(|| {
                        invalid_realization("region row has no exact admitted test orientation")
                    })?;
                    Ok((
                        relation,
                        DynQuantity::new(sign * scale.value(), scale.dim()),
                    ))
                })
                .collect::<Result<BTreeMap<_, _>, Diagnostic>>()?;
            let time = RegionTimeBinding {
                step: plan.time_step().duration(),
                states: plan
                    .time_step()
                    .eliminated_states()
                    .iter()
                    .copied()
                    .filter(|state| {
                        fields
                            .iter()
                            .any(|field| field.field == state.pair().rate().erase())
                    })
                    .collect(),
            };
            form.bind(reference, &fields, &rows, Some(&time))
                .map(|bound| (domain, bound))
        })
        .collect()
}

/// Project authenticated Model/Plan Fields and geometric partition into the
/// same global map used by every region packet consumer.
pub(super) fn layout(
    model: &FixedReferenceFsiCartesianModel2d,
    forms: &BTreeMap<RawId, BoundRegionForm>,
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
    let mapping = RegionDofMap::new(
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
    )
}
