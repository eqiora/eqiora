//! Bind Model-derived region equations to the exact resolved numerical choices.

use eqiora_solver::AlgebraicBlock;
use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, DynQuantity, RawId};
use eqiora_meshing::ReferenceCell;
use eqiora_realization::CoupledFieldwiseRealizationPlan;

use crate::canonical_fsi::FixedReferenceFsiCartesianModel2d;
use crate::form_compiler::region::{BoundRegionForm, RegionFieldBinding, RegionTimeBinding};

use super::validate::{fluid_pressure, fluid_velocity, invalid_realization, solid_velocity};

mod cells;
pub(super) use cells::prepare_cells;

pub(super) fn bind(
    model: &FixedReferenceFsiCartesianModel2d,
    plan: &CoupledFieldwiseRealizationPlan,
) -> Result<BTreeMap<RawId, BoundRegionForm>, Diagnostic> {
    let reference = ReferenceCell::simplex(2)?;
    let functional = plan.scaling().weak_functional_scale().quantity();
    let state = plan.time_step().eliminated_state();
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
                    let sign = if tested == fluid_pressure(model).erase() {
                        -1.0
                    } else if tested == fluid_velocity(model).erase() {
                        model.fluid.momentum_orientation()
                    } else if tested == solid_velocity(model).erase() {
                        model.solid.momentum_orientation()
                    } else {
                        return Err(invalid_realization(
                            "region row has no admitted test orientation",
                        ));
                    };
                    Ok((
                        relation,
                        DynQuantity::new(sign * scale.value(), scale.dim()),
                    ))
                })
                .collect::<Result<BTreeMap<_, _>, Diagnostic>>()?;
            let time = RegionTimeBinding {
                step: plan.time_step().duration(),
                states: fields
                    .iter()
                    .any(|field| field.field == state.pair().rate().erase())
                    .then_some(state)
                    .into_iter()
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
    mesh: &eqiora_meshing::SimplicialMesh,
    partition: &crate::simplicial_fsi::FixedReferenceFsiPartition<2>,
    boundary: &crate::simplicial_fsi::FixedReferenceFsiBoundary<2>,
) -> Result<crate::simplicial_fsi::layout::FsiLayout<2>, Diagnostic> {
    use super::validate::{fluid_domain, solid_domain, trace_quotient};
    use crate::region_assembly::mapping::{RegionDofMap, TraceBinding, TraceFacet};
    use eqiora_meshing::{MeshEntity, MeshTopology};
    let layouts = forms
        .iter()
        .map(|(domain, form)| (*domain, form.fields().to_vec()))
        .collect();
    let mut domains = vec![None; partition.cell_count()];
    for (domain, cells) in [
        (fluid_domain(model).erase(), partition.fluid_cells()),
        (solid_domain(model).erase(), partition.solid_cells()),
    ] {
        for cell in cells {
            domains[cell.index()] = Some(domain);
        }
    }
    let domains = domains
        .into_iter()
        .map(|domain| {
            domain.ok_or_else(|| invalid_realization("authenticated partition omits a Domain cell"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let quotient = trace_quotient(model);
    let facets = partition
        .interface_facets()
        .iter()
        .map(|facet| {
            let facet = MeshEntity::new(1, facet.index());
            let incidence = mesh
                .incidence(facet, 2)
                .ok_or_else(|| invalid_realization("interface has no exact parent incidence"))?;
            let mut sides = Vec::new();
            for endpoint in quotient.endpoints() {
                sides.push(
                    *incidence
                        .iter()
                        .find(|side| domains[side.entity.index()] == endpoint.domain().erase())
                        .ok_or_else(|| {
                            invalid_realization("interface has no exact Domain owner")
                        })?,
                );
            }
            Ok(TraceFacet {
                facet,
                sides: sides.try_into().expect("two quotient endpoints"),
            })
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    let mapping = RegionDofMap::new(
        mesh,
        &layouts,
        ReferenceCell::simplex(2)?,
        &domains,
        &[TraceBinding { quotient, facets }],
        &BTreeMap::new(),
    )?;
    crate::simplicial_fsi::layout::FsiLayout::new(
        mesh,
        partition,
        boundary,
        &mapping,
        [
            fluid_velocity(model).erase(),
            fluid_pressure(model).erase(),
            solid_velocity(model).erase(),
        ],
    )
}
