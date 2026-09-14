//! Project authenticated FSI support and state into ordinary region cell packets.

use std::collections::BTreeMap;

use eqiora_assembly::{AssemblyPacketSetIdentityV1, TargetAssemblyMap};
use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::{MeshEntity, MeshGeometry, QuadratureRule, SimplicialMesh};

use crate::canonical_fsi::FixedReferenceFsiCartesianModel2d;
use crate::form_compiler::region::BoundRegionForm;
use crate::region_assembly::{PreparedRegionAssembly, RegionAssemblyCell};
use crate::simplicial_fsi::{
    FixedReferenceFsiPartition, FixedReferenceFsiState, PreparedFixedReferenceFsiAssembly,
};

use super::super::validate::invalid_realization;

#[allow(clippy::too_many_arguments)]
pub(in super::super) fn prepare_cells(
    _model: &FixedReferenceFsiCartesianModel2d,
    forms: &BTreeMap<RawId, BoundRegionForm>,
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
    previous: &FixedReferenceFsiState<2>,
    quadrature: &QuadratureRule,
    prepared: &PreparedFixedReferenceFsiAssembly<'_, 2>,
    packet_set: AssemblyPacketSetIdentityV1,
) -> Result<PreparedRegionAssembly, Diagnostic> {
    let mut domains = vec![None; partition.cell_count()];
    let mut cells = Vec::new();
    for (&domain, form) in forms {
        let selected = partition
            .domain_cells(domain.downcast().expect("Domain"))
            .ok_or_else(|| invalid_realization("region Domain has no exact partition support"))?;
        for cell in selected {
            let index = cell.index();
            if domains[index].replace(domain).is_some() {
                return Err(invalid_realization(
                    "region cell has multiple Domain owners",
                ));
            }
            let entity = MeshEntity::new(2, index);
            let geometry = mesh
                .geometry_map(entity)
                .ok_or_else(|| invalid_realization("region cell has no affine geometry"))?;
            let maps = [true, false]
                .into_iter()
                .map(|reduced| {
                    let target = if reduced {
                        prepared.target_roles().reduced()
                    } else {
                        prepared.target_roles().full()
                    };
                    Ok(TargetAssemblyMap::new(
                        target,
                        prepared.layout().cell_map(index, reduced)?,
                    ))
                })
                .collect::<Result<Vec<_>, Diagnostic>>()?;
            let mut history = BTreeMap::new();
            for (&field, layout) in form.previous_fields() {
                let physical = previous.fields.get(&field).ok_or_else(|| {
                    invalid_realization("region history omits exact physical Field")
                })?;
                if physical.domain != domain || physical.value_type != layout.value_type {
                    return Err(invalid_realization(
                        "region history has stale Domain or ValueType",
                    ));
                }
                let algebraic = if prepared.layout().mapping().field_layout(field).is_some() {
                    field
                } else {
                    prepared
                        .layout()
                        .state_bindings()
                        .iter()
                        .find(|binding| binding.pair().state().erase() == field)
                        .ok_or_else(|| {
                            invalid_realization("history has no exact state/rate binding")
                        })?
                        .pair()
                        .rate()
                        .erase()
                };
                let local = prepared
                    .layout()
                    .mapping()
                    .cell_field_keys(index, algebraic)?
                    .into_iter()
                    .map(|key| {
                        physical
                            .coefficients
                            .get(&crate::region_assembly::mapping::FieldDof { field, ..key })
                            .copied()
                            .ok_or_else(|| {
                                invalid_realization("region history omits exact local coordinate")
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if local.len() != layout.range.len() {
                    return Err(invalid_realization(
                        "region history differs from exact local Field layout",
                    ));
                }
                history.insert(field, local);
            }
            cells.push(RegionAssemblyCell {
                index,
                geometry,
                mappings: maps,
                previous: history,
            });
        }
    }
    let domains = domains
        .into_iter()
        .map(|domain| {
            domain.ok_or_else(|| invalid_realization("region partition omits a mesh cell"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    PreparedRegionAssembly::new(
        packet_set,
        prepared.plan(),
        forms
            .values()
            .cloned()
            .map(|form| (form, quadrature.clone()))
            .collect(),
        &domains,
        cells,
        Vec::new(),
    )
}
