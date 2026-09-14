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

use super::super::validate::{
    fluid_domain, fluid_velocity, invalid_realization, solid_displacement, solid_domain,
    solid_velocity,
};

#[allow(clippy::too_many_arguments)]
pub(in super::super) fn prepare_cells(
    model: &FixedReferenceFsiCartesianModel2d,
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
        let selected = if domain == fluid_domain(model).erase() {
            partition.fluid_cells()
        } else if domain == solid_domain(model).erase() {
            partition.solid_cells()
        } else {
            return Err(invalid_realization(
                "region Domain has no authenticated mesh partition",
            ));
        };
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
            let vertices = mesh
                .entity_vertices(entity)
                .ok_or_else(|| invalid_realization("region cell has no vertex closure"))?;
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
            for field in form.previous_fields().keys().copied() {
                let values = if field == solid_displacement(model).erase() {
                    previous.solid_displacement()
                } else if field == fluid_velocity(model).erase()
                    || field == solid_velocity(model).erase()
                {
                    previous.vertex_velocity()
                } else {
                    return Err(invalid_realization(
                        "region history has no exact physical State Field",
                    ));
                };
                let mut local = vertices
                    .iter()
                    .flat_map(|vertex| values[vertex.index()])
                    .collect::<Vec<_>>();
                if field == fluid_velocity(model).erase() {
                    local.extend_from_slice(
                        previous
                            .fluid_cell_bubble_velocity()
                            .get(&eqiora_meshing::CellId::new(index))
                            .ok_or_else(|| {
                                invalid_realization("region history omits exact bubble cell")
                            })?,
                    );
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
