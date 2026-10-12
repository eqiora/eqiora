//! Independent local-operator reference for the physical acceptance test.
use super::*;
use crate::simplicial_fsi::contract::FixedReferenceFsiBoundary;
use crate::simplicial_fsi::element::{fluid_local, solid_local};
use eqiora_assembly::{
    AssemblyBackend, AssemblyPacket, AssemblyPacketSetIdentityV1, TargetAssemblyMap,
};
use eqiora_meshing::{MeshEntity, MeshGeometry};

#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize_fixed_reference_fsi_step_with_packet_set<const D: usize>(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    boundary: &FixedReferenceFsiBoundary<D>,
    previous: &FixedReferenceFsiState<D>,
    config: FixedReferenceFsiStepConfig<D>,
    quadrature: &QuadratureRule,
    packet_set: AssemblyPacketSetIdentityV1,
    assembly: &dyn AssemblyBackend<f64>,
    layout: &FsiLayout<D>,
) -> Result<FinalizedFixedReferenceFsiStep<D>, Diagnostic> {
    layout.require_boundary(boundary)?;
    let prepared = PreparedFixedReferenceFsiAcceptance::new(
        mesh,
        partition,
        previous,
        config,
        quadrature,
        layout.mapping().assembly_plan()?,
        layout.clone(),
    )?;
    let work = ReferenceFsiAssembly {
        prepared,
        packet_set,
    };
    let result = assembly.assemble(work.prepared.plan(), &work)?;
    let reactions = work.prepared.reactions(&work)?;
    work.prepared.finish(result, reactions)
}

#[derive(Debug)]
struct ReferenceFsiAssembly<'a, const D: usize> {
    prepared: PreparedFixedReferenceFsiAcceptance<'a, D>,
    packet_set: AssemblyPacketSetIdentityV1,
}

impl<const D: usize> AssemblyWork<f64> for ReferenceFsiAssembly<'_, D> {
    fn packet_set_identity(&self) -> AssemblyPacketSetIdentityV1 {
        self.packet_set
    }

    fn packet_count(&self) -> usize {
        self.prepared.cell_count
    }

    fn evaluate(&self, packet_index: usize) -> Result<AssemblyPacket<f64>, Diagnostic> {
        let prepared = &self.prepared;
        if packet_index >= prepared.cell_count {
            return Err(invalid(format!(
                "fixed-reference FSI packet {packet_index} is outside cell count {}",
                prepared.cell_count
            )));
        }
        let cell = MeshEntity::new(D, packet_index);
        let geometry = prepared.mesh.geometry_map(cell).ok_or_else(|| {
            invalid(format!(
                "fixed-reference FSI cell packet {packet_index} has no affine geometry"
            ))
        })?;
        let vertices = prepared.mesh.entity_vertices(cell).ok_or_else(|| {
            invalid(format!(
                "fixed-reference FSI cell packet {packet_index} has no vertex closure"
            ))
        })?;
        match prepared.layout.cell_domain(packet_index)? {
            domain if prepared.layout.pressure_field(domain).is_some() => {
                let cell = eqiora_meshing::CellId::new(packet_index);
                let local = fluid_local(
                    &geometry,
                    prepared.quadrature,
                    &prepared.config,
                    &vertices,
                    prepared.previous,
                    cell,
                    prepared
                        .layout
                        .velocity_field(domain)?
                        .downcast()
                        .expect("Field"),
                )?;
                let reduced = prepared.layout.fluid_map(cell, &vertices, true)?;
                let full = prepared.layout.fluid_map(cell, &vertices, false)?;
                AssemblyPacket::new(
                    local,
                    vec![
                        TargetAssemblyMap::new(prepared.target_roles.reduced(), reduced),
                        TargetAssemblyMap::new(prepared.target_roles.full(), full),
                    ],
                )
            }
            domain if prepared.layout.state_field(domain).is_some() => {
                let local = solid_local(
                    &geometry,
                    prepared.quadrature,
                    &prepared.config,
                    &vertices,
                    prepared.previous,
                    prepared
                        .layout
                        .velocity_field(domain)?
                        .downcast()
                        .expect("Field"),
                    prepared
                        .layout
                        .state_field(domain)
                        .expect("state role")
                        .downcast()
                        .expect("Field"),
                )?;
                let reduced = prepared.layout.solid_map(packet_index, &vertices, true)?;
                let full = prepared.layout.solid_map(packet_index, &vertices, false)?;
                AssemblyPacket::new(
                    local,
                    vec![
                        TargetAssemblyMap::new(prepared.target_roles.reduced(), reduced),
                        TargetAssemblyMap::new(prepared.target_roles.full(), full),
                    ],
                )
            }
            _ => Err(invalid(format!(
                "fixed-reference FSI cell packet {packet_index} has no material assignment"
            ))),
        }
    }
}
