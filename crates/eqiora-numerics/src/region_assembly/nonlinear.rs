//! Retain primal residuals with their assembled Jacobians, independent of Field roles.
use std::sync::OnceLock;

use eqiora_assembly::{
    AssemblyBackend, AssemblyPacket, AssemblyPacketSetIdentityV1, AssemblyPlan, AssemblyReport,
    IndexedAssemblyWork, LinearSystem,
};
use eqiora_core::Diagnostic;

use super::{invalid, residual::scatter_residual};

#[derive(Clone)]
pub(crate) struct EvaluatedRegionPacket {
    pub(crate) assembly: AssemblyPacket<f64>,
    pub(crate) residual: Vec<f64>,
}

pub(crate) struct RegionAssemblyLinearization {
    pub(crate) systems: Vec<LinearSystem<f64>>,
    pub(crate) residuals: Vec<Vec<f64>>,
    pub(crate) report: AssemblyReport,
    pub(crate) packets: Vec<EvaluatedRegionPacket>,
}

/// Evaluate each local action once, even if a backend requests it repeatedly.
/// Residuals are scattered directly in logical order, never recovered by J*x-b.
/// An error leaves no partial assembly or residual available to the caller.
pub(crate) fn assemble_linearization(
    plan: &AssemblyPlan,
    packet_set: AssemblyPacketSetIdentityV1,
    packet_count: usize,
    backend: &dyn AssemblyBackend<f64>,
    evaluate: impl Fn(usize) -> Result<EvaluatedRegionPacket, Diagnostic> + Sync,
) -> Result<RegionAssemblyLinearization, Diagnostic> {
    let evaluated = (0..packet_count)
        .map(|_| OnceLock::<Result<EvaluatedRegionPacket, Diagnostic>>::new())
        .collect::<Vec<_>>();
    let work = IndexedAssemblyWork::for_packet_set(packet_set, packet_count, |index: usize| {
        evaluated[index]
            .get_or_init(|| {
                let packet = evaluate(index)?;
                if packet.residual.len() != packet.assembly.local().rows()
                    || packet.residual.iter().any(|value| !value.is_finite())
                {
                    return Err(invalid(
                        "region packet requires an exact finite primal residual",
                    ));
                }
                Ok(packet)
            })
            .as_ref()
            .map(|packet| packet.assembly.clone())
            .map_err(Clone::clone)
    });
    let (systems, report) = backend.assemble(plan, &work)?.into_parts();
    let packets = evaluated
        .into_iter()
        .map(|value| {
            value
                .into_inner()
                .ok_or_else(|| invalid("assembly backend omitted a prepared region packet"))?
        })
        .collect::<Result<Vec<_>, _>>()?;
    if systems.len() != plan.target_count() {
        return Err(invalid(
            "region assembly returned a foreign target inventory",
        ));
    }
    let mut residuals = Vec::with_capacity(plan.target_count());
    for index in 0..plan.target_count() {
        let target = plan.target_id(index).expect("enumerated target");
        let mut residual = vec![0.0; plan.target(target).expect("owned target").size()];
        for packet in &packets {
            for mapping in packet.assembly.mappings() {
                if mapping.target() == target {
                    scatter_residual(&mut residual, mapping.map(), &packet.residual)?;
                }
            }
        }
        if residual.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "region residual assembly produced a non-finite value",
            ));
        }
        residuals.push(residual);
    }
    Ok(RegionAssemblyLinearization {
        systems,
        residuals,
        report,
        packets,
    })
}

#[cfg(test)]
mod tests;
