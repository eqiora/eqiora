use super::{
    contract::{Identity, invalid, reserve},
    finite,
};
use crate::cartesian_fvm_geometry::CartesianCellMetrics;
use eqiora_core::{Diagnostic, RawId};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FaceActionPacket {
    pub(super) face: usize,
    pub(super) owner: usize,
    pub(super) neighbor: Option<usize>,
    pub(super) boundary: Option<RawId>,
    pub(super) normal: Vec<f64>,
    pub(super) area: f64,
    pub(super) normal_flux: Vec<f64>,
    pub(super) wave_bound: Option<f64>,
    pub(super) integrated: Vec<f64>,
}

impl FaceActionPacket {
    pub(crate) fn owner(&self) -> usize {
        self.owner
    }
    pub(crate) fn neighbor(&self) -> Option<usize> {
        self.neighbor
    }
    pub(crate) fn normal_flux(&self) -> &[f64] {
        &self.normal_flux
    }
    pub(crate) fn integrated_outward_flux(&self) -> &[f64] {
        &self.integrated
    }
    /// One stored packet, not an independently evaluated adjacent-cell copy.
    pub(crate) fn scatter(&self, cell: usize, component: usize) -> Option<f64> {
        let value = *self.integrated.get(component)?;
        if cell == self.owner {
            Some(-value)
        } else if Some(cell) == self.neighbor {
            Some(value)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConservativeActionReceipt {
    pub(super) identity: Identity,
    pub(super) packets: Vec<FaceActionPacket>,
    pub(super) inventory_rates: Vec<f64>,
    pub(super) average_rates: Vec<f64>,
    pub(super) outward_boundary_flux: Vec<f64>,
    pub(super) balance_defect: Vec<f64>,
    pub(super) accounting_tolerance: Vec<f64>,
}

impl ConservativeActionReceipt {
    pub(crate) fn packets(&self) -> &[FaceActionPacket] {
        &self.packets
    }
    pub(crate) fn inventory_rates(&self) -> &[f64] {
        &self.inventory_rates
    }
    pub(crate) fn average_rates(&self) -> &[f64] {
        &self.average_rates
    }
    pub(crate) fn outward_boundary_flux(&self) -> &[f64] {
        &self.outward_boundary_flux
    }
    pub(crate) fn balance_defect(&self) -> &[f64] {
        &self.balance_defect
    }
    pub(crate) fn accounting_tolerance(&self) -> &[f64] {
        &self.accounting_tolerance
    }
}

pub(super) fn assemble<const D: usize>(
    identity: &Identity,
    cells: &[CartesianCellMetrics<D>],
    packets: Vec<FaceActionPacket>,
) -> Result<ConservativeActionReceipt, Diagnostic> {
    let k = identity.semantic.components.len();
    let mut inventory_rates = reserve(cells.len() * k)?;
    inventory_rates.resize(cells.len() * k, 0.0);
    let mut outward_boundary_flux = vec![0.0; k];
    let mut absolute_packets = vec![0.0; k];
    let mut additions = 0_usize;
    for packet in &packets {
        for (component, value) in packet.integrated.iter().enumerate() {
            inventory_rates[packet.owner * k + component] -= value;
            if let Some(neighbor) = packet.neighbor {
                inventory_rates[neighbor * k + component] += value;
            } else {
                outward_boundary_flux[component] += value;
            }
            absolute_packets[component] += 2.0 * value.abs();
        }
        additions += 2; // Per component: two scatters, or scatter + boundary accumulation.
    }
    let mut average_rates = reserve(inventory_rates.len())?;
    let mut sum = vec![0.0; k];
    for (cell, values) in cells.iter().zip(inventory_rates.chunks_exact(k)) {
        for (component, value) in values.iter().enumerate() {
            finite(*value, "integrated residual")?;
            average_rates.push(finite(value / cell.measure, "cell-average derivative")?);
            sum[component] += value;
        }
    }
    // gamma_n forward-error bound for all scatter, final cell/boundary sums and
    // absolute-magnitude accumulation. Epsilon (rather than half epsilon) plus
    // a second factor two covers rounding of the magnitude sum itself. Underflow
    // is covered additively by one minimum subnormal per counted operation.
    let operations = additions
        .checked_add(cells.len())
        .and_then(|n| n.checked_add(packets.len()))
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| invalid("accounting operation count overflow"))?;
    let error = operations as f64 * f64::EPSILON;
    if error >= 0.5 {
        return Err(invalid(
            "accounting operation count cannot bound floating-point error",
        ));
    }
    let gamma = error / (1.0 - error);
    let mut balance_defect = Vec::with_capacity(k);
    let mut accounting_tolerance = Vec::with_capacity(k);
    for component in 0..k {
        let defect = finite(
            sum[component] + outward_boundary_flux[component],
            "accumulated balance",
        )?;
        let tolerance = finite(
            2.0 * gamma * absolute_packets[component] + operations as f64 * f64::from_bits(1),
            "accounting bound",
        )?;
        if defect.abs() > tolerance {
            return Err(invalid(
                "accumulated conservation defect exceeds rounding bound",
            ));
        }
        balance_defect.push(defect);
        accounting_tolerance.push(tolerance);
    }
    Ok(ConservativeActionReceipt {
        identity: identity.clone(),
        packets,
        inventory_rates,
        average_rates,
        outward_boundary_flux,
        balance_defect,
        accounting_tolerance,
    })
}
