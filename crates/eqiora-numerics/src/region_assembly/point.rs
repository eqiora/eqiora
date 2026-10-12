//! Original and linearized actions at one exact constrained Region point.
use super::nonlinear::{
    EvaluatedRegionPacket, RegionAssemblyLinearization, assemble_linearization, mapped_local_point,
    scatter_residual,
};
use super::*;
use eqiora_assembly::{AssemblyBackend, AssemblyMap, AssemblyTargetId};

impl PreparedRegionAssembly<f64> {
    pub(crate) fn linearize(
        &self,
        plan: &AssemblyPlan,
        target: AssemblyTargetId,
        point: &[f64],
        backend: &dyn AssemblyBackend<f64>,
    ) -> Result<RegionAssemblyLinearization, Diagnostic> {
        self.validate_point(plan, target, point)?;
        assemble_linearization(
            plan,
            self.packet_set,
            self.packet_count(),
            backend,
            |index| {
                let cell = self.packet_cell(index)?;
                let map = candidate_map(&cell.mappings, target)?;
                let mut local_point = mapped_local_point(map, point)?;
                for (value, sign) in local_point.iter_mut().zip(&cell.orientation) {
                    *value *= f64::from(*sign);
                }
                let action = if index < self.cells.len() {
                    self.prepared[index].linearize(&cell.previous, &local_point)?
                } else {
                    self.boundary_packets[index - self.cells.len()]
                        .1
                        .linearize(&local_point)?
                };
                let residual = action
                    .residual
                    .iter()
                    .zip(&cell.orientation)
                    .map(|(value, sign)| *value * f64::from(*sign))
                    .collect();
                Ok(EvaluatedRegionPacket {
                    assembly: AssemblyPacket::new(
                        action
                            .into_contribution(&local_point)?
                            .reoriented(&cell.orientation, &cell.orientation)?,
                        cell.mappings.clone(),
                    )?,
                    residual,
                })
            },
        )
    }

    /// Independently evaluate primal weak actions; never recover R by J*x-b.
    pub(crate) fn residual(
        &self,
        plan: &AssemblyPlan,
        target: AssemblyTargetId,
        point: &[f64],
    ) -> Result<Vec<f64>, Diagnostic> {
        self.validate_point(plan, target, point)?;
        let mut residual = vec![0.; point.len()];
        for index in 0..self.packet_count() {
            let cell = self.packet_cell(index)?;
            let map = candidate_map(&cell.mappings, target)?;
            let mut local_point = mapped_local_point(map, point)?;
            for (value, sign) in local_point.iter_mut().zip(&cell.orientation) {
                *value *= f64::from(*sign);
            }
            let mut local = if index < self.cells.len() {
                self.prepared[index].residual(&cell.previous, &local_point)?
            } else {
                self.boundary_packets[index - self.cells.len()]
                    .1
                    .residual(&local_point)?
            };
            for (value, sign) in local.iter_mut().zip(&cell.orientation) {
                *value *= f64::from(*sign);
            }
            scatter_residual(&mut residual, map, &local)?;
        }
        if residual.iter().any(|value| !value.is_finite()) {
            return Err(invalid("Region primal residual accumulation is non-finite"));
        }
        Ok(residual)
    }

    fn validate_point(
        &self,
        plan: &AssemblyPlan,
        target: AssemblyTargetId,
        point: &[f64],
    ) -> Result<(), Diagnostic> {
        let selected = plan
            .target(target)
            .ok_or_else(|| invalid("Region point has a foreign assembly target"))?;
        if point.len() != selected.size() || point.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "Region point requires exact finite target coordinates",
            ));
        }
        Ok(())
    }
}

fn candidate_map(
    mappings: &[TargetAssemblyMap<f64>],
    target: AssemblyTargetId,
) -> Result<&AssemblyMap<f64>, Diagnostic> {
    mappings
        .iter()
        .find(|mapping| mapping.target() == target)
        .map(|mapping| mapping.map())
        .ok_or_else(|| invalid("Region packet omits its candidate target"))
}
