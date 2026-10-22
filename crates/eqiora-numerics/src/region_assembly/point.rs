//! Original and linearized actions at one exact constrained Region point.
use super::nonlinear::{
    EvaluatedRegionPacket, RegionAssemblyLinearization, assemble_linearization,
};
use super::residual::{mapped_local_point, scatter_residual};
use super::*;
use crate::form_compiler::region::{FieldPullback, RegionLinearization};
use eqiora_assembly::{AssemblyBackend, AssemblyMap, AssemblyTargetId};

/// A bound solve consumes its authenticated history; an implicit point supplies
/// independent rates in an unconstrained target's normalized coordinates.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RegionPoint<'a> {
    Bound(&'a [f64]),
    Independent {
        state: &'a [f64],
        rate_target: AssemblyTargetId,
        rates: &'a [f64],
        pullback: FieldPullback<'a>,
    },
}

impl<'a> RegionPoint<'a> {
    fn state(self) -> &'a [f64] {
        match self {
            Self::Bound(state) | Self::Independent { state, .. } => state,
        }
    }
}

impl<S: Coefficient + Send + Sync, P: Projection<S, Scalar = f64>> PreparedRegionAssembly<S, P> {
    pub(crate) fn linearize(
        &self,
        plan: &AssemblyPlan,
        target: AssemblyTargetId,
        point: RegionPoint<'_>,
        backend: &dyn AssemblyBackend<f64>,
    ) -> Result<RegionAssemblyLinearization, Diagnostic> {
        self.validate_point(plan, target, point)?;
        assemble_linearization(
            plan,
            self.packet_set,
            self.packet_count(),
            backend,
            |index| {
                let packet = self.packet(index)?;
                let (local_point, action) = self.local_action(target, point, index, true)?;
                let residual = action
                    .residual
                    .iter()
                    .zip(&packet.orientation)
                    .map(|(value, sign)| *value * f64::from(*sign))
                    .collect();
                Ok(EvaluatedRegionPacket {
                    assembly: AssemblyPacket::new(
                        action
                            .into_contribution(&local_point)?
                            .reoriented(&packet.orientation, &packet.orientation)?,
                        packet.mappings.clone(),
                    )?,
                    residual,
                })
            },
        )
    }

    /// Evaluate original weak actions; never recover R by subtracting J*x-b.
    pub(crate) fn residual(
        &self,
        plan: &AssemblyPlan,
        target: AssemblyTargetId,
        point: RegionPoint<'_>,
    ) -> Result<Vec<f64>, Diagnostic> {
        self.validate_point(plan, target, point)?;
        let mut residual = vec![0.; point.state().len()];
        for index in 0..self.packet_count() {
            let packet = self.packet(index)?;
            let map = candidate_map(&packet.mappings, target)?;
            let (_, mut action) = self.local_action(target, point, index, false)?;
            for (value, sign) in action.residual.iter_mut().zip(&packet.orientation) {
                *value *= f64::from(*sign);
            }
            scatter_residual(&mut residual, map, &action.residual)?;
        }
        if residual.iter().any(|value| !value.is_finite()) {
            return Err(invalid("Region primal residual accumulation is non-finite"));
        }
        Ok(residual)
    }

    fn local_action(
        &self,
        target: AssemblyTargetId,
        point: RegionPoint<'_>,
        index: usize,
        derivative: bool,
    ) -> Result<(Vec<f64>, RegionLinearization), Diagnostic> {
        let packet = self.packet(index)?;
        let map = candidate_map(&packet.mappings, target)?;
        let mut state = mapped_local_point(map, point.state())?;
        orient(&mut state, &packet.orientation);
        let action = match point {
            RegionPoint::Bound(_) => match &packet.action {
                RegionPacketAction::Cell { prepared, previous } => {
                    if derivative {
                        prepared.linearize(previous, &state)?
                    } else {
                        RegionLinearization {
                            residual: prepared.residual(previous, &state)?,
                            jacobian: Vec::new(),
                        }
                    }
                }
                RegionPacketAction::Boundary(facet) => {
                    if derivative {
                        facet.linearize(&state)?
                    } else {
                        RegionLinearization {
                            residual: facet.residual(&state)?,
                            jacobian: Vec::new(),
                        }
                    }
                }
            },
            RegionPoint::Independent {
                rate_target,
                rates,
                pullback,
                ..
            } => {
                let local_weights;
                let pullback = match pullback {
                    FieldPullback::Coordinates(weights) => {
                        local_weights = map
                            .unknowns()
                            .iter()
                            .map(|unknown| match unknown {
                                LocalUnknown::Free(dof) => weights[dof.index()],
                                LocalUnknown::Fixed(_) => [None, None],
                            })
                            .collect::<Vec<_>>();
                        FieldPullback::Coordinates(&local_weights)
                    }
                    uniform => uniform,
                };
                match &packet.action {
                    RegionPacketAction::Cell { prepared, .. } => {
                        let rate_map = candidate_map(&packet.mappings, rate_target)?;
                        // Fixed entries contain State values. They are not time derivatives
                        // of boundary data, so a rate point must supply every coordinate.
                        if rate_map
                            .unknowns()
                            .iter()
                            .any(|value| matches!(value, LocalUnknown::Fixed(_)))
                        {
                            return Err(invalid(
                                "independent rates require an unconstrained target with explicit boundary rates",
                            ));
                        }
                        let mut rates = mapped_local_point(rate_map, rates)?;
                        orient(&mut rates, &packet.orientation);
                        let rates = prepared.physical_coefficients(&rates)?;
                        prepared.implicit_action(
                            &state,
                            Some(&rates),
                            derivative.then_some(pullback),
                        )?
                    }
                    RegionPacketAction::Boundary(facet) => {
                        facet.action(&state, derivative.then_some(pullback))?
                    }
                }
            }
        };
        Ok((state, action))
    }

    fn validate_point(
        &self,
        plan: &AssemblyPlan,
        target: AssemblyTargetId,
        point: RegionPoint<'_>,
    ) -> Result<(), Diagnostic> {
        validate_values(plan, target, point.state())?;
        if let RegionPoint::Independent {
            rate_target,
            rates,
            pullback,
            ..
        } = point
        {
            validate_values(plan, rate_target, rates)?;
            pullback.validate(point.state().len())?;
        }
        Ok(())
    }
}

fn validate_values(
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

fn orient(point: &mut [f64], signs: &[i8]) {
    for (value, sign) in point.iter_mut().zip(signs) {
        *value *= f64::from(*sign);
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
