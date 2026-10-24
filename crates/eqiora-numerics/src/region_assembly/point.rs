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
    /// Same bound residual, differentiated in its consumed predecessor.
    Previous(&'a [f64]),
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
            Self::Bound(state) | Self::Previous(state) | Self::Independent { state, .. } => state,
        }
    }
}

impl<S: Coefficient, P: Projection<S, Scalar = f64>> PreparedRegionPacket<S, P> {
    fn physical_rates(
        &self,
        target: AssemblyTargetId,
        rates: &[f64],
    ) -> Result<Vec<f64>, Diagnostic> {
        let RegionPacketAction::Cell { prepared, .. } = &self.action else {
            return Err(invalid(
                "independent rate interpolation requires a cell action",
            ));
        };
        let rate_map = candidate_map(&self.mappings, target)?;
        // Fixed entries are State values, not derivatives of boundary data.
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
        orient(&mut rates, &self.orientation);
        prepared.physical_coefficients(&rates)
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

    /// Apply the same local tangents without assembling a global matrix.
    /// Prescribed coordinates have zero candidate direction; functional
    /// orientations act on both the gathered direction and scattered equation.
    pub(crate) fn jacobian_action(
        &self,
        plan: &AssemblyPlan,
        target: AssemblyTargetId,
        point: RegionPoint<'_>,
        direction: &[f64],
    ) -> Result<Vec<f64>, Diagnostic> {
        self.validate_point(plan, target, point)?;
        validate_values(plan, target, direction)?;
        let mut output = vec![0.; direction.len()];
        for index in 0..self.packet_count() {
            let packet = self.packet(index)?;
            let map = candidate_map(&packet.mappings, target)?;
            let (_, action) = self.local_action(target, point, index, true)?;
            let local_direction = map
                .unknowns()
                .iter()
                .zip(&packet.orientation)
                .map(|(unknown, sign)| match unknown {
                    LocalUnknown::Free(dof) => direction[dof.index()] * f64::from(*sign),
                    LocalUnknown::Fixed(_) => 0.,
                })
                .collect::<Vec<_>>();
            let local = action
                .jacobian
                .chunks_exact(local_direction.len())
                .zip(&packet.orientation)
                .map(|(row, sign)| {
                    row.iter()
                        .zip(&local_direction)
                        .map(|(entry, direction)| entry * direction)
                        .sum::<f64>()
                        * f64::from(*sign)
                })
                .collect::<Vec<_>>();
            scatter_residual(&mut output, map, &local)?;
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(invalid("Region tangent action accumulation is non-finite"));
        }
        Ok(output)
    }

    pub(crate) fn projected_input_partial(
        &self,
        plan: &AssemblyPlan,
        target: AssemblyTargetId,
        point: RegionPoint<'_>,
        coefficient_partial: (&Self, &AssemblyPlan),
        coordinate: &eqiora_ir::ScalarSymbolCoordinate,
        mass: &eqiora_time::ConstantDerivativeMatrixProof,
    ) -> Result<Vec<f64>, Diagnostic> {
        use crate::form_compiler::spatial::InputAction;
        use num_rational::BigRational;
        use num_traits::{ToPrimitive, Zero};
        self.validate_point(plan, target, point)?;
        let RegionPoint::Independent {
            rate_target, rates, ..
        } = point
        else {
            return Err(invalid(
                "projected input partial requires independent rates",
            ));
        };
        if mass.dimension() != point.state().len() {
            return Err(invalid("input partial mass proof has a foreign dimension"));
        }
        let mut actions: Vec<InputAction<S>> = Vec::new();
        for (assembly, assembly_plan, coefficients) in [
            (self, plan, false),
            (coefficient_partial.0, coefficient_partial.1, true),
        ] {
            assembly.validate_point(assembly_plan, target, point)?;
            for packet in &assembly.packets {
                let RegionPacketAction::Cell { prepared, .. } = &packet.action else {
                    continue;
                };
                let map = candidate_map(&packet.mappings, target)?;
                let mut state = mapped_local_point(map, point.state())?;
                orient(&mut state, &packet.orientation);
                let rates = packet.physical_rates(rate_target, rates)?;
                let local = if coefficients {
                    prepared.coefficient_input_actions(&state, &rates)?
                } else {
                    prepared.component_input_actions(&state, &rates, coordinate)?
                };
                for mut action in local {
                    for (value, sign) in action.coefficients.iter_mut().zip(&packet.orientation) {
                        *value *= BigRational::from_integer((*sign).into());
                    }
                    let at = if let Some(at) = actions
                        .iter()
                        .position(|item| item.atom.same_atom(&action.atom))
                    {
                        at
                    } else {
                        actions.push(InputAction {
                            atom: action.atom.clone(),
                            coefficients: vec![BigRational::zero(); mass.dimension()],
                        });
                        actions.len() - 1
                    };
                    scatter_residual(&mut actions[at].coefficients, map, &action.coefficients)?;
                }
            }
        }
        let mut partial = vec![0_f64; mass.dimension()];
        for action in actions {
            if mass.is_compatible(&[], &action.coefficients)? {
                continue;
            }
            let derivative = action.atom.partial()?;
            for (value, coefficient) in partial.iter_mut().zip(action.coefficients) {
                *value += coefficient
                    .to_f64()
                    .ok_or_else(|| invalid("input action exceeds binary64"))?
                    * derivative;
            }
        }
        if partial.iter().any(|value| !value.is_finite()) {
            return Err(invalid("projected input partial is non-finite"));
        }
        Ok(partial)
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
            RegionPoint::Bound(_) | RegionPoint::Previous(_) => match &packet.action {
                RegionPacketAction::Cell { prepared, previous } => {
                    if derivative {
                        if matches!(point, RegionPoint::Previous(_)) {
                            prepared.linearize_previous(previous, &state)?
                        } else {
                            prepared.linearize(previous, &state)?
                        }
                    } else {
                        RegionLinearization {
                            residual: prepared.residual(previous, &state)?,
                            jacobian: Vec::new(),
                        }
                    }
                }
                RegionPacketAction::Boundary(facet) => {
                    if derivative {
                        if matches!(point, RegionPoint::Previous(_)) {
                            facet.action(
                                &state,
                                Some(FieldPullback::Uniform {
                                    current: None,
                                    rate: None,
                                }),
                            )?
                        } else {
                            facet.linearize(&state)?
                        }
                    } else {
                        RegionLinearization {
                            residual: facet.residual(&state)?,
                            jacobian: Vec::new(),
                        }
                    }
                }
                RegionPacketAction::Affine(local) => affine_action(
                    local,
                    &state,
                    derivative.then_some(FieldPullback::Uniform {
                        current: (!matches!(point, RegionPoint::Previous(_))).then_some(1.),
                        rate: None,
                    }),
                )?,
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
                        let rates = packet.physical_rates(rate_target, rates)?;
                        prepared.implicit_action(
                            &state,
                            Some(&rates),
                            derivative.then_some(pullback),
                        )?
                    }
                    RegionPacketAction::Boundary(facet) => {
                        facet.action(&state, derivative.then_some(pullback))?
                    }
                    RegionPacketAction::Affine(local) => {
                        affine_action(local, &state, derivative.then_some(pullback))?
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

fn affine_action(
    local: &eqiora_assembly::LocalContribution<f64>,
    state: &[f64],
    pullback: Option<FieldPullback<'_>>,
) -> Result<RegionLinearization, Diagnostic> {
    let residual = super::residual::evaluate_local_residual(local, state)?;
    let jacobian = match pullback {
        Some(pullback) => {
            pullback.validate(state.len())?;
            local
                .matrix()
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    *value
                        * pullback
                            .weight(index % local.columns(), false)
                            .unwrap_or(0.)
                })
                .collect()
        }
        None => Vec::new(),
    };
    Ok(RegionLinearization { residual, jacobian })
}
