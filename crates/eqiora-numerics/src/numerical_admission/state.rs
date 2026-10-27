use super::*;
mod grid;
mod scalar;
use crate::prepared_execution::advance_prepared_actions;
pub(super) use grid::TimeGrid;
use std::ops::ControlFlow;

struct CommonAcceptedActions {
    state: CommonState,
    outputs: Vec<(usize, CommonAcceptedSpatialStep)>,
}

#[allow(clippy::type_complexity)]
fn advance_common_prepared_actions<P>(
    initial: CommonState,
    maximum_actions: usize,
    output_actions: &[usize],
    step_s: f64,
    prepare: impl FnOnce(&CommonState) -> Result<P, Diagnostic>,
    mut advance: impl FnMut(&mut P, &CommonState) -> Result<CommonAcceptedSpatialStep, Diagnostic>,
    mut stop_at_boundary: impl FnMut(usize, &CommonState) -> bool,
) -> Result<ControlFlow<(usize, CommonState), Vec<(usize, CommonAcceptedSpatialStep)>>, Diagnostic>
{
    let context = CommonAcceptedActions {
        state: initial,
        outputs: Vec::with_capacity(output_actions.len()),
    };
    match advance_prepared_actions(
        context,
        maximum_actions,
        |context| prepare(&context.state),
        |step, context| {
            eqiora_execution::telemetry_span!(time_step(
                step,
                context.state.time_s().expect("admitted transient State") + step_s,
                step_s
            ))
        },
        |prepared, context| advance(prepared, &context.state),
        |context, accepted_actions, candidate| {
            if output_actions.binary_search(&accepted_actions).is_ok() {
                context.outputs.push((accepted_actions, candidate.clone()));
            }
            context.state = candidate.state;
            Ok(())
        },
        |accepted_actions, context| {
            stop_at_boundary(accepted_actions, &context.state)
                .then(|| (accepted_actions, context.state.clone()))
        },
    )? {
        ControlFlow::Break(stopped) => Ok(ControlFlow::Break(stopped)),
        ControlFlow::Continue(context) => Ok(ControlFlow::Continue(context.outputs)),
    }
}

impl CommonTransientRunRequest {
    /// Convert an exact fixed-grid step request to physical times once.
    pub fn from_steps(
        plan: ResolvedCommonPlan,
        state: CommonState,
        steps: usize,
        output_steps: Vec<usize>,
    ) -> Result<Self, Diagnostic> {
        if steps == 0
            || output_steps.is_empty()
            || output_steps.iter().any(|step| *step == 0 || *step > steps)
            || output_steps.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                "output_steps must be nonempty, strictly increasing accepted-step indices within the inclusive horizon",
            ));
        }
        let dt = plan
            .backward_euler()
            .ok_or_else(|| invalid("fixed-grid transient Run requires BackwardEuler"))?
            .value();
        let grid = state.time_grid()?;
        let until_s = grid.advance(steps)?.time(dt)?;
        let outputs = output_steps
            .into_iter()
            .map(|step| grid.advance(step)?.time(dt))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_times(plan, state, until_s, outputs)
    }

    /// Bind a physical horizon and observations independently of internal steps.
    /// Backward Euler retains its exact-grid output contract.
    pub fn from_times(
        plan: ResolvedCommonPlan,
        state: CommonState,
        until_s: f64,
        output_times_s: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        let start = state.require_time_s()?;
        if state.state_space_identity() != plan.spatial_state_space_identity()? {
            return Err(invalid(
                "transient Run State belongs to a different exact common state space",
            ));
        }
        if !until_s.is_finite() || until_s <= start || !(until_s - start).is_finite() {
            return Err(invalid(
                "transient Run until_s must be finite and later than State.time_s",
            ));
        }
        if output_times_s.is_empty()
            || output_times_s
                .iter()
                .any(|time| !time.is_finite() || *time <= start || *time > until_s)
            || output_times_s.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                "output_times_s must be finite, nonempty, strictly increasing, and within the Run horizon",
            ));
        }
        if let Some(temporal) = plan.backward_euler() {
            let grid = state.time_grid()?;
            let dt = temporal.value();
            grid.steps_until(until_s, dt, "until_s")?;
            for &time in &output_times_s {
                grid.steps_until(time, dt, "output_times_s")?;
            }
        } else if plan.ode_temporal().is_none() || plan.as_spatial().is_none() {
            return Err(invalid(
                "spatial transient Run requires its admitted temporal policy",
            ));
        }
        let mut bytes = Vec::new();
        push_framed(&mut bytes, plan.identity().as_bytes());
        push_framed(&mut bytes, state.identity().as_bytes());
        bytes.extend_from_slice(&until_s.to_bits().to_be_bytes());
        for time in &output_times_s {
            bytes.extend_from_slice(&time.to_bits().to_be_bytes());
        }
        let identity =
            domain_separated_identity(b"eqiora.common-transient-run-request/v2\0", &bytes);
        Ok(Self {
            plan,
            state,
            until_s,
            output_times_s,
            identity,
        })
    }

    #[must_use]
    pub const fn plan(&self) -> &ResolvedCommonPlan {
        &self.plan
    }
    #[must_use]
    pub const fn state(&self) -> &CommonState {
        &self.state
    }
    #[must_use]
    pub const fn until_s(&self) -> f64 {
        self.until_s
    }
    #[must_use]
    pub fn output_times_s(&self) -> &[f64] {
        &self.output_times_s
    }
    /// Exact fixed-grid count, unavailable for continuous time policies.
    pub fn accepted_steps(&self) -> Result<NonZeroUsize, Diagnostic> {
        let dt = self
            .plan
            .backward_euler()
            .ok_or_else(|| {
                invalid("accepted step count is determined by execution for this policy")
            })?
            .value();
        NonZeroUsize::new(
            self.state
                .time_grid()?
                .steps_until(self.until_s, dt, "until_s")?,
        )
        .ok_or_else(|| invalid("transient Run horizon contains no accepted step"))
    }
    /// Fixed-grid output ordinals, available only when observations select grid points.
    pub fn output_steps(&self) -> Result<Vec<usize>, Diagnostic> {
        let dt = self
            .plan
            .backward_euler()
            .ok_or_else(|| {
                invalid("observation times are not accepted-step ordinals for this policy")
            })?
            .value();
        let grid = self.state.time_grid()?;
        self.output_times_s
            .iter()
            .map(|&time| grid.steps_until(time, dt, "output_times_s"))
            .collect()
    }
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Authenticate this exact Run once, then advance its accepted actions.
    ///
    /// The callback is observed only before action one and after a successful
    /// action. Returning `true` stops with the last accepted State.
    #[allow(clippy::type_complexity)]
    pub fn advance_accepted_actions(
        &self,
        backend: &dyn LinearSolverBackend,
        stop_at_boundary: impl FnMut(usize, &CommonState) -> bool,
    ) -> Result<
        ControlFlow<(usize, CommonState), Vec<(usize, CommonAcceptedSpatialStep)>>,
        Diagnostic,
    > {
        if let ResolvedCommonPlan::Spatial(plan) = &self.plan {
            let step_s = plan
                .admission
                .temporal
                .as_ref()
                .expect("validated temporal")
                .require_backward_euler()?
                .value();
            return advance_common_prepared_actions(
                self.state.clone(),
                self.accepted_steps()?.get(),
                &self.output_steps()?,
                step_s,
                |state| scalar::PreparedScalarExecution::new(plan, state, backend),
                |prepared, state| prepared.advance(state),
                stop_at_boundary,
            );
        }
        let ResolvedCommonPlan::TransientFlow(plan) = &self.plan else {
            return Err(invalid("unsupported transient Plan"));
        };
        let step_s = plan.temporal().value();
        let start_s = self.state().require_time_s()?;
        let mut accepted = 0;
        advance_common_prepared_actions(
            self.state.clone(),
            self.accepted_steps()?.get(),
            &self.output_steps()?,
            step_s,
            |state| plan.prepare_execution(state, backend),
            |prepared, state| {
                accepted += 1;
                prepared
                    .advance(state, start_s + accepted as f64 * step_s)
                    .and_then(|next| CommonAcceptedSpatialStep::new(state, next))
            },
            stop_at_boundary,
        )
    }
}

impl CommonState {
    pub(super) fn new(
        state_space_identity: String,
        time_s: Option<f64>,
        model: Arc<ModelEnvelope>,
        resources: Arc<NativeMeshResources>,
        kind: CommonStateKind,
    ) -> Result<Self, Diagnostic> {
        if time_s.is_some_and(|time| {
            !time.is_finite() || time < 0.0 || time.to_bits() == (-0.0_f64).to_bits()
        }) {
            return Err(invalid("State time_s must be finite and non-negative"));
        }
        let mut bytes = Vec::new();
        push_framed(&mut bytes, state_space_identity.as_bytes());
        match time_s {
            Some(time) => {
                bytes.push(1);
                bytes.extend_from_slice(&time.to_bits().to_be_bytes());
            }
            None => bytes.push(0),
        }
        match &kind {
            CommonStateKind::Spatial { values, previous } => {
                push_framed(&mut bytes, b"spatial-nodal/v3");
                bytes.extend_from_slice(&(values.len() as u64).to_be_bytes());
                for value in values {
                    bytes.extend_from_slice(&value.to_bits().to_be_bytes());
                }
                match previous {
                    None => bytes.push(0),
                    Some(previous) => {
                        bytes.push(1);
                        bytes.extend_from_slice(&previous.grid.origin_s.to_bits().to_be_bytes());
                        bytes.extend_from_slice(&(previous.grid.index as u64).to_be_bytes());
                        bytes.extend_from_slice(&(previous.values.len() as u64).to_be_bytes());
                        for value in &previous.values {
                            bytes.extend_from_slice(&value.to_bits().to_be_bytes());
                        }
                    }
                }
            }
            CommonStateKind::MiniP1(state) => {
                push_framed(&mut bytes, b"mini-p1/backward-euler/no-extra-history/v1");
                for value in state
                    .velocity()
                    .vertex_values()
                    .iter()
                    .flatten()
                    .chain(state.velocity().cell_bubble_values().iter().flatten())
                    .chain(state.pressure().vertex_values())
                {
                    bytes.extend_from_slice(&value.to_bits().to_be_bytes());
                }
                match state.pressure_reference() {
                    SteadyStokesPressureReference2d::ZeroIntegral { multiplier } => {
                        push_framed(&mut bytes, b"zero-integral");
                        bytes.extend_from_slice(&multiplier.to_bits().to_be_bytes());
                    }
                    SteadyStokesPressureReference2d::BoundaryTraction => {
                        push_framed(&mut bytes, b"boundary-traction");
                    }
                }
            }
            CommonStateKind::CellCentered(state) => {
                push_framed(
                    &mut bytes,
                    b"cell-centered/backward-euler/bdf1-previous-accepted-face-volume-flux/v1",
                );
                for value in state
                    .velocity()
                    .values()
                    .iter()
                    .flatten()
                    .chain(state.pressure().values())
                    .chain(std::iter::once(&state.gauge_multiplier()))
                    .chain(state.previous_face_volume_fluxes())
                {
                    bytes.extend_from_slice(&value.to_bits().to_be_bytes());
                }
            }
        }
        let identity = hex_bytes(&Sha256::digest(
            [b"eqiora.common-state/v3\0".as_slice(), bytes.as_slice()].concat(),
        ));
        Ok(Self {
            state_space_identity,
            identity,
            time_s,
            model,
            resources,
            kind,
        })
    }

    /// Exact identity of Model, Mesh, complete fields/spaces, gauge, layout, and history schema.
    #[must_use]
    pub fn state_space_identity(&self) -> &str {
        &self.state_space_identity
    }

    /// Content identity of this exact accepted state occurrence.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Exact coherent-SI model time; stationary numerical states have no time.
    #[must_use]
    pub const fn time_s(&self) -> Option<f64> {
        self.time_s
    }

    pub(crate) fn require_time_s(&self) -> Result<f64, Diagnostic> {
        self.time_s
            .ok_or_else(|| invalid("transient operation requires a State with model Time"))
    }

    #[must_use]
    pub fn velocity_vertex_values(&self) -> Option<&[[f64; 2]]> {
        match &self.kind {
            CommonStateKind::Spatial { .. } => None,
            CommonStateKind::MiniP1(state) => Some(state.velocity().vertex_values()),
            CommonStateKind::CellCentered(_) => None,
        }
    }

    #[must_use]
    pub fn velocity_cell_values(&self) -> Option<Vec<[f64; 2]>> {
        match &self.kind {
            CommonStateKind::Spatial { .. } => None,
            CommonStateKind::MiniP1(state) => Some(state.velocity().cell_bubble_values().to_vec()),
            CommonStateKind::CellCentered(state) => Some(state.velocity().values().to_vec()),
        }
    }

    #[must_use]
    pub fn pressure_vertex_values(&self) -> Option<&[f64]> {
        match &self.kind {
            CommonStateKind::Spatial { .. } => None,
            CommonStateKind::MiniP1(state) => Some(state.pressure().vertex_values()),
            CommonStateKind::CellCentered(_) => None,
        }
    }

    #[must_use]
    pub fn pressure_cell_values(&self) -> Option<&[f64]> {
        match &self.kind {
            CommonStateKind::Spatial { .. } | CommonStateKind::MiniP1(_) => None,
            CommonStateKind::CellCentered(state) => Some(state.pressure().values()),
        }
    }

    #[cfg(test)]
    pub(super) fn method_history_values(&self) -> &[f64] {
        match &self.kind {
            CommonStateKind::Spatial { .. } | CommonStateKind::MiniP1(_) => &[],
            CommonStateKind::CellCentered(state) => state.previous_face_volume_fluxes(),
        }
    }
}
