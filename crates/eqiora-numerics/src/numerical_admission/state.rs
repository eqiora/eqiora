use super::*;
mod grid;
use crate::prepared_execution::advance_prepared_actions;
pub(super) use grid::TimeGrid;
use std::ops::ControlFlow;

struct CommonAcceptedActions {
    state: CommonState,
    outputs: Vec<(usize, CommonAcceptedSpatialStep)>,
}

impl CommonRunSchedule {
    fn new(
        identity_domain: &[u8],
        identity_trailer: &[u8],
        plan_identity: &str,
        state: CommonState,
        accepted_steps: NonZeroUsize,
        output_steps: Vec<usize>,
        invalid_outputs: &'static str,
    ) -> Result<Self, Diagnostic> {
        if output_steps.is_empty()
            || output_steps.windows(2).any(|pair| pair[0] >= pair[1])
            || output_steps
                .iter()
                .any(|step| *step == 0 || *step > accepted_steps.get())
        {
            return Err(invalid(invalid_outputs));
        }
        let mut bytes = Vec::new();
        push_framed(&mut bytes, plan_identity.as_bytes());
        push_framed(&mut bytes, state.identity().as_bytes());
        bytes.extend_from_slice(&(accepted_steps.get() as u64).to_be_bytes());
        for output in &output_steps {
            bytes.extend_from_slice(&(*output as u64).to_be_bytes());
        }
        bytes.extend_from_slice(identity_trailer);
        let identity = domain_separated_identity(identity_domain, &bytes);
        Ok(Self {
            state,
            accepted_steps,
            output_steps,
            identity,
        })
    }
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
    /// Canonicalize a step-count horizon and explicit accepted-step outputs.
    pub fn from_steps(
        plan: ResolvedCommonPlan,
        state: CommonState,
        steps: usize,
        output_steps: Vec<usize>,
    ) -> Result<Self, Diagnostic> {
        let accepted_steps = NonZeroUsize::new(steps)
            .ok_or_else(|| invalid("transient Run steps must be strictly positive"))?;
        Self::new(plan, state, accepted_steps, output_steps)
    }

    /// Canonicalize an exact Backward-Euler time horizon and output-time grid.
    pub fn from_times(
        plan: ResolvedCommonPlan,
        state: CommonState,
        until_s: f64,
        output_times_s: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        if !until_s.is_finite() || until_s <= state.require_time_s()? {
            return Err(invalid(
                "transient Run until_s must be finite and later than State.time_s",
            ));
        }
        if output_times_s.is_empty()
            || output_times_s.iter().any(|value| !value.is_finite())
            || output_times_s.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                "output_times_s must be finite, nonempty, and strictly increasing",
            ));
        }
        let step_s = plan
            .backward_euler()
            .ok_or_else(|| invalid("transient Run requires BackwardEuler"))?
            .step()
            .value();
        let grid = state.time_grid()?;
        let accepted_steps = grid.steps_until(until_s, step_s, "until_s")?;
        let output_steps = output_times_s
            .into_iter()
            .map(|time| grid.steps_until(time, step_s, "output_times_s"))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(
            plan,
            state,
            NonZeroUsize::new(accepted_steps)
                .ok_or_else(|| invalid("transient Run horizon contains no accepted step"))?,
            output_steps,
        )
    }

    pub(super) fn new(
        plan: ResolvedCommonPlan,
        state: CommonState,
        accepted_steps: NonZeroUsize,
        output_steps: Vec<usize>,
    ) -> Result<Self, Diagnostic> {
        state.require_time_s()?;
        if plan.backward_euler().is_none() {
            return Err(invalid("fixed-grid transient Run requires BackwardEuler"));
        }
        if state.state_space_identity() != plan.spatial_state_space_identity()? {
            return Err(invalid(
                "transient Run State belongs to a different exact common state space",
            ));
        }
        let schedule = CommonRunSchedule::new(
            b"eqiora.common-transient-run-request/v1\0",
            &1_u64.to_be_bytes(),
            plan.identity(),
            state,
            accepted_steps,
            output_steps,
            "output_steps must be nonempty, strictly increasing accepted-step indices within the inclusive horizon",
        )?;
        Ok(Self { plan, schedule })
    }

    #[must_use]
    pub const fn plan(&self) -> &ResolvedCommonPlan {
        &self.plan
    }
    #[must_use]
    pub const fn state(&self) -> &CommonState {
        &self.schedule.state
    }
    #[must_use]
    pub const fn accepted_steps(&self) -> NonZeroUsize {
        self.schedule.accepted_steps
    }
    #[must_use]
    pub fn output_steps(&self) -> &[usize] {
        &self.schedule.output_steps
    }
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.schedule.identity
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
                .step()
                .value();
            return advance_common_prepared_actions(
                self.schedule.state.clone(),
                self.schedule.accepted_steps.get(),
                &self.schedule.output_steps,
                step_s,
                |_| Ok(()),
                |(), state| plan.advance_scalar(state, backend),
                stop_at_boundary,
            );
        }
        let ResolvedCommonPlan::TransientFlow(plan) = &self.plan else {
            return Err(invalid("unsupported transient Plan"));
        };
        let step_s = plan.temporal().step().value();
        let start_s = self.state().require_time_s()?;
        let mut accepted = 0;
        advance_common_prepared_actions(
            self.schedule.state.clone(),
            self.schedule.accepted_steps.get(),
            &self.schedule.output_steps,
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
