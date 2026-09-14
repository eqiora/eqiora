use super::*;
use crate::prepared_execution::advance_prepared_actions;
use std::ops::ControlFlow;

struct CommonAcceptedActions {
    state: CommonState,
    outputs: Vec<(usize, CommonState)>,
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
    mut advance: impl FnMut(&P, &CommonState) -> Result<CommonState, Diagnostic>,
    mut stop_at_boundary: impl FnMut(usize, &CommonState) -> bool,
) -> Result<ControlFlow<(usize, CommonState), Vec<(usize, CommonState)>>, Diagnostic> {
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
                context.state.time_s() + step_s,
                step_s
            ))
        },
        |prepared, context| advance(prepared, &context.state),
        |context, accepted_actions, candidate| {
            context.state = candidate;
            if output_actions.binary_search(&accepted_actions).is_ok() {
                context
                    .outputs
                    .push((accepted_actions, context.state.clone()));
            }
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

impl CommonFsiRunRequest {
    pub fn from_steps(
        plan: CommonFsiPlan,
        state: CommonState,
        steps: usize,
        output_steps: Vec<usize>,
    ) -> Result<Self, Diagnostic> {
        let accepted_steps = NonZeroUsize::new(steps)
            .ok_or_else(|| invalid("FSI Run steps must be strictly positive"))?;
        Self::new(plan, state, accepted_steps, output_steps)
    }

    pub fn from_times(
        plan: CommonFsiPlan,
        state: CommonState,
        until_s: f64,
        output_times_s: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        let step_s = plan.temporal().step().value();
        let accepted = exact_grid_index(state.time_s(), until_s, step_s, "until_s")?;
        let outputs = output_times_s
            .into_iter()
            .map(|time| exact_grid_index(state.time_s(), time, step_s, "output_times_s"))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(
            plan,
            state,
            NonZeroUsize::new(accepted)
                .ok_or_else(|| invalid("FSI Run horizon contains no accepted step"))?,
            outputs,
        )
    }

    pub(super) fn new(
        plan: CommonFsiPlan,
        state: CommonState,
        accepted_steps: NonZeroUsize,
        output_steps: Vec<usize>,
    ) -> Result<Self, Diagnostic> {
        if state.state_space_identity() != plan.state_space_identity() {
            return Err(invalid(
                "FSI Run State belongs to an incompatible common state space",
            ));
        }
        let schedule = CommonRunSchedule::new(
            b"eqiora.common-fsi-run-request/v1\0",
            &[],
            plan.identity(),
            state,
            accepted_steps,
            output_steps,
            "FSI output_steps must be nonempty, strictly increasing, and within the horizon",
        )?;
        Ok(Self { plan, schedule })
    }

    #[must_use]
    pub const fn plan(&self) -> &CommonFsiPlan {
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
    ) -> Result<ControlFlow<(usize, CommonState), Vec<(usize, CommonState)>>, Diagnostic> {
        advance_common_prepared_actions(
            self.schedule.state.clone(),
            self.schedule.accepted_steps.get(),
            &self.schedule.output_steps,
            self.plan.temporal().step().value(),
            |state| self.plan.prepare_execution(state, backend),
            |prepared, state| prepared.advance(state),
            stop_at_boundary,
        )
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
        if !until_s.is_finite() || until_s <= state.time_s() {
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
        let accepted_steps = exact_grid_index(state.time_s(), until_s, step_s, "until_s")?;
        let output_steps = output_times_s
            .into_iter()
            .map(|time| exact_grid_index(state.time_s(), time, step_s, "output_times_s"))
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
    ) -> Result<ControlFlow<(usize, CommonState), Vec<(usize, CommonState)>>, Diagnostic> {
        if let ResolvedCommonPlan::Scalar(plan) = &self.plan {
            let step_s = plan
                .admission
                .temporal
                .expect("validated temporal")
                .step()
                .value();
            let start_s = self.state().time_s();
            let mut accepted = 0;
            return advance_common_prepared_actions(
                self.schedule.state.clone(),
                self.schedule.accepted_steps.get(),
                &self.schedule.output_steps,
                plan.admission
                    .temporal
                    .expect("validated temporal")
                    .step()
                    .value(),
                |_| Ok(()),
                |(), state| {
                    accepted += 1;
                    plan.advance_scalar(state, backend, start_s + accepted as f64 * step_s)
                },
                stop_at_boundary,
            );
        }
        let ResolvedCommonPlan::TransientFlow(plan) = &self.plan else {
            return Err(invalid("unsupported transient Plan"));
        };
        advance_common_prepared_actions(
            self.schedule.state.clone(),
            self.schedule.accepted_steps.get(),
            &self.schedule.output_steps,
            plan.temporal().step().value(),
            |state| plan.prepare_execution(state, backend),
            |prepared, state| prepared.advance(state),
            stop_at_boundary,
        )
    }
}

fn exact_grid_index(
    start_s: f64,
    target_s: f64,
    step_s: f64,
    label: &str,
) -> Result<usize, Diagnostic> {
    if !target_s.is_finite() || target_s <= start_s {
        return Err(invalid(format!(
            "{label} values must be finite and later than State.time_s"
        )));
    }
    let raw = (target_s - start_s) / step_s;
    if !raw.is_finite() || raw < 1.0 || raw.fract() != 0.0 || raw > usize::MAX as f64 {
        return Err(invalid(format!(
            "{label} values must align exactly to the Plan Backward-Euler grid"
        )));
    }
    let index = raw as usize;
    let reconstructed = start_s + step_s * index as f64;
    if reconstructed.to_bits() != target_s.to_bits() {
        return Err(invalid(format!(
            "{label} values must align exactly to the Plan Backward-Euler grid"
        )));
    }
    Ok(index)
}

impl CommonState {
    pub(super) fn new(
        state_space_identity: String,
        time_s: f64,
        model: Arc<ModelEnvelope>,
        resources: Arc<NativeMeshResources>,
        kind: CommonStateKind,
    ) -> Result<Self, Diagnostic> {
        Self::new_with_boundary_forces(
            state_space_identity,
            time_s,
            model,
            resources,
            kind,
            Vec::new(),
        )
    }

    pub(super) fn new_with_boundary_forces(
        state_space_identity: String,
        time_s: f64,
        model: Arc<ModelEnvelope>,
        resources: Arc<NativeMeshResources>,
        kind: CommonStateKind,
        mut named_boundary_forces_on_domain: Vec<(String, [f64; 2])>,
    ) -> Result<Self, Diagnostic> {
        if !time_s.is_finite() || time_s < 0.0 || time_s.to_bits() == (-0.0_f64).to_bits() {
            return Err(invalid("State time_s must be finite and non-negative"));
        }
        named_boundary_forces_on_domain.sort_by(|left, right| left.0.cmp(&right.0));
        if named_boundary_forces_on_domain
            .windows(2)
            .any(|pair| pair[0].0 == pair[1].0)
            || named_boundary_forces_on_domain.iter().any(|(name, force)| {
                name.is_empty() || force.iter().any(|value| !value.is_finite())
            })
        {
            return Err(invalid(
                "State named boundary forces must have unique non-empty names and finite components",
            ));
        }
        let mut bytes = Vec::new();
        push_framed(&mut bytes, state_space_identity.as_bytes());
        bytes.extend_from_slice(&time_s.to_bits().to_be_bytes());
        match &kind {
            CommonStateKind::Scalar(values) => {
                push_framed(&mut bytes, b"scalar-q1/backward-euler");
                for value in values {
                    bytes.extend_from_slice(&value.to_bits().to_be_bytes());
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
            CommonStateKind::Fsi {
                state, pressure, ..
            } => {
                push_framed(
                    &mut bytes,
                    b"fixed-reference-fsi/mini-p1+p1/backward-euler/v1",
                );
                for value in state
                    .vertex_velocity()
                    .iter()
                    .flatten()
                    .chain(state.fluid_cell_bubble_velocity().values().flatten())
                    .chain(pressure.iter())
                    .chain(state.solid_displacement().iter().flatten())
                {
                    bytes.extend_from_slice(&value.to_bits().to_be_bytes());
                }
            }
        }
        for (name, force) in &named_boundary_forces_on_domain {
            push_framed(&mut bytes, name.as_bytes());
            for component in force {
                bytes.extend_from_slice(&component.to_bits().to_be_bytes());
            }
        }
        let identity = hex_bytes(&Sha256::digest(
            [b"eqiora.common-state/v1\0".as_slice(), bytes.as_slice()].concat(),
        ));
        Ok(Self {
            state_space_identity,
            identity,
            time_s,
            model,
            resources,
            kind,
            named_boundary_forces_on_domain,
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

    /// Exact coherent-SI model time.
    #[must_use]
    pub const fn time_s(&self) -> f64 {
        self.time_s
    }

    /// Intrinsic-2D force exerted on the fluid domain by one named boundary.
    #[must_use]
    pub fn named_boundary_force_on_domain(&self, name: &str) -> Option<[f64; 2]> {
        self.named_boundary_forces_on_domain
            .iter()
            .find_map(|(candidate, force)| (candidate == name).then_some(*force))
    }

    #[must_use]
    pub fn velocity_vertex_values(&self) -> Option<&[[f64; 2]]> {
        match &self.kind {
            CommonStateKind::Scalar(_) => None,
            CommonStateKind::MiniP1(state) => Some(state.velocity().vertex_values()),
            CommonStateKind::CellCentered(_) => None,
            CommonStateKind::Fsi { state, .. } => Some(state.vertex_velocity()),
        }
    }

    #[must_use]
    pub fn velocity_cell_values(&self) -> Vec<[f64; 2]> {
        match &self.kind {
            CommonStateKind::Scalar(_) => Vec::new(),
            CommonStateKind::MiniP1(state) => state.velocity().cell_bubble_values().to_vec(),
            CommonStateKind::CellCentered(state) => state.velocity().values().to_vec(),
            CommonStateKind::Fsi { state, .. } => state
                .fluid_cell_bubble_velocity()
                .values()
                .copied()
                .collect(),
        }
    }

    #[must_use]
    pub fn pressure_vertex_values(&self) -> Option<&[f64]> {
        match &self.kind {
            CommonStateKind::Scalar(_) => None,
            CommonStateKind::MiniP1(state) => Some(state.pressure().vertex_values()),
            CommonStateKind::CellCentered(_) => None,
            CommonStateKind::Fsi { pressure, .. } => Some(pressure),
        }
    }

    #[must_use]
    pub fn pressure_cell_values(&self) -> Option<&[f64]> {
        match &self.kind {
            CommonStateKind::Scalar(_) | CommonStateKind::MiniP1(_) => None,
            CommonStateKind::CellCentered(state) => Some(state.pressure().values()),
            CommonStateKind::Fsi { .. } => None,
        }
    }

    #[must_use]
    pub fn fsi_solid_displacement_values(&self) -> Option<&[[f64; 2]]> {
        match &self.kind {
            CommonStateKind::Fsi { state, .. } => Some(state.solid_displacement()),
            _ => None,
        }
    }

    #[must_use]
    pub fn fsi_accepted_solution(&self) -> Option<&ResolvedFixedReferenceFsiSolution2d> {
        match &self.kind {
            CommonStateKind::Fsi { accepted, .. } => accepted.as_deref(),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(super) fn method_history_values(&self) -> &[f64] {
        match &self.kind {
            CommonStateKind::Scalar(_) | CommonStateKind::MiniP1(_) => &[],
            CommonStateKind::CellCentered(state) => state.previous_face_volume_fluxes(),
            CommonStateKind::Fsi { .. } => &[],
        }
    }
}
