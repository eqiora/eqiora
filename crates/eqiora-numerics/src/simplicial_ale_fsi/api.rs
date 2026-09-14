//! Accepted outputs and falsifying evidence for fixed-topology ALE FSI.
//!
//! This surface records independently checkable residual, interface,
//! geometry, and linearization evidence. It deliberately makes no fixed-domain
//! energy-equality claim: moving-volume energetics require a separate theorem
//! and acceptance contract.

use eqiora_assembly::AssemblyReport;
use eqiora_core::Diagnostic;
use eqiora_meshing::FixedTopologyGeometryAction;
use eqiora_meshing::VertexId;
use eqiora_solver::{ExecutionTopology, SolveReport};

use super::{AleFsiState, AleFsiStepPlan, invalid};

/// Independently recovered fluid and solid actions on one interface vertex.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AleFsiInterfaceAction<const D: usize> {
    vertex: VertexId,
    fluid: [f64; D],
    solid: [f64; D],
}

impl<const D: usize> AleFsiInterfaceAction<D> {
    /// Admit one finite pair of physical nodal actions.
    pub(super) fn new(
        vertex: VertexId,
        fluid: [f64; D],
        solid: [f64; D],
    ) -> Result<Self, Diagnostic> {
        if !matches!(D, 2 | 3) {
            return Err(invalid(
                "fixed-topology ALE FSI interface actions admit dimensions two and three",
            ));
        }
        if fluid.iter().chain(&solid).any(|value| !value.is_finite()) {
            return Err(invalid(
                "fixed-topology ALE FSI interface actions must be finite",
            ));
        }
        Ok(Self {
            vertex,
            fluid,
            solid,
        })
    }

    /// Shared interface vertex in immutable reference order.
    #[must_use]
    pub const fn vertex(self) -> VertexId {
        self.vertex
    }

    /// Fluid-side physical nodal action.
    #[must_use]
    pub const fn fluid(self) -> [f64; D] {
        self.fluid
    }

    /// Solid-side physical nodal action after the explicit configuration bridge.
    #[must_use]
    pub const fn solid(self) -> [f64; D] {
        self.solid
    }

    /// Fluid-plus-solid action which vanishes on an unconstrained shared trace.
    #[must_use]
    pub fn imbalance(self) -> [f64; D] {
        std::array::from_fn(|component| self.fluid[component] + self.solid[component])
    }

    /// Euclidean magnitude of [`Self::imbalance`].
    #[must_use]
    pub fn imbalance_norm(self) -> f64 {
        self.imbalance()
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
    }

    /// Fluid-side power at one finite shared physical interface velocity.
    ///
    /// # Errors
    /// Returns `EQ0801` when `shared_velocity` is non-finite or the dot product
    /// overflows.
    pub fn fluid_power(self, shared_velocity: [f64; D]) -> Result<f64, Diagnostic> {
        finite_power(self.fluid, shared_velocity)
    }

    /// Solid-side power at one finite shared physical interface velocity.
    ///
    /// # Errors
    /// Returns `EQ0801` when `shared_velocity` is non-finite or the dot product
    /// overflows.
    pub fn solid_power(self, shared_velocity: [f64; D]) -> Result<f64, Diagnostic> {
        finite_power(self.solid, shared_velocity)
    }

    /// Signed fluid-plus-solid interface power defect.
    ///
    /// # Errors
    /// Returns `EQ0801` when either side's evaluation is non-finite.
    pub fn power_imbalance(self, shared_velocity: [f64; D]) -> Result<f64, Diagnostic> {
        let value = self.fluid_power(shared_velocity)? + self.solid_power(shared_velocity)?;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(invalid(
                "fixed-topology ALE FSI interface-power imbalance overflowed",
            ))
        }
    }
}

/// Named internal measurements submitted to the acceptance boundary.
///
/// Geometry quality, metric identity, action imbalance, power imbalance, and
/// the nonlinear target are intentionally absent: the constructor derives
/// those from the bound geometry action, accepted state, interface actions,
/// and common step plan.
pub(super) struct AleFsiStepEvidenceInput<const D: usize> {
    pub(super) nonlinear_iterations: usize,
    pub(super) initial_residual_norm: f64,
    pub(super) final_residual_norm: f64,
    pub(super) continuity_residual_norm: f64,
    pub(super) kinematic_residual_norm: f64,
    pub(super) interface_velocity_jump_norm: f64,
    pub(super) interface_actions: Vec<AleFsiInterfaceAction<D>>,
    pub(super) probed_moving_fluid_cell_count: usize,
    pub(super) gcl_active_moving_fluid_cell_count: usize,
    pub(super) compatible_constant_free_stream_residual_norm: f64,
    pub(super) omitted_gcl_witness_norm: f64,
    pub(super) assembly_report: AssemblyReport,
    pub(super) nonlinear_linear_solves: Vec<SolveReport>,
}

/// Independently accepted evidence for one monolithic ALE FSI step.
#[derive(Debug, Clone, PartialEq)]
pub struct AleFsiStepEvidence<const D: usize> {
    accepted_time: f64,
    nonlinear_iterations: usize,
    initial_residual_norm: f64,
    residual_target: f64,
    final_residual_norm: f64,
    continuity_residual_norm: f64,
    kinematic_residual_norm: f64,
    interface_velocity_jump_norm: f64,
    interface_actions: Vec<AleFsiInterfaceAction<D>>,
    interface_action_imbalance_norm: f64,
    interface_power_imbalance: f64,
    maximum_affine_metric_identity_defect: f64,
    minimum_current_mean_ratio: f64,
    minimum_current_signed_jacobian: f64,
    minimum_path_signed_jacobian: f64,
    probed_moving_fluid_cell_count: usize,
    gcl_active_moving_fluid_cell_count: usize,
    compatible_constant_free_stream_residual_norm: f64,
    omitted_gcl_witness_norm: f64,
    assembly_report: AssemblyReport,
    nonlinear_linear_solves: Vec<SolveReport>,
}

impl<const D: usize> AleFsiStepEvidence<D> {
    /// Bind independently measured residuals to one exact accepted geometry.
    ///
    /// # Errors
    /// Returns `EQ0801` for non-finite or sign-invalid evidence, a stale state
    /// or geometry action, non-canonical interface actions, an unaccepted
    /// nonlinear residual, inconsistent Newton/Krylov counts or plans, or
    /// execution evidence outside the serial-host reference boundary.
    pub(super) fn new(
        plan: AleFsiStepPlan<D>,
        geometry: &FixedTopologyGeometryAction<D>,
        accepted: &AleFsiState<D>,
        input: AleFsiStepEvidenceInput<D>,
    ) -> Result<Self, Diagnostic> {
        if geometry.time_step() != plan.time_step() || geometry.current() != accepted.geometry() {
            return Err(invalid(
                "fixed-topology ALE FSI evidence must bind the accepted state and exact step geometry",
            ));
        }

        let scalar_norms = [
            input.initial_residual_norm,
            input.final_residual_norm,
            input.continuity_residual_norm,
            input.kinematic_residual_norm,
            input.interface_velocity_jump_norm,
            input.compatible_constant_free_stream_residual_norm,
            input.omitted_gcl_witness_norm,
        ];
        if scalar_norms
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(invalid(
                "fixed-topology ALE FSI residual and verification norms must be finite and non-negative",
            ));
        }
        let free_stream_tolerance =
            65_536.0 * f64::EPSILON * (1.0 + input.omitted_gcl_witness_norm);
        if input.gcl_active_moving_fluid_cell_count > input.probed_moving_fluid_cell_count
            || (input.probed_moving_fluid_cell_count == 0
                && (input.compatible_constant_free_stream_residual_norm != 0.0
                    || input.omitted_gcl_witness_norm != 0.0))
            || (input.gcl_active_moving_fluid_cell_count > 0
                && input.omitted_gcl_witness_norm == 0.0)
            || input.compatible_constant_free_stream_residual_norm > free_stream_tolerance
        {
            return Err(invalid(
                "fixed-topology ALE FSI constant-free-stream probe is inconsistent with its moving-cell and omitted-GCL witness",
            ));
        }

        let nonlinear = plan.nonlinear();
        let residual_target = nonlinear
            .absolute_tolerance()
            .max(nonlinear.relative_tolerance() * input.initial_residual_norm);
        if !residual_target.is_finite()
            || input.final_residual_norm > residual_target
            || input.final_residual_norm > input.initial_residual_norm
            || input.nonlinear_iterations > nonlinear.maximum_iterations().get()
            || input.nonlinear_linear_solves.len() != input.nonlinear_iterations
            || (input.nonlinear_iterations == 0
                && (input.initial_residual_norm > residual_target
                    || input.final_residual_norm != input.initial_residual_norm))
            || (input.nonlinear_iterations > 0 && input.initial_residual_norm <= residual_target)
        {
            return Err(invalid(
                "fixed-topology ALE FSI nonlinear evidence is inconsistent with the common acceptance plan",
            ));
        }
        if input.nonlinear_linear_solves.iter().any(|report| {
            report.solver_plan() != plan.linear_solver()
                || !serial_host(report.execution().topology())
                || !serial_host(report.verification().topology())
        }) {
            return Err(invalid(
                "fixed-topology ALE FSI Krylov evidence must use the exact step plan and serial-host execution",
            ));
        }
        if input.assembly_report.packet_count() == 0
            || input.assembly_report.target_count() == 0
            || !serial_host(input.assembly_report.execution().topology())
        {
            return Err(invalid(
                "fixed-topology ALE FSI final assembly evidence must be non-empty and serial-host",
            ));
        }
        validate_interface_order(&input.interface_actions)?;

        let interface_action_imbalance_norm = input
            .interface_actions
            .iter()
            .flat_map(|action| action.imbalance())
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let signed_power_imbalance = input.interface_actions.iter().try_fold(
            0.0,
            |sum, action| -> Result<f64, Diagnostic> {
                let mut velocities = plan
                    .material()
                    .kinetic_fields()
                    .filter_map(|field| {
                        accepted
                            .physical_state()
                            .vector_vertices(field)
                            .ok()
                            .and_then(|values| values.get(&action.vertex()).copied())
                    });
                let velocity = velocities.next().ok_or_else(|| {
                    invalid(
                        "fixed-topology ALE FSI interface action is outside the exact kinetic Field inventory",
                    )
                })?;
                if velocities.any(|other| other != velocity) {
                    return Err(invalid(
                        "fixed-topology ALE FSI interface kinetic Fields disagree on their shared trace",
                    ));
                }
                let next = sum + action.power_imbalance(velocity)?;
                if next.is_finite() {
                    Ok(next)
                } else {
                    Err(invalid(
                        "fixed-topology ALE FSI global interface-power imbalance overflowed",
                    ))
                }
            },
        )?;
        let interface_power_imbalance = signed_power_imbalance.abs();
        if !interface_action_imbalance_norm.is_finite() || !interface_power_imbalance.is_finite() {
            return Err(invalid(
                "fixed-topology ALE FSI interface balance evidence must be finite",
            ));
        }

        let maximum_affine_metric_identity_defect = geometry
            .cells()
            .iter()
            .map(|cell| cell.metric_identity_defect().abs())
            .fold(0.0_f64, f64::max);
        let quality = geometry.current().quality_report();
        let minimum_current_mean_ratio = quality.minimum_mean_ratio();
        let minimum_current_signed_jacobian = quality.minimum_signed_measure_scale();
        let minimum_path_signed_jacobian = geometry.minimum_path_signed_measure_scale();
        if !maximum_affine_metric_identity_defect.is_finite()
            || maximum_affine_metric_identity_defect < 0.0
            || !minimum_current_mean_ratio.is_finite()
            || !(0.0..=1.0).contains(&minimum_current_mean_ratio)
            || !minimum_current_signed_jacobian.is_finite()
            || minimum_current_signed_jacobian <= 0.0
            || !minimum_path_signed_jacobian.is_finite()
            || minimum_path_signed_jacobian <= 0.0
        {
            return Err(invalid(
                "fixed-topology ALE FSI geometry evidence must be finite, positively oriented, and quality-admitted",
            ));
        }

        Ok(Self {
            accepted_time: accepted.time(),
            nonlinear_iterations: input.nonlinear_iterations,
            initial_residual_norm: input.initial_residual_norm,
            residual_target,
            final_residual_norm: input.final_residual_norm,
            continuity_residual_norm: input.continuity_residual_norm,
            kinematic_residual_norm: input.kinematic_residual_norm,
            interface_velocity_jump_norm: input.interface_velocity_jump_norm,
            interface_actions: input.interface_actions,
            interface_action_imbalance_norm,
            interface_power_imbalance,
            maximum_affine_metric_identity_defect,
            minimum_current_mean_ratio,
            minimum_current_signed_jacobian,
            minimum_path_signed_jacobian,
            probed_moving_fluid_cell_count: input.probed_moving_fluid_cell_count,
            gcl_active_moving_fluid_cell_count: input.gcl_active_moving_fluid_cell_count,
            compatible_constant_free_stream_residual_norm: input
                .compatible_constant_free_stream_residual_norm,
            omitted_gcl_witness_norm: input.omitted_gcl_witness_norm,
            assembly_report: input.assembly_report,
            nonlinear_linear_solves: input.nonlinear_linear_solves,
        })
    }

    /// Accepted state time to which this evidence is bound.
    #[must_use]
    pub const fn accepted_time(&self) -> f64 {
        self.accepted_time
    }

    /// Accepted damped-Newton updates.
    #[must_use]
    pub const fn nonlinear_iterations(&self) -> usize {
        self.nonlinear_iterations
    }

    /// Residual norm at the previous-state warm start.
    #[must_use]
    pub const fn initial_residual_norm(&self) -> f64 {
        self.initial_residual_norm
    }

    /// Frozen nonlinear acceptance threshold derived from the common plan.
    #[must_use]
    pub const fn residual_target(&self) -> f64 {
        self.residual_target
    }

    /// Independently reassembled final nonlinear residual norm.
    #[must_use]
    pub const fn final_residual_norm(&self) -> f64 {
        self.final_residual_norm
    }

    /// Weak fluid incompressibility residual norm.
    #[must_use]
    pub const fn continuity_residual_norm(&self) -> f64 {
        self.continuity_residual_norm
    }

    /// Solid backward-Euler kinematic residual norm.
    #[must_use]
    pub const fn kinematic_residual_norm(&self) -> f64 {
        self.kinematic_residual_norm
    }

    /// Physical velocity jump across the shared interface trace.
    #[must_use]
    pub const fn interface_velocity_jump_norm(&self) -> f64 {
        self.interface_velocity_jump_norm
    }

    /// Independently recovered interface actions in strict vertex order.
    #[must_use]
    pub fn interface_actions(&self) -> &[AleFsiInterfaceAction<D>] {
        &self.interface_actions
    }

    /// Euclidean norm of all fluid-plus-solid interface action components.
    #[must_use]
    pub const fn interface_action_imbalance_norm(&self) -> f64 {
        self.interface_action_imbalance_norm
    }

    /// Absolute global fluid-plus-solid interface power defect.
    #[must_use]
    pub const fn interface_power_imbalance(&self) -> f64 {
        self.interface_power_imbalance
    }

    /// Maximum absolute affine `dJ/dt - J div(w)` defect over all cells.
    #[must_use]
    pub const fn maximum_affine_metric_identity_defect(&self) -> f64 {
        self.maximum_affine_metric_identity_defect
    }

    /// Minimum current-cell mean-ratio quality.
    #[must_use]
    pub const fn minimum_current_mean_ratio(&self) -> f64 {
        self.minimum_current_mean_ratio
    }

    /// Minimum positive current-cell signed Jacobian.
    #[must_use]
    pub const fn minimum_current_signed_jacobian(&self) -> f64 {
        self.minimum_current_signed_jacobian
    }

    /// Minimum positive signed Jacobian over every complete affine path.
    #[must_use]
    pub const fn minimum_path_signed_jacobian(&self) -> f64 {
        self.minimum_path_signed_jacobian
    }

    /// Moving fluid cells tested with the compatible constant-stream probe.
    ///
    /// Momentum is tested only with the cell bubble whose trace vanishes.
    /// The value therefore does not claim that a nonzero constant velocity is
    /// admissible under the model's homogeneous exterior boundary condition.
    #[must_use]
    pub const fn probed_moving_fluid_cell_count(&self) -> usize {
        self.probed_moving_fluid_cell_count
    }

    /// Probed cells whose nonzero mesh divergence activates the GCL term.
    #[must_use]
    pub const fn gcl_active_moving_fluid_cell_count(&self) -> usize {
        self.gcl_active_moving_fluid_cell_count
    }

    /// Dimensionless residual norm of the compatible constant-stream probe.
    #[must_use]
    pub const fn compatible_constant_free_stream_residual_norm(&self) -> f64 {
        self.compatible_constant_free_stream_residual_norm
    }

    /// Norm that would remain on the same probe if the GCL correction vanished.
    ///
    /// This witness may be zero for static or exactly isochoric grid motion;
    /// it must be nonzero whenever a probed cell has nonzero mesh divergence.
    #[must_use]
    pub const fn omitted_gcl_witness_norm(&self) -> f64 {
        self.omitted_gcl_witness_norm
    }

    /// Accepted placement and packet shape of final independent reassembly.
    #[must_use]
    pub const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    /// Common-contract Krylov reports in nonlinear-update order.
    ///
    /// Harmonic influence-column reports remain on
    /// `P1HarmonicMeshMotionAction<D>`; they are not duplicated here.
    #[must_use]
    pub fn nonlinear_linear_solves(&self) -> &[SolveReport] {
        &self.nonlinear_linear_solves
    }
}

/// Initial state followed by accepted moving states and one evidence record per step.
#[derive(Debug, Clone, PartialEq)]
pub struct AleFsiTrajectory<const D: usize> {
    states: Vec<AleFsiState<D>>,
    steps: Vec<AleFsiStepEvidence<D>>,
}

impl<const D: usize> AleFsiTrajectory<D> {
    pub(crate) fn new(initial: AleFsiState<D>) -> Self {
        Self {
            states: vec![initial],
            steps: Vec::new(),
        }
    }

    /// Append one already accepted state/evidence pair atomically.
    ///
    /// Exact agreement with the step duration is checked by the solver before
    /// evidence construction. This container additionally requires strict time
    /// increase and the evidence's exact accepted-state time.
    pub(crate) fn push(
        &mut self,
        state: AleFsiState<D>,
        evidence: AleFsiStepEvidence<D>,
    ) -> Result<(), Diagnostic> {
        let previous = self
            .states
            .last()
            .expect("ALE FSI trajectory owns its initial state");
        if state.time() <= previous.time() || evidence.accepted_time() != state.time() {
            return Err(invalid(
                "accepted fixed-topology ALE FSI states must increase strictly and match their evidence time",
            ));
        }
        self.states
            .try_reserve(1)
            .map_err(|_| invalid("fixed-topology ALE FSI trajectory state allocation failed"))?;
        self.steps
            .try_reserve(1)
            .map_err(|_| invalid("fixed-topology ALE FSI trajectory evidence allocation failed"))?;
        self.states.push(state);
        self.steps.push(evidence);
        Ok(())
    }

    /// Initial state followed by accepted states in strict model-time order.
    #[must_use]
    pub fn states(&self) -> &[AleFsiState<D>] {
        &self.states
    }

    /// One evidence record per transition between adjacent states.
    #[must_use]
    pub fn steps(&self) -> &[AleFsiStepEvidence<D>] {
        &self.steps
    }

    /// Initial state of the trajectory.
    #[must_use]
    pub fn initial_state(&self) -> &AleFsiState<D> {
        self.states
            .first()
            .expect("ALE FSI trajectory owns its initial state")
    }

    /// Most recently accepted state, or the initial state before any step.
    #[must_use]
    pub fn final_state(&self) -> &AleFsiState<D> {
        self.states
            .last()
            .expect("ALE FSI trajectory owns its initial state")
    }
}

fn finite_power<const D: usize>(
    action: [f64; D],
    shared_velocity: [f64; D],
) -> Result<f64, Diagnostic> {
    if shared_velocity.iter().any(|value| !value.is_finite()) {
        return Err(invalid(
            "fixed-topology ALE FSI interface velocity must be finite",
        ));
    }
    let power = action
        .iter()
        .zip(shared_velocity)
        .map(|(action, velocity)| action * velocity)
        .sum::<f64>();
    if power.is_finite() {
        Ok(power)
    } else {
        Err(invalid(
            "fixed-topology ALE FSI interface-power evaluation overflowed",
        ))
    }
}

fn validate_interface_order<const D: usize>(
    actions: &[AleFsiInterfaceAction<D>],
) -> Result<(), Diagnostic> {
    if actions.is_empty()
        || actions
            .windows(2)
            .any(|pair| pair[0].vertex().index() >= pair[1].vertex().index())
    {
        return Err(invalid(
            "fixed-topology ALE FSI evidence requires non-empty interface actions in strict vertex order",
        ));
    }
    Ok(())
}

fn serial_host(topology: ExecutionTopology) -> bool {
    matches!(
        topology,
        ExecutionTopology::Host { workers } if workers == std::num::NonZeroUsize::MIN
    )
}

#[cfg(test)]
mod tests;
