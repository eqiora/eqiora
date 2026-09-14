//! Accepted physical fields and falsifying numerical evidence.

use eqiora_meshing::CellId;
use std::collections::BTreeMap;
use std::sync::Arc;

use eqiora_assembly::AssemblyReport;
use eqiora_meshing::VertexId;
use eqiora_solver::{CanonicalCsrSystemView, SolveReport};

/// Independently recovered fluid and solid actions on one free interface vertex.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedReferenceFsiInterfaceAction<const D: usize> {
    pub(super) vertex: VertexId,
    pub(super) fluid: [f64; D],
    pub(super) solid: [f64; D],
}

impl<const D: usize> FixedReferenceFsiInterfaceAction<D> {
    /// Shared interface vertex.
    #[must_use]
    pub const fn vertex(self) -> VertexId {
        self.vertex
    }

    /// Fluid-side discrete action in the complete physical system.
    #[must_use]
    pub const fn fluid(self) -> [f64; D] {
        self.fluid
    }

    /// Solid-side discrete action in the complete physical system.
    #[must_use]
    pub const fn solid(self) -> [f64; D] {
        self.solid
    }

    /// Sum which must vanish at an unconstrained shared interface unknown.
    #[must_use]
    pub fn imbalance(self) -> [f64; D] {
        std::array::from_fn(|component| self.fluid[component] + self.solid[component])
    }
}

/// Discrete backward-Euler energy identity terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedReferenceFsiEnergyBalance {
    pub(super) previous_kinetic: f64,
    pub(super) next_kinetic: f64,
    pub(super) previous_elastic: f64,
    pub(super) next_elastic: f64,
    pub(super) kinetic_increment: f64,
    pub(super) elastic_increment: f64,
    pub(super) viscous_dissipation: f64,
    pub(super) defect: f64,
}

impl FixedReferenceFsiEnergyBalance {
    /// Previous fluid-plus-solid kinetic energy.
    #[must_use]
    pub const fn previous_kinetic(self) -> f64 {
        self.previous_kinetic
    }

    /// Accepted next fluid-plus-solid kinetic energy.
    #[must_use]
    pub const fn next_kinetic(self) -> f64 {
        self.next_kinetic
    }

    /// Previous solid elastic energy.
    #[must_use]
    pub const fn previous_elastic(self) -> f64 {
        self.previous_elastic
    }

    /// Accepted next solid elastic energy.
    #[must_use]
    pub const fn next_elastic(self) -> f64 {
        self.next_elastic
    }

    /// Backward-Euler kinetic numerical dissipation.
    #[must_use]
    pub const fn kinetic_increment(self) -> f64 {
        self.kinetic_increment
    }

    /// Backward-Euler elastic numerical dissipation.
    #[must_use]
    pub const fn elastic_increment(self) -> f64 {
        self.elastic_increment
    }

    /// Time-integrated physical viscous dissipation.
    #[must_use]
    pub const fn viscous_dissipation(self) -> f64 {
        self.viscous_dissipation
    }

    /// Sum of all energy-identity terms; zero external work is explicit.
    #[must_use]
    pub const fn defect(self) -> f64 {
        self.defect
    }
}

/// Accepted fields and falsifying numerical evidence for one FSI step.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedReferenceFsiSolution<const D: usize> {
    pub(super) vertex_velocity: Vec<[f64; D]>,
    pub(super) fluid_cell_bubble_velocity: BTreeMap<CellId, [f64; D]>,
    pub(super) fluid_pressure_vertices: Vec<VertexId>,
    pub(super) fluid_pressure: Vec<f64>,
    pub(super) solid_displacement: Vec<[f64; D]>,
    pub(super) algebraic_values: Vec<f64>,
    pub(super) canonical_system: Arc<CanonicalCsrSystemView>,
    pub(super) pressure_constant_action_norm: f64,
    pub(super) residual_norm: f64,
    pub(super) continuity_residual_norm: f64,
    pub(super) kinematic_residual_norm: f64,
    pub(super) interface_velocity_jump_norm: f64,
    pub(super) interface_actions: Vec<FixedReferenceFsiInterfaceAction<D>>,
    pub(super) interface_action_imbalance_norm: f64,
    pub(super) energy: FixedReferenceFsiEnergyBalance,
    pub(super) assembly_report: AssemblyReport,
    pub(super) solve_report: SolveReport,
}

impl<const D: usize> FixedReferenceFsiSolution<D> {
    /// Accepted shared mesh-vertex velocity coefficients.
    #[must_use]
    pub fn vertex_velocity(&self) -> &[[f64; D]] {
        &self.vertex_velocity
    }

    /// Accepted fluid MINI bubble coefficients keyed by exact fluid `CellId`.
    #[must_use]
    pub fn fluid_cell_bubble_velocity(&self) -> &BTreeMap<CellId, [f64; D]> {
        &self.fluid_cell_bubble_velocity
    }

    /// Fluid pressure vertex identities in coefficient order.
    #[must_use]
    pub fn fluid_pressure_vertices(&self) -> &[VertexId] {
        &self.fluid_pressure_vertices
    }

    /// Accepted fluid P1 pressure coefficients.
    #[must_use]
    pub fn fluid_pressure(&self) -> &[f64] {
        &self.fluid_pressure
    }

    /// Accepted next solid displacement, exact zero outside the solid closure.
    #[must_use]
    pub fn solid_displacement(&self) -> &[[f64; D]] {
        &self.solid_displacement
    }

    /// Dimensionless reduced values in deterministic velocity/bubble/pressure block order.
    #[must_use]
    pub fn algebraic_values(&self) -> &[f64] {
        &self.algebraic_values
    }

    /// Exact captured CSR system submitted to the solver.
    #[must_use]
    pub fn linear_system(&self) -> &CanonicalCsrSystemView {
        self.canonical_system.as_ref()
    }

    /// Dimensionless complete-operator action norm for unit dimensionless pressure.
    #[must_use]
    pub const fn pressure_constant_action_norm(&self) -> f64 {
        self.pressure_constant_action_norm
    }

    /// Independently reapplied dimensionless reduced CSR residual norm.
    #[must_use]
    pub const fn residual_norm(&self) -> f64 {
        self.residual_norm
    }

    /// Dimensionless norm of all fluid P1 incompressibility rows.
    #[must_use]
    pub const fn continuity_residual_norm(&self) -> f64 {
        self.continuity_residual_norm
    }

    /// Norm of `d_next - d_previous - dt * v_next` on the solid closure.
    #[must_use]
    pub const fn kinematic_residual_norm(&self) -> f64 {
        self.kinematic_residual_norm
    }

    /// Shared-trace velocity jump.  It is structurally zero for this quotient.
    #[must_use]
    pub const fn interface_velocity_jump_norm(&self) -> f64 {
        self.interface_velocity_jump_norm
    }

    /// Independently recovered free-interface actions.
    #[must_use]
    pub fn interface_actions(&self) -> &[FixedReferenceFsiInterfaceAction<D>] {
        &self.interface_actions
    }

    /// Euclidean norm of the summed fluid/solid free-interface actions.
    #[must_use]
    pub const fn interface_action_imbalance_norm(&self) -> f64 {
        self.interface_action_imbalance_norm
    }

    /// Accepted zero-load backward-Euler energy identity.
    #[must_use]
    pub const fn energy_balance(&self) -> FixedReferenceFsiEnergyBalance {
        self.energy
    }

    /// Complete assembly placement evidence.
    #[must_use]
    pub const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    /// Selected backend convergence evidence.
    #[must_use]
    pub const fn solve_report(&self) -> &SolveReport {
        &self.solve_report
    }
}
