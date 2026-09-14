//! Finalized assembly, solver handoff, and accepted-step construction.

use std::sync::Arc;

#[cfg(test)]
use eqiora_assembly::{AssemblyBackend, REFERENCE_ASSEMBLY_BACKEND};
use eqiora_assembly::{
    AssemblyPacket, AssemblyPacketSetIdentityV1, AssemblyPlan, AssemblyReport, AssemblyResult,
    AssemblyTarget, AssemblyTargetId, AssemblyWork, LinearSystem, TargetAssemblyMap,
};
use eqiora_core::Diagnostic;
use eqiora_meshing::{MeshEntity, MeshGeometry, MeshTopology, QuadratureRule, SimplicialMesh};
use eqiora_solver::{
    CanonicalCsrSystemView, LinearOperatorProperties, LinearSolution, LinearSolveRequest,
};

use super::acceptance::{
    EnergyEvaluation, apply_canonical, energy_balance, kinematic_residual_norm, norm,
    require_pressure_closed_by_complete_operator, require_symmetric,
};
use super::api::{FixedReferenceFsiInterfaceAction, FixedReferenceFsiSolution};
#[cfg(test)]
use super::contract::FixedReferenceFsiBoundary;
use super::contract::{FixedReferenceFsiState, FixedReferenceFsiStepConfig, validate_problem};
use super::element::{fluid_local, solid_local};
use super::invalid;
use super::layout::FsiLayout;
use super::partition::{CellMaterial, FixedReferenceFsiPartition};
use crate::region_assembly::InterfaceReactions;

/// Captured symmetric-indefinite step plus private acceptance state.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalizedFixedReferenceFsiStep<const D: usize> {
    canonical_system: Arc<CanonicalCsrSystemView>,
    state: FinalizedState<D>,
}

impl<const D: usize> FinalizedFixedReferenceFsiStep<D> {
    pub(crate) fn free_field_dof(
        &self,
        key: crate::region_assembly::mapping::FieldDof,
    ) -> Option<eqiora_assembly::DofId> {
        self.state.layout.free_field_dof(key)
    }
    pub(crate) fn canonical_system_arc(&self) -> Arc<CanonicalCsrSystemView> {
        Arc::clone(&self.canonical_system)
    }

    /// Exact semantic roles of the two assembly targets retained for the
    /// distributed-assembly composition boundary.
    pub(crate) const fn assembly_target_roles(&self) -> FixedReferenceFsiAssemblyTargetRoles {
        self.state.assembly_target_roles
    }

    /// Capture the retained full reconstruction system for an exact
    /// property-free assembly-identity comparison.
    pub(crate) fn full_canonical_system_view(&self) -> Result<CanonicalCsrSystemView, Diagnostic> {
        CanonicalCsrSystemView::new(&self.state.full_system, LinearOperatorProperties::General)
    }

    /// Exact captured CSR operator and RHS.
    #[must_use]
    pub fn linear_system(&self) -> &CanonicalCsrSystemView {
        self.canonical_system.as_ref()
    }

    /// Complete packet/placement evidence available before execution.
    #[must_use]
    pub const fn assembly_report(&self) -> &AssemblyReport {
        &self.state.assembly_report
    }

    /// Finish one solution returned for this exact captured system.
    ///
    /// # Errors
    /// Rejects a shape mismatch, excessive independently recomputed residual,
    /// failed incompressibility/kinematics/interface balance, or a failed
    /// zero-load backward-Euler energy identity.
    pub fn finish(
        self,
        solved: LinearSolution,
    ) -> Result<FixedReferenceFsiSolution<D>, Diagnostic> {
        self.state.finish(solved, self.canonical_system)
    }

    /// Execute and accept this exact finalized problem through one solver.
    ///
    /// # Errors
    /// Preserves solver and solution-acceptance diagnostics.
    pub fn solve(
        self,
        solver: LinearSolveRequest<'_>,
    ) -> Result<FixedReferenceFsiSolution<D>, Diagnostic> {
        let solved = solver.solve(&self.canonical_system.linear_problem()?)?;
        self.finish(solved)
    }
}

/// Finalize one three-dimensional fixed-reference monolithic step.
///
/// # Errors
/// Preserves generic fixed-reference admission and assembly diagnostics.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn finalize_fixed_reference_fsi_step_3d(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<3>,
    boundary: &FixedReferenceFsiBoundary<3>,
    previous: &FixedReferenceFsiState<3>,
    config: FixedReferenceFsiStepConfig<3>,
    quadrature: &QuadratureRule,
    layout: &FsiLayout<3>,
) -> Result<FinalizedFixedReferenceFsiStep<3>, Diagnostic> {
    finalize_fixed_reference_fsi_step_with_assembly(
        mesh,
        partition,
        boundary,
        previous,
        config,
        quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        layout,
    )
}

/// Finalize one dimension-typed fixed-reference step through an explicit
/// ordered assembly backend.
///
/// # Errors
/// Preserves admission, local-action, and selected assembly diagnostics.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn finalize_fixed_reference_fsi_step_with_assembly<const D: usize>(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    boundary: &FixedReferenceFsiBoundary<D>,
    previous: &FixedReferenceFsiState<D>,
    config: FixedReferenceFsiStepConfig<D>,
    quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend,
    layout: &FsiLayout<D>,
) -> Result<FinalizedFixedReferenceFsiStep<D>, Diagnostic> {
    finalize_fixed_reference_fsi_step_with_packet_set(
        mesh,
        partition,
        boundary,
        previous,
        config,
        quadrature,
        AssemblyPacketSetIdentityV1::Unbound,
        assembly,
        layout,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn finalize_fixed_reference_fsi_step_with_packet_set<const D: usize>(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    boundary: &FixedReferenceFsiBoundary<D>,
    previous: &FixedReferenceFsiState<D>,
    config: FixedReferenceFsiStepConfig<D>,
    quadrature: &QuadratureRule,
    packet_set: AssemblyPacketSetIdentityV1,
    assembly: &dyn AssemblyBackend,
    layout: &FsiLayout<D>,
) -> Result<FinalizedFixedReferenceFsiStep<D>, Diagnostic> {
    layout.require_boundary(boundary)?;
    let prepared = PreparedFixedReferenceFsiAssembly::new(
        mesh,
        partition,
        previous,
        config,
        quadrature,
        packet_set,
        layout.clone(),
    )?;
    let result = assembly.assemble(prepared.plan(), &prepared)?;
    let reactions = prepared.reactions(&prepared)?;
    prepared.finish(result, reactions)
}

/// Admitted fixed-reference data and the sole cell-indexed assembly work.
///
/// Packet index is exactly the canonical mesh-cell index. Backends may choose
/// placement, but they cannot introduce a second packet identity or local
/// operator path.
#[derive(Debug)]
pub(crate) struct PreparedFixedReferenceFsiAssembly<'a, const D: usize> {
    mesh: &'a SimplicialMesh,
    partition: &'a FixedReferenceFsiPartition<D>,
    previous: &'a FixedReferenceFsiState<D>,
    config: FixedReferenceFsiStepConfig<D>,
    quadrature: &'a QuadratureRule,
    layout: FsiLayout<D>,
    plan: AssemblyPlan,
    target_roles: FixedReferenceFsiAssemblyTargetRoles,
    cell_count: usize,
    packet_set: AssemblyPacketSetIdentityV1,
}

/// Plan-local identities of the physical-acceptance assembly targets.
///
/// The identities are deliberately crate-private: they let the later
/// distributed bridge bind evidence to the exact reduced solve system and
/// full reconstruction system without exposing either as a physics IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FixedReferenceFsiAssemblyTargetRoles {
    reduced: AssemblyTargetId,
    full: AssemblyTargetId,
}

impl FixedReferenceFsiAssemblyTargetRoles {
    fn from_plan(plan: &AssemblyPlan) -> Result<Self, Diagnostic> {
        if plan.target_count() != 2 {
            return Err(invalid(
                "fixed-reference FSI assembly plan must own exactly reduced and full targets",
            ));
        }
        Ok(Self {
            reduced: plan
                .target_id(0)
                .ok_or_else(|| invalid("fixed-reference FSI plan omits its reduced target"))?,
            full: plan
                .target_id(1)
                .ok_or_else(|| invalid("fixed-reference FSI plan omits its full target"))?,
        })
    }

    /// Essential-boundary-eliminated system submitted to the solver.
    pub(crate) const fn reduced(self) -> AssemblyTargetId {
        self.reduced
    }

    /// Full system retained for physical residual and reaction acceptance.
    pub(crate) const fn full(self) -> AssemblyTargetId {
        self.full
    }
}

impl<'a, const D: usize> PreparedFixedReferenceFsiAssembly<'a, D> {
    pub(crate) fn new(
        mesh: &'a SimplicialMesh,
        partition: &'a FixedReferenceFsiPartition<D>,
        previous: &'a FixedReferenceFsiState<D>,
        config: FixedReferenceFsiStepConfig<D>,
        quadrature: &'a QuadratureRule,
        packet_set: AssemblyPacketSetIdentityV1,
        layout: FsiLayout<D>,
    ) -> Result<Self, Diagnostic> {
        let boundary = layout.boundary();
        validate_problem(mesh, partition, boundary, previous, config, quadrature)?;
        layout.require_reference(mesh, partition)?;
        layout.require_boundary(boundary)?;
        layout.require_scale(config.scale())?;
        let plan = AssemblyPlan::new(vec![
            AssemblyTarget::new(layout.reduced_size())?,
            AssemblyTarget::new(layout.full_size())?,
        ])?;
        let target_roles = FixedReferenceFsiAssemblyTargetRoles::from_plan(&plan)?;
        let cell_count = mesh.entity_count(D).ok_or_else(|| {
            invalid("fixed-reference FSI mesh omits its top-dimensional cell stratum")
        })?;
        Ok(Self {
            mesh,
            partition,
            previous,
            config,
            quadrature,
            layout,
            plan,
            target_roles,
            cell_count,
            packet_set,
        })
    }

    pub(crate) const fn plan(&self) -> &AssemblyPlan {
        &self.plan
    }

    pub(crate) const fn layout(&self) -> &FsiLayout<D> {
        &self.layout
    }

    pub(crate) const fn target_roles(&self) -> FixedReferenceFsiAssemblyTargetRoles {
        self.target_roles
    }

    pub(crate) fn reactions(
        &self,
        work: &dyn AssemblyWork,
    ) -> Result<InterfaceReactions, Diagnostic> {
        self.layout.reactions(work, self.target_roles.full())
    }

    pub(crate) fn finish(
        self,
        result: AssemblyResult,
        reactions: InterfaceReactions,
    ) -> Result<FinalizedFixedReferenceFsiStep<D>, Diagnostic> {
        let (systems, assembly_report) = result.into_parts();
        if assembly_report.packet_count() != self.cell_count
            || assembly_report.target_count() != self.plan.target_count()
        {
            return Err(invalid(
                "fixed-reference FSI assembly evidence differs from its prepared cell/target inventory",
            ));
        }
        let systems: [LinearSystem; 2] = systems.try_into().map_err(|systems: Vec<_>| {
            invalid(format!(
                "fixed-reference FSI assembly returned {} systems for its exact two-target plan",
                systems.len()
            ))
        })?;
        let [reduced_system, full_system] = systems;
        require_system_shape(&reduced_system, self.layout.reduced_size(), "reduced")?;
        require_system_shape(&full_system, self.layout.full_size(), "full")?;
        require_symmetric(reduced_system.matrix())?;
        let pressure_constant_action_norm =
            require_pressure_closed_by_complete_operator(&reduced_system, &self.layout)?;
        let canonical_system = Arc::new(CanonicalCsrSystemView::new(
            &reduced_system,
            LinearOperatorProperties::SymmetricIndefinite,
        )?);
        Ok(FinalizedFixedReferenceFsiStep {
            canonical_system,
            state: FinalizedState {
                mesh: self.mesh.clone(),
                partition: self.partition.clone(),
                previous: self.previous.clone(),
                config: self.config,
                quadrature: self.quadrature.clone(),
                layout: self.layout,
                full_system,
                assembly_target_roles: self.target_roles,
                pressure_constant_action_norm,
                assembly_report,
                reactions,
            },
        })
    }
}

impl<const D: usize> AssemblyWork for PreparedFixedReferenceFsiAssembly<'_, D> {
    fn packet_set_identity(&self) -> AssemblyPacketSetIdentityV1 {
        self.packet_set
    }

    fn packet_count(&self) -> usize {
        self.cell_count
    }

    fn evaluate(&self, packet_index: usize) -> Result<AssemblyPacket, Diagnostic> {
        if packet_index >= self.cell_count {
            return Err(invalid(format!(
                "fixed-reference FSI packet {packet_index} is outside cell count {}",
                self.cell_count
            )));
        }
        let cell = MeshEntity::new(D, packet_index);
        let geometry = self.mesh.geometry_map(cell).ok_or_else(|| {
            invalid(format!(
                "fixed-reference FSI cell packet {packet_index} has no affine geometry"
            ))
        })?;
        let vertices = self.mesh.entity_vertices(cell).ok_or_else(|| {
            invalid(format!(
                "fixed-reference FSI cell packet {packet_index} has no vertex closure"
            ))
        })?;
        match self.partition.material(packet_index) {
            CellMaterial::Fluid => {
                let position = self.partition.fluid_position(packet_index).ok_or_else(|| {
                    invalid(format!(
                        "fixed-reference FSI fluid packet {packet_index} has no bubble position"
                    ))
                })?;
                let local = fluid_local(
                    &geometry,
                    self.quadrature,
                    self.config,
                    &vertices,
                    self.previous,
                    position,
                )?;
                let reduced = self.layout.fluid_map(position, &vertices, true)?;
                let full = self.layout.fluid_map(position, &vertices, false)?;
                AssemblyPacket::new(
                    local,
                    vec![
                        TargetAssemblyMap::new(self.target_roles.reduced(), reduced),
                        TargetAssemblyMap::new(self.target_roles.full(), full),
                    ],
                )
            }
            CellMaterial::Solid => {
                let local = solid_local(
                    &geometry,
                    self.quadrature,
                    self.config,
                    &vertices,
                    self.previous,
                )?;
                let reduced = self.layout.solid_map(packet_index, &vertices, true)?;
                let full = self.layout.solid_map(packet_index, &vertices, false)?;
                AssemblyPacket::new(
                    local,
                    vec![
                        TargetAssemblyMap::new(self.target_roles.reduced(), reduced),
                        TargetAssemblyMap::new(self.target_roles.full(), full),
                    ],
                )
            }
            CellMaterial::Unassigned => Err(invalid(format!(
                "fixed-reference FSI cell packet {packet_index} has no material assignment"
            ))),
        }
    }
}

fn require_system_shape(
    system: &LinearSystem,
    expected: usize,
    target: &'static str,
) -> Result<(), Diagnostic> {
    if system.matrix().rows() != expected
        || system.matrix().columns() != expected
        || system.rhs().len() != expected
    {
        return Err(invalid(format!(
            "fixed-reference FSI {target} assembly system differs from its prepared target shape"
        )));
    }
    Ok(())
}

/// Finalize, execute, and accept one three-dimensional fixed-reference step.
///
/// # Errors
/// Preserves all generic admission, assembly, solver, and acceptance diagnostics.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn solve_fixed_reference_fsi_step_3d(
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<3>,
    boundary: &FixedReferenceFsiBoundary<3>,
    previous: &FixedReferenceFsiState<3>,
    config: FixedReferenceFsiStepConfig<3>,
    quadrature: &QuadratureRule,
    solver: LinearSolveRequest<'_>,
    layout: &FsiLayout<3>,
) -> Result<FixedReferenceFsiSolution<3>, Diagnostic> {
    finalize_fixed_reference_fsi_step_3d(
        mesh, partition, boundary, previous, config, quadrature, layout,
    )?
    .solve(solver)
}

#[derive(Debug, Clone, PartialEq)]
struct FinalizedState<const D: usize> {
    mesh: SimplicialMesh,
    partition: FixedReferenceFsiPartition<D>,
    previous: FixedReferenceFsiState<D>,
    config: FixedReferenceFsiStepConfig<D>,
    quadrature: QuadratureRule,
    layout: FsiLayout<D>,
    full_system: LinearSystem,
    assembly_target_roles: FixedReferenceFsiAssemblyTargetRoles,
    pressure_constant_action_norm: f64,
    assembly_report: AssemblyReport,
    reactions: InterfaceReactions,
}

impl<const D: usize> FinalizedState<D> {
    fn finish(
        self,
        solved: LinearSolution,
        canonical_system: Arc<CanonicalCsrSystemView>,
    ) -> Result<FixedReferenceFsiSolution<D>, Diagnostic> {
        if solved.values().len() != canonical_system.rows() {
            return Err(invalid(
                "fixed-reference FSI solver result differs from its captured system",
            ));
        }
        let residual_target = solved.report().residual_target();
        let (algebraic_values, solve_report) = solved.into_parts();
        let (dimensionless_vertex_velocity, dimensionless_fluid_bubbles, dimensionless_pressure) =
            self.layout
                .reconstruct_primal(&algebraic_values, self.partition.fluid_cells().len())?;
        let full_values = self.layout.fill_full(
            &dimensionless_vertex_velocity,
            &dimensionless_fluid_bubbles,
            &dimensionless_pressure,
        );
        let (vertex_velocity, fluid_bubbles, pressure) =
            self.layout.reconstruct_physical(&algebraic_values)?;

        let mut reduced_residual = apply_canonical(&canonical_system, &algebraic_values)?;
        for (value, rhs) in reduced_residual
            .iter_mut()
            .zip(canonical_system.right_hand_side())
        {
            *value -= rhs;
        }
        let residual_norm = norm(&reduced_residual);
        let residual_tolerance = residual_target
            + 4096.0
                * f64::EPSILON
                * (1.0 + norm(&algebraic_values) + norm(canonical_system.right_hand_side()));
        if !residual_norm.is_finite() || residual_norm > residual_tolerance {
            return Err(invalid(format!(
                "fixed-reference FSI reapplied CSR residual {residual_norm:e} exceeds {residual_tolerance:e}"
            )));
        }

        let mut full_residual = self.full_system.matrix().multiply(&full_values)?;
        for (value, rhs) in full_residual.iter_mut().zip(self.full_system.rhs()) {
            *value -= rhs;
        }
        let continuity_residual_norm = norm(
            &self
                .layout
                .full_pressure_dofs()
                .into_iter()
                .map(|dof| full_residual[dof])
                .collect::<Vec<_>>(),
        );
        if continuity_residual_norm > residual_tolerance {
            return Err(invalid(format!(
                "fixed-reference FSI incompressibility residual {continuity_residual_norm:e} exceeds {residual_tolerance:e}"
            )));
        }

        let mut solid_displacement = self.previous.solid_displacement().to_vec();
        for vertex in self.partition.solid_vertices() {
            for component in 0..D {
                solid_displacement[vertex.index()][component] +=
                    self.config.time_step() * vertex_velocity[vertex.index()][component];
            }
        }
        let kinematic_residual_norm = kinematic_residual_norm(
            &self.partition,
            &self.previous,
            &vertex_velocity,
            &solid_displacement,
            self.config.time_step(),
        );
        let kinematic_tolerance = 4096.0
            * f64::EPSILON
            * self.config.scale().length()
            * (D as f64 * self.partition.solid_vertices().len() as f64).sqrt();
        if kinematic_residual_norm > kinematic_tolerance {
            return Err(invalid(format!(
                "fixed-reference FSI kinematic residual {kinematic_residual_norm:e} exceeds {kinematic_tolerance:e}"
            )));
        }

        let reactions = self.reactions.recover(&full_values)?;
        let dimensionless_to_action = self.config.scale().power() / self.config.scale().velocity();
        let interface_actions = self
            .partition
            .interface_vertices()
            .iter()
            .copied()
            .filter(|vertex| !self.layout.fixed_velocity(vertex.index()))
            .map(|vertex| {
                let (fluid, solid) = self.layout.interface_actions(&reactions, vertex.index())?;
                Ok(FixedReferenceFsiInterfaceAction {
                    vertex,
                    fluid: fluid.map(|value| dimensionless_to_action * value),
                    solid: solid.map(|value| dimensionless_to_action * value),
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let interface_action_imbalance_norm = interface_actions
            .iter()
            .flat_map(|action| action.imbalance())
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let interface_tolerance = residual_target * dimensionless_to_action
            + 8192.0
                * f64::EPSILON
                * self.config.scale().action()
                * (1.0 + D as f64 * interface_actions.len() as f64).sqrt();
        if interface_actions.is_empty()
            || !interface_action_imbalance_norm.is_finite()
            || interface_action_imbalance_norm > interface_tolerance
        {
            return Err(invalid(format!(
                "fixed-reference FSI interface action imbalance {interface_action_imbalance_norm:e} exceeds {interface_tolerance:e}"
            )));
        }

        let energy = energy_balance(EnergyEvaluation {
            mesh: &self.mesh,
            partition: &self.partition,
            previous: &self.previous,
            next_vertex_velocity: &vertex_velocity,
            next_bubbles: &fluid_bubbles,
            next_displacement: &solid_displacement,
            config: self.config,
            quadrature: &self.quadrature,
        })?;
        let energy_tolerance = self.config.time_step()
            * self.config.scale().power()
            * residual_target
            * (1.0 + norm(&algebraic_values))
            + 65_536.0
                * f64::EPSILON
                * self.config.scale().energy()
                * (1.0
                    + (energy.previous_kinetic.abs()
                        + energy.next_kinetic.abs()
                        + energy.previous_elastic.abs()
                        + energy.next_elastic.abs())
                        / self.config.scale().energy());
        if energy.defect.abs() > energy_tolerance {
            return Err(invalid(format!(
                "fixed-reference FSI energy defect {:e} exceeds {:e}",
                energy.defect, energy_tolerance
            )));
        }

        Ok(FixedReferenceFsiSolution {
            vertex_velocity,
            fluid_cell_bubble_velocity: fluid_bubbles,
            fluid_pressure_vertices: self.layout.pressure_vertices().to_vec(),
            fluid_pressure: pressure,
            solid_displacement,
            algebraic_values,
            canonical_system,
            pressure_constant_action_norm: self.pressure_constant_action_norm,
            residual_norm,
            continuity_residual_norm,
            kinematic_residual_norm,
            interface_velocity_jump_norm: 0.0,
            interface_actions,
            interface_action_imbalance_norm,
            energy,
            assembly_report: self.assembly_report,
            solve_report,
        })
    }
}
