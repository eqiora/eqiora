//! Accepted state and step-policy contracts for fixed-topology ALE FSI.
//!
//! Coordinates are deliberately absent from the public state constructor.
//! One sealed harmonic-motion action maps absolute solid displacement to the
//! complete vertex displacement, from which the current geometry is rebuilt
//! against immutable reference topology.  The step plan likewise consumes the
//! common nonlinear and linear solver contracts directly; it does not create a
//! method-specific Krylov configuration.

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_meshing::CellId;
use eqiora_meshing::{FixedTopologyGeometryAction, FixedTopologyGeometryState};
use eqiora_meshing::{MeshTopology, SimplicialMesh};
use eqiora_realization::{NonlinearSolvePlan, Target};
use eqiora_solver::{LinearOperatorProperties, LinearSolver, SolverPlan};
use std::collections::BTreeMap;

use super::{P1HarmonicMeshMotionAction, invalid};
use crate::simplicial_fsi::{
    FixedReferenceFsiBoundary, FixedReferenceFsiLoad, FixedReferenceFsiMaterial,
    FixedReferenceFsiPartition, FixedReferenceFsiScale, FixedReferenceFsiState,
    FixedReferenceFsiStepConfig,
};

/// Homogeneous physical-velocity boundary used by the bounded ALE slice.
///
/// Mesh-motion boundary ownership remains sealed in
/// [`P1HarmonicMeshMotionAction`]; this alias describes only the physical velocity
/// closure and therefore reuses the fixed-reference FSI contract exactly.
pub type AleFsiBoundary<const D: usize> = FixedReferenceFsiBoundary<D>;

/// One accepted or restartable state on immutable reference topology.
///
/// Velocity and displacement coefficients use reference-vertex order. MINI
/// bubble velocity is keyed by exact fluid `CellId`, while pressure uses
/// its fluid-vertex order. The stored geometry is a derived value, never
/// independent state.
#[derive(Debug, Clone, PartialEq)]
pub struct AleFsiState<const D: usize> {
    time: f64,
    vertex_velocity: Vec<[f64; D]>,
    fluid_cell_bubble_velocity: BTreeMap<CellId, [f64; D]>,
    fluid_pressure: Vec<f64>,
    solid_displacement: Vec<[f64; D]>,
    geometry: FixedTopologyGeometryState<D>,
}

impl<const D: usize> AleFsiState<D> {
    /// Derive and admit one complete moving-domain state.
    ///
    /// `solid_displacement` is the sole geometry driver. It must use reference
    /// vertex order and be exact zero outside the solid closure. The sealed
    /// harmonic action supplies interface continuity, fixed fluid-exterior
    /// values, and every fluid-interior value before coordinates are formed as
    /// `reference + absolute_displacement`.
    ///
    /// # Errors
    /// Returns `EQ0801` for non-finite/negative time, an incompatible sealed
    /// reference or partition, a non-finite or incorrectly shaped field,
    /// displacement outside the solid closure, or coordinate overflow. Mesh
    /// orientation and quality failures retain their `EQ0803` diagnostic.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        time: f64,
        reference_mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        motion: &P1HarmonicMeshMotionAction<D>,
        vertex_velocity: Vec<[f64; D]>,
        fluid_cell_bubble_velocity: BTreeMap<CellId, [f64; D]>,
        fluid_pressure: Vec<f64>,
        solid_displacement: Vec<[f64; D]>,
    ) -> Result<Self, Diagnostic> {
        if !time.is_finite() || time < 0.0 {
            return Err(invalid(
                "fixed-topology ALE FSI state time must be finite and non-negative",
            ));
        }
        motion.validate_reference(reference_mesh, partition)?;
        validate_state_fields(
            reference_mesh,
            partition,
            &vertex_velocity,
            &fluid_cell_bubble_velocity,
            &fluid_pressure,
            &solid_displacement,
        )?;

        let displacement = motion.apply(&solid_displacement)?;
        let coordinates = current_coordinates(reference_mesh, &displacement)?;
        let geometry = FixedTopologyGeometryState::<D>::new(reference_mesh, coordinates)?;
        let value = Self {
            time,
            vertex_velocity,
            fluid_cell_bubble_velocity,
            fluid_pressure,
            solid_displacement,
            geometry,
        };
        value.validate_against(reference_mesh, partition, motion)?;
        Ok(value)
    }

    /// Model time in coherent seconds.
    #[must_use]
    pub const fn time(&self) -> f64 {
        self.time
    }

    /// Shared fluid/solid P1 velocity in reference-vertex order.
    #[must_use]
    pub fn vertex_velocity(&self) -> &[[f64; D]] {
        &self.vertex_velocity
    }

    /// Fluid MINI bubble velocity keyed by exact owned `CellId`.
    #[must_use]
    pub fn fluid_cell_bubble_velocity(&self) -> &BTreeMap<CellId, [f64; D]> {
        &self.fluid_cell_bubble_velocity
    }

    /// Fluid P1 pressure in canonical fluid-vertex order.
    #[must_use]
    pub fn fluid_pressure(&self) -> &[f64] {
        &self.fluid_pressure
    }

    /// Absolute solid P1 displacement in reference-vertex order.
    ///
    /// Entries outside the solid closure are exact zero.
    #[must_use]
    pub fn solid_displacement(&self) -> &[[f64; D]] {
        &self.solid_displacement
    }

    /// Current coordinates and recomputed quality derived from solid motion.
    #[must_use]
    pub const fn geometry(&self) -> &FixedTopologyGeometryState<D> {
        &self.geometry
    }

    /// Revalidate this state against the exact sealed reference root.
    ///
    /// This is the restart/replay gate. In addition to field shape and support,
    /// it independently reapplies the harmonic action and requires exact
    /// equality with the stored derived geometry.
    ///
    /// # Errors
    /// Returns `EQ0801` if any state field or derived geometry cannot replay
    /// against the supplied immutable reference, partition, and motion action;
    /// mesh reconstruction failures retain their `EQ0803` diagnostic.
    pub fn validate_against(
        &self,
        reference_mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        motion: &P1HarmonicMeshMotionAction<D>,
    ) -> Result<(), Diagnostic> {
        if !self.time.is_finite() || self.time < 0.0 {
            return Err(invalid(
                "fixed-topology ALE FSI state time must be finite and non-negative",
            ));
        }
        motion.validate_reference(reference_mesh, partition)?;
        validate_state_fields(
            reference_mesh,
            partition,
            &self.vertex_velocity,
            &self.fluid_cell_bubble_velocity,
            &self.fluid_pressure,
            &self.solid_displacement,
        )?;
        let displacement = motion.apply(&self.solid_displacement)?;
        let coordinates = current_coordinates(reference_mesh, &displacement)?;
        let replayed = FixedTopologyGeometryState::<D>::new(reference_mesh, coordinates)?;
        if replayed != self.geometry {
            return Err(invalid(
                "fixed-topology ALE FSI geometry must equal reference coordinates plus replayed absolute harmonic motion",
            ));
        }
        self.geometry.reconstruct_mesh(reference_mesh)?;
        Ok(())
    }

    /// Exact bridge to the unchanged reference-layout velocity/displacement state.
    pub(crate) fn to_fixed_reference_state(
        &self,
        reference_mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
    ) -> Result<FixedReferenceFsiState<D>, Diagnostic> {
        FixedReferenceFsiState::<D>::new(
            reference_mesh,
            partition,
            self.vertex_velocity.clone(),
            self.fluid_cell_bubble_velocity.clone(),
            self.solid_displacement.clone(),
        )
    }
}

/// Complete bounded policy for one monolithic backward-Euler ALE FSI step.
///
/// Material, scale, and load retain their existing physical meaning. The
/// common [`NonlinearSolvePlan`] and [`SolverPlan`] remain the sole nonlinear
/// and linear controls; this type only closes their ALE-FSI compatibility and
/// serial reference placement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AleFsiStepPlan<const D: usize> {
    fixed_reference: FixedReferenceFsiStepConfig<D>,
    nonlinear: NonlinearSolvePlan,
    linear_solver: SolverPlan,
    target: Target,
}

impl<const D: usize> AleFsiStepPlan<D> {
    /// Admit the bounded serial-host nonlinear ALE FSI policy.
    ///
    /// # Errors
    /// Returns `EQ0801` for an invalid duration and `EQ0807` unless the load is
    /// the explicit zero-load slice, the common linear plan selects BiCGSTAB
    /// for the general Newton action, and placement is exactly one host worker.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        time_step: f64,
        material: FixedReferenceFsiMaterial<D>,
        scale: FixedReferenceFsiScale<D>,
        load: FixedReferenceFsiLoad,
        nonlinear: NonlinearSolvePlan,
        linear_solver: SolverPlan,
        target: Target,
    ) -> Result<Self, Diagnostic> {
        let fixed_reference =
            FixedReferenceFsiStepConfig::<D>::new(time_step, material, scale, load)?;
        if load != FixedReferenceFsiLoad::Zero {
            return Err(invalid_realization(
                "fixed-topology ALE FSI v1 admits only the explicit zero-load policy",
            ));
        }
        if linear_solver.algorithm() != LinearSolver::BiConjugateGradientStabilized
            || !linear_solver
                .algorithm()
                .accepts(LinearOperatorProperties::General)
        {
            return Err(invalid_realization(
                "fixed-topology ALE FSI Newton actions require the common general-operator BiCGSTAB plan",
            ));
        }
        if target
            != (Target::HostCpu {
                threads: std::num::NonZeroUsize::MIN,
            })
        {
            return Err(invalid_realization(
                "fixed-topology ALE FSI v1 requires the serial HostCpu target",
            ));
        }
        Ok(Self {
            fixed_reference,
            nonlinear,
            linear_solver,
            target,
        })
    }

    /// Backward-Euler step width.
    #[must_use]
    pub const fn time_step(self) -> f64 {
        self.fixed_reference.time_step()
    }

    /// Stable Newtonian-fluid and linear-solid material data.
    #[must_use]
    pub const fn material(self) -> FixedReferenceFsiMaterial<D> {
        self.fixed_reference.material()
    }

    /// Characteristic acceptance scales.
    #[must_use]
    pub const fn scale(self) -> FixedReferenceFsiScale<D> {
        self.fixed_reference.scale()
    }

    /// Explicit bounded load policy.
    #[must_use]
    pub const fn load(self) -> FixedReferenceFsiLoad {
        self.fixed_reference.load()
    }

    /// Common nonlinear convergence and globalization policy.
    #[must_use]
    pub const fn nonlinear(self) -> NonlinearSolvePlan {
        self.nonlinear
    }

    /// Common linear plan used for every general Newton action.
    #[must_use]
    pub const fn linear_solver(self) -> SolverPlan {
        self.linear_solver
    }

    /// Mathematical class of every admitted Newton action.
    #[must_use]
    pub const fn operator_properties(self) -> LinearOperatorProperties {
        LinearOperatorProperties::General
    }

    /// Exact one-worker host placement of the bounded reference slice.
    #[must_use]
    pub const fn target(self) -> Target {
        self.target
    }

    /// Unchanged material/scale/load bridge for reference-solid assembly.
    pub(crate) const fn fixed_reference_config(self) -> FixedReferenceFsiStepConfig<D> {
        self.fixed_reference
    }

    /// Revalidate two accepted states and derive their sole geometry action.
    ///
    /// The current time must be the exact result of adding this plan's duration
    /// to the previous time. Mesh velocity and all GCL coefficients then come
    /// only from the returned consecutive geometry action.
    pub(crate) fn geometry_action(
        self,
        reference_mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        motion: &P1HarmonicMeshMotionAction<D>,
        previous: &AleFsiState<D>,
        current: &AleFsiState<D>,
    ) -> Result<FixedTopologyGeometryAction<D>, Diagnostic> {
        previous.validate_against(reference_mesh, partition, motion)?;
        current.validate_against(reference_mesh, partition, motion)?;
        let expected_time = previous.time + self.time_step();
        if !expected_time.is_finite()
            || expected_time <= previous.time
            || current.time != expected_time
        {
            return Err(invalid(
                "fixed-topology ALE FSI states must advance by the exact plan duration",
            ));
        }
        FixedTopologyGeometryAction::<D>::new(
            reference_mesh,
            previous.geometry(),
            current.geometry(),
            self.time_step(),
        )
    }
}

fn validate_state_fields<const D: usize>(
    reference_mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    vertex_velocity: &[[f64; D]],
    fluid_cell_bubble_velocity: &BTreeMap<CellId, [f64; D]>,
    fluid_pressure: &[f64],
    solid_displacement: &[[f64; D]],
) -> Result<(), Diagnostic> {
    if !matches!(D, 2 | 3)
        || reference_mesh.topological_dimension() != D
        || reference_mesh
            .vertices()
            .iter()
            .any(|coordinates| coordinates.len() != D)
        || fluid_pressure.len() != partition.fluid_vertices().len()
        || fluid_pressure.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(format!(
            "fixed-topology ALE FSI state must own finite pressure in canonical fluid-vertex order on one intrinsic {D}D reference mesh with D equal to 2 or 3"
        )));
    }
    FixedReferenceFsiState::<D>::new(
        reference_mesh,
        partition,
        vertex_velocity.to_vec(),
        fluid_cell_bubble_velocity.clone(),
        solid_displacement.to_vec(),
    )?;
    Ok(())
}

fn current_coordinates<const D: usize>(
    reference_mesh: &SimplicialMesh,
    displacement: &[[f64; D]],
) -> Result<Vec<Vec<f64>>, Diagnostic> {
    if displacement.len() != reference_mesh.vertices().len()
        || reference_mesh
            .vertices()
            .iter()
            .any(|coordinates| coordinates.len() != D)
    {
        return Err(invalid(format!(
            "fixed-topology ALE FSI motion must cover the exact intrinsic-{D}D reference vertex inventory"
        )));
    }
    let coordinates = reference_mesh
        .vertices()
        .iter()
        .zip(displacement)
        .map(|(reference, displacement)| {
            reference
                .iter()
                .zip(displacement)
                .map(|(reference, displacement)| reference + displacement)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if coordinates.iter().flatten().any(|value| !value.is_finite()) {
        return Err(invalid(
            "fixed-topology ALE FSI current-coordinate derivation overflowed",
        ));
    }
    Ok(coordinates)
}

fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

#[cfg(test)]
mod tests;
