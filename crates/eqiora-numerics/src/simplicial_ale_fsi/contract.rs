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
use eqiora_meshing::{FixedTopologyGeometryAction, FixedTopologyGeometryState};
use eqiora_meshing::{MeshTopology, SimplicialMesh};
use eqiora_realization::{NonlinearSolvePlan, Target};
use eqiora_solver::{LinearOperatorProperties, LinearSolver, SolverPlan};

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

/// One accepted or restartable state on exact Field/entity and reference topology inventories.
#[derive(Debug, Clone, PartialEq)]
pub struct AleFsiState<const D: usize> {
    time: f64,
    physical: FixedReferenceFsiState<D>,
    geometry: FixedTopologyGeometryState<D>,
}

impl<const D: usize> AleFsiState<D> {
    /// Derive geometry from the exact admitted driver Field in one complete physical State.
    /// # Errors
    /// Rejects nonfinite time, stale Field/Domain/entity ownership and invalid derived geometry.
    pub fn new(
        time: f64,
        reference_mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        motion: &P1HarmonicMeshMotionAction<D>,
        physical: FixedReferenceFsiState<D>,
    ) -> Result<Self, Diagnostic> {
        if !time.is_finite() || time < 0.0 {
            return Err(invalid("ALE state time must be finite and nonnegative"));
        }
        motion.validate_reference(reference_mesh, partition)?;
        validate_state_fields(reference_mesh, partition, &physical)?;
        let field = motion.policy().solid_displacement();
        let displacement = motion.apply(field, &physical.vector_vertices(field)?)?;
        let geometry = FixedTopologyGeometryState::<D>::new(
            reference_mesh,
            current_coordinates(reference_mesh, &displacement)?,
        )?;
        Ok(Self {
            time,
            physical,
            geometry,
        })
    }
    /// Model time in coherent seconds.
    pub const fn time(&self) -> f64 {
        self.time
    }
    /// Every physical Field with exact Domain/entity/component ownership.
    pub const fn physical_state(&self) -> &FixedReferenceFsiState<D> {
        &self.physical
    }
    /// Current geometry derived from the sole exact driver Field.
    pub const fn geometry(&self) -> &FixedTopologyGeometryState<D> {
        &self.geometry
    }
    /// Reapply the exact harmonic driver and reauthenticate stored derived geometry.
    pub fn validate_against(
        &self,
        reference_mesh: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        motion: &P1HarmonicMeshMotionAction<D>,
    ) -> Result<(), Diagnostic> {
        let replayed = Self::new(
            self.time,
            reference_mesh,
            partition,
            motion,
            self.physical.clone(),
        )?;
        if replayed.geometry != self.geometry {
            return Err(invalid("ALE geometry differs from exact driver replay"));
        }
        self.geometry.reconstruct_mesh(reference_mesh)?;
        Ok(())
    }
}

/// Complete bounded policy for one monolithic backward-Euler ALE FSI step.
///
/// Material, scale, and load retain their existing physical meaning. The
/// common [`NonlinearSolvePlan`] and [`SolverPlan`] remain the sole nonlinear
/// and linear controls; this type only closes their ALE-FSI compatibility and
/// serial reference placement.
#[derive(Debug, Clone, PartialEq)]
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
    pub const fn time_step(&self) -> f64 {
        self.fixed_reference.time_step()
    }

    /// Stable Newtonian-fluid and linear-solid material data.
    #[must_use]
    pub fn material(&self) -> &FixedReferenceFsiMaterial<D> {
        self.fixed_reference.material()
    }

    /// Characteristic acceptance scales.
    #[must_use]
    pub const fn scale(&self) -> FixedReferenceFsiScale<D> {
        self.fixed_reference.scale()
    }

    /// Explicit bounded load policy.
    #[must_use]
    pub const fn load(&self) -> FixedReferenceFsiLoad {
        self.fixed_reference.load()
    }

    /// Common nonlinear convergence and globalization policy.
    #[must_use]
    pub const fn nonlinear(&self) -> NonlinearSolvePlan {
        self.nonlinear
    }

    /// Common linear plan used for every general Newton action.
    #[must_use]
    pub const fn linear_solver(&self) -> SolverPlan {
        self.linear_solver
    }

    /// Mathematical class of every admitted Newton action.
    #[must_use]
    pub const fn operator_properties(&self) -> LinearOperatorProperties {
        LinearOperatorProperties::General
    }

    /// Exact one-worker host placement of the bounded reference slice.
    #[must_use]
    pub const fn target(&self) -> Target {
        self.target
    }

    /// Unchanged material/scale/load bridge for reference-solid assembly.
    pub(crate) const fn fixed_reference_config(&self) -> &FixedReferenceFsiStepConfig<D> {
        &self.fixed_reference
    }

    /// Revalidate two accepted states and derive their sole geometry action.
    ///
    /// The current time must be the exact result of adding this plan's duration
    /// to the previous time. Mesh velocity and all GCL coefficients then come
    /// only from the returned consecutive geometry action.
    pub(crate) fn geometry_action(
        &self,
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
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    physical: &FixedReferenceFsiState<D>,
) -> Result<(), Diagnostic> {
    if !matches!(D, 2 | 3)
        || reference.topological_dimension() != D
        || physical
            .fields
            .values()
            .map(|field| field.domain)
            .collect::<std::collections::BTreeSet<_>>()
            != partition.domains().map(|domain| domain.erase()).collect()
    {
        return Err(invalid(
            "ALE physical State differs from complete intrinsic Domain inventory",
        ));
    }
    for (&id, field) in &physical.fields {
        for (key, value) in &field.coefficients {
            if key.field != id
                || !value.is_finite()
                || reference
                    .entity_count(key.entity.dimension())
                    .is_none_or(|count| key.entity.index() >= count)
            {
                return Err(invalid(
                    "ALE physical State has stale exact coordinates or nonfinite values",
                ));
            }
        }
    }
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
