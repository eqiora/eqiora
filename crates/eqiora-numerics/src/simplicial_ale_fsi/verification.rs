//! Explicit implementation verification for monolithic ALE FSI.

use crate::simplicial_fsi::layout::FsiLayout;
use eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND;
use eqiora_core::Diagnostic;
use eqiora_ir::{LinearizedRelation, RelationTangent};
use eqiora_meshing::{QuadratureRule, SimplicialMesh};

use super::assembly::{
    assemble_step_linearization_prepared, assemble_step_residual_prepared,
    build_step_jacobian_pattern,
};
use super::boundary_step::PreparedAleFsiBoundaryStep;
use super::contract::{AleFsiBoundary, AleFsiState, AleFsiStepPlan};
use super::{P1HarmonicMeshMotionAction, invalid};
use crate::jacobian_audit::{CenteredJacobianVerification, audit_centered_jacobian};
use crate::simplicial_fsi::FixedReferenceFsiPartition;

impl<const D: usize> AleFsiStepPlan<D> {
    /// Explicitly verify the analytic Jacobian at one accepted ALE FSI step.
    ///
    /// Returns `(columns, colors, globally_coupled_singletons,
    /// residual_assemblies, maximum_error)`.
    ///
    /// # Errors
    /// Returns a diagnostic when the states do not describe this plan's step,
    /// when assembly fails, or when an analytic column differs from the
    /// centered residual reconstruction.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn verify_accepted_jacobian(
        &self,
        reference: &SimplicialMesh,
        partition: &FixedReferenceFsiPartition<D>,
        boundary: &AleFsiBoundary<D>,
        motion: &P1HarmonicMeshMotionAction<D>,
        previous: &AleFsiState<D>,
        accepted: &AleFsiState<D>,
        quadrature: &QuadratureRule,
        base_layout: &FsiLayout<D>,
    ) -> Result<(usize, usize, usize, usize, f64), Diagnostic> {
        verify_simplicial_ale_fsi_jacobian(
            reference,
            partition,
            boundary,
            motion,
            previous,
            accepted,
            self,
            quadrature,
            base_layout,
        )
        .map(|verification| {
            (
                verification.column_count(),
                verification.color_count(),
                verification.globally_coupled_singleton_count(),
                verification.residual_assembly_count(),
                verification.maximum_error(),
            )
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_simplicial_ale_fsi_jacobian<const D: usize>(
    reference: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<D>,
    boundary: &AleFsiBoundary<D>,
    motion: &P1HarmonicMeshMotionAction<D>,
    previous: &AleFsiState<D>,
    accepted: &AleFsiState<D>,
    plan: &AleFsiStepPlan<D>,
    quadrature: &QuadratureRule,
    base_layout: &FsiLayout<D>,
) -> Result<CenteredJacobianVerification, Diagnostic> {
    let prepared = match PreparedAleFsiBoundaryStep::from_boundary(boundary) {
        Some(prepared) => prepared,
        None => PreparedAleFsiBoundaryStep::homogeneous(
            reference,
            boundary,
            previous.time(),
            previous.time() + plan.time_step(),
            plan.scale().velocity(),
        )?,
    };
    prepared.validate_inputs(reference, partition, motion, previous, plan, quadrature)?;
    accepted.validate_against(reference, partition, motion)?;
    let layout = prepared.layout(base_layout)?;
    let point = prepared.reduce_current_point(accepted, plan, &layout)?;
    let assembled = assemble_step_linearization_prepared(
        reference,
        partition,
        &prepared,
        motion,
        previous,
        &point,
        plan,
        quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        base_layout,
    )?;
    if assembled.current_state() != accepted {
        return Err(invalid(
            "explicit ALE FSI verification did not reconstruct the supplied accepted state",
        ));
    }
    let pattern = build_step_jacobian_pattern(reference, partition, boundary, motion, base_layout)?;
    audit_centered_jacobian(
        &point,
        &pattern,
        2.0e-5,
        "ALE FSI",
        |candidate| {
            assemble_step_residual_prepared::<D>(
                reference,
                partition,
                &prepared,
                motion,
                previous,
                candidate,
                plan,
                quadrature,
                base_layout,
            )
        },
        |column, analytic| {
            let mut direction = vec![0.0; point.len()];
            direction[column] = 1.0;
            assembled
                .relation
                .jvp(RelationTangent::Unknown(&direction), analytic)
        },
    )
}
