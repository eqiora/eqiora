//! Access to one completed ALE FSI assembly result.

use std::sync::Arc;

use crate::region_assembly::{
    InterfaceReactionProjection, RecoveredDomainReactions, RecoveredInterfaceReactions,
};
use eqiora_assembly::AssemblyReport;
use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::FixedTopologyGeometryAction;

use super::{AleFsiState, FsiLayout, finite_norm};
use crate::assembled_linearization::AssembledLinearizedRelation;

/// One assembled Newton point and independently evaluated exact Domain residuals.
pub(in crate::simplicial_ale_fsi) struct StepAssembly<const D: usize> {
    pub(in crate::simplicial_ale_fsi) relation: AssembledLinearizedRelation,
    pub(in crate::simplicial_ale_fsi) current: AleFsiState<D>,
    pub(in crate::simplicial_ale_fsi) geometry_action: FixedTopologyGeometryAction<D>,
    pub(in crate::simplicial_ale_fsi) residual: Vec<f64>,
    pub(in crate::simplicial_ale_fsi) domain_residuals: RecoveredDomainReactions<f64>,
    pub(super) reactions: Arc<InterfaceReactionProjection>,
    pub(in crate::simplicial_ale_fsi) layout: Arc<FsiLayout<D>>,
    pub(in crate::simplicial_ale_fsi) assembly_report: AssemblyReport,
}

impl<const D: usize> StepAssembly<D> {
    pub(in crate::simplicial_ale_fsi) fn residual_norm(&self) -> Result<f64, Diagnostic> {
        finite_norm(&self.residual, "ALE FSI reduced residual")
    }

    pub(in crate::simplicial_ale_fsi) fn residual(&self) -> &[f64] {
        &self.residual
    }

    pub(in crate::simplicial_ale_fsi) fn algebraic_values(&self) -> &[f64] {
        self.relation.accepted_unknowns()
    }

    pub(in crate::simplicial_ale_fsi) const fn current_state(&self) -> &AleFsiState<D> {
        &self.current
    }

    pub(in crate::simplicial_ale_fsi) const fn geometry_action(
        &self,
    ) -> &FixedTopologyGeometryAction<D> {
        &self.geometry_action
    }

    pub(in crate::simplicial_ale_fsi) fn domain_residual(
        &self,
        domain: RawId,
    ) -> Result<&[f64], Diagnostic> {
        self.domain_residuals
            .values
            .get(&domain)
            .map(Vec::as_slice)
            .ok_or_else(|| super::invalid("residual request has no exact owning Domain"))
    }

    pub(in crate::simplicial_ale_fsi) fn interface_reactions(
        &self,
    ) -> Result<RecoveredInterfaceReactions<f64>, Diagnostic> {
        self.reactions.recover(&self.domain_residuals)
    }

    pub(in crate::simplicial_ale_fsi) const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }
}
