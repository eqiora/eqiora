//! Midpoint observations share the ordinary sampler and separate constraint recovery.
use super::*;
use crate::common_trajectory::SpatialTimeStage;
use eqiora_differentiation::PrimalAcceptance;
use eqiora_time::{ImplicitMidpointTimeBackend, TimeHistoryStep};

pub(in crate::numerical_admission) struct MidpointObservation {
    pub(in crate::numerical_admission) time: f64,
    pub(in crate::numerical_admission) physical: Vec<f64>,
    pub(in crate::numerical_admission) recovery: Option<SpatialTimeStage>,
    pub(in crate::numerical_admission) weights: [f64; 2],
}

impl<S: Coefficient + Send + Sync> SpatialImplicitSystem<'_, S>
where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    fn midpoint_observation_fixed(
        &self,
        step: &TimeHistoryStep,
        physical_endpoint: &[f64],
        endpoint: Option<&SpatialTimeStage>,
        time: f64,
    ) -> Result<(Vec<f64>, [f64; 2]), Diagnostic> {
        if endpoint.is_some() == self.algebraic.is_empty() {
            return Err(invalid(
                "midpoint observation has a foreign endpoint inventory",
            ));
        }
        let weights = ImplicitMidpointTimeBackend::sample_step_weights(step, time)?;
        let sampled = ImplicitMidpointTimeBackend::sample_step(step, time)?;
        let fixed = self.lift_differential(&sampled, physical_endpoint)?;
        Ok((fixed, weights))
    }

    /// Recover each actual requested observation independently, including exact endpoints.
    /// An endpoint witness supplies a seed only; prior observations never do.
    pub(in crate::numerical_admission) fn solve_midpoint_observation(
        &self,
        step: &TimeHistoryStep,
        physical_endpoint: &[f64],
        endpoint: Option<&SpatialTimeStage>,
        time: f64,
        backend: &dyn LinearSolverBackend,
    ) -> Result<MidpointObservation, Diagnostic> {
        let (fixed, weights) =
            self.midpoint_observation_fixed(step, physical_endpoint, endpoint, time)?;
        let (physical, recovery) = if let Some(endpoint) = endpoint {
            let h = step.end_time() - step.start_time();
            let (unknown, evidence) =
                self.recover_consistent(time, h, &fixed, &endpoint.unknown, backend)?;
            let (physical, _) = self.recovery_coordinates(h, &fixed, &unknown)?;
            (
                physical,
                Some(SpatialTimeStage::from_newton(unknown, evidence)),
            )
        } else {
            (fixed, None)
        };
        Ok(MidpointObservation {
            time,
            physical,
            recovery,
            weights,
        })
    }

    /// Reaccept the same observation equation and its stopping witness without a solve.
    pub(in crate::numerical_admission) fn accept_midpoint_observation(
        &self,
        request: &CommonTransientRunRequest,
        step: &TimeHistoryStep,
        physical_endpoint: &[f64],
        endpoint: Option<&SpatialTimeStage>,
        time: f64,
        recovery: Option<&SpatialTimeStage>,
    ) -> Result<(MidpointObservation, Option<PrimalAcceptance>), Diagnostic> {
        let (fixed, weights) =
            self.midpoint_observation_fixed(step, physical_endpoint, endpoint, time)?;
        if recovery.is_some() != endpoint.is_some() {
            return Err(invalid(
                "midpoint observation omitted or added a recovery witness",
            ));
        }
        let (physical, acceptance) = if let Some(recovery) = recovery {
            let policy = request
                .plan()
                .ode_temporal()
                .ok_or_else(|| invalid("midpoint observation requires its temporal policy"))?;
            let h = step.end_time() - step.start_time();
            let acceptance = self.validate_newton(
                request,
                recovery,
                &endpoint.expect("checked inventory").unknown,
                &self.correction_tolerances(policy)?,
                |unknown| self.recovery_point(time, h, &fixed, unknown),
            )?;
            (
                self.recovery_coordinates(h, &fixed, &recovery.unknown)?.0,
                Some(acceptance),
            )
        } else {
            (fixed, None)
        };
        Ok((
            MidpointObservation {
                time,
                physical,
                recovery: recovery.cloned(),
                weights,
            },
            acceptance,
        ))
    }
}
