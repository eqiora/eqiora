use super::*;

pub(in crate::numerical_admission) struct PreparedCommonTransientExecution<'a> {
    pub(super) plan: &'a CommonTransientFlowPlan,
    pub(super) backend: super::native::ProfileCheckedBackend<'a>,
    pub(super) prepared_linear: Option<Box<dyn eqiora_solver::PreparedLinearSolver>>,
    pub(super) method: PreparedCommonTransientMethod<'a>,
}

pub(super) enum PreparedCommonTransientMethod<'a> {
    MiniP1(Box<PreparedResolvedTransientMiniRun2d<'a>>),
    GeometryMiniP1(Box<PreparedResolvedTransientGeometryMiniRun2d<'a>>),
    CellCentered(Box<PreparedResolvedTransientCellCenteredRun2d<'a>>),
}

impl PreparedCommonTransientExecution<'_> {
    pub(in crate::numerical_admission) fn advance(
        &mut self,
        state: &CommonState,
        next_time: f64,
    ) -> Result<CommonState, Diagnostic> {
        let run = TransientNavierStokesRun2d::new(NonZeroStepCount::new(NonZeroUsize::MIN));
        match &self.method {
            PreparedCommonTransientMethod::MiniP1(prepared) => {
                let CommonStateKind::MiniP1(initial) = &state.kind else {
                    return Err(invalid(
                        "prepared MINI Run received a non-MINI common State",
                    ));
                };
                let trajectory = prepared.advance_with_linear(
                    initial.as_ref().clone(),
                    run,
                    &self.backend,
                    self.prepared_linear.as_deref_mut(),
                )?;
                let accepted = trajectory
                    .states()
                    .last()
                    .ok_or_else(|| invalid("MINI transient step returned no accepted State"))?;
                let NativeMeshResources::AffineTriangleSimplicial { mesh, .. } =
                    self.plan.admission.resources()
                else {
                    unreachable!("prepared MINI Run owns affine-triangle resources")
                };
                self.accept_mini(mesh, state, next_time, accepted)
            }
            PreparedCommonTransientMethod::GeometryMiniP1(prepared) => {
                let CommonStateKind::MiniP1(initial) = &state.kind else {
                    return Err(invalid(
                        "prepared MINI Run received a non-MINI common State",
                    ));
                };
                let states = prepared.advance_with_linear(
                    initial.as_ref().clone(),
                    run,
                    &self.backend,
                    self.prepared_linear.as_deref_mut(),
                )?;
                let accepted = states.last().ok_or_else(|| {
                    invalid("Geometry MINI transient step returned no accepted State")
                })?;
                let NativeMeshResources::GmshSimplicial { mesh, .. } =
                    self.plan.admission.resources()
                else {
                    unreachable!("prepared Geometry MINI Run owns Gmsh resources")
                };
                self.accept_mini(mesh, state, next_time, accepted)
            }
            PreparedCommonTransientMethod::CellCentered(prepared) => {
                let CommonStateKind::CellCentered(initial) = &state.kind else {
                    return Err(invalid(
                        "prepared cell-centered Run received incompatible method history",
                    ));
                };
                let trajectory = prepared.advance(initial.as_ref().clone(), run, &self.backend)?;
                let accepted = trajectory.states().last().ok_or_else(|| {
                    invalid("cell-centered transient step returned no accepted State")
                })?;
                self.accept(
                    state,
                    next_time,
                    CommonStateKind::CellCentered(Box::new(
                        prepared.initial_from_accepted(accepted)?,
                    )),
                    Vec::new(),
                )
            }
        }
    }

    fn accept_mini(
        &self,
        mesh: &SimplicialMeshEnvelopeV1,
        previous: &CommonState,
        next_time: f64,
        accepted: &ResolvedTransientNavierStokesState2d,
    ) -> Result<CommonState, Diagnostic> {
        self.accept(
            previous,
            next_time,
            CommonStateKind::MiniP1(Box::new(mini_initial_from_resolved(
                self.plan, mesh, accepted,
            )?)),
            accepted.named_boundary_forces_on_domain().to_vec(),
        )
    }

    fn accept(
        &self,
        previous: &CommonState,
        next_time: f64,
        kind: CommonStateKind,
        named_boundary_forces_on_domain: Vec<(String, [f64; 2])>,
    ) -> Result<CommonState, Diagnostic> {
        CommonState::new_with_boundary_forces(
            self.plan.state_space_identity(),
            next_time,
            Arc::clone(&previous.model),
            Arc::clone(&previous.resources),
            kind,
            named_boundary_forces_on_domain,
        )
    }
}
