use super::*;

pub(crate) fn replay_program(
    model: &ModelEnvelope,
    geometry: &CanonicalGeometryV1,
    additional: &[CanonicalGeometryV1],
) -> Result<KernelProgram, Diagnostic> {
    let reference = model.artifact_reference()?;
    let (transaction, model_id) = model.to_transaction().map_err(first)?;
    let store = InMemoryGraphStore::restore_snapshot(
        transaction,
        eqiora_graph::Revision(reference.semantic_revision().get()),
    )
    .map_err(first)?;
    let snapshot = store.snapshot();
    let program = if model.requires_geometry_admission()? {
        let geometries = std::iter::once(geometry)
            .chain(additional)
            .collect::<Vec<_>>();
        KernelProgram::from_snapshot_with_geometry(&snapshot, model_id, &geometries)
            .map_err(first)?
    } else {
        if !additional.is_empty() {
            return Err(invalid("Model has unreferenced Geometry dependencies"));
        }
        KernelProgram::from_snapshot(&snapshot, model_id).map_err(first)?
    };
    if program.model() != reference.model()
        || program.revision().0 != reference.semantic_revision().get()
    {
        return Err(invalid(
            "replayed Model identity differs from exact caller Model",
        ));
    }
    Ok(program)
}

pub(crate) fn policy_identity(
    spatial: &NativeSpatialPolicy,
    linear: &NativeLinearPolicy,
    temporal: Option<&CommonTemporalPolicy>,
    time_backend: Option<eqiora_time::TimeBackendCapabilities>,
    nonlinear: Option<NonlinearSolvePlan>,
) -> Result<String, Diagnostic> {
    let mut bytes = Vec::new();
    push_framed(
        &mut bytes,
        match linear.scalar_domain {
            eqiora_core::ScalarDomain::Real => b"real",
            eqiora_core::ScalarDomain::Complex => b"complex",
            _ => unreachable!("native linear policies use floating-point scalars"),
        },
    );
    match spatial {
        NativeSpatialPolicy::CoordinateCellConstant => {
            bytes.extend_from_slice(b"coordinate-cell-average-gauss2")
        }
        NativeSpatialPolicy::FiniteElement(request) => match request {
            CommonSpatialRequest::Uniform(policy) => {
                bytes.extend_from_slice(spatial_policy_tag(*policy))
            }
            CommonSpatialRequest::Scoped(bindings) => {
                bytes.extend_from_slice(b"domain-scoped-linear-finite-element-v1");
                bytes.extend_from_slice(&(bindings.len() as u64).to_be_bytes());
                for binding in bindings {
                    push_framed(&mut bytes, binding.model().to_string().as_bytes());
                    bytes.extend_from_slice(&binding.domain().ulid().to_bytes());
                    push_framed(&mut bytes, spatial_policy_tag(binding.policy()));
                }
            }
        },
        NativeSpatialPolicy::ScalarTpfa(constraint) => {
            bytes.extend_from_slice(b"scalar-tpfa");
            if let Some(constraint) = constraint {
                bytes.extend_from_slice(b"zero-integral");
                bytes.extend_from_slice(&constraint.field().ulid().to_bytes());
            }
        }
        NativeSpatialPolicy::StokesMiniP1(scales) => {
            bytes.extend_from_slice(b"stokes-mini-p1");
            bytes.extend_from_slice(&scales.length().value().to_bits().to_be_bytes());
            bytes.extend_from_slice(&scales.velocity().value().to_bits().to_be_bytes());
            bytes.extend_from_slice(&scales.pressure().value().to_bits().to_be_bytes());
        }
        NativeSpatialPolicy::TransientMiniP1(scales)
        | NativeSpatialPolicy::TransientCellCentered(scales) => {
            bytes.extend_from_slice(match spatial {
                NativeSpatialPolicy::TransientMiniP1(_) => b"transient-mini-p1",
                NativeSpatialPolicy::TransientCellCentered(_) => b"transient-cell-centered",
                _ => unreachable!(),
            });
            bytes.extend_from_slice(&scales.length().value().to_bits().to_be_bytes());
            bytes.extend_from_slice(&scales.velocity().value().to_bits().to_be_bytes());
            bytes.extend_from_slice(&scales.pressure().value().to_bits().to_be_bytes());
        }
    }
    if let Some(temporal) = temporal {
        push_framed(
            &mut bytes,
            &super::super::plan_artifact::temporal_policy_bytes(temporal)?,
        );
    }

    if let Some(backend) = time_backend {
        push_framed(&mut bytes, backend.identity().id().as_bytes());
        push_framed(&mut bytes, backend.identity().version().as_bytes());
    }

    if let Some(nonlinear) = nonlinear {
        bytes.extend_from_slice(&nonlinear.relative_tolerance().to_bits().to_be_bytes());
        bytes.extend_from_slice(&nonlinear.absolute_tolerance().to_bits().to_be_bytes());
        bytes.extend_from_slice(&nonlinear.maximum_iterations().get().to_be_bytes());
        bytes.extend_from_slice(&nonlinear.maximum_line_search_steps().to_be_bytes());
    }
    push_linear_policy_identity(&mut bytes, linear);
    Ok(domain_separated_identity(POLICY_DOMAIN, &bytes))
}

/// Effective solver and planning provenance share one byte owner across Plan families.
pub(in crate::numerical_admission) fn push_linear_policy_identity(
    bytes: &mut Vec<u8>,
    linear: &NativeLinearPolicy,
) {
    push_framed(bytes, linear_solver_identity(linear.solver.algorithm()));
    push_framed(
        bytes,
        preconditioner_identity(linear.solver.preconditioner()),
    );
    push_framed(bytes, reduction_identity(linear.solver.reduction()));
    bytes.extend_from_slice(&linear.solver.relative_tolerance().to_bits().to_be_bytes());
    bytes.extend_from_slice(&linear.solver.absolute_tolerance().to_bits().to_be_bytes());
    bytes.extend_from_slice(&linear.solver.maximum_iterations().get().to_be_bytes());
    if let Some(objective) = linear.planning_objective {
        push_framed(bytes, b"program-controlled");
        push_framed(
            bytes,
            match objective {
                SolverPlanningObjective::Robust => b"robust",
                SolverPlanningObjective::Fast => b"fast",
                SolverPlanningObjective::LowMemory => b"low-memory",
            },
        );
        push_framed(
            bytes,
            linear
                .planning_policy_id
                .expect("program-controlled policy retains its identity")
                .as_bytes(),
        );
        push_framed(
            bytes,
            linear
                .selected_candidate_id
                .expect("program-controlled policy retains its selected candidate")
                .as_bytes(),
        );
    }
    push_framed(bytes, linear.provider.id().as_str().as_bytes());
    push_framed(bytes, linear.provider.implementation_version().as_bytes());
    for library in linear.provider.libraries() {
        push_framed(bytes, library.name().as_bytes());
        push_framed(bytes, library.version().as_bytes());
    }
    push_framed(bytes, linear.execution.id().as_str().as_bytes());
    push_framed(bytes, linear.execution.implementation_version().as_bytes());
    for library in linear.execution.libraries() {
        push_framed(bytes, library.name().as_bytes());
        push_framed(bytes, library.version().as_bytes());
    }
    bytes.extend_from_slice(&linear.workers.get().to_be_bytes());
}

pub(crate) const fn linear_solver_identity(value: LinearSolver) -> &'static [u8] {
    match value {
        LinearSolver::ConjugateGradient => b"conjugate-gradient",
        LinearSolver::MinimumResidual => b"minimum-residual",
        LinearSolver::BiConjugateGradientStabilized => b"bicgstab",
        LinearSolver::SparseLu => b"sparse-lu",
    }
}

pub(crate) const fn preconditioner_identity(value: PreconditionerPolicy) -> &'static [u8] {
    match value {
        PreconditionerPolicy::Identity => b"identity",
        PreconditionerPolicy::Jacobi => b"jacobi",
    }
}

pub(crate) const fn reduction_identity(value: ReductionPolicy) -> &'static [u8] {
    match value {
        ReductionPolicy::Reproducible => b"reproducible",
        ReductionPolicy::Fast => b"fast",
    }
}

pub(crate) fn push_framed(target: &mut Vec<u8>, value: &[u8]) {
    target.extend_from_slice(&value.len().to_be_bytes());
    target.extend_from_slice(value);
}

pub(crate) fn domain_separated_identity(domain: &[u8], payload: &[u8]) -> String {
    hex_bytes(&Sha256::digest([domain, payload].concat()))
}

pub(crate) fn static_plan_identity_lineage(
    admission: &NativeNumericalAdmission,
    realization_digest: &str,
) -> Result<(ResourceDigests, Vec<u8>), Diagnostic> {
    let digests = resource_digests(admission.resources())?;
    let mut bytes = Vec::new();
    push_framed(&mut bytes, admission.model_digest().as_bytes());
    digests.bind(&mut bytes);
    for value in [realization_digest, admission.policy_identity()] {
        push_framed(&mut bytes, value.as_bytes());
    }
    Ok((digests, bytes))
}

pub(crate) fn space_identity(space: Space) -> &'static [u8] {
    match space.family() {
        SpaceFamily::SimplexP1Bubble => b"simplex-p1-bubble",
        SpaceFamily::CellConstant => b"cell-constant",
        SpaceFamily::TetrahedralEdge => b"tetrahedral-edge",
        SpaceFamily::TetrahedralFace => b"tetrahedral-face",
        SpaceFamily::ContinuousLagrange { order } if order.get() == 1 => b"continuous-lagrange-p1",
        SpaceFamily::ContinuousLagrange { .. } => {
            unreachable!("closed common transient resolver only admits continuous P1")
        }
    }
}

pub(crate) fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(crate) fn first(diagnostics: Vec<Diagnostic>) -> Diagnostic {
    diagnostics
        .into_iter()
        .next()
        .unwrap_or_else(|| invalid("Model replay failed without a diagnostic"))
}

pub(crate) fn require_portable_realization(
    owned: &PortableRealizationGraph,
    materialized: PortableRealizationGraph,
) -> Result<(), Diagnostic> {
    if &materialized != owned {
        return Err(invalid(
            "common Plan's execution materialization differs from its portable realization graph",
        ));
    }
    Ok(())
}

pub(crate) fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

fn spatial_policy_tag(policy: CommonSpatialPolicy) -> &'static [u8] {
    match policy {
        CommonSpatialPolicy::Q1 | CommonSpatialPolicy::P1 => b"scalar-continuous-lagrange-1",
        CommonSpatialPolicy::MiniP1 => b"linear-mini-p1-duffy4",
        CommonSpatialPolicy::TetrahedralEdge => b"tetrahedral-edge-moment-duffy3",
        CommonSpatialPolicy::TetrahedralFace => b"tetrahedral-face-moment-duffy3",
        CommonSpatialPolicy::CellCentered => b"linear-cell-constant",
        CommonSpatialPolicy::CellCenteredTpfa => b"scalar-tpfa",
    }
}
