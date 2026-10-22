use super::solver_planning::resolve_linear;
use super::spatial_planning::{
    TransientSpatialDecision, resolve_scalar, resolve_stokes, resolve_transient,
};
use super::*;

impl ResolvedCommonPlan {
    #[allow(
        clippy::too_many_arguments,
        reason = "the opaque compiler projection is a separate Formulation input, not numerical policy"
    )]
    pub fn resolve(
        model: &ModelEnvelope,
        owner: AuthenticatedCommonMesh,
        method: impl Into<CommonMethodRequest>,
        solve: CommonSolvePolicy,
        scaling: Option<CommonScalingRequest>,
        temporal: Option<CommonTemporalPolicy>,
        linear_backend: &dyn LinearSolverBackend,
        time_backend: Option<eqiora_time::TimeBackendCapabilities>,
        authored_formulation: Option<&AuthoredFormulationProjection>,
    ) -> Result<ResolvedCommonPlan, Diagnostic> {
        let method = method.into();
        if let Some(form) =
            authored_formulation.filter(|form| form.harmonic_angular_frequency().is_some())
        {
            if !matches!(
                method,
                CommonMethodRequest::Uniform(CommonSpatialPolicy::Q1)
            ) || !matches!(solve, CommonSolvePolicy::Linear(_))
                || scaling.is_some()
                || temporal.is_some()
                || time_backend.is_some()
            {
                return Err(invalid(
                    "harmonic spatial response requires Q1 and a linear solve without time stepping or flow scaling",
                ));
            }
            let geometry = owner.resources.geometry()?;
            let program = replay_program(model, geometry, &owner.model_geometries)?;
            let reduction = crate::form_compiler::harmonic::HarmonicReduction::derive(
                &program,
                form,
                Some(geometry),
            )?;
            let inner = Self::resolve(
                &reduction.reduced,
                owner,
                method,
                solve,
                None,
                None,
                linear_backend,
                None,
                None,
            )?;
            let Self::Spatial(plan) = inner else {
                return Err(invalid(
                    "harmonic spatial reduction requires the shared complex scalar path",
                ));
            };
            return plan
                .with_harmonic(reduction, form.clone())
                .map(|plan| Self::Spatial(Box::new(plan)));
        }
        let (spatial, formulation) = method.split();
        let recognized = RecognizedNativeAdmission::recognize(
            model,
            owner,
            &spatial,
            temporal.as_ref().map(|_| 0.),
        )?;
        if let Some(projection) = authored_formulation {
            crate::form_compiler::check_authored_dependence(projection, &recognized.program)?;
        }
        let (scaling, system_scaling) = match scaling {
            None => (None, eqiora_realization::SystemScaling::Dimensional),
            Some(CommonScalingRequest::Incompressible(request)) => (
                Some(request),
                eqiora_realization::SystemScaling::Dimensional,
            ),
            Some(CommonScalingRequest::SymmetricCongruence(request)) => {
                if !matches!(&recognized.recognized, RecognizedNativeModel::Spatial(_)) {
                    return Err(invalid(
                        "symmetric congruence scaling requires a real linear finite-element system",
                    ));
                }
                (
                    None,
                    eqiora_realization::SystemScaling::SymmetricCongruence(request),
                )
            }
        };
        match &recognized.recognized {
            RecognizedNativeModel::Coordinates(projection) => {
                if spatial != CommonSpatialRequest::Uniform(CommonSpatialPolicy::CellCentered)
                    || formulation.is_some()
                    || authored_formulation.is_some()
                {
                    return Err(invalid(
                        "coordinate conservation requires automatic CellCentered realization without an authored Formulation",
                    ));
                }
                let structure =
                    eqiora_solver::AlgebraicStructure::new([projection.primary_field()], [])?;
                let (linear, temporal) = resolve_linear_requirements(
                    solve,
                    scaling,
                    temporal,
                    false,
                    "coordinate cell-integrated equality",
                    LinearOperatorProperties::SymmetricPositiveDefinite,
                    Some(structure),
                    linear_backend,
                )?;
                let admission = recognized.complete(
                    NativeSpatialPolicy::CoordinateCellConstant,
                    linear,
                    temporal,
                    time_backend,
                    None,
                    eqiora_realization::SystemScaling::Dimensional,
                )?;
                CommonSpatialPlan::from_coordinate_admission(model, admission)
                    .map(|plan| ResolvedCommonPlan::Spatial(Box::new(plan)))
            }

            RecognizedNativeModel::Spatial(equations) => {
                let mut spatial = resolve_scalar(
                    spatial,
                    matches!(
                        recognized.resources,
                        NativeMeshResources::GmshSimplicial { .. }
                            | NativeMeshResources::AffineTriangleSimplicial { .. }
                    ),
                )?;
                if matches!(spatial, NativeSpatialPolicy::ScalarTpfa(_)) {
                    let constraint = authored_formulation
                        .map(|form| {
                            scalar::interval::admit_gauge(&recognized.program, equations, form)
                        })
                        .transpose()?
                        .flatten();
                    spatial = NativeSpatialPolicy::ScalarTpfa(constraint);
                }
                let formulation_selection = match spatial {
                    NativeSpatialPolicy::FiniteElement(_) => Some(resolve_formulation_request(
                        formulation,
                        equations.finite_element_formulation(),
                        "linear finite-element form",
                    )?),
                    NativeSpatialPolicy::ScalarTpfa(_) => {
                        if authored_formulation.is_some_and(|form| form.interval().is_none()) {
                            return Err(invalid(
                                "TPFA requires an integral-conservative authored form",
                            ));
                        }
                        Some(resolve_formulation_request(
                            formulation,
                            FormulationKind::IntegralConservative,
                            "scalar-elliptic TPFA",
                        )?)
                    }
                    _ => unreachable!("scalar resolution returns only scalar spatial policies"),
                };
                let structure = equations.algebraic_structure(spatial.scalar_constraint())?;
                let nonlinear_equations = equations
                    .regions
                    .iter()
                    .any(|region| region.form.is_nonlinear());
                let (linear_request, nonlinear) =
                    resolve_spatial_strategy(solve, &spatial, nonlinear_equations)?;
                let properties = recognized.operator_properties(&spatial, nonlinear)?;
                let (linear, temporal) = resolve_linear_requirements(
                    CommonSolvePolicy::Linear(linear_request),
                    scaling,
                    temporal,
                    equations
                        .regions
                        .iter()
                        .any(|region| region.form.is_transient()),
                    "linear conservation form",
                    properties,
                    Some(structure),
                    linear_backend,
                )?;
                let admission = recognized.complete(
                    spatial,
                    linear,
                    temporal,
                    time_backend,
                    nonlinear,
                    system_scaling,
                )?;
                CommonSpatialPlan::from_admission(
                    model,
                    admission,
                    formulation_selection,
                    authored_formulation,
                )
                .map(|plan| ResolvedCommonPlan::Spatial(Box::new(plan)))
            }
            RecognizedNativeModel::ComplexSpatial(equations) => {
                let spatial = resolve_scalar(
                    spatial,
                    matches!(
                        recognized.resources,
                        NativeMeshResources::GmshSimplicial { .. }
                            | NativeMeshResources::AffineTriangleSimplicial { .. }
                    ),
                )?;
                if scaling.is_some() {
                    return Err(invalid(
                        "complex spatial Plan does not admit incompressible scaling",
                    ));
                }
                let selection = resolve_formulation_request(
                    formulation,
                    if matches!(spatial, NativeSpatialPolicy::ScalarTpfa(_)) {
                        FormulationKind::IntegralConservative
                    } else {
                        equations.finite_element_formulation()
                    },
                    "complex spatial form",
                )?;
                let nonlinear_equations = equations
                    .regions
                    .iter()
                    .any(|region| region.form.is_nonlinear());
                let (request, nonlinear) =
                    resolve_spatial_strategy(solve, &spatial, nonlinear_equations)?;
                let temporal = resolve_temporal_requirement(
                    equations
                        .regions
                        .iter()
                        .any(|region| region.form.is_transient()),
                    temporal,
                    "complex spatial form",
                )?;
                let properties = recognized.operator_properties(&spatial, nonlinear)?;
                let structure = equations.algebraic_structure(None)?;
                let domain = recognized.numerical_scalar_domain(
                    &spatial,
                    &system_scaling,
                    nonlinear,
                    temporal.as_ref(),
                )?;
                let mut linear = if domain == eqiora_core::ScalarDomain::Complex {
                    super::solver_planning::resolve_complex_linear(
                        request,
                        properties,
                        linear_backend,
                    )?
                } else {
                    super::solver_planning::resolve_linear(
                        request,
                        properties,
                        None,
                        None,
                        Some(structure.clone()),
                        linear_backend,
                    )?
                };
                let profile =
                    eqiora_solver::HostSerialSolverProfile::canonical_csr(properties, None, None)
                        .with_structure(structure)?;
                profile.require_plan(linear.solver)?;
                linear.planning_profile = Some(profile);
                let admission = recognized.complete(
                    spatial,
                    linear,
                    temporal,
                    time_backend,
                    nonlinear,
                    system_scaling,
                )?;
                CommonSpatialPlan::from_complex_admission(
                    model,
                    admission,
                    selection,
                    authored_formulation,
                )
                .map(|plan| ResolvedCommonPlan::Spatial(Box::new(plan)))
            }
            RecognizedNativeModel::Stokes(binding) => {
                reject_authored_scalar_form(
                    authored_formulation,
                    "steady incompressible mixed form",
                )?;
                let formulation_selection = resolve_formulation_request(
                    formulation,
                    FormulationKind::MixedGalerkin,
                    "steady-Stokes MINI/P1",
                )?;
                let CommonSolvePolicy::Linear(solve) = solve else {
                    return Err(invalid(
                        "steady-Stokes mathematics requires Linear solve policy",
                    ));
                };
                if temporal.is_some() {
                    return Err(invalid(
                        "steady-Stokes mathematics does not admit a temporal policy",
                    ));
                }
                let spatial = resolve_stokes(spatial)?;
                let scaling = binding.resolve_incompressible_scaling(model, scaling)?;
                let linear = resolve_linear(
                    solve,
                    LinearOperatorProperties::SymmetricIndefinite,
                    None,
                    None,
                    Some(binding.algebraic_structure()?),
                    linear_backend,
                )?;
                let admission = recognized.complete(
                    spatial.with_scaling(scaling.scales()),
                    linear,
                    None,
                    time_backend,
                    None,
                    eqiora_realization::SystemScaling::Dimensional,
                )?;
                CommonSteadyStokesPlan::from_admission(
                    model,
                    admission,
                    formulation_selection,
                    scaling,
                )
                .map(|plan| ResolvedCommonPlan::SteadyStokes(Box::new(plan)))
            }
            RecognizedNativeModel::Transient(_) | RecognizedNativeModel::TransientGeometry(_) => {
                reject_authored_scalar_form(authored_formulation, "transient storage form")?;
                let spatial = resolve_transient(spatial)?;
                let effective_formulation = match spatial {
                    TransientSpatialDecision::MiniP1 => FormulationKind::MixedGalerkin,
                    TransientSpatialDecision::CellCentered => FormulationKind::IntegralConservative,
                };
                let formulation_selection = resolve_formulation_request(
                    formulation,
                    effective_formulation,
                    "transient incompressible-flow spatial policy",
                )?;
                let CommonSolvePolicy::Newton { nonlinear, linear } = solve else {
                    return Err(invalid(
                        "transient incompressible-flow mathematics requires Newton(linear=...) policy",
                    ));
                };
                let temporal = temporal
                    .ok_or_else(|| {
                        invalid("transient incompressible-flow mathematics requires BackwardEuler")
                    })?
                    .require_backward_euler()?;
                let (geometry, mesh, correspondence, _) =
                    resource_artifact_digests(&recognized.resources)?;
                let scaling = resolve_complete_manual_incompressible_scaling_2d(
                    scaling,
                    model.digest()?,
                    geometry,
                    correspondence,
                    mesh,
                )?;
                let linear = resolve_linear(
                    linear,
                    LinearOperatorProperties::General,
                    None,
                    None,
                    Some(transient_algebraic_structure(
                        &recognized.recognized,
                        spatial,
                    )?),
                    linear_backend,
                )?;
                let native_spatial = spatial.with_scaling(scaling.scales());
                let admission = recognized.complete(
                    native_spatial,
                    linear,
                    Some(temporal.into()),
                    time_backend,
                    Some(nonlinear),
                    eqiora_realization::SystemScaling::Dimensional,
                )?;
                CommonTransientFlowPlan::from_admission(
                    model,
                    admission,
                    formulation_selection,
                    scaling,
                    temporal,
                    nonlinear,
                )
                .map(|plan| ResolvedCommonPlan::TransientFlow(Box::new(plan)))
            }
        }
    }
}

fn resolve_spatial_strategy(
    solve: CommonSolvePolicy,
    spatial: &NativeSpatialPolicy,
    nonlinear_equations: bool,
) -> Result<(CommonLinearRequest, Option<NonlinearSolvePlan>), Diagnostic> {
    Ok(match solve {
        CommonSolvePolicy::Linear(linear) if !nonlinear_equations => (linear, None),
        CommonSolvePolicy::Linear(_) => {
            return Err(invalid(
                "nonlinear Region equations require an explicit Newton policy",
            ));
        }
        CommonSolvePolicy::Newton { linear, nonlinear } => {
            if !matches!(
                spatial,
                NativeSpatialPolicy::FiniteElement(_) | NativeSpatialPolicy::ScalarTpfa(_)
            ) {
                return Err(invalid(
                    "Region Newton requires a mapped spatial discretization",
                ));
            }
            (linear, Some(nonlinear))
        }
    })
}

#[allow(
    clippy::too_many_arguments,
    reason = "each argument is an existing exact mathematical or numerical requirement"
)]
fn resolve_linear_requirements(
    solve: CommonSolvePolicy,
    scaling: Option<IncompressibleScalingRequest2d>,
    temporal: Option<CommonTemporalPolicy>,
    has_storage: bool,
    mathematical_form: &str,
    properties: LinearOperatorProperties,
    structure: Option<eqiora_solver::AlgebraicStructure>,
    backend: &dyn LinearSolverBackend,
) -> Result<(NativeLinearPolicy, Option<CommonTemporalPolicy>), Diagnostic> {
    let CommonSolvePolicy::Linear(solve) = solve else {
        return Err(invalid(format!(
            "{mathematical_form} requires Linear solve policy"
        )));
    };
    let temporal = resolve_temporal_requirement(has_storage, temporal, mathematical_form)?;
    if scaling.is_some() {
        return Err(invalid(format!(
            "{mathematical_form} does not admit incompressible-flow scaling"
        )));
    }
    resolve_linear(solve, properties, None, None, structure, backend)
        .map(|linear| (linear, temporal))
}

fn resolve_temporal_requirement(
    has_storage: bool,
    temporal: Option<CommonTemporalPolicy>,
    mathematical_form: &str,
) -> Result<Option<CommonTemporalPolicy>, Diagnostic> {
    Ok(match (has_storage, temporal) {
        (false, None) => None,
        (false, Some(_)) => {
            return Err(invalid(format!(
                "steady {mathematical_form} does not admit a temporal policy"
            )));
        }
        (true, Some(temporal)) => Some(temporal),
        (true, None) => {
            return Err(invalid(format!(
                "{mathematical_form} with storage requires an explicit temporal policy"
            )));
        }
    })
}

fn reject_authored_scalar_form(
    authored: Option<&AuthoredFormulationProjection>,
    mathematical_form: &str,
) -> Result<(), Diagnostic> {
    if authored.is_some() {
        return Err(invalid(format!(
            "authored scalar Formulation does not match the admitted {mathematical_form}"
        )));
    }
    Ok(())
}

fn transient_algebraic_structure(
    recognized: &RecognizedNativeModel,
    spatial: TransientSpatialDecision,
) -> Result<eqiora_solver::AlgebraicStructure, Diagnostic> {
    use crate::canonical_stokes::{
        transient_cell_centered_algebraic_structure, transient_mini_algebraic_structure,
    };
    match (recognized, spatial) {
        (RecognizedNativeModel::Transient(model), TransientSpatialDecision::MiniP1) => {
            transient_mini_algebraic_structure(model)
        }
        (RecognizedNativeModel::TransientGeometry(binding), TransientSpatialDecision::MiniP1) => {
            transient_mini_algebraic_structure(binding.model())
        }
        (RecognizedNativeModel::Transient(model), TransientSpatialDecision::CellCentered) => {
            transient_cell_centered_algebraic_structure(model)
        }
        _ => Err(invalid(
            "transient algebraic structure requires matching Model and spatial meaning",
        )),
    }
}

fn resolve_formulation_request(
    requested: Option<FormulationKind>,
    effective: FormulationKind,
    consumer: &str,
) -> Result<FormulationSelectionMode, Diagnostic> {
    match requested {
        None => Ok(FormulationSelectionMode::Automatic),
        Some(requested) if requested == effective => Ok(FormulationSelectionMode::Exact),
        Some(requested) => Err(invalid(format!(
            "requested {requested:?} formulation is incompatible with {consumer}; the only admitted effective formulation is {effective:?}"
        ))),
    }
}
