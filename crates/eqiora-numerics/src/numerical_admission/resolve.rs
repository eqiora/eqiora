use super::solver_planning::resolve_linear;
use super::spatial_planning::{
    TransientSpatialDecision, require_fixed_reference_fsi, resolve_elasticity, resolve_scalar,
    resolve_stokes, resolve_transient,
};
use super::*;

#[allow(
    clippy::too_many_arguments,
    reason = "the opaque compiler projection is a separate Formulation input, not numerical policy"
)]
pub fn resolve_common_plan(
    model: &ModelEnvelope,
    owner: AuthenticatedCommonMesh,
    method: impl Into<CommonMethodRequest>,
    solve: CommonSolvePolicy,
    scaling: Option<IncompressibleScalingRequest2d>,
    temporal: Option<CommonBackwardEuler>,
    stokes_backend: &dyn LinearSolverBackend,
    authored_formulation: Option<&AuthoredFormulationProjection>,
) -> Result<ResolvedCommonPlan, Diagnostic> {
    let recognized = RecognizedNativeAdmission::recognize(model, owner)?;
    let (spatial, formulation) = method.into().split();
    match &recognized.recognized {
        RecognizedNativeModel::Scalar(equations) => {
            let spatial = resolve_scalar(spatial)?;
            let (formulation_selection, properties) = match spatial {
                NativeSpatialPolicy::ScalarQ1 => (
                    Some(resolve_formulation_request(
                        formulation,
                        FormulationKind::PrimalGalerkin,
                        "scalar-elliptic Q1",
                    )?),
                    LinearOperatorProperties::General,
                ),
                NativeSpatialPolicy::ScalarTpfa => {
                    if authored_formulation.is_some_and(|form| form.interval().is_none()) {
                        return Err(invalid(
                            "TPFA requires an integral-conservative authored form",
                        ));
                    }
                    (
                        Some(resolve_formulation_request(
                            formulation,
                            FormulationKind::IntegralConservative,
                            "scalar-elliptic TPFA",
                        )?),
                        LinearOperatorProperties::SymmetricPositiveDefinite,
                    )
                }
                _ => unreachable!("scalar resolution returns only scalar spatial policies"),
            };
            let structure = equations.algebraic_structure()?;
            let (linear, temporal) = resolve_linear_requirements(
                solve,
                scaling,
                temporal,
                false,
                "scalar conservation form",
                properties,
                Some(structure),
                stokes_backend,
            )?;
            let admission = recognized.complete(spatial, linear, temporal, None)?;
            CommonScalarPlan::from_admission(
                model,
                admission,
                formulation_selection,
                authored_formulation,
            )
            .map(|plan| ResolvedCommonPlan::Scalar(Box::new(plan)))
        }
        RecognizedNativeModel::Elasticity(continuum) => {
            if authored_formulation.is_some() {
                return Err(invalid(
                    "authored scalar Formulation does not match the vector small-strain form",
                ));
            }
            reject_unsupported_formulation_request(formulation, "isotropic small-strain form")?;
            let spatial = resolve_elasticity(spatial)?;
            let structure = super::elasticity::algebraic_structure(continuum)?;
            let (linear, temporal) = resolve_linear_requirements(
                solve,
                scaling,
                temporal,
                false,
                "isotropic small-strain form",
                LinearOperatorProperties::SymmetricPositiveDefinite,
                Some(structure),
                stokes_backend,
            )?;
            let admission = recognized.complete(spatial, linear, temporal, None)?;
            CommonElasticityPlan::from_admission(model, admission)
                .map(|plan| ResolvedCommonPlan::Elasticity(Box::new(plan)))
        }
        RecognizedNativeModel::Stokes(binding) => {
            reject_authored_scalar_form(authored_formulation, "steady incompressible mixed form")?;
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
                stokes_backend,
            )?;
            let admission =
                recognized.complete(spatial.with_scaling(scaling.scales()), linear, None, None)?;
            CommonSteadyStokesPlan::from_admission(model, admission, formulation_selection, scaling)
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
            let temporal = temporal.ok_or_else(|| {
                invalid("transient incompressible-flow mathematics requires BackwardEuler")
            })?;
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
                stokes_backend,
            )?;
            let native_spatial = spatial.with_scaling(scaling.scales());
            let admission =
                recognized.complete(native_spatial, linear, Some(temporal), Some(nonlinear))?;
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
        RecognizedNativeModel::Fsi(canonical) => {
            reject_authored_scalar_form(authored_formulation, "coupled interface form")?;
            reject_unsupported_formulation_request(formulation, "fixed-reference FSI")?;
            let CommonSolvePolicy::Linear(linear) = solve else {
                return Err(invalid(
                    "fixed-reference FSI mathematics requires Linear solve policy",
                ));
            };
            let temporal = temporal
                .ok_or_else(|| invalid("fixed-reference FSI mathematics requires BackwardEuler"))?;
            require_fixed_reference_fsi(model, canonical, spatial)?;
            let effective_linear = resolve_linear(
                linear,
                LinearOperatorProperties::SymmetricIndefinite,
                None,
                // The admitted common host FSI execution owns reproducible reductions.
                Some(ReductionPolicy::Reproducible),
                Some(canonical.algebraic_structure()?),
                stokes_backend,
            )?;
            CommonFsiPlan::from_recognized(model, recognized, scaling, temporal, effective_linear)
                .map(|plan| ResolvedCommonPlan::Fsi(Box::new(plan)))
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "each argument is an existing exact mathematical or numerical requirement"
)]
fn resolve_linear_requirements(
    solve: CommonSolvePolicy,
    scaling: Option<IncompressibleScalingRequest2d>,
    temporal: Option<CommonBackwardEuler>,
    has_storage: bool,
    mathematical_form: &str,
    properties: LinearOperatorProperties,
    structure: Option<eqiora_solver::AlgebraicStructure>,
    backend: &dyn LinearSolverBackend,
) -> Result<(NativeLinearPolicy, Option<CommonBackwardEuler>), Diagnostic> {
    let CommonSolvePolicy::Linear(solve) = solve else {
        return Err(invalid(format!(
            "{mathematical_form} requires Linear solve policy"
        )));
    };
    let temporal = match (has_storage, temporal) {
        (false, None) => None,
        (false, Some(_)) => {
            return Err(invalid(format!(
                "steady {mathematical_form} does not admit a temporal policy"
            )));
        }
        (true, Some(temporal)) => Some(temporal),
        (true, None) => {
            return Err(invalid(format!(
                "{mathematical_form} with storage requires an explicit BackwardEuler policy"
            )));
        }
    };
    if scaling.is_some() {
        return Err(invalid(format!(
            "{mathematical_form} does not admit incompressible-flow scaling"
        )));
    }
    resolve_linear(solve, properties, None, None, structure, backend)
        .map(|linear| (linear, temporal))
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
            transient_mini_algebraic_structure(&model.common_projection())
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

fn reject_unsupported_formulation_request(
    requested: Option<FormulationKind>,
    consumer: &str,
) -> Result<(), Diagnostic> {
    if requested.is_some() {
        return Err(invalid(format!(
            "{consumer} does not yet admit an exact formulation request"
        )));
    }
    Ok(())
}
