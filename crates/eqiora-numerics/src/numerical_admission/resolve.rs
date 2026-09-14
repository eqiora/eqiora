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
    if authored_formulation.is_some()
        && !matches!(recognized.capability, NativeCapability::ScalarElliptic)
    {
        return Err(invalid(
            "authored scalar-primal Formulation requires scalar Q1 mathematics",
        ));
    }
    let (spatial, formulation) = method.into().split();
    match recognized.capability {
        NativeCapability::ScalarElliptic | NativeCapability::IsotropicElasticity => {
            // Preserve exact-request rejection before policy checks for the current
            // elasticity realization. Both stationary forms then share admission.
            let (consumer, scaling_subject) = match recognized.capability {
                NativeCapability::ScalarElliptic => ("scalar-elliptic", "scalar-elliptic Model"),
                NativeCapability::IsotropicElasticity => {
                    reject_unsupported_formulation_request(formulation, "linear-elasticity")?;
                    ("linear-elasticity", "linear-elasticity")
                }
                _ => unreachable!("stationary scalar or elasticity form"),
            };
            let CommonSolvePolicy::Linear(solve) = solve else {
                return Err(invalid(format!(
                    "{consumer} mathematics requires Linear solve policy"
                )));
            };
            if temporal.is_some() {
                return Err(invalid(format!(
                    "steady {consumer} mathematics does not admit a temporal policy"
                )));
            }
            if scaling.is_some() {
                return Err(invalid(format!(
                    "{scaling_subject} mathematics does not admit incompressible-flow scaling"
                )));
            }
            let spatial = match recognized.capability {
                NativeCapability::ScalarElliptic => resolve_scalar(spatial)?,
                NativeCapability::IsotropicElasticity => resolve_elasticity(spatial)?,
                _ => unreachable!("stationary scalar or elasticity form"),
            };
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
                NativeSpatialPolicy::ElasticityQ1 => {
                    (None, LinearOperatorProperties::SymmetricPositiveDefinite)
                }
                _ => unreachable!("stationary scalar or elasticity spatial policy"),
            };
            let structure = match &recognized.recognized {
                RecognizedNativeModel::Scalar(equations) => Some(equations.algebraic_structure()?),
                RecognizedNativeModel::Elasticity(continuum) => {
                    Some(super::elasticity::algebraic_structure(continuum)?)
                }
                _ => None,
            };
            let linear = resolve_linear(solve, properties, None, None, structure, stokes_backend)?;
            let admission = recognized.complete(spatial, linear, None, None)?;
            match spatial {
                NativeSpatialPolicy::ScalarQ1 | NativeSpatialPolicy::ScalarTpfa => {
                    CommonScalarPlan::from_admission(
                        model,
                        admission,
                        formulation_selection,
                        authored_formulation,
                    )
                    .map(|plan| ResolvedCommonPlan::Scalar(Box::new(plan)))
                }
                NativeSpatialPolicy::ElasticityQ1 => {
                    CommonElasticityPlan::from_admission(model, admission)
                        .map(|plan| ResolvedCommonPlan::Elasticity(Box::new(plan)))
                }
                _ => unreachable!("stationary scalar or elasticity spatial policy"),
            }
        }
        NativeCapability::SteadyIncompressibleStokes => {
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
            let RecognizedNativeModel::Stokes(binding) = &recognized.recognized else {
                unreachable!("steady-Stokes capability recognition returns a Stokes binding")
            };
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
        NativeCapability::TransientIncompressibleFlow => {
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
        NativeCapability::FixedReferenceFsi => {
            reject_unsupported_formulation_request(formulation, "fixed-reference FSI")?;
            let CommonSolvePolicy::Linear(linear) = solve else {
                return Err(invalid(
                    "fixed-reference FSI mathematics requires Linear solve policy",
                ));
            };
            let temporal = temporal
                .ok_or_else(|| invalid("fixed-reference FSI mathematics requires BackwardEuler"))?;
            let RecognizedNativeModel::Fsi(canonical) = &recognized.recognized else {
                unreachable!("FSI capability owns recognized FSI meaning")
            };
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
