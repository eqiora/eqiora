//! Constitutive proofs and scientific projections for the shared linear Plan.
use super::*;

pub(super) fn recognize(
    program: &KernelProgram,
    resources: &NativeMeshResources,
) -> Option<IsotropicElasticityContinuum<2>> {
    let NativeMeshResources::Cartesian {
        geometry,
        mesh,
        correspondence,
        ..
    } = resources
    else {
        return None;
    };
    lower_isotropic_elasticity_geometry_2d(program, geometry, mesh, correspondence).ok()
}

impl CommonLinearPlan {
    pub(crate) fn require_elasticity_observation(
        &self,
        bounds: Option<[[f64; 2]; 2]>,
    ) -> Result<(), Diagnostic> {
        let expected = if self.admission.elasticity_proof().is_some() {
            self.admission
                .resources()
                .geometry()?
                .planar_rectangle_bounds()
                .copied()
        } else {
            None
        };
        if bounds != expected {
            return Err(invalid(
                "elastic observation differs from the exact linear Plan and Geometry",
            ));
        }
        Ok(())
    }

    pub(crate) fn elasticity_observation(
        &self,
        reactions: Option<&crate::region_assembly::RecoveredInterfaceReactions<f64>>,
    ) -> Result<Option<CommonElasticityObservation>, Diagnostic> {
        let Some(continuum) = self.admission.elasticity_proof() else {
            return Ok(None);
        };
        let reactions = reactions.ok_or_else(|| {
            invalid("elastic observation requires the shared exact reaction recovery")
        })?;
        let mut constrained_reaction = [0.0; 2];
        let mut integrated_body_force = [0.0; 2];
        for (values, sums) in [
            (&reactions.constrained_actions, &mut constrained_reaction),
            (&reactions.volume_loads, &mut integrated_body_force),
        ] {
            let mut count = 0;
            for (key, value) in values
                .iter()
                .filter(|(key, _)| key.field == continuum.displacement())
            {
                let sum = sums
                    .get_mut(key.component)
                    .ok_or_else(|| invalid("elastic dual has a foreign component"))?;
                *sum += value;
                count += 1;
            }
            if count == 0 || sums.iter().any(|value| !value.is_finite()) {
                return Err(invalid("elastic dual is missing or non-finite"));
            }
        }
        let exact_bounds = self
            .admission
            .resources()
            .geometry()?
            .planar_rectangle_bounds()
            .copied()
            .ok_or_else(|| invalid("elastic observation requires exact rectangular Geometry"))?;
        Ok(Some(CommonElasticityObservation {
            constrained_reaction,
            integrated_body_force,
            exact_bounds,
        }))
    }
}

pub(super) fn describe_formulation(
    admission: &NativeNumericalAdmission,
    continuum: &IsotropicElasticityContinuum<2>,
    selection: FormulationSelectionMode,
    authored: Option<&AuthoredFormulationProjection>,
) -> Result<Option<CommonFormulationDescription>, Diagnostic> {
    let derived = crate::form_compiler::derive_elasticity_correspondence(
        admission.program(),
        continuum,
        authored,
    )?;
    let Some((kind, boundary, rules)) = derived else {
        if authored.is_some() || selection != FormulationSelectionMode::Automatic {
            return Err(invalid(
                "elastic primal Formulation requires admitted trace or homogeneous natural boundary laws",
            ));
        }
        return Ok(None);
    };
    let mut description = super::scalar::describe_primal(
        kind,
        boundary,
        rules,
        if authored.is_some() {
            FormulationSelectionMode::Authored
        } else {
            selection
        },
    );
    description.requested_source_identity = authored.map(|form| form.source_identity().to_owned());
    Ok(Some(description))
}
