//! Harmonic reduction precedes ordinary complex scalar recognition and assembly.
use super::*;
use crate::form_compiler::harmonic::HarmonicReduction;

impl CommonScalarPlan {
    pub(in crate::numerical_admission) fn with_harmonic(
        self,
        reduction: HarmonicReduction,
        form: AuthoredFormulationProjection,
    ) -> Result<Self, Diagnostic> {
        if self.admission.model() != &reduction.reduced
            || self.harmonic.is_some()
            || !matches!(
                self.admission.recognized_model(),
                RecognizedNativeModel::ComplexScalar(_)
            )
        {
            return Err(invalid(
                "harmonic reduction does not own this exact complex scalar Model",
            ));
        }
        let mut description = CommonFormulationDescription::harmonic(&form);
        if let Some(inner) = &self.formulation {
            let mut rules = description.rule_ids.into_vec();
            rules.extend_from_slice(&inner.rule_ids);
            description.rule_ids = rules.into_boxed_slice();
        }
        let mut plan = Self::finish_admission(
            &reduction.reduced,
            self.admission,
            self.cells,
            self.fields,
            self.portable,
            Some(description),
            Some(form),
        )?;
        plan.harmonic = Some(reduction);
        Ok(plan)
    }

    /// Original real time-domain Model including the retained initial conditions.
    pub fn harmonic_original_model(&self) -> Option<&ModelEnvelope> {
        self.harmonic.as_ref().map(|reduction| &reduction.original)
    }

    /// Positive angular frequency in the retained negative-exponential, peak convention.
    pub fn harmonic_angular_frequency(&self) -> Option<f64> {
        self.harmonic
            .as_ref()
            .map(|reduction| reduction.angular_frequency)
    }

    /// Reconstruct one original real Field's spatial coefficient block at model time.
    /// Coefficients retain the amplitude Result's association and logical block shape,
    /// and the original Field's units. They do not satisfy arbitrary original initial conditions.
    pub fn reconstruct_harmonic_field_block(
        &self,
        result: &crate::CommonResult,
        original: eqiora_core::Id<eqiora_core::entity::kinds::Field>,
        block: usize,
        time: DynQuantity,
    ) -> Result<Vec<f64>, Diagnostic> {
        let reduction = self
            .harmonic
            .as_ref()
            .ok_or_else(|| invalid("Plan has no harmonic reconstruction"))?;
        if result.plan().as_scalar() != Some(self) {
            return Err(invalid(
                "harmonic reconstruction requires this exact Plan's accepted Result",
            ));
        }
        let amplitude = reduction
            .amplitudes
            .iter()
            .find(|(_, field, _)| *field == original)
            .map(|(_, _, amplitude)| amplitude.ulid().to_string())
            .ok_or_else(|| invalid("Field has no mapping in this original harmonic Model"))?;
        let field = (0..result.field_count())
            .find(|&field| {
                result
                    .field(field)
                    .is_some_and(|(id, _, _, _)| id == amplitude)
            })
            .ok_or_else(|| invalid("harmonic Result omitted a mapped amplitude"))?;
        let (_, values, _) = result
            .field_block(field, block)
            .ok_or_else(|| invalid("harmonic amplitude block is absent"))?;
        if result.field_scalar_domain(field) != Some(eqiora_core::ScalarDomain::Complex)
            || !values.len().is_multiple_of(2)
        {
            return Err(invalid(
                "harmonic amplitude block has incompatible complex storage",
            ));
        }
        reduction.reconstruct(
            time,
            values
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[real, imag]| (real, imag)),
        )
    }

    /// Ordered amplitude names and exact original/derived Field identities.
    pub fn harmonic_amplitudes(
        &self,
    ) -> impl Iterator<
        Item = (
            &str,
            eqiora_core::Id<eqiora_core::entity::kinds::Field>,
            eqiora_core::Id<eqiora_core::entity::kinds::Field>,
        ),
    > {
        self.harmonic
            .iter()
            .flat_map(|reduction| reduction.amplitudes.iter())
            .map(|(name, original, amplitude)| (name.as_str(), *original, *amplitude))
    }

    pub(super) fn reauthenticate_harmonic(&self) -> Result<(), Diagnostic> {
        let reduction = self.harmonic.as_ref().expect("harmonic replay dispatch");
        let form = self
            .authored_formulation
            .as_ref()
            .ok_or_else(|| invalid("harmonic Plan lost its original request"))?;
        let geometry = self.admission.resources().geometry()?;
        let original = replay_program(&reduction.original, geometry)?;
        let repeated = HarmonicReduction::derive(&original, form, Some(geometry))?;
        if repeated != *reduction {
            return Err(invalid(
                "harmonic Plan differs from its repeated original-source reduction",
            ));
        }
        self.admission.revalidate()?;
        let repeated = Self::from_complex_admission(
            self.admission.model(),
            self.admission.clone(),
            FormulationSelectionMode::Automatic,
            None,
        )?
        .with_harmonic(repeated, form.clone())?;
        if repeated != *self {
            return Err(invalid("harmonic scalar Plan changed during exact replay"));
        }
        Ok(())
    }
}
