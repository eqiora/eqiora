use super::*;

impl CommonFormulationDescription {
    pub(super) fn harmonic(form: &AuthoredFormulationProjection) -> Self {
        Self {
            requested: FormulationSelectionMode::Authored,
            kind: FormulationKind::HarmonicResponse,
            boundary_treatment: "all-original-relations-and-explicit-harmonic-inputs",
            rule_ids: Box::new([
                "harmonic.derive.v1.fixed-domain-real-lti",
                "harmonic.derive.v1.negative-exponential-peak",
                "harmonic.derive.v1.complete-source-boundary-transformation",
                "harmonic.derive.v1.response-restriction-not-initial-value-equivalence",
            ]),
            selection_reason_codes: Box::new(["eqiora.formulation.authored.harmonic-response/v1"]),
            requested_source_identity: Some(form.source_identity().to_owned()),
            source_relation: None,
            state_coordinates: Box::new([]),
        }
    }
    pub(super) fn finite_hermitian(plan: &CommonEigenPlan) -> Self {
        let projected = plan.coordinate_embeddings().next().is_some();
        let mut rules = vec![
            "spectral.derive.v1.homogeneous-affine-pencil",
            "spectral.derive.v1.exact-complex-linear-action",
            "spectral.derive.v1.positive-metric-equation-orientation",
            "spectral.derive.v1.positive-metric-unit-normalization",
        ];
        if projected {
            rules.extend([
                "spectral.derive.v1.source-coordinate-embedding",
                "spectral.derive.v1.lifted-original-pencil-verification",
            ]);
        }
        let authored = plan.authored_formulation.as_ref();
        if authored.is_some() {
            rules.push("finite.derive.v1.exact-conjugate-test-residual");
        }
        Self {
            requested: if authored.is_some() {
                FormulationSelectionMode::Authored
            } else {
                FormulationSelectionMode::Automatic
            },
            kind: FormulationKind::FiniteHermitianPencil,
            boundary_treatment: if projected {
                "source-coordinate-embedding"
            } else {
                "complete-finite-space"
            },
            rule_ids: rules.into_boxed_slice(),
            selection_reason_codes: Box::new([if authored.is_some() {
                "eqiora.formulation.authored.finite-hermitian-pencil/v1"
            } else {
                "eqiora.formulation.auto.finite-hermitian-pencil/v1"
            }]),
            requested_source_identity: authored.map(|form| form.source_identity().to_owned()),
            source_relation: Some(plan.relation()),
            state_coordinates: Box::new([]),
        }
    }

    pub(super) fn first_order(proof: &eqiora_time::TimeLoweringProof) -> Self {
        Self {
            requested: FormulationSelectionMode::Automatic,
            kind: FormulationKind::FirstOrderEvolution,
            boundary_treatment: "not-applicable",
            rule_ids: Box::new([
                "time.derive.v1.source-derivative-coordinates",
                "time.derive.v1.companion-equations",
            ]),
            selection_reason_codes: Box::new(["eqiora.formulation.auto.first-order-evolution/v1"]),
            requested_source_identity: None,
            source_relation: Some(proof.relation()),
            state_coordinates: proof.state_coordinates().into(),
        }
    }

    /// Authored Relation retained by source-preserving time or spectral derivation.
    #[must_use]
    pub const fn source_relation(
        &self,
    ) -> Option<eqiora_core::Id<eqiora_core::entity::kinds::Relation>> {
        self.source_relation
    }

    /// Ordered time coordinates (source Field, derivative order). Consecutive orders
    /// of a Field obey D(q_k) = q_(k+1); its highest rate enters the authored Relation.
    /// Empty for spatial Formulations, whose field/space correspondence has its own owner.
    #[must_use]
    pub fn state_coordinates(&self) -> &[eqiora_core::TimeStateCoordinate] {
        &self.state_coordinates
    }

    pub(super) fn mixed(
        correspondence: &crate::form_compiler::vocabulary::MixedGalerkinCorrespondence,
        requested: FormulationSelectionMode,
        reason: &'static str,
    ) -> Self {
        Self {
            requested,
            kind: correspondence.formulation.kind,
            boundary_treatment: correspondence.formulation.boundary_treatment.id(),
            rule_ids: correspondence
                .formulation
                .rules
                .map(crate::form_compiler::vocabulary::MixedFormulationRule::id)
                .into(),
            selection_reason_codes: Box::new([reason]),
            requested_source_identity: None,
            source_relation: None,
            state_coordinates: Box::new([]),
        }
    }

    pub(super) fn integral(
        correspondence: &crate::form_compiler::vocabulary::IntegralConservativeCorrespondence,
        requested: FormulationSelectionMode,
    ) -> Self {
        Self {
            requested,
            kind: correspondence.formulation.kind,
            boundary_treatment: correspondence.formulation.boundary_treatment.id(),
            rule_ids: correspondence
                .formulation
                .rules
                .map(crate::form_compiler::vocabulary::IntegralConservativeRule::id)
                .into(),
            selection_reason_codes: Box::new([match requested {
                FormulationSelectionMode::Automatic => {
                    "eqiora.formulation.auto.integral-conservative-for-cell-centered-fvm/v1"
                }
                FormulationSelectionMode::Exact => {
                    "eqiora.formulation.exact.integral-conservative-admitted/v1"
                }
                FormulationSelectionMode::Authored => {
                    unreachable!("authored integral-conservative forms are not admitted")
                }
            }]),
            requested_source_identity: None,
            source_relation: None,
            state_coordinates: Box::new([]),
        }
    }

    /// Requested selection mode. The first inspection slice is automatic-only.
    #[must_use]
    pub const fn requested(&self) -> FormulationSelectionMode {
        self.requested
    }

    /// Fresh-compile source identity when selection admitted an authored form.
    #[must_use]
    pub fn requested_source_identity(&self) -> Option<&str> {
        self.requested_source_identity.as_deref()
    }

    /// Exact effective mathematical form.
    #[must_use]
    pub const fn effective(&self) -> FormulationKind {
        self.kind
    }

    /// Versioned boundary-treatment identifier.
    #[must_use]
    pub const fn boundary_treatment(&self) -> &'static str {
        self.boundary_treatment
    }

    /// Complete ordered closed-rule inventory consumed by derivation.
    #[must_use]
    pub fn rule_ids(&self) -> &[&'static str] {
        &self.rule_ids
    }

    /// Stable reasons for the automatic choice.
    #[must_use]
    pub fn selection_reason_codes(&self) -> &[&'static str] {
        &self.selection_reason_codes
    }
}
