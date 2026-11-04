use eqiora_core::RawId;
use eqiora_schema::kernel::ExprId;

use super::{BoundaryTreatment, FormulationKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MixedFormulationRule {
    MomentumTestPairing,
    StressDivergenceByParts,
    PressureVelocityCoupling,
    ContinuityConstraintPairing,
    SourcePairing,
    ExplicitBoundaryLaw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectionalProof {
    StrongImpliesWeak,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum MixedTermRole {
    SourceDefinition,
    MomentumResidualPairing,
    MomentumStressDivergence,
    MomentumViscousStress,
    MomentumPressureCoupling,
    MomentumBodySource,
    ContinuityConstraint,
    BoundaryLaw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MixedTermSign {
    Positive,
    Negative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MixedNormalOrientation {
    NotApplicable,
    ParentOutward,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MixedBoundaryDisposition {
    NotBoundary,
    EssentialTrace,
    NaturalFlux,
    PrescribedTrace,
    PrescribedFlux,
    PortBinding,
}

/// One source-DAG term consumed by the bounded mixed Galerkin derivation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MixedCertificateEntry {
    pub(crate) relation: RawId,
    pub(crate) source_node: ExprId,
    pub(crate) support: RawId,
    pub(crate) rule: MixedFormulationRule,
    pub(crate) direction: DirectionalProof,
    pub(crate) source_sign: MixedTermSign,
    pub(crate) produced_sign: MixedTermSign,
    pub(crate) role: MixedTermRole,
    pub(crate) test: Option<RawId>,
    pub(crate) trial: Option<RawId>,
    pub(crate) normal: MixedNormalOrientation,
    pub(crate) boundary_disposition: MixedBoundaryDisposition,
    pub(crate) assumptions: &'static [&'static str],
}

impl MixedFormulationRule {
    pub(crate) const fn id(self) -> &'static str {
        match self {
            Self::MomentumTestPairing => "fem.mixed.v1.momentum-test-pairing",
            Self::StressDivergenceByParts => "fem.mixed.v1.stress-divergence-by-parts",
            Self::PressureVelocityCoupling => "fem.mixed.v1.pressure-velocity-coupling",
            Self::ContinuityConstraintPairing => "fem.mixed.v1.continuity-constraint-pairing",
            Self::SourcePairing => "fem.mixed.v1.source-pairing",
            Self::ExplicitBoundaryLaw => "fem.mixed.v1.explicit-boundary-law",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MixedLawIdentity {
    pub(crate) domain: RawId,
    pub(crate) velocity: RawId,
    pub(crate) pressure: RawId,
    pub(crate) source: Option<(RawId, RawId)>,
    pub(crate) momentum_relation: RawId,
    pub(crate) incompressibility_relation: RawId,
    pub(crate) boundary_relations: Vec<RawId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MixedGalerkinFormulation {
    pub(crate) kind: FormulationKind,
    pub(crate) velocity_trial: RawId,
    pub(crate) velocity_test: RawId,
    pub(crate) pressure_trial: RawId,
    pub(crate) pressure_test: RawId,
    pub(crate) boundary_treatment: BoundaryTreatment,
    pub(crate) rules: [MixedFormulationRule; 6],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MixedGalerkinSource<'a> {
    pub(crate) domain: RawId,
    pub(crate) velocity: RawId,
    pub(crate) pressure: RawId,
    pub(crate) source: Option<(RawId, RawId)>,
    pub(crate) momentum_relation: RawId,
    pub(crate) incompressibility_relation: RawId,
    pub(crate) boundary_relations: &'a [RawId],
}

/// Exact correspondence between an admitted mixed Law and its method-level
/// Galerkin interpretation. Discrete spaces, quadrature, gauge, and solver are
/// deliberately absent and remain Realization concerns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MixedGalerkinCorrespondence {
    pub(crate) law: MixedLawIdentity,
    pub(crate) formulation: MixedGalerkinFormulation,
    pub(crate) entries: Vec<MixedCertificateEntry>,
}

impl MixedGalerkinCorrespondence {
    pub(crate) fn derive(source: MixedGalerkinSource<'_>) -> Self {
        Self {
            law: MixedLawIdentity {
                domain: source.domain,
                velocity: source.velocity,
                pressure: source.pressure,
                source: source.source,
                momentum_relation: source.momentum_relation,
                incompressibility_relation: source.incompressibility_relation,
                boundary_relations: source.boundary_relations.to_vec(),
            },
            formulation: MixedGalerkinFormulation {
                kind: FormulationKind::MixedGalerkin,
                velocity_trial: source.velocity,
                velocity_test: source.velocity,
                pressure_trial: source.pressure,
                pressure_test: source.pressure,
                boundary_treatment: BoundaryTreatment::ExplicitTraceFluxLaws,
                rules: [
                    MixedFormulationRule::MomentumTestPairing,
                    MixedFormulationRule::StressDivergenceByParts,
                    MixedFormulationRule::PressureVelocityCoupling,
                    MixedFormulationRule::ContinuityConstraintPairing,
                    MixedFormulationRule::SourcePairing,
                    MixedFormulationRule::ExplicitBoundaryLaw,
                ],
            },
            entries: Vec::new(),
        }
    }

    pub(crate) fn with_entries(mut self, entries: Vec<MixedCertificateEntry>) -> Self {
        self.entries = entries;
        self
    }

    pub(crate) fn replay(&self, source: MixedGalerkinSource<'_>) -> Result<(), &'static str> {
        let expected = Self::derive(source);
        if self.law != expected.law || self.formulation != expected.formulation {
            return Err("mixed Law identity or effective Formulation is stale");
        }
        let mut consumed = std::collections::BTreeSet::new();
        for entry in &self.entries {
            if !consumed.insert((entry.relation, entry.source_node, entry.role)) {
                return Err("mixed correspondence consumes one source term more than once");
            }
            if entry.direction != DirectionalProof::StrongImpliesWeak
                || entry.test.is_some_and(|test| {
                    test != self.formulation.velocity_test && test != self.formulation.pressure_test
                })
                || entry.trial.is_some_and(|trial| {
                    trial != self.formulation.velocity_trial
                        && trial != self.formulation.pressure_trial
                })
            {
                return Err("mixed correspondence contains a cross-wired term role");
            }
        }
        Ok(())
    }
}
