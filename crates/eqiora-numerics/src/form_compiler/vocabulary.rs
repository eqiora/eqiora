//! Shared private vocabulary for proof-carrying mathematical formulations.

use eqiora_core::RawId;
use eqiora_schema::kernel::ExprId;

mod mixed;
mod replay;
pub(crate) use mixed::{
    DirectionalProof, MixedBoundaryDisposition, MixedCertificateEntry, MixedFormulationRule,
    MixedGalerkinCorrespondence, MixedGalerkinSource, MixedNormalOrientation, MixedTermRole,
    MixedTermSign,
};

pub(super) const TEST_PAIRING: &str = "fem.derive.v1.test-pairing";
pub(super) const DIVERGENCE_BY_PARTS: &str = "fem.derive.v1.divergence-by-parts";
pub(super) const ZERO_TEST_TRACE_DISCHARGE: &str =
    "fem.derive.v2.boundary-discharge.zero-test-trace";
pub(super) const SOURCE_PAIRING: &str = "fem.derive.v1.source-pairing";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MatrixSlot {
    Test,
    Trial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WeakTermSlot {
    TestPairing { test: MatrixSlot },
    Bilinear { test: MatrixSlot, trial: MatrixSlot },
    Boundary { test: MatrixSlot },
    Linear { test: MatrixSlot },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WeakSign {
    Positive,
    Negative,
}

/// Mathematical form selected between an exact Model and its numerical Realization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormulationKind {
    /// Primal test/trial pairing produced by Galerkin derivation.
    PrimalGalerkin,
    /// Mixed test/trial pairing with more than one field role.
    MixedGalerkin,
    /// Arbitrary-subdomain conservation balance consumed by conservative methods.
    IntegralConservative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoundaryTreatment {
    CompleteEssential,
    ExplicitTraceFluxLaws,
}

impl BoundaryTreatment {
    pub(crate) const fn id(self) -> &'static str {
        match self {
            Self::CompleteEssential => "complete-essential",
            Self::ExplicitTraceFluxLaws => "explicit-trace-flux-laws",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FormulationRule {
    TestPairing,
    DivergenceByParts,
    ZeroTestTraceDischarge,
    SourcePairing,
}

impl FormulationRule {
    pub(super) const fn id(self) -> &'static str {
        match self {
            Self::TestPairing => TEST_PAIRING,
            Self::DivergenceByParts => DIVERGENCE_BY_PARTS,
            Self::ZeroTestTraceDischarge => ZERO_TEST_TRACE_DISCHARGE,
            Self::SourcePairing => SOURCE_PAIRING,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LawIdentity {
    pub(super) domain: RawId,
    pub(super) unknown: RawId,
    pub(super) relations: Vec<RawId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct EffectiveFormulation {
    pub(super) kind: FormulationKind,
    pub(super) trial: RawId,
    pub(super) test: RawId,
    pub(super) boundary_treatment: BoundaryTreatment,
    pub(super) rules: [FormulationRule; 4],
    pub(super) zero_on: Vec<RawId>,
    pub(super) direction: DirectionalProof,
    pub(super) assumptions: Vec<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CertificateEntry {
    pub(super) rule_id: &'static str,
    pub(super) relation: RawId,
    pub(super) source_node: ExprId,
    pub(super) slot: WeakTermSlot,
    pub(super) sign: WeakSign,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct BoundarySource {
    pub(super) domain: RawId,
    pub(super) relation: RawId,
    pub(super) trace_node: ExprId,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PrimalGalerkinSource<'a> {
    pub(super) domain: RawId,
    pub(super) unknown: RawId,
    pub(super) volume_relation: RawId,
    pub(super) root: ExprId,
    pub(super) divergence: ExprId,
    pub(super) divergence_sign: WeakSign,
    pub(super) source: ExprId,
    pub(super) boundaries: &'a [BoundarySource],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PrimalGalerkinCorrespondence {
    pub(super) law: LawIdentity,
    pub(super) formulation: EffectiveFormulation,
    pub(super) entries: Vec<CertificateEntry>,
}

impl PrimalGalerkinCorrespondence {
    pub(super) fn derive(source: PrimalGalerkinSource<'_>) -> Self {
        let rules = [
            FormulationRule::TestPairing,
            FormulationRule::DivergenceByParts,
            FormulationRule::ZeroTestTraceDischarge,
            FormulationRule::SourcePairing,
        ];
        let mut relations = Vec::with_capacity(source.boundaries.len() + 1);
        relations.push(source.volume_relation);
        relations.extend(source.boundaries.iter().map(|boundary| boundary.relation));

        let mut entries = Vec::with_capacity(source.boundaries.len() + 3);
        entries.push(CertificateEntry {
            rule_id: rules[0].id(),
            relation: source.volume_relation,
            source_node: source.root,
            slot: WeakTermSlot::TestPairing {
                test: MatrixSlot::Test,
            },
            sign: WeakSign::Positive,
        });
        entries.push(CertificateEntry {
            rule_id: rules[1].id(),
            relation: source.volume_relation,
            source_node: source.divergence,
            slot: WeakTermSlot::Bilinear {
                test: MatrixSlot::Test,
                trial: MatrixSlot::Trial,
            },
            sign: source.divergence_sign,
        });
        entries.extend(source.boundaries.iter().map(|boundary| CertificateEntry {
            rule_id: rules[2].id(),
            relation: boundary.relation,
            source_node: boundary.trace_node,
            slot: WeakTermSlot::Boundary {
                test: MatrixSlot::Test,
            },
            sign: match source.divergence_sign {
                WeakSign::Positive => WeakSign::Negative,
                WeakSign::Negative => WeakSign::Positive,
            },
        }));
        entries.push(CertificateEntry {
            rule_id: rules[3].id(),
            relation: source.volume_relation,
            source_node: source.source,
            slot: WeakTermSlot::Linear {
                test: MatrixSlot::Test,
            },
            sign: WeakSign::Positive,
        });

        Self {
            law: LawIdentity {
                domain: source.domain,
                unknown: source.unknown,
                relations,
            },
            formulation: EffectiveFormulation {
                kind: FormulationKind::PrimalGalerkin,
                trial: source.unknown,
                test: source.unknown,
                boundary_treatment: BoundaryTreatment::CompleteEssential,
                zero_on: source
                    .boundaries
                    .iter()
                    .map(|boundary| boundary.domain)
                    .collect(),
                direction: DirectionalProof::StrongImpliesWeak,
                assumptions: eqiora_compiler::AuthoredFormulationProjection::required_assumptions()
                    .to_vec(),
                rules,
            },
            entries,
        }
    }
}

/// Closed mathematical transformations from conservative differential Laws
/// to arbitrary-subdomain integral balances. These rules contain no mesh,
/// control-volume layout, numerical face flux, quadrature, or solver choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IntegralConservativeRule {
    ArbitrarySubdomainBalance,
    TransientStorageIntegral,
    PhysicalMomentumFlux,
    PhysicalStressFlux,
    BodySourceIntegral,
    IncompressibilityFluxBalance,
    ExplicitBoundaryLaw,
}

impl IntegralConservativeRule {
    pub(crate) const fn id(self) -> &'static str {
        match self {
            Self::ArbitrarySubdomainBalance => {
                "conservative.integral.v1.arbitrary-subdomain-balance"
            }
            Self::TransientStorageIntegral => "conservative.integral.v1.transient-storage-integral",
            Self::PhysicalMomentumFlux => "conservative.integral.v1.physical-momentum-flux",
            Self::PhysicalStressFlux => "conservative.integral.v1.physical-stress-flux",
            Self::BodySourceIntegral => "conservative.integral.v1.body-source-integral",
            Self::IncompressibilityFluxBalance => {
                "conservative.integral.v1.incompressibility-flux-balance"
            }
            Self::ExplicitBoundaryLaw => "conservative.integral.v1.explicit-boundary-law",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ConservativeFlowLawIdentity {
    pub(crate) domain: RawId,
    pub(crate) velocity: RawId,
    pub(crate) pressure: RawId,
    pub(crate) source: RawId,
    pub(crate) source_definition: RawId,
    pub(crate) momentum_relation: RawId,
    pub(crate) incompressibility_relation: RawId,
    pub(crate) boundary_relations: Vec<RawId>,
}

/// Effective integral form consumed by a conservative Realization. `domain`
/// denotes an arbitrary mathematical subdomain; it is not a mesh cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IntegralConservativeFormulation {
    pub(crate) kind: FormulationKind,
    pub(crate) domain: RawId,
    pub(crate) momentum_unknown: RawId,
    pub(crate) pressure_role: RawId,
    pub(crate) boundary_treatment: BoundaryTreatment,
    pub(crate) rules: [IntegralConservativeRule; 7],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct IntegralConservativeSource<'a> {
    pub(crate) domain: RawId,
    pub(crate) velocity: RawId,
    pub(crate) pressure: RawId,
    pub(crate) source: RawId,
    pub(crate) source_definition: RawId,
    pub(crate) momentum_relation: RawId,
    pub(crate) incompressibility_relation: RawId,
    pub(crate) boundary_relations: &'a [RawId],
}

/// Exact directional correspondence between the recognized physical Laws and
/// their integral-conservative form. A later FVM Realization may map its
/// control volumes to the arbitrary-subdomain role and choose numerical face
/// fluxes, without inserting those numerical choices into this certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IntegralConservativeCorrespondence {
    pub(crate) law: ConservativeFlowLawIdentity,
    pub(crate) formulation: IntegralConservativeFormulation,
}

impl IntegralConservativeCorrespondence {
    pub(crate) fn derive(source: IntegralConservativeSource<'_>) -> Self {
        Self {
            law: ConservativeFlowLawIdentity {
                domain: source.domain,
                velocity: source.velocity,
                pressure: source.pressure,
                source: source.source,
                source_definition: source.source_definition,
                momentum_relation: source.momentum_relation,
                incompressibility_relation: source.incompressibility_relation,
                boundary_relations: source.boundary_relations.to_vec(),
            },
            formulation: IntegralConservativeFormulation {
                kind: FormulationKind::IntegralConservative,
                domain: source.domain,
                momentum_unknown: source.velocity,
                pressure_role: source.pressure,
                boundary_treatment: BoundaryTreatment::ExplicitTraceFluxLaws,
                rules: [
                    IntegralConservativeRule::ArbitrarySubdomainBalance,
                    IntegralConservativeRule::TransientStorageIntegral,
                    IntegralConservativeRule::PhysicalMomentumFlux,
                    IntegralConservativeRule::PhysicalStressFlux,
                    IntegralConservativeRule::BodySourceIntegral,
                    IntegralConservativeRule::IncompressibilityFluxBalance,
                    IntegralConservativeRule::ExplicitBoundaryLaw,
                ],
            },
        }
    }
}
