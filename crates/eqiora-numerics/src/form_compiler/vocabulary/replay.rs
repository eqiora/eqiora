//! Check the closed correspondence inventory without invoking its generator.
//!
//! Source recognition remains the caller's responsibility. This checker binds the
//! resulting source roles; it does not establish regularity or reverse implication.
//! Replay scans the ordered resources once and allocates no replacement certificate.

use super::*;

impl PrimalGalerkinCorrespondence {
    pub(in crate::form_compiler) fn replay_authored_restriction(
        &self,
        authored: &eqiora_compiler::AuthoredFormulationProjection,
    ) -> Result<(), &'static str> {
        if authored.equations().len() != 1
            || authored.trial_ulids().len() != 1
            || authored.test_restrictions().len() != 1
        {
            return Err("scalar primal correspondence requires one equation and test/trial");
        }
        if self
            .law
            .relations
            .first()
            .map(|id| id.ulid().to_string())
            .as_deref()
            != Some(authored.equations()[0].0.as_str())
        {
            return Err("Relation differs from the admitted strong Law");
        }
        if self.law.domain.ulid().to_string() != authored.domain_ulid() {
            return Err("integration support differs from the admitted Domain");
        }
        if self.formulation.trial.ulid().to_string() != authored.trial_ulids()[0].as_str() {
            return Err("trial/test Field differs from the admitted unknown");
        }
        let mut boundaries = self
            .formulation
            .zero_on
            .iter()
            .map(|id| id.ulid().to_string())
            .collect::<Vec<_>>();
        boundaries.sort();
        if authored
            .test_restrictions()
            .first()
            .map(|(_, _, bounds)| bounds.as_slice())
            != Some(boundaries.as_slice())
        {
            return Err("test zero_on restriction differs from the complete essential boundary");
        }
        if authored.implication() != "strong-implies-weak"
            || !authored.assumptions().iter().map(String::as_str).eq(self
                .formulation
                .assumptions
                .iter()
                .copied())
        {
            return Err(
                "authored direction or regularity hypotheses differ from the checked implication",
            );
        }
        Ok(())
    }

    pub(in crate::form_compiler) fn replay(
        &self,
        source: PrimalGalerkinSource<'_>,
    ) -> Result<(), &'static str> {
        if self.law.domain != source.domain
            || self.law.unknown != source.unknown
            || self.law.relations.first() != Some(&source.volume_relation)
            || !self
                .law
                .relations
                .iter()
                .skip(1)
                .copied()
                .eq(source.boundaries.iter().map(|boundary| boundary.relation))
        {
            return Err("scalar Law identity or ordered boundary resources are stale");
        }
        if self.formulation.direction != DirectionalProof::StrongImpliesWeak
            || self.formulation.assumptions
                != eqiora_compiler::AuthoredFormulationProjection::required_assumptions()
            || !self
                .formulation
                .zero_on
                .iter()
                .copied()
                .eq(source.boundaries.iter().map(|boundary| boundary.domain))
            || self.formulation.kind != FormulationKind::PrimalGalerkin
            || self.formulation.trial != source.unknown
            || self.formulation.test != source.unknown
            || self.formulation.boundary_treatment != BoundaryTreatment::CompleteEssential
            || !matches!(
                self.formulation.rules,
                [
                    FormulationRule::TestPairing,
                    FormulationRule::DivergenceByParts,
                    FormulationRule::ZeroTestTraceDischarge,
                    FormulationRule::SourcePairing,
                ]
            )
        {
            return Err("scalar effective Formulation or closed rule inventory is stale");
        }

        // The retained order is test introduction, parts, every essential trace,
        // then source pairing. Requiring exhaustion rejects extra/missing terms.
        let mut entries = self.entries.iter();
        check_entry(
            entries.next(),
            TEST_PAIRING,
            source.volume_relation,
            source.root,
            WeakTermSlot::TestPairing {
                test: MatrixSlot::Test,
            },
            WeakSign::Positive,
        )?;
        check_entry(
            entries.next(),
            DIVERGENCE_BY_PARTS,
            source.volume_relation,
            source.divergence,
            WeakTermSlot::Bilinear {
                test: MatrixSlot::Test,
                trial: MatrixSlot::Trial,
            },
            source.divergence_sign,
        )?;
        for boundary in source.boundaries {
            check_entry(
                entries.next(),
                ZERO_TEST_TRACE_DISCHARGE,
                boundary.relation,
                boundary.trace_node,
                WeakTermSlot::Boundary {
                    test: MatrixSlot::Test,
                },
                match source.divergence_sign {
                    WeakSign::Positive => WeakSign::Negative,
                    WeakSign::Negative => WeakSign::Positive,
                },
            )?;
        }
        check_entry(
            entries.next(),
            SOURCE_PAIRING,
            source.volume_relation,
            source.source,
            WeakTermSlot::Linear {
                test: MatrixSlot::Test,
            },
            WeakSign::Positive,
        )?;
        if entries.next().is_some() {
            return Err("scalar correspondence has unconsumed entries");
        }
        Ok(())
    }
}

fn check_entry(
    entry: Option<&CertificateEntry>,
    rule: &str,
    relation: RawId,
    node: ExprId,
    slot: WeakTermSlot,
    sign: WeakSign,
) -> Result<(), &'static str> {
    let Some(entry) = entry else {
        return Err("scalar correspondence is missing a required entry");
    };
    if entry.rule_id != rule
        || entry.relation != relation
        || entry.source_node != node
        || entry.slot != slot
        || entry.sign != sign
    {
        return Err("scalar correspondence rule, source occurrence, role, or sign is stale");
    }
    Ok(())
}

impl IntegralConservativeCorrespondence {
    pub(crate) fn replay(
        &self,
        source: IntegralConservativeSource<'_>,
    ) -> Result<(), &'static str> {
        if self.law.domain != source.domain
            || self.law.velocity != source.velocity
            || self.law.pressure != source.pressure
            || self.law.source != source.source
            || self.law.source_definition != source.source_definition
            || self.law.momentum_relation != source.momentum_relation
            || self.law.incompressibility_relation != source.incompressibility_relation
            || self.law.boundary_relations != source.boundary_relations
        {
            return Err("conservative Law identity or ordered boundary resources are stale");
        }
        if self.formulation.kind != FormulationKind::IntegralConservative
            || self.formulation.domain != source.domain
            || self.formulation.momentum_unknown != source.velocity
            || self.formulation.pressure_role != source.pressure
            || self.formulation.boundary_treatment != BoundaryTreatment::ExplicitTraceFluxLaws
            || !matches!(
                self.formulation.rules,
                [
                    IntegralConservativeRule::ArbitrarySubdomainBalance,
                    IntegralConservativeRule::TransientStorageIntegral,
                    IntegralConservativeRule::PhysicalMomentumFlux,
                    IntegralConservativeRule::PhysicalStressFlux,
                    IntegralConservativeRule::BodySourceIntegral,
                    IntegralConservativeRule::IncompressibilityFluxBalance,
                    IntegralConservativeRule::ExplicitBoundaryLaw,
                ]
            )
        {
            return Err("conservative effective Formulation or closed rule inventory is stale");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
