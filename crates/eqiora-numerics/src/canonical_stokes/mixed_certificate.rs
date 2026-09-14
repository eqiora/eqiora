//! Term-complete directional certificate for the bounded steady Stokes path.

use std::collections::BTreeSet;
mod authored;
pub use authored::check_authored_mixed_formulation;

use eqiora_core::{Diagnostic, RawId};
use eqiora_schema::kernel::{ExprId, ExprNode};
use eqiora_sem::KernelProgram;

use crate::additive_residual::{AdditiveResidualView, AdditiveSign};
use crate::canonical_boundary::BoundaryRelationBinding;
use crate::form_compiler::vocabulary::{
    DirectionalProof, MixedBoundaryDisposition, MixedCertificateEntry, MixedFormulationRule,
    MixedNormalOrientation, MixedTermRole, MixedTermSign,
};

use super::support::{lowering_error, typed_relation, unique_root};

const TYPED_VOLUME_ASSUMPTIONS: &[&str] = &["typed-continuum-relation"];
const STRESS_BY_PARTS_ASSUMPTIONS: &[&str] = &[
    "typed-continuum-relation",
    "symmetric-newtonian-stress",
    "admitted-regularity-for-divergence-by-parts",
];
const TRACE_BOUNDARY_ASSUMPTIONS: &[&str] = &["complete-explicit-boundary-closure"];
const FLUX_BOUNDARY_ASSUMPTIONS: &[&str] = &[
    "complete-explicit-boundary-closure",
    "parent-outward-normal",
];

pub(super) struct SteadyStokesCertificateSource<'a> {
    pub(super) domain: RawId,
    pub(super) velocity: RawId,
    pub(super) pressure: RawId,
    pub(super) source_definition: RawId,
    pub(super) source_node: ExprId,
    pub(super) momentum_relation: RawId,
    pub(super) incompressibility_relation: RawId,
    pub(super) boundaries: &'a [BoundaryRelationBinding],
    pub(super) boundary_dispositions: &'a std::collections::BTreeMap<
        RawId,
        crate::canonical_boundary::PhysicalBoundaryDisposition,
    >,
}

pub(super) fn derive(
    program: &KernelProgram,
    source: &SteadyStokesCertificateSource<'_>,
) -> Result<Vec<MixedCertificateEntry>, Diagnostic> {
    let source_expression = typed_relation(program, source.source_definition)?;
    let source_root = unique_root(source_expression.expression(), source.source_definition)?;
    let momentum = typed_relation(program, source.momentum_relation)?;
    let momentum_root = unique_root(momentum.expression(), source.momentum_relation)?;
    let momentum_view = AdditiveResidualView::derive(
        momentum.expression(),
        momentum_root,
        source.momentum_relation,
    )?;
    let divergence = momentum_view
        .leaves()
        .iter()
        .find(|leaf| {
            matches!(
                momentum.expression().node(leaf.value()),
                Some(ExprNode::Divergence(_))
            )
        })
        .ok_or_else(|| {
            lowering_error(
                source.momentum_relation,
                "certified Stokes momentum has no stress divergence",
            )
        })?;
    let forcing = momentum_view
        .leaves()
        .iter()
        .find(|leaf| {
            matches!(
                momentum.expression().node(leaf.value()),
                Some(ExprNode::Gradient(_))
            )
        })
        .ok_or_else(|| {
            lowering_error(
                source.momentum_relation,
                "certified Stokes momentum has no body-source gradient",
            )
        })?;
    let (viscous_stress_node, pressure_node) = match momentum.expression().node(divergence.value())
    {
        Some(ExprNode::Divergence(stress)) => match momentum.expression().node(*stress) {
            Some(ExprNode::Sub(viscous, pressure)) => (*viscous, *pressure),
            _ => {
                return Err(lowering_error(
                    source.momentum_relation,
                    "certified Stokes stress has no signed pressure component",
                ));
            }
        },
        _ => unreachable!("divergence leaf was selected above"),
    };
    let continuity = typed_relation(program, source.incompressibility_relation)?;
    let continuity_root = unique_root(continuity.expression(), source.incompressibility_relation)?;
    let continuity_view = AdditiveResidualView::derive(
        continuity.expression(),
        continuity_root,
        source.incompressibility_relation,
    )?;
    let [continuity_leaf] = continuity_view.leaves() else {
        return Err(lowering_error(
            source.incompressibility_relation,
            "certified Stokes continuity must contain exactly one signed source term",
        ));
    };

    let mut entries = vec![
        entry(
            source.source_definition,
            source.source_node,
            source.domain,
            MixedFormulationRule::SourcePairing,
            sign_of_definition(
                source_expression.expression(),
                source_root,
                source.source_node,
                source.source_definition,
            ),
            sign_of_definition(
                source_expression.expression(),
                source_root,
                source.source_node,
                source.source_definition,
            ),
            MixedTermRole::SourceDefinition,
            None,
            None,
            MixedNormalOrientation::NotApplicable,
            MixedBoundaryDisposition::NotBoundary,
            TYPED_VOLUME_ASSUMPTIONS,
        ),
        entry(
            source.momentum_relation,
            momentum_root,
            source.domain,
            MixedFormulationRule::MomentumTestPairing,
            MixedTermSign::Positive,
            MixedTermSign::Positive,
            MixedTermRole::MomentumResidualPairing,
            Some(source.velocity),
            None,
            MixedNormalOrientation::NotApplicable,
            MixedBoundaryDisposition::NotBoundary,
            TYPED_VOLUME_ASSUMPTIONS,
        ),
        entry(
            source.momentum_relation,
            divergence.value(),
            source.domain,
            MixedFormulationRule::StressDivergenceByParts,
            sign(divergence.sign()),
            opposite(sign(divergence.sign())),
            MixedTermRole::MomentumStressDivergence,
            Some(source.velocity),
            Some(source.velocity),
            MixedNormalOrientation::NotApplicable,
            MixedBoundaryDisposition::NotBoundary,
            STRESS_BY_PARTS_ASSUMPTIONS,
        ),
        entry(
            source.momentum_relation,
            viscous_stress_node,
            source.domain,
            MixedFormulationRule::StressDivergenceByParts,
            sign(divergence.sign()),
            opposite(sign(divergence.sign())),
            MixedTermRole::MomentumViscousStress,
            Some(source.velocity),
            Some(source.velocity),
            MixedNormalOrientation::NotApplicable,
            MixedBoundaryDisposition::NotBoundary,
            STRESS_BY_PARTS_ASSUMPTIONS,
        ),
        entry(
            source.momentum_relation,
            pressure_node,
            source.domain,
            MixedFormulationRule::PressureVelocityCoupling,
            opposite(sign(divergence.sign())),
            sign(divergence.sign()),
            MixedTermRole::MomentumPressureCoupling,
            Some(source.velocity),
            Some(source.pressure),
            MixedNormalOrientation::NotApplicable,
            MixedBoundaryDisposition::NotBoundary,
            STRESS_BY_PARTS_ASSUMPTIONS,
        ),
        entry(
            source.momentum_relation,
            forcing.value(),
            source.domain,
            MixedFormulationRule::SourcePairing,
            sign(forcing.sign()),
            sign(forcing.sign()),
            MixedTermRole::MomentumBodySource,
            Some(source.velocity),
            None,
            MixedNormalOrientation::NotApplicable,
            MixedBoundaryDisposition::NotBoundary,
            TYPED_VOLUME_ASSUMPTIONS,
        ),
        entry(
            source.incompressibility_relation,
            continuity_leaf.value(),
            source.domain,
            MixedFormulationRule::ContinuityConstraintPairing,
            sign(continuity_leaf.sign()),
            sign(continuity_leaf.sign()),
            MixedTermRole::ContinuityConstraint,
            Some(source.pressure),
            Some(source.velocity),
            MixedNormalOrientation::NotApplicable,
            MixedBoundaryDisposition::NotBoundary,
            TYPED_VOLUME_ASSUMPTIONS,
        ),
    ];
    for binding in source.boundaries {
        let residual = typed_relation(program, binding.relation())?;
        let disposition = source
            .boundary_dispositions
            .get(&binding.boundary())
            .ok_or_else(|| {
                lowering_error(
                    binding.relation(),
                    "certified boundary Relation has no exact boundary disposition",
                )
            })?;
        for &root in residual.expression().roots() {
            let (boundary_disposition, normal, assumptions) = boundary_root_metadata(
                *disposition,
                residual.expression(),
                root,
                binding.relation(),
            )?;
            entries.push(entry(
                binding.relation(),
                root,
                binding.boundary(),
                MixedFormulationRule::ExplicitBoundaryLaw,
                MixedTermSign::Positive,
                MixedTermSign::Positive,
                MixedTermRole::BoundaryLaw,
                Some(source.velocity),
                None,
                normal,
                boundary_disposition,
                assumptions,
            ));
        }
    }
    check(program, source, &entries, None)?;
    Ok(entries)
}

/// Replay a certificate against the live typed DAG without reconstructing the
/// correspondence object or invoking its constructor.
pub(super) fn check(
    program: &KernelProgram,
    source: &SteadyStokesCertificateSource<'_>,
    entries: &[MixedCertificateEntry],
    authored: Option<&eqiora_compiler::AuthoredFormulationProjection>,
) -> Result<(), Diagnostic> {
    let expected_relations = [
        source.source_definition,
        source.momentum_relation,
        source.incompressibility_relation,
    ]
    .into_iter()
    .chain(source.boundaries.iter().map(|binding| binding.relation()))
    .collect::<BTreeSet<_>>();
    let mut consumed = BTreeSet::new();
    for certificate in entries {
        if !expected_relations.contains(&certificate.relation)
            || program.node(certificate.support).is_none()
            || typed_relation(program, certificate.relation)?
                .expression()
                .node(certificate.source_node)
                .is_none()
        {
            return Err(lowering_error(
                certificate.relation,
                "mixed certificate references a foreign Relation, support, or source node",
            ));
        }
        if !consumed.insert((
            certificate.relation,
            certificate.source_node,
            certificate.role,
        )) {
            return Err(lowering_error(
                certificate.relation,
                "mixed certificate consumes a source term more than once",
            ));
        }
        let roles_match = match certificate.role {
            MixedTermRole::SourceDefinition => {
                certificate.relation == source.source_definition
                    && certificate.source_node == source.source_node
                    && certificate.support == source.domain
                    && certificate.rule == MixedFormulationRule::SourcePairing
                    && certificate.source_sign
                        == live_additive_sign(
                            program,
                            certificate.relation,
                            certificate.source_node,
                        )?
                    && certificate.test.is_none()
                    && certificate.trial.is_none()
                    && certificate.produced_sign == certificate.source_sign
                    && volume_metadata_matches(certificate, TYPED_VOLUME_ASSUMPTIONS)
            }
            MixedTermRole::MomentumStressDivergence => {
                certificate.relation == source.momentum_relation
                    && certificate.support == source.domain
                    && certificate.rule == MixedFormulationRule::StressDivergenceByParts
                    && matches!(
                        typed_relation(program, certificate.relation)?
                            .expression()
                            .node(certificate.source_node),
                        Some(ExprNode::Divergence(_))
                    )
                    && certificate.source_sign
                        == live_additive_sign(
                            program,
                            certificate.relation,
                            certificate.source_node,
                        )?
                    && certificate.test == Some(source.velocity)
                    && certificate.trial == Some(source.velocity)
                    && certificate.produced_sign == opposite(certificate.source_sign)
                    && volume_metadata_matches(certificate, STRESS_BY_PARTS_ASSUMPTIONS)
            }
            MixedTermRole::MomentumResidualPairing => {
                certificate.relation == source.momentum_relation
                    && certificate.support == source.domain
                    && certificate.rule == MixedFormulationRule::MomentumTestPairing
                    && typed_relation(program, certificate.relation)?
                        .expression()
                        .roots()
                        .contains(&certificate.source_node)
                    && certificate.source_sign == MixedTermSign::Positive
                    && certificate.produced_sign == MixedTermSign::Positive
                    && certificate.test == Some(source.velocity)
                    && certificate.trial.is_none()
                    && volume_metadata_matches(certificate, TYPED_VOLUME_ASSUMPTIONS)
            }
            MixedTermRole::MomentumViscousStress => {
                certificate.relation == source.momentum_relation
                    && certificate.support == source.domain
                    && certificate.rule == MixedFormulationRule::StressDivergenceByParts
                    && viscous_node_matches(
                        program,
                        certificate.relation,
                        certificate.source_node,
                        certificate.source_sign,
                        certificate.produced_sign,
                    )?
                    && certificate.test == Some(source.velocity)
                    && certificate.trial == Some(source.velocity)
                    && volume_metadata_matches(certificate, STRESS_BY_PARTS_ASSUMPTIONS)
            }
            MixedTermRole::MomentumBodySource => {
                certificate.relation == source.momentum_relation
                    && certificate.support == source.domain
                    && certificate.rule == MixedFormulationRule::SourcePairing
                    && matches!(
                        typed_relation(program, certificate.relation)?
                            .expression()
                            .node(certificate.source_node),
                        Some(ExprNode::Gradient(_))
                    )
                    && certificate.source_sign
                        == live_additive_sign(
                            program,
                            certificate.relation,
                            certificate.source_node,
                        )?
                    && certificate.test == Some(source.velocity)
                    && certificate.trial.is_none()
                    && certificate.produced_sign == certificate.source_sign
                    && volume_metadata_matches(certificate, TYPED_VOLUME_ASSUMPTIONS)
            }
            MixedTermRole::MomentumPressureCoupling => {
                certificate.relation == source.momentum_relation
                    && certificate.support == source.domain
                    && certificate.rule == MixedFormulationRule::PressureVelocityCoupling
                    && pressure_node_matches(
                        program,
                        certificate.relation,
                        certificate.source_node,
                        certificate.source_sign,
                        certificate.produced_sign,
                    )?
                    && certificate.test == Some(source.velocity)
                    && certificate.trial == Some(source.pressure)
                    && volume_metadata_matches(certificate, STRESS_BY_PARTS_ASSUMPTIONS)
            }
            MixedTermRole::ContinuityConstraint => {
                certificate.relation == source.incompressibility_relation
                    && certificate.support == source.domain
                    && certificate.rule == MixedFormulationRule::ContinuityConstraintPairing
                    && matches!(
                        typed_relation(program, certificate.relation)?
                            .expression()
                            .node(certificate.source_node),
                        Some(ExprNode::Divergence(_))
                    )
                    && certificate.source_sign
                        == live_additive_sign(
                            program,
                            certificate.relation,
                            certificate.source_node,
                        )?
                    && certificate.test == Some(source.pressure)
                    && certificate.trial == Some(source.velocity)
                    && certificate.produced_sign == certificate.source_sign
                    && volume_metadata_matches(certificate, TYPED_VOLUME_ASSUMPTIONS)
            }
            MixedTermRole::BoundaryLaw => {
                let expected_disposition = source
                    .boundary_dispositions
                    .get(&certificate.support)
                    .copied()
                    .map(|disposition| {
                        boundary_root_metadata(
                            disposition,
                            typed_relation(program, certificate.relation)?.expression(),
                            certificate.source_node,
                            certificate.relation,
                        )
                    })
                    .transpose()?;
                source.boundaries.iter().any(|binding| {
                    binding.relation() == certificate.relation
                        && binding.boundary() == certificate.support
                }) && expected_disposition
                    == Some((
                        certificate.boundary_disposition,
                        certificate.normal,
                        certificate.assumptions,
                    ))
                    && certificate.rule == MixedFormulationRule::ExplicitBoundaryLaw
                    && typed_relation(program, certificate.relation)?
                        .expression()
                        .roots()
                        .contains(&certificate.source_node)
                    && certificate.source_sign == MixedTermSign::Positive
                    && certificate.produced_sign == MixedTermSign::Positive
                    && certificate.test == Some(source.velocity)
                    && certificate.trial.is_none()
            }
        };
        if !roles_match || certificate.direction != DirectionalProof::StrongImpliesWeak {
            return Err(lowering_error(
                certificate.relation,
                "mixed certificate has a sign/support/test-trial/role cross-wire",
            ));
        }
    }
    let expected_boundary_terms = source
        .boundaries
        .iter()
        .try_fold(0usize, |count, binding| {
            Ok::<_, Diagnostic>(
                count
                    + typed_relation(program, binding.relation())?
                        .expression()
                        .roots()
                        .len(),
            )
        })?;
    let counts = |role| entries.iter().filter(|entry| entry.role == role).count();
    if counts(MixedTermRole::SourceDefinition) != 1
        || counts(MixedTermRole::MomentumStressDivergence) != 1
        || counts(MixedTermRole::MomentumResidualPairing) != 1
        || counts(MixedTermRole::MomentumViscousStress) != 1
        || counts(MixedTermRole::MomentumPressureCoupling) != 1
        || counts(MixedTermRole::MomentumBodySource) != 1
        || counts(MixedTermRole::ContinuityConstraint) != 1
        || counts(MixedTermRole::BoundaryLaw) != expected_boundary_terms
        || entries.len() != expected_boundary_terms + 7
    {
        return Err(lowering_error(
            source.domain,
            "mixed certificate leaves a source term or boundary law missing or unconsumed",
        ));
    }
    if let Some(authored) = authored {
        self::authored::check_terms(program, source, entries, authored)?;
    }
    Ok(())
}

pub(super) fn check_model(
    program: &KernelProgram,
    model: &super::api::SteadyIncompressibleStokesModel2d,
    entries: &[MixedCertificateEntry],
    authored: Option<&eqiora_compiler::AuthoredFormulationProjection>,
) -> Result<(), Diagnostic> {
    let definition = typed_relation(program, model.force_potential_definition())?;
    let root = unique_root(definition.expression(), model.force_potential_definition())?;
    let source_node = super::expression::additive_load_definition_root(
        definition.expression(),
        root,
        model.force_potential(),
        model.force_potential_definition(),
    )?
    .ok_or_else(|| {
        lowering_error(
            model.domain(),
            "live source definition no longer owns its forcing expression",
        )
    })?;
    check(
        program,
        &SteadyStokesCertificateSource {
            domain: model.domain(),
            velocity: model.velocity(),
            pressure: model.pressure(),
            source_definition: model.force_potential_definition(),
            source_node,
            momentum_relation: model.momentum_relation(),
            incompressibility_relation: model.incompressibility_relation(),
            boundaries: model.boundary_relations(),
            boundary_dispositions: &model
                .boundary_entries()
                .map(|(_, entry)| (entry.boundary, entry.disposition))
                .collect(),
        },
        entries,
        authored,
    )
}

#[allow(clippy::too_many_arguments)]
fn entry(
    relation: RawId,
    source_node: ExprId,
    support: RawId,
    rule: MixedFormulationRule,
    source_sign: MixedTermSign,
    produced_sign: MixedTermSign,
    role: MixedTermRole,
    test: Option<RawId>,
    trial: Option<RawId>,
    normal: MixedNormalOrientation,
    boundary_disposition: MixedBoundaryDisposition,
    assumptions: &'static [&'static str],
) -> MixedCertificateEntry {
    MixedCertificateEntry {
        relation,
        source_node,
        support,
        rule,
        direction: DirectionalProof::StrongImpliesWeak,
        source_sign,
        produced_sign,
        role,
        test,
        trial,
        normal,
        boundary_disposition,
        assumptions,
    }
}

fn sign(value: AdditiveSign) -> MixedTermSign {
    match value {
        AdditiveSign::Positive => MixedTermSign::Positive,
        AdditiveSign::Negative => MixedTermSign::Negative,
    }
}

fn opposite(value: MixedTermSign) -> MixedTermSign {
    match value {
        MixedTermSign::Positive => MixedTermSign::Negative,
        MixedTermSign::Negative => MixedTermSign::Positive,
    }
}

fn boundary_metadata(
    disposition: crate::canonical_boundary::PhysicalBoundaryDisposition,
) -> (
    MixedBoundaryDisposition,
    MixedNormalOrientation,
    &'static [&'static str],
) {
    use crate::canonical_boundary::{PhysicalBoundaryDisposition, PhysicalBoundaryQuantity};
    match disposition {
        PhysicalBoundaryDisposition::TraceZero => (
            MixedBoundaryDisposition::EssentialTrace,
            MixedNormalOrientation::NotApplicable,
            TRACE_BOUNDARY_ASSUMPTIONS,
        ),
        PhysicalBoundaryDisposition::FluxZero => (
            MixedBoundaryDisposition::NaturalFlux,
            MixedNormalOrientation::ParentOutward,
            FLUX_BOUNDARY_ASSUMPTIONS,
        ),
        PhysicalBoundaryDisposition::Prescribed(law) => match law.quantity() {
            PhysicalBoundaryQuantity::Trace => (
                MixedBoundaryDisposition::PrescribedTrace,
                MixedNormalOrientation::NotApplicable,
                TRACE_BOUNDARY_ASSUMPTIONS,
            ),
            PhysicalBoundaryQuantity::Flux => (
                MixedBoundaryDisposition::PrescribedFlux,
                MixedNormalOrientation::ParentOutward,
                FLUX_BOUNDARY_ASSUMPTIONS,
            ),
        },
        PhysicalBoundaryDisposition::PortBinding { .. } => (
            MixedBoundaryDisposition::PortBinding,
            MixedNormalOrientation::ParentOutward,
            FLUX_BOUNDARY_ASSUMPTIONS,
        ),
    }
}

fn boundary_root_metadata(
    disposition: crate::canonical_boundary::PhysicalBoundaryDisposition,
    expression: &eqiora_schema::kernel::ExprDag,
    root: ExprId,
    relation: RawId,
) -> Result<
    (
        MixedBoundaryDisposition,
        MixedNormalOrientation,
        &'static [&'static str],
    ),
    Diagnostic,
> {
    if !matches!(
        disposition,
        crate::canonical_boundary::PhysicalBoundaryDisposition::PortBinding { .. }
    ) {
        return Ok(boundary_metadata(disposition));
    }
    let view = AdditiveResidualView::derive(expression, root, relation)?;
    let is_flux = view.leaves().iter().any(|leaf| {
        matches!(
            expression.node(leaf.value()),
            Some(ExprNode::NormalComponent(_))
                | Some(ExprNode::Symbol(
                    eqiora_schema::kernel::SymbolRef::PortFlux(_)
                ))
        )
    });
    let is_trace = view.leaves().iter().any(|leaf| {
        matches!(
            expression.node(leaf.value()),
            Some(ExprNode::Trace(_))
                | Some(ExprNode::Symbol(
                    eqiora_schema::kernel::SymbolRef::PortTrace(_)
                ))
        )
    });
    match (is_trace, is_flux) {
        (true, false) => Ok((
            MixedBoundaryDisposition::PortBinding,
            MixedNormalOrientation::NotApplicable,
            TRACE_BOUNDARY_ASSUMPTIONS,
        )),
        (false, true) => Ok((
            MixedBoundaryDisposition::PortBinding,
            MixedNormalOrientation::ParentOutward,
            FLUX_BOUNDARY_ASSUMPTIONS,
        )),
        _ => Err(lowering_error(
            relation,
            "port boundary certificate root is not one exact trace or parent-outward flux law",
        )),
    }
}

fn sign_of_definition(
    expression: &eqiora_schema::kernel::ExprDag,
    root: ExprId,
    source: ExprId,
    owner: RawId,
) -> MixedTermSign {
    AdditiveResidualView::derive(expression, root, owner)
        .ok()
        .and_then(|view| {
            view.leaves()
                .iter()
                .find(|leaf| leaf.value() == source)
                .map(|leaf| sign(leaf.sign()))
        })
        .unwrap_or(MixedTermSign::Negative)
}

fn volume_metadata_matches(
    entry: &MixedCertificateEntry,
    assumptions: &'static [&'static str],
) -> bool {
    entry.normal == MixedNormalOrientation::NotApplicable
        && entry.boundary_disposition == MixedBoundaryDisposition::NotBoundary
        && entry.assumptions == assumptions
}

fn live_additive_sign(
    program: &KernelProgram,
    relation: RawId,
    node: ExprId,
) -> Result<MixedTermSign, Diagnostic> {
    let residual = typed_relation(program, relation)?;
    let root = unique_root(residual.expression(), relation)?;
    let view = AdditiveResidualView::derive(residual.expression(), root, relation)?;
    let matching = view
        .leaves()
        .iter()
        .filter(|leaf| leaf.value() == node)
        .collect::<Vec<_>>();
    let [leaf] = matching.as_slice() else {
        return Err(lowering_error(
            relation,
            "mixed certificate source node is missing, duplicated, or not an additive term",
        ));
    };
    Ok(sign(leaf.sign()))
}

fn pressure_node_matches(
    program: &KernelProgram,
    relation: RawId,
    pressure_node: ExprId,
    source_sign: MixedTermSign,
    produced_sign: MixedTermSign,
) -> Result<bool, Diagnostic> {
    let residual = typed_relation(program, relation)?;
    let root = unique_root(residual.expression(), relation)?;
    let view = AdditiveResidualView::derive(residual.expression(), root, relation)?;
    let divergence = view.leaves().iter().find(|leaf| {
        matches!(
            residual.expression().node(leaf.value()),
            Some(ExprNode::Divergence(_))
        )
    });
    let Some(divergence) = divergence else {
        return Ok(false);
    };
    let Some(ExprNode::Divergence(stress)) = residual.expression().node(divergence.value()) else {
        return Ok(false);
    };
    let Some(ExprNode::Sub(_, live_pressure)) = residual.expression().node(*stress) else {
        return Ok(false);
    };
    Ok(*live_pressure == pressure_node
        && opposite(sign(divergence.sign())) == source_sign
        && sign(divergence.sign()) == produced_sign)
}

fn viscous_node_matches(
    program: &KernelProgram,
    relation: RawId,
    viscous_node: ExprId,
    source_sign: MixedTermSign,
    produced_sign: MixedTermSign,
) -> Result<bool, Diagnostic> {
    let residual = typed_relation(program, relation)?;
    let root = unique_root(residual.expression(), relation)?;
    let view = AdditiveResidualView::derive(residual.expression(), root, relation)?;
    let divergence = view.leaves().iter().find(|leaf| {
        matches!(
            residual.expression().node(leaf.value()),
            Some(ExprNode::Divergence(_))
        )
    });
    let Some(divergence) = divergence else {
        return Ok(false);
    };
    let Some(ExprNode::Divergence(stress)) = residual.expression().node(divergence.value()) else {
        return Ok(false);
    };
    let Some(ExprNode::Sub(live_viscous, _)) = residual.expression().node(*stress) else {
        return Ok(false);
    };
    Ok(*live_viscous == viscous_node
        && sign(divergence.sign()) == source_sign
        && opposite(sign(divergence.sign())) == produced_sign)
}
