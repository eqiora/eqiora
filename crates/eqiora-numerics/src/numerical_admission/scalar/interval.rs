//! Checked mathematical interval admission for the existing scalar TPFA executor.
use super::*;

pub(super) fn admit(
    admission: &NativeNumericalAdmission,
    lowered: &ExecutableScalarEquations,
    requested: FormulationSelectionMode,
    authored: Option<&AuthoredFormulationProjection>,
) -> Result<Option<CommonFormulationDescription>, Diagnostic> {
    let unavailable = || {
        if authored.is_some() || requested != FormulationSelectionMode::Automatic {
            Err(invalid(
                "interval TPFA formulation requires one steady real scalar Law on a one-dimensional Geometry support",
            ))
        } else {
            Ok(None)
        }
    };
    let [region] = lowered.regions.as_slice() else {
        return unavailable();
    };
    if region.form.dimension() != 1 || region.form.fields().len() != 1 {
        return unavailable();
    }
    let program = admission.program();
    if !matches!(program.node(region.form.domain()), Some(eqiora_schema::kernel::KernelNode::Domain(domain)) if matches!(domain.kind(), eqiora_schema::kernel::DomainKind::GeometryRegion { .. }))
    {
        return unavailable();
    }
    let laws = program
        .nodes()
        .filter_map(|node| match node {
            eqiora_schema::kernel::KernelNode::Relation(law)
                if matches!(
                    law.meaning(),
                    eqiora_schema::kernel::RelationMeaning::Conservation(_)
                ) && program.edges().iter().any(|edge| {
                    edge.from() == law.id().erase()
                        && edge.to() == region.form.domain()
                        && edge.kind() == eqiora_graph::EdgeKind::AppliesOn
                }) =>
            {
                Some(law.id())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let [law] = laws.as_slice() else {
        return unavailable();
    };
    // Keep solver suitability and the existing TPFA boundary admission separate
    // from the mathematical divergence-theorem implication.
    lowered.conservation_descriptor(program)?;
    let (transaction, _) = admission
        .model()
        .to_transaction()
        .map_err(|errors| invalid(format!("interval Model snapshot rejected: {errors:?}")))?;
    let geometry = admission.resources().geometry();
    if let Some(authored) = authored {
        if authored.relation_ulid() != law.ulid().to_string()
            || authored.trial_ulid() != region.form.fields()[0].0.ulid().to_string()
        {
            return Err(invalid(
                "authored interval references a foreign executable Law or Field",
            ));
        }
        authored.check_interval(&transaction, geometry)?;
    } else {
        if eqiora_compiler::check_derived_interval_conservation(&transaction, geometry, *law)?
            .is_none()
        {
            return unavailable();
        }
    }
    let requested = if authored.is_some() {
        FormulationSelectionMode::Authored
    } else {
        requested
    };
    Ok(Some(CommonFormulationDescription {
        requested,
        kind: FormulationKind::IntegralConservative,
        boundary_treatment: "oriented-mathematical-interval-endpoints",
        rule_ids: Box::new([
            "conservative.interval.v1.divergence-theorem",
            "conservative.interval.v1.outward-endpoints",
            "conservative.interval.v1.source-integral",
        ]),
        selection_reason_codes: Box::new([match requested {
            FormulationSelectionMode::Automatic => {
                "eqiora.formulation.auto.interval-conservative-for-tpfa/v1"
            }
            FormulationSelectionMode::Exact => {
                "eqiora.formulation.exact.interval-conservative-admitted/v1"
            }
            FormulationSelectionMode::Authored => {
                "eqiora.formulation.authored.interval-conservative-admitted/v1"
            }
        }]),
        requested_source_identity: authored.map(|form| form.source_identity().to_owned()),
    }))
}
