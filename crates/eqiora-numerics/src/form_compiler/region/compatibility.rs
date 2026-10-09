//! Differential compatibility belongs to binding, before any cell assembly.
use eqiora_realization::SpaceFamily;

use super::*;

pub(super) fn validate<S: Coefficient>(
    form: &CompiledRegionForm<S>,
    fields: &[RegionFieldLayout],
    time: Option<&RegionTimeBinding>,
) -> Result<(), Diagnostic> {
    let moment = |family| {
        matches!(
            family,
            SpaceFamily::TetrahedralEdge | SpaceFamily::TetrahedralFace
        )
    };
    if !fields.iter().any(|field| moment(field.space.family())) {
        return Ok(());
    }
    // Existing history and nonlinear trace evaluators are componentwise nodal.
    // Do not admit moment coordinates into those paths until they consume the
    // same oriented coefficient interpretation as the static linear operator.
    if time.is_some() || form.rows.iter().any(|row| !row.dyadics.is_empty()) {
        return Err(invalid(
            "moment bindings currently require static linear region forms",
        ));
    }
    let family = |id| {
        fields
            .iter()
            .find(|field| field.field == id)
            .map(|field| field.space.family())
            .ok_or_else(|| invalid("differential pairing requires an exact bound Field"))
    };
    let gradient = |family| {
        matches!(
            family,
            SpaceFamily::ContinuousLagrange { .. } | SpaceFamily::SimplexP1Bubble
        )
    };
    let curl = |family| gradient(family) || family == SpaceFamily::TetrahedralEdge;
    let divergence = |family| gradient(family) || family == SpaceFamily::TetrahedralFace;
    for row in &form.rows {
        let test = family(row.tested)?;
        for term in &row.terms {
            let trial = family(term.trial)?;
            let accepted = match term.pairing {
                Pairing::Value => true,
                Pairing::Gradient | Pairing::SymmetricGradient => gradient(test) && gradient(trial),
                Pairing::Curl => curl(test) && curl(trial),
                Pairing::Divergence => divergence(test) && divergence(trial),
                Pairing::TestDivergenceTrialValue => divergence(test),
                Pairing::TestValueTrialDivergence => divergence(trial),
            };
            if !accepted {
                return Err(invalid(
                    "weak differential pairing is incompatible with the bound Space",
                ));
            }
        }
        if !divergence(test)
            && row
                .flux
                .iter()
                .any(|term| matches!(term, flux::FluxTerm::Isotropic(_)))
        {
            return Err(invalid(
                "isotropic flux requires a divergence-conforming test Space",
            ));
        }
    }
    Ok(())
}
