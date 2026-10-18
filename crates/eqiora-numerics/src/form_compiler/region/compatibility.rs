//! Differential compatibility belongs to binding, before any cell assembly.
use eqiora_realization::SpaceFamily;

use super::*;

pub(super) fn validate<S: Coefficient>(
    form: &CompiledRegionForm<S>,
    fields: &[RegionFieldLayout],
    states: &BTreeMap<RawId, RegionFieldLayout>,
    time: Option<&RegionTimeBinding>,
) -> Result<(), Diagnostic> {
    let moment = |family| {
        matches!(
            family,
            SpaceFamily::TetrahedralEdge | SpaceFamily::TetrahedralFace
        )
    };
    if fields.iter().any(|field| {
        moment(field.space.family()) && !spatial_vector(&field.value_type, form.dimension)
    }) {
        return Err(invalid(
            "moment bindings require exact spatial vector components, not equal-sized channels",
        ));
    }
    // Existing history and nonlinear trace evaluators are componentwise nodal.
    // Do not admit moment coordinates into those paths until they consume the
    // same oriented coefficient interpretation as the static linear operator.
    if fields.iter().any(|field| moment(field.space.family()))
        && (time.is_some() || form.rows.iter().any(|row| !row.dyadics.is_empty()))
    {
        return Err(invalid(
            "moment bindings currently require static linear region forms",
        ));
    }
    let family = |id| {
        fields
            .iter()
            .find(|field| field.field == id)
            .or_else(|| states.get(&id))
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
        for term in &row.dyadics {
            if !gradient(test) || !gradient(family(term.field)?) {
                return Err(invalid(
                    "dyadic flux requires gradient-conforming test and trial Spaces",
                ));
            }
        }
        for term in &row.terms {
            let trial = family(term.trial)?;
            let accepted = match term.pairing {
                Pairing::Value => true,
                Pairing::Components {
                    test: test_component,
                    trial: trial_component,
                } => {
                    (!test_component.requires_gradient() || gradient(test))
                        && (!trial_component.requires_gradient() || gradient(trial))
                }
                Pairing::TestGradientTrialValue(_) => gradient(test),
                Pairing::Gradient | Pairing::SkewGradient(_) | Pairing::SymmetricGradient => {
                    gradient(test) && gradient(trial)
                }
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
        if !gradient(test)
            && row
                .flux
                .iter()
                .any(|term| matches!(term, flux::FluxTerm::Prescribed { .. }))
        {
            return Err(invalid(
                "component flux load requires a gradient-conforming test Space",
            ));
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

impl<S: Coefficient> CompiledRegionForm<S> {
    pub(in crate::form_compiler) fn stored_fields(
        &self,
    ) -> Result<BTreeMap<RawId, Data<S>>, Diagnostic> {
        let mut storage = BTreeMap::new();
        for row in &self.rows {
            for term in row.terms.iter().filter(|term| term.derivative) {
                if self.roles.relations.values().any(
                    |role| matches!(role.kind, Role::Kinematic {state,..} if state == term.trial),
                ) {
                    continue;
                }
                if term.trial != row.tested || term.pairing != Pairing::Value {
                    return Err(invalid(
                        "Backward Euler requires each row to store its exact Field",
                    ));
                }
                storage
                    .entry(row.tested)
                    .and_modify(|value: &mut Data<S>| {
                        *value = value.clone().add(term.coefficient.clone())
                    })
                    .or_insert_with(|| term.coefficient.clone());
            }
        }
        Ok(storage)
    }
}
