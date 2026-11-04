//! Exact finite Geometry members enter the same complete-exterior proof as source bindings.
use super::*;

#[allow(clippy::too_many_arguments)]
pub(in crate::hierarchy) fn external_complete_exterior_set<I: Clone + Ord>(
    file: &str,
    slot: &str,
    parent: I,
    dimensions: usize,
    members: Vec<(String, I)>,
    resolve: impl Fn(&I) -> Option<ExteriorDomain<I>>,
    budget: &mut CompleteExteriorMembershipBudget,
) -> Result<ResolvedBoundarySet<I>, Diagnostic> {
    let range = TextRange::default();
    budget.charge(members.len()).map_err(|error| {
        source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            file,
            range,
            format!("external complete-exterior membership resource limit: {error:?}"),
        )
    })?;
    let mut members = members
        .into_iter()
        .map(|(target, exact_identity)| ResolvedBoundaryMember {
            target,
            exact_identity,
            source_range: range,
        })
        .collect::<Vec<_>>();
    let witness = prove_complete_exterior(
        parent.clone(),
        members.iter().map(|member| member.exact_identity.clone()),
        |identity| {
            resolve(identity).or_else(|| {
                (identity == &parent).then_some(ExteriorDomain::Volume {
                    ambient_dimension: dimensions,
                })
            })
        },
    )
    .map_err(|error| complete_exterior_proof_diagnostic(file, slot, range, &members, error))?;
    members.sort_unstable_by(|a, b| a.exact_identity.cmp(&b.exact_identity));
    Ok(ResolvedBoundarySet::Explicit(ExplicitBoundarySetBinding {
        proved: Arc::new(ProvedBoundarySet {
            members: members.into_boxed_slice(),
            witness,
        }),
        source_range: range,
    }))
}

/// Retain a parent/dimension obligation during reusable-definition checking.
/// No synthetic members or successful witness exist until a concrete occurrence
/// supplies and validates the complete partition.
pub(in crate::hierarchy) fn symbolic_complete_exterior_set(
    file: &str,
    slot_name: &str,
    contract: &CompleteExteriorSlotContract,
    source_range: TextRange,
) -> Result<ResolvedBoundarySet<String>, Diagnostic> {
    let _ = (file, slot_name);
    Ok(ResolvedBoundarySet::Deferred {
        exact_parent: contract.parent_slot().to_owned(),
        ambient_dimension: contract.ambient_dimension(),
        source_range,
    })
}
