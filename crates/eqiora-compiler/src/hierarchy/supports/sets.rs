//! Exact finite Geometry members enter the same complete-exterior proof as source bindings.
use super::*;

#[allow(clippy::too_many_arguments)]
pub(in crate::hierarchy) fn external_complete_exterior_set<I: Clone + Ord>(
    file: &str,
    slot: &str,
    parent: I,
    dimensions: usize,
    members: Vec<(String, I)>,
    resolve: impl Fn(&I) -> Option<CartesianDomain<I>>,
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
    let witness = prove_complete_cartesian_exterior(
        parent.clone(),
        members.iter().map(|member| member.exact_identity.clone()),
        |identity| {
            if identity == &parent {
                Some(CartesianDomain::Volume {
                    ambient_dimension: dimensions,
                })
            } else {
                resolve(identity)
            }
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

/// Build a proved stand-in for an enclosing complete-exterior obligation
/// during occurrence-free reusable-definition checking.
///
/// The identities are compiler-private and deterministic. They are never
/// staged, projected, serialized, or consumed by flattening. Importantly, the
/// stand-in still passes through the same pure Cartesian proof as a concrete
/// occurrence, so symbolic checking cannot accept a weaker set shape.
pub(in crate::hierarchy) fn symbolic_complete_exterior_set(
    file: &str,
    slot_name: &str,
    contract: &CompleteExteriorSlotContract,
    source_range: TextRange,
) -> Result<ResolvedBoundarySet<String>, Diagnostic> {
    let side_count = contract.ambient_dimension().checked_mul(2).ok_or_else(|| {
        source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            file,
            source_range,
            format!("symbolic complete-exterior side count overflows usize for slot `{slot_name}`"),
        )
    })?;
    let mut exact_members = Vec::new();
    let mut resolved_members = Vec::new();
    let mut metadata = Vec::new();
    if exact_members.try_reserve_exact(side_count).is_err()
        || resolved_members.try_reserve_exact(side_count).is_err()
        || metadata.try_reserve_exact(side_count).is_err()
    {
        return Err(source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            file,
            source_range,
            format!(
                "cannot reserve {side_count} symbolic complete-exterior members for slot `{slot_name}`"
            ),
        ));
    }
    for axis in 0..contract.ambient_dimension() {
        for side in [BoundarySide::Lower, BoundarySide::Upper] {
            let side_name = boundary_side_name(side);
            let identity = format!("@symbolic/{slot_name}/axis/{axis}/{side_name}");
            exact_members.push(identity.clone());
            resolved_members.push(ResolvedBoundaryMember {
                target: identity.clone(),
                exact_identity: identity.clone(),
                source_range,
            });
            metadata.push((
                identity,
                CartesianDomain::Boundary {
                    exact_parent: contract.parent_slot().to_owned(),
                    ambient_dimension: contract.ambient_dimension(),
                    axis,
                    side,
                },
            ));
        }
    }
    metadata.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    let exact_parent = contract.parent_slot().to_owned();
    let witness =
        prove_complete_cartesian_exterior(exact_parent.clone(), exact_members, |identity| {
            if identity == &exact_parent {
                Some(CartesianDomain::Volume {
                    ambient_dimension: contract.ambient_dimension(),
                })
            } else {
                metadata
                    .binary_search_by(|candidate| candidate.0.cmp(identity))
                    .ok()
                    .map(|index| metadata[index].1.clone())
            }
        })
        .map_err(|error| {
            complete_exterior_proof_diagnostic(
                file,
                slot_name,
                source_range,
                &resolved_members,
                error,
            )
        })?;
    resolved_members
        .sort_unstable_by(|left, right| left.exact_identity().cmp(right.exact_identity()));
    Ok(ResolvedBoundarySet::Forwarded(
        ForwardedBoundarySetBinding {
            target: slot_name.to_owned(),
            proved: Arc::new(ProvedBoundarySet {
                members: resolved_members.into_boxed_slice(),
                witness,
            }),
            source_range,
        },
    ))
}
