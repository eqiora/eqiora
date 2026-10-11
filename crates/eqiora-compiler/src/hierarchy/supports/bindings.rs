//! Singular and finite boundary-set bindings share one resolver; only definition checking may defer abstract topology.
use super::*;

#[allow(clippy::too_many_arguments)]
pub(in crate::hierarchy) fn resolve_instance_support_bindings<I: Clone + Ord>(
    binding_file: &str,
    component: &ComponentDecl,
    interface: &SupportInterface,
    instance: &InstanceDecl,
    resolve_singular: impl FnMut(&str) -> Option<SpatialSupport<I>>,
    resolve_boundary_name: impl FnMut(&str) -> Option<ResolvedBoundaryTarget<I>>,
    resolve_cartesian_domain: impl FnMut(&I) -> Option<CartesianDomain<I>>,
    resolve_forwarded_set: impl FnMut(&str) -> Option<ResolvedBoundarySet<I>>,
    membership_budget: &mut CompleteExteriorMembershipBudget,
) -> Result<ResolvedSupportBindings<I>, Vec<Diagnostic>> {
    resolve_definition_support_bindings(
        binding_file,
        component,
        interface,
        instance,
        resolve_singular,
        resolve_boundary_name,
        resolve_cartesian_domain,
        resolve_forwarded_set,
        |_| false,
        membership_budget,
    )
}

/// Resolve singular supports and complete-exterior occurrence bindings.
///
/// `resolve_boundary_name` maps one lexical member to a stable enclosing-scope
/// locator and exact elaboration identity. `resolve_cartesian_domain` supplies
/// authoritative metadata by exact identity. `resolve_forwarded_set` only
/// accepts an already proved enclosing set; forwarding therefore cannot weaken
/// the proof obligation.
#[allow(clippy::too_many_arguments)]
pub(in crate::hierarchy) fn resolve_definition_support_bindings<I: Clone + Ord>(
    binding_file: &str,
    component: &ComponentDecl,
    interface: &SupportInterface,
    instance: &InstanceDecl,
    mut resolve_singular: impl FnMut(&str) -> Option<SpatialSupport<I>>,
    mut resolve_boundary_name: impl FnMut(&str) -> Option<ResolvedBoundaryTarget<I>>,
    mut resolve_cartesian_domain: impl FnMut(&I) -> Option<CartesianDomain<I>>,
    mut resolve_forwarded_set: impl FnMut(&str) -> Option<ResolvedBoundarySet<I>>,
    mut abstract_boundary: impl FnMut(&str) -> bool,
    membership_budget: &mut CompleteExteriorMembershipBudget,
) -> Result<ResolvedSupportBindings<I>, Vec<Diagnostic>> {
    let mut diagnostics = Vec::new();
    let mut targets = BTreeMap::new();
    let mut actual = BTreeMap::new();
    let mut boundary_sets = BTreeMap::new();
    let mut deferred_topology = BTreeSet::new();
    let mut forwarded = Vec::new();
    let mut explicit = Vec::new();
    let mut seen = BTreeSet::new();

    for binding in crate::hierarchy::named_bindings::references(
        binding_file,
        instance,
        |binding| {
            (interface.get(binding.name()).is_some()
                || interface.complete_exterior(binding.name()).is_some())
                && !crate::hierarchy::named_bindings::is_boundary_set(binding)
        },
        &mut diagnostics,
    ) {
        if !seen.insert(binding.slot()) {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                binding_file,
                binding.range(),
                format!(
                    "duplicate binding for support slot `{}` in instance `{}`",
                    binding.slot(),
                    instance.name()
                ),
            ));
            continue;
        }
        if let Some(contract) = interface.complete_exterior(binding.slot()) {
            if contract.visibility() != VisibilitySyntax::Public {
                diagnostics.push(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    binding_file,
                    binding.range(),
                    format!(
                        "private complete-exterior support slot `{}` cannot be bound on instance `{}`",
                        binding.slot(),
                        instance.name()
                    ),
                ));
                continue;
            }
            let Some(source_set) = resolve_forwarded_set(binding.target()) else {
                diagnostics.push(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    binding_file,
                    binding.range(),
                    format!(
                        "complete-exterior binding target `{}` is not a proved enclosing BoundarySet",
                        binding.target()
                    ),
                ));
                continue;
            };
            forwarded.push((binding, contract, source_set));
            continue;
        }
        let Some(slot) = interface.get(binding.slot()) else {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                binding_file,
                binding.range(),
                format!(
                    "unknown support slot `{}` on component `{}`",
                    binding.slot(),
                    component.name()
                ),
            ));
            continue;
        };
        if slot.visibility() != VisibilitySyntax::Public {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                binding_file,
                binding.range(),
                format!(
                    "private support slot `{}` cannot be bound on instance `{}`",
                    binding.slot(),
                    instance.name()
                ),
            ));
            continue;
        }
        let Some(support) = resolve_singular(binding.target()) else {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                binding_file,
                binding.range(),
                format!(
                    "support binding target `{}` is not an enclosing spatial support",
                    binding.target()
                ),
            ));
            continue;
        };
        targets.insert(binding.slot().to_owned(), binding.target().to_owned());
        actual.insert(binding.slot().to_owned(), (support, binding.range()));
    }

    for binding in crate::hierarchy::named_bindings::boundary_sets(
        binding_file,
        instance,
        |name| interface.get(name).is_some() || interface.complete_exterior(name).is_some(),
        &mut diagnostics,
    ) {
        if !seen.insert(binding.slot()) {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                binding_file,
                binding.range(),
                format!(
                    "duplicate binding for support slot `{}` in instance `{}`",
                    binding.slot(),
                    instance.name()
                ),
            ));
            continue;
        }
        let Some(contract) = interface.complete_exterior(binding.slot()) else {
            let message = if interface.get(binding.slot()).is_some() {
                format!(
                    "singular support slot `{}` cannot be bound with `boundaries(...)`",
                    binding.slot()
                )
            } else {
                format!(
                    "unknown support slot `{}` on component `{}`",
                    binding.slot(),
                    component.name()
                )
            };
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                binding_file,
                binding.range(),
                message,
            ));
            continue;
        };
        explicit.push((binding, contract));
    }

    validate_singular_support_shapes(binding_file, interface, &actual, &mut diagnostics);

    for (binding, contract, source_set) in forwarded {
        let Some((SpatialSupport::Volume { domain, dimensions }, _)) =
            actual.get(contract.parent_slot())
        else {
            continue;
        };
        if dimensions != &contract.ambient_dimension() {
            diagnostics.push(source_error(
                codes::LANGUAGE_LOWERING_ERROR,
                binding_file,
                binding.range(),
                format!(
                    "complete-exterior slot `{}` has inconsistent definition-time dimension",
                    binding.slot()
                ),
            ));
            continue;
        }
        if source_set.witness().exact_parent() != domain {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                binding_file,
                binding.range(),
                format!(
                    "forwarded BoundarySet for slot `{}` does not have the exact bound parent slot `{}`",
                    binding.slot(),
                    contract.parent_slot()
                ),
            ));
            continue;
        }
        if source_set.witness().ambient_dimension() != *dimensions {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                binding_file,
                binding.range(),
                format!(
                    "forwarded BoundarySet for slot `{}` requires ambient dimension {}",
                    binding.slot(),
                    dimensions
                ),
            ));
            continue;
        }
        boundary_sets.insert(
            binding.slot().to_owned(),
            ResolvedBoundarySet::Forwarded(ForwardedBoundarySetBinding {
                target: binding.target().to_owned(),
                proved: Arc::clone(source_set.proved()),
                source_range: binding.range(),
            }),
        );
    }

    for (binding, contract) in explicit {
        if let Err(error) = membership_budget.charge(binding.members().len()) {
            diagnostics.push(complete_exterior_budget_diagnostic(
                binding_file,
                &binding,
                error,
            ));
            continue;
        }
        let Some((SpatialSupport::Volume { domain, dimensions }, _)) =
            actual.get(contract.parent_slot())
        else {
            continue;
        };
        if dimensions != &contract.ambient_dimension() {
            diagnostics.push(source_error(
                codes::LANGUAGE_LOWERING_ERROR,
                binding_file,
                binding.range(),
                format!(
                    "complete-exterior slot `{}` has inconsistent definition-time dimension",
                    binding.slot()
                ),
            ));
            continue;
        }

        let member_count = binding.members().len();
        let mut exact_members = Vec::new();
        let mut resolved_members = Vec::new();
        if exact_members.try_reserve_exact(member_count).is_err()
            || resolved_members.try_reserve_exact(member_count).is_err()
        {
            diagnostics.push(source_error(
                codes::LANGUAGE_LOWERING_ERROR,
                binding_file,
                binding.range(),
                format!(
                    "cannot reserve {member_count} exact members for complete-exterior slot `{}`",
                    binding.slot()
                ),
            ));
            continue;
        }
        for member in binding.members() {
            let Some(target) = resolve_boundary_name(member.target()) else {
                diagnostics.push(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    binding_file,
                    member.range(),
                    format!(
                        "BoundarySet member `{}` is not an enclosing Cartesian Domain",
                        member.target()
                    ),
                ));
                continue;
            };
            exact_members.push(target.exact_identity.clone());
            resolved_members.push(ResolvedBoundaryMember {
                target: target.target,
                exact_identity: target.exact_identity,
                source_range: member.range(),
            });
        }
        if exact_members.len() != member_count {
            continue;
        }

        // Only unsupplied abstract signature members can leave a topology obligation.
        // An exact witness is never manufactured here; occurrence resolution is strict.
        if member_count > 0
            && binding
                .members()
                .iter()
                .all(|member| abstract_boundary(member.target()))
        {
            let mut unique = BTreeSet::new();
            for member in binding.members() {
                let Some(SpatialSupport::Boundary {
                    domain: identity,
                    parent,
                    dimensions: member_dimensions,
                }) = resolve_singular(member.target())
                else {
                    diagnostics.push(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        binding_file,
                        member.range(),
                        "abstract complete-exterior member requires an exact boundary support",
                    ));
                    continue;
                };
                if &parent != domain || member_dimensions != *dimensions {
                    diagnostics.push(source_error(codes::LANGUAGE_TYPE_ERROR, binding_file, member.range(), "abstract complete-exterior member does not have the exact symbolic parent and dimension"));
                }
                if !unique.insert(identity) {
                    diagnostics.push(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        binding_file,
                        member.range(),
                        "abstract complete-exterior contains an exact member more than once",
                    ));
                }
            }
            if dimensions.checked_mul(2) != Some(member_count) {
                diagnostics.push(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    binding_file,
                    binding.range(),
                    "abstract complete-exterior member count does not cover every Cartesian side",
                ));
            }
            deferred_topology.insert(binding.slot().to_owned());
            continue;
        }

        let exact_parent = domain.clone();
        let result =
            prove_complete_cartesian_exterior(exact_parent.clone(), exact_members, |identity| {
                if identity == &exact_parent {
                    Some(CartesianDomain::Volume {
                        ambient_dimension: *dimensions,
                    })
                } else {
                    resolve_cartesian_domain(identity)
                }
            });
        match result {
            Ok(witness) => {
                resolved_members.sort_unstable_by(|left, right| {
                    left.exact_identity().cmp(right.exact_identity())
                });
                boundary_sets.insert(
                    binding.slot().to_owned(),
                    ResolvedBoundarySet::Explicit(ExplicitBoundarySetBinding {
                        proved: Arc::new(ProvedBoundarySet {
                            members: resolved_members.into_boxed_slice(),
                            witness,
                        }),
                        source_range: binding.range(),
                    }),
                );
            }
            Err(error) => diagnostics.push(complete_exterior_proof_diagnostic(
                binding_file,
                binding.slot(),
                binding.range(),
                &resolved_members,
                error,
            )),
        }
    }

    for (name, slot) in interface.iter() {
        if slot.visibility() == VisibilitySyntax::Public && !actual.contains_key(name) {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                binding_file,
                instance.range(),
                format!(
                    "instance `{}` has no binding for required support slot `{name}`",
                    instance.name()
                ),
            ));
        }
    }
    for (name, contract) in interface.complete_exteriors() {
        if contract.visibility() == VisibilitySyntax::Public
            && !boundary_sets.contains_key(name)
            && !deferred_topology.contains(name)
        {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                binding_file,
                instance.range(),
                format!(
                    "instance `{}` has no binding for required complete-exterior support slot `{name}`",
                    instance.name()
                ),
            ));
        }
    }

    if diagnostics.is_empty() {
        Ok(ResolvedSupportBindings {
            singular_targets: targets,
            singular_supports: actual
                .into_iter()
                .map(|(slot, (support, _))| (slot, support))
                .collect(),
            boundary_sets,
        })
    } else {
        Err(diagnostics)
    }
}
