//! Occurrence-bound spatial-support interfaces for reusable Components.
//!
//! Support slots are definition-time obligations. They are resolved to exact
//! existing Domain identities before occurrence expansion and never become
//! Kernel entities, values, mesh handles, or inference rules.

mod bindings;
pub(super) use bindings::{resolve_definition_support_bindings, resolve_instance_support_bindings};
mod sets;
pub(super) use sets::{external_complete_exterior_set, symbolic_complete_exterior_set};

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_lang::{
    ComponentDecl, DomainSyntax, InstanceDecl, Item, ModelDecl, SignatureItem, SupportSlotSyntax,
    TextRange, VisibilitySyntax,
};
use eqiora_schema::kernel::{BoundarySide, typing::SpatialSupport};

use crate::diagnostics::source_error;

use super::CompleteExteriorLimits;
use super::complete_exterior::{
    CartesianDomain, CompleteExteriorError, CompleteExteriorWitness,
    prove_complete_cartesian_exterior,
};

#[derive(Clone, Debug)]
pub(super) struct SupportSlotContract {
    visibility: VisibilitySyntax,
    support: SpatialSupport<String>,
}

impl SupportSlotContract {
    pub(super) const fn visibility(&self) -> VisibilitySyntax {
        self.visibility
    }

    pub(super) const fn support(&self) -> &SpatialSupport<String> {
        &self.support
    }
}

/// Definition-time obligation for a complete Cartesian exterior.
///
/// The parent remains a slot name until occurrence binding. No inferred set,
/// geometry handle, or Kernel node is stored in a component interface.
#[derive(Clone, Debug)]
pub(super) struct CompleteExteriorSlotContract {
    visibility: VisibilitySyntax,
    parent_slot: String,
    ambient_dimension: usize,
}

impl CompleteExteriorSlotContract {
    pub(super) const fn visibility(&self) -> VisibilitySyntax {
        self.visibility
    }

    pub(super) fn parent_slot(&self) -> &str {
        &self.parent_slot
    }

    pub(super) const fn ambient_dimension(&self) -> usize {
        self.ambient_dimension
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct SupportInterface {
    slots: BTreeMap<String, SupportSlotContract>,
    complete_exteriors: BTreeMap<String, CompleteExteriorSlotContract>,
}

impl SupportInterface {
    pub(super) fn get(&self, name: &str) -> Option<&SupportSlotContract> {
        self.slots.get(name)
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (&str, &SupportSlotContract)> {
        self.slots
            .iter()
            .map(|(name, contract)| (name.as_str(), contract))
    }

    pub(super) fn complete_exterior(&self, name: &str) -> Option<&CompleteExteriorSlotContract> {
        self.complete_exteriors.get(name)
    }

    pub(super) fn complete_exteriors(
        &self,
    ) -> impl Iterator<Item = (&str, &CompleteExteriorSlotContract)> {
        self.complete_exteriors
            .iter()
            .map(|(name, contract)| (name.as_str(), contract))
    }

    pub(super) fn visible_support(&self, name: &str) -> Option<&SpatialSupport<String>> {
        self.slots.get(name).and_then(|contract| {
            (contract.visibility == VisibilitySyntax::Public).then_some(&contract.support)
        })
    }
}

pub(super) fn component_support_interface(
    file: &str,
    component: &ComponentDecl,
) -> Result<SupportInterface, Vec<Diagnostic>> {
    signature_support_interface(file, component.signature())
}

pub(super) fn signature_support_interface(
    file: &str,
    signature: &[SignatureItem],
) -> Result<SupportInterface, Vec<Diagnostic>> {
    let declarations = signature
        .iter()
        .filter_map(|item| match item {
            SignatureItem::Support(declaration) => {
                Some((declaration.name().to_owned(), declaration))
            }
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    let mut slots = BTreeMap::new();
    let mut complete_exteriors = BTreeMap::new();
    let mut diagnostics = Vec::new();

    for (name, declaration) in &declarations {
        if declaration.visibility() != VisibilitySyntax::Public {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                declaration.range(),
                format!(
                    "support slot `{name}` must be public; private occurrence support is not representable in v1"
                ),
            ));
            continue;
        }
        if let SupportSlotSyntax::Interval { dimension } = declaration.syntax() {
            match crate::dimensions::lower_dimension(file, dimension) {
                Ok(unit) => {
                    slots.insert(
                        name.clone(),
                        SupportSlotContract {
                            visibility: declaration.visibility(),
                            support: SpatialSupport::Coordinates {
                                domain: name.clone(),
                                factors: vec![(name.clone(), unit, 1)],
                            },
                        },
                    );
                }
                Err(error) => diagnostics.push(error),
            }
            continue;
        }
        let SupportSlotSyntax::Volume { ambient_dimension } = declaration.syntax() else {
            continue;
        };
        if *ambient_dimension == 0 {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                declaration.range(),
                format!("volume support slot `{name}` requires a positive ambient dimension"),
            ));
            continue;
        }
        slots.insert(
            name.clone(),
            SupportSlotContract {
                visibility: declaration.visibility(),
                support: SpatialSupport::Volume {
                    domain: name.clone(),
                    dimensions: *ambient_dimension,
                },
            },
        );
    }

    for (name, declaration) in &declarations {
        if declaration.visibility() != VisibilitySyntax::Public {
            continue;
        }
        let SupportSlotSyntax::Boundary { parent } = declaration.syntax() else {
            continue;
        };
        let Some(parent_contract) = slots.get(parent) else {
            let message = match declarations.get(parent) {
                Some(_) => {
                    format!("boundary support slot `{name}` requires volume parent slot `{parent}`")
                }
                None => format!(
                    "boundary support slot `{name}` refers to unknown parent slot `{parent}`"
                ),
            };
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                declaration.range(),
                message,
            ));
            continue;
        };
        let SpatialSupport::Volume { dimensions, .. } = parent_contract.support() else {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                declaration.range(),
                format!("boundary support slot `{name}` requires volume parent slot `{parent}`"),
            ));
            continue;
        };
        if declaration.visibility() == VisibilitySyntax::Public
            && parent_contract.visibility() != VisibilitySyntax::Public
        {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                declaration.range(),
                format!(
                    "public boundary support slot `{name}` requires public parent slot `{parent}`"
                ),
            ));
            continue;
        }
        slots.insert(
            name.clone(),
            SupportSlotContract {
                visibility: declaration.visibility(),
                support: SpatialSupport::Boundary {
                    domain: name.clone(),
                    parent: parent.clone(),
                    dimensions: *dimensions,
                },
            },
        );
    }

    for (name, declaration) in &declarations {
        if declaration.visibility() != VisibilitySyntax::Public {
            continue;
        }
        let SupportSlotSyntax::CompleteExterior { parent } = declaration.syntax() else {
            continue;
        };
        let Some(parent_contract) = slots.get(parent) else {
            let message = match declarations.get(parent) {
                Some(_) => format!(
                    "complete-exterior support slot `{name}` requires volume parent slot `{parent}`"
                ),
                None => format!(
                    "complete-exterior support slot `{name}` refers to unknown parent slot `{parent}`"
                ),
            };
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                declaration.range(),
                message,
            ));
            continue;
        };
        let SpatialSupport::Volume { dimensions, .. } = parent_contract.support() else {
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                declaration.range(),
                format!(
                    "complete-exterior support slot `{name}` requires volume parent slot `{parent}`"
                ),
            ));
            continue;
        };
        complete_exteriors.insert(
            name.clone(),
            CompleteExteriorSlotContract {
                visibility: declaration.visibility(),
                parent_slot: parent.clone(),
                ambient_dimension: *dimensions,
            },
        );
    }

    if diagnostics.is_empty() {
        Ok(SupportInterface {
            slots,
            complete_exteriors,
        })
    } else {
        Err(diagnostics)
    }
}

mod spatial;
pub(super) use spatial::{component_spatial_supports, model_spatial_supports};

/// Per-elaboration accounting for explicit complete-exterior memberships.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CompleteExteriorMembershipBudget {
    limits: CompleteExteriorLimits,
    total_memberships: usize,
}

impl CompleteExteriorMembershipBudget {
    pub(super) const fn new(limits: CompleteExteriorLimits) -> Self {
        Self {
            limits,
            total_memberships: 0,
        }
    }

    #[cfg(test)]
    pub(super) const fn total_memberships(&self) -> usize {
        self.total_memberships
    }

    fn charge(&mut self, count: usize) -> Result<(), CompleteExteriorBudgetError> {
        if count > self.limits.max_members_per_set {
            return Err(CompleteExteriorBudgetError::SetTooLarge {
                count,
                limit: self.limits.max_members_per_set,
            });
        }
        let total = self
            .total_memberships
            .checked_add(count)
            .ok_or(CompleteExteriorBudgetError::TotalOverflow)?;
        if total > self.limits.max_total_memberships {
            return Err(CompleteExteriorBudgetError::TotalExceeded {
                total,
                limit: self.limits.max_total_memberships,
            });
        }
        self.total_memberships = total;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CompleteExteriorBudgetError {
    SetTooLarge { count: usize, limit: usize },
    TotalOverflow,
    TotalExceeded { total: usize, limit: usize },
}

/// One enclosing-scope boundary locator resolved to an exact identity.
///
/// Flattening supplies a stable internal symbol locator when one is available;
/// symbolic definition checking may use the enclosing lexical name.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ResolvedBoundaryTarget<I> {
    target: String,
    exact_identity: I,
}

impl<I> ResolvedBoundaryTarget<I> {
    pub(super) fn new(target: String, exact_identity: I) -> Self {
        Self {
            target,
            exact_identity,
        }
    }
}

/// One explicit source member resolved to its stable locator and exact
/// elaboration identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ResolvedBoundaryMember<I> {
    target: String,
    exact_identity: I,
    source_range: TextRange,
}

impl<I> ResolvedBoundaryMember<I> {
    pub(super) fn target(&self) -> &str {
        &self.target
    }

    pub(super) const fn exact_identity(&self) -> &I {
        &self.exact_identity
    }

    pub(super) const fn source_range(&self) -> TextRange {
        self.source_range
    }
}

/// Shared, immutable result of one complete-exterior proof.
///
/// Forwarding shares this exact catalog rather than copying or re-proving it.
/// Members are sorted by exact identity; position has no semantic meaning.
#[derive(Clone, Debug, Eq, PartialEq)]
struct ProvedBoundarySet<I> {
    members: Box<[ResolvedBoundaryMember<I>]>,
    witness: CompleteExteriorWitness<I>,
}

impl<I: Ord> ProvedBoundarySet<I> {
    fn member(&self, exact_identity: &I) -> Option<&ResolvedBoundaryMember<I>> {
        self.members
            .binary_search_by(|member| member.exact_identity().cmp(exact_identity))
            .ok()
            .map(|index| &self.members[index])
    }
}

/// A source `boundaries(...)` binding and its geometric proof.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ExplicitBoundarySetBinding<I> {
    proved: Arc<ProvedBoundarySet<I>>,
    source_range: TextRange,
}

impl<I: Ord> ExplicitBoundarySetBinding<I> {
    /// Members in exact-identity order. Position is not semantic.
    #[cfg(test)]
    pub(super) fn members(&self) -> &[ResolvedBoundaryMember<I>] {
        &self.proved.members
    }

    pub(super) fn member(&self, exact_identity: &I) -> Option<&ResolvedBoundaryMember<I>> {
        self.proved.member(exact_identity)
    }

    /// Canonical axis-major side bijection consumed by flattening.
    pub(super) fn witness(&self) -> &CompleteExteriorWitness<I> {
        &self.proved.witness
    }
}

/// A binding that forwards an already proved enclosing complete exterior.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ForwardedBoundarySetBinding<I> {
    target: String,
    proved: Arc<ProvedBoundarySet<I>>,
    source_range: TextRange,
}

impl<I: Ord> ForwardedBoundarySetBinding<I> {
    #[cfg(test)]
    pub(super) fn target(&self) -> &str {
        &self.target
    }

    pub(super) fn witness(&self) -> &CompleteExteriorWitness<I> {
        &self.proved.witness
    }

    #[cfg(test)]
    pub(super) fn members(&self) -> &[ResolvedBoundaryMember<I>] {
        &self.proved.members
    }

    pub(super) fn member(&self, exact_identity: &I) -> Option<&ResolvedBoundaryMember<I>> {
        self.proved.member(exact_identity)
    }
}

/// The two intentionally distinct ways an occurrence can satisfy a set slot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum ResolvedBoundarySet<I> {
    Explicit(ExplicitBoundarySetBinding<I>),
    Forwarded(ForwardedBoundarySetBinding<I>),
}

impl<I: Ord> ResolvedBoundarySet<I> {
    pub(super) fn witness(&self) -> &CompleteExteriorWitness<I> {
        match self {
            Self::Explicit(binding) => binding.witness(),
            Self::Forwarded(binding) => binding.witness(),
        }
    }

    /// Exact-identity keyed locator records; slice order is non-semantic.
    #[cfg(test)]
    pub(super) fn members(&self) -> &[ResolvedBoundaryMember<I>] {
        match self {
            Self::Explicit(binding) => binding.members(),
            Self::Forwarded(binding) => binding.members(),
        }
    }

    pub(super) fn member(&self, exact_identity: &I) -> Option<&ResolvedBoundaryMember<I>> {
        match self {
            Self::Explicit(binding) => binding.member(exact_identity),
            Self::Forwarded(binding) => binding.member(exact_identity),
        }
    }

    /// Source range of the explicit or forwarded binding that established
    /// this occurrence's complete-exterior proof.
    pub(super) const fn source_range(&self) -> TextRange {
        match self {
            Self::Explicit(binding) => binding.source_range,
            Self::Forwarded(binding) => binding.source_range,
        }
    }

    fn proved(&self) -> &Arc<ProvedBoundarySet<I>> {
        match self {
            Self::Explicit(binding) => &binding.proved,
            Self::Forwarded(binding) => &binding.proved,
        }
    }
}

/// Fully checked occurrence support bindings.
///
/// Lexical singular targets are retained for the existing symbol-forwarding
/// path. Exact singular supports and proved sets are the typed inputs for new
/// boundary-family flattening.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ResolvedSupportBindings<I> {
    singular_targets: BTreeMap<String, String>,
    singular_supports: BTreeMap<String, SpatialSupport<I>>,
    boundary_sets: BTreeMap<String, ResolvedBoundarySet<I>>,
}

impl<I: Ord> ResolvedSupportBindings<I> {
    pub(super) const fn singular_targets(&self) -> &BTreeMap<String, String> {
        &self.singular_targets
    }

    pub(super) const fn singular_supports(&self) -> &BTreeMap<String, SpatialSupport<I>> {
        &self.singular_supports
    }

    pub(super) fn boundary_set(&self, slot: &str) -> Option<&ResolvedBoundarySet<I>> {
        self.boundary_sets.get(slot)
    }

    pub(super) fn boundary_sets(&self) -> impl Iterator<Item = (&str, &ResolvedBoundarySet<I>)> {
        self.boundary_sets
            .iter()
            .map(|(slot, set)| (slot.as_str(), set))
    }

    #[cfg(test)]
    fn into_singular_targets(self) -> BTreeMap<String, String> {
        self.singular_targets
    }
}

/// Compatibility surface for singular occurrence supports.
///
/// The richer resolver is the sole implementation. Existing callers retain
/// their lexical target map until boundary-family flattening consumes the
/// typed result directly.
#[cfg(test)]
pub(super) fn resolve_instance_supports<I: Clone + Ord>(
    binding_file: &str,
    component: &ComponentDecl,
    interface: &SupportInterface,
    instance: &InstanceDecl,
    resolve_singular: impl FnMut(&str) -> Option<SpatialSupport<I>>,
) -> Result<BTreeMap<String, String>, Vec<Diagnostic>> {
    let mut membership_budget =
        CompleteExteriorMembershipBudget::new(CompleteExteriorLimits::default());
    resolve_instance_support_bindings(
        binding_file,
        component,
        interface,
        instance,
        resolve_singular,
        |_| None,
        |_| None,
        |_| None,
        &mut membership_budget,
    )
    .map(ResolvedSupportBindings::into_singular_targets)
}

fn validate_singular_support_shapes<I: Eq>(
    binding_file: &str,
    interface: &SupportInterface,
    actual: &BTreeMap<String, (SpatialSupport<I>, TextRange)>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for (name, slot) in interface.iter() {
        let Some((bound, range)) = actual.get(name) else {
            continue;
        };
        match (slot.support(), bound) {
            (
                SpatialSupport::Coordinates {
                    factors: expected, ..
                },
                SpatialSupport::Coordinates { domain, factors },
            ) if expected.len() == 1
                && factors.len() == 1
                && &factors[0].0 == domain
                && expected[0].1 == factors[0].1
                && factors[0].2 == 1 => {}
            (SpatialSupport::Coordinates { .. }, _) | (_, SpatialSupport::Coordinates { .. }) => {
                diagnostics.push(source_error(codes::LANGUAGE_TYPE_ERROR, binding_file, *range,
                    format!("support slot `{name}` requires its declared factor kind and coordinate units")));
            }
            (
                SpatialSupport::Volume {
                    dimensions: expected,
                    ..
                },
                SpatialSupport::Volume {
                    dimensions: actual, ..
                },
            ) if expected == actual => {}
            (SpatialSupport::Volume { dimensions, .. }, SpatialSupport::Volume { .. }) => {
                diagnostics.push(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    binding_file,
                    *range,
                    format!("volume support slot `{name}` requires ambient dimension {dimensions}"),
                ));
            }
            (SpatialSupport::Volume { .. }, SpatialSupport::Boundary { .. }) => {
                diagnostics.push(kind_mismatch(
                    binding_file,
                    *range,
                    name,
                    "volume",
                    "boundary",
                ));
            }
            (
                SpatialSupport::Boundary {
                    parent: parent_slot,
                    dimensions: expected,
                    ..
                },
                SpatialSupport::Boundary {
                    parent: actual_parent,
                    dimensions: actual_dimensions,
                    ..
                },
            ) => {
                let expected_parent =
                    actual
                        .get(parent_slot)
                        .and_then(|(support, _)| match support {
                            SpatialSupport::Volume { domain, .. } => Some(domain),
                            SpatialSupport::Coordinates { .. }
                            | SpatialSupport::Boundary { .. }
                            | SpatialSupport::Interface { .. }
                            | SpatialSupport::PhysicalInterface { .. } => None,
                        });
                if expected != actual_dimensions {
                    diagnostics.push(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        binding_file,
                        *range,
                        format!(
                            "boundary support slot `{name}` requires ambient dimension {expected}"
                        ),
                    ));
                } else if expected_parent != Some(actual_parent) {
                    diagnostics.push(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        binding_file,
                        *range,
                        format!(
                            "boundary support slot `{name}` is not BoundaryOf its exact bound parent slot `{parent_slot}`"
                        ),
                    ));
                }
            }
            (SpatialSupport::Boundary { .. }, SpatialSupport::Volume { .. }) => {
                diagnostics.push(kind_mismatch(
                    binding_file,
                    *range,
                    name,
                    "boundary",
                    "volume",
                ));
            }
            (SpatialSupport::Interface { .. } | SpatialSupport::PhysicalInterface { .. }, _)
            | (_, SpatialSupport::Interface { .. } | SpatialSupport::PhysicalInterface { .. }) => {
                diagnostics.push(source_error(
                    codes::LANGUAGE_LOWERING_ERROR,
                    binding_file,
                    *range,
                    "derived interface support cannot be bound as a source support slot",
                ));
            }
        }
    }
}

fn complete_exterior_budget_diagnostic(
    file: &str,
    binding: &super::named_bindings::BoundarySetBinding<'_>,
    error: CompleteExteriorBudgetError,
) -> Diagnostic {
    let message = match error {
        CompleteExteriorBudgetError::SetTooLarge { count, limit } => format!(
            "complete-exterior binding for slot `{}` has {count} members, exceeding the {limit} member limit",
            binding.slot()
        ),
        CompleteExteriorBudgetError::TotalOverflow => {
            "complete-exterior membership count overflows usize".to_owned()
        }
        CompleteExteriorBudgetError::TotalExceeded { total, limit } => format!(
            "complete-exterior bindings require {total} total memberships, exceeding the {limit} membership limit"
        ),
    };
    source_error(
        codes::LANGUAGE_LOWERING_ERROR,
        file,
        binding.range(),
        message,
    )
}

fn complete_exterior_proof_diagnostic<I: Eq>(
    file: &str,
    slot: &str,
    binding_range: TextRange,
    members: &[ResolvedBoundaryMember<I>],
    error: CompleteExteriorError<I>,
) -> Diagnostic {
    let range = complete_exterior_error_range(&error, members).unwrap_or(binding_range);
    let (code, message) = match &error {
        CompleteExteriorError::Empty => (
            codes::LANGUAGE_TYPE_ERROR,
            format!("complete-exterior slot `{slot}` requires a nonempty BoundarySet"),
        ),
        CompleteExteriorError::DuplicateExactMember { .. } => (
            codes::LANGUAGE_TYPE_ERROR,
            format!("complete-exterior slot `{slot}` contains one exact Boundary more than once"),
        ),
        CompleteExteriorError::UnknownDomain { .. } => (
            codes::LANGUAGE_TYPE_ERROR,
            format!(
                "complete-exterior slot `{slot}` contains unresolved Cartesian Domain metadata"
            ),
        ),
        CompleteExteriorError::ExactParentIsNotVolume { .. } => (
            codes::LANGUAGE_TYPE_ERROR,
            format!("complete-exterior slot `{slot}` does not have an exact volume parent"),
        ),
        CompleteExteriorError::ExactParentHasZeroDimension { .. } => (
            codes::LANGUAGE_TYPE_ERROR,
            format!("complete-exterior slot `{slot}` has a zero-dimensional parent"),
        ),
        CompleteExteriorError::MemberIsVolume { .. } => (
            codes::LANGUAGE_TYPE_ERROR,
            format!("complete-exterior slot `{slot}` contains a volume instead of a Boundary"),
        ),
        CompleteExteriorError::BoundaryOfBoundary { .. } => (
            codes::LANGUAGE_TYPE_ERROR,
            format!("complete-exterior slot `{slot}` contains a boundary-of-boundary"),
        ),
        CompleteExteriorError::WrongParent { .. } => (
            codes::LANGUAGE_TYPE_ERROR,
            format!(
                "complete-exterior slot `{slot}` contains a Boundary of a different exact parent"
            ),
        ),
        CompleteExteriorError::WrongDimension {
            expected, actual, ..
        } => (
            codes::LANGUAGE_TYPE_ERROR,
            format!(
                "complete-exterior slot `{slot}` requires ambient dimension {expected}, found {actual}"
            ),
        ),
        CompleteExteriorError::AxisOutsideParentDimension {
            axis,
            ambient_dimension,
            ..
        } => (
            codes::LANGUAGE_TYPE_ERROR,
            format!(
                "complete-exterior slot `{slot}` contains axis {axis}, outside parent dimension {ambient_dimension}"
            ),
        ),
        CompleteExteriorError::DuplicateGeometry { axis, side, .. } => (
            codes::LANGUAGE_TYPE_ERROR,
            format!(
                "complete-exterior slot `{slot}` contains Cartesian side ({axis}, {}) more than once",
                boundary_side_name(*side)
            ),
        ),
        CompleteExteriorError::MissingSide { axis, side } => (
            codes::LANGUAGE_TYPE_ERROR,
            format!(
                "complete-exterior slot `{slot}` is missing Cartesian side ({axis}, {})",
                boundary_side_name(*side)
            ),
        ),
        CompleteExteriorError::SideCountOverflow { ambient_dimension } => (
            codes::LANGUAGE_LOWERING_ERROR,
            format!(
                "complete-exterior side count overflows usize for dimension {ambient_dimension}"
            ),
        ),
        CompleteExteriorError::Allocation { resource, count } => (
            codes::LANGUAGE_LOWERING_ERROR,
            format!("cannot reserve {count} entries for {resource}"),
        ),
    };
    source_error(code, file, range, message)
}

fn complete_exterior_error_range<I: Eq>(
    error: &CompleteExteriorError<I>,
    members: &[ResolvedBoundaryMember<I>],
) -> Option<TextRange> {
    let identity = match error {
        CompleteExteriorError::DuplicateExactMember { member }
        | CompleteExteriorError::MemberIsVolume { member }
        | CompleteExteriorError::BoundaryOfBoundary { member, .. }
        | CompleteExteriorError::WrongParent { member, .. }
        | CompleteExteriorError::WrongDimension { member, .. }
        | CompleteExteriorError::AxisOutsideParentDimension { member, .. } => Some(member),
        CompleteExteriorError::DuplicateGeometry { duplicate, .. } => Some(duplicate),
        CompleteExteriorError::UnknownDomain { domain } => Some(domain),
        CompleteExteriorError::Empty
        | CompleteExteriorError::ExactParentIsNotVolume { .. }
        | CompleteExteriorError::ExactParentHasZeroDimension { .. }
        | CompleteExteriorError::MissingSide { .. }
        | CompleteExteriorError::SideCountOverflow { .. }
        | CompleteExteriorError::Allocation { .. } => None,
    };
    identity.and_then(|identity| {
        members
            .iter()
            .rev()
            .find(|member| member.exact_identity() == identity)
            .map(ResolvedBoundaryMember::source_range)
    })
}

const fn boundary_side_name(side: BoundarySide) -> &'static str {
    match side {
        BoundarySide::Lower => "lower",
        BoundarySide::Upper => "upper",
    }
}

fn kind_mismatch(
    file: &str,
    range: TextRange,
    slot: &str,
    expected: &str,
    actual: &str,
) -> Diagnostic {
    source_error(
        codes::LANGUAGE_TYPE_ERROR,
        file,
        range,
        format!("support slot `{slot}` requires {expected} support, found {actual}"),
    )
}

#[cfg(test)]
mod tests;
