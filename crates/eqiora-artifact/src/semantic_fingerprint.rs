//! Alpha-normalized structural identity of one accepted Semantic Model.
//!
//! Exact Model artifacts retain occurrence ULIDs and remain authoritative for
//! replay, provenance, and mutation.  This module instead constructs a closed,
//! versioned projection of the accepted kernel graph, canonically labels that
//! graph without consulting occurrence IDs, and hashes the resulting bytes.

mod canonical;
mod expression;
use expression::{canonical_expr_id, encode_expression};
mod projection;
mod property;
mod relation;
mod values;

use core::fmt;
use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, RawId, ValueLiteral};
use eqiora_graph::EdgeKind;
use eqiora_schema::kernel::{
    ActivationKind, BoundaryPairing, BoundarySide, CartesianCoordinateSource, ClockKind,
    ConnectionSemantics, DomainKind, EventDirection, ExprDag, ExprNode, KernelNode, PortPayload,
    RelationConditionKind, RelationMeaning, RepresentationKind, SignalDirection, SymbolRef,
    UnaryMathFunction,
};
use eqiora_sem::KernelProgram;
use sha2::{Digest, Sha256};

use crate::{ArtifactDigest, invalid_artifact};
use canonical::{Canonicalizer, Encoder};
use projection::{ConstructionBudget, ProjectionGraph, Reference};
use values::{
    encode_literal, encode_optional_literal, encode_quantity, encode_value_type, type_reference,
};

const FINGERPRINT_DOMAIN_V25: &[u8] = b"eqiora.structural-semantic-fingerprint/v25\0";
const PROJECTION_MAGIC: &[u8; 8] = b"EQIORASF";
const GENERATION_V25: u16 = 25;

/// Current generation of the structural semantic projection.
///
/// Generations are intentionally independent of Model artifact wire versions.
/// Equality is defined only within one explicitly equal generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum SemanticFingerprintGeneration {
    /// Closed projection retaining Boolean and exact integer payloads, nominal references,
    /// ordered equation sides, comparisons and finite extrema, initialization,
    /// sample/hold transitions, typed operators, conditional value guards, and
    /// nominal records with ordered heterogeneous member expressions, and typed
    /// Observables with exact expression and reduction support, analytic/table
    /// property derivative profiles, and exclusive equality, inequality,
    /// complementarity and conservation relation meaning, exact event priorities,
    /// finite coordinate duality and ordered linear-map basis references, and
    /// ordered partial sources, selected local bindings and checked derivative values.
    V25,
}

impl SemanticFingerprintGeneration {
    /// Stable external spelling of this comparison generation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::V25 => "eqiora.structural-semantic-fingerprint/v25",
        }
    }

    const fn code(self) -> u16 {
        match self {
            Self::V25 => GENERATION_V25,
        }
    }

    const fn hash_domain(self) -> &'static [u8] {
        match self {
            Self::V25 => FINGERPRINT_DOMAIN_V25,
        }
    }
}

/// Comparison/cache evidence for one alpha-normalized Semantic Model graph.
///
/// This value is deliberately not a Model artifact identity.  It cannot be
/// used as an execution input, replay key, provenance reference, or mutation
/// precondition.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StructuralSemanticFingerprint {
    generation: SemanticFingerprintGeneration,
    digest: ArtifactDigest,
}

impl StructuralSemanticFingerprint {
    /// Construct the current bounded structural projection.
    ///
    /// # Errors
    /// Returns `EQ0901` when the program contains unsupported vocabulary or
    /// exact canonical labeling exceeds the selected generation's fixed
    /// resource policy.
    pub fn from_program(program: &KernelProgram) -> Result<Self, Diagnostic> {
        ProjectionIdentity::from_program(program, SemanticFingerprintLimits::default())
            .map(|identity| identity.fingerprint)
    }

    /// Construct with an explicit bounded canonicalization policy.
    ///
    /// Limits affect admission only. Every accepted construction produces
    /// exactly the same bytes and digest for its selected generation.
    ///
    /// # Errors
    /// Returns `EQ0901` for unsupported meaning or exhausted limits.
    #[cfg(test)]
    fn from_program_with_limits(
        program: &KernelProgram,
        limits: SemanticFingerprintLimits,
    ) -> Result<Self, Diagnostic> {
        ProjectionIdentity::from_program(program, limits).map(|identity| identity.fingerprint)
    }

    /// Exact structural comparison generation.
    #[must_use]
    pub const fn generation(&self) -> SemanticFingerprintGeneration {
        self.generation
    }

    /// Hexadecimal domain-separated SHA-256 of the closed canonical projection.
    ///
    /// The view deliberately does not expose [`ArtifactDigest`], which is an
    /// authority-bearing input to artifact and Run lineage constructors.
    #[must_use]
    pub fn digest(&self) -> &str {
        self.digest.as_str()
    }
}

impl fmt::Display for StructuralSemanticFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}", self.generation.as_str(), self.digest)
    }
}

/// Resource policy for exact graph canonicalization.
///
/// The algorithm never falls back to occurrence ordering or a probabilistic
/// refinement.  A pathological symmetry that exceeds these limits is rejected
/// instead of producing a route-dependent fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SemanticFingerprintLimits {
    /// Maximum kernel vertices in one selected Model.
    max_nodes: usize,
    /// Maximum graph edges plus nominal and expression references.
    max_references: usize,
    /// Maximum expression nodes summed across Relations and Activations.
    max_expression_nodes: usize,
    /// Maximum bytes in one canonical projection or intermediate label set.
    max_canonical_bytes: usize,
    /// Maximum individualization/refinement search states.
    max_search_states: usize,
    /// Maximum recursive individualization depth.
    max_individualization_depth: usize,
    /// Maximum vertex/reference visits across refinement rounds.
    max_refinement_work: usize,
    /// Maximum canonical bytes produced across every discrete search leaf.
    max_serialization_work: usize,
}

impl Default for SemanticFingerprintLimits {
    fn default() -> Self {
        Self {
            max_nodes: 100_000,
            max_references: 1_000_000,
            max_expression_nodes: 1_000_000,
            max_canonical_bytes: 128 * 1_024 * 1_024,
            max_search_states: 100_000,
            max_individualization_depth: 256,
            max_refinement_work: 100_000_000,
            max_serialization_work: 512 * 1_024 * 1_024,
        }
    }
}

/// Compare two programs through the same closed canonical projection.
///
/// Unlike comparing only fingerprints, this bounded consumer also compares
/// canonical bytes after equal digests.  A cryptographic collision therefore
/// fails closed rather than being reported as semantic equality.
///
/// # Errors
/// Returns `EQ0901` for unsupported meaning, exhausted construction limits,
/// or a digest collision between unequal canonical projections.
pub fn structurally_equivalent(
    left: &KernelProgram,
    right: &KernelProgram,
) -> Result<bool, Diagnostic> {
    let limits = SemanticFingerprintLimits::default();
    let left = ProjectionIdentity::from_program(left, limits)?;
    let right = ProjectionIdentity::from_program(right, limits)?;
    if left.fingerprint != right.fingerprint {
        return Ok(false);
    }
    if left.canonical != right.canonical {
        return Err(fingerprint_error(
            "structural semantic fingerprint collision between unequal canonical projections",
        ));
    }
    Ok(true)
}

struct ProjectionIdentity {
    fingerprint: StructuralSemanticFingerprint,
    canonical: Vec<u8>,
}

impl ProjectionIdentity {
    fn from_program(
        program: &KernelProgram,
        limits: SemanticFingerprintLimits,
    ) -> Result<Self, Diagnostic> {
        validate_limits(limits)?;
        let generation = SemanticFingerprintGeneration::V25;
        let graph = ProjectionGraph::from_program(program, limits)?;
        let canonical = Canonicalizer::new(&graph, limits).canonicalize()?;
        let mut hasher = Sha256::new();
        hasher.update(generation.hash_domain());
        hasher.update(&canonical);
        let digest = ArtifactDigest::from_sha256(hasher.finalize().into());
        Ok(Self {
            fingerprint: StructuralSemanticFingerprint { generation, digest },
            canonical,
        })
    }
}

fn encode_node(
    node: &KernelNode,
    current_value: Option<&ValueLiteral>,
    boundary: bool,
    ids: &BTreeMap<RawId, usize>,
    references: &mut Vec<Reference>,
    budget: &mut ConstructionBudget,
) -> Result<Vec<u8>, Diagnostic> {
    let mut encoder = Encoder::new(budget.limits.max_canonical_bytes);
    match node {
        KernelNode::Record(definition) => {
            encoder.u8(13)?;
            encoder.len(definition.members().len())?;
            for (index, (name, ty)) in definition.members().iter().enumerate() {
                encoder.bytes(name.as_bytes())?;
                encode_value_type(&mut encoder, ty)?;
                let mut label = Encoder::new(16);
                label.u8(5)?;
                label.len(index)?;
                type_reference(ty, label.finish()?, ids, references, budget)?;
            }
        }
        KernelNode::RecordInstance(instance) => {
            encoder.u8(14)?;
            push_reference(
                references,
                vec![6],
                lookup(ids, instance.definition().erase(), "record declaration")?,
                budget,
            )?;
            encode_expression(
                &mut encoder,
                instance.expression(),
                &[],
                4,
                ids,
                references,
                budget,
            )?;
        }
        KernelNode::Observable(definition) => {
            encoder.u8(15)?;
            encode_value_type(&mut encoder, definition.value_type())?;
            type_reference(
                definition.value_type(),
                nominal_label(15),
                ids,
                references,
                budget,
            )?;
            match definition.reduction() {
                eqiora_schema::kernel::ObservableReduction::Value => encoder.u8(0)?,
                eqiora_schema::kernel::ObservableReduction::SpatialIntegral { domain, measure } => {
                    encoder.u8(match measure {
                        eqiora_schema::kernel::ObservableMeasure::Volume => 1,
                        eqiora_schema::kernel::ObservableMeasure::Boundary => 2,
                    })?;
                    push_reference(
                        references,
                        nominal_label(16),
                        lookup(ids, domain.erase(), "Observable integration Domain")?,
                        budget,
                    )?;
                }
            }
            encode_expression(
                &mut encoder,
                definition.expression(),
                &[],
                4,
                ids,
                references,
                budget,
            )?;
        }
        KernelNode::Enum(definition) => {
            encoder.u8(12)?;
            encoder.len(definition.members().len())?;
            for member in definition.members() {
                encoder.bytes(member.as_bytes())?;
            }
        }
        KernelNode::FiniteSpace(space) => {
            encoder.u8(10)?;
            if let Some(factors) = space.factors() {
                encoder.u8(2)?;
                for (position, factor) in factors.into_iter().enumerate() {
                    encoder.u32(factor.extent())?;
                    push_reference(
                        references,
                        vec![14, position as u8],
                        lookup(
                            ids,
                            factor.space().expect("atomic factor").erase(),
                            "product factor",
                        )?,
                        budget,
                    )?;
                }
            } else {
                encoder.u8(1)?;
                let labels = space.labels().expect("atomic labels");
                encoder.len(labels.len())?;
                for label in labels {
                    encoder.bytes(label.as_bytes())?;
                }
            }
        }
        KernelNode::IndexSet(set) => {
            encoder.u8(11)?;
            encoder.u32(set.extent())?;
        }
        KernelNode::Domain(domain) => {
            encoder.u8(1)?;
            encode_domain_kind(&mut encoder, domain.kind(), ids, references, budget)?;
        }
        KernelNode::Representation(representation) => {
            encoder.u8(2)?;
            match representation.kind() {
                RepresentationKind::Abstract => encoder.u8(1)?,
                RepresentationKind::Continuum => encoder.u8(2)?,
                _ => return Err(newer_vocabulary("Representation kind")),
            }
        }
        KernelNode::Field(field) => {
            encoder.u8(3)?;
            encode_value_type(&mut encoder, field.value_type())?;
            type_reference(
                field.value_type(),
                nominal_label(5),
                ids,
                references,
                budget,
            )?;
            encoder.u8(match field.role() {
                eqiora_schema::kernel::FieldRole::Variable => 0,
                eqiora_schema::kernel::FieldRole::State => 1,
            })?;
        }
        KernelNode::Parameter(parameter) => {
            encoder.u8(4)?;
            encode_literal(&mut encoder, parameter.value())?;
            type_reference(
                parameter.value().value_type(),
                nominal_label(6),
                ids,
                references,
                budget,
            )?;
        }
        KernelNode::Port(port) => {
            encoder.u8(5)?;
            match port.payload() {
                PortPayload::Signal {
                    direction,
                    value_type,
                } => {
                    encoder.u8(1)?;
                    encode_signal_direction(&mut encoder, direction)?;
                    encode_value_type(&mut encoder, &value_type)?;
                    type_reference(&value_type, nominal_label(7), ids, references, budget)?;
                }
                PortPayload::ScalarPhysical { domain } => {
                    encoder.u8(3)?;
                    push_reference(
                        references,
                        nominal_label(1),
                        lookup(ids, domain.erase(), "scalar physical Port Domain")?,
                        budget,
                    )?;
                }
                PortPayload::BoundaryPhysical {
                    connector,
                    boundary,
                } => {
                    encoder.u8(4)?;
                    push_reference(
                        references,
                        nominal_label(2),
                        lookup(ids, connector.erase(), "boundary Port connector")?,
                        budget,
                    )?;
                    push_reference(
                        references,
                        nominal_label(3),
                        lookup(ids, boundary.erase(), "boundary Port support")?,
                        budget,
                    )?;
                }
                _ => return Err(newer_vocabulary("Port payload")),
            }
        }
        KernelNode::Relation(relation) => {
            relation::encode(relation, &mut encoder, ids, references, budget)?;
        }
        KernelNode::Activation(activation) => {
            encoder.u8(7)?;
            match activation.kind() {
                ActivationKind::Continuous => encoder.u8(1)?,
                ActivationKind::Periodic => encoder.u8(2)?,
                ActivationKind::Event {
                    guard,
                    direction,
                    priority,
                } => {
                    encoder.u8(3)?;
                    encode_event_direction(&mut encoder, *direction)?;
                    encoder.raw(&priority.to_be_bytes())?;
                    encode_expression(&mut encoder, guard, &[], 2, ids, references, budget)?;
                }
                ActivationKind::Guard { guard } => {
                    encoder.u8(4)?;
                    encode_expression(&mut encoder, guard, &[], 3, ids, references, budget)?;
                }
                _ => return Err(newer_vocabulary("Activation kind")),
            }
        }
        KernelNode::Connection(connection) => {
            encoder.u8(8)?;
            match connection.semantics() {
                ConnectionSemantics::Signal { driver } => {
                    encoder.u8(1)?;
                    push_reference(
                        references,
                        nominal_label(4),
                        lookup(ids, driver.erase(), "signal Connection driver Port")?,
                        budget,
                    )?;
                }
                ConnectionSemantics::Conserving => encoder.u8(2)?,
                ConnectionSemantics::SpatialPeriodic => encoder.u8(3)?,
                _ => return Err(newer_vocabulary("Connection semantics")),
            }
        }
        KernelNode::ClockDomain(clock) => {
            encoder.u8(9)?;
            match clock.kind() {
                ClockKind::Continuous => encoder.u8(1)?,
                ClockKind::Periodic { period, phase } => {
                    encoder.u8(2)?;
                    encoder.u64(period.numerator())?;
                    encoder.u64(period.denominator())?;
                    encoder.u64(phase.numerator())?;
                    encoder.u64(phase.denominator())?;
                }
                ClockKind::Aperiodic => encoder.u8(3)?,
                ClockKind::Inherited => encoder.u8(4)?,
                _ => return Err(newer_vocabulary("ClockDomain kind")),
            }
        }
        _ => return Err(newer_vocabulary("Semantic Kernel node")),
    }
    encode_optional_literal(&mut encoder, current_value)?;
    if let Some(value) = current_value {
        type_reference(
            value.value_type(),
            nominal_label(8),
            ids,
            references,
            budget,
        )?;
    }
    encoder.bool(boundary)?;
    encoder.finish()
}

fn encode_domain_kind(
    encoder: &mut Encoder,
    kind: &DomainKind,
    ids: &BTreeMap<RawId, usize>,
    references: &mut Vec<Reference>,
    budget: &mut ConstructionBudget,
) -> Result<(), Diagnostic> {
    match kind {
        DomainKind::Abstract => encoder.u8(1),
        DomainKind::CartesianBox { coordinates } => {
            encoder.u8(2)?;
            encoder.len(coordinates.len())?;
            for (axis_index, axis) in coordinates.iter().enumerate() {
                for (endpoint, source) in [(1, axis.lower()), (2, axis.upper())] {
                    match source {
                        CartesianCoordinateSource::Fixed(value) => {
                            encoder.u8(1)?;
                            encode_quantity(encoder, value)?;
                        }
                        CartesianCoordinateSource::Parameter(parameter) => {
                            encoder.u8(2)?;
                            let mut label = Encoder::new(32);
                            label.u8(4)?;
                            label.usize(axis_index)?;
                            label.u8(endpoint)?;
                            push_reference(
                                references,
                                label.finish()?,
                                lookup(ids, parameter.erase(), "Cartesian coordinate Parameter")?,
                                budget,
                            )?;
                        }
                    }
                }
            }
            Ok(())
        }
        DomainKind::CartesianBoundary { axis, side } => {
            encoder.u8(3)?;
            encoder.usize(*axis)?;
            encode_boundary_side(encoder, *side)
        }
        DomainKind::ScalarPhysical {
            across_type,
            through_type,
        } => {
            encoder.u8(4)?;
            encode_value_type(encoder, across_type)?;
            type_reference(across_type, nominal_label(9), ids, references, budget)?;
            type_reference(through_type, nominal_label(10), ids, references, budget)?;
            encode_value_type(encoder, through_type)
        }
        DomainKind::BoundaryPhysical { connector } => {
            encoder.u8(5)?;
            for (role, value_type) in [(11, connector.trace_type()), (12, connector.flux_type())] {
                encode_value_type(encoder, value_type)?;
                type_reference(value_type, nominal_label(role), ids, references, budget)?;
            }
            match connector.pairing() {
                BoundaryPairing::EuclideanBoundaryDuality => encoder.u8(1),
            }
        }
        DomainKind::GeometryRegion {
            geometry,
            entity_set,
        } => {
            encoder.u8(6)?;
            encoder.raw(&geometry.bytes())?;
            encoder.bytes(entity_set.as_bytes())
        }
        DomainKind::GeometryBoundary { entity_set } => {
            encoder.u8(7)?;
            encoder.bytes(entity_set.as_bytes())
        }
        _ => Err(newer_vocabulary("Domain kind")),
    }
}

fn push_reference(
    references: &mut Vec<Reference>,
    label: Vec<u8>,
    target: usize,
    budget: &mut ConstructionBudget,
) -> Result<(), Diagnostic> {
    budget.account_reference()?;
    budget.account_bytes(label.len())?;
    references
        .try_reserve(1)
        .map_err(|_| fingerprint_error("cannot reserve semantic projection reference"))?;
    references.push(Reference { label, target });
    Ok(())
}

fn nominal_label(role: u8) -> Vec<u8> {
    vec![2, role]
}

fn edge_label(kind: EdgeKind) -> Result<Vec<u8>, Diagnostic> {
    let tag = match kind {
        EdgeKind::DefinedOn => 1,
        EdgeKind::AppliesOn => 2,
        EdgeKind::BoundaryOf => 3,
        EdgeKind::DependsOn => 4,
        EdgeKind::HasPort => 5,
        EdgeKind::Activates => 6,
        EdgeKind::Connects => 7,
        EdgeKind::ClockedBy => 8,
        EdgeKind::StructurallyDependsOn => 9,
        _ => return Err(newer_vocabulary("Semantic Model edge")),
    };
    Ok(vec![1, tag])
}

fn lookup(ids: &BTreeMap<RawId, usize>, id: RawId, role: &str) -> Result<usize, Diagnostic> {
    ids.get(&id).copied().ok_or_else(|| {
        fingerprint_error(format!(
            "{role} {id} is outside the accepted Semantic Model projection"
        ))
    })
}

fn encode_boundary_side(encoder: &mut Encoder, side: BoundarySide) -> Result<(), Diagnostic> {
    match side {
        BoundarySide::Lower => encoder.u8(1),
        BoundarySide::Upper => encoder.u8(2),
    }
}

fn encode_signal_direction(
    encoder: &mut Encoder,
    direction: SignalDirection,
) -> Result<(), Diagnostic> {
    match direction {
        SignalDirection::Input => encoder.u8(1),
        SignalDirection::Output => encoder.u8(2),
    }
}

fn encode_event_direction(
    encoder: &mut Encoder,
    direction: EventDirection,
) -> Result<(), Diagnostic> {
    match direction {
        EventDirection::Any => encoder.u8(1),
        EventDirection::Rising => encoder.u8(2),
        EventDirection::Falling => encoder.u8(3),
    }
}

fn validate_limits(limits: SemanticFingerprintLimits) -> Result<(), Diagnostic> {
    if limits.max_nodes == 0
        || limits.max_references == 0
        || limits.max_expression_nodes == 0
        || limits.max_canonical_bytes < PROJECTION_MAGIC.len() + 16
        || limits.max_search_states == 0
        || limits.max_individualization_depth == 0
        || limits.max_refinement_work == 0
        || limits.max_serialization_work == 0
    {
        return Err(fingerprint_error(
            "structural semantic fingerprint limits must all admit non-empty bounded work",
        ));
    }
    Ok(())
}

fn newer_vocabulary(subject: &str) -> Diagnostic {
    fingerprint_error(format!(
        "{subject} is newer than structural semantic fingerprint generation v25"
    ))
}

fn fingerprint_error(message: impl Into<String>) -> Diagnostic {
    invalid_artifact(message)
}

#[cfg(test)]
mod tests;
