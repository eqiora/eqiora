//! Canonical, bounded wire for a resolved portable realization graph.

mod basic;
mod transformation;

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, OntologyId};
use eqiora_schema::Model;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use ulid::Ulid;

use self::basic::{
    WireAlgebraicBlock, WireDiscretization, WireNonlinearPlan, WireOperatorProperties,
    WirePositiveScale, WireQuantity, WireScalarType, WireScaling, WireSchedule, WireSolverPlan,
    WireSpace, WireVectorLayout, decode_constraint, decode_nonzero_usize, decode_usize,
    encode_constraint, encode_usize, normalize_zero, parse_id,
};
use self::transformation::WireTransformation;
use super::*;
use crate::{
    AleGeometryQualityGate, DefaultPolicyVersion, RealizationRevision, ResolutionSource,
    SemanticRevision, invalid_realization,
};

const SCHEMA: &str = "eqiora.portable-realization-graph/v4";
const ENCODING: &str = "eqiora.canonical-json/v1";
const DIGEST_DOMAIN: &[u8] = b"eqiora.portable-realization-graph/v4\0";
const MAX_BYTES: usize = 8 * 1024 * 1024;
const MAX_NODES_PER_ARENA: usize = 100_000;

impl PortableRealizationGraph {
    /// Encode this resolved Plan graph as deterministic canonical JSON bytes.
    ///
    /// The payload is portable: it contains no filesystem path, runtime
    /// handle, device ordinal, communicator, credential, or deployment binding.
    ///
    /// # Errors
    /// Returns `EQ0807` if the graph is invalid, a platform count cannot be
    /// represented by the wire, or serialization unexpectedly fails.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        self.validate()?;
        let wire = WireGraph::encode(self)?;
        serde_json::to_vec(&wire).map_err(|error| {
            invalid_realization(format!(
                "cannot serialize portable realization graph: {error}"
            ))
        })
    }

    /// Decode and revalidate one exact portable Plan graph.
    ///
    /// Unknown versions, fields, noncanonical JSON, excessive input, malformed
    /// identities, disconnected nodes, and invalid numerical combinations fail
    /// closed before a graph is returned.
    ///
    /// # Errors
    /// Returns `EQ0807` when the bytes are not the exact bounded v4 wire.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Diagnostic> {
        if bytes.len() > MAX_BYTES {
            return Err(invalid_realization(
                "portable realization graph exceeds the 8 MiB decode bound",
            ));
        }
        let wire: WireGraph = serde_json::from_slice(bytes).map_err(|error| {
            invalid_realization(format!("invalid portable realization graph JSON: {error}"))
        })?;
        wire.validate_limits()?;
        let canonical = serde_json::to_vec(&wire).map_err(|error| {
            invalid_realization(format!(
                "cannot canonicalize portable realization graph: {error}"
            ))
        })?;
        if canonical != bytes {
            return Err(invalid_realization(
                "portable realization graph JSON is not canonical",
            ));
        }
        let graph = wire.decode()?;
        let normalized = serde_json::to_vec(&WireGraph::encode(&graph)?).map_err(|error| {
            invalid_realization(format!(
                "cannot normalize portable realization graph: {error}"
            ))
        })?;
        if normalized != bytes {
            return Err(invalid_realization(
                "portable realization graph uses a noncanonical semantic spelling",
            ));
        }
        Ok(graph)
    }

    /// Domain-separated SHA-256 identity of the exact canonical graph bytes.
    ///
    /// # Errors
    /// Returns `EQ0807` if canonical encoding unexpectedly fails.
    pub fn digest(&self) -> Result<[u8; 32], Diagnostic> {
        let bytes = self.to_bytes()?;
        let mut digest = Sha256::new();
        digest.update(DIGEST_DOMAIN);
        digest.update(bytes);
        Ok(digest.finalize().into())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireGraph {
    schema: String,
    encoding: String,
    lineage: WireLineage,
    domains: Vec<WireDomain>,
    fields: Vec<WireField>,
    geometry_actions: Vec<WireGeometryAction>,
    transformations: Vec<WireTransformation>,
    systems: Vec<WireSystem>,
    linear_solves: Vec<WireLinearSolve>,
    nonlinear_solves: Vec<WireNonlinearSolve>,
    placements: Vec<WirePlacement>,
    root: WireRoot,
}

impl WireGraph {
    fn encode(value: &PortableRealizationGraph) -> Result<Self, Diagnostic> {
        Ok(Self {
            schema: SCHEMA.to_owned(),
            encoding: ENCODING.to_owned(),
            lineage: WireLineage::encode(value.lineage()),
            domains: value
                .domains()
                .iter()
                .map(WireDomain::encode)
                .collect::<Result<_, _>>()?,
            fields: value
                .fields()
                .iter()
                .copied()
                .map(WireField::encode)
                .collect::<Result<_, _>>()?,
            geometry_actions: value
                .geometry_actions()
                .iter()
                .copied()
                .map(WireGeometryAction::encode)
                .collect::<Result<_, _>>()?,
            transformations: value
                .transformations()
                .iter()
                .copied()
                .map(WireTransformation::encode)
                .collect::<Result<_, _>>()?,
            systems: value
                .systems()
                .iter()
                .map(WireSystem::encode)
                .collect::<Result<_, _>>()?,
            linear_solves: value
                .linear_solves()
                .iter()
                .copied()
                .map(WireLinearSolve::encode)
                .collect::<Result<_, _>>()?,
            nonlinear_solves: value
                .nonlinear_solves()
                .iter()
                .copied()
                .map(WireNonlinearSolve::encode)
                .collect::<Result<_, _>>()?,
            placements: value
                .placements()
                .iter()
                .copied()
                .map(WirePlacement::encode)
                .collect::<Result<_, _>>()?,
            root: WireRoot::encode(value.root())?,
        })
    }

    fn validate_limits(&self) -> Result<(), Diagnostic> {
        if self.schema != SCHEMA || self.encoding != ENCODING {
            return Err(invalid_realization(
                "portable realization graph schema or encoding is unsupported",
            ));
        }
        for (label, count) in [
            ("Domain", self.domains.len()),
            ("Field", self.fields.len()),
            ("geometry action", self.geometry_actions.len()),
            ("transformation", self.transformations.len()),
            ("system", self.systems.len()),
            ("linear solve", self.linear_solves.len()),
            ("nonlinear solve", self.nonlinear_solves.len()),
            ("placement", self.placements.len()),
        ] {
            if count > MAX_NODES_PER_ARENA {
                return Err(invalid_realization(format!(
                    "portable realization graph {label} inventory exceeds its decode bound"
                )));
            }
        }
        for system in &self.systems {
            if system.blocks.len() > MAX_NODES_PER_ARENA
                || system.transformations.len() > MAX_NODES_PER_ARENA
            {
                return Err(invalid_realization(
                    "portable realization system inventory exceeds its decode bound",
                ));
            }
            if let WireScaling::SymmetricCongruence { block_scales, .. } = &system.scaling
                && block_scales.len() > MAX_NODES_PER_ARENA
            {
                return Err(invalid_realization(
                    "portable realization scaling inventory exceeds its decode bound",
                ));
            }
        }
        Ok(())
    }

    fn decode(self) -> Result<PortableRealizationGraph, Diagnostic> {
        let graph = PortableRealizationGraph {
            lineage: self.lineage.decode()?,
            domains: self
                .domains
                .into_iter()
                .map(WireDomain::decode)
                .collect::<Result<_, _>>()?,
            fields: self
                .fields
                .into_iter()
                .map(WireField::decode)
                .collect::<Result<_, _>>()?,
            geometry_actions: self
                .geometry_actions
                .into_iter()
                .map(WireGeometryAction::decode)
                .collect::<Result<_, _>>()?,
            transformations: self
                .transformations
                .into_iter()
                .map(WireTransformation::decode)
                .collect::<Result<_, _>>()?,
            systems: self
                .systems
                .into_iter()
                .map(WireSystem::decode)
                .collect::<Result<_, _>>()?,
            linear_solves: self
                .linear_solves
                .into_iter()
                .map(WireLinearSolve::decode)
                .collect::<Result<_, _>>()?,
            nonlinear_solves: self
                .nonlinear_solves
                .into_iter()
                .map(WireNonlinearSolve::decode)
                .collect::<Result<_, _>>()?,
            placements: self
                .placements
                .into_iter()
                .map(WirePlacement::decode)
                .collect::<Result<_, _>>()?,
            root: self.root.decode()?,
        };
        graph.validate()?;
        Ok(graph)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireLineage {
    model_ulid: String,
    semantic_revision: u64,
    source: WireSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireSource {
    Default { policy_version: u16 },
    Explicit { realization_revision: u64 },
}

impl WireLineage {
    fn encode(value: RealizationLineage) -> Self {
        Self {
            model_ulid: value.model().ulid().to_string(),
            semantic_revision: value.semantic_revision().get(),
            source: match value.source() {
                ResolutionSource::Default(policy) => WireSource::Default {
                    policy_version: policy.get(),
                },
                ResolutionSource::Explicit(revision) => WireSource::Explicit {
                    realization_revision: revision.get(),
                },
            },
        }
    }

    fn decode(self) -> Result<RealizationLineage, Diagnostic> {
        let model = self
            .model_ulid
            .parse::<Ulid>()
            .map(OntologyId::<Model>::from_ulid)
            .map_err(|_| invalid_realization("portable graph contains an invalid Model ULID"))?;
        let semantic_revision = SemanticRevision::new(self.semantic_revision);
        let source = match self.source {
            WireSource::Default { policy_version }
                if policy_version == DefaultPolicyVersion::V0.get() =>
            {
                ResolutionSource::Default(DefaultPolicyVersion::V0)
            }
            WireSource::Default { .. } => {
                return Err(invalid_realization(
                    "portable graph contains an unsupported default policy version",
                ));
            }
            WireSource::Explicit {
                realization_revision,
            } => ResolutionSource::Explicit(RealizationRevision::new(realization_revision)),
        };
        Ok(RealizationLineage::new(model, semantic_revision, source))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireDomain {
    domain_ulid: String,
    coordinates: WireCoordinates,
    configuration: WireConfiguration,
    discretization: WireDiscretization,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireCoordinates {
    Physical,
    Scaled { scale: WirePositiveScale },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireConfiguration {
    FixedGeometry,
    ReferenceConfiguration,
    CurrentAleGeometry { geometry_action: u64 },
}

impl WireDomain {
    fn encode(value: &DomainDiscretizationNode) -> Result<Self, Diagnostic> {
        Ok(Self {
            domain_ulid: value.domain().ulid().to_string(),
            coordinates: match value.coordinates() {
                CoordinateTreatment::Physical => WireCoordinates::Physical,
                CoordinateTreatment::Scaled(scale) => WireCoordinates::Scaled {
                    scale: WirePositiveScale::encode(scale),
                },
            },
            configuration: match value.configuration() {
                DomainConfiguration::FixedGeometry => WireConfiguration::FixedGeometry,
                DomainConfiguration::ReferenceConfiguration => {
                    WireConfiguration::ReferenceConfiguration
                }
                DomainConfiguration::CurrentAleGeometry { action } => {
                    WireConfiguration::CurrentAleGeometry {
                        geometry_action: encode_index(action.index(), "geometry action")?,
                    }
                }
            },
            discretization: WireDiscretization::encode(value.discretization())?,
        })
    }

    fn decode(self) -> Result<DomainDiscretizationNode, Diagnostic> {
        Ok(DomainDiscretizationNode {
            domain: parse_id(&self.domain_ulid)?,
            coordinates: match self.coordinates {
                WireCoordinates::Physical => CoordinateTreatment::Physical,
                WireCoordinates::Scaled { scale } => CoordinateTreatment::Scaled(scale.decode()?),
            },
            configuration: match self.configuration {
                WireConfiguration::FixedGeometry => DomainConfiguration::FixedGeometry,
                WireConfiguration::ReferenceConfiguration => {
                    DomainConfiguration::ReferenceConfiguration
                }
                WireConfiguration::CurrentAleGeometry { geometry_action } => {
                    DomainConfiguration::CurrentAleGeometry {
                        action: GeometryActionId::new(decode_index(
                            geometry_action,
                            "geometry action",
                        )?),
                    }
                }
            },
            discretization: self.discretization.decode()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireField {
    domain: u64,
    field_ulid: String,
    space: WireSpace,
}

impl WireField {
    fn encode(value: FieldRepresentationNode) -> Result<Self, Diagnostic> {
        Ok(Self {
            domain: encode_index(value.domain().index(), "Domain reference")?,
            field_ulid: value.field().ulid().to_string(),
            space: WireSpace::encode(value.space()),
        })
    }

    fn decode(self) -> Result<FieldRepresentationNode, Diagnostic> {
        Ok(FieldRepresentationNode {
            domain: DomainDiscretizationId::new(decode_index(self.domain, "Domain reference")?),
            field: parse_id(&self.field_ulid)?,
            space: self.space.decode()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireGeometryAction {
    P1HarmonicExtension {
        fluid_domain: u64,
        solid_domain: u64,
        driver: u64,
        interface_ulid: String,
        duration: WireQuantity,
        minimum_mean_ratio: f64,
        solver: WireSolverPlan,
    },
}

impl WireGeometryAction {
    fn encode(value: GeometryActionNode) -> Result<Self, Diagnostic> {
        match value {
            GeometryActionNode::P1HarmonicExtension {
                fluid_domain,
                solid_domain,
                driver,
                interface,
                duration,
                quality_gate,
                solver,
            } => Ok(Self::P1HarmonicExtension {
                fluid_domain: encode_index(fluid_domain.index(), "fluid Domain")?,
                solid_domain: encode_index(solid_domain.index(), "solid Domain")?,
                driver: encode_index(driver.index(), "geometry driver")?,
                interface_ulid: interface.ulid().to_string(),
                duration: WireQuantity::encode(duration),
                minimum_mean_ratio: normalize_zero(quality_gate.minimum_mean_ratio()),
                solver: WireSolverPlan::encode(solver)?,
            }),
        }
    }

    fn decode(self) -> Result<GeometryActionNode, Diagnostic> {
        match self {
            Self::P1HarmonicExtension {
                fluid_domain,
                solid_domain,
                driver,
                interface_ulid,
                duration,
                minimum_mean_ratio,
                solver,
            } => Ok(GeometryActionNode::P1HarmonicExtension {
                fluid_domain: DomainDiscretizationId::new(decode_index(
                    fluid_domain,
                    "fluid Domain",
                )?),
                solid_domain: DomainDiscretizationId::new(decode_index(
                    solid_domain,
                    "solid Domain",
                )?),
                driver: FieldRepresentationId::new(decode_index(driver, "geometry driver")?),
                interface: parse_id::<kinds::Connection>(&interface_ulid)?,
                duration: duration.decode(),
                quality_gate: AleGeometryQualityGate::new(minimum_mean_ratio)?,
                solver: solver.decode()?,
            }),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSystem {
    blocks: Vec<WireSystemBlock>,
    transformations: Vec<u64>,
    scaling: WireScaling,
    operator_properties: WireOperatorProperties,
    scalar_type: WireScalarType,
    partition: WireVectorLayout,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireSystemBlock {
    Field { representation: u64 },
    ConstraintMultiplier { constraint: WireAlgebraicBlock },
}

impl WireSystem {
    fn encode(value: &AlgebraicSystemNode) -> Result<Self, Diagnostic> {
        Ok(Self {
            blocks: value
                .blocks()
                .iter()
                .copied()
                .map(|block| match block {
                    SystemBlock::Field(field) => Ok(WireSystemBlock::Field {
                        representation: encode_index(field.index(), "system Field")?,
                    }),
                    SystemBlock::ConstraintMultiplier(constraint) => {
                        Ok(WireSystemBlock::ConstraintMultiplier {
                            constraint: encode_constraint(constraint),
                        })
                    }
                })
                .collect::<Result<_, Diagnostic>>()?,
            transformations: value
                .transformations()
                .iter()
                .map(|id| encode_index(id.index(), "system transformation"))
                .collect::<Result<_, _>>()?,
            scaling: WireScaling::encode(value.scaling()),
            operator_properties: WireOperatorProperties::encode(value.operator_properties()),
            scalar_type: WireScalarType::encode(value.scalar_type()),
            partition: WireVectorLayout::encode(value.partition()),
        })
    }

    fn decode(self) -> Result<AlgebraicSystemNode, Diagnostic> {
        Ok(AlgebraicSystemNode {
            blocks: self
                .blocks
                .into_iter()
                .map(|block| match block {
                    WireSystemBlock::Field { representation } => Ok(SystemBlock::Field(
                        FieldRepresentationId::new(decode_index(representation, "system Field")?),
                    )),
                    WireSystemBlock::ConstraintMultiplier { constraint } => Ok(
                        SystemBlock::ConstraintMultiplier(decode_constraint(constraint)?),
                    ),
                })
                .collect::<Result<_, Diagnostic>>()?,
            transformations: self
                .transformations
                .into_iter()
                .map(|index| {
                    decode_index(index, "system transformation").map(TransformationId::new)
                })
                .collect::<Result<_, _>>()?,
            scaling: self.scaling.decode()?,
            operator_properties: self.operator_properties.decode(),
            scalar_type: self.scalar_type.decode(),
            partition: self.partition.decode(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireLinearSolve {
    system: u64,
    plan: WireSolverPlan,
    placement: u64,
    schedule: WireSchedule,
}

impl WireLinearSolve {
    fn encode(value: LinearSolveNode) -> Result<Self, Diagnostic> {
        Ok(Self {
            system: encode_index(value.system().index(), "linear system")?,
            plan: WireSolverPlan::encode(value.plan())?,
            placement: encode_index(value.placement().index(), "linear placement")?,
            schedule: WireSchedule::encode(value.schedule()),
        })
    }

    fn decode(self) -> Result<LinearSolveNode, Diagnostic> {
        Ok(LinearSolveNode {
            system: AlgebraicSystemId::new(decode_index(self.system, "linear system")?),
            plan: self.plan.decode()?,
            placement: PlacementRequirementId::new(decode_index(
                self.placement,
                "linear placement",
            )?),
            schedule: self.schedule.decode()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireNonlinearSolve {
    residual_system: u64,
    linearization: u64,
    plan: WireNonlinearPlan,
}

impl WireNonlinearSolve {
    fn encode(value: NonlinearSolveNode) -> Result<Self, Diagnostic> {
        Ok(Self {
            residual_system: encode_index(
                value.residual_system().index(),
                "nonlinear residual system",
            )?,
            linearization: encode_index(value.linearization().index(), "nonlinear linearization")?,
            plan: WireNonlinearPlan::encode(value.plan())?,
        })
    }

    fn decode(self) -> Result<NonlinearSolveNode, Diagnostic> {
        Ok(NonlinearSolveNode {
            residual_system: AlgebraicSystemId::new(decode_index(
                self.residual_system,
                "nonlinear residual system",
            )?),
            linearization: LinearSolveId::new(decode_index(
                self.linearization,
                "nonlinear linearization",
            )?),
            plan: self.plan.decode()?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WirePlacement {
    HostWorkers { workers_per_partition: u64 },
    CudaDevices { devices_per_partition: u64 },
}

impl WirePlacement {
    fn encode(value: PlacementRequirementNode) -> Result<Self, Diagnostic> {
        match value {
            PlacementRequirementNode::HostWorkers {
                workers_per_partition,
            } => Ok(Self::HostWorkers {
                workers_per_partition: encode_usize(
                    workers_per_partition.get(),
                    "host workers per partition",
                )?,
            }),
            PlacementRequirementNode::CudaDevices {
                devices_per_partition,
            } => Ok(Self::CudaDevices {
                devices_per_partition: encode_usize(
                    devices_per_partition.get(),
                    "CUDA devices per partition",
                )?,
            }),
        }
    }

    fn decode(self) -> Result<PlacementRequirementNode, Diagnostic> {
        match self {
            Self::HostWorkers {
                workers_per_partition,
            } => Ok(PlacementRequirementNode::HostWorkers {
                workers_per_partition: decode_nonzero_usize(
                    workers_per_partition,
                    "host workers per partition",
                )?,
            }),
            Self::CudaDevices {
                devices_per_partition,
            } => Ok(PlacementRequirementNode::CudaDevices {
                devices_per_partition: decode_nonzero_usize(
                    devices_per_partition,
                    "CUDA devices per partition",
                )?,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireRoot {
    Linear { solve: u64 },
    Nonlinear { solve: u64 },
}

impl WireRoot {
    fn encode(value: SolveRoot) -> Result<Self, Diagnostic> {
        match value {
            SolveRoot::Linear(id) => Ok(Self::Linear {
                solve: encode_index(id.index(), "linear root")?,
            }),
            SolveRoot::Nonlinear(id) => Ok(Self::Nonlinear {
                solve: encode_index(id.index(), "nonlinear root")?,
            }),
        }
    }

    fn decode(self) -> Result<SolveRoot, Diagnostic> {
        match self {
            Self::Linear { solve } => Ok(SolveRoot::Linear(LinearSolveId::new(decode_index(
                solve,
                "linear root",
            )?))),
            Self::Nonlinear { solve } => Ok(SolveRoot::Nonlinear(NonlinearSolveId::new(
                decode_index(solve, "nonlinear root")?,
            ))),
        }
    }
}

fn encode_index(value: usize, label: &str) -> Result<u64, Diagnostic> {
    encode_usize(value, label)
}

fn decode_index(value: u64, label: &str) -> Result<usize, Diagnostic> {
    decode_usize(value, label)
}
