use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use eqiora_core::Diagnostic;
use eqiora_realization::{Target, VectorLayoutKind};
use eqiora_solver::{ExecutionProvider, ReductionPolicy, SolverProvider};
use serde::{Deserialize, Serialize};

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, CanonicalRealizationArtifact, JsonDecoderLimits,
    LayoutArtifacts, check_json_limits, invalid_artifact, validate_setting_key, validate_text,
};

const RUN_V2_SCHEMA: &str = "eqiora.run-manifest/v2";

/// MPI thread-support level resolved by a transport adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpiThreadSupportV1 {
    /// Only the initializing thread calls MPI.
    Single,
    /// Only the initializing thread calls MPI, but other threads may exist.
    Funneled,
    /// Multiple threads may call MPI, but never concurrently.
    Serialized,
    /// Multiple threads may call MPI concurrently.
    Multiple,
}

/// Transport identity for one distributed execution group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributedTransportV1 {
    /// Deterministic one-process protocol oracle.
    Loopback,
    /// A resolved system MPI implementation.
    Mpi {
        /// Implementation identity such as `openmpi` or `mpich`.
        implementation: String,
        /// Implementation version reported at execution time.
        version: String,
        /// Thread-support level actually provided by initialization.
        thread_support: MpiThreadSupportV1,
    },
}

/// Resolved physical/logical placement of one execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionTopologyV1 {
    /// One host process with a bounded worker pool.
    Host {
        /// Worker threads admitted to the run.
        workers: NonZeroUsize,
    },
    /// One explicitly partitioned execution group.
    Distributed {
        /// Number of unique-owner partitions/ranks.
        partitions: NonZeroUsize,
        /// Worker threads admitted within each partition.
        workers_per_partition: NonZeroUsize,
        /// Transport actually used by the execution group.
        transport: DistributedTransportV1,
    },
    /// One resolved CUDA device.
    Cuda {
        /// Device ordinal resolved by the deployment environment.
        device: u16,
        /// Device name reported by the CUDA runtime.
        device_name: String,
        /// CUDA compute-capability major component.
        compute_capability_major: u16,
        /// CUDA compute-capability minor component.
        compute_capability_minor: u16,
        /// CUDA driver version reported at execution time.
        driver_version: String,
    },
}

/// Typed backend, library, topology, and reduction provenance for one run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionProvenanceV1 {
    wire: WireExecutionProvenanceV1,
}

/// Domain-separated agreement identity of one exact runtime observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExecutionProvenanceFingerprintV1([u8; 32]);

impl ExecutionProvenanceFingerprintV1 {
    /// Complete fingerprint bytes for fixed-size collective agreement.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

impl ExecutionProvenanceV1 {
    /// Construct validated execution provenance.
    ///
    /// # Errors
    /// Returns `EQ0901` for empty/control-containing identities or invalid
    /// target-specific topology data.
    pub fn new(
        adapter: impl Into<String>,
        adapter_version: impl Into<String>,
        solver_backend: impl Into<String>,
        solver_backend_version: impl Into<String>,
        topology: ExecutionTopologyV1,
        reduction: ReductionPolicy,
    ) -> Result<Self, Diagnostic> {
        let wire = WireExecutionProvenanceV1 {
            adapter: adapter.into(),
            adapter_version: adapter_version.into(),
            solver_backend: solver_backend.into(),
            solver_backend_version: solver_backend_version.into(),
            libraries: BTreeMap::new(),
            topology: WireExecutionTopologyV1::encode(topology)?,
            reduction: WireReduction::encode(reduction),
        };
        let value = Self { wire };
        value.validate()?;
        Ok(value)
    }

    /// Construct v1 execution provenance from the two primary provider
    /// releases and additional observed runtime components.
    ///
    /// Provider dependency releases and runtime observations enter one sorted
    /// component inventory. Repeated equal observations are deduplicated;
    /// conflicting versions under one component name fail closed. This does
    /// not turn the flat v1 inventory into a role-preserving provider graph.
    ///
    /// # Errors
    /// Returns `EQ0901` for an invalid provider, component, topology, or a
    /// contradictory component version.
    pub fn from_provider_releases<I, N, V>(
        solver: SolverProvider,
        execution: ExecutionProvider,
        topology: ExecutionTopologyV1,
        reduction: ReductionPolicy,
        additional_components: I,
    ) -> Result<Self, Diagnostic>
    where
        I: IntoIterator<Item = (N, V)>,
        N: Into<String>,
        V: Into<String>,
    {
        solver.validate().map_err(|diagnostic| {
            invalid_artifact(format!("invalid solver provider release: {diagnostic}"))
        })?;
        execution.validate().map_err(|diagnostic| {
            invalid_artifact(format!("invalid execution provider release: {diagnostic}"))
        })?;

        let mut libraries = BTreeMap::new();
        for library in solver.libraries().iter().chain(execution.libraries()) {
            merge_component_version(
                &mut libraries,
                library.name().to_owned(),
                library.version().to_owned(),
            )?;
        }
        for (name, version) in additional_components {
            merge_component_version(&mut libraries, name.into(), version.into())?;
        }

        let mut value = Self::new(
            execution.id().as_str(),
            execution.implementation_version(),
            solver.id().as_str(),
            solver.implementation_version(),
            topology,
            reduction,
        )?;
        value.wire.libraries = libraries;
        value.validate()?;
        Ok(value)
    }

    /// Add one resolved library/runtime version.
    ///
    /// # Errors
    /// Returns `EQ0901` for an invalid/duplicate component name or invalid
    /// version text.
    pub fn with_library(
        mut self,
        component: impl Into<String>,
        version: impl Into<String>,
    ) -> Result<Self, Diagnostic> {
        let component = component.into();
        let version = version.into();
        validate_setting_key(&component)?;
        validate_text("library version", &version)?;
        if self
            .wire
            .libraries
            .insert(component.clone(), version)
            .is_some()
        {
            return Err(invalid_artifact(format!(
                "duplicate execution library `{component}`"
            )));
        }
        Ok(self)
    }

    /// Stable execution adapter identity.
    #[must_use]
    pub fn adapter(&self) -> &str {
        &self.wire.adapter
    }

    /// Resolved execution adapter version.
    #[must_use]
    pub fn adapter_version(&self) -> &str {
        &self.wire.adapter_version
    }

    /// Stable solver backend identity, separate from operator placement.
    #[must_use]
    pub fn solver_backend(&self) -> &str {
        &self.wire.solver_backend
    }

    /// Resolved solver backend version.
    #[must_use]
    pub fn solver_backend_version(&self) -> &str {
        &self.wire.solver_backend_version
    }

    /// Resolved topology.
    ///
    /// # Errors
    /// Returns `EQ0901` only if internal validated state was corrupted.
    pub fn topology(&self) -> Result<ExecutionTopologyV1, Diagnostic> {
        self.wire.topology.decode()
    }

    /// Numerical reduction policy actually used.
    #[must_use]
    pub const fn reduction(&self) -> ReductionPolicy {
        self.wire.reduction.decode()
    }

    /// Sorted library/runtime version map.
    #[must_use]
    pub const fn libraries(&self) -> &BTreeMap<String, String> {
        &self.wire.libraries
    }

    /// Compute a stable identity over the complete validated runtime observation.
    ///
    /// This identity supports in-memory agreement and does not turn execution
    /// provenance into a standalone durable artifact.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization unexpectedly fails.
    pub fn agreement_fingerprint(&self) -> Result<ExecutionProvenanceFingerprintV1, Diagnostic> {
        let bytes = serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!(
                "cannot serialize execution provenance for agreement: {error}"
            ))
        })?;
        Ok(ExecutionProvenanceFingerprintV1(
            ArtifactDigest::compute(b"eqiora.execution-provenance-agreement/v1", &bytes)
                .sha256_bytes(),
        ))
    }

    fn validate(&self) -> Result<(), Diagnostic> {
        validate_text("execution adapter", &self.wire.adapter)?;
        validate_text("execution adapter version", &self.wire.adapter_version)?;
        validate_text("solver backend", &self.wire.solver_backend)?;
        validate_text("solver backend version", &self.wire.solver_backend_version)?;
        for (component, version) in &self.wire.libraries {
            validate_setting_key(component)?;
            validate_text("library version", version)?;
        }
        self.wire.topology.decode()?;
        Ok(())
    }
}

fn merge_component_version(
    libraries: &mut BTreeMap<String, String>,
    component: String,
    version: String,
) -> Result<(), Diagnostic> {
    validate_setting_key(&component)?;
    validate_text("library version", &version)?;
    match libraries.get(&component) {
        Some(existing) if existing != &version => Err(invalid_artifact(format!(
            "execution library `{component}` has contradictory versions `{existing}` and `{version}`"
        ))),
        Some(_) => Ok(()),
        None => {
            libraries.insert(component, version);
            Ok(())
        }
    }
}

/// Reproducible inputs, resolved execution provenance, and outputs of one run.
///
/// V2 requires a typed Realization artifact and replaces v1's opaque numerical
/// setting map with exact policy in that artifact plus typed resolved evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunManifestV2 {
    wire: WireRunManifestV2,
}

impl RunManifestV2 {
    /// Start a v2 manifest from one typed Realization artifact.
    ///
    /// # Errors
    /// Returns `EQ0901` if execution provenance is invalid or contradicts the
    /// realization target, layout, worker count, or reduction policy.
    pub fn new(
        realization: &(impl CanonicalRealizationArtifact + ?Sized),
        execution: ExecutionProvenanceV1,
    ) -> Result<Self, Diagnostic> {
        execution.validate()?;
        let realization = realization.artifact_reference()?;
        let manifest = Self {
            wire: WireRunManifestV2 {
                schema: RUN_V2_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                model_sha256: realization.model_artifact().to_string(),
                semantic_revision: realization.semantic_revision().get(),
                realization_sha256: realization.artifact().to_string(),
                execution: execution.wire,
                output_sha256: Vec::new(),
            },
        };
        manifest.validate_against(&realization)?;
        Ok(manifest)
    }

    /// Add a content-addressed output artifact.
    #[must_use]
    pub fn with_output(mut self, output: ArtifactDigest) -> Self {
        self.wire.output_sha256.push(output.0);
        self.wire.output_sha256.sort();
        self.wire.output_sha256.dedup();
        self
    }

    /// Decode and validate a v2 run manifest.
    ///
    /// # Errors
    /// Returns `EQ0901` for oversized, malformed, unknown-version, duplicate,
    /// or non-canonical field data.
    pub fn from_json(bytes: &[u8], limits: JsonDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits)?;
        let wire = serde_json::from_slice(bytes)
            .map_err(|error| invalid_artifact(format!("invalid run manifest v2 JSON: {error}")))?;
        let manifest = Self { wire };
        manifest.validate()?;
        Ok(manifest)
    }

    /// Deterministic canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire)
            .map_err(|error| invalid_artifact(format!("cannot serialize run manifest v2: {error}")))
    }

    /// Domain-separated SHA-256 identity of canonical manifest bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            RUN_V2_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Referenced canonical Semantic Model artifact.
    #[must_use]
    pub fn model(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.model_sha256.clone())
    }

    /// Referenced typed Realization artifact.
    #[must_use]
    pub fn realization(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.realization_sha256.clone())
    }

    /// Semantic graph revision referenced by the linked Realization.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.wire.semantic_revision
    }

    /// Typed resolved execution provenance.
    #[must_use]
    pub fn execution(&self) -> ExecutionProvenanceV1 {
        ExecutionProvenanceV1 {
            wire: self.wire.execution.clone(),
        }
    }

    /// Sorted content-addressed output artifacts.
    #[must_use]
    pub fn outputs(&self) -> Vec<ArtifactDigest> {
        self.wire
            .output_sha256
            .iter()
            .cloned()
            .map(ArtifactDigest)
            .collect()
    }

    /// Validate external model/realization linkage and execution consistency.
    ///
    /// This check is required after loading a manifest and realization as
    /// separate content-addressed artifacts.
    ///
    /// # Errors
    /// Returns `EQ0901` for a digest/revision mismatch or execution provenance
    /// that contradicts typed realization policy.
    pub fn validate_against(
        &self,
        realization: &(impl CanonicalRealizationArtifact + ?Sized),
    ) -> Result<(), Diagnostic> {
        let realization = realization.artifact_reference()?;
        if &self.model() != realization.model_artifact()
            || self.wire.semantic_revision != realization.semantic_revision().get()
            || &self.realization() != realization.artifact()
        {
            return Err(invalid_artifact(
                "run manifest model/revision/realization linkage does not match the realization artifact",
            ));
        }
        let execution = self.execution();
        if execution.reduction() != realization.reduction() {
            return Err(invalid_artifact(
                "run reduction policy contradicts the realization solver plan",
            ));
        }
        match (
            realization.target(),
            realization.vector_layout(),
            realization.layout_artifacts(),
            execution.topology()?,
        ) {
            (
                Target::HostCpu { threads },
                VectorLayoutKind::Replicated,
                LayoutArtifacts::Replicated,
                ExecutionTopologyV1::Host { workers },
            ) if threads == workers => Ok(()),
            (
                Target::HostCpu { threads },
                VectorLayoutKind::Distributed,
                LayoutArtifacts::Distributed { .. },
                ExecutionTopologyV1::Distributed {
                    workers_per_partition,
                    ..
                },
            ) if threads == workers_per_partition => Ok(()),
            (
                Target::CudaGpu { device: planned },
                VectorLayoutKind::Replicated,
                LayoutArtifacts::Replicated,
                ExecutionTopologyV1::Cuda {
                    device: resolved, ..
                },
            ) if planned == resolved => Ok(()),
            _ => Err(invalid_artifact(
                "run topology contradicts the realization target, layout, or worker count",
            )),
        }
    }

    fn validate(&self) -> Result<(), Diagnostic> {
        if self.wire.schema != RUN_V2_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported run-manifest/v2 schema or canonical encoding",
            ));
        }
        ArtifactDigest::from_hex(self.wire.model_sha256.clone())?;
        ArtifactDigest::from_hex(self.wire.realization_sha256.clone())?;
        ExecutionProvenanceV1 {
            wire: self.wire.execution.clone(),
        }
        .validate()?;
        let mut ordered = self.wire.output_sha256.clone();
        for output in &ordered {
            ArtifactDigest::from_hex(output.clone())?;
        }
        ordered.sort();
        ordered.dedup();
        if ordered != self.wire.output_sha256 {
            return Err(invalid_artifact(
                "run manifest v2 outputs must be sorted and unique",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRunManifestV2 {
    schema: String,
    encoding: String,
    model_sha256: String,
    semantic_revision: u64,
    realization_sha256: String,
    execution: WireExecutionProvenanceV1,
    output_sha256: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireExecutionProvenanceV1 {
    adapter: String,
    adapter_version: String,
    solver_backend: String,
    solver_backend_version: String,
    libraries: BTreeMap<String, String>,
    topology: WireExecutionTopologyV1,
    reduction: WireReduction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireExecutionTopologyV1 {
    Host {
        workers: u64,
    },
    Distributed {
        partitions: u64,
        workers_per_partition: u64,
        transport: WireDistributedTransportV1,
    },
    Cuda {
        device: u16,
        device_name: String,
        compute_capability_major: u16,
        compute_capability_minor: u16,
        driver_version: String,
    },
}

impl WireExecutionTopologyV1 {
    fn encode(value: ExecutionTopologyV1) -> Result<Self, Diagnostic> {
        match value {
            ExecutionTopologyV1::Host { workers } => Ok(Self::Host {
                workers: encode_usize(workers.get(), "host workers")?,
            }),
            ExecutionTopologyV1::Distributed {
                partitions,
                workers_per_partition,
                transport,
            } => Ok(Self::Distributed {
                partitions: encode_usize(partitions.get(), "distributed partitions")?,
                workers_per_partition: encode_usize(
                    workers_per_partition.get(),
                    "workers per partition",
                )?,
                transport: WireDistributedTransportV1::encode(transport),
            }),
            ExecutionTopologyV1::Cuda {
                device,
                device_name,
                compute_capability_major,
                compute_capability_minor,
                driver_version,
            } => {
                validate_text("CUDA device name", &device_name)?;
                validate_text("CUDA driver version", &driver_version)?;
                if compute_capability_major == 0 {
                    return Err(invalid_artifact(
                        "CUDA compute-capability major version must be non-zero",
                    ));
                }
                Ok(Self::Cuda {
                    device,
                    device_name,
                    compute_capability_major,
                    compute_capability_minor,
                    driver_version,
                })
            }
        }
    }

    fn decode(&self) -> Result<ExecutionTopologyV1, Diagnostic> {
        match self {
            Self::Host { workers } => Ok(ExecutionTopologyV1::Host {
                workers: decode_nonzero_usize(*workers, "host workers")?,
            }),
            Self::Distributed {
                partitions,
                workers_per_partition,
                transport,
            } => Ok(ExecutionTopologyV1::Distributed {
                partitions: decode_nonzero_usize(*partitions, "distributed partitions")?,
                workers_per_partition: decode_nonzero_usize(
                    *workers_per_partition,
                    "workers per partition",
                )?,
                transport: transport.decode()?,
            }),
            Self::Cuda {
                device,
                device_name,
                compute_capability_major,
                compute_capability_minor,
                driver_version,
            } => {
                validate_text("CUDA device name", device_name)?;
                validate_text("CUDA driver version", driver_version)?;
                if *compute_capability_major == 0 {
                    return Err(invalid_artifact(
                        "CUDA compute-capability major version must be non-zero",
                    ));
                }
                Ok(ExecutionTopologyV1::Cuda {
                    device: *device,
                    device_name: device_name.clone(),
                    compute_capability_major: *compute_capability_major,
                    compute_capability_minor: *compute_capability_minor,
                    driver_version: driver_version.clone(),
                })
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireDistributedTransportV1 {
    Loopback,
    Mpi {
        implementation: String,
        version: String,
        thread_support: WireMpiThreadSupport,
    },
}

impl WireDistributedTransportV1 {
    fn encode(value: DistributedTransportV1) -> Self {
        match value {
            DistributedTransportV1::Loopback => Self::Loopback,
            DistributedTransportV1::Mpi {
                implementation,
                version,
                thread_support,
            } => Self::Mpi {
                implementation,
                version,
                thread_support: WireMpiThreadSupport::encode(thread_support),
            },
        }
    }

    fn decode(&self) -> Result<DistributedTransportV1, Diagnostic> {
        match self {
            Self::Loopback => Ok(DistributedTransportV1::Loopback),
            Self::Mpi {
                implementation,
                version,
                thread_support,
            } => {
                validate_text("MPI implementation", implementation)?;
                validate_text("MPI implementation version", version)?;
                Ok(DistributedTransportV1::Mpi {
                    implementation: implementation.clone(),
                    version: version.clone(),
                    thread_support: thread_support.decode(),
                })
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireMpiThreadSupport {
    Single,
    Funneled,
    Serialized,
    Multiple,
}

impl WireMpiThreadSupport {
    const fn encode(value: MpiThreadSupportV1) -> Self {
        match value {
            MpiThreadSupportV1::Single => Self::Single,
            MpiThreadSupportV1::Funneled => Self::Funneled,
            MpiThreadSupportV1::Serialized => Self::Serialized,
            MpiThreadSupportV1::Multiple => Self::Multiple,
        }
    }

    const fn decode(self) -> MpiThreadSupportV1 {
        match self {
            Self::Single => MpiThreadSupportV1::Single,
            Self::Funneled => MpiThreadSupportV1::Funneled,
            Self::Serialized => MpiThreadSupportV1::Serialized,
            Self::Multiple => MpiThreadSupportV1::Multiple,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireReduction {
    Reproducible,
    Fast,
}

impl WireReduction {
    const fn encode(value: ReductionPolicy) -> Self {
        match value {
            ReductionPolicy::Reproducible => Self::Reproducible,
            ReductionPolicy::Fast => Self::Fast,
        }
    }

    const fn decode(self) -> ReductionPolicy {
        match self {
            Self::Reproducible => ReductionPolicy::Reproducible,
            Self::Fast => ReductionPolicy::Fast,
        }
    }
}

fn encode_usize(value: usize, label: &str) -> Result<u64, Diagnostic> {
    u64::try_from(value).map_err(|_| invalid_artifact(format!("{label} exceeds wire u64")))
}

fn decode_nonzero_usize(value: u64, label: &str) -> Result<NonZeroUsize, Diagnostic> {
    let value = usize::try_from(value)
        .map_err(|_| invalid_artifact(format!("{label} exceeds local usize")))?;
    NonZeroUsize::new(value).ok_or_else(|| invalid_artifact(format!("{label} must be non-zero")))
}
