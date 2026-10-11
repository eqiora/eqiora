//! Value-free post-run bindings for durable physical exposure projections.

use eqiora_core::Diagnostic;
use serde::{Deserialize, Serialize};

use crate::{
    ArtifactDigest, CANONICAL_ENCODING, JsonDecoderLimits, RunManifestV1, RunManifestV2,
    check_json_limits, invalid_artifact,
};

use super::PhysicalExposureCatalogEnvelopeV1;

const OBSERVATION_SCHEMA: &str = "eqiora.physical-exposure-observation-binding/v1";

/// Closed mathematical quantity projected from one physical exposure cut.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalExposureQuantityV1 {
    /// Common across value or field trace on the maximal connection class.
    Common,
    /// Sum of through values or outward flux over the exact interior cut.
    NetOutward,
}

/// Post-run lineage binding from one typed projection to one output artifact.
///
/// The binding is intentionally value-free. It says which output a producer
/// designated for one exact projection quantity; it does not claim numerical
/// acceptance and cannot introduce a cycle into the Run manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicalExposureObservationBindingV1 {
    wire: WirePhysicalExposureObservationV1,
}

impl PhysicalExposureObservationBindingV1 {
    /// Bind one catalog projection and quantity to an existing Run v1 output.
    ///
    /// # Errors
    /// Returns `EQ0901` for missing projection, wrong Model/revision, or an
    /// output digest not registered by the Run.
    pub fn new_v1(
        catalog: &PhysicalExposureCatalogEnvelopeV1,
        projection: ArtifactDigest,
        quantity: PhysicalExposureQuantityV1,
        run: &RunManifestV1,
        result: ArtifactDigest,
    ) -> Result<Self, Diagnostic> {
        Self::new_inner(
            catalog,
            projection,
            quantity,
            ResolvedRunReferenceV1::from_v1(run)?,
            result,
        )
    }

    /// Bind one catalog projection and quantity to an existing typed Run v2
    /// output.
    ///
    /// # Errors
    /// Returns `EQ0901` for missing projection, wrong Model/revision, or an
    /// output digest not registered by the Run.
    pub fn new_v2(
        catalog: &PhysicalExposureCatalogEnvelopeV1,
        projection: ArtifactDigest,
        quantity: PhysicalExposureQuantityV1,
        run: &RunManifestV2,
        result: ArtifactDigest,
    ) -> Result<Self, Diagnostic> {
        Self::new_inner(
            catalog,
            projection,
            quantity,
            ResolvedRunReferenceV1::from_v2(run)?,
            result,
        )
    }

    fn new_inner(
        catalog: &PhysicalExposureCatalogEnvelopeV1,
        projection: ArtifactDigest,
        quantity: PhysicalExposureQuantityV1,
        run: ResolvedRunReferenceV1,
        result: ArtifactDigest,
    ) -> Result<Self, Diagnostic> {
        if catalog.projection(&projection).is_none() {
            return Err(invalid_artifact(
                "physical observation references no catalog projection",
            ));
        }
        if run.model != catalog.model_artifact()
            || run.semantic_revision != catalog.semantic_revision()
            || !run.outputs.contains(&result)
        {
            return Err(invalid_artifact(
                "physical observation Run Model/revision or result output differs",
            ));
        }
        let binding = Self {
            wire: WirePhysicalExposureObservationV1 {
                schema: OBSERVATION_SCHEMA.to_owned(),
                encoding: CANONICAL_ENCODING.to_owned(),
                model_sha256: catalog.model_artifact().to_string(),
                semantic_revision: catalog.semantic_revision(),
                catalog_sha256: catalog.digest()?.to_string(),
                projection_sha256: projection.to_string(),
                quantity: WirePhysicalExposureQuantityV1::encode(quantity),
                run: run.wire,
                result_sha256: result.to_string(),
            },
        };
        binding.validate_local()?;
        Ok(binding)
    }

    /// Decode and locally validate a binding.
    ///
    /// # Errors
    /// Returns `EQ0901` for malformed or unknown-version data.
    pub fn from_json(bytes: &[u8], limits: JsonDecoderLimits) -> Result<Self, Diagnostic> {
        check_json_limits(bytes, limits)?;
        let wire = serde_json::from_slice(bytes).map_err(|error| {
            invalid_artifact(format!(
                "invalid physical observation binding JSON: {error}"
            ))
        })?;
        let binding = Self { wire };
        binding.validate_local()?;
        Ok(binding)
    }

    /// Canonical JSON bytes.
    ///
    /// # Errors
    /// Returns `EQ0901` if serialization unexpectedly fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&self.wire).map_err(|error| {
            invalid_artifact(format!(
                "cannot serialize physical observation binding: {error}"
            ))
        })
    }

    /// Domain-separated content identity.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<ArtifactDigest, Diagnostic> {
        Ok(ArtifactDigest::compute(
            OBSERVATION_SCHEMA.as_bytes(),
            &self.canonical_json()?,
        ))
    }

    /// Independently replay every catalog, Run v1, and result-output link.
    ///
    /// # Errors
    /// Returns `EQ0901` for any stale or mismatched identity.
    pub fn validate_against_v1(
        &self,
        catalog: &PhysicalExposureCatalogEnvelopeV1,
        run: &RunManifestV1,
    ) -> Result<(), Diagnostic> {
        self.validate_against_inner(catalog, ResolvedRunReferenceV1::from_v1(run)?)
    }

    /// Independently replay every catalog, typed Run v2, and result-output
    /// link.
    ///
    /// # Errors
    /// Returns `EQ0901` for any stale or mismatched identity.
    pub fn validate_against_v2(
        &self,
        catalog: &PhysicalExposureCatalogEnvelopeV1,
        run: &RunManifestV2,
    ) -> Result<(), Diagnostic> {
        self.validate_against_inner(catalog, ResolvedRunReferenceV1::from_v2(run)?)
    }

    fn validate_against_inner(
        &self,
        catalog: &PhysicalExposureCatalogEnvelopeV1,
        run: ResolvedRunReferenceV1,
    ) -> Result<(), Diagnostic> {
        self.validate_local()?;
        let result = ArtifactDigest::from_hex(self.wire.result_sha256.clone())?;
        if self.wire.model_sha256 != catalog.model_artifact().as_str()
            || self.wire.semantic_revision != catalog.semantic_revision()
            || self.wire.catalog_sha256 != catalog.digest()?.as_str()
            || catalog
                .projection(&ArtifactDigest::from_hex(
                    self.wire.projection_sha256.clone(),
                )?)
                .is_none()
            || self.wire.run != run.wire
            || run.model != catalog.model_artifact()
            || run.semantic_revision != catalog.semantic_revision()
            || !run.outputs.contains(&result)
        {
            return Err(invalid_artifact(
                "physical observation catalog, projection, Run, or result linkage differs",
            ));
        }
        Ok(())
    }

    /// Projected common or net-outward quantity.
    #[must_use]
    pub const fn quantity(&self) -> PhysicalExposureQuantityV1 {
        self.wire.quantity.decode()
    }

    /// Bound output artifact digest.
    #[must_use]
    pub fn result(&self) -> ArtifactDigest {
        ArtifactDigest(self.wire.result_sha256.clone())
    }

    fn validate_local(&self) -> Result<(), Diagnostic> {
        if self.wire.schema != OBSERVATION_SCHEMA || self.wire.encoding != CANONICAL_ENCODING {
            return Err(invalid_artifact(
                "unsupported physical-exposure-observation schema or canonical encoding",
            ));
        }
        for digest in [
            &self.wire.model_sha256,
            &self.wire.catalog_sha256,
            &self.wire.projection_sha256,
            &self.wire.result_sha256,
        ] {
            ArtifactDigest::from_hex(digest.clone())?;
        }
        self.wire.run.validate()?;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct WirePhysicalExposureObservationV1 {
    schema: String,
    encoding: String,
    model_sha256: String,
    semantic_revision: u64,
    catalog_sha256: String,
    projection_sha256: String,
    quantity: WirePhysicalExposureQuantityV1,
    run: WireRunReferenceV1,
    result_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum WireRunReferenceV1 {
    RunManifestV1 { sha256: String },
    RunManifestV2 { sha256: String },
}

impl WireRunReferenceV1 {
    fn validate(&self) -> Result<(), Diagnostic> {
        let sha256 = match self {
            Self::RunManifestV1 { sha256 } | Self::RunManifestV2 { sha256 } => sha256,
        };
        ArtifactDigest::from_hex(sha256.clone()).map(|_| ())
    }
}

struct ResolvedRunReferenceV1 {
    wire: WireRunReferenceV1,
    model: ArtifactDigest,
    semantic_revision: u64,
    outputs: Vec<ArtifactDigest>,
}

impl ResolvedRunReferenceV1 {
    fn from_v1(run: &RunManifestV1) -> Result<Self, Diagnostic> {
        Ok(Self {
            wire: WireRunReferenceV1::RunManifestV1 {
                sha256: run.digest()?.to_string(),
            },
            model: run.model(),
            semantic_revision: run.semantic_revision(),
            outputs: run.outputs(),
        })
    }

    fn from_v2(run: &RunManifestV2) -> Result<Self, Diagnostic> {
        Ok(Self {
            wire: WireRunReferenceV1::RunManifestV2 {
                sha256: run.digest()?.to_string(),
            },
            model: run.model(),
            semantic_revision: run.semantic_revision(),
            outputs: run.outputs(),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WirePhysicalExposureQuantityV1 {
    Common,
    NetOutward,
}

impl WirePhysicalExposureQuantityV1 {
    const fn encode(value: PhysicalExposureQuantityV1) -> Self {
        match value {
            PhysicalExposureQuantityV1::Common => Self::Common,
            PhysicalExposureQuantityV1::NetOutward => Self::NetOutward,
        }
    }

    const fn decode(self) -> PhysicalExposureQuantityV1 {
        match self {
            Self::Common => PhysicalExposureQuantityV1::Common,
            Self::NetOutward => PhysicalExposureQuantityV1::NetOutward,
        }
    }
}
