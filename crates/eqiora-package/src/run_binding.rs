use serde::{Deserialize, Serialize};

use crate::canonical;
use crate::{
    CanonicalModelDigest, CanonicalRunDigest, ContractError, PackageCompilationDigest,
    PackageCompilationRecordV2, PackageRunBindingDigest,
};

const SCHEMA: &str = "eqiora.package-run-binding.v1";
const ENCODING: &str = "eqiora.canonical-json.v1";
const MAX_WIRE_BYTES: usize = 4 * 1024;

/// Exact run-manifest schema whose externally computed identity is bound.
///
/// This is deliberately closed rather than an arbitrary string. Supporting a
/// different run family requires an explicit package-lineage contract change.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum BoundRunManifestSchemaV1 {
    /// The original portable run-manifest schema.
    #[serde(rename = "eqiora.run-manifest/v1")]
    RunManifestV1,
}

/// Content-addressed lineage edge from one exact package compilation to one
/// canonical run manifest.
///
/// This record binds identities only. It does not independently prove that an
/// execution occurred or that its results were accepted. The composing layer
/// must validate the linked run under its own contract before constructing or
/// accepting this edge.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PackageRunBindingV1 {
    schema: String,
    encoding: String,
    model_sha256: CanonicalModelDigest,
    package_compilation_sha256: PackageCompilationDigest,
    run_schema: BoundRunManifestSchemaV1,
    run_sha256: CanonicalRunDigest,
}

impl PackageRunBindingV1 {
    /// Bind an externally validated run identity to one exact package
    /// compilation.
    ///
    /// This constructor derives the model and compilation identities from the
    /// compilation record. The caller remains responsible for proving that the
    /// supplied run schema and digest came from a run over that same model.
    pub fn new(
        compilation: &PackageCompilationRecordV2,
        run_schema: BoundRunManifestSchemaV1,
        run_sha256: CanonicalRunDigest,
    ) -> Result<Self, ContractError> {
        let binding = Self {
            schema: SCHEMA.to_owned(),
            encoding: ENCODING.to_owned(),
            model_sha256: compilation.model_digest(),
            package_compilation_sha256: compilation.digest()?,
            run_schema,
            run_sha256,
        };
        binding.validate_local()?;
        Ok(binding)
    }

    /// Decode and locally validate one bounded lineage edge.
    ///
    /// External compilation and run artifacts must subsequently be supplied
    /// to [`Self::validate_against`].
    pub fn from_json(bytes: &[u8]) -> Result<Self, ContractError> {
        let binding = canonical::from_slice_with_limit::<Self>(bytes, MAX_WIRE_BYTES)?;
        binding.validate_local()?;
        Ok(binding)
    }

    /// Emit deterministic canonical JSON under the binding-specific byte
    /// bound.
    pub fn canonical_json(&self) -> Result<Vec<u8>, ContractError> {
        self.validate_local()?;
        canonical::to_bytes_with_limit(self, MAX_WIRE_BYTES)
    }

    /// Domain-separated identity of the complete lineage edge.
    pub fn digest(&self) -> Result<PackageRunBindingDigest, ContractError> {
        Ok(PackageRunBindingDigest::compute(&self.canonical_json()?))
    }

    /// Replay the exact compilation and run identities named by this edge.
    ///
    /// This verifies content linkage, not execution acceptance. A higher layer
    /// must first validate the concrete run artifact and its model linkage.
    pub fn validate_against(
        &self,
        compilation: &PackageCompilationRecordV2,
        run_schema: BoundRunManifestSchemaV1,
        run_sha256: CanonicalRunDigest,
    ) -> Result<(), ContractError> {
        self.validate_local()?;
        if self.model_sha256 != compilation.model_digest()
            || self.package_compilation_sha256 != compilation.digest()?
            || self.run_schema != run_schema
            || self.run_sha256 != run_sha256
        {
            return Err(ContractError::new(
                "package run binding does not match the supplied compilation and run identities",
            ));
        }
        Ok(())
    }

    /// Canonical model shared by the package compilation and linked run.
    #[must_use]
    pub const fn model_digest(&self) -> CanonicalModelDigest {
        self.model_sha256
    }

    /// Exact package-compilation record linked by this edge.
    #[must_use]
    pub const fn compilation_digest(&self) -> PackageCompilationDigest {
        self.package_compilation_sha256
    }

    /// Closed schema of the linked run manifest.
    #[must_use]
    pub const fn run_schema(&self) -> BoundRunManifestSchemaV1 {
        self.run_schema
    }

    /// Externally computed identity of the linked run manifest.
    #[must_use]
    pub const fn run_digest(&self) -> CanonicalRunDigest {
        self.run_sha256
    }

    fn validate_local(&self) -> Result<(), ContractError> {
        if self.schema != SCHEMA || self.encoding != ENCODING {
            return Err(ContractError::new(
                "unsupported package run-binding schema or encoding",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_support::compilation_with;

    fn compilation(model_byte: &str, resolution_byte: &str) -> PackageCompilationRecordV2 {
        compilation_with(model_byte, resolution_byte, "34", "0.1.0")
    }

    #[test]
    fn binding_round_trips_and_replays_exact_identities() {
        let compilation = compilation("56", "78");
        let run = CanonicalRunDigest::parse(&"9a".repeat(32)).expect("run");
        let binding =
            PackageRunBindingV1::new(&compilation, BoundRunManifestSchemaV1::RunManifestV1, run)
                .expect("binding");
        let bytes = binding.canonical_json().expect("canonical JSON");
        let decoded = PackageRunBindingV1::from_json(&bytes).expect("decoded binding");

        assert_eq!(decoded, binding);
        assert_eq!(decoded.model_digest(), compilation.model_digest());
        assert_eq!(decoded.compilation_digest(), compilation.digest().unwrap());
        assert_eq!(decoded.run_digest(), run);
        assert_eq!(decoded.digest(), binding.digest());
        decoded
            .validate_against(&compilation, BoundRunManifestSchemaV1::RunManifestV1, run)
            .expect("exact replay");
    }

    #[test]
    fn malformed_unknown_and_oversized_wires_fail_closed() {
        let compilation = compilation("56", "78");
        let run = CanonicalRunDigest::parse(&"9a".repeat(32)).expect("run");
        let binding =
            PackageRunBindingV1::new(&compilation, BoundRunManifestSchemaV1::RunManifestV1, run)
                .expect("binding");
        let bytes = String::from_utf8(binding.canonical_json().expect("JSON")).expect("UTF-8");

        let uppercase_digest = bytes.replace(&"9a".repeat(32), &"9A".repeat(32));
        assert!(PackageRunBindingV1::from_json(uppercase_digest.as_bytes()).is_err());

        let unknown_schema = bytes.replace("eqiora.run-manifest/v1", "eqiora.run-manifest/v3");
        assert!(PackageRunBindingV1::from_json(unknown_schema.as_bytes()).is_err());

        let unknown_binding_schema = bytes.replace(
            "eqiora.package-run-binding.v1",
            "eqiora.package-run-binding.v2",
        );
        assert!(PackageRunBindingV1::from_json(unknown_binding_schema.as_bytes()).is_err());

        let unknown_field = bytes.replacen('{', "{\"payload\":null,", 1);
        assert!(PackageRunBindingV1::from_json(unknown_field.as_bytes()).is_err());

        assert!(PackageRunBindingV1::from_json(&vec![b' '; MAX_WIRE_BYTES + 1]).is_err());
    }

    #[test]
    fn replay_rejects_any_changed_compilation_identity_even_for_the_same_model() {
        let original = compilation("56", "78");
        let changed_resolution = compilation("56", "ab");
        let changed_source = compilation_with("56", "78", "cd", "0.1.0");
        let changed_toolchain = compilation_with("56", "78", "34", "0.2.0");

        let run = CanonicalRunDigest::parse(&"9a".repeat(32)).expect("run");
        let binding =
            PackageRunBindingV1::new(&original, BoundRunManifestSchemaV1::RunManifestV1, run)
                .expect("binding");
        for changed in [&changed_resolution, &changed_source, &changed_toolchain] {
            assert_eq!(original.model_digest(), changed.model_digest());
            assert_ne!(original.digest(), changed.digest());
            assert!(
                binding
                    .validate_against(changed, BoundRunManifestSchemaV1::RunManifestV1, run,)
                    .is_err()
            );
        }
    }
}
