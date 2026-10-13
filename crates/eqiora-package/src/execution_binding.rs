use serde::{Deserialize, Serialize};

use crate::canonical;
use crate::{
    CanonicalModelDigest, CanonicalRealizationDigest, CanonicalRunDigest, ContractError,
    PackageCompilationDigest, PackageCompilationRecordV2, PackageExecutionBindingDigest,
};

const SCHEMA: &str = "eqiora.package-execution-binding.v1";
const ENCODING: &str = "eqiora.canonical-json.v1";
const MAX_WIRE_BYTES: usize = 8 * 1024;

/// Exact typed Realization schema named by package execution lineage v1.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum BoundRealizationSchemaV1 {
    /// The current portable-graph Realization artifact schema.
    #[serde(rename = "eqiora.realization-envelope/v11")]
    RealizationEnvelope,
}

/// Exact typed Run schema named by package execution lineage v1.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum BoundExecutionRunSchemaV1 {
    /// The typed Realization-linked Run manifest schema.
    #[serde(rename = "eqiora.run-manifest/v2")]
    RunManifestV2,
}

/// Content-addressed lineage from an exact package compilation through one
/// typed Realization to one typed Run.
///
/// The edge binds identities only. It neither attests execution nor proves
/// numerical acceptance. The composing application layer must validate the
/// concrete Model, Realization, and Run artifacts before constructing it.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PackageExecutionBindingV1 {
    schema: String,
    encoding: String,
    model_sha256: CanonicalModelDigest,
    semantic_revision: u64,
    package_compilation_sha256: PackageCompilationDigest,
    realization_schema: BoundRealizationSchemaV1,
    realization_sha256: CanonicalRealizationDigest,
    run_schema: BoundExecutionRunSchemaV1,
    run_sha256: CanonicalRunDigest,
}

impl PackageExecutionBindingV1 {
    /// Bind externally validated typed execution identities to one exact
    /// package compilation.
    pub fn new(
        compilation: &PackageCompilationRecordV2,
        semantic_revision: u64,
        realization_sha256: CanonicalRealizationDigest,
        run_sha256: CanonicalRunDigest,
    ) -> Result<Self, ContractError> {
        let binding = Self {
            schema: SCHEMA.to_owned(),
            encoding: ENCODING.to_owned(),
            model_sha256: compilation.model_digest(),
            semantic_revision,
            package_compilation_sha256: compilation.digest()?,
            realization_schema: BoundRealizationSchemaV1::RealizationEnvelope,
            realization_sha256,
            run_schema: BoundExecutionRunSchemaV1::RunManifestV2,
            run_sha256,
        };
        binding.validate_local()?;
        Ok(binding)
    }

    /// Decode and locally validate one bounded lineage edge.
    ///
    /// Concrete external artifacts must subsequently be supplied to
    /// [`Self::validate_against`].
    pub fn from_json(bytes: &[u8]) -> Result<Self, ContractError> {
        let binding = canonical::from_slice_with_limit::<Self>(bytes, MAX_WIRE_BYTES)?;
        binding.validate_local()?;
        Ok(binding)
    }

    /// Emit deterministic canonical JSON.
    pub fn canonical_json(&self) -> Result<Vec<u8>, ContractError> {
        self.validate_local()?;
        canonical::to_bytes_with_limit(self, MAX_WIRE_BYTES)
    }

    /// Domain-separated identity of the complete lineage edge.
    pub fn digest(&self) -> Result<PackageExecutionBindingDigest, ContractError> {
        Ok(PackageExecutionBindingDigest::compute(
            &self.canonical_json()?,
        ))
    }

    /// Replay every exact identity named by this edge.
    ///
    /// This verifies content linkage only. The caller remains responsible for
    /// validating the concrete typed artifacts before invoking this check.
    pub fn validate_against(
        &self,
        compilation: &PackageCompilationRecordV2,
        semantic_revision: u64,
        realization_sha256: CanonicalRealizationDigest,
        run_sha256: CanonicalRunDigest,
    ) -> Result<(), ContractError> {
        self.validate_local()?;
        if self.model_sha256 != compilation.model_digest()
            || self.semantic_revision != semantic_revision
            || self.package_compilation_sha256 != compilation.digest()?
            || self.realization_schema != BoundRealizationSchemaV1::RealizationEnvelope
            || self.realization_sha256 != realization_sha256
            || self.run_schema != BoundExecutionRunSchemaV1::RunManifestV2
            || self.run_sha256 != run_sha256
        {
            return Err(ContractError::new(
                "package execution binding does not match the supplied compilation, revision, Realization, and Run identities",
            ));
        }
        Ok(())
    }

    /// Canonical Model shared by compilation, Realization, and Run.
    #[must_use]
    pub const fn model_digest(&self) -> CanonicalModelDigest {
        self.model_sha256
    }

    /// Exact semantic revision shared by the typed artifacts.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.semantic_revision
    }

    /// Exact package-compilation record linked by this edge.
    #[must_use]
    pub const fn compilation_digest(&self) -> PackageCompilationDigest {
        self.package_compilation_sha256
    }

    /// Closed schema of the linked Realization.
    #[must_use]
    pub const fn realization_schema(&self) -> BoundRealizationSchemaV1 {
        self.realization_schema
    }

    /// Externally computed identity of the linked Realization.
    #[must_use]
    pub const fn realization_digest(&self) -> CanonicalRealizationDigest {
        self.realization_sha256
    }

    /// Closed schema of the linked typed Run.
    #[must_use]
    pub const fn run_schema(&self) -> BoundExecutionRunSchemaV1 {
        self.run_schema
    }

    /// Externally computed identity of the linked typed Run.
    #[must_use]
    pub const fn run_digest(&self) -> CanonicalRunDigest {
        self.run_sha256
    }

    fn validate_local(&self) -> Result<(), ContractError> {
        if self.schema != SCHEMA || self.encoding != ENCODING {
            return Err(ContractError::new(
                "unsupported package execution-binding schema or encoding",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compilation_with(
        model_byte: &str,
        resolution_byte: &str,
        source_byte: &str,
        compiler_version: &str,
    ) -> PackageCompilationRecordV2 {
        let model = model_byte.repeat(32);
        let resolution = resolution_byte.repeat(32);
        let semantic = "12".repeat(32);
        let source = source_byte.repeat(32);
        let json = format!(
            r#"{{"schema":"eqiora.package-compilation.v2","encoding":"eqiora.canonical-json.v1","model_sha256":"{model}","root":{{"name":"org.example.Main","version":"1.0.0","semantic_digest":"{semantic}"}},"resolution_digest":"{resolution}","packages":[{{"package":{{"name":"org.example.Main","version":"1.0.0","semantic_digest":"{semantic}"}},"source_digest":"{source}"}}],"toolchain":{{"compiler":"Eqiora.Compiler","compiler_version":"{compiler_version}","semantic_canonicalization_version":2,"source_bundle_version":1,"resolution_version":1}}}}"#,
        );
        PackageCompilationRecordV2::from_json(json.as_bytes()).expect("compilation")
    }

    fn compilation() -> PackageCompilationRecordV2 {
        compilation_with("56", "78", "34", "0.1.0")
    }

    fn realization(byte: &str) -> CanonicalRealizationDigest {
        CanonicalRealizationDigest::parse(&byte.repeat(32)).expect("Realization digest")
    }

    fn run(byte: &str) -> CanonicalRunDigest {
        CanonicalRunDigest::parse(&byte.repeat(32)).expect("Run digest")
    }

    #[test]
    fn binding_round_trips_and_replays_all_exact_identities() {
        let compilation = compilation();
        let binding = PackageExecutionBindingV1::new(&compilation, 7, realization("9a"), run("bc"))
            .expect("binding");
        let bytes = binding.canonical_json().expect("canonical JSON");
        let decoded = PackageExecutionBindingV1::from_json(&bytes).expect("decoded binding");

        assert_eq!(decoded, binding);
        assert_eq!(decoded.model_digest(), compilation.model_digest());
        assert_eq!(decoded.semantic_revision(), 7);
        assert_eq!(decoded.compilation_digest(), compilation.digest().unwrap());
        assert_eq!(
            decoded.realization_schema(),
            BoundRealizationSchemaV1::RealizationEnvelope
        );
        assert_eq!(decoded.realization_digest(), realization("9a"));
        assert_eq!(
            decoded.run_schema(),
            BoundExecutionRunSchemaV1::RunManifestV2
        );
        assert_eq!(decoded.run_digest(), run("bc"));
        decoded
            .validate_against(&compilation, 7, realization("9a"), run("bc"))
            .expect("exact replay");
    }

    #[test]
    fn every_identity_substitution_fails_closed() {
        let compilation = compilation();
        let binding = PackageExecutionBindingV1::new(&compilation, 7, realization("9a"), run("bc"))
            .expect("binding");
        let changed_model = compilation_with("ab", "78", "34", "0.1.0");
        let changed_compilation = compilation_with("56", "ab", "34", "0.1.0");
        let changed_source = compilation_with("56", "78", "cd", "0.1.0");
        let changed_toolchain = compilation_with("56", "78", "34", "0.2.0");

        for changed in [&changed_compilation, &changed_source, &changed_toolchain] {
            assert_eq!(compilation.model_digest(), changed.model_digest());
            assert_ne!(compilation.digest(), changed.digest());
            assert!(
                binding
                    .validate_against(changed, 7, realization("9a"), run("bc"))
                    .is_err()
            );
        }
        assert!(
            binding
                .validate_against(&compilation, 8, realization("9a"), run("bc"))
                .is_err()
        );
        assert_ne!(compilation.model_digest(), changed_model.model_digest());
        assert!(
            binding
                .validate_against(&changed_model, 7, realization("9a"), run("bc"))
                .is_err()
        );
        assert!(
            binding
                .validate_against(&compilation, 7, realization("de"), run("bc"))
                .is_err()
        );
        assert!(
            binding
                .validate_against(&compilation, 7, realization("9a"), run("ef"))
                .is_err()
        );
    }

    #[test]
    fn malformed_unknown_and_oversized_wires_fail_closed() {
        let binding =
            PackageExecutionBindingV1::new(&compilation(), 7, realization("9a"), run("bc"))
                .expect("binding");
        let bytes = String::from_utf8(binding.canonical_json().expect("JSON")).expect("UTF-8");

        assert!(
            PackageExecutionBindingV1::from_json(
                bytes
                    .replace("eqiora.run-manifest/v2", "eqiora.run-manifest/v3")
                    .as_bytes()
            )
            .is_err()
        );
        assert!(
            PackageExecutionBindingV1::from_json(
                bytes
                    .replace(
                        "eqiora.realization-envelope/v11",
                        "eqiora.realization-envelope/v2"
                    )
                    .as_bytes()
            )
            .is_err()
        );
        assert!(
            PackageExecutionBindingV1::from_json(
                bytes
                    .replace(
                        "eqiora.package-execution-binding.v1",
                        "eqiora.package-execution-binding.v2"
                    )
                    .as_bytes()
            )
            .is_err()
        );
        assert!(
            PackageExecutionBindingV1::from_json(
                bytes.replacen('{', "{\"payload\":null,", 1).as_bytes()
            )
            .is_err()
        );
        assert!(
            PackageExecutionBindingV1::from_json(
                bytes.replace(&"bc".repeat(32), &"BC".repeat(32)).as_bytes()
            )
            .is_err()
        );
        assert!(PackageExecutionBindingV1::from_json(&vec![b' '; MAX_WIRE_BYTES + 1]).is_err());
    }
}
