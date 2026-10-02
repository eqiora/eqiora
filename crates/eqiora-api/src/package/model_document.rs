use eqiora_artifact::{
    ArtifactDigest, PhysicalExposureCatalogEnvelopeV1, PhysicalExposureObservationBindingV1,
    PhysicalExposureProjectionV1, PhysicalExposureQuantityV1, PhysicalExposureSourceOriginV1,
    PhysicalExposureSourceSpanV1, RealizationEnvelopeV1, RunManifestV1, RunManifestV2,
};
use eqiora_compiler::projection::{PhysicalExposureContract, PhysicalExposureProjectionMap};
use eqiora_compiler::provenance::ProvenanceMap;
use eqiora_compiler::{AnalyzedResolvedHierarchy, analyze_resolved_hierarchy};
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_package::{
    BoundRunManifestSchemaV1, CanonicalModelDigest, CanonicalRealizationDigest, CanonicalRunDigest,
    CompilationToolchainV2, ExactResolver, ExactVersion, PackageCompilationRecordV2,
    PackageExecutionBindingV1, PackageRunBindingV1, PackageStore, QualifiedName,
    ResolutionRecordV1,
};
use eqiora_schema::kernel::PropertyMeaning;

use super::{
    PackageCompilationError, PackageExecutionBindingError, PackageRunBindingError,
    compilation_namespaces, compiler_input, verify_semantic_content,
};
use crate::ModelDocument;

const COMPILER_IDENTITY: &str = "Eqiora.Compiler";

fn collect_property_bindings(
    analyzed: &AnalyzedResolvedHierarchy,
) -> Box<[PropertyBindingProjection]> {
    analyzed
        .property_bindings()
        .map(
            |(
                composition,
                contract,
                release,
                component,
                requirement,
                meaning,
                validity,
                citation,
                license,
            )| {
                PropertyBindingProjection {
                    composition: composition.map(str::to_owned),
                    contract: contract.to_owned(),
                    release: release.to_owned(),
                    component: component.to_owned(),
                    requirement: requirement.to_owned(),
                    meaning: meaning.clone(),
                    validity: validity.to_owned(),
                    citation: citation.to_owned(),
                    license: license.to_owned(),
                }
            },
        )
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

/// One admitted canonical model plus exact package compilation provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct PackagedModelDocument {
    model: ModelDocument,
    compilation: PackageCompilationRecordV2,
    provenance: ProvenanceMap,
    physical_exposures: PhysicalExposureProjectionMap,
    physical_exposure_catalog: Option<PhysicalExposureCatalogEnvelopeV1>,
    property_bindings: Box<[PropertyBindingProjection]>,
}

#[derive(Debug, Clone, PartialEq)]
struct PropertyBindingProjection {
    composition: Option<String>,
    contract: String,
    release: String,
    component: String,
    requirement: String,
    meaning: PropertyMeaning,
    validity: String,
    citation: String,
    license: String,
}

impl PackagedModelDocument {
    /// Resolve a locked package graph offline, verify source semantics, and
    /// compile one root-local or directly imported public Model.
    ///
    /// Resolution has no discovery, version selection, network, environment,
    /// or fallback path. Every source unit is analyzed and compared with its
    /// release's canonical semantic content before the selected root is
    /// elaborated or any graph transaction is committed.
    /// `entry_model` is either a root-local name or exactly `alias.Model` for
    /// one direct dependency declared by the root package.
    ///
    /// # Errors
    /// Returns the original resolver error, package contract error, complete
    /// compiler diagnostic set, or a typed semantic-content mismatch.
    pub fn compile_locked(
        store: &impl PackageStore,
        resolution: &ResolutionRecordV1,
        entry_model: &str,
    ) -> Result<Self, PackageCompilationError> {
        Self::compile_selected(store, resolution, entry_model, &[])
    }

    /// Compile a selected locked Model or Component with exact static bindings.
    ///
    /// # Errors
    /// Returns package, signature binding, compiler, or artifact diagnostics.
    pub fn compile_selected(
        store: &impl PackageStore,
        resolution: &ResolutionRecordV1,
        entry: &str,
        bindings: &[(&str, eqiora_compiler::StaticBindingValue<'_>)],
    ) -> Result<Self, PackageCompilationError> {
        let resolved = ExactResolver.resolve(resolution, store)?;
        let namespaces = compilation_namespaces(&resolved)?;
        let input = compiler_input(&resolved, &namespaces)?;
        let analyzed = analyze_resolved_hierarchy(input)?;
        verify_semantic_content(&resolved, &namespaces, &analyzed)?;
        let property_bindings = collect_property_bindings(&analyzed);
        let validated = analyzed.validate_definitions()?;

        let compiled = validated.compile_selected(entry, bindings)?;
        let provenance = compiled.provenance().cloned().ok_or_else(|| {
            PackageCompilationError::Diagnostics(vec![Diagnostic::error(
                codes::LANGUAGE_LOWERING_ERROR,
                "locked package compilation did not produce hierarchy provenance",
            )])
        })?;
        let physical_exposures = compiled.physical_exposures().clone();
        let model = ModelDocument::accept_bound_compiled(compiled, bindings)?;
        let model_digest = CanonicalModelDigest::parse(&model.digest()?)?;
        let toolchain = CompilationToolchainV2::new(
            QualifiedName::parse(COMPILER_IDENTITY)?,
            ExactVersion::parse(env!("CARGO_PKG_VERSION"))?,
        );
        let compilation = PackageCompilationRecordV2::new(model_digest, &resolved, toolchain)?;
        let physical_exposure_catalog = (!physical_exposures.is_empty())
            .then(|| {
                seal_physical_exposure_catalog(
                    &model,
                    &compilation,
                    &provenance,
                    &physical_exposures,
                )
            })
            .transpose()?;

        Ok(Self {
            model,
            compilation,
            provenance,
            physical_exposures,
            physical_exposure_catalog,
            property_bindings,
        })
    }

    /// Canonical model admitted through the ordinary application boundary.
    #[must_use]
    pub const fn model(&self) -> &ModelDocument {
        &self.model
    }

    /// Exact resolution, package inventory, toolchain, and model digest.
    #[must_use]
    pub const fn compilation(&self) -> &PackageCompilationRecordV2 {
        &self.compilation
    }

    /// Package-qualified definition, instance, and binding source locations.
    #[must_use]
    pub const fn provenance(&self) -> &ProvenanceMap {
        &self.provenance
    }

    /// Exact nominal property bindings used by this compilation.
    ///
    /// Each item is `(composition, contract, release, component, requirement,
    /// meaning, validity, citation, license)`.
    pub fn property_bindings(
        &self,
    ) -> impl ExactSizeIterator<
        Item = (
            Option<&str>,
            &str,
            &str,
            &str,
            &str,
            &PropertyMeaning,
            &str,
            &str,
            &str,
        ),
    > {
        self.property_bindings.iter().map(|value| {
            (
                value.composition.as_deref(),
                value.contract.as_str(),
                value.release.as_str(),
                value.component.as_str(),
                value.requirement.as_str(),
                &value.meaning,
                value.validity.as_str(),
                value.citation.as_str(),
                value.license.as_str(),
            )
        })
    }

    /// Versioned, content-addressed observation cuts for every ownerless
    /// public physical Port eliminated during exact package compilation.
    #[must_use]
    pub const fn physical_exposure_catalog(&self) -> Option<&PhysicalExposureCatalogEnvelopeV1> {
        self.physical_exposure_catalog.as_ref()
    }

    /// Independently replay a decoded physical exposure catalog against the
    /// exact package compilation and compiler-derived occurrence cuts.
    ///
    /// A flat Kernel graph alone cannot reconstruct which proper subset came
    /// from an eliminated hierarchy occurrence. Replay therefore checks the
    /// resolution and recompilation-owned sidecars, not only graph membership.
    ///
    /// # Errors
    /// Returns a typed resolution/package failure or `EQ0901` diagnostic for
    /// any stale Model, source lineage, occurrence cut, contract, or source
    /// provenance.
    pub fn validate_physical_exposure_catalog(
        &self,
        catalog: &PhysicalExposureCatalogEnvelopeV1,
        resolution: &ResolutionRecordV1,
    ) -> Result<(), PackageCompilationError> {
        self.compilation.validate_against(resolution)?;
        if self.physical_exposures.is_empty() {
            return Err(Diagnostic::error(
                codes::INVALID_ARTIFACT,
                "package compilation has no eliminated physical exposure catalog",
            )
            .into());
        }
        let expected = seal_physical_exposure_catalog(
            &self.model,
            &self.compilation,
            &self.provenance,
            &self.physical_exposures,
        )?;
        catalog.validate_against(
            expected.model_artifact(),
            self.model.program(),
            expected.package_compilation(),
        )?;
        if catalog != &expected {
            return Err(Diagnostic::error(
                codes::INVALID_ARTIFACT,
                "physical exposure catalog differs from exact compiler projection or source provenance",
            )
            .into());
        }
        Ok(())
    }

    /// Bind one cataloged physical exposure quantity to an output of an exact
    /// package-linked Run v1.
    ///
    /// This is a value-free lineage operation. The output artifact must
    /// already be listed by the Run; numerical acceptance and payload typing
    /// remain the responsibility of the producing result adapter.
    ///
    /// # Errors
    /// Returns a typed package/Run linkage or artifact failure for a missing
    /// catalog, projection, exact Model/revision mismatch, or absent output.
    pub fn bind_physical_observation_v1(
        &self,
        projection: ArtifactDigest,
        quantity: PhysicalExposureQuantityV1,
        run: &RunManifestV1,
        result: ArtifactDigest,
    ) -> Result<PhysicalExposureObservationBindingV1, PackageRunBindingError> {
        self.validate_run_v1_model(run)?;
        let catalog = self.physical_exposure_catalog.as_ref().ok_or_else(|| {
            Diagnostic::error(
                codes::INVALID_ARTIFACT,
                "package compilation has no eliminated physical exposure catalog",
            )
        })?;
        PhysicalExposureObservationBindingV1::new_v1(catalog, projection, quantity, run, result)
            .map_err(Into::into)
    }

    /// Independently replay one physical observation binding against the
    /// exact package resolution, sealed catalog, Run, and registered output.
    ///
    /// # Errors
    /// Returns a typed package/Run linkage or artifact failure for any stale
    /// source lineage, catalog, projection, Run, or output identity.
    pub fn validate_physical_observation_v1(
        &self,
        binding: &PhysicalExposureObservationBindingV1,
        run: &RunManifestV1,
        resolution: &ResolutionRecordV1,
    ) -> Result<(), PackageRunBindingError> {
        self.compilation.validate_against(resolution)?;
        self.validate_run_v1_model(run)?;
        let catalog = self.physical_exposure_catalog.as_ref().ok_or_else(|| {
            Diagnostic::error(
                codes::INVALID_ARTIFACT,
                "package compilation has no eliminated physical exposure catalog",
            )
        })?;
        binding
            .validate_against_v1(catalog, run)
            .map_err(Into::into)
    }

    /// Bind one caller-designated v1 Run manifest to this exact package
    /// compilation identity.
    ///
    /// The Run must reference the admitted Model exactly. This constructs a
    /// content-addressed lineage edge; it does not prove that an execution
    /// occurred or that the caller accepted it. Evidence producers should
    /// invoke it only after their independent acceptance checks succeed.
    ///
    /// # Errors
    /// Returns a typed artifact, package-contract, or Model-linkage failure.
    pub fn bind_run_v1(
        &self,
        run: &RunManifestV1,
    ) -> Result<PackageRunBindingV1, PackageRunBindingError> {
        let run_digest = self.validate_run_v1_model(run)?;
        PackageRunBindingV1::new(
            &self.compilation,
            BoundRunManifestSchemaV1::RunManifestV1,
            run_digest,
        )
        .map_err(Into::into)
    }

    /// Independently replay an exact v1 Run lineage edge.
    ///
    /// This revalidates the complete resolution record against the package
    /// compilation, the admitted Model against the compilation, and the Run's
    /// Model and canonical digest against the binding.
    ///
    /// # Errors
    /// Returns a typed artifact, package-contract, or Model-linkage failure.
    pub fn validate_run_v1_binding(
        &self,
        binding: &PackageRunBindingV1,
        run: &RunManifestV1,
        resolution: &ResolutionRecordV1,
    ) -> Result<(), PackageRunBindingError> {
        self.compilation.validate_against(resolution)?;
        let run_digest = self.validate_run_v1_model(run)?;
        binding.validate_against(
            &self.compilation,
            BoundRunManifestSchemaV1::RunManifestV1,
            run_digest,
        )?;
        Ok(())
    }

    /// Bind one typed Realization and its validated Run v2 to this exact
    /// package compilation.
    ///
    /// This method validates the complete Model/revision/Realization/Run chain
    /// before constructing a separate content-addressed lineage edge. The edge
    /// changes none of the linked artifacts and does not independently prove
    /// execution or numerical acceptance.
    ///
    /// # Errors
    /// Returns a typed artifact, package-contract, or exact-linkage failure.
    pub fn bind_execution_v2(
        &self,
        realization: &RealizationEnvelopeV1,
        run: &RunManifestV2,
    ) -> Result<PackageExecutionBindingV1, PackageExecutionBindingError> {
        let identities = self.validate_execution_v2_artifacts(realization, run)?;
        PackageExecutionBindingV1::new(
            &self.compilation,
            identities.semantic_revision,
            identities.realization,
            identities.run,
        )
        .map_err(Into::into)
    }

    /// Independently replay exact package execution lineage against all
    /// concrete artifacts.
    ///
    /// Resolution and compilation inventory are replayed first. The admitted
    /// Model, typed Realization, and Run v2 are then revalidated before their
    /// exact identities are compared with the lineage edge.
    ///
    /// # Errors
    /// Returns a typed artifact, package-contract, or exact-linkage failure.
    pub fn validate_execution_v2_binding(
        &self,
        binding: &PackageExecutionBindingV1,
        realization: &RealizationEnvelopeV1,
        run: &RunManifestV2,
        resolution: &ResolutionRecordV1,
    ) -> Result<(), PackageExecutionBindingError> {
        self.compilation.validate_against(resolution)?;
        let identities = self.validate_execution_v2_artifacts(realization, run)?;
        binding.validate_against(
            &self.compilation,
            identities.semantic_revision,
            identities.realization,
            identities.run,
        )?;
        Ok(())
    }

    /// Transfer the canonical model while deliberately discarding its package
    /// compilation and source-provenance sidecars.
    #[must_use]
    pub fn into_model(self) -> ModelDocument {
        self.model
    }

    fn validate_run_v1_model(
        &self,
        run: &RunManifestV1,
    ) -> Result<CanonicalRunDigest, PackageRunBindingError> {
        let model = CanonicalModelDigest::parse(&self.model.digest()?)?;
        let compilation = self.compilation.model_digest();
        if model != compilation {
            return Err(PackageRunBindingError::CompilationModelMismatch { model, compilation });
        }
        let run_model = CanonicalModelDigest::parse(run.model().as_str())?;
        if run_model != compilation {
            return Err(PackageRunBindingError::RunModelMismatch {
                compilation,
                run: run_model,
            });
        }
        let model_revision = self.model.program().revision().0;
        if run.semantic_revision() != model_revision {
            return Err(PackageRunBindingError::RunRevisionMismatch {
                model: model_revision,
                run: run.semantic_revision(),
            });
        }
        CanonicalRunDigest::parse(run.digest()?.as_str()).map_err(Into::into)
    }

    fn validate_execution_v2_artifacts(
        &self,
        realization: &RealizationEnvelopeV1,
        run: &RunManifestV2,
    ) -> Result<TypedExecutionIdentities, PackageExecutionBindingError> {
        let model = CanonicalModelDigest::parse(&self.model.digest()?)?;
        let compilation = self.compilation.model_digest();
        if model != compilation {
            return Err(PackageExecutionBindingError::CompilationModelMismatch {
                model,
                compilation,
            });
        }

        let realization_model =
            CanonicalModelDigest::parse(&realization.model_artifact().to_string())?;
        if realization_model != compilation {
            return Err(PackageExecutionBindingError::RealizationModelMismatch {
                compilation,
                realization: realization_model,
            });
        }

        let model_revision = self.model.program().revision().0;
        let realization_revision = realization.semantic_revision().get();
        if realization_revision != model_revision {
            return Err(PackageExecutionBindingError::RealizationRevisionMismatch {
                model: model_revision,
                realization: realization_revision,
            });
        }
        if realization.model()? != self.model.program().model() {
            return Err(PackageExecutionBindingError::RealizationOntologyMismatch);
        }

        run.validate_against(realization)?;
        Ok(TypedExecutionIdentities {
            semantic_revision: model_revision,
            realization: CanonicalRealizationDigest::parse(&realization.digest()?.to_string())?,
            run: CanonicalRunDigest::parse(&run.digest()?.to_string())?,
        })
    }
}

fn seal_physical_exposure_catalog(
    model: &ModelDocument,
    compilation: &PackageCompilationRecordV2,
    provenance: &ProvenanceMap,
    projections: &PhysicalExposureProjectionMap,
) -> Result<PhysicalExposureCatalogEnvelopeV1, PackageCompilationError> {
    let entries = projections
        .iter()
        .map(|projection| {
            let origins = provenance.get(projection.exposure()).ok_or_else(|| {
                Diagnostic::error(
                    codes::LANGUAGE_LOWERING_ERROR,
                    "physical exposure projection has no exact source provenance",
                )
            })?;
            let origins = origins
                .origins()
                .iter()
                .map(|origin| {
                    let definition = source_span(origin.definition_span())?;
                    let instance = source_span(origin.instance_span())?;
                    let bindings = origin
                        .binding_spans()
                        .iter()
                        .map(source_span)
                        .collect::<Result<Vec<_>, Diagnostic>>()?;
                    PhysicalExposureSourceOriginV1::new(definition, instance, bindings)
                })
                .collect::<Result<Vec<_>, Diagnostic>>()?;
            let exposure = *projection.exposure().as_bytes();
            let connection = *projection.connection().full_identity().as_bytes();
            let interior = projection
                .interior()
                .iter()
                .map(|port| *port.full_identity().as_bytes())
                .collect();
            match projection.contract() {
                PhysicalExposureContract::ScalarPhysical { connector } => {
                    PhysicalExposureProjectionV1::scalar(
                        projection.selector(),
                        exposure,
                        connection,
                        interior,
                        *connector.full_identity().as_bytes(),
                        origins,
                    )
                }
                PhysicalExposureContract::FieldBoundary {
                    connector,
                    boundary,
                } => PhysicalExposureProjectionV1::field_boundary(
                    projection.selector(),
                    exposure,
                    connection,
                    interior,
                    *connector.full_identity().as_bytes(),
                    *boundary.full_identity().as_bytes(),
                    origins,
                ),
            }
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    let model_artifact = ArtifactDigest::from_hex(model.digest()?)?;
    let package_compilation = ArtifactDigest::from_hex(compilation.digest()?.to_hex())
        .map_err(PackageCompilationError::from)?;
    PhysicalExposureCatalogEnvelopeV1::new(
        model_artifact,
        model.program(),
        package_compilation,
        entries,
    )
    .map_err(Into::into)
}

fn source_span(span: &eqiora_core::Span) -> Result<PhysicalExposureSourceSpanV1, Diagnostic> {
    PhysicalExposureSourceSpanV1::new(span.file.clone(), span.start, span.end)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TypedExecutionIdentities {
    semantic_revision: u64,
    realization: CanonicalRealizationDigest,
    run: CanonicalRunDigest,
}
