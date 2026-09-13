//! Shared application operations for thin Eqiora clients.
//!
//! This L4 crate composes compiler, transaction wire, graph commit, immutable
//! model artifact, and reference execution contracts without introducing a
//! second model semantics. Python and Studio adapters add language- or
//! UI-specific ergonomics around these owned Rust values.

mod cad;
pub mod control;
mod differentiation;
pub mod editor;
mod evaluation_map;
mod execution_session;
#[cfg(any(feature = "vtu", feature = "xdmf"))]
mod external_data;
mod external_spatial;
mod geometry_edit;
mod ml_dataset;
pub mod package;
mod parameter_regeneration;
mod remeshing_trajectory;
mod rendering;
mod run_request;
mod sampling;
mod value_edit;

pub use cad::*;
pub use differentiation::*;
pub use eqiora_artifact::{SemanticFingerprintGeneration, StructuralSemanticFingerprint};
pub use evaluation_map::{
    CompleteEvaluationMap, EvaluationMapExecutionPolicy, EvaluationMapJvp, EvaluationMapOccurrence,
    EvaluationMapPlan, EvaluationMapProducts, EvaluationMapRetention, EvaluationMapTerminalReport,
    EvaluationMapVjp,
};
#[cfg(any(feature = "vtu", feature = "xdmf"))]
pub use external_data::*;
pub use geometry_edit::{CartesianDomainEditPlan, CartesianDomainEditResult};
pub use ml_dataset::*;
pub use parameter_regeneration::{
    ParameterGeometryRegenerationPlan, ParameterGeometryRegenerationResult,
};
pub use remeshing_trajectory::RemeshingTrajectoryReplayInputV1;
#[cfg(feature = "hdf5")]
pub use remeshing_trajectory::{
    VerifiedXdmfHdf5TrajectoryExportV1, XdmfHdf5TrajectoryExportArtifactsV1,
    XdmfHdf5TrajectoryExportLimits, export_xdmf_hdf5_trajectory_v1,
    verify_xdmf_hdf5_trajectory_storage_v1,
};
pub use rendering::{MathReference, MathRendering};
pub use run_request::RunRequest;
pub use sampling::{
    ParameterSampler, SampledParameterPoint, SamplingCoupling, SamplingGenerator, SamplingIdentity,
};
pub use value_edit::{ValueEditPlan, ValueEditResult};

use std::collections::BTreeMap;

use eqiora_artifact::{
    AcceptedModelArtifact, CanonicalModelArtifact, ModelArtifactReference, ModelDecoderLimits,
    ModelTransactionEnvelope, ReplayableCanonicalModelArtifact,
};
use eqiora_compiler::{
    AuthoredFormulationProjection, CompilationNamespaceId, CompiledAuthoredFormulation,
    CompiledModel, ModelSymbols, ResolvedHierarchyInput, ResolvedSourceUnit,
};
use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, RawId};
use eqiora_geometry::CanonicalGeometryV1;
use eqiora_graph::{GraphStore, InMemoryGraphStore, Revision};
use eqiora_lang::Module;
use eqiora_sem::KernelProgram;

/// One immutable, validated canonical model revision plus non-semantic source
/// aliases used by client presentation layers.
#[derive(Debug, Clone)]
pub struct ModelDocument {
    notation: eqiora_compiler::ModelNotation,
    program: KernelProgram,
    artifact: AcceptedModelArtifact,
    aliases: BTreeMap<String, RawId>,
    store: InMemoryGraphStore,
    geometry_authority: Vec<CanonicalGeometryV1>,
    authored_formulations: Vec<CompiledAuthoredFormulation>,
}

impl PartialEq for ModelDocument {
    fn eq(&self, other: &Self) -> bool {
        self.program == other.program
            && self.artifact == other.artifact
            && self.aliases == other.aliases
            && self.geometry_authority == other.geometry_authority
    }
}

impl ModelDocument {
    /// Full-Model declaration labels, shared unchanged by every notation subview.
    /// Bare artifact replay exposes exact-identity labels, not guessed source metadata.
    #[must_use]
    pub const fn notation(&self) -> &eqiora_compiler::ModelNotation {
        &self.notation
    }
    /// Compile exactly one source model with the current semantic vocabulary.
    ///
    /// # Errors
    /// Returns all compiler/semantic diagnostics, or one artifact diagnostic,
    /// when no unique valid model revision can be constructed.
    pub fn compile(filename: &str, source: &str) -> Result<Self, Vec<Diagnostic>> {
        let mut compiled = eqiora_compiler::compile(filename, source)?;
        if compiled.len() != 1 {
            return Err(vec![Diagnostic::error(
                codes::LANGUAGE_LOWERING_ERROR,
                format!(
                    "compile requires exactly one model declaration, found {}",
                    compiled.len()
                ),
            )]);
        }
        Self::accept_compiled(compiled.remove(0))
    }

    /// Compile one selected Model from an already closed semantic module graph.
    ///
    /// Every source import resolves inside `input`; this operation performs no
    /// filesystem discovery, package resolution, network access, or implicit
    /// source inclusion. `entry_model` is either root-local or exactly
    /// `alias.Model` for one directly imported public Model.
    ///
    /// # Errors
    /// Returns graph, parser, visibility, type, lowering, or artifact
    /// diagnostics before exposing a partial Model.
    pub fn compile_modules(
        input: ResolvedHierarchyInput,
        entry_model: &str,
        bindings: &[(&str, eqiora_compiler::StaticBindingValue<'_>)],
    ) -> Result<Self, Vec<Diagnostic>> {
        let compiled = eqiora_compiler::analyze_resolved_hierarchy(input)?
            .validate_definitions()?
            .compile_selected(entry_model, bindings)?;
        Self::accept_bound_compiled(compiled, bindings)
    }

    /// Compile one selected Model from a closed inventory of local project sources.
    ///
    /// Each `path` is validated as a portable normalized path below `src/` and
    /// is the source's logical module identity. Moving a source therefore
    /// renames its module. ASCII-case path collisions are rejected so one
    /// accepted inventory names the same files on case-sensitive and
    /// case-insensitive hosts.
    /// `entry_model` is either root-local or exactly `alias.Model` for one
    /// directly imported public Model.
    ///
    /// This operation performs no filesystem discovery, package resolution,
    /// network access, or implicit source inclusion. Callers must supply the
    /// complete UTF-8 source closure, and every import must resolve inside it.
    ///
    /// # Errors
    /// Returns path, graph, parser, visibility, type, lowering, or artifact
    /// diagnostics before exposing a partial Model.
    pub fn compile_project_sources<I, P, S>(
        root_module: &str,
        sources: I,
        entry_model: &str,
    ) -> Result<Self, Vec<Diagnostic>>
    where
        I: IntoIterator<Item = (P, S)>,
        P: Into<String>,
        S: Into<String>,
    {
        let owner =
            CompilationNamespaceId::new(["eqiora.local_project"]).map_err(single_diagnostic)?;
        let mut paths = BTreeMap::<String, String>::new();
        let mut units = Vec::new();
        let mut diagnostics = Vec::new();

        for (path, source) in sources {
            let path = match eqiora_package::NormalizedRelativePath::parse(path) {
                Ok(path) => path,
                Err(error) => {
                    diagnostics.push(Diagnostic::error(
                        codes::LANGUAGE_LOWERING_ERROR,
                        format!("invalid project source path: {error}"),
                    ));
                    continue;
                }
            };
            let collision_key = path.as_str().to_ascii_lowercase();
            if let Some(previous) = paths.get(&collision_key) {
                diagnostics.push(Diagnostic::error(
                    codes::LANGUAGE_LOWERING_ERROR,
                    format!(
                        "project source path `{}` collides with `{previous}` under portable ASCII-case comparison",
                        path.as_str()
                    ),
                ));
                continue;
            }
            paths.insert(collision_key, path.as_str().to_owned());
            match ResolvedSourceUnit::new(owner.clone(), path.as_str(), source) {
                Ok(unit) => units.push(unit),
                Err(diagnostic) => diagnostics.push(diagnostic),
            }
        }

        if !diagnostics.is_empty() {
            return Err(diagnostics);
        }
        let input =
            ResolvedHierarchyInput::with_root_module(owner, root_module.split('.'), units, vec![])
                .map_err(single_diagnostic)?;
        Self::compile_modules(input, entry_model, &[])
    }

    /// Discover and compile one bounded local `.eqi` source directory.
    ///
    /// The directory adapter performs deterministic capability-rooted
    /// traversal without following symbolic links, then this operation admits
    /// the result through [`Self::compile_project_sources`]. The caller's root
    /// path is the operation's sole ambient filesystem lookup.
    ///
    /// # Errors
    /// Returns one filesystem discovery diagnostic or the complete set of
    /// project-source compilation diagnostics.
    #[cfg(feature = "project-filesystem")]
    pub fn compile_project_directory(
        root: impl Into<std::path::PathBuf>,
        root_module: &str,
        entry_model: &str,
    ) -> Result<Self, Vec<Diagnostic>> {
        let directory = eqiora_package::PackageDirectory::open_ambient(root).map_err(|error| {
            single_diagnostic(Diagnostic::error(
                codes::LANGUAGE_LOWERING_ERROR,
                format!("project source discovery failed: {error}"),
            ))
        })?;
        let sources = directory.discover_project_sources().map_err(|error| {
            single_diagnostic(Diagnostic::error(
                codes::LANGUAGE_LOWERING_ERROR,
                format!("project source discovery failed: {error}"),
            ))
        })?;
        Self::compile_project_sources(
            root_module,
            sources
                .into_iter()
                .map(|(path, source)| (path.as_str().to_owned(), source)),
            entry_model,
        )
    }

    /// Define exactly one model from immutable client-neutral declarations
    /// using the current semantic vocabulary.
    ///
    /// Native construction skips text parsing only. It crosses the same typed
    /// compiler lowerer, versioned transaction wire, atomic graph commit, and
    /// canonical artifact reconstruction as [`Self::compile`].
    ///
    /// # Errors
    /// Returns graph-path compiler/semantic diagnostics, or one artifact
    /// diagnostic, when no valid model revision can be constructed.
    pub fn compile_module(
        module: &Module,
        entry: Option<&str>,
        bindings: &[(&str, eqiora_compiler::StaticBindingValue<'_>)],
    ) -> Result<Self, Vec<Diagnostic>> {
        Self::accept_bound_compiled(
            eqiora_compiler::lower_module(module, entry, bindings)?,
            bindings,
        )
    }

    pub(crate) fn accept_compiled(compiled: CompiledModel) -> Result<Self, Vec<Diagnostic>> {
        Self::accept_compiled_with_geometry(compiled, &[])
    }

    pub(crate) fn accept_compiled_with_geometry(
        compiled: CompiledModel,
        geometries: &[&eqiora_geometry::CanonicalGeometryV1],
    ) -> Result<Self, Vec<Diagnostic>> {
        let aliases = aliases(compiled.symbols());
        let authored_formulations = compiled.authored_formulations().cloned().collect();
        let notation = compiled.notation().clone();
        let model = compiled.model();

        // Every source/UI/language client crosses the same bounded,
        // versioned transaction representation before graph mutation.
        let transaction = ModelTransactionEnvelope::from_transaction(compiled.transaction())
            .and_then(|envelope| envelope.to_transaction())
            .map_err(single_diagnostic)?;

        let mut store = InMemoryGraphStore::new();
        store.commit(transaction)?;
        let program =
            KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, geometries)?;
        let geometry_authority = geometries
            .iter()
            .map(|geometry| (*geometry).clone())
            .collect();
        let mut document =
            Self::from_store(store, program, aliases, geometry_authority, Some(notation))?;
        document.authored_formulations = authored_formulations;
        Ok(document)
    }

    /// Replay one self-contained artifact through the single current Model
    /// contract.
    ///
    /// Historical schemas reject before semantic use; this path never sniffs,
    /// retries, or migrates bytes. Geometry-referencing artifact bytes require
    /// the exact external Geometry closure and therefore reject this
    /// resource-free entry rather than fabricating spatial authority.
    ///
    /// # Errors
    /// Returns decoder, graph, or whole-Model validation diagnostics.
    pub fn replay(data: &[u8]) -> Result<Self, Vec<Diagnostic>> {
        let artifact = AcceptedModelArtifact::from_json(data, ModelDecoderLimits::default())
            .map_err(single_diagnostic)?;
        let (transaction, model) = artifact.to_transaction()?;
        let store = InMemoryGraphStore::restore_snapshot(
            transaction,
            Revision(artifact.source_revision()),
        )?;
        let program = KernelProgram::from_snapshot(&store.snapshot(), model)?;
        Ok(Self {
            notation: eqiora_compiler::ModelNotation::from_kernel(program.model(), program.nodes()),
            program,
            artifact,
            aliases: BTreeMap::new(),
            store,
            geometry_authority: Vec::new(),
            authored_formulations: Vec::new(),
        })
    }

    fn from_store(
        store: InMemoryGraphStore,
        program: KernelProgram,
        aliases: BTreeMap<String, RawId>,
        geometry_authority: Vec<eqiora_geometry::CanonicalGeometryV1>,
        notation: Option<eqiora_compiler::ModelNotation>,
    ) -> Result<Self, Vec<Diagnostic>> {
        let geometries = geometry_authority.iter().collect::<Vec<_>>();
        let program = KernelProgram::from_snapshot_with_geometry(
            &store.snapshot(),
            program.model(),
            &geometries,
        )?;
        let artifact = AcceptedModelArtifact::from_program(&program).map_err(single_diagnostic)?;
        // Reconstruct once more from the public artifact so client behavior
        // cannot accidentally depend on an in-memory compiler-only state.
        let bytes = artifact.canonical_json().map_err(single_diagnostic)?;
        let artifact = AcceptedModelArtifact::from_json(&bytes, ModelDecoderLimits::default())
            .map_err(single_diagnostic)?;
        let document = Self {
            notation: notation.unwrap_or_else(|| {
                eqiora_compiler::ModelNotation::from_kernel(program.model(), program.nodes())
            }),
            program,
            artifact,
            aliases,
            store,
            geometry_authority,
            authored_formulations: Vec::new(),
        };
        document
            .replay_with_retained_geometry()
            .map_err(single_diagnostic)?;
        Ok(document)
    }

    /// Typed authored mathematics available only after fresh source compilation.
    ///
    /// Canonical Model artifacts deliberately exclude this compiler sidecar;
    /// replay therefore returns an empty slice instead of fabricating a form.
    #[must_use]
    pub fn authored_formulations(
        &self,
    ) -> impl ExactSizeIterator<Item = &CompiledAuthoredFormulation> {
        self.authored_formulations.iter()
    }

    /// Closed typed scalar-primal projection consumed by common resolution.
    ///
    /// The compiler owns the versioned representation and its canonical codec;
    /// callers may inspect the closed typed expression vocabulary but cannot
    /// construct an unchecked projection.
    ///
    /// # Errors
    /// Returns a diagnostic if compilation retained more than one authored form.
    pub fn authored_scalar_primal_projection(
        &self,
    ) -> Result<Option<&AuthoredFormulationProjection>, Diagnostic> {
        match self.authored_formulations.as_slice() {
            [] => Ok(None),
            [form] => Ok(Some(form.projection())),
            _ => Err(Diagnostic::error(
                codes::LANGUAGE_TYPE_ERROR,
                "common scalar resolve accepts exactly one authored primal Formulation",
            )),
        }
    }

    /// Validated immutable execution input.
    #[must_use]
    pub const fn program(&self) -> &KernelProgram {
        &self.program
    }

    /// Typed identity of the current Model artifact.
    ///
    /// # Errors
    /// Returns an artifact diagnostic only if validated envelope state cannot
    /// be decoded.
    pub fn artifact_reference(&self) -> Result<ModelArtifactReference, Diagnostic> {
        self.artifact.artifact_reference()
    }

    /// Non-semantic source aliases in deterministic lexical order.
    #[must_use]
    pub const fn aliases(&self) -> &BTreeMap<String, RawId> {
        &self.aliases
    }

    /// Canonical compact JSON bytes for this immutable model.
    ///
    /// # Errors
    /// Returns an artifact diagnostic if invariant replay fails.
    pub fn canonical_json(&self) -> Result<Vec<u8>, Diagnostic> {
        let bytes = self.artifact.canonical_json()?;
        self.replay_with_retained_geometry()?;
        Ok(bytes)
    }

    /// Domain-separated exact Model artifact digest, including occurrence identity.
    ///
    /// # Errors
    /// Returns an artifact diagnostic if invariant replay fails.
    pub fn digest(&self) -> Result<String, Diagnostic> {
        self.replay_with_retained_geometry()?;
        self.artifact.digest().map(|digest| digest.to_string())
    }

    /// Alpha-normalized structural comparison evidence for this Model.
    ///
    /// The fingerprint intentionally excludes occurrence IDs, source names,
    /// graph revision, and exact artifact codec.  It never substitutes for
    /// [`Self::artifact_reference`] in execution, replay, provenance, or
    /// mutation preconditions.
    ///
    /// # Errors
    /// Returns an artifact diagnostic when exact bounded graph
    /// canonicalization cannot represent this fingerprint generation.
    pub fn structural_fingerprint(&self) -> Result<StructuralSemanticFingerprint, Diagnostic> {
        StructuralSemanticFingerprint::from_program(&self.program)
    }

    /// Compare alpha-normalized Semantic Kernel structure without weakening
    /// either exact Model artifact identity.
    ///
    /// Equal digests are followed by exact private canonical-byte comparison,
    /// so a cryptographic collision fails closed.
    ///
    /// # Errors
    /// Returns an artifact diagnostic for unsupported meaning, exhausted
    /// canonicalization limits, or unequal projections with a colliding hash.
    pub fn structurally_equivalent(&self, other: &Self) -> Result<bool, Diagnostic> {
        eqiora_artifact::structurally_equivalent(&self.program, &other.program)
    }

    fn replay_with_retained_geometry(&self) -> Result<KernelProgram, Diagnostic> {
        if self.geometry_authority.is_empty() {
            return self
                .artifact
                .replay_model()
                .map(|replayed| replayed.program().clone());
        }
        let reference = self.artifact.artifact_reference()?;
        let (transaction, model) = self.artifact.to_transaction().map_err(first_diagnostic)?;
        let store = InMemoryGraphStore::restore_snapshot(
            transaction,
            Revision(self.artifact.source_revision()),
        )
        .map_err(first_diagnostic)?;
        let geometries = self.geometry_authority.iter().collect::<Vec<_>>();
        let program =
            KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &geometries)
                .map_err(first_diagnostic)?;
        if program.model() != reference.model()
            || program.revision().0 != reference.semantic_revision().get()
        {
            return Err(Diagnostic::error(
                codes::INVALID_ARTIFACT,
                "replayed Model identity or semantic revision differs from its exact artifact reference",
            ));
        }
        Ok(program)
    }
}

fn aliases(symbols: &ModelSymbols) -> BTreeMap<String, RawId> {
    symbols
        .iter()
        .map(|(name, id)| (name.to_owned(), id))
        .collect()
}

fn single_diagnostic(diagnostic: Diagnostic) -> Vec<Diagnostic> {
    vec![diagnostic]
}

fn first_diagnostic(diagnostics: Vec<Diagnostic>) -> Diagnostic {
    diagnostics.into_iter().next().unwrap_or_else(|| {
        Diagnostic::error(
            codes::INVALID_ARTIFACT,
            "Model replay failed without a diagnostic",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::ModelDocument;
    use eqiora_artifact::ReplayableCanonicalModelArtifact;
    use eqiora_compiler::{CompilationNamespaceId, ResolvedHierarchyInput, ResolvedSourceUnit};
    use eqiora_core::{DimExponents, EntityKind, RawId, ValueLiteral};
    use eqiora_lang::{DraftExpression, DraftField, DraftParameter, DraftRelation, Module};

    const SOURCE: &str = r#"
model decay() {
  state x: 1;
  initial { x = 1; }
  parameter rate: 1 / s = 1;
  relation flow {
    derivative(x) + rate * x = 0;
  }
}
"#;

    #[test]
    fn one_model_path_closes_compile_wire_and_artifact() {
        let document = ModelDocument::compile("decay.eqi", SOURCE).unwrap();
        let bytes = document.canonical_json().unwrap();
        let digest = document.digest().unwrap();
        let reconstructed = ModelDocument::replay(&bytes).unwrap();
        assert_eq!(reconstructed.canonical_json().unwrap(), bytes);
        assert_eq!(reconstructed.digest().unwrap(), digest);

        let mut zero_revision: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        zero_revision["source_revision"] = serde_json::Value::from(0);
        let diagnostics =
            ModelDocument::replay(&serde_json::to_vec(&zero_revision).unwrap()).unwrap_err();
        assert_eq!(diagnostics[0].code().0, "EQ0901");
    }

    #[test]
    fn closed_local_module_graph_compiles_through_the_model_boundary() {
        let owner =
            CompilationNamespaceId::new(["org.example.project", "1.0.0", "local-semantic-closure"])
                .unwrap();
        let input = |reverse: bool| {
            let mut units = vec![
                ResolvedSourceUnit::new(
                    owner.clone(),
                    "src/models/main.eqi",
                    "import org.example.project.library.parts as lib; model Main() { instance part: lib.Part(p = 1); }",
                )
                .unwrap(),
                ResolvedSourceUnit::new(
                    owner.clone(),
                    "src/library/parts.eqi",
                    "public component Part(\n  parameter p: 1\n) { relation law { p - 1 = 0; } }",
                )
                .unwrap(),
            ];
            if reverse {
                units.reverse();
            }
            ResolvedHierarchyInput::with_root_module(
                owner.clone(),
                "models.main".split('.'),
                units,
                vec![],
            )
            .unwrap()
        };

        let document = ModelDocument::compile_modules(input(false), "Main", &[]).unwrap();
        let reversed = ModelDocument::compile_modules(input(true), "Main", &[]).unwrap();
        let bytes = document.canonical_json().unwrap();
        assert_eq!(reversed.canonical_json().unwrap(), bytes);
        assert_eq!(
            ModelDocument::replay(&bytes)
                .unwrap()
                .canonical_json()
                .unwrap(),
            bytes
        );
    }

    #[test]
    fn project_sources_derive_modules_from_paths_deterministically() {
        let main = r#"
import eqiora.local_project.library.parts as lib;
model Main() { instance load: lib.Resistor(resistance = 2); }
"#;
        let library = r#"
public component Resistor(
  parameter resistance: 1
) {
  relation law { resistance - 2 = 0; }
}
"#;
        let compile = |main_path, library_path, reverse| {
            let mut sources = vec![(main_path, main), (library_path, library)];
            if reverse {
                sources.reverse();
            }
            ModelDocument::compile_project_sources("models.main", sources, "Main").unwrap()
        };

        let original = compile("src/models/main.eqi", "src/library/parts.eqi", false);
        let reordered = compile("src/models/main.eqi", "src/library/parts.eqi", true);
        assert_eq!(
            original.canonical_json().unwrap(),
            reordered.canonical_json().unwrap()
        );
    }

    #[test]
    fn project_sources_compile_a_directly_imported_public_model() {
        let imported = ModelDocument::compile_project_sources(
            "main",
            [
                (
                    "src/main.eqi",
                    "import eqiora.local_project.library.entries as lib; model Local() {}",
                ),
                (
                    "src/library/entries.eqi",
                    "public model Shared() { parameter gain: 1 = 2; relation law { gain - 2 = 0; } }",
                ),
            ],
            "lib.Shared",
        )
        .expect("directly imported public Model compiles");
        assert!(imported.aliases().contains_key("law"));

        let diagnostics = ModelDocument::compile_project_sources(
            "main",
            [
                (
                    "src/main.eqi",
                    "import eqiora.local_project.library.entries as lib; model Local() {}",
                ),
                ("src/library/entries.eqi", "model Hidden() {}"),
            ],
            "lib.Hidden",
        )
        .expect_err("private imported Model fails closed");
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message().contains("private Model")),
            "{diagnostics:?}"
        );
    }

    #[test]
    fn project_sources_reject_nonportable_or_colliding_paths_together() {
        let diagnostics = ModelDocument::compile_project_sources(
            "models.main",
            [
                ("../main.eqi", "model Main() {}"),
                ("src/Part.eqi", "public component One() {}"),
                ("src/part.eqi", "public component Two() {}"),
            ],
            "Main",
        )
        .unwrap_err();

        assert_eq!(diagnostics.len(), 2);
        assert!(
            diagnostics[0]
                .message()
                .contains("non-portable path segment")
        );
        assert!(diagnostics[1].message().contains("ASCII-case comparison"));
    }

    #[test]
    fn native_definition_closes_an_equivalent_independent_artifact() {
        let state = DraftField::new(
            "x",
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
            )
            .expect("valid scalar type"),
            eqiora_lang::FieldRoleSyntax::State,
        );
        let rate = DraftParameter::new(
            "rate",
            ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(
                    eqiora_core::ScalarDomain::Real,
                    DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).expect("bounded dimension"),
                )
                .expect("valid scalar type"),
                1.0,
            )
            .unwrap(),
        );
        let flow = DraftRelation::continuous(
            "flow",
            [(
                DraftExpression::derivative(&state) + rate.expression() * state.expression(),
                DraftExpression::constant(eqiora_lang::DecimalLiteral::from_f64(0.0).unwrap()),
            )],
        );
        let initial = eqiora_lang::DraftDeclaration::Initial(vec![(
            state.expression(),
            DraftExpression::constant(eqiora_lang::DecimalLiteral::from_f64(1.0).unwrap()),
        )]);
        let draft =
            Module::new("decay", [state.into(), rate.into(), flow.into(), initial]).unwrap();

        let native = ModelDocument::compile_module(&draft, None, &[]).unwrap();
        let bytes = native.canonical_json().unwrap();
        let reconstructed = ModelDocument::replay(&bytes).unwrap();
        assert_eq!(reconstructed.canonical_json().unwrap(), bytes);
        for (name, kind) in [
            ("x", EntityKind::Field),
            ("rate", EntityKind::Parameter),
            ("flow", EntityKind::Relation),
        ] {
            assert_eq!(native.aliases()[name].kind(), kind);
        }

        let source = ModelDocument::compile("decay.eqi", SOURCE).unwrap();
        assert_ne!(native.digest().unwrap(), source.digest().unwrap());
        assert!(native.structurally_equivalent(&source).unwrap());
        assert_eq!(
            native.structural_fingerprint().unwrap(),
            source.structural_fingerprint().unwrap()
        );
    }

    #[test]
    fn current_authoring_retains_tensor_vocabulary() {
        let source = r#"
model elastic_relation() {
  domain body = box(0, 1, 0, 1);
  variable displacement: vector<m, 2> on body;
  parameter mu: kg / (m * s ^ 2) = 2;
  parameter lambda: kg / (m * s ^ 2) = 3;
  relation balance on body {
    -div(
      2 * mu * symmetric_part(grad(displacement))
      + lambda * isotropic_lift(div(displacement))
    ) = 0;
  }
}
"#;
        let document = ModelDocument::compile("elastic.eqi", source).unwrap();
        let bytes = document.canonical_json().unwrap();
        assert!(String::from_utf8_lossy(&bytes).contains("symmetric-part"));
        assert!(String::from_utf8_lossy(&bytes).contains("isotropic-lift"));
        let replay = ModelDocument::replay(&bytes).unwrap();
        assert_eq!(replay.canonical_json().unwrap(), bytes);
        assert_eq!(replay.digest().unwrap(), document.digest().unwrap());
    }

    #[test]
    fn current_authoring_closes_generic_pure_operators() {
        let source = r#"
public operator dyadic(input left: spatial[1], input right: spatial[1]): spatial[2]
  = component(left, 0) * component(right, 1);
model pure_relation() {
  domain body = box(0, 1, 0, 1);
  variable left: vector<1, 2> on body;
  variable right: vector<1, 2> on body;
  relation balance on body {
    div(div(dyadic(left = left, right = right))) = 0;
  }
}
"#;
        let current = ModelDocument::compile("pure-relation.eqi", source).unwrap();

        let bytes = current.canonical_json().unwrap();
        let json = String::from_utf8_lossy(&bytes);
        assert!(json.contains("pure-operator-application"));
        assert!(json.contains("eqiora.model-envelope/v26"));
        let replay = ModelDocument::replay(&bytes).unwrap();
        assert_eq!(replay.canonical_json().unwrap(), bytes);
        assert_eq!(replay.digest().unwrap(), current.digest().unwrap());
    }

    #[test]
    fn invalid_source_remains_a_structured_diagnostic() {
        let diagnostics =
            ModelDocument::compile("broken.eqi", "model broken() { field ; }").unwrap_err();
        assert!(!diagnostics.is_empty());
        assert!(diagnostics[0].code().0.starts_with("EQ"));
    }

    fn scalar_edit_value(document: &ModelDocument, target: RawId, value: f64) -> ValueLiteral {
        ValueLiteral::from_real(
            document
                .program()
                .typed_value(target)
                .unwrap()
                .value_type()
                .clone(),
            value,
        )
        .unwrap()
    }

    #[test]
    fn value_edit_retains_current_transaction_and_artifact_lineage() {
        let document = ModelDocument::compile("decay.eqi", SOURCE).unwrap();
        let base_digest = document.digest().unwrap();
        let rate = document.aliases()["rate"];
        let relation = document.aliases()["flow"];

        let plan = document
            .preview_value_edit(rate, scalar_edit_value(&document, rate, 2.0))
            .unwrap();
        assert_eq!(plan.base_digest(), base_digest);
        assert_eq!(plan.base_revision().0, 1);
        assert_eq!(plan.target(), rate);
        assert_eq!(plan.before().real_scalar_value().unwrap().value(), 1.0);
        assert_eq!(plan.after().real_scalar_value().unwrap().value(), 2.0);
        assert_eq!(
            plan.before().value_type().dimension(),
            plan.after().value_type().dimension()
        );
        assert!(plan.key().starts_with("eqiora.value-edit-plan/v1:"));
        assert!(
            String::from_utf8(plan.transaction_json().unwrap())
                .unwrap()
                .contains("eqiora.model-transaction-envelope/v26")
        );

        let result = document.commit_value_edit(plan.clone()).unwrap();
        assert_eq!(result.plan(), &plan);
        assert_eq!(result.result_revision().0, 2);
        assert_ne!(result.result_digest(), base_digest);
        assert_eq!(document.program().revision().0, 1);
        assert_eq!(document.program().value(rate).unwrap().value(), 1.0);
        assert_eq!(
            result.document().program().value(rate).unwrap().value(),
            2.0
        );

        let public_artifact_replay = result.document().artifact.replay_model().unwrap();
        assert_eq!(public_artifact_replay.program().revision().0, 2);
        assert_eq!(
            public_artifact_replay
                .artifact_reference()
                .semantic_revision()
                .get(),
            2
        );

        let child_bytes = result.document().canonical_json().unwrap();
        let replayed_child = ModelDocument::replay(&child_bytes).unwrap();
        let grandchild_plan = replayed_child
            .preview_value_edit(rate, scalar_edit_value(&replayed_child, rate, 3.0))
            .unwrap();
        let grandchild = replayed_child.commit_value_edit(grandchild_plan).unwrap();
        assert_eq!(grandchild.result_revision().0, 3);
        assert_eq!(
            grandchild.document().program().value(rate).unwrap().value(),
            3.0
        );

        assert_eq!(
            result.document().commit_value_edit(plan).unwrap_err()[0]
                .code()
                .0,
            "EQ0106"
        );
        assert_eq!(
            document
                .preview_value_edit(rate, scalar_edit_value(&document, rate, 1.0))
                .unwrap_err()
                .code()
                .0,
            "EQ0105"
        );
        assert!(
            ValueLiteral::from_real(
                document
                    .program()
                    .typed_value(rate)
                    .unwrap()
                    .value_type()
                    .clone(),
                f64::NAN
            )
            .is_err()
        );
        assert_eq!(
            document
                .preview_value_edit(relation, scalar_edit_value(&document, rate, 2.0))
                .unwrap_err()
                .code()
                .0,
            "EQ0105"
        );
    }

    #[test]
    fn value_edit_identity_includes_the_exact_base_artifact() {
        let source = SOURCE.replace("model decay() {", "model decay() { parameter probe: 1 = 0;");
        let base = ModelDocument::compile("decay.eqi", &source).unwrap();
        let probe = base.aliases()["probe"];
        let rate = base.aliases()["rate"];

        let left = base
            .commit_value_edit(
                base.preview_value_edit(rate, scalar_edit_value(&base, rate, 2.0))
                    .unwrap(),
            )
            .unwrap()
            .into_document();
        let right = base
            .commit_value_edit(
                base.preview_value_edit(rate, scalar_edit_value(&base, rate, 3.0))
                    .unwrap(),
            )
            .unwrap()
            .into_document();
        let left_plan = left
            .preview_value_edit(probe, scalar_edit_value(&left, probe, 4.0))
            .unwrap();
        let right_plan = right
            .preview_value_edit(probe, scalar_edit_value(&right, probe, 4.0))
            .unwrap();

        assert_eq!(left_plan.base_revision(), right_plan.base_revision());
        assert_eq!(
            left_plan.transaction_digest(),
            right_plan.transaction_digest(),
            "the same graph-local transaction is reusable only against its exact base artifact"
        );
        assert_ne!(left_plan.base_digest(), right_plan.base_digest());
        assert_ne!(left_plan.key(), right_plan.key());
        assert_ne!(left_plan, right_plan);
        assert_eq!(
            left.commit_value_edit(right_plan).unwrap_err()[0].code().0,
            "EQ0106"
        );
    }
}
