//! Immutable Python model identities and optimistic value edits.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use eqiora::api::package::PackagedModelDocument;
use eqiora::api::{ModelDocument, StructuralSemanticFingerprint, ValueEditPlan};
use eqiora::artifact::{CanonicalModelArtifact, ModelDecoderLimits, ModelEnvelope};
use eqiora::diagnostic::codes;
use eqiora::graph::Op;
use eqiora::package::PackageCompilationRecordV2;
use eqiora::{Diagnostic, EntityKind, RawId};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyModule, PyTuple};

use crate::error::{diagnostic_error, internal_diagnostic_error, panic_boundary, validation_error};
use crate::geometry::PyGeometry;
use crate::model_io::{self, DecodedModel};

mod authored_formulation;
mod enumeration;
mod notation;
pub(crate) use notation::PyQuantityLabel;
pub(crate) use notation::parse_profile;
mod rendering;
pub(crate) use rendering::{PyMathReference, PyMathRendering};
mod activation_ref;
pub(crate) mod constraint;
mod observable_ref;
pub(crate) use activation_ref::PyActivationRef;
mod parameter_ref;
mod property;
pub(crate) use observable_ref::PyObservableRef;
pub(crate) use parameter_ref::PyModelParameterRef;
use property::{PyPropertyBinding, property_bindings};

/// Exact identity of one immutable canonical Model artifact.
#[pyclass(
    name = "Revision",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyRevision {
    model_id: String,
    digest: String,
    number: u64,
}

/// Alpha-normalized comparison/cache evidence, never exact Model identity.
#[pyclass(
    name = "StructuralSemanticFingerprint",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyStructuralSemanticFingerprint {
    value: StructuralSemanticFingerprint,
}

#[pymethods]
impl PyStructuralSemanticFingerprint {
    /// Exact generation of the structural projection.
    #[getter]
    fn generation(&self) -> &'static str {
        self.value.generation().as_str()
    }

    /// Domain-separated digest of the alpha-normalized projection.
    #[getter]
    fn digest(&self) -> &str {
        self.value.digest()
    }

    fn __repr__(&self) -> String {
        format!(
            "StructuralSemanticFingerprint(generation={:?}, digest={:?})",
            self.generation(),
            self.digest()
        )
    }
}

#[pymethods]
impl PyRevision {
    /// Canonical typed Model ontology identity.
    #[getter]
    fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Domain-separated canonical Model content digest.
    #[getter]
    fn digest(&self) -> &str {
        &self.digest
    }

    /// Semantic graph revision serialized by the Model artifact.
    #[getter]
    const fn number(&self) -> u64 {
        self.number
    }

    fn __repr__(&self) -> String {
        format!(
            "Revision(model_id={:?}, number={}, digest={:?})",
            self.model_id, self.number, self.digest
        )
    }
}

/// One immutable, exact-base value edit prepared by the shared Rust facade.
#[pyclass(
    name = "ValueEdit",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyValueEdit {
    plan: ValueEditPlan,
}

/// Exact canonical Field selected from one immutable Model.
#[pyclass(
    name = "FieldRef",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyModelFieldRef {
    model_digest: String,
    id: String,
}

/// Exact canonical Domain selected from one immutable Model.
#[pyclass(
    name = "DomainRef",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyModelDomainRef {
    model_digest: String,
    id: String,
}

impl PyModelDomainRef {
    pub(crate) fn from_exact(model_digest: String, id: String) -> Self {
        Self { model_digest, id }
    }

    pub(crate) fn exact_model_digest(&self) -> &str {
        &self.model_digest
    }
    pub(crate) fn exact_id(&self) -> &str {
        &self.id
    }
}

#[pymethods]
impl PyModelDomainRef {
    #[getter]
    fn model_digest(&self) -> &str {
        &self.model_digest
    }

    #[getter]
    fn id(&self) -> &str {
        &self.id
    }

    fn __repr__(&self) -> String {
        format!(
            "DomainRef(id={:?}, model_digest={:?})",
            self.id, self.model_digest
        )
    }
}

impl PyModelFieldRef {
    pub(crate) fn from_exact(model_digest: String, id: String) -> Self {
        Self { model_digest, id }
    }

    pub(crate) fn exact_model_digest(&self) -> &str {
        &self.model_digest
    }

    pub(crate) fn exact_id(&self) -> &str {
        &self.id
    }
}

#[pymethods]
impl PyModelFieldRef {
    /// Exact canonical Model artifact digest.
    #[getter]
    fn model_digest(&self) -> &str {
        &self.model_digest
    }

    /// Stable canonical Field ULID.
    #[getter]
    fn id(&self) -> &str {
        &self.id
    }

    fn __repr__(&self) -> String {
        format!(
            "FieldRef(id={:?}, model_digest={:?})",
            self.id, self.model_digest
        )
    }
}

#[pymethods]
impl PyValueEdit {
    /// Exact-plan key over the base artifact and ordered transaction wire.
    #[getter]
    fn key(&self) -> String {
        self.plan.key()
    }

    /// Canonical identity of the immutable base Model content.
    #[getter]
    fn base_digest(&self) -> &str {
        self.plan.base_digest()
    }

    /// Exact graph revision required by the optimistic precondition.
    #[getter]
    fn base_revision(&self) -> u64 {
        self.plan.base_revision().0
    }

    /// Stable canonical target identity.
    #[getter]
    fn target_id(&self) -> String {
        self.plan.target().ulid().to_string()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, Self>>()
            .is_ok_and(|other| self.plan.key() == other.plan.key())
    }

    fn __ne__(&self, other: &Bound<'_, PyAny>) -> bool {
        !self.__eq__(other)
    }

    fn __hash__(&self) -> isize {
        hash_value(&self.plan.key()) as isize
    }

    fn __repr__(&self) -> String {
        format!(
            "ValueEdit(base_revision={}, target_id={:?}, key={:?})",
            self.base_revision(),
            self.target_id(),
            self.key()
        )
    }
}

/// One immutable canonical Model artifact, semantically admitted when closed.
#[pyclass(name = "Model", module = "eqiora._eqiora", frozen, skip_from_py_object)]
#[derive(Debug)]
pub(crate) struct PyModel {
    document: Option<ModelDocument>,
    artifact: ModelEnvelope,
    revision: PyRevision,
    package_compilation: Option<PackageCompilationRecordV2>,
    property_bindings: Box<[PyPropertyBinding]>,
    /// Retains the exact caller-owned Python Geometry handle for a fresh
    /// geometry-closed compilation; artifact identity remains Rust-owned.
    _geometry: Option<Py<PyGeometry>>,
}

impl PyModel {
    pub(crate) fn package_compilation_digest_value(&self) -> Result<Option<String>, Diagnostic> {
        self.package_compilation
            .as_ref()
            .map(|compilation| compilation.digest().map(|digest| digest.to_hex()))
            .transpose()
            .map_err(|_| {
                Diagnostic::error(
                    codes::INTERNAL_FAILURE,
                    "accepted package-compilation lineage could not be projected",
                )
            })
    }

    pub(crate) fn from_document(py: Python<'_>, document: ModelDocument) -> PyResult<Self> {
        let reference = document
            .artifact_reference()
            .map_err(|diagnostic| internal_diagnostic_error(py, &[diagnostic]))?;
        let artifact = document
            .canonical_json()
            .and_then(|bytes| ModelEnvelope::from_json(&bytes, ModelDecoderLimits::default()))
            .map_err(|diagnostic| internal_diagnostic_error(py, &[diagnostic]))?;
        let property_bindings = property_bindings(document.program().nodes());
        Ok(Self {
            revision: PyRevision {
                model_id: reference.model().ulid().to_string(),
                digest: reference.artifact().to_string(),
                number: reference.semantic_revision().get(),
            },
            document: Some(document),
            artifact,
            package_compilation: None,
            property_bindings,
            _geometry: None,
        })
    }

    pub(crate) fn from_document_with_geometry(
        py: Python<'_>,
        document: ModelDocument,
        geometry: Py<PyGeometry>,
    ) -> PyResult<Self> {
        let mut model = Self::from_document(py, document)?;
        model._geometry = Some(geometry);
        Ok(model)
    }

    pub(crate) fn from_artifact(py: Python<'_>, artifact: ModelEnvelope) -> PyResult<Self> {
        let reference = artifact
            .artifact_reference()
            .map_err(|diagnostic| internal_diagnostic_error(py, &[diagnostic]))?;
        // Typed artifact reconstruction authenticates expression metadata while
        // preserving deferred admission of an external Geometry closure.
        let (transaction, _) = artifact
            .to_transaction()
            .map_err(|diagnostics| internal_diagnostic_error(py, &diagnostics))?;
        let property_bindings = property_bindings(transaction.ops().iter().filter_map(
            |operation| match operation {
                Op::DefineKernelNode { node } => Some(node),
                _ => None,
            },
        ));
        Ok(Self {
            revision: PyRevision {
                model_id: reference.model().ulid().to_string(),
                digest: reference.artifact().to_string(),
                number: reference.semantic_revision().get(),
            },
            document: None,
            artifact,
            package_compilation: None,
            property_bindings,
            _geometry: None,
        })
    }

    pub(crate) fn from_packaged(py: Python<'_>, packaged: PackagedModelDocument) -> PyResult<Self> {
        let compilation = packaged.compilation().clone();
        let mut model = Self::from_document(py, packaged.model().clone())?;
        model.package_compilation = Some(compilation);
        Ok(model)
    }

    pub(crate) fn from_packaged_with_geometry(
        py: Python<'_>,
        packaged: PackagedModelDocument,
        geometry: Py<PyGeometry>,
    ) -> PyResult<Self> {
        let mut model = Self::from_packaged(py, packaged)?;
        model._geometry = Some(geometry);
        Ok(model)
    }

    pub(crate) fn document(&self) -> Result<&ModelDocument, Diagnostic> {
        self.document.as_ref().ok_or_else(deferred_admission)
    }

    pub(crate) const fn artifact(&self) -> &ModelEnvelope {
        &self.artifact
    }

    pub(crate) fn authored_scalar_primal_projection(
        &self,
    ) -> Result<Option<&eqiora::compiler::AuthoredFormulationProjection>, Diagnostic> {
        self.document
            .as_ref()
            .map(ModelDocument::authored_scalar_primal_projection)
            .transpose()
            .map(Option::flatten)
    }

    #[allow(dead_code)]
    pub(crate) fn field_ref_from_id(
        &self,
        py: Python<'_>,
        field: eqiora::Id<eqiora::kinds::Field>,
    ) -> PyResult<PyModelFieldRef> {
        let reference = self
            .artifact
            .artifact_reference()
            .map_err(|diagnostic| internal_diagnostic_error(py, &[diagnostic]))?;
        Ok(PyModelFieldRef::from_exact(
            reference.artifact().to_string(),
            field.ulid().to_string(),
        ))
    }

    fn artifact_ids(&self, kind: EntityKind) -> Result<Vec<String>, Vec<Diagnostic>> {
        let (transaction, _) = self.artifact.to_transaction()?;
        Ok(transaction
            .ops()
            .iter()
            .filter_map(|operation| match operation {
                Op::DefineKernelNode { node } if node.id().kind() == kind => {
                    Some(node.id().ulid().to_string())
                }
                _ => None,
            })
            .collect())
    }

    fn resolve_edit_target(&self, target: &str) -> Result<RawId, Diagnostic> {
        let document = self.document()?;
        if let Some(&id) = document.aliases().get(target)
            && id.kind() == EntityKind::Parameter
        {
            return Ok(id);
        }
        document
            .program()
            .nodes()
            .map(|node| node.id())
            .find(|id| {
                id.kind() == EntityKind::Parameter
                    && id.ulid().to_string() == target
            })
            .ok_or_else(|| {
                Diagnostic::error(
                    codes::NODE_NOT_FOUND,
                    format!(
                        "value-edit target {target:?} is neither a Parameter alias nor an exact canonical ID in this Model revision"
                    ),
                )
            })
    }
}

fn deferred_admission() -> Diagnostic {
    Diagnostic::error(
        codes::NOT_IMPLEMENTED,
        "this current Model requires application-specific artifact admission before semantic use",
    )
}

#[pymethods]
impl PyModel {
    /// Exact canonical, versioned compiled Model artifact bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        panic_boundary(py, || {
            self.artifact
                .canonical_json()
                .map(|bytes| PyBytes::new(py, &bytes))
                .map_err(|diagnostic| internal_diagnostic_error(py, &[diagnostic]))
        })
    }

    /// Decode exact canonical compiled Model artifact bytes.
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        panic_boundary(py, || {
            let data = data.to_vec();
            py.detach(move || model_io::decode_model(&data))
                .map_err(|diagnostics| crate::error::compatibility_error(py, &diagnostics))
                .and_then(|decoded| match decoded {
                    DecodedModel::Document(document) => Self::from_document(py, *document),
                    DecodedModel::Deferred(artifact) => Self::from_artifact(py, artifact),
                })
        })
    }

    /// Atomically write this compiled Model to a `.eqmodel` file.
    fn write(&self, py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        panic_boundary(py, || {
            let path = model_io::unicode_artifact_path(py, path)?;
            let bytes = self
                .artifact
                .canonical_json()
                .map_err(|diagnostic| internal_diagnostic_error(py, &[diagnostic]))?;
            py.detach(move || model_io::write_model_bytes(&path, &bytes))
                .map_err(|diagnostic| crate::error::compatibility_error(py, &[diagnostic]))
        })
    }

    /// Read and decode one exact canonical compiled Model `.eqmodel` file.
    #[staticmethod]
    fn read(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<Self> {
        panic_boundary(py, || {
            let path = model_io::unicode_artifact_path(py, path)?;
            let bytes = py
                .detach(move || model_io::read_model_bytes(&path))
                .map_err(|diagnostic| crate::error::compatibility_error(py, &[diagnostic]))?;
            let decoded = py
                .detach(move || model_io::decode_model(&bytes))
                .map_err(|diagnostics| crate::error::compatibility_error(py, &diagnostics))?;
            match decoded {
                DecodedModel::Document(document) => Self::from_document(py, *document),
                DecodedModel::Deferred(artifact) => Self::from_artifact(py, artifact),
            }
        })
    }

    /// Exact immutable artifact-bound revision identity.
    #[getter]
    fn revision(&self) -> PyRevision {
        self.revision.clone()
    }

    /// Typed Semantic Model identity retained by the canonical artifact.
    #[getter]
    fn model_id(&self) -> &str {
        &self.revision.model_id
    }

    /// Domain-separated semantic content digest.
    #[getter]
    fn digest(&self) -> &str {
        &self.revision.digest
    }

    /// Exact accepted package-compilation lineage, absent after derivation or decoding.
    #[getter]
    fn package_compilation_digest(&self, py: Python<'_>) -> PyResult<Option<String>> {
        self.package_compilation_digest_value().map_err(|_| {
            internal_diagnostic_error(
                py,
                &[Diagnostic::error(
                    codes::INTERNAL_FAILURE,
                    "accepted package-compilation lineage could not be projected",
                )],
            )
        })
    }

    /// Exact release bindings retained on Model expression occurrences.
    #[getter]
    fn property_bindings(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let bindings = self
            .property_bindings
            .iter()
            .cloned()
            .map(|binding| Py::new(py, binding))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, bindings)?.unbind())
    }

    /// Authored mathematics retained by fresh source compilation only.
    #[getter]
    fn authored_formulations(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        authored_formulation::project(py, self.document.as_ref())
    }

    /// Full-scope labels or an exact identity-selected subview, never re-resolved.
    #[pyo3(signature = (profile="latex", *, identities=None))]
    fn notation_labels(
        &self,
        py: Python<'_>,
        profile: &str,
        identities: Option<Vec<String>>,
    ) -> PyResult<Py<PyTuple>> {
        notation::project(self, py, profile, identities)
    }

    /// Render exact retained equations without substituting identifier text.
    #[pyo3(signature = (relation, profile="latex"))]
    fn render_equations(
        &self,
        py: Python<'_>,
        relation: &str,
        profile: &str,
    ) -> PyResult<Py<PyTuple>> {
        rendering::equations(self, py, relation, profile)
    }

    /// Render retained authored forms through the same accessible projections.
    #[pyo3(signature = (profile="latex"))]
    fn render_formulations(&self, py: Python<'_>, profile: &str) -> PyResult<Py<PyTuple>> {
        rendering::formulations(self, py, profile)
    }

    /// Alpha-normalized structural evidence, separate from exact artifact identity.
    #[getter]
    fn structural_fingerprint(&self, py: Python<'_>) -> PyResult<PyStructuralSemanticFingerprint> {
        panic_boundary(py, || {
            let document = self
                .document()
                .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))?
                .clone();
            py.detach(move || document.structural_fingerprint())
                .map(|value| PyStructuralSemanticFingerprint { value })
                .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))
        })
    }

    /// Compare structural meaning without changing exact Model equality.
    fn structurally_equivalent(&self, py: Python<'_>, other: &PyModel) -> PyResult<bool> {
        panic_boundary(py, || {
            let left = self
                .document()
                .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))?
                .clone();
            let right = other
                .document()
                .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))?
                .clone();
            py.detach(move || left.structurally_equivalent(&right))
                .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))
        })
    }

    /// Stable ULIDs of Fields that can appear in results.
    #[getter]
    fn field_ids(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        match &self.document {
            Some(document) => Ok(document
                .program()
                .nodes()
                .filter(|node| node.id().kind() == EntityKind::Field)
                .map(|node| node.id().ulid().to_string())
                .collect()),
            None => self
                .artifact_ids(EntityKind::Field)
                .map_err(|diagnostics| diagnostic_error(py, &diagnostics)),
        }
    }

    /// Stable ULIDs of Parameters addressable by value edits.
    #[getter]
    fn parameter_ids(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        Ok(self
            .document()
            .map_err(|diagnostic| diagnostic_error(py, &[diagnostic]))?
            .program()
            .nodes()
            .filter(|node| node.id().kind() == EntityKind::Parameter)
            .map(|node| node.id().ulid().to_string())
            .collect())
    }

    /// Stable ULIDs of semantic Domains in this exact Model.
    #[getter]
    fn domain_ids(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        match &self.document {
            Some(document) => Ok(document
                .program()
                .nodes()
                .filter(|node| node.id().kind() == EntityKind::Domain)
                .map(|node| node.id().ulid().to_string())
                .collect()),
            None => self
                .artifact_ids(EntityKind::Domain)
                .map_err(|diagnostics| diagnostic_error(py, &diagnostics)),
        }
    }

    /// Start reference execution with complete clock-indexed input tables.
    #[pyo3(signature = (*, end_time_s, max_step_s, inputs))]
    fn execution_session(
        &self,
        py: Python<'_>,
        end_time_s: f64,
        max_step_s: f64,
        inputs: &Bound<'_, pyo3::types::PyDict>,
    ) -> PyResult<crate::execution_session::PyExecutionSession> {
        let document = self
            .document()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        crate::execution_session::start(py, document, end_time_s, max_step_s, inputs)
    }

    /// Resume one accepted in-memory checkpoint on this exact immutable Model.
    fn resume_execution(
        &self,
        py: Python<'_>,
        checkpoint: &crate::execution_session::PyExecutionCheckpoint,
    ) -> PyResult<crate::execution_session::PyExecutionSession> {
        let document = self
            .document()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        crate::execution_session::resume(py, document, checkpoint)
    }

    /// Inspect an exact enum declaration by source alias or retained canonical ULID.
    #[pyo3(name = "enum")]
    fn enum_definition(
        &self,
        py: Python<'_>,
        selection: &str,
    ) -> PyResult<crate::modeling::enumeration::PyEnum> {
        enumeration::select(self, py, selection)
    }

    /// Resolve a source alias or exact ULID once into an exact Parameter role.
    fn parameter(&self, py: Python<'_>, selection: &str) -> PyResult<PyModelParameterRef> {
        panic_boundary(py, || {
            let document = self
                .document()
                .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
            PyModelParameterRef::from_document(document, selection)
                .map_err(|diagnostic| validation_error(py, &[diagnostic]))
        })
    }

    /// Select an exact inequality or complementarity by Relation and condition ordinal.
    fn constraint(
        &self,
        py: Python<'_>,
        selection: &str,
        ordinal: u32,
    ) -> PyResult<constraint::PyConstraintRef> {
        constraint::select(self, py, selection, ordinal)
    }

    /// Select an exact Activation from this Model's aliases or canonical IDs.
    fn activation(&self, py: Python<'_>, selection: &str) -> PyResult<PyActivationRef> {
        activation_ref::select(self, py, selection)
    }

    /// Select an exact derived output from this Model's aliases or canonical IDs.
    fn observable(&self, py: Python<'_>, selection: &str) -> PyResult<PyObservableRef> {
        observable_ref::select(self, py, selection)
    }

    /// Resolve a source alias or exact ULID once into an exact Field role.
    fn field(&self, py: Python<'_>, selection: &str) -> PyResult<PyModelFieldRef> {
        panic_boundary(py, || {
            if let Some(document) = &self.document {
                let value = document
                    .field_ref(selection)
                    .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
                return Ok(PyModelFieldRef::from_exact(
                    value.model().artifact().to_string(),
                    value.id().ulid().to_string(),
                ));
            }
            let field_ids = self
                .artifact_ids(EntityKind::Field)
                .map_err(|diagnostics| validation_error(py, &diagnostics))?;
            if !field_ids.iter().any(|id| id == selection) {
                return Err(validation_error(
                    py,
                    &[Diagnostic::error(
                        codes::NODE_NOT_FOUND,
                        "deferred-admission Model field selection requires an exact Field ULID",
                    )],
                ));
            }
            let reference = self
                .artifact
                .artifact_reference()
                .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
            Ok(PyModelFieldRef::from_exact(
                reference.artifact().to_string(),
                selection.to_owned(),
            ))
        })
    }

    /// Resolve a source alias or exact ULID once into an exact Domain role.
    fn domain(&self, py: Python<'_>, selection: &str) -> PyResult<PyModelDomainRef> {
        panic_boundary(py, || {
            let (model_digest, id) = if let Some(document) = &self.document {
                let value = document
                    .domain_ref(selection)
                    .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
                (
                    value.model().artifact().to_string(),
                    value.id().ulid().to_string(),
                )
            } else {
                let ids = self
                    .artifact_ids(EntityKind::Domain)
                    .map_err(|diagnostics| validation_error(py, &diagnostics))?;
                if !ids.iter().any(|id| id == selection) {
                    return Err(validation_error(
                        py,
                        &[Diagnostic::error(
                            codes::NODE_NOT_FOUND,
                            "deferred-admission Model domain selection requires an exact Domain ULID",
                        )],
                    ));
                }
                (self.revision.digest.clone(), selection.to_owned())
            };
            Ok(PyModelDomainRef { model_digest, id })
        })
    }

    /// Prepare an exact-base complete typed value edit without mutating this Model.
    fn preview_value_edit(
        &self,
        py: Python<'_>,
        target: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyValueEdit> {
        panic_boundary(py, || {
            let target = self
                .resolve_edit_target(target)
                .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
            let document = self
                .document()
                .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
            let before = document.program().typed_value(target).ok_or_else(|| {
                PyTypeError::new_err("value edit target has no complete Parameter value")
            })?;
            let value =
                crate::modeling::value_literal::from_python(value, before.value_type().clone())?;
            document
                .preview_value_edit(target, value)
                .map(|plan| PyValueEdit { plan })
                .map_err(|diagnostic| validation_error(py, &[diagnostic]))
        })
    }

    /// Atomically commit an exact-base edit into a new immutable child Model.
    fn commit(&self, py: Python<'_>, edit: &PyValueEdit) -> PyResult<Self> {
        panic_boundary(py, || {
            let document = self
                .document()
                .map_err(|diagnostic| validation_error(py, &[diagnostic]))?
                .clone();
            let plan = edit.plan.clone();
            let child = py
                .detach(move || document.commit_value_edit(plan))
                .map_err(|diagnostics| validation_error(py, &diagnostics))?
                .into_document();
            Self::from_document(py, child)
        })
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, Self>>()
            .is_ok_and(|other| self.revision == other.revision)
    }

    fn __ne__(&self, other: &Bound<'_, PyAny>) -> bool {
        !self.__eq__(other)
    }

    fn __hash__(&self) -> isize {
        hash_value(&self.revision) as isize
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(model_id={:?}, revision={}, digest={:?})",
            self.revision.model_id, self.revision.number, self.revision.digest
        )
    }
}

fn hash_value(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    authored_formulation::register(module)?;
    module.add_class::<PyStructuralSemanticFingerprint>()?;
    module.add_class::<PyPropertyBinding>()?;
    module.add_class::<PyRevision>()?;
    module.add_class::<PyValueEdit>()?;
    module.add_class::<PyModelParameterRef>()?;
    module.add_class::<PyModelFieldRef>()?;
    module.add_class::<PyModelDomainRef>()?;
    module.add_class::<PyObservableRef>()?;
    module.add_class::<constraint::PyConstraintRef>()?;
    module.add_class::<PyActivationRef>()?;
    module.add_class::<PyModel>()?;
    Ok(())
}
