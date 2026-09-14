use super::*;
use eqiora::ValueFrame;
use std::collections::BTreeSet;

type ExactFsiEntityComponents = BTreeMap<usize, BTreeMap<usize, BTreeMap<(usize, usize), f64>>>;

/// Immutable exact-Field-bound coherent-SI initial coefficients.
#[pyclass(
    name = "InitialField",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug)]
pub(crate) struct PyInitialField {
    pub(crate) native: CommonInitialField,
    field: Py<PyModelFieldRef>,
}

#[pymethods]
impl PyInitialField {
    #[new]
    #[pyo3(signature = (field, /, *, vertex_values=None, cell_values=None))]
    fn new(
        py: Python<'_>,
        field: Py<PyModelFieldRef>,
        vertex_values: Option<&Bound<'_, PyAny>>,
        cell_values: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let field_ref = field.borrow(py);
        let model = ArtifactDigest::from_hex(field_ref.exact_model_digest().to_owned())
            .map_err(|diagnostic| crate::error::validation_error(py, &[diagnostic]))?;
        let id = ulid::Ulid::from_string(field_ref.exact_id())
            .map(eqiora::Id::<eqiora::kinds::Field>::from_ulid)
            .map_err(|_| PyValueError::new_err("FieldRef contains an invalid exact Field ULID"))?;
        let vertex = vertex_values.map(extract_initial_values).transpose()?;
        let cell = cell_values.map(extract_initial_values).transpose()?;
        let native = CommonInitialField::new(model, id, vertex, cell)
            .map_err(|diagnostic| crate::error::validation_error(py, &[diagnostic]))?;
        drop(field_ref);
        Ok(Self { native, field })
    }

    #[getter]
    fn field(&self, py: Python<'_>) -> Py<PyModelFieldRef> {
        self.field.clone_ref(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "InitialField(field={:?}, vertex_values={}, cell_values={})",
            self.native.field().to_string(),
            self.native.vertex().is_some(),
            self.native.cell().is_some(),
        )
    }
}

fn extract_initial_values(value: &Bound<'_, PyAny>) -> PyResult<CommonInitialValues> {
    let normalized = value
        .cast::<PySequence>()
        .is_err()
        .then(|| value.call_method0("tolist"))
        .transpose()
        .map_err(|_| {
            PyValueError::new_err(
                "InitialField values must be a finite scalar or 2-vector sequence",
            )
        })?;
    let sequence = normalized
        .as_ref()
        .unwrap_or(value)
        .cast::<PySequence>()
        .map_err(|_| {
            PyValueError::new_err(
                "InitialField values must be a finite scalar or 2-vector sequence",
            )
        })?;
    let length = sequence.len()?;
    if length == 0 {
        return Err(PyValueError::new_err(
            "InitialField value sequences must be nonempty",
        ));
    }
    let first = sequence.get_item(0)?;
    if first.cast::<PySequence>().is_ok() && !first.is_instance_of::<pyo3::types::PyString>() {
        let mut values = Vec::with_capacity(length);
        for index in 0..length {
            let row = sequence
                .get_item(index)?
                .cast_into::<PySequence>()
                .map_err(|_| PyValueError::new_err("InitialField vector rows must be sequences"))?;
            if row.len()? != 2 {
                return Err(PyValueError::new_err(
                    "InitialField vectors must have exactly two components",
                ));
            }
            let mut vector = [0.0; 2];
            for (component, value) in vector.iter_mut().enumerate() {
                let item = row.get_item(component)?;
                if item.is_instance_of::<PyBool>() {
                    return Err(PyValueError::new_err("InitialField values reject booleans"));
                }
                *value = item.extract::<f64>().map_err(|_| {
                    PyValueError::new_err("InitialField values must be finite real numbers")
                })?;
            }
            values.push(vector);
        }
        Ok(CommonInitialValues::Vector2(values.into_boxed_slice()))
    } else {
        let mut values = Vec::with_capacity(length);
        for index in 0..length {
            let item = sequence.get_item(index)?;
            if item.is_instance_of::<PyBool>() {
                return Err(PyValueError::new_err("InitialField values reject booleans"));
            }
            values.push(item.extract::<f64>().map_err(|_| {
                PyValueError::new_err("InitialField values must be finite real numbers")
            })?);
        }
        Ok(CommonInitialValues::Scalar(values.into_boxed_slice()))
    }
}

enum ProjectedValues {
    Scalar(ReadOnlyVector<f64>),
    Vector(ReadOnlyMatrix<f64>),
}

impl ProjectedValues {
    fn numpy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Scalar(values) => Ok(values.numpy(py)?.into_any()),
            Self::Vector(values) => Ok(values.numpy(py)?.into_any()),
        }
    }
}

struct ProjectedBlock {
    association: &'static str,
    digest: String,
    values: ProjectedValues,
    support_indices: Arc<ReadOnlyVector<u32>>,
}

/// One exact semantic Field observation in an accepted trajectory state.
#[pyclass(
    name = "FieldSnapshot",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
pub(crate) struct PyFieldSnapshot {
    digest: String,
    mesh_digest: String,
    field: Py<PyModelFieldRef>,
    field_id: String,
    support_domain_id: String,
    dimension: DimExponents,
    value_shape: Vec<u32>,
    frame: &'static str,
    blocks: Vec<ProjectedBlock>,
}

/// Immutable typed result of one mathematical operation over an accepted
/// spatial Field snapshot.
#[pyclass(
    name = "DerivedFieldSnapshot",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
pub(crate) struct PyDerivedFieldSnapshot {
    digest: String,
    source_state_digest: String,
    source_field: Py<PyModelFieldRef>,
    source_field_id: String,
    mesh_digest: String,
    support_domain_id: String,
    operator: &'static str,
    dimension: DimExponents,
    value_shape: Vec<u32>,
    frame: &'static str,
    blocks: Vec<ProjectedBlock>,
}

impl PartialEq for PyDerivedFieldSnapshot {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest
    }
}

impl Eq for PyDerivedFieldSnapshot {}

impl Hash for PyDerivedFieldSnapshot {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.digest.hash(state);
    }
}

impl PyDerivedFieldSnapshot {
    pub(super) fn from_cell_average_curl(
        py: Python<'_>,
        model_digest: &str,
        source_state_digest: &str,
        source_field: &PyModelFieldRef,
        mesh_digest: &str,
        support_domain_id: &str,
        values: &[f64],
    ) -> PyResult<Self> {
        const INVERSE_TIME: DimExponents =
            DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).expect("bounded dimension");
        let source_field_id = source_field.exact_id().to_owned();
        let block = common_scalar_block("cell", values)?;
        let mut hasher = Sha256::new();
        hasher.update(b"eqiora.common-derived-field-snapshot/v1\0");
        for value in [
            model_digest,
            source_state_digest,
            source_field_id.as_str(),
            mesh_digest,
            support_domain_id,
            "curl/cell-average-2d",
            block.digest.as_str(),
        ] {
            hasher.update((value.len() as u64).to_be_bytes());
            hasher.update(value.as_bytes());
        }
        Ok(Self {
            digest: hex_sha256(hasher.finalize().as_slice()),
            source_state_digest: source_state_digest.to_owned(),
            source_field: Py::new(
                py,
                PyModelFieldRef::from_exact(model_digest.to_owned(), source_field_id.clone()),
            )?,
            source_field_id,
            mesh_digest: mesh_digest.to_owned(),
            support_domain_id: support_domain_id.to_owned(),
            operator: "curl",
            dimension: INVERSE_TIME,
            value_shape: Vec::new(),
            frame: "spatial-axial",
            blocks: vec![block],
        })
    }
}

#[pymethods]
impl PyDerivedFieldSnapshot {
    #[getter]
    fn digest(&self) -> &str {
        &self.digest
    }

    #[getter]
    fn source_state_digest(&self) -> &str {
        &self.source_state_digest
    }

    #[getter]
    fn source_field(&self, py: Python<'_>) -> Py<PyModelFieldRef> {
        self.source_field.clone_ref(py)
    }

    #[getter]
    fn mesh_digest(&self) -> &str {
        &self.mesh_digest
    }

    #[getter]
    fn support_domain_id(&self) -> &str {
        &self.support_domain_id
    }

    #[getter]
    fn operator(&self) -> &'static str {
        self.operator
    }

    #[getter]
    fn dimension(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::modeling::dimension::exponents(py, self.dimension)
    }

    #[getter]
    fn value_shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.value_shape.iter().copied())?.unbind())
    }

    #[getter]
    fn frame(&self) -> &'static str {
        self.frame
    }

    #[getter]
    fn associations(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.blocks.iter().map(|block| block.association))?.unbind())
    }

    #[getter]
    fn block_digests(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let entries = self
            .blocks
            .iter()
            .map(|block| (block.association, block.digest.as_str()));
        Ok(PyTuple::new(py, entries)?.unbind())
    }

    #[pyo3(signature = (association, /))]
    fn values(&self, py: Python<'_>, association: &str) -> PyResult<Py<PyAny>> {
        self.blocks
            .iter()
            .find(|block| block.association == association)
            .ok_or_else(|| PyKeyError::new_err(association.to_owned()))?
            .values
            .numpy(py)
    }

    #[pyo3(signature = (association, /))]
    fn support_indices(&self, py: Python<'_>, association: &str) -> PyResult<Py<PyArray1<u32>>> {
        self.blocks
            .iter()
            .find(|block| block.association == association)
            .ok_or_else(|| PyKeyError::new_err(association.to_owned()))?
            .support_indices
            .numpy(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "DerivedFieldSnapshot(operator={:?}, source_field_id={:?}, digest={:?})",
            self.operator, self.source_field_id, self.digest,
        )
    }
}

impl PartialEq for PyFieldSnapshot {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest
    }
}

impl Eq for PyFieldSnapshot {}

impl Hash for PyFieldSnapshot {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.digest.hash(state);
    }
}

impl PyFieldSnapshot {
    pub(super) fn from_common_fsi(
        py: Python<'_>,
        plan: &eqiora_numerics::CommonFsiPlan,
        state: &CommonState,
        mesh_digest: &str,
    ) -> PyResult<Vec<Self>> {
        let physical = state
            .fsi_fields()
            .ok_or_else(|| PyRuntimeError::new_err("common FSI State omitted exact Fields"))?;
        let fields = physical
            .fields()
            .map(|(field, domain, value_type)| (field, domain, value_type.clone()))
            .collect::<Vec<_>>();
        let state_field_ids = fields
            .iter()
            .map(|(field, _, _)| field.to_string())
            .collect::<BTreeSet<_>>();
        let state_domain_ids = fields
            .iter()
            .map(|(_, domain, _)| domain.to_string())
            .collect::<BTreeSet<_>>();
        if state_field_ids != plan.field_ids().iter().cloned().collect()
            || state_domain_ids != plan.domain_ids().iter().cloned().collect()
        {
            return Err(PyRuntimeError::new_err(
                "common FSI State Field/Domain inventory differs from its exact Plan",
            ));
        }

        fields
            .into_iter()
            .map(|(field, domain, value_type)| {
                let coefficients = physical.coefficients(field).ok_or_else(|| {
                    PyRuntimeError::new_err("common FSI State omitted an exact Field")
                })?;
                let mut entities = ExactFsiEntityComponents::new();
                for (entity, slot, component, value) in coefficients {
                    entities
                        .entry(entity.dimension())
                        .or_default()
                        .entry(entity.index())
                        .or_default()
                        .insert((slot, component), value);
                }
                let component_count = value_type.shape().component_count().ok_or_else(|| {
                    PyRuntimeError::new_err("FSI Field component count is not representable")
                })?;
                let blocks = common_fsi_blocks(entities, component_count)?;
                Self::from_common_exact_parts(
                    py,
                    plan.model_digest(),
                    mesh_digest,
                    &field.to_string(),
                    &domain.to_string(),
                    value_type.dimension(),
                    value_type
                        .shape()
                        .extents()
                        .iter()
                        .map(|extent| extent.get())
                        .collect(),
                    match value_type.frame() {
                        ValueFrame::Invariant => "invariant",
                        ValueFrame::SpatialCartesian => "spatial-cartesian",
                    },
                    blocks,
                )
            })
            .collect()
    }
}

#[pymethods]
impl PyFieldSnapshot {
    /// Exact accepted Field snapshot identity.
    #[getter]
    fn digest(&self) -> &str {
        &self.digest
    }

    /// Exact Mesh artifact on which this observation is defined.
    #[getter]
    fn mesh_digest(&self) -> &str {
        &self.mesh_digest
    }

    /// Exact Model-bound semantic Field identity.
    #[getter]
    fn field(&self, py: Python<'_>) -> Py<PyModelFieldRef> {
        self.field.clone_ref(py)
    }

    /// Exact volume Domain supporting this Field.
    #[getter]
    fn support_domain_id(&self) -> &str {
        &self.support_domain_id
    }

    /// Coherent-SI base exponents in M,L,T,I,Theta,N,J order.
    #[getter]
    fn dimension(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::modeling::dimension::exponents(py, self.dimension)
    }

    /// Exact mathematical component shape; an empty tuple is scalar.
    #[getter]
    fn value_shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.value_shape.iter().copied())?.unbind())
    }

    /// Coordinate-frame meaning of the mathematical components.
    #[getter]
    fn frame(&self) -> &'static str {
        self.frame
    }

    /// Coefficient associations in exact snapshot-edge order.
    #[getter]
    fn associations(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.blocks.iter().map(|block| block.association))?.unbind())
    }

    /// Exact block identities paired with their coefficient associations.
    #[getter]
    fn block_digests(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let entries = self
            .blocks
            .iter()
            .map(|block| (block.association, block.digest.as_str()));
        Ok(PyTuple::new(py, entries)?.unbind())
    }

    /// Read-only NumPy coefficients for one exact association.
    #[pyo3(signature = (association, /))]
    fn values(&self, py: Python<'_>, association: &str) -> PyResult<Py<PyAny>> {
        self.blocks
            .iter()
            .find(|block| block.association == association)
            .ok_or_else(|| PyKeyError::new_err(association.to_owned()))?
            .values
            .numpy(py)
    }

    /// Read-only exact global mesh-entity indices in the Field support.
    #[pyo3(signature = (association, /))]
    fn support_indices(&self, py: Python<'_>, association: &str) -> PyResult<Py<PyArray1<u32>>> {
        self.blocks
            .iter()
            .find(|block| block.association == association)
            .ok_or_else(|| PyKeyError::new_err(association.to_owned()))?
            .support_indices
            .numpy(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "FieldSnapshot(field_id={:?}, digest={:?})",
            self.field_id, self.digest,
        )
    }
}

/// One accepted physical state in exact trajectory order.
impl PyFieldSnapshot {
    pub(super) fn from_common_velocity(
        py: Python<'_>,
        plan: &eqiora_numerics::CommonTransientFlowPlan,
        state: &CommonState,
        mesh_digest: &str,
    ) -> PyResult<Self> {
        const VELOCITY: DimExponents =
            DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
        let mut blocks = Vec::new();
        if let Some(values) = state.velocity_vertex_values() {
            blocks.push(common_vector_block("vertex", values)?);
        }
        if let Some(values) = state.velocity_cell_values() {
            blocks.push(common_vector_block("cell", &values)?);
        }
        Self::from_common_parts(
            py,
            plan,
            mesh_digest,
            plan.velocity_field_id(),
            VELOCITY,
            vec![2],
            "spatial-cartesian",
            blocks,
        )
    }

    pub(super) fn from_common_pressure(
        py: Python<'_>,
        plan: &eqiora_numerics::CommonTransientFlowPlan,
        state: &CommonState,
        mesh_digest: &str,
    ) -> PyResult<Self> {
        const PRESSURE: DimExponents =
            DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");
        let block = match (state.pressure_vertex_values(), state.pressure_cell_values()) {
            (Some(values), None) => common_scalar_block("vertex", values)?,
            (None, Some(values)) => common_scalar_block("cell", values)?,
            _ => {
                return Err(PyRuntimeError::new_err(
                    "common pressure State lost its exact coefficient association",
                ));
            }
        };
        Self::from_common_parts(
            py,
            plan,
            mesh_digest,
            plan.pressure_field_id(),
            PRESSURE,
            Vec::new(),
            "invariant",
            vec![block],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_common_parts(
        py: Python<'_>,
        plan: &eqiora_numerics::CommonTransientFlowPlan,
        mesh_digest: &str,
        field_id: &str,
        dimension: DimExponents,
        value_shape: Vec<u32>,
        frame: &'static str,
        blocks: Vec<ProjectedBlock>,
    ) -> PyResult<Self> {
        let mut hasher = Sha256::new();
        hasher.update(b"eqiora.common-field-snapshot/v1\0");
        hasher.update(plan.model_digest().as_bytes());
        hasher.update(mesh_digest.as_bytes());
        hasher.update(field_id.as_bytes());
        for block in &blocks {
            hasher.update(block.digest.as_bytes());
        }
        let digest = hex_sha256(hasher.finalize().as_slice());
        let field = Py::new(
            py,
            PyModelFieldRef::from_exact(plan.model_digest().to_owned(), field_id.to_owned()),
        )?;
        Ok(Self {
            digest,
            mesh_digest: mesh_digest.to_owned(),
            field,
            field_id: field_id.to_owned(),
            support_domain_id: plan.domain_id(),
            dimension,
            value_shape,
            frame,
            blocks,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn from_common_exact_parts(
        py: Python<'_>,
        model_digest: &str,
        mesh_digest: &str,
        field_id: &str,
        support_domain_id: &str,
        dimension: DimExponents,
        value_shape: Vec<u32>,
        frame: &'static str,
        blocks: Vec<ProjectedBlock>,
    ) -> PyResult<Self> {
        let mut hasher = Sha256::new();
        hasher.update(b"eqiora.common-field-snapshot/v1\0");
        hasher.update(model_digest.as_bytes());
        hasher.update(mesh_digest.as_bytes());
        hasher.update(field_id.as_bytes());
        for block in &blocks {
            hasher.update(block.digest.as_bytes());
        }
        Ok(Self {
            digest: hex_sha256(hasher.finalize().as_slice()),
            mesh_digest: mesh_digest.to_owned(),
            field: Py::new(
                py,
                PyModelFieldRef::from_exact(model_digest.to_owned(), field_id.to_owned()),
            )?,
            field_id: field_id.to_owned(),
            support_domain_id: support_domain_id.to_owned(),
            dimension,
            value_shape,
            frame,
            blocks,
        })
    }
}

fn common_vector_block(association: &'static str, values: &[[f64; 2]]) -> PyResult<ProjectedBlock> {
    let coefficients = values.iter().flatten().copied().collect::<Vec<_>>();
    common_block(
        association,
        &coefficients.clone(),
        ProjectedValues::Vector(ReadOnlyMatrix::new(values.len(), 2, coefficients)),
        values.len(),
    )
}

fn common_scalar_block(association: &'static str, values: &[f64]) -> PyResult<ProjectedBlock> {
    common_block(
        association,
        values,
        ProjectedValues::Scalar(ReadOnlyVector::new(values.to_vec())),
        values.len(),
    )
}

fn common_fsi_blocks(
    entities: ExactFsiEntityComponents,
    component_count: usize,
) -> PyResult<Vec<ProjectedBlock>> {
    entities
        .into_iter()
        .map(|(dimension, entities)| {
            let association = match dimension {
                0 => "vertex",
                2 => "cell",
                _ => {
                    return Err(PyRuntimeError::new_err(
                        "FSI Field has an unsupported exact entity association",
                    ));
                }
            };
            let indices = entities.keys().copied().collect::<Vec<_>>();
            match component_count {
                1 => {
                    let values = entities
                        .into_values()
                        .map(|components| exact_fsi_component(&components, 1, 0))
                        .collect::<PyResult<Vec<_>>>()?;
                    common_scalar_block_at(association, &values, &indices)
                }
                2 => {
                    let values = entities
                        .into_values()
                        .map(|components| {
                            Ok([
                                exact_fsi_component(&components, 2, 0)?,
                                exact_fsi_component(&components, 2, 1)?,
                            ])
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    common_vector_block_at(association, &values, &indices)
                }
                _ => Err(PyRuntimeError::new_err(
                    "FSI Field snapshot supports exact scalar and 2-vector values",
                )),
            }
        })
        .collect()
}

fn exact_fsi_component(
    components: &BTreeMap<(usize, usize), f64>,
    component_count: usize,
    component: usize,
) -> PyResult<f64> {
    if components.len() == component_count
        && let Some(value) = components.get(&(0, component))
    {
        return Ok(*value);
    }
    Err(PyRuntimeError::new_err(
        "FSI Field entity lacks one exact single-slot component",
    ))
}

fn common_vector_block_at(
    association: &'static str,
    values: &[[f64; 2]],
    indices: &[usize],
) -> PyResult<ProjectedBlock> {
    if values.len() != indices.len() {
        return Err(PyRuntimeError::new_err(
            "FSI vector block cardinality differs from exact support",
        ));
    }
    let coefficients = values.iter().flatten().copied().collect::<Vec<_>>();
    common_block_at(
        association,
        &coefficients,
        ProjectedValues::Vector(ReadOnlyMatrix::new(values.len(), 2, coefficients.clone())),
        indices,
    )
}

fn common_scalar_block_at(
    association: &'static str,
    values: &[f64],
    indices: &[usize],
) -> PyResult<ProjectedBlock> {
    if values.len() != indices.len() {
        return Err(PyRuntimeError::new_err(
            "FSI scalar block cardinality differs from exact support",
        ));
    }
    common_block_at(
        association,
        values,
        ProjectedValues::Scalar(ReadOnlyVector::new(values.to_vec())),
        indices,
    )
}

fn common_block_at(
    association: &'static str,
    coefficients: &[f64],
    values: ProjectedValues,
    indices: &[usize],
) -> PyResult<ProjectedBlock> {
    let support_indices = indices
        .iter()
        .map(|&index| {
            u32::try_from(index)
                .map_err(|_| PyOverflowError::new_err("Field support index exceeds Python uint32"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mut hasher = Sha256::new();
    hasher.update(b"eqiora.common-field-block/v1\0");
    hasher.update(association.as_bytes());
    for value in coefficients {
        hasher.update(value.to_bits().to_be_bytes());
    }
    Ok(ProjectedBlock {
        association,
        digest: hex_sha256(hasher.finalize().as_slice()),
        values,
        support_indices: Arc::new(ReadOnlyVector::new(support_indices)),
    })
}

fn common_block(
    association: &'static str,
    coefficients: &[f64],
    values: ProjectedValues,
    count: usize,
) -> PyResult<ProjectedBlock> {
    let support_indices = (0..count)
        .map(|index| {
            u32::try_from(index)
                .map_err(|_| PyOverflowError::new_err("Field support index exceeds Python uint32"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mut hasher = Sha256::new();
    hasher.update(b"eqiora.common-field-block/v1\0");
    hasher.update(association.as_bytes());
    for value in coefficients {
        hasher.update(value.to_bits().to_be_bytes());
    }
    Ok(ProjectedBlock {
        association,
        digest: hex_sha256(hasher.finalize().as_slice()),
        values,
        support_indices: Arc::new(ReadOnlyVector::new(support_indices)),
    })
}

impl PyFieldSnapshot {
    pub(super) fn from_common_scalar(
        py: Python<'_>,
        plan: &eqiora_numerics::CommonScalarPlan,
        state: &CommonState,
        mesh_digest: &str,
    ) -> PyResult<Self> {
        let (field, value_type) = plan.fields().next().expect("scalar storage Field");
        let values = state
            .scalar_values()
            .ok_or_else(|| PyValueError::new_err("missing scalar State values"))?;
        Self::from_common_exact_parts(
            py,
            plan.model_digest(),
            mesh_digest,
            &field.ulid().to_string(),
            &plan
                .storage_domain_id()
                .map_err(|d| crate::error::validation_error(py, &[d]))?,
            value_type.dimension(),
            vec![],
            "invariant",
            vec![common_scalar_block("vertex", values)?],
        )
    }
}
