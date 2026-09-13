//! Checked nominal declarations shared by native drafts and Python Source scopes.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use eqiora::kernel::{FiniteSpaceDef, IndexSetDef};
use eqiora::language::{NamePath, TextRange, ValueTypeSyntax};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt, PyTuple};

use super::PyValueType;

pub(super) fn name_path(name: &str) -> PyResult<NamePath> {
    NamePath::from_segments([name], TextRange::default())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn qualified_name_path(name: &str) -> PyResult<NamePath> {
    NamePath::from_segments(name.split('.'), TextRange::default())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Immutable exact finite basis, independent of numerical discretization spaces.
#[pyclass(
    name = "FiniteSpace",
    module = "eqiora._eqiora",
    frozen,
    eq,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PyFiniteSpace {
    pub(crate) name: String,
    pub(crate) value: FiniteSpaceDef,
}

#[pymethods]
impl PyFiniteSpace {
    #[new]
    #[pyo3(signature = (name, *, labels))]
    fn new(name: String, labels: Vec<String>) -> PyResult<Self> {
        name_path(&name)?;
        for label in &labels {
            name_path(label)?;
        }
        let value = FiniteSpaceDef::new(eqiora::Id::new(), labels)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        ValueTypeSyntax::validate_checked(&value.coordinates())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { name, value })
    }
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }
    #[getter]
    fn id(&self) -> String {
        self.value.id().to_string()
    }
    #[getter]
    fn labels(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.value.labels())?.unbind())
    }
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.value.id().hash(&mut hasher);
        hasher.finish()
    }
    fn __repr__(&self) -> String {
        format!(
            "FiniteSpace(name={:?}, id={:?}, labels={:?})",
            self.name,
            self.id(),
            self.value.labels()
        )
    }
}

/// Immutable nominal zero-based index set with a checked constant extent.
#[pyclass(
    name = "IndexSet",
    module = "eqiora._eqiora",
    frozen,
    eq,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PyIndexSet {
    pub(crate) name: String,
    pub(crate) value: IndexSetDef,
}

#[pymethods]
impl PyIndexSet {
    #[new]
    #[pyo3(signature = (name, *, extent))]
    fn new(name: String, extent: &Bound<'_, PyAny>) -> PyResult<Self> {
        name_path(&name)?;
        if extent.is_instance_of::<PyBool>() || !extent.is_instance_of::<PyInt>() {
            return Err(PyTypeError::new_err(
                "IndexSet extent must be a constant exact int, not bool or expression",
            ));
        }
        let value = IndexSetDef::new(eqiora::Id::new(), extent.extract::<u32>()?)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { name, value })
    }
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }
    #[getter]
    fn id(&self) -> String {
        self.value.id().to_string()
    }
    #[getter]
    fn extent(&self) -> u32 {
        self.value.extent()
    }
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.value.id().hash(&mut hasher);
        hasher.finish()
    }
    fn __repr__(&self) -> String {
        format!(
            "IndexSet(name={:?}, id={:?}, extent={})",
            self.name,
            self.id(),
            self.extent()
        )
    }
}

#[pyfunction]
pub(crate) fn _nominal_type(
    value_type: &PyValueType,
    spaces: Vec<(PyRef<'_, PyFiniteSpace>, String)>,
    sets: Vec<(PyRef<'_, PyIndexSet>, String)>,
    enums: Vec<(PyRef<'_, super::enumeration::PyEnum>, String)>,
) -> PyResult<crate::authoring::PyAstType> {
    let mut names = Vec::with_capacity(spaces.len() + sets.len() + enums.len());
    for (space, name) in spaces {
        names.push((space.value.id().erase(), qualified_name_path(&name)?));
    }
    for (set, name) in sets {
        names.push((set.value.id().erase(), qualified_name_path(&name)?));
    }
    for (definition, name) in enums {
        names.push((definition.value.id().erase(), qualified_name_path(&name)?));
    }
    ValueTypeSyntax::from_checked(&value_type.value, |id| {
        names
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, name)| name.clone())
    })
    // The handle resolves lexical ownership, not this module's canonical ID.
    .and_then(|value| {
        eqiora::language::SourceAstFactory::value_type(value.kind().clone(), value.range())
    })
    .map(|value| crate::authoring::PyAstType { value })
    .map_err(|error| PyValueError::new_err(error.to_string()))
}
