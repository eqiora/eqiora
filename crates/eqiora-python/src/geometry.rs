//! Bounded Python authoring over Rust-owned exact geometry meaning.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use eqiora::Diagnostic;
use eqiora::diagnostic::codes;
use eqiora::geometry::{CanonicalGeometryV1, NamedEntitySet};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule, PyTuple};

use crate::error::validation_error;

pub(crate) fn digest_to_hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// One immutable exact Geometry produced by an accepted authored graph.
///
/// Construction and observations delegate to the common canonical geometry
/// owner. Primitive shape does not create a parallel Python product type.
#[pyclass(
    name = "Geometry",
    module = "eqiora._eqiora",
    frozen,
    eq,
    skip_from_py_object
)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PyGeometry {
    geometry: CanonicalGeometryV1,
}

/// Immutable named selection bound to one exact Geometry revision.
#[pyclass(
    name = "GeometrySelection",
    module = "eqiora._eqiora",
    frozen,
    eq,
    skip_from_py_object
)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PyGeometrySelection {
    source_digest: String,
    name: String,
    dimension: usize,
}

#[pymethods]
impl PyGeometrySelection {
    #[getter]
    fn source_digest(&self) -> &str {
        &self.source_digest
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    const fn dimension(&self) -> usize {
        self.dimension
    }

    fn __repr__(&self) -> String {
        format!(
            "GeometrySelection(name={:?}, dimension={}, source_digest={:?})",
            self.name, self.dimension, self.source_digest,
        )
    }

    /// Hash the same revision-bound identity used by equality.
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.source_digest.hash(&mut hasher);
        self.name.hash(&mut hasher);
        self.dimension.hash(&mut hasher);
        hasher.finish()
    }
}

impl PyGeometrySelection {
    pub(crate) fn bound_source_digest(&self) -> &str {
        &self.source_digest
    }

    pub(crate) fn canonical_name(&self) -> &str {
        &self.name
    }

    pub(crate) const fn canonical_dimension(&self) -> usize {
        self.dimension
    }
}

#[pymethods]
impl PyGeometry {
    /// Intrinsic and coordinate dimension of this accepted Geometry.
    #[getter]
    fn dimension(&self) -> usize {
        self.geometry.ambient_dimension()
    }

    /// Exact Cartesian bounds, one pair per coordinate axis, in metres.
    #[getter]
    fn bounds(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let bounds = self.exact_bounds();
        Ok(PyTuple::new(py, bounds.into_iter().map(|[lower, upper]| (lower, upper)))?.unbind())
    }

    /// Producer classification tolerance in metres, absent for source-owned topology.
    #[getter]
    fn classification_tolerance(&self) -> Option<f64> {
        self.geometry.classification_tolerance_m()
    }

    /// Exact canonical JSON, without a trailing newline.
    #[getter]
    fn canonical_bytes(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, self.geometry.canonical_bytes()).unbind()
    }

    /// Lowercase domain-separated SHA-256 identity.
    #[getter]
    fn digest(&self) -> String {
        digest_to_hex(&self.geometry.digest_bytes())
    }

    /// Canonically ordered semantic selection names.
    #[getter]
    fn selection_names(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(
            py,
            self.geometry.entity_sets().iter().map(NamedEntitySet::name),
        )?
        .unbind())
    }

    /// Exact topological dimension of one fixed-role selection.
    fn selection_dimension(&self, py: Python<'_>, name: &str) -> PyResult<usize> {
        self.geometry
            .entity_set(name)
            .map(NamedEntitySet::dimension)
            .ok_or_else(|| {
                let diagnostic = Diagnostic::error(
                    codes::INVALID_ARTIFACT,
                    format!("Geometry has no selection named {name:?}"),
                );
                validation_error(py, std::slice::from_ref(&diagnostic))
            })
    }

    /// Resolve one canonical name into an immutable revision-bound selection.
    fn selection(&self, py: Python<'_>, name: &str) -> PyResult<PyGeometrySelection> {
        let entity_set = self.geometry.entity_set(name).ok_or_else(|| {
            let diagnostic = Diagnostic::error(
                codes::INVALID_ARTIFACT,
                format!("Geometry has no selection named {name:?}"),
            );
            validation_error(py, std::slice::from_ref(&diagnostic))
        })?;
        Ok(PyGeometrySelection {
            source_digest: self.digest(),
            name: entity_set.name().to_owned(),
            dimension: entity_set.dimension(),
        })
    }

    /// Hash the same canonical identity used by equality.
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.geometry.canonical_bytes().hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!(
            "Geometry(dimension={}, bounds={}, selections={}, digest={:?})",
            self.dimension(),
            self.bounds_repr(),
            self.geometry.entity_sets().len(),
            self.digest(),
        )
    }
}

impl PyGeometry {
    fn exact_bounds(&self) -> Vec<[f64; 2]> {
        self.geometry
            .cartesian_box_bounds()
            .map(<[_]>::to_vec)
            .or_else(|| {
                self.geometry
                    .planar_rectangle_bounds()
                    .or_else(|| self.geometry.circular_hole_bounds())
                    .map(|bounds| bounds.to_vec())
            })
            .or_else(|| {
                self.geometry
                    .planar_adjacent_rectangle_partition()
                    .map(|(bounds, _)| bounds.to_vec())
            })
            .or_else(|| {
                let vertices = self.geometry.region()?.vertices();
                let mut bounds = [[f64::INFINITY, f64::NEG_INFINITY]; 2];
                for vertex in vertices {
                    for axis in 0..2 {
                        bounds[axis][0] = bounds[axis][0].min(vertex[axis]);
                        bounds[axis][1] = bounds[axis][1].max(vertex[axis]);
                    }
                }
                Some(bounds.to_vec())
            })
            .expect("the admitted Geometry has exact Cartesian bounds")
    }

    fn bounds_repr(&self) -> String {
        let axes = self
            .exact_bounds()
            .into_iter()
            .map(|[lower, upper]| format!("({lower:?}, {upper:?})"))
            .collect::<Vec<_>>();
        if axes.len() == 1 {
            format!("({},)", axes[0])
        } else {
            format!("({})", axes.join(", "))
        }
    }

    pub(crate) const fn from_geometry(geometry: CanonicalGeometryV1) -> Self {
        Self { geometry }
    }

    pub(crate) const fn geometry(&self) -> &CanonicalGeometryV1 {
        &self.geometry
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGeometry>()?;
    module.add_class::<PyGeometrySelection>()
}
