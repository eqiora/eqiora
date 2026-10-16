//! Immutable mesh intent and complete resolved provider choices.

use eqiora::artifact::{AffineTriangleMeshPolicy, CartesianMeshCellsV2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyTuple};

use super::gmsh;
use super::request_error;
use crate::error::validation_error;
use crate::geometry::{PyGeometry, digest_to_hex};
use crate::panic_boundary;

/// Exact Gmsh provider selection.
#[pyclass(
    name = "GmshMesher",
    module = "eqiora._eqiora",
    frozen,
    eq,
    skip_from_py_object
)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PyGmshMesher {
    pub(super) maximum_boundary_error: f64,
    pub(super) minimum_mean_ratio: f64,
    pub(super) maximum_boundary_facets: usize,
    pub(super) maximum_target_size: Option<f64>,
}

#[pymethods]
impl PyGmshMesher {
    #[new]
    #[pyo3(signature = (*, maximum_boundary_error=1.0e-4, maximum_target_size=None, minimum_mean_ratio=1.0e-5, maximum_boundary_facets=50))]
    fn new(
        py: Python<'_>,
        maximum_boundary_error: f64,
        maximum_target_size: Option<f64>,
        minimum_mean_ratio: f64,
        maximum_boundary_facets: usize,
    ) -> PyResult<Self> {
        if maximum_target_size.is_some_and(|value| !value.is_finite() || value <= 0.0) {
            return Err(request_error(
                py,
                "maximum_target_size must be None or a finite positive value",
            ));
        }
        validate_numerical_policy(
            py,
            maximum_boundary_error,
            minimum_mean_ratio,
            maximum_boundary_facets,
        )?;
        Ok(Self {
            maximum_boundary_error,
            minimum_mean_ratio,
            maximum_boundary_facets,
            maximum_target_size,
        })
    }

    #[getter]
    const fn maximum_boundary_error(&self) -> f64 {
        self.maximum_boundary_error
    }

    #[getter]
    const fn minimum_mean_ratio(&self) -> f64 {
        self.minimum_mean_ratio
    }

    #[getter]
    const fn maximum_boundary_facets(&self) -> usize {
        self.maximum_boundary_facets
    }

    #[getter]
    const fn maximum_target_size(&self) -> Option<f64> {
        self.maximum_target_size
    }

    fn __repr__(&self) -> String {
        provider_repr(*self)
    }
}

/// Deterministic structured Cartesian provider selection.
#[pyclass(
    name = "CartesianMesher",
    module = "eqiora._eqiora",
    frozen,
    eq,
    skip_from_py_object
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PyCartesianMesher {
    pub(super) policy: CartesianMeshCellsV2,
}

#[pymethods]
impl PyCartesianMesher {
    #[new]
    #[pyo3(signature = (*, cells))]
    fn new(py: Python<'_>, cells: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let parsed = parse_cells(py, cells)?;
        let policy = CartesianMeshCellsV2::new(parsed)
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        Ok(Self { policy })
    }

    #[getter]
    fn cells(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.policy.cells().iter().copied())?.unbind())
    }

    fn __repr__(&self) -> String {
        let cells = self
            .policy
            .cells()
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let suffix = if self.policy.cells().len() == 1 {
            ","
        } else {
            ""
        };
        format!("CartesianMesher(cells=({cells}{suffix}))")
    }
}

/// Deterministic fixed-diagonal affine-triangle provider selection.
#[pyclass(
    name = "AffineTriangleMesher",
    module = "eqiora._eqiora",
    frozen,
    eq,
    skip_from_py_object
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PyAffineTriangleMesher {
    pub(super) policy: AffineTriangleMeshPolicy,
}

#[pymethods]
impl PyAffineTriangleMesher {
    #[new]
    #[pyo3(signature = (*, cells))]
    fn new(py: Python<'_>, cells: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let parsed: [usize; 2] = parse_cells(py, cells)?
            .try_into()
            .map_err(|_| request_error(py, "AffineTriangleMesher requires exactly (nx, ny)"))?;
        let policy = AffineTriangleMeshPolicy::new(parsed)
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        Ok(Self { policy })
    }

    #[getter]
    const fn cells(&self) -> (usize, usize) {
        let [nx, ny] = self.policy.cells();
        (nx, ny)
    }

    #[getter]
    const fn diagonal(&self) -> &'static str {
        self.policy.diagonal()
    }

    fn __repr__(&self) -> String {
        let (nx, ny) = self.cells();
        format!("AffineTriangleMesher(cells=({nx}, {ny}))")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum MeshProviderPolicy {
    Gmsh(PyGmshMesher),
    Cartesian(PyCartesianMesher),
    AffineTriangle(PyAffineTriangleMesher),
}

impl MeshProviderPolicy {
    fn to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Self::Gmsh(provider) => Py::new(py, *provider).map(Py::into_any),
            Self::Cartesian(provider) => Py::new(py, provider.clone()).map(Py::into_any),
            Self::AffineTriangle(provider) => Py::new(py, provider.clone()).map(Py::into_any),
        }
    }

    fn representation(&self) -> String {
        match self {
            Self::Gmsh(provider) => provider.__repr__(),
            Self::Cartesian(provider) => provider.__repr__(),
            Self::AffineTriangle(provider) => provider.__repr__(),
        }
    }
}

fn extract_provider(py: Python<'_>, provider: &Bound<'_, PyAny>) -> PyResult<MeshProviderPolicy> {
    if let Ok(provider) = provider.extract::<PyRef<'_, PyGmshMesher>>() {
        Ok(MeshProviderPolicy::Gmsh(*provider))
    } else if let Ok(provider) = provider.extract::<PyRef<'_, PyCartesianMesher>>() {
        Ok(MeshProviderPolicy::Cartesian(provider.clone()))
    } else if let Ok(provider) = provider.extract::<PyRef<'_, PyAffineTriangleMesher>>() {
        Ok(MeshProviderPolicy::AffineTriangle(provider.clone()))
    } else {
        Err(request_error(
            py,
            "provider must be GmshMesher, CartesianMesher, or AffineTriangleMesher",
        ))
    }
}

fn parse_cells(py: Python<'_>, cells: &Bound<'_, PyTuple>) -> PyResult<Vec<usize>> {
    if !(1..=3).contains(&cells.len()) {
        return Err(request_error(
            py,
            "cells must contain between one and three axis counts",
        ));
    }
    let mut parsed = Vec::with_capacity(cells.len());
    for value in cells.iter() {
        if value.is_instance_of::<PyBool>() {
            return Err(request_error(py, "cells must contain positive integers"));
        }
        parsed.push(
            value
                .extract::<usize>()
                .map_err(|_| request_error(py, "cells must contain positive integers"))?,
        );
    }
    Ok(parsed)
}

fn validate_numerical_policy(
    py: Python<'_>,
    maximum_boundary_error: f64,
    minimum_mean_ratio: f64,
    maximum_boundary_facets: usize,
) -> PyResult<()> {
    if !maximum_boundary_error.is_finite() || maximum_boundary_error <= 0.0 {
        return Err(request_error(
            py,
            "maximum_boundary_error must be finite and positive",
        ));
    }
    eqiora::meshing::MeshQualityGate::new(minimum_mean_ratio)
        .map_err(|diagnostic| validation_error(py, std::slice::from_ref(&diagnostic)))?;
    if maximum_boundary_facets < 8 {
        return Err(request_error(
            py,
            "maximum_boundary_facets must be at least eight",
        ));
    }
    Ok(())
}

fn provider_repr(provider: PyGmshMesher) -> String {
    let maximum_target_size = provider
        .maximum_target_size
        .map_or_else(|| "None".to_owned(), |value| value.to_string());
    format!(
        "GmshMesher(maximum_boundary_error={}, maximum_target_size={}, minimum_mean_ratio={}, maximum_boundary_facets={})",
        provider.maximum_boundary_error,
        maximum_target_size,
        provider.minimum_mean_ratio,
        provider.maximum_boundary_facets,
    )
}

/// Complete resolved provider choice bound to one exact Geometry.
///
/// Resolution owns the deterministic accepted resource privately so
/// `generate` can only publish the exact resource inspected through this plan.
#[pyclass(
    name = "MeshPlan",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
pub(super) struct PyMeshPlan {
    source_digest: String,
    pub(super) source: eqiora::geometry::CanonicalGeometryV1,
    pub(super) provider: MeshProviderPolicy,
    pub(super) planned: PlannedMesh,
}

pub(super) enum PlannedMesh {
    Gmsh(gmsh::GmshSizingReceipt),
    Cartesian,
    AffineTriangle,
}

#[pymethods]
impl PyMeshPlan {
    #[getter]
    fn source_digest(&self) -> &str {
        &self.source_digest
    }

    #[getter]
    fn provider(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.provider.to_python(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "MeshPlan(provider={}, source_digest={:?})",
            self.provider.representation(),
            self.source_digest,
        )
    }
}

/// Resolve one complete provider plan for the exact supplied Geometry.
#[pyfunction]
#[pyo3(signature = (geometry, provider, /))]
pub(super) fn resolve(
    py: Python<'_>,
    geometry: &PyGeometry,
    provider: &Bound<'_, PyAny>,
) -> PyResult<PyMeshPlan> {
    panic_boundary(py, || {
        let provider = extract_provider(py, provider)?;
        let planned = match &provider {
            MeshProviderPolicy::Gmsh(provider) => {
                let sizing = gmsh::plan(
                    geometry.geometry(),
                    provider.maximum_boundary_error,
                    provider.minimum_mean_ratio,
                    provider.maximum_boundary_facets,
                    provider.maximum_target_size,
                )
                .map_err(|diagnostic| validation_error(py, std::slice::from_ref(&diagnostic)))?;
                PlannedMesh::Gmsh(sizing)
            }
            MeshProviderPolicy::Cartesian(provider) => {
                let dimension = geometry.geometry().ambient_dimension();
                let admitted = geometry.geometry().planar_rectangle_bounds().is_some()
                    || geometry.geometry().cartesian_box_bounds().is_some();
                if !admitted || provider.policy.cells().len() != dimension {
                    return Err(request_error(
                        py,
                        "CartesianMesher requires dimension-matched exact Cartesian Geometry",
                    ));
                }
                validate_cartesian_boundary_extent(py, provider.policy.cells())?;
                PlannedMesh::Cartesian
            }
            MeshProviderPolicy::AffineTriangle(provider) => {
                let is_rectangle = geometry.geometry().planar_rectangle_bounds().is_some();
                let is_partition = geometry
                    .geometry()
                    .planar_adjacent_rectangle_partition()
                    .is_some();
                if !is_rectangle && !is_partition {
                    return Err(request_error(
                        py,
                        "AffineTriangleMesher requires planar rectangle or adjacent-partition Geometry v2",
                    ));
                }
                if is_partition && provider.policy.cells() != [2, 2] {
                    return Err(request_error(
                        py,
                        "the admitted adjacent-partition AffineTriangleMesher plan requires cells=(2, 2)",
                    ));
                }
                let cells = provider.policy.cells();
                validate_rectangle_boundary_extent(py, cells)?;
                PlannedMesh::AffineTriangle
            }
        };
        Ok(PyMeshPlan {
            source_digest: digest_to_hex(&geometry.geometry().digest_bytes()),
            source: geometry.geometry().clone(),
            provider,
            planned,
        })
    })
}

fn validate_rectangle_boundary_extent(py: Python<'_>, cells: [usize; 2]) -> PyResult<()> {
    cells[0]
        .checked_add(cells[1])
        .and_then(|sum| sum.checked_mul(2))
        .map(|_| ())
        .ok_or_else(|| request_error(py, "planned rectangle boundary extent overflows usize"))
}

fn validate_cartesian_boundary_extent(py: Python<'_>, cells: &[usize]) -> PyResult<()> {
    cells
        .iter()
        .try_fold(0_usize, |sum, &count| sum.checked_add(count))
        .and_then(|sum| sum.checked_mul(2))
        .map(|_| ())
        .ok_or_else(|| request_error(py, "planned Cartesian boundary extent overflows usize"))
}
