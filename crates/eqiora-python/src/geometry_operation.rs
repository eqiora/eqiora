//! Python projection of the shared handle-first Geometry operation core.

use std::collections::BTreeMap;

use eqiora::geometry::{
    GeometryBoundaryHandle, GeometryGraph, GeometryOperation, GeometryRegionHandle,
    GeometryTopologyHandle,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyMapping, PyModule, PySequence, PyTuple};

use crate::cad_authored::{
    PyGeometryBuildReceipt, PyGeometrySolidOperation,
    extract_named_topology as extract_solid_names, extract_rectangle_pair, extract_sequence_pair,
};
use crate::error::validation_error;
use crate::geometry::PyGeometry;

#[pyclass(name = "GeometryGraph", module = "eqiora._eqiora", frozen)]
pub(crate) struct PyGeometryGraph {
    graph: GeometryGraph,
}

#[pyclass(
    name = "GeometryOperation",
    module = "eqiora._eqiora",
    frozen,
    eq,
    skip_from_py_object
)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PyGeometryOperation {
    operation: GeometryOperation,
}

#[pyclass(
    name = "GeometryRegionHandle",
    module = "eqiora._eqiora",
    frozen,
    eq,
    skip_from_py_object
)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PyGeometryRegionHandle {
    handle: GeometryRegionHandle,
}

#[pyclass(
    name = "GeometryBoundaryHandle",
    module = "eqiora._eqiora",
    frozen,
    eq,
    skip_from_py_object
)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PyGeometryBoundaryHandle {
    handle: GeometryBoundaryHandle,
}

#[pymethods]
impl PyGeometryGraph {
    #[new]
    fn new() -> Self {
        Self {
            graph: GeometryGraph::new(),
        }
    }

    /// Construct one simple polygon; coordinates and tolerance are metres.
    #[pyo3(signature = (*, vertices, tolerance))]
    fn polygon(
        &self,
        py: Python<'_>,
        vertices: Vec<[f64; 2]>,
        tolerance: f64,
    ) -> PyResult<PyGeometryOperation> {
        self.graph
            .polygon(vertices, tolerance)
            .map(|operation| PyGeometryOperation { operation })
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }

    /// Construct one closed convex shell from outward-oriented faces, in metres.
    #[pyo3(signature = (*, vertices, faces, tolerance))]
    fn convex_polyhedron(
        &self,
        py: Python<'_>,
        vertices: Vec<[f64; 3]>,
        faces: Vec<Vec<usize>>,
        tolerance: f64,
    ) -> PyResult<PyGeometryOperation> {
        self.graph
            .convex_polyhedron(vertices, faces, tolerance)
            .map(|operation| PyGeometryOperation { operation })
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }

    #[pyo3(signature = (*, x_bounds, y_bounds))]
    fn rectangle(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = extract_sequence_pair)] x_bounds: [f64; 2],
        #[pyo3(from_py_with = extract_sequence_pair)] y_bounds: [f64; 2],
    ) -> PyResult<PyGeometryOperation> {
        self.graph
            .rectangle(x_bounds, y_bounds)
            .map(|operation| PyGeometryOperation { operation })
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }

    #[pyo3(signature = (*, x_bounds, y_bounds, z_bounds))]
    fn cuboid(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = extract_sequence_pair)] x_bounds: [f64; 2],
        #[pyo3(from_py_with = extract_sequence_pair)] y_bounds: [f64; 2],
        #[pyo3(from_py_with = extract_sequence_pair)] z_bounds: [f64; 2],
    ) -> PyResult<PyGeometryOperation> {
        self.graph
            .cuboid(x_bounds, y_bounds, z_bounds)
            .map(|operation| PyGeometryOperation { operation })
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }

    #[pyo3(signature = (*, bounds))]
    fn interval(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = extract_sequence_pair)] bounds: [f64; 2],
    ) -> PyResult<PyGeometryOperation> {
        self.graph
            .interval(bounds)
            .map(|operation| PyGeometryOperation { operation })
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }

    #[pyo3(signature = (*, center, radius))]
    fn circle(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = extract_sequence_pair)] center: [f64; 2],
        radius: f64,
    ) -> PyResult<PyGeometryOperation> {
        self.graph
            .circle(center, radius)
            .map(|operation| PyGeometryOperation { operation })
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }

    fn subtract(
        &self,
        py: Python<'_>,
        rectangle: &PyGeometryOperation,
        circle: &PyGeometryOperation,
    ) -> PyResult<PyGeometryOperation> {
        self.graph
            .subtract(&rectangle.operation, &circle.operation)
            .map(|operation| PyGeometryOperation { operation })
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }

    #[pyo3(signature = (left, right, /, *, interface))]
    fn partition(
        &self,
        py: Python<'_>,
        left: &PyGeometryOperation,
        right: &PyGeometryOperation,
        interface: &Bound<'_, PyTuple>,
    ) -> PyResult<PyGeometryOperation> {
        if interface.len() != 2 {
            return Err(PyTypeError::new_err(
                "interface must contain exactly two boundary handles in parent order",
            ));
        }
        let first = interface
            .get_item(0)?
            .extract::<PyRef<'_, PyGeometryBoundaryHandle>>()?;
        let second = interface
            .get_item(1)?
            .extract::<PyRef<'_, PyGeometryBoundaryHandle>>()?;
        self.graph
            .partition(
                &left.operation,
                &right.operation,
                [first.handle, second.handle],
            )
            .map(|operation| PyGeometryOperation { operation })
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }

    /// Construct the one admitted exact rectangle-extrusion solid operation.
    #[pyo3(signature = (
        *,
        x_bounds,
        y_bounds,
        plane_z,
        depth,
        modeling_tolerance
    ))]
    fn rectangle_extrusion(
        &self,
        py: Python<'_>,
        #[pyo3(from_py_with = extract_rectangle_pair)] x_bounds: (f64, f64),
        #[pyo3(from_py_with = extract_rectangle_pair)] y_bounds: (f64, f64),
        plane_z: f64,
        depth: f64,
        modeling_tolerance: f64,
    ) -> PyResult<PyGeometrySolidOperation> {
        self.graph
            .rectangle_extrusion(x_bounds, y_bounds, plane_z, depth, modeling_tolerance)
            .map(|graph| PyGeometrySolidOperation { graph })
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }

    /// Reconstruct one admitted solid operation and bind it to this graph session.
    fn decode_solid(&self, py: Python<'_>, data: &[u8]) -> PyResult<PyGeometrySolidOperation> {
        self.graph
            .decode_solid(data)
            .map(|graph| PyGeometrySolidOperation { graph })
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }

    /// Append the one admitted circular through-cut to a solid operation.
    #[pyo3(signature = (target, /, *, center, radius, boolean_tolerance))]
    fn circular_through_cut(
        &self,
        py: Python<'_>,
        target: &PyGeometrySolidOperation,
        #[pyo3(from_py_with = extract_sequence_pair)] center: [f64; 2],
        radius: f64,
        boolean_tolerance: f64,
    ) -> PyResult<PyGeometrySolidOperation> {
        self.graph
            .circular_through_cut(&target.graph, center, radius, boolean_tolerance)
            .map(|graph| PyGeometrySolidOperation { graph })
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }

    /// Build either typed operation family through this graph-owned boundary.
    #[pyo3(signature = (operation, /, *, named_topology=None))]
    fn build(
        &self,
        py: Python<'_>,
        operation: &Bound<'_, PyAny>,
        named_topology: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if let Ok(operation) = operation.extract::<PyRef<'_, PyGeometryOperation>>() {
            let named_topology = named_topology.ok_or_else(|| {
                PyTypeError::new_err("GeometryOperation build requires named_topology=")
            })?;
            let named_topology = extract_named_topology(named_topology)?;
            return self
                .graph
                .build(&operation.operation, &named_topology)
                .map(PyGeometry::from_geometry)
                .map_err(|diagnostic| validation_error(py, &[diagnostic]))
                .and_then(|geometry| Py::new(py, geometry).map(Py::into_any));
        }
        if let Ok(operation) = operation.extract::<PyRef<'_, PyGeometrySolidOperation>>() {
            let build = self
                .graph
                .build_solid(&operation.graph)
                .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
            let Some(named_topology) = named_topology else {
                return Py::new(py, PyGeometryBuildReceipt { build }).map(Py::into_any);
            };
            let named_topology = extract_solid_names(named_topology)?;
            return self
                .graph
                .build_solid_geometry(&operation.graph, &named_topology)
                .map(PyGeometry::from_geometry)
                .map_err(|diagnostic| validation_error(py, &[diagnostic]))
                .and_then(|geometry| Py::new(py, geometry).map(Py::into_any));
        }
        Err(PyTypeError::new_err(
            "operation must be a GeometryOperation or GeometrySolidOperation",
        ))
    }
}

#[pymethods]
impl PyGeometryOperation {
    #[getter]
    fn region(&self) -> PyGeometryRegionHandle {
        PyGeometryRegionHandle {
            handle: self.operation.region(),
        }
    }

    #[getter]
    fn boundaries(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let handles = self
            .operation
            .boundaries()
            .into_iter()
            .map(|handle| Py::new(py, PyGeometryBoundaryHandle { handle }))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, handles)?.unbind())
    }
}

#[pymethods]
impl PyGeometryRegionHandle {
    #[getter]
    fn dimension(&self) -> usize {
        self.handle.dimension()
    }
}

#[pymethods]
impl PyGeometryBoundaryHandle {
    #[getter]
    fn dimension(&self) -> usize {
        self.handle.dimension()
    }
}

fn extract_named_topology(
    value: &Bound<'_, PyAny>,
) -> PyResult<BTreeMap<String, Vec<GeometryTopologyHandle>>> {
    let mapping = value.cast::<PyMapping>().map_err(|_| {
        PyTypeError::new_err("named_topology must be one mapping from strings to topology handles")
    })?;
    let mut result = BTreeMap::new();
    for item in mapping.items()?.try_iter()? {
        let pair = item?.cast_into::<PyTuple>()?;
        let name = pair
            .get_item(0)?
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("named_topology keys must be strings"))?;
        let raw = pair.get_item(1)?;
        let handles = if let Some(handle) = extract_handle(&raw) {
            vec![handle]
        } else {
            let sequence = raw.cast::<PySequence>().map_err(|_| {
                PyTypeError::new_err(
                    "named_topology values must be a topology handle or sequence of handles",
                )
            })?;
            sequence
                .try_iter()?
                .map(|member| {
                    let member = member?;
                    extract_handle(&member).ok_or_else(|| {
                        PyTypeError::new_err(
                            "named_topology sequences must contain only topology handles",
                        )
                    })
                })
                .collect::<PyResult<Vec<_>>>()?
        };
        if result.insert(name, handles).is_some() {
            return Err(PyValueError::new_err(
                "named_topology mapping contains a duplicate name",
            ));
        }
    }
    Ok(result)
}

fn extract_handle(value: &Bound<'_, PyAny>) -> Option<GeometryTopologyHandle> {
    if let Ok(handle) = value.extract::<PyRef<'_, PyGeometryRegionHandle>>() {
        Some(handle.handle.into())
    } else if let Ok(handle) = value.extract::<PyRef<'_, PyGeometryBoundaryHandle>>() {
        Some(handle.handle.into())
    } else {
        None
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyGeometryGraph>()?;
    module.add_class::<PyGeometryOperation>()?;
    module.add_class::<PyGeometryRegionHandle>()?;
    module.add_class::<PyGeometryBoundaryHandle>()
}
