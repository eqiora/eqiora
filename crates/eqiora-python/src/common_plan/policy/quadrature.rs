//! Thin Python inputs for the shared realization quadrature policy.
use super::*;
use pyo3::exceptions::PyValueError;

#[pyclass(
    name = "GaussLegendre",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PyGaussLegendre {
    #[pyo3(get)]
    points_per_axis: usize,
}

#[pymethods]
impl PyGaussLegendre {
    #[new]
    #[pyo3(signature = (points_per_axis, /))]
    fn new(points_per_axis: usize) -> PyResult<Self> {
        if points_per_axis == 0 {
            return Err(PyValueError::new_err(
                "Gauss points per axis must be positive",
            ));
        }
        Ok(Self { points_per_axis })
    }
}

#[pyclass(
    name = "SimplexDuffyGaussLegendre",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PySimplexDuffyGaussLegendre {
    #[pyo3(get)]
    spatial_dimension: usize,
    #[pyo3(get)]
    points_per_axis: usize,
}

#[pymethods]
impl PySimplexDuffyGaussLegendre {
    #[new]
    #[pyo3(signature = (spatial_dimension, points_per_axis, /))]
    fn new(spatial_dimension: usize, points_per_axis: usize) -> PyResult<Self> {
        if !(1..=3).contains(&spatial_dimension) || points_per_axis == 0 {
            return Err(PyValueError::new_err(
                "simplex quadrature requires dimension 1..3 and positive points per axis",
            ));
        }
        Ok(Self {
            spatial_dimension,
            points_per_axis,
        })
    }
}

pub(in crate::common_plan) fn request(
    value: &Bound<'_, PyAny>,
) -> PyResult<eqiora::realization::QuadraturePolicy> {
    if let Ok(policy) = value.extract::<PyRef<'_, PyGaussLegendre>>() {
        Ok(eqiora::realization::QuadraturePolicy::GaussLegendre {
            points_per_axis: NonZeroUsize::new(policy.points_per_axis)
                .expect("constructor checked positive points"),
        })
    } else if let Ok(policy) = value.extract::<PyRef<'_, PySimplexDuffyGaussLegendre>>() {
        Ok(
            eqiora::realization::QuadraturePolicy::SimplexDuffyGaussLegendre {
                spatial_dimension: NonZeroUsize::new(policy.spatial_dimension)
                    .expect("constructor checked dimension"),
                points_per_axis: NonZeroUsize::new(policy.points_per_axis)
                    .expect("constructor checked positive points"),
            },
        )
    } else {
        Err(PyTypeError::new_err(
            "quadrature must be fem.GaussLegendre or fem.SimplexDuffyGaussLegendre",
        ))
    }
}
