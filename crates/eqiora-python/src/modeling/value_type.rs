use eqiora::{DimExponents, ScalarDomain, ValueFrame, ValueShape, ValueType};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt, PyTuple};

use super::PyDimension;

fn extent(value: &Bound<'_, PyAny>) -> PyResult<u32> {
    if !value.is_instance_of::<PyInt>() || value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err("type extent must be an integer"));
    }
    value.extract()
}

/// Immutable mathematical type shared with native model admission.
#[pyclass(
    name = "ValueType",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyValueType {
    pub(crate) value: ValueType,
}

impl PyValueType {
    fn scalar(domain: ScalarDomain, dimension: Option<&PyDimension>) -> PyResult<Self> {
        Ok(Self {
            value: ValueType::scalar(
                domain,
                dimension.map_or(DimExponents::DIMENSIONLESS, |dimension| dimension.value),
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    fn finite_scalar(scalar: &Self) -> PyResult<()> {
        if !scalar.value.shape().is_scalar()
            || scalar.value.frame() != ValueFrame::Invariant
            || scalar.value.array_rank() != 0
        {
            return Err(PyValueError::new_err(
                "finite components require an invariant scalar type",
            ));
        }
        Ok(())
    }

    fn spatial(scalar: &Self, extents: Vec<u32>) -> PyResult<Self> {
        if !scalar.value.shape().is_scalar() {
            return Err(PyValueError::new_err(
                "spatial components require a scalar type",
            ));
        }
        let shape =
            ValueShape::new(extents).map_err(|error| PyValueError::new_err(error.to_string()))?;
        let value = ValueType::shaped(
            scalar.value.scalar_domain(),
            scalar.value.dimension(),
            shape,
            ValueFrame::SpatialCartesian,
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { value })
    }
}

#[pymethods]
impl PyValueType {
    /// Emit the bounded canonical language type using the native formatter.
    fn to_eqi(&self) -> PyResult<String> {
        eqiora::language::ValueTypeSyntax::from_checked(&self.value, |_| None, |_| None)
            .map(|syntax| syntax.to_source())
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    /// Present this exact checked type without inferring type from symbol style.
    #[pyo3(signature = (profile="latex"))]
    fn render(&self, py: Python<'_>, profile: &str) -> PyResult<crate::model::PyMathRendering> {
        eqiora::api::MathRendering::value_type(&self.value, crate::model::parse_profile(profile)?)
            .map(|value| crate::model::PyMathRendering { value })
            .map_err(|error| crate::error::diagnostic_error(py, &[error]))
    }

    #[staticmethod]
    fn boolean() -> Self {
        Self {
            value: ValueType::boolean(),
        }
    }

    /// Exact signed 64-bit, dimensionless integer type.
    #[staticmethod]
    fn integer() -> PyResult<Self> {
        Self::scalar(ScalarDomain::Integer, None)
    }

    #[staticmethod]
    #[pyo3(signature = (dimension=None))]
    fn real(dimension: Option<&PyDimension>) -> PyResult<Self> {
        Self::scalar(ScalarDomain::Real, dimension)
    }

    #[staticmethod]
    #[pyo3(signature = (dimension=None))]
    fn complex(dimension: Option<&PyDimension>) -> PyResult<Self> {
        Self::scalar(ScalarDomain::Complex, dimension)
    }

    /// Coordinates retain their exact basis, scalar domain, units, and dual role.
    #[staticmethod]
    #[pyo3(signature = (scalar, space, *, dual=false))]
    fn coordinates(
        scalar: &Self,
        space: &super::nominal::PyFiniteSpace,
        dual: bool,
    ) -> PyResult<Self> {
        Self::finite_scalar(scalar)?;
        let basis = space.value.basis();
        let basis = if dual { basis.dual() } else { basis };
        Ok(Self {
            value: ValueType::coordinates(
                basis,
                scalar.value.scalar_domain(),
                scalar.value.dimension(),
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    /// A finite map has ordered input and output bases, independent of equal extents.
    #[staticmethod]
    #[pyo3(signature = (scalar, source, target, *, source_dual=false, target_dual=false))]
    fn linear_map(
        scalar: &Self,
        source: &super::nominal::PyFiniteSpace,
        target: &super::nominal::PyFiniteSpace,
        source_dual: bool,
        target_dual: bool,
    ) -> PyResult<Self> {
        Self::finite_scalar(scalar)?;
        let source = source.value.basis();
        let target = target.value.basis();
        Ok(Self {
            value: ValueType::linear_map(
                if source_dual { source.dual() } else { source },
                if target_dual { target.dual() } else { target },
                scalar.value.scalar_domain(),
                scalar.value.dimension(),
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[staticmethod]
    fn counts(space: &super::nominal::PyFiniteSpace) -> PyResult<Self> {
        Ok(Self {
            value: space
                .value
                .counts()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        })
    }

    #[staticmethod]
    fn index(set: &super::nominal::PyIndexSet) -> Self {
        Self {
            value: set.value.value_type(),
        }
    }

    #[staticmethod]
    fn vector(scalar: &Self, #[pyo3(from_py_with = extent)] extent: u32) -> PyResult<Self> {
        Self::spatial(scalar, vec![extent])
    }

    #[staticmethod]
    #[pyo3(signature = (scalar, *extents))]
    fn tensor(scalar: &Self, extents: &Bound<'_, PyTuple>) -> PyResult<Self> {
        if extents.len() < 2 {
            return Err(PyValueError::new_err(
                "tensor requires at least two spatial axes",
            ));
        }
        Self::spatial(
            scalar,
            extents
                .iter()
                .map(|value| extent(&value))
                .collect::<PyResult<_>>()?,
        )
    }

    #[staticmethod]
    fn array(element: &Self, #[pyo3(from_py_with = extent)] extent: u32) -> PyResult<Self> {
        let value = element
            .value
            .clone()
            .array(extent)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { value })
    }

    #[getter]
    fn scalar_domain(&self) -> &'static str {
        match self.value.scalar_domain() {
            ScalarDomain::Enum => "enum",
            ScalarDomain::Boolean => "boolean",
            ScalarDomain::Integer => "integer",
            ScalarDomain::Real => "real",
            ScalarDomain::Complex => "complex",
        }
    }

    #[getter]
    fn dimension(&self) -> PyDimension {
        PyDimension {
            value: self.value.dimension(),
        }
    }

    #[getter]
    fn shape(&self) -> Vec<u32> {
        self.value
            .shape()
            .extents()
            .iter()
            .map(|extent| extent.get())
            .collect()
    }

    #[getter]
    fn array_rank(&self) -> usize {
        self.value.array_rank()
    }

    #[getter]
    fn frame(&self) -> &'static str {
        match self.value.frame() {
            ValueFrame::Invariant => "invariant",
            ValueFrame::SpatialCartesian => "spatial_cartesian",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ValueType(domain={:?}, dimension={:?}, shape={:?}, array_rank={}, frame={:?})",
            self.scalar_domain(),
            self.value.dimension().exponents(),
            self.shape(),
            self.array_rank(),
            self.frame()
        )
    }
}
