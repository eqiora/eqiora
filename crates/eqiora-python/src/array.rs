//! Bounded Python data-plane projection for immutable CPU result buffers.

use std::mem;

use numpy::{Complex64, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyBufferError, PyIndexError};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::types::{PyAny, PyComplex, PyDict, PyFloat, PyModule};

mod storage;
use storage::OwnedBuffer;

const DLPACK_CPU_DEVICE_TYPE: i32 = 1;
const DLPACK_CPU_DEVICE_ID: i32 = 0;
const DLPACK_MAJOR_VERSION: u32 = 1;
const DLPACK_MINOR_VERSION: u32 = 0;

/// One immutable, dense, one-dimensional CPU `float64` or `complex128` buffer.
///
/// The native result allocation is transferred into an opaque Python owner.
/// NumPy can therefore borrow it without copying while retaining its lifetime
/// independently of the originating Result. DLPack exports use a snapshot:
/// its read-only bit is advisory for consumers and cannot protect canonical
/// Result evidence by itself.
#[pyclass(name = "Array", module = "eqiora._eqiora", frozen)]
pub(crate) struct PyArrayBuffer {
    len: usize,
    storage: ArrayStorage,
}

enum ArrayStorage {
    Real(OwnedBuffer<f64>),
    Complex(OwnedBuffer<Complex64>),
}

impl PyArrayBuffer {
    pub(crate) fn from_owned_result(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                len: values.len(),
                storage: ArrayStorage::Real(OwnedBuffer::new(values)),
            },
        )
    }

    pub(crate) fn from_owned_complex_result(
        py: Python<'_>,
        values: Vec<Complex64>,
    ) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                len: values.len(),
                storage: ArrayStorage::Complex(OwnedBuffer::new(values)),
            },
        )
    }

    pub(crate) fn numpy_array(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.storage {
            ArrayStorage::Real(values) => Ok(values.numpy(py)?.into_any()),
            ArrayStorage::Complex(values) => Ok(values.numpy(py)?.into_any()),
        }
    }

    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    /// Real-only staging and visualization must never discard an imaginary component.
    pub(crate) fn snapshot(&self, py: Python<'_>) -> PyResult<Vec<f64>> {
        match &self.storage {
            ArrayStorage::Real(values) => values.snapshot(py),
            ArrayStorage::Complex(_) => Err(PyBufferError::new_err(
                "this operation requires a float64 Array; complex128 is not implicitly projected to real",
            )),
        }
    }
}

/// Admit one exact CPU/native-`f64` rank-one input and copy it into
/// Eqiora-owned staging before native execution detaches from Python.
pub(crate) fn stage_f64_input(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    expected_len: usize,
    label: &str,
) -> PyResult<Vec<f64>> {
    stage_f64_shaped_input(py, value, &[expected_len], label)
}

/// Stage an exact dense shape without changing its coordinate interpretation.
pub(crate) fn stage_f64_shaped_input(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    expected_shape: &[usize],
    label: &str,
) -> PyResult<Vec<f64>> {
    stage_tensor(py, value, Some(expected_shape), isize::MAX as usize, label)
        .map(|(values, _)| values)
}

/// Capture actual shape and owned values once, including DLPack consumption.
pub(crate) fn stage_f64_tensor_input(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    bytes_limit: usize,
    label: &str,
) -> PyResult<(Vec<f64>, Vec<usize>)> {
    stage_tensor(py, value, None, bytes_limit, label)
}

fn stage_tensor(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    expected_shape: Option<&[usize]>,
    bytes_limit: usize,
    label: &str,
) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let (values, shape) = if let Ok(buffer) = value.extract::<PyRef<'_, PyArrayBuffer>>() {
        let shape = vec![buffer.len()];
        check_shape(&shape, expected_shape, bytes_limit, label)?;
        (buffer.snapshot(py)?, shape)
    } else if let Ok(array) = value.cast::<PyArrayDyn<f64>>() {
        copy_exact_f64_array(array, expected_shape, bytes_limit, label)?
    } else {
        stage_f64_dlpack_input(py, value, expected_shape, bytes_limit, label)?
    };
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        let mut remainder = index;
        let mut coordinates = vec![0; shape.len()];
        for (coordinate, &extent) in coordinates.iter_mut().zip(&shape).rev() {
            *coordinate = remainder % extent;
            remainder /= extent;
        }
        let component = coordinates.pop().unwrap_or(0);
        let axis = if label == "cotangent" || label.starts_with("output") {
            "output"
        } else {
            "input"
        };
        return Err(PyBufferError::new_err(format!(
            "{label} must contain finite float64 values (grid coordinates {coordinates:?}, {axis} coordinate {component})",
        )));
    }
    Ok((values, shape))
}

fn stage_f64_dlpack_input(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    expected_shape: Option<&[usize]>,
    bytes_limit: usize,
    label: &str,
) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let has_capsule = value.hasattr("__dlpack__")?;
    let has_device = value.hasattr("__dlpack_device__")?;
    if !has_capsule || !has_device {
        return Err(PyBufferError::new_err(format!(
            "{label} must be an Eqiora Array, an exact NumPy float64 array, or a complete DLPack producer",
        )));
    }

    let reported_device = value.call_method0("__dlpack_device__")?;
    let device = reported_device.extract::<(i32, i32)>().map_err(|_| {
        PyBufferError::new_err(format!(
            "{label} DLPack producer must report one integer device pair",
        ))
    })?;
    if device != (DLPACK_CPU_DEVICE_TYPE, DLPACK_CPU_DEVICE_ID) {
        return Err(PyBufferError::new_err(format!(
            "{label} DLPack input must already reside on CPU device 0",
        )));
    }

    // NumPy owns capsule version negotiation, exactly-once consumption, and
    // the producer deleter. copy=False forbids a hidden producer transfer;
    // Eqiora performs the one documented staging copy only after validation.
    let kwargs = PyDict::new(py);
    kwargs.set_item("device", "cpu")?;
    kwargs.set_item("copy", false)?;
    let imported = py
        .import("numpy")?
        .getattr("from_dlpack")?
        .call((value,), Some(&kwargs))?;
    let array = imported.cast::<PyArrayDyn<f64>>().map_err(|_| {
        PyBufferError::new_err(format!(
            "{label} DLPack input must be an exact float64 array",
        ))
    })?;
    copy_exact_f64_array(array, expected_shape, bytes_limit, label)
}

fn copy_exact_f64_array(
    array: &Bound<'_, PyArrayDyn<f64>>,
    expected_shape: Option<&[usize]>,
    bytes_limit: usize,
    label: &str,
) -> PyResult<(Vec<f64>, Vec<usize>)> {
    if !array.is_c_contiguous()
        || !array.is_aligned()
        || array.dtype().is_native_byteorder() != Some(true)
    {
        return Err(PyBufferError::new_err(format!(
            "{label} must be C-contiguous, aligned, and native-endian",
        )));
    }
    check_shape(array.shape(), expected_shape, bytes_limit, label)?;
    let readonly = array.readonly();
    let borrowed = readonly
        .as_slice()
        .map_err(|_| PyBufferError::new_err(format!("{label} cannot be borrowed safely")))?;
    Ok((borrowed.to_vec(), array.shape().to_vec()))
}

fn check_shape(
    shape: &[usize],
    expected: Option<&[usize]>,
    bytes_limit: usize,
    label: &str,
) -> PyResult<()> {
    if shape.len() > 33 {
        return Err(PyBufferError::new_err(format!("{label} rank exceeds 33")));
    }
    if expected.is_some_and(|expected| shape != expected) {
        return Err(PyBufferError::new_err(format!(
            "{label} requires shape {:?}, received {shape:?}",
            expected.unwrap(),
        )));
    }
    let nonzero = shape
        .iter()
        .try_fold(size_of::<f64>(), |n, &extent| {
            n.checked_mul(extent.max(1))
                .filter(|&n| n <= isize::MAX as usize)
        })
        .ok_or_else(|| {
            PyBufferError::new_err(format!("{label} shape overflows addressable storage"))
        })?;
    let bytes = if shape.contains(&0) { 0 } else { nonzero };
    if bytes > bytes_limit {
        return Err(PyBufferError::new_err(format!(
            "{label} exceeds input byte limit {bytes_limit}"
        )));
    }
    Ok(())
}

#[pymethods]
impl PyArrayBuffer {
    /// Project this immutable host buffer into NumPy.
    ///
    /// `False` and `None` return the exact read-only zero-copy array. `True`
    /// creates an independent writable NumPy copy.
    /// The NumPy array protocol, so `numpy.asarray(array)` needs no method call.
    ///
    /// `copy` carries the same meaning as [`Self::numpy`], which is already
    /// NumPy's: `None` and `False` never copy, `True` always does. A requested
    /// `dtype` is honoured through `astype` rather than ignored, so a caller
    /// asking for `float32` is not silently handed `float64`.
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__(
        &self,
        py: Python<'_>,
        dtype: Option<Bound<'_, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let array = self.numpy(py, copy)?.into_bound(py).into_any();
        match dtype {
            None => Ok(array.unbind()),
            Some(dtype) => Ok(array
                .call_method(
                    "astype",
                    (dtype,),
                    Some(&[("copy", false)].into_py_dict(py)?),
                )?
                .unbind()),
        }
    }

    #[pyo3(signature = (*, copy=None))]
    fn numpy(&self, py: Python<'_>, copy: Option<bool>) -> PyResult<Py<PyAny>> {
        let array = self.numpy_array(py)?;
        if copy == Some(true) {
            Ok(array.bind(py).call_method0("copy")?.unbind())
        } else {
            Ok(array)
        }
    }

    /// DLPack device pair `(kDLCPU, 0)`.
    fn __dlpack_device__(&self) -> (i32, i32) {
        (DLPACK_CPU_DEVICE_TYPE, DLPACK_CPU_DEVICE_ID)
    }

    /// Export one fresh, versioned DLPack snapshot.
    ///
    /// DLPack read-only flags are advisory, so the immutable Result allocation
    /// is never shared through this protocol. `copy=False`, legacy capsules,
    /// streams, and non-CPU device requests fail rather than weakening that
    /// boundary. NumPy owns capsule consumption and its exactly-once deleter.
    #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(
        &self,
        py: Python<'_>,
        stream: Option<Py<PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        if stream.is_some() {
            return Err(PyBufferError::new_err(
                "CPU Array DLPack export accepts stream=None only",
            ));
        }
        let Some((major, _minor)) = max_version else {
            return Err(PyBufferError::new_err(
                "Array exports versioned DLPack 1.x capsules only",
            ));
        };
        if major != DLPACK_MAJOR_VERSION {
            return Err(PyBufferError::new_err(format!(
                "Array supports DLPack major version {DLPACK_MAJOR_VERSION}, not {major}",
            )));
        }
        if let Some(device) = dl_device
            && device != (DLPACK_CPU_DEVICE_TYPE, DLPACK_CPU_DEVICE_ID)
        {
            return Err(PyBufferError::new_err(
                "Array DLPack export cannot transfer away from CPU device 0",
            ));
        }
        if copy == Some(false) {
            return Err(PyBufferError::new_err(
                "immutable Result evidence requires a DLPack snapshot; copy=False is unsupported",
            ));
        }

        let kwargs = PyDict::new(py);
        kwargs.set_item("max_version", (DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION))?;
        kwargs.set_item("copy", true)?;
        Ok(self
            .numpy_array(py)?
            .bind(py)
            .call_method("__dlpack__", (), Some(&kwargs))?
            .unbind())
    }

    /// Device on which this bounded array resides.
    #[getter]
    fn device(&self) -> &'static str {
        "cpu"
    }

    /// Exact CPU device ordinal.
    #[getter]
    const fn device_id(&self) -> u32 {
        DLPACK_CPU_DEVICE_ID as u32
    }

    /// Exact scalar dtype.
    #[getter]
    fn dtype(&self) -> &'static str {
        match self.storage {
            ArrayStorage::Real(_) => "float64",
            ArrayStorage::Complex(_) => "complex128",
        }
    }

    /// Native byte order of the stored scalar components.
    #[getter]
    const fn byte_order(&self) -> &'static str {
        if cfg!(target_endian = "little") {
            "little"
        } else {
            "big"
        }
    }

    /// Logical array shape.
    #[getter]
    const fn shape(&self) -> (usize,) {
        (self.len,)
    }

    /// Byte strides in logical-axis order.
    #[getter]
    const fn strides(&self) -> (isize,) {
        (match self.storage {
            ArrayStorage::Real(_) => mem::size_of::<f64>(),
            ArrayStorage::Complex(_) => mem::size_of::<Complex64>(),
        } as isize,)
    }

    /// This producer slice admits C-contiguous storage only.
    #[getter]
    const fn c_contiguous(&self) -> bool {
        true
    }

    /// The native allocation satisfies its exact scalar dtype alignment.
    #[getter]
    const fn aligned(&self) -> bool {
        true
    }

    /// Canonical Result storage is immutable.
    #[getter]
    const fn readonly(&self) -> bool {
        true
    }

    /// The Array owns its lifetime anchor rather than borrowing a Result.
    #[getter]
    fn ownership(&self) -> &'static str {
        "owned"
    }

    /// Whether construction from the native Result copied its values.
    #[getter]
    const fn origin_copy_occurred(&self) -> bool {
        false
    }

    const fn __len__(&self) -> usize {
        self.len()
    }

    /// Read one scalar without forcing NumPy materialization.
    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<Py<PyAny>> {
        let index = if index < 0 {
            self.len.checked_sub(index.unsigned_abs())
        } else {
            usize::try_from(index)
                .ok()
                .filter(|&index| index < self.len)
        }
        .ok_or_else(|| PyIndexError::new_err("Array index out of range"))?;

        match &self.storage {
            ArrayStorage::Real(values) => {
                Ok(PyFloat::new(py, values.get(py, index)?).into_any().unbind())
            }
            ArrayStorage::Complex(values) => {
                let value = values.get(py, index)?;
                Ok(PyComplex::from_doubles(py, value.re, value.im)
                    .into_any()
                    .unbind())
            }
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Array(shape=({},), dtype='{}', device='cpu:0', readonly=True)",
            self.len,
            self.dtype()
        )
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyArrayBuffer>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_constants_match_the_native_allocation_contract() {
        assert_eq!(mem::size_of::<f64>(), 8);
        assert_eq!(mem::align_of::<f64>(), 8);
        assert_eq!(mem::size_of::<Complex64>(), 16);
        assert_eq!(DLPACK_CPU_DEVICE_TYPE, 1);
        assert_eq!(DLPACK_CPU_DEVICE_ID, 0);
    }

    #[test]
    fn real_input_staging_rejects_complex_arrays_without_projection() -> PyResult<()> {
        Python::initialize();
        Python::attach(|py| {
            for value in [Complex64::new(2., 0.), Complex64::new(2., 3.)] {
                let array = PyArrayBuffer::from_owned_complex_result(py, vec![value])?;
                let error = stage_f64_input(py, array.bind(py).as_any(), 1, "input")
                    .expect_err("complex dtype cannot enter real-only staging");
                assert!(error.is_instance_of::<PyBufferError>(py));
                assert!(error.to_string().contains("not implicitly projected"));
            }
            Ok(())
        })
    }
}
