//! One ownership and lazy materialization path for each admitted scalar dtype.
use numpy::ndarray::{Array, Dimension, Ix1, Ix2, IxDyn};
use numpy::{Element, IntoPyArray, PyArray, PyArrayMethods};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use std::{mem, sync::Mutex};

pub(crate) struct OwnedBuffer<T: Element + Copy, D: Dimension = Ix1> {
    shape: D,
    state: Mutex<State<T, D>>,
}

enum State<T: Element, D: Dimension> {
    Native(Vec<T>),
    Materializing,
    Numpy(Py<PyArray<T, D>>),
}

impl<T: Element + Copy> OwnedBuffer<T> {
    pub(crate) fn new(values: Vec<T>) -> Self {
        Self::with_shape(numpy::ndarray::Dim([values.len()]), values)
    }
}

impl<T: Element + Copy> OwnedBuffer<T, Ix2> {
    pub(crate) fn matrix(rows: usize, columns: usize, values: Vec<T>) -> Self {
        Self::with_shape(numpy::ndarray::Dim([rows, columns]), values)
    }

    pub(crate) fn shape(&self) -> (usize, usize) {
        (self.shape[0], self.shape[1])
    }
}

impl<T: Element + Copy> OwnedBuffer<T, IxDyn> {
    pub(crate) fn shaped(shape: Vec<usize>, values: Vec<T>) -> Self {
        Self::with_shape(IxDyn(&shape), values)
    }
}

impl<T: Element + Copy, D: Dimension> OwnedBuffer<T, D> {
    fn with_shape(shape: D, values: Vec<T>) -> Self {
        debug_assert_eq!(shape.size(), values.len());
        Self {
            shape,
            state: Mutex::new(State::Native(values)),
        }
    }

    pub(crate) fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    pub(crate) fn numpy(&self, py: Python<'_>) -> PyResult<Py<PyArray<T, D>>> {
        let values = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| PyRuntimeError::new_err("Array storage lock is poisoned"))?;
            match &*state {
                State::Numpy(array) => return Ok(array.clone_ref(py)),
                State::Materializing => {
                    return Err(PyRuntimeError::new_err(
                        "Array NumPy materialization is already in progress",
                    ));
                }
                State::Native(_) => {}
            }
            let State::Native(values) = mem::replace(&mut *state, State::Materializing) else {
                return Err(PyRuntimeError::new_err(
                    "Array storage changed before NumPy materialization",
                ));
            };
            values
        };
        // Import hooks may re-enter this buffer. Never hold the lock across NumPy initialization.
        let native = Array::from_shape_vec(self.shape.clone(), values)
            .expect("validated native array shape");
        let array = native.into_pyarray(py);
        let owned = array.clone().unbind();
        drop(array.readwrite().make_nonwriteable());
        let mut state = self
            .state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Array storage lock is poisoned"))?;
        if !matches!(*state, State::Materializing) {
            return Err(PyRuntimeError::new_err(
                "Array storage changed during NumPy materialization",
            ));
        }
        *state = State::Numpy(owned.clone_ref(py));
        Ok(owned)
    }

    pub(crate) fn snapshot(&self, py: Python<'_>) -> PyResult<Vec<T>> {
        let array = {
            let state = self
                .state
                .lock()
                .map_err(|_| PyRuntimeError::new_err("Array storage lock is poisoned"))?;
            match &*state {
                State::Native(values) => return Ok(values.clone()),
                State::Materializing => {
                    return Err(PyRuntimeError::new_err(
                        "Array NumPy materialization is already in progress",
                    ));
                }
                State::Numpy(array) => array.clone_ref(py),
            }
        };
        Ok(array.bind(py).readonly().as_slice()?.to_vec())
    }

    pub(crate) fn get(&self, py: Python<'_>, index: usize) -> PyResult<T> {
        let array = {
            let state = self
                .state
                .lock()
                .map_err(|_| PyRuntimeError::new_err("Array storage lock is poisoned"))?;
            match &*state {
                State::Native(values) => return Ok(values[index]),
                State::Materializing => {
                    return Err(PyRuntimeError::new_err(
                        "Array NumPy materialization is already in progress",
                    ));
                }
                State::Numpy(array) => array.clone_ref(py),
            }
        };
        Ok(array.bind(py).readonly().as_slice()?[index])
    }
}
