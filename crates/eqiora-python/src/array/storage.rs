//! One ownership and lazy materialization path for each admitted scalar dtype.
use numpy::{Element, IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use std::{mem, sync::Mutex};

pub(super) struct OwnedBuffer<T: Element + Copy> {
    state: Mutex<State<T>>,
}

enum State<T: Element> {
    Native(Vec<T>),
    Materializing,
    Numpy(Py<PyArray1<T>>),
}

impl<T: Element + Copy> OwnedBuffer<T> {
    pub(super) fn new(values: Vec<T>) -> Self {
        Self {
            state: Mutex::new(State::Native(values)),
        }
    }

    pub(super) fn numpy(&self, py: Python<'_>) -> PyResult<Py<PyArray1<T>>> {
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
        let array = values.into_pyarray(py);
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

    pub(super) fn snapshot(&self, py: Python<'_>) -> PyResult<Vec<T>> {
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

    pub(super) fn get(&self, py: Python<'_>, index: usize) -> PyResult<T> {
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
