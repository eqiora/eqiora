//! Private immutable native-to-NumPy matrix transfer.

use std::mem;
use std::sync::Mutex;

use numpy::ndarray::Array1;
use numpy::ndarray::Array2;
use numpy::{Element, IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

enum MatrixStorage<T: Element> {
    Native(Vec<T>),
    Materializing,
    Numpy(Py<PyArray2<T>>),
}

enum VectorStorage<T: Element> {
    Native(Vec<T>),
    Materializing,
    Numpy(Py<PyArray1<T>>),
}

/// One native vector whose first NumPy projection becomes its immutable owner.
pub(crate) struct ReadOnlyVector<T: Element> {
    storage: Mutex<VectorStorage<T>>,
}

impl<T: Element> ReadOnlyVector<T> {
    pub(crate) fn new(values: Vec<T>) -> Self {
        Self {
            storage: Mutex::new(VectorStorage::Native(values)),
        }
    }

    pub(crate) fn numpy(&self, py: Python<'_>) -> PyResult<Py<PyArray1<T>>> {
        let values = {
            let mut storage = self
                .storage
                .lock()
                .map_err(|_| PyRuntimeError::new_err("vector storage lock is poisoned"))?;
            match &*storage {
                VectorStorage::Numpy(array) => return Ok(array.clone_ref(py)),
                VectorStorage::Materializing => {
                    return Err(PyRuntimeError::new_err(
                        "vector NumPy materialization is already in progress",
                    ));
                }
                VectorStorage::Native(_) => {}
            }
            let VectorStorage::Native(values) =
                mem::replace(&mut *storage, VectorStorage::Materializing)
            else {
                return Err(PyRuntimeError::new_err(
                    "vector storage changed before NumPy materialization",
                ));
            };
            values
        };

        let vector = Array1::from_vec(values).into_pyarray(py);
        drop(vector.readwrite().make_nonwriteable());
        let owned = vector.unbind();

        let mut storage = self
            .storage
            .lock()
            .map_err(|_| PyRuntimeError::new_err("vector storage lock is poisoned"))?;
        if !matches!(*storage, VectorStorage::Materializing) {
            return Err(PyRuntimeError::new_err(
                "vector storage changed during NumPy materialization",
            ));
        }
        *storage = VectorStorage::Numpy(owned.clone_ref(py));
        Ok(owned)
    }

    pub(crate) fn snapshot(&self, py: Python<'_>) -> PyResult<Vec<T>>
    where
        T: Copy,
    {
        let numpy = {
            let storage = self
                .storage
                .lock()
                .map_err(|_| PyRuntimeError::new_err("vector storage lock is poisoned"))?;
            match &*storage {
                VectorStorage::Native(values) => return Ok(values.clone()),
                VectorStorage::Materializing => {
                    return Err(PyRuntimeError::new_err(
                        "vector NumPy materialization is already in progress",
                    ));
                }
                VectorStorage::Numpy(array) => array.clone_ref(py),
            }
        };
        Ok(numpy.bind(py).readonly().as_slice()?.to_vec())
    }
}

/// One native matrix whose first NumPy projection becomes its immutable owner.
pub(crate) struct ReadOnlyMatrix<T: Element> {
    rows: usize,
    columns: usize,
    storage: Mutex<MatrixStorage<T>>,
}

impl<T: Element> ReadOnlyMatrix<T> {
    pub(crate) fn new(rows: usize, columns: usize, values: Vec<T>) -> Self {
        debug_assert_eq!(values.len(), rows * columns);
        Self {
            rows,
            columns,
            storage: Mutex::new(MatrixStorage::Native(values)),
        }
    }

    pub(crate) fn numpy(&self, py: Python<'_>) -> PyResult<Py<PyArray2<T>>> {
        let values = {
            let mut storage = self
                .storage
                .lock()
                .map_err(|_| PyRuntimeError::new_err("matrix storage lock is poisoned"))?;
            match &*storage {
                MatrixStorage::Numpy(array) => return Ok(array.clone_ref(py)),
                MatrixStorage::Materializing => {
                    return Err(PyRuntimeError::new_err(
                        "matrix NumPy materialization is already in progress",
                    ));
                }
                MatrixStorage::Native(_) => {}
            }
            let MatrixStorage::Native(values) =
                mem::replace(&mut *storage, MatrixStorage::Materializing)
            else {
                return Err(PyRuntimeError::new_err(
                    "matrix storage changed before NumPy materialization",
                ));
            };
            values
        };

        // NumPy import hooks can execute Python. Never hold the storage lock
        // across allocation, and make the transferred native allocation
        // immutable before publishing it.
        let native = Array2::from_shape_vec((self.rows, self.columns), values)
            .expect("validated native matrix shape");
        let matrix = native.into_pyarray(py);
        drop(matrix.readwrite().make_nonwriteable());
        let owned = matrix.unbind();

        let mut storage = self
            .storage
            .lock()
            .map_err(|_| PyRuntimeError::new_err("matrix storage lock is poisoned"))?;
        if !matches!(*storage, MatrixStorage::Materializing) {
            return Err(PyRuntimeError::new_err(
                "matrix storage changed during NumPy materialization",
            ));
        }
        *storage = MatrixStorage::Numpy(owned.clone_ref(py));
        Ok(owned)
    }

    pub(crate) const fn shape(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    pub(crate) fn snapshot(&self, py: Python<'_>) -> PyResult<Vec<T>>
    where
        T: Copy,
    {
        let numpy = {
            let storage = self
                .storage
                .lock()
                .map_err(|_| PyRuntimeError::new_err("matrix storage lock is poisoned"))?;
            match &*storage {
                MatrixStorage::Native(values) => return Ok(values.clone()),
                MatrixStorage::Materializing => {
                    return Err(PyRuntimeError::new_err(
                        "matrix NumPy materialization is already in progress",
                    ));
                }
                MatrixStorage::Numpy(array) => array.clone_ref(py),
            }
        };
        Ok(numpy.bind(py).readonly().as_slice()?.to_vec())
    }
}
