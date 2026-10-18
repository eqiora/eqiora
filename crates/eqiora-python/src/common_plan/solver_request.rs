//! Python values over the existing exact solver/provider authority.

use eqiora::backends::faer::FaerLinearSolver;
use eqiora::solver::{
    LinearSolver, PreconditionerPolicy, REFERENCE_LINEAR_SOLVER, ReductionPolicy, SolverProvider,
};
use pyo3::prelude::*;
use std::hash::{Hash, Hasher};

macro_rules! solver_enum {
    ($name:ident, $python:literal, $native:ident, $($variant:ident),+ $(,)?) => {
        #[pyclass(name = $python, module = "eqiora._eqiora", frozen, eq, hash, from_py_object)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub(crate) enum $name { $($variant),+ }
        impl From<$name> for $native {
            fn from(value: $name) -> Self { match value { $($name::$variant => Self::$variant),+ } }
        }
        impl From<$native> for $name {
            fn from(value: $native) -> Self { match value { $($native::$variant => Self::$variant),+ } }
        }
    }
}

solver_enum!(
    PyLinearSolver,
    "LinearSolver",
    LinearSolver,
    ConjugateGradient,
    MinimumResidual,
    BiConjugateGradientStabilized,
    SparseLu
);
solver_enum!(
    PyPreconditioner,
    "Preconditioner",
    PreconditionerPolicy,
    Identity,
    Jacobi
);
solver_enum!(
    PyReduction,
    "Reduction",
    ReductionPolicy,
    Reproducible,
    Fast
);

/// An immutable observation of one compiled solver's complete release identity.
#[pyclass(
    name = "SolverProvider",
    module = "eqiora._eqiora",
    frozen,
    eq,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PySolverProvider {
    pub(super) native: SolverProvider,
}

#[pymethods]
impl PySolverProvider {
    #[staticmethod]
    fn reference() -> Self {
        Self {
            native: REFERENCE_LINEAR_SOLVER.provider(),
        }
    }

    #[staticmethod]
    fn faer() -> Self {
        Self {
            native: FaerLinearSolver.provider(),
        }
    }

    #[getter]
    fn id(&self) -> &'static str {
        self.native.id().as_str()
    }

    #[getter]
    fn implementation_version(&self) -> &'static str {
        self.native.implementation_version()
    }

    #[getter]
    fn libraries(&self) -> Vec<(&'static str, &'static str)> {
        self.native
            .libraries()
            .iter()
            .map(|library| (library.name(), library.version()))
            .collect()
    }

    fn __hash__(&self) -> isize {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.native.id().as_str().hash(&mut hasher);
        self.native.implementation_version().hash(&mut hasher);
        self.native.libraries().hash(&mut hasher);
        hasher.finish() as isize
    }

    fn __repr__(&self) -> String {
        format!(
            "SolverProvider(id={:?}, implementation_version={:?}, libraries={:?})",
            self.id(),
            self.implementation_version(),
            self.libraries()
        )
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLinearSolver>()?;
    module.add_class::<PyPreconditioner>()?;
    module.add_class::<PyReduction>()?;
    module.add_class::<PySolverProvider>()
}
