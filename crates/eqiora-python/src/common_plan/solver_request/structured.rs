//! Immutable Python values over complete core preconditioner policies.
use super::*;
use crate::error::validation_error;
use eqiora::solver::{AlgebraicBlock, FixedFactorizationChoice, UpperDiagonalSchurConfig};
use pyo3::exceptions::PyTypeError;

#[pyclass(
    name = "AlgebraicBlock",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyAlgebraicBlock {
    native: AlgebraicBlock,
}
#[pymethods]
impl PyAlgebraicBlock {
    #[staticmethod]
    fn field(field: &crate::model::PyModelFieldRef) -> PyResult<Self> {
        Self::from_field(field, false)
    }
    #[staticmethod]
    fn constraint_multiplier(field: &crate::model::PyModelFieldRef) -> PyResult<Self> {
        Self::from_field(field, true)
    }
    #[getter]
    fn kind(&self) -> &'static str {
        match self.native {
            AlgebraicBlock::Field(_) => "field",
            _ => "constraint-multiplier",
        }
    }
    #[getter]
    fn field_id(&self) -> String {
        match self.native {
            AlgebraicBlock::Field(field) | AlgebraicBlock::ConstraintMultiplier { field } => {
                field.ulid().to_string()
            }
        }
    }
}
impl PyAlgebraicBlock {
    fn from_field(field: &crate::model::PyModelFieldRef, multiplier: bool) -> PyResult<Self> {
        let field = ulid::Ulid::from_string(field.exact_id())
            .map(eqiora::Id::from_ulid)
            .map_err(|_| PyTypeError::new_err("invalid exact FieldRef"))?;
        Ok(Self {
            native: if multiplier {
                AlgebraicBlock::ConstraintMultiplier { field }
            } else {
                AlgebraicBlock::Field(field)
            },
        })
    }
}

#[pyclass(
    name = "FixedFactorization",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyFixedFactorization {
    native: FixedFactorizationChoice,
}
#[pymethods]
impl PyFixedFactorization {
    #[new]
    #[pyo3(signature=(*, provider, algorithm=PyLinearSolver::SparseLu))]
    fn new(
        py: Python<'_>,
        provider: &PySolverProvider,
        algorithm: PyLinearSolver,
    ) -> PyResult<Self> {
        FixedFactorizationChoice::new(algorithm.into(), provider.native.id().as_str())
            .map(|native| Self { native })
            .map_err(|d| validation_error(py, &[d]))
    }
    #[getter]
    fn algorithm(&self) -> PyLinearSolver {
        self.native.algorithm().into()
    }
    #[getter]
    fn provider(&self) -> PyResult<PySolverProvider> {
        [
            REFERENCE_LINEAR_SOLVER.provider(),
            FaerLinearSolver.provider(),
        ]
        .into_iter()
        .find(|p| p.id().as_str() == self.native.provider())
        .map(|native| PySolverProvider { native })
        .ok_or_else(|| {
            PyTypeError::new_err("factorization provider is not in the compiled registry")
        })
    }
}

#[pyclass(
    name = "UpperDiagonalSchurConfig",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyUpperDiagonalSchurConfig {
    native: UpperDiagonalSchurConfig,
}
#[pymethods]
impl PyUpperDiagonalSchurConfig {
    #[getter]
    fn groups(&self) -> (Vec<PyAlgebraicBlock>, Vec<PyAlgebraicBlock>) {
        let g = self.native.groups();
        (
            g[0].iter()
                .map(|native| PyAlgebraicBlock { native: *native })
                .collect(),
            g[1].iter()
                .map(|native| PyAlgebraicBlock { native: *native })
                .collect(),
        )
    }
    #[getter]
    fn a_factorization(&self) -> PyFixedFactorization {
        PyFixedFactorization {
            native: self.native.a_factorization().clone(),
        }
    }
    #[getter]
    fn schur_factorization(&self) -> PyFixedFactorization {
        PyFixedFactorization {
            native: self.native.schur_factorization().clone(),
        }
    }
    /// Maximum retained Schur entries, excluding factor memory and RSS.
    #[getter]
    fn maximum_entries(&self) -> usize {
        self.native.maximum_schur_entries().get()
    }
    /// Maximum scalar product contributions during Schur construction.
    #[getter]
    fn maximum_products(&self) -> usize {
        self.native.maximum_schur_products().get()
    }
}

#[pyclass(
    name = "Preconditioner",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyPreconditioner {
    native: PreconditionerPolicy,
}
impl From<PyPreconditioner> for PreconditionerPolicy {
    fn from(v: PyPreconditioner) -> Self {
        v.native
    }
}
impl From<&PreconditionerPolicy> for PyPreconditioner {
    fn from(v: &PreconditionerPolicy) -> Self {
        Self { native: v.clone() }
    }
}
#[pymethods]
impl PyPreconditioner {
    #[classattr]
    #[allow(non_snake_case)]
    fn Identity() -> Self {
        Self {
            native: PreconditionerPolicy::Identity,
        }
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn Jacobi() -> Self {
        Self {
            native: PreconditionerPolicy::Jacobi,
        }
    }
    #[staticmethod]
    #[pyo3(signature=(*, groups, a_factorization, schur_factorization, maximum_entries, maximum_products))]
    fn upper_diagonal_schur(
        py: Python<'_>,
        groups: (Vec<PyAlgebraicBlock>, Vec<PyAlgebraicBlock>),
        a_factorization: PyFixedFactorization,
        schur_factorization: PyFixedFactorization,
        maximum_entries: usize,
        maximum_products: usize,
    ) -> PyResult<Self> {
        UpperDiagonalSchurConfig::new(
            [
                groups.0.into_iter().map(|v| v.native).collect(),
                groups.1.into_iter().map(|v| v.native).collect(),
            ],
            a_factorization.native,
            schur_factorization.native,
            maximum_entries,
            maximum_products,
        )
        .map(|v| Self {
            native: PreconditionerPolicy::upper_diagonal_schur(v),
        })
        .map_err(|d| validation_error(py, &[d]))
    }
    #[getter]
    fn kind(&self) -> &'static str {
        match self.native {
            PreconditionerPolicy::Identity => "identity",
            PreconditionerPolicy::Jacobi => "jacobi",
            PreconditionerPolicy::UpperDiagonalSchur(_) => "upper-diagonal-schur",
        }
    }
    #[getter]
    fn upper_diagonal_schur_config(&self) -> Option<PyUpperDiagonalSchurConfig> {
        match &self.native {
            PreconditionerPolicy::UpperDiagonalSchur(native) => Some(PyUpperDiagonalSchurConfig {
                native: native.as_ref().clone(),
            }),
            _ => None,
        }
    }
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPreconditioner>()?;
    module.add_class::<PyAlgebraicBlock>()?;
    module.add_class::<PyFixedFactorization>()?;
    module.add_class::<PyUpperDiagonalSchurConfig>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_policy_values_validate_and_preserve_explicit_gauges() -> PyResult<()> {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "structured_test")?;
            register(&module)?;
            module.add_class::<PySolverProvider>()?;
            module.add_class::<PyLinearSolver>()?;
            module.add_class::<crate::common_plan::policy::PyLinear>()?;
            module.add_class::<crate::common_plan::policy::PySolverPlanningObjective>()?;
            module.add_class::<PyReduction>()?;
            let field = crate::model::PyModelFieldRef::from_exact(
                "test-model".into(),
                ulid::Ulid::new().to_string(),
            );
            module.add("field", Py::new(py, field)?)?;
            py.run(
                pyo3::ffi::c_str!(
                    r#"
provider = SolverProvider.reference()
factor = FixedFactorization(provider=provider)
assert factor.algorithm == LinearSolver.SparseLu
assert factor.provider == provider
block = AlgebraicBlock.field(field)
gauge = AlgebraicBlock.constraint_multiplier(field)
assert block.field_id == gauge.field_id == field.id
assert block.kind == "field" and gauge.kind == "constraint-multiplier"
settings = dict(groups=([block], [gauge]), a_factorization=factor,
                schur_factorization=factor, maximum_entries=16, maximum_products=64)
policy = Preconditioner.upper_diagonal_schur(**settings)
assert policy == Preconditioner.upper_diagonal_schur(**settings)
assert hash(policy) == hash(Preconditioner.upper_diagonal_schur(**settings))
assert policy.kind == "upper-diagonal-schur"
controls = dict(relative_tolerance=1e-10, absolute_tolerance=1e-12, maximum_iterations=100)
ranked = Linear(**controls, objective=SolverPlanningObjective.Robust, preconditioner=policy)
exact = Linear(**controls, algorithm=LinearSolver.BiConjugateGradientStabilized,
               reduction=Reduction.Reproducible, provider=provider, preconditioner=policy)
assert ranked.preconditioner == exact.preconditioner == policy
assert ranked.algorithm is None and exact.algorithm == LinearSolver.BiConjugateGradientStabilized
assert Linear(**controls, objective=SolverPlanningObjective.Robust).preconditioner is None

config = policy.upper_diagonal_schur_config
assert config.groups == ([block], [gauge])
assert config.a_factorization == config.schur_factorization == factor
assert config.maximum_entries == 16 and config.maximum_products == 64
copy = config.groups
copy[0].clear()
assert config.groups == ([block], [gauge])
assert Preconditioner.Identity.kind == "identity"
assert Preconditioner.Jacobi.upper_diagonal_schur_config is None
for replacement in ({"groups": ([block], [block])}, {"groups": ([], [gauge])},
                    {"maximum_entries": 0}, {"maximum_products": 0}):
    try:
        Preconditioner.upper_diagonal_schur(**(settings | replacement))
    except Exception:
        pass
    else:
        raise AssertionError("invalid core policy was accepted")
try:
    FixedFactorization(provider=provider, algorithm=LinearSolver.ConjugateGradient)
except Exception:
    pass
else:
    raise AssertionError("iterative factorization was accepted")
try:
    config.maximum_entries = 3
except AttributeError:
    pass
else:
    raise AssertionError("configuration is mutable")
"#
                ),
                Some(&module.dict()),
                None,
            )
        })
    }
}
