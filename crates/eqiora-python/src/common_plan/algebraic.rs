//! Thin finite algebraic policy projection.
use super::*;
mod scaling;

#[pyclass(name = "AlgebraicPlanView", module = "eqiora._eqiora", frozen)]
pub(super) struct PyAlgebraicPlanView {
    #[pyo3(get)]
    pub(super) unknown_count: usize,
    kind: &'static str,
}

#[pymethods]
impl PyAlgebraicPlanView {
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind
    }
}

pub(super) fn resolve(
    py: Python<'_>,
    model: Py<PyModel>,
    solve: Option<&Bound<'_, PyAny>>,
    formulation: Option<&Bound<'_, PyAny>>,
    scaling: Option<&Bound<'_, PyAny>>,
    enforcement: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyPlan> {
    if formulation.is_some_and(|v| !v.is_none()) {
        return Err(PyTypeError::new_err(
            "finite algebraic resolve accepts Model and solve controls",
        ));
    }
    let solve = solve
        .ok_or_else(|| PyTypeError::new_err("finite resolve requires Linear or Newton controls"))?;
    let request = if let Ok(linear) = solve.extract::<Py<PyLinear>>() {
        CommonSolvePolicy::Linear(linear.borrow(py).native)
    } else if let Ok(newton) = solve.extract::<Py<PyNewton>>() {
        let newton = newton.borrow(py);
        CommonSolvePolicy::Newton {
            linear: newton.linear.borrow(py).native,
            nonlinear: newton.native,
        }
    } else {
        return Err(PyTypeError::new_err(
            "finite solve requires Linear or Newton",
        ));
    };
    let enforcement = enforcement.map(super::enforcement::extract).transpose()?;
    if let Some((model_digest, _)) = &enforcement {
        let reference = model
            .borrow(py)
            .artifact()
            .artifact_reference()
            .map_err(|error| validation_error(py, &[error]))?;
        if *model_digest != reference.artifact().to_string() {
            return Err(PyTypeError::new_err(
                "finite enforcement belongs to another exact Model",
            ));
        }
    }
    let scales = scaling::extract(py, &model.borrow(py), scaling)?;
    let native = eqiora_numerics::CommonAlgebraicPlan::resolve(
        model.borrow(py).artifact(),
        request,
        enforcement.map(|(_, native)| native),
        &scales,
        model
            .borrow(py)
            .authored_formulation_projection()
            .map_err(|d| validation_error(py, &[d]))?,
        &FaerLinearSolver,
    )
    .map_err(|d| validation_error(py, &[d]))?;
    PyPlan::from_native_artifact(py, ResolvedCommonPlan::Algebraic(Box::new(native)))
}

pub(super) fn view(
    py: Python<'_>,
    plan: &eqiora_numerics::CommonAlgebraicPlan,
) -> PyResult<Py<PyAny>> {
    Py::new(
        py,
        PyAlgebraicPlanView {
            unknown_count: plan.coordinate_count(),
            kind: if plan.nonlinear().is_some() {
                "finite-nonlinear"
            } else {
                "finite-affine"
            },
        },
    )
    .map(Py::into_any)
}
