//! Typed capability-specific views projected from one resolved root Plan.

use eqiora::realization::{Space, SpaceFamily};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use eqiora_numerics::{CommonFormulationDescription, FormulationKind, FormulationSelectionMode};

use crate::model::PyModelFieldRef;

use super::policy::PyPressureGauge2d;
use super::scaling::{PyIncompressibleScales, PyIncompressibleScalingReceipt2d};

pub(super) fn space_name(space: Space) -> &'static str {
    match space.family() {
        SpaceFamily::SimplexP1Bubble => "simplex-p1-bubble",
        SpaceFamily::ContinuousLagrange { order } if order.get() == 1 => "continuous-lagrange-p1",
        SpaceFamily::CellConstant => "cell-constant",
        SpaceFamily::ContinuousLagrange { .. } => {
            unreachable!("common Plan only publishes the admitted P1 continuous space")
        }
    }
}

/// Closed mathematical Formulation families accepted by exact override.
///
/// Authority: `crates/eqiora-python/src/common_plan/capability_view.rs::PyFormulationKind`.
#[pyclass(
    name = "FormulationKind",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyFormulationKind {
    PrimalGalerkin,
    MixedGalerkin,
    IntegralConservative,
}

impl From<FormulationKind> for PyFormulationKind {
    fn from(value: FormulationKind) -> Self {
        match value {
            FormulationKind::PrimalGalerkin => Self::PrimalGalerkin,
            FormulationKind::MixedGalerkin => Self::MixedGalerkin,
            FormulationKind::IntegralConservative => Self::IntegralConservative,
        }
    }
}

impl From<PyFormulationKind> for FormulationKind {
    fn from(value: PyFormulationKind) -> Self {
        match value {
            PyFormulationKind::PrimalGalerkin => Self::PrimalGalerkin,
            PyFormulationKind::MixedGalerkin => Self::MixedGalerkin,
            PyFormulationKind::IntegralConservative => Self::IntegralConservative,
        }
    }
}

/// Whether resolution selected a Formulation or admitted an exact request.
///
/// Authority: `crates/eqiora-python/src/common_plan/capability_view.rs::PyFormulationSelectionMode`.
#[pyclass(
    name = "FormulationSelectionMode",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyFormulationSelectionMode {
    Automatic,
    Exact,
    Authored,
}

/// Inspectable effective mathematical Formulation selected before Realization.
///
/// Authority: `crates/eqiora-python/src/common_plan/capability_view.rs::PyFormulationView`.
#[pyclass(
    name = "FormulationView",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug)]
pub(crate) struct PyFormulationView {
    requested: PyFormulationSelectionMode,
    effective: PyFormulationKind,
    boundary_treatment: &'static str,
    rule_ids: Vec<&'static str>,
    selection_reason_codes: Vec<&'static str>,
    requested_source_identity: Option<String>,
}

impl PyFormulationView {
    pub(crate) fn from_native(description: CommonFormulationDescription) -> Self {
        let requested = match description.requested() {
            FormulationSelectionMode::Automatic => PyFormulationSelectionMode::Automatic,
            FormulationSelectionMode::Exact => PyFormulationSelectionMode::Exact,
            FormulationSelectionMode::Authored => PyFormulationSelectionMode::Authored,
        };
        let effective = description.effective().into();
        let requested_source_identity = description.requested_source_identity().map(str::to_owned);
        Self {
            requested,
            effective,
            boundary_treatment: description.boundary_treatment(),
            rule_ids: description.rule_ids().to_vec(),
            selection_reason_codes: description.selection_reason_codes().to_vec(),
            requested_source_identity,
        }
    }
}

#[pymethods]
impl PyFormulationView {
    #[getter]
    const fn requested(&self) -> PyFormulationSelectionMode {
        self.requested
    }
    #[getter]
    const fn effective(&self) -> PyFormulationKind {
        self.effective
    }
    #[getter]
    const fn boundary_treatment(&self) -> &'static str {
        self.boundary_treatment
    }
    #[getter]
    fn rule_ids(&self) -> Vec<&'static str> {
        self.rule_ids.clone()
    }
    #[getter]
    fn selection_reason_codes(&self) -> Vec<&'static str> {
        self.selection_reason_codes.clone()
    }
    #[getter]
    fn requested_source_identity(&self) -> Option<String> {
        self.requested_source_identity.clone()
    }
    fn __repr__(&self) -> String {
        format!(
            "FormulationView(requested={:?}, effective={:?})",
            self.requested, self.effective
        )
    }
}

/// Resolved no-Mesh ODE capability.
#[pyclass(
    name = "OdePlanView",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug)]
pub(crate) struct PyOdePlanView {
    pub(super) backend: &'static str,
    pub(super) backend_version: &'static str,
}

#[pymethods]
impl PyOdePlanView {
    #[getter]
    const fn kind(&self) -> &'static str {
        "ode"
    }
    #[getter]
    const fn backend(&self) -> &'static str {
        self.backend
    }
    #[getter]
    const fn backend_version(&self) -> &'static str {
        self.backend_version
    }
    fn __repr__(&self) -> String {
        format!("OdePlanView(backend={:?})", self.backend)
    }
}

/// Scalar-valued Fields resolved from one Model.
#[pyclass(
    name = "ScalarPlanView",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug)]
pub(crate) struct PyScalarPlanView {
    pub(super) fields: Vec<PyModelFieldRef>,
    pub(super) coefficient_sampling: &'static str,
    pub(super) face_coefficient_policy: &'static str,
}

#[pymethods]
impl PyScalarPlanView {
    #[getter]
    const fn kind(&self) -> &'static str {
        "scalar"
    }

    #[getter]
    fn fields(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let fields = self
            .fields
            .iter()
            .cloned()
            .map(|field| Py::new(py, field))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, fields)?.unbind())
    }
    #[getter]
    const fn coefficient_sampling(&self) -> &'static str {
        self.coefficient_sampling
    }
    #[getter]
    const fn face_coefficient_policy(&self) -> &'static str {
        self.face_coefficient_policy
    }
    fn __repr__(&self) -> String {
        format!(
            "ScalarPlanView(fields={:?}, coefficient_sampling={:?}, face_coefficient_policy={:?})",
            self.fields
                .iter()
                .map(PyModelFieldRef::exact_id)
                .collect::<Vec<_>>(),
            self.coefficient_sampling,
            self.face_coefficient_policy,
        )
    }
}

#[cfg(test)]
mod scalar_fields_tests {
    use super::*;

    #[test]
    fn scalar_view_preserves_every_field_in_order_without_singleton_alias() -> PyResult<()> {
        Python::initialize();
        Python::attach(|py| {
            let fields = ["first", "second", "third"]
                .map(|id| PyModelFieldRef::from_exact("model".to_owned(), id.to_owned()));
            let expected = PyTuple::new(
                py,
                fields
                    .iter()
                    .cloned()
                    .map(|field| Py::new(py, field))
                    .collect::<PyResult<Vec<_>>>()?,
            )?;
            let view = Py::new(
                py,
                PyScalarPlanView {
                    fields: fields.to_vec(),
                    coefficient_sampling: "quadrature-point",
                    face_coefficient_policy: "not-applicable",
                },
            )?;
            assert!(view.bind(py).getattr("fields")?.eq(expected)?);
            assert!(!view.bind(py).hasattr("field")?);
            Ok(())
        })
    }
}

/// Resolved linear-elasticity field roles.
#[pyclass(
    name = "ElasticityPlanView",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug)]
pub(crate) struct PyElasticityPlanView {
    pub(super) displacement: PyModelFieldRef,
}

#[pymethods]
impl PyElasticityPlanView {
    #[getter]
    const fn kind(&self) -> &'static str {
        "linear-elasticity"
    }

    #[getter]
    fn displacement(&self) -> PyModelFieldRef {
        self.displacement.clone()
    }
    fn __repr__(&self) -> String {
        format!(
            "ElasticityPlanView(displacement={:?})",
            self.displacement.exact_id()
        )
    }
}

/// Resolved incompressible-flow roles, spaces, gauge, and scales.
#[pyclass(
    name = "IncompressibleFlowPlanView",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug)]
pub(crate) struct PyIncompressibleFlowPlanView {
    pub(super) kind: &'static str,
    pub(super) velocity: PyModelFieldRef,
    pub(super) pressure: PyModelFieldRef,
    pub(super) velocity_space: &'static str,
    pub(super) pressure_space: &'static str,
    pub(super) pressure_gauge: Option<PyPressureGauge2d>,
    pub(super) scaling: Py<PyIncompressibleScales>,
    pub(super) scaling_receipt: Py<PyIncompressibleScalingReceipt2d>,
}

#[pymethods]
impl PyIncompressibleFlowPlanView {
    #[getter]
    const fn kind(&self) -> &'static str {
        self.kind
    }
    #[getter]
    fn velocity(&self) -> PyModelFieldRef {
        self.velocity.clone()
    }
    #[getter]
    fn pressure(&self) -> PyModelFieldRef {
        self.pressure.clone()
    }
    #[getter]
    const fn velocity_space(&self) -> &'static str {
        self.velocity_space
    }
    #[getter]
    const fn pressure_space(&self) -> &'static str {
        self.pressure_space
    }
    #[getter]
    const fn pressure_gauge(&self) -> Option<PyPressureGauge2d> {
        self.pressure_gauge
    }
    #[getter]
    fn scaling(&self, py: Python<'_>) -> Py<PyIncompressibleScales> {
        self.scaling.clone_ref(py)
    }
    #[getter]
    fn scaling_receipt(&self, py: Python<'_>) -> Py<PyIncompressibleScalingReceipt2d> {
        self.scaling_receipt.clone_ref(py)
    }
    fn __repr__(&self) -> String {
        format!("IncompressibleFlowPlanView(kind={:?})", self.kind)
    }
}

/// Resolved scales for fixed-reference FSI.
#[pyclass(
    name = "FixedReferenceFsiPlanView",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
#[derive(Debug)]
pub(crate) struct PyFixedReferenceFsiPlanView {
    pub(super) scaling: Py<PyIncompressibleScales>,
    pub(super) scaling_receipt: Py<PyIncompressibleScalingReceipt2d>,
}

#[pymethods]
impl PyFixedReferenceFsiPlanView {
    #[getter]
    const fn kind(&self) -> &'static str {
        "fixed-reference-fsi"
    }
    #[getter]
    fn scaling(&self, py: Python<'_>) -> Py<PyIncompressibleScales> {
        self.scaling.clone_ref(py)
    }
    #[getter]
    fn scaling_receipt(&self, py: Python<'_>) -> Py<PyIncompressibleScalingReceipt2d> {
        self.scaling_receipt.clone_ref(py)
    }
    fn __repr__(&self) -> &'static str {
        "FixedReferenceFsiPlanView()"
    }
}
