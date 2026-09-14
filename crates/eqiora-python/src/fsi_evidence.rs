//! Observation-only evidence projected from accepted common FSI states.

use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

use crate::common_plan::PyPlan;
use crate::matrix::{ReadOnlyMatrix, ReadOnlyVector};
use crate::realization::PyLinearSolveSummary;
use crate::result::PyRunResult;
use crate::trajectory::{PyState, PyTrajectory};

#[pyclass(name = "FsiDomainEvidence", module = "eqiora._eqiora", frozen)]
pub(crate) struct PyFsiDomainEvidence {
    identity: String,
    cells: ReadOnlyVector<u32>,
}

#[pymethods]
impl PyFsiDomainEvidence {
    #[getter]
    fn identity(&self) -> &str {
        &self.identity
    }
    #[getter]
    fn cells(&self, py: Python<'_>) -> PyResult<Py<PyArray1<u32>>> {
        self.cells.numpy(py)
    }
}

#[pyclass(name = "FsiConnectionEvidence", module = "eqiora._eqiora", frozen)]
pub(crate) struct PyFsiConnectionEvidence {
    identity: String,
    endpoint_domains: [String; 2],
    endpoint_fields: [String; 2],
    facets: ReadOnlyMatrix<u32>,
}

#[pymethods]
impl PyFsiConnectionEvidence {
    #[getter]
    fn identity(&self) -> &str {
        &self.identity
    }
    #[getter]
    fn endpoint_domains(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, &self.endpoint_domains)?.unbind())
    }
    #[getter]
    fn endpoint_fields(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, &self.endpoint_fields)?.unbind())
    }
    #[getter]
    fn facets(&self, py: Python<'_>) -> PyResult<Py<PyArray2<u32>>> {
        self.facets.numpy(py)
    }
}

/// One exact recovered action. Endpoint action row order matches the endpoint identities.
#[pyclass(name = "FsiInterfaceActionEvidence", module = "eqiora._eqiora", frozen)]
pub(crate) struct PyFsiInterfaceActionEvidence {
    connection: String,
    entity_dimension: usize,
    entity_index: usize,
    slot: usize,
    endpoint_domains: [String; 2],
    endpoint_fields: [String; 2],
    endpoint_actions: ReadOnlyMatrix<f64>,
    imbalance: ReadOnlyVector<f64>,
}

#[pymethods]
impl PyFsiInterfaceActionEvidence {
    #[getter]
    fn connection(&self) -> &str {
        &self.connection
    }
    #[getter]
    const fn entity_dimension(&self) -> usize {
        self.entity_dimension
    }
    #[getter]
    const fn entity_index(&self) -> usize {
        self.entity_index
    }
    #[getter]
    const fn slot(&self) -> usize {
        self.slot
    }
    #[getter]
    fn endpoint_domains(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, &self.endpoint_domains)?.unbind())
    }
    #[getter]
    fn endpoint_fields(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, &self.endpoint_fields)?.unbind())
    }
    #[getter]
    fn endpoint_actions(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        self.endpoint_actions.numpy(py)
    }
    #[getter]
    fn imbalance(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        self.imbalance.numpy(py)
    }
}

#[pyclass(
    name = "FsiStateEvidence",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
pub(crate) struct PyFsiStateEvidence {
    state_digest: String,
    interface_actions: Vec<Py<PyFsiInterfaceActionEvidence>>,
    previous_kinetic_energy_j_per_m: f64,
    next_kinetic_energy_j_per_m: f64,
    previous_elastic_energy_j_per_m: f64,
    next_elastic_energy_j_per_m: f64,
    kinetic_increment_j_per_m: f64,
    elastic_increment_j_per_m: f64,
    viscous_dissipation_j_per_m: f64,
    energy_defect_j_per_m: f64,
    numerical_residual_norm: f64,
    continuity_residual_norm: f64,
    kinematic_residual_norm: f64,
    interface_velocity_jump_norm: f64,
    interface_action_imbalance_n_per_m: f64,
    solve: Py<PyLinearSolveSummary>,
    assembly_packets: usize,
    assembly_targets: usize,
}

#[pymethods]
impl PyFsiStateEvidence {
    #[getter]
    fn state_digest(&self) -> &str {
        &self.state_digest
    }
    #[getter]
    fn interface_actions(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(
            py,
            self.interface_actions
                .iter()
                .map(|action| action.clone_ref(py)),
        )?
        .unbind())
    }
    #[getter]
    const fn previous_kinetic_energy_j_per_m(&self) -> f64 {
        self.previous_kinetic_energy_j_per_m
    }
    #[getter]
    const fn next_kinetic_energy_j_per_m(&self) -> f64 {
        self.next_kinetic_energy_j_per_m
    }
    #[getter]
    const fn previous_elastic_energy_j_per_m(&self) -> f64 {
        self.previous_elastic_energy_j_per_m
    }
    #[getter]
    const fn next_elastic_energy_j_per_m(&self) -> f64 {
        self.next_elastic_energy_j_per_m
    }
    #[getter]
    const fn kinetic_increment_j_per_m(&self) -> f64 {
        self.kinetic_increment_j_per_m
    }
    #[getter]
    const fn elastic_increment_j_per_m(&self) -> f64 {
        self.elastic_increment_j_per_m
    }
    #[getter]
    const fn viscous_dissipation_j_per_m(&self) -> f64 {
        self.viscous_dissipation_j_per_m
    }
    #[getter]
    const fn energy_defect_j_per_m(&self) -> f64 {
        self.energy_defect_j_per_m
    }
    #[getter]
    const fn numerical_residual_norm(&self) -> f64 {
        self.numerical_residual_norm
    }
    #[getter]
    const fn continuity_residual_norm(&self) -> f64 {
        self.continuity_residual_norm
    }
    #[getter]
    const fn kinematic_residual_norm(&self) -> f64 {
        self.kinematic_residual_norm
    }
    #[getter]
    const fn interface_velocity_jump_norm(&self) -> f64 {
        self.interface_velocity_jump_norm
    }
    #[getter]
    const fn interface_action_imbalance_n_per_m(&self) -> f64 {
        self.interface_action_imbalance_n_per_m
    }
    #[getter]
    fn solve(&self, py: Python<'_>) -> Py<PyLinearSolveSummary> {
        self.solve.clone_ref(py)
    }
    #[getter]
    const fn assembly_packets(&self) -> usize {
        self.assembly_packets
    }
    #[getter]
    const fn assembly_targets(&self) -> usize {
        self.assembly_targets
    }
}

#[pyclass(
    name = "FsiEvidence",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
pub(crate) struct PyFsiEvidence {
    model_digest: String,
    request_identity: String,
    domains: Vec<Py<PyFsiDomainEvidence>>,
    connections: Vec<Py<PyFsiConnectionEvidence>>,
    state_owners: Vec<Py<PyState>>,
    states: Vec<Py<PyFsiStateEvidence>>,
}

#[pymethods]
impl PyFsiEvidence {
    #[getter]
    fn request_identity(&self) -> &str {
        &self.request_identity
    }
    #[getter]
    fn domains(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.domains.iter().map(|value| value.clone_ref(py)))?.unbind())
    }
    #[getter]
    fn connections(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.connections.iter().map(|value| value.clone_ref(py)))?.unbind())
    }
    #[getter]
    fn states(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.states.iter().map(|state| state.clone_ref(py)))?.unbind())
    }
    fn state(
        &self,
        py: Python<'_>,
        state: &Bound<'_, PyState>,
    ) -> PyResult<Py<PyFsiStateEvidence>> {
        if state.borrow().model_digest_value() != self.model_digest {
            return Err(PyValueError::new_err(
                "State belongs to a different exact Model artifact",
            ));
        }
        let position = self
            .state_owners
            .iter()
            .position(|owner| owner.bind(py).is(state))
            .ok_or_else(|| {
                PyValueError::new_err("State belongs to a different Result occurrence")
            })?;
        Ok(self.states[position].clone_ref(py))
    }
}

impl PyFsiEvidence {
    pub(crate) fn from_common(
        py: Python<'_>,
        plan: &PyPlan,
        trajectory: &PyTrajectory,
        request_identity: &str,
        result: &eqiora_numerics::CommonResult,
    ) -> PyResult<Self> {
        let native_plan = plan
            .fsi_native()
            .ok_or_else(|| PyValueError::new_err("FSI evidence requires an FSI Plan"))?;
        let owners = trajectory.state_handles(py);
        if owners.len() != result.fsi_state_count() {
            return Err(PyValueError::new_err(
                "FSI Result evidence disagrees with its Trajectory State count",
            ));
        }
        let mut states = Vec::with_capacity(owners.len());
        for (index, owner) in owners.iter().enumerate() {
            let state = owner.borrow(py);
            if Some(state.digest_value()) != result.fsi_state_identity(index) {
                return Err(PyValueError::new_err(
                    "FSI Result evidence crossed a different output State",
                ));
            }
            let action_count = result.fsi_interface_action_count(index);
            let mut interface_actions = Vec::with_capacity(action_count);
            for action in 0..action_count {
                let action = *result.fsi_interface_action(index, action).ok_or_else(|| {
                    PyValueError::new_err("FSI Result omitted an interface action")
                })?;
                let endpoints = action.endpoints();
                interface_actions.push(Py::new(
                    py,
                    PyFsiInterfaceActionEvidence {
                        connection: action.connection().ulid().to_string(),
                        entity_dimension: action.entity().dimension(),
                        entity_index: action.entity().index(),
                        slot: action.slot(),
                        endpoint_domains: endpoints
                            .each_ref()
                            .map(|value| value.0.ulid().to_string()),
                        endpoint_fields: endpoints
                            .each_ref()
                            .map(|value| value.1.ulid().to_string()),
                        endpoint_actions: ReadOnlyMatrix::new(
                            2,
                            2,
                            endpoints.into_iter().flat_map(|value| value.2).collect(),
                        ),
                        imbalance: ReadOnlyVector::new(action.imbalance().to_vec()),
                    },
                )?);
            }
            let metrics = result
                .fsi_state_metrics(index)
                .ok_or_else(|| PyValueError::new_err("FSI Result omitted State metrics"))?;
            let (assembly_packets, assembly_targets) = result
                .fsi_state_assembly_counts(index)
                .ok_or_else(|| PyValueError::new_err("FSI Result omitted assembly evidence"))?;
            let solve = PyLinearSolveSummary::from_common_result(result, Some(index))
                .ok_or_else(|| PyValueError::new_err("FSI Result omitted solve evidence"))?;
            states.push(Py::new(
                py,
                PyFsiStateEvidence {
                    state_digest: state.digest_value().to_owned(),
                    interface_actions,
                    previous_kinetic_energy_j_per_m: metrics[0],
                    next_kinetic_energy_j_per_m: metrics[1],
                    previous_elastic_energy_j_per_m: metrics[2],
                    next_elastic_energy_j_per_m: metrics[3],
                    kinetic_increment_j_per_m: metrics[4],
                    elastic_increment_j_per_m: metrics[5],
                    viscous_dissipation_j_per_m: metrics[6],
                    energy_defect_j_per_m: metrics[7],
                    numerical_residual_norm: metrics[8],
                    continuity_residual_norm: metrics[9],
                    kinematic_residual_norm: metrics[10],
                    interface_velocity_jump_norm: metrics[11],
                    interface_action_imbalance_n_per_m: metrics[12],
                    solve: Py::new(py, solve)?,
                    assembly_packets,
                    assembly_targets,
                },
            )?);
        }
        let convert = |values: Vec<usize>, noun: &str| {
            values
                .into_iter()
                .map(|value| {
                    u32::try_from(value).map_err(|_| {
                        PyOverflowError::new_err(format!("FSI {noun} index exceeds uint32"))
                    })
                })
                .collect::<PyResult<Vec<_>>>()
        };
        let domains = native_plan
            .domain_cell_inventories()
            .map(|inventory| {
                Py::new(
                    py,
                    PyFsiDomainEvidence {
                        identity: inventory.domain().ulid().to_string(),
                        cells: ReadOnlyVector::new(convert(inventory.cells().to_vec(), "cell")?),
                    },
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        let connections = native_plan
            .connection_inventories()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .into_iter()
            .map(|inventory| {
                let quotient = inventory.quotient();
                let endpoints = quotient.endpoints();
                let facets = inventory.facets();
                let facet_count = facets.len();
                Py::new(
                    py,
                    PyFsiConnectionEvidence {
                        identity: quotient.connection().ulid().to_string(),
                        endpoint_domains: endpoints.map(|value| value.domain().ulid().to_string()),
                        endpoint_fields: endpoints.map(|value| value.field().ulid().to_string()),
                        facets: ReadOnlyMatrix::new(
                            facet_count,
                            2,
                            convert(facets.iter().flatten().copied().collect(), "facet vertex")?,
                        ),
                    },
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            model_digest: native_plan.model_digest().to_owned(),
            request_identity: request_identity.to_owned(),
            domains,
            connections,
            state_owners: owners,
            states,
        })
    }
}

#[pyfunction]
fn evidence(py: Python<'_>, result: &PyRunResult) -> PyResult<Py<PyFsiEvidence>> {
    result.fsi_evidence(py)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyFsiDomainEvidence>()?;
    module.add_class::<PyFsiConnectionEvidence>()?;
    module.add_class::<PyFsiInterfaceActionEvidence>()?;
    module.add_class::<PyFsiStateEvidence>()?;
    module.add_class::<PyFsiEvidence>()?;
    module.add_function(wrap_pyfunction!(evidence, module)?)?;
    Ok(())
}
