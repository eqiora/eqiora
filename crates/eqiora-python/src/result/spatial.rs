//! Read-only observations reconstructed from retained common spatial steps.
use super::*;
use crate::modeling::PyDimension;
use eqiora_numerics::{SpatialBalanceRole, SpatialBoundarySource};
use pyo3::exceptions::PyIndexError;
use pyo3::types::PyTuple;

/// One directed volume term and its exact equation/Field lineage.
#[pyclass(name = "SpatialBalanceTerm", module = "eqiora._eqiora", frozen)]
pub(crate) struct PySpatialBalanceTerm {
    #[pyo3(get)]
    domain_id: String,
    #[pyo3(get)]
    cell_index: usize,
    #[pyo3(get)]
    relation_id: String,
    #[pyo3(get)]
    tested_field_id: String,
    #[pyo3(get)]
    trial_field_id: Option<String>,
    #[pyo3(get)]
    term_index: usize,
    #[pyo3(get)]
    role: &'static str,
    #[pyo3(get)]
    dimension: PyDimension,
    #[pyo3(get)]
    previous: f64,
    #[pyo3(get)]
    current: f64,
    #[pyo3(get)]
    increment: f64,
    #[pyo3(get)]
    work: f64,
}

#[pymethods]
impl PySpatialBalanceTerm {
    fn __repr__(&self) -> String {
        format!(
            "SpatialBalanceTerm(role={:?}, domain_id={:?}, cell_index={}, work={}, dimension={:?})",
            self.role, self.domain_id, self.cell_index, self.work, self.dimension.value
        )
    }
}

/// Signed boundary work and the weak action at one coefficient, when available.
#[pyclass(name = "SpatialBoundaryWork", module = "eqiora._eqiora", frozen)]
pub(crate) struct PySpatialBoundaryWork {
    #[pyo3(get)]
    domain_id: String,
    #[pyo3(get)]
    kind: &'static str,
    #[pyo3(get)]
    field_id: String,
    #[pyo3(get)]
    connection_id: Option<String>,
    #[pyo3(get)]
    cell_index: Option<usize>,
    #[pyo3(get)]
    entity: Option<(usize, usize)>,
    #[pyo3(get)]
    slot: Option<usize>,
    #[pyo3(get)]
    component: Option<usize>,
    #[pyo3(get)]
    work: f64,
    #[pyo3(get)]
    work_dimension: PyDimension,
    #[pyo3(get)]
    action: Option<f64>,
    #[pyo3(get)]
    action_dimension: Option<PyDimension>,
}

#[pymethods]
impl PySpatialBoundaryWork {
    fn __repr__(&self) -> String {
        format!(
            "SpatialBoundaryWork(kind={:?}, field_id={:?}, entity={:?}, action={:?}, work={})",
            self.kind, self.field_id, self.entity, self.action, self.work
        )
    }
}

/// Signed residual work for one Domain and physical dimension.
#[pyclass(name = "SpatialWorkDefect", module = "eqiora._eqiora", frozen)]
pub(crate) struct PySpatialWorkDefect {
    #[pyo3(get)]
    domain_id: String,
    #[pyo3(get)]
    dimension: PyDimension,
    #[pyo3(get)]
    value: f64,
}

#[pymethods]
impl PySpatialWorkDefect {
    fn __repr__(&self) -> String {
        format!(
            "SpatialWorkDefect(domain_id={:?}, value={}, dimension={:?})",
            self.domain_id, self.value, self.dimension.value
        )
    }
}

/// Solver metadata and physical work belonging to one retained spatial output.
#[pyclass(name = "SpatialStepObservation", module = "eqiora._eqiora", frozen)]
pub(crate) struct PySpatialStepObservation {
    #[pyo3(get)]
    result_identity: String,
    #[pyo3(get)]
    trajectory_identity: String,
    #[pyo3(get)]
    output_index: usize,
    #[pyo3(get)]
    step_index: usize,
    #[pyo3(get)]
    previous_state_identity: String,
    #[pyo3(get)]
    state_identity: String,
    #[pyo3(get)]
    assembly_counts: Option<(usize, usize)>,
    #[pyo3(get)]
    balance_error: Option<String>,
    solve: Option<Py<PyLinearSolveSummary>>,
    volume_terms: Vec<Py<PySpatialBalanceTerm>>,
    boundary_work: Vec<Py<PySpatialBoundaryWork>>,
    defects: Vec<Py<PySpatialWorkDefect>>,
}

#[pymethods]
impl PySpatialStepObservation {
    fn __repr__(&self) -> String {
        format!(
            "SpatialStepObservation(output_index={}, step_index={}, state_identity={:?}, volume_terms={}, boundary_work={}, balance_error={:?})",
            self.output_index,
            self.step_index,
            self.state_identity,
            self.volume_terms.len(),
            self.boundary_work.len(),
            self.balance_error
        )
    }

    #[getter]
    fn solve(&self, py: Python<'_>) -> Option<Py<PyLinearSolveSummary>> {
        self.solve.as_ref().map(|solve| solve.clone_ref(py))
    }
    #[getter]
    fn volume_terms(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.volume_terms.iter().map(|term| term.clone_ref(py)))?.unbind())
    }
    #[getter]
    fn boundary_work(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.boundary_work.iter().map(|term| term.clone_ref(py)))?.unbind())
    }
    #[getter]
    fn defects(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.defects.iter().map(|term| term.clone_ref(py)))?.unbind())
    }
}

impl PyRunResult {
    /// Observe a retained output by position, including its actual predecessor.
    /// A sparse output's step_index can differ from its output_index.
    pub(super) fn spatial_step_observation(
        &self,
        py: Python<'_>,
        output_index: usize,
    ) -> PyResult<PySpatialStepObservation> {
        let trajectory = self
            .native
            .trajectory()
            .ok_or_else(|| capability_error(py, "this Result has no spatial Trajectory"))?;
        let steps = trajectory
            .spatial_steps()
            .ok_or_else(|| capability_error(py, "this Result has no spatial steps"))?;
        let (step_index, step) = steps.get(output_index).ok_or_else(|| {
            PyIndexError::new_err("spatial output index is outside the retained schedule")
        })?;
        let solve = PyLinearSolveSummary::from_common_result(&self.native, Some(output_index))
            .map(|summary| Py::new(py, summary))
            .transpose()?;
        let mut observation = PySpatialStepObservation {
            result_identity: self.native.identity().to_owned(),
            trajectory_identity: trajectory.identity().to_owned(),
            output_index,
            step_index: *step_index,
            previous_state_identity: step.previous_state().identity().to_owned(),
            state_identity: step.state().identity().to_owned(),
            assembly_counts: self.native.spatial_step_assembly_counts(output_index),
            balance_error: None,
            solve,
            volume_terms: Vec::new(),
            boundary_work: Vec::new(),
            defects: Vec::new(),
        };
        let balance = match step.balance() {
            Ok(balance) => balance,
            Err(reason) => {
                observation.balance_error = Some(reason.to_owned());
                return Ok(observation);
            }
        };
        for (cell_index, domain, term) in balance.volume_terms() {
            observation.volume_terms.push(Py::new(
                py,
                PySpatialBalanceTerm {
                    domain_id: domain.ulid().to_string(),
                    cell_index,
                    relation_id: term.relation().ulid().to_string(),
                    tested_field_id: term.tested_field().ulid().to_string(),
                    trial_field_id: term.trial_field().map(|field| field.ulid().to_string()),
                    term_index: term.term_index(),
                    role: match term.role() {
                        SpatialBalanceRole::Storage => "storage",
                        SpatialBalanceRole::StatePotential => "state-potential",
                        SpatialBalanceRole::Dissipation => "dissipation",
                        SpatialBalanceRole::ConstraintWork => "constraint-work",
                        SpatialBalanceRole::AppliedVolumeWork => "applied-volume-work",
                    },
                    dimension: PyDimension::from_native(term.work().dim()),
                    previous: term.previous().value(),
                    current: term.current().value(),
                    increment: term.increment().value(),
                    work: term.work().value(),
                },
            )?);
        }
        for term in balance.boundary_work() {
            let mut projected = PySpatialBoundaryWork {
                domain_id: term.domain().ulid().to_string(),
                kind: "natural",
                field_id: String::new(),
                connection_id: None,
                cell_index: None,
                entity: None,
                slot: None,
                component: None,
                work: term.work().value(),
                work_dimension: PyDimension::from_native(term.work().dim()),
                action: term.action().map(|action| action.value()),
                action_dimension: term
                    .action()
                    .map(|action| PyDimension::from_native(action.dim())),
            };
            match term.source() {
                SpatialBoundarySource::Natural { cell, field } => {
                    projected.cell_index = Some(*cell);
                    projected.field_id = field.ulid().to_string();
                }
                SpatialBoundarySource::Prescribed {
                    field,
                    entity,
                    slot,
                    component,
                }
                | SpatialBoundarySource::Interface {
                    field,
                    entity,
                    slot,
                    component,
                    ..
                } => {
                    projected.kind = "prescribed";
                    projected.field_id = field.ulid().to_string();
                    projected.entity = Some((entity.dimension(), entity.index()));
                    projected.slot = Some(*slot);
                    projected.component = Some(*component);
                    if let SpatialBoundarySource::Interface { connection, .. } = term.source() {
                        projected.kind = "interface";
                        projected.connection_id = Some(connection.ulid().to_string());
                    }
                }
            }
            observation.boundary_work.push(Py::new(py, projected)?);
        }
        for (domain, defect) in balance
            .defects()
            .map_err(|error| crate::diagnostic_error(py, &[error]))?
        {
            observation.defects.push(Py::new(
                py,
                PySpatialWorkDefect {
                    domain_id: domain.ulid().to_string(),
                    dimension: PyDimension::from_native(defect.dim()),
                    value: defect.value(),
                },
            )?);
        }
        Ok(observation)
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySpatialBalanceTerm>()?;
    module.add_class::<PySpatialBoundaryWork>()?;
    module.add_class::<PySpatialWorkDefect>()?;
    module.add_class::<PySpatialStepObservation>()?;
    Ok(())
}
