//! Typed Python projections delegate evaluation and tangent admission to CommonResult.
use super::*;
use crate::model::PyObservableRef;
use crate::modeling::{PyDimension, PyValueType};
use eqiora::graph::Op;
use eqiora::kernel::{ExprNode, KernelNode, ObservableMeasure, ObservableReduction, SymbolRef};
use eqiora::meshing::{MeshTopology, QuadratureRule};
use eqiora::{DynQuantity, Id, ValueLiteral, kinds};
use pyo3::types::PyDict;
use std::collections::{BTreeMap, HashMap, HashSet};

/// Typed Result-owned value with its effective numerical quadrature.
#[pyclass(
    name = "Observation",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
pub(crate) struct PyObservation {
    value: ValueLiteral,
    #[pyo3(get)]
    result_identity: String,
    #[pyo3(get)]
    observable_id: String,
    #[pyo3(get)]
    evaluation_kind: &'static str,
    #[pyo3(get)]
    quadratures: BTreeMap<String, (&'static str, usize, usize)>,
}

#[pymethods]
impl PyObservation {
    fn __repr__(&self) -> String {
        format!(
            "Observation(observable_id={:?}, evaluation_kind={:?}, result_identity={:?})",
            self.observable_id, self.evaluation_kind, self.result_identity
        )
    }

    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::modeling::value_literal::to_python(py, &self.value)
    }
    #[getter]
    fn value_type(&self) -> PyValueType {
        PyValueType {
            value: self.value.value_type().clone(),
        }
    }
}

/// Dimensioned Field coefficient variation bound to one exact accepted Result.
#[pyclass(
    name = "ObservableStateTangent",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
pub(crate) struct PyObservableStateTangent {
    native: eqiora_numerics::CommonObservableStateTangent,
    #[pyo3(get)]
    result_identity: String,
}

#[pymethods]
impl PyObservableStateTangent {
    fn __repr__(&self) -> String {
        format!(
            "ObservableStateTangent(result_identity={:?})",
            self.result_identity
        )
    }
}

impl PyRunResult {
    fn observation_rules(
        &self,
        py: Python<'_>,
        observable: &PyObservableRef,
        points: Option<usize>,
    ) -> PyResult<HashMap<Id<kinds::Domain>, QuadratureRule>> {
        if observable.model_digest != self.identity.model_digest() {
            return Err(PyValueError::new_err(
                "ObservableRef belongs to a different exact Model artifact",
            ));
        }
        let (transaction, _) = self
            .native
            .plan()
            .model_artifact()
            .to_transaction()
            .map_err(|errors| diagnostic_error(py, &errors))?;
        let definitions = transaction
            .ops()
            .iter()
            .filter_map(|operation| match operation {
                Op::DefineKernelNode {
                    node: KernelNode::Observable(definition),
                } => Some((definition.id(), definition)),
                _ => None,
            })
            .collect::<HashMap<_, _>>();
        let mut pending = vec![observable.id];
        let mut visited = HashSet::new();
        let mut measures = HashMap::new();
        while let Some(id) = pending.pop() {
            if !visited.insert(id) {
                continue;
            }
            let Some(definition) = definitions.get(&id) else {
                return Err(PyKeyError::new_err(
                    "Observable is outside this exact Result Model",
                ));
            };
            if let ObservableReduction::SpatialIntegral { domain, measure } = definition.reduction()
            {
                measures.insert(domain, measure);
            }
            for node in definition.expression().nodes() {
                if let ExprNode::Symbol(SymbolRef::Observable(dependency)) = node {
                    pending.push(*dependency);
                }
            }
        }
        let Some(points) = points else {
            return Ok(HashMap::new());
        };
        if measures.is_empty() {
            return Err(PyValueError::new_err(
                "finite Observable does not accept spatial quadrature",
            ));
        }
        let owner = self.native.plan().authenticated_mesh().ok_or_else(|| {
            PyValueError::new_err("Observable requires an authenticated Result mesh")
        })?;
        let mesh = owner.cartesian_mesh().ok_or_else(|| {
            PyValueError::new_err(
                "Observable quadrature requires an admitted Cartesian Result mesh",
            )
        })?;
        let mut rules = HashMap::new();
        for (domain, measure) in &measures {
            let dimension = mesh
                .mesh()
                .topological_dimension()
                .checked_sub(usize::from(*measure == ObservableMeasure::Boundary))
                .ok_or_else(|| {
                    PyValueError::new_err("Observable boundary has no ambient dimension")
                })?;
            let rule = if dimension == 0 {
                if points != 1
                    && measures
                        .values()
                        .all(|measure| *measure == ObservableMeasure::Boundary)
                {
                    return Err(PyValueError::new_err(
                        "point measure requires quadrature_points=1",
                    ));
                }
                QuadratureRule::point()
            } else {
                QuadratureRule::tensor_product_gauss_legendre(dimension, points)
                    .map_err(|error| diagnostic_error(py, &[error]))?
            };
            rules.insert(*domain, rule);
        }
        Ok(rules)
    }

    pub(super) fn observe_value(
        &self,
        py: Python<'_>,
        observable: &PyObservableRef,
        points: Option<usize>,
    ) -> PyResult<PyObservation> {
        let rules = self.observation_rules(py, observable, points)?;
        let value = self
            .native
            .observe(self.native.plan().model_artifact(), observable.id, &rules)
            .map_err(|error| diagnostic_error(py, &[error]))?;
        Ok(PyObservation {
            value: value.value().clone(),
            result_identity: value.result_identity().to_owned(),
            observable_id: value.observable().ulid().to_string(),
            evaluation_kind: "value",
            quadratures: quadrature_metadata(value.quadratures(), points.unwrap_or(0)),
        })
    }

    pub(super) fn bind_observable_tangent(
        &self,
        py: Python<'_>,
        directions: &Bound<'_, PyDict>,
    ) -> PyResult<PyObservableStateTangent> {
        let mut fields = Vec::with_capacity(directions.len());
        for (field, direction) in directions.iter() {
            let field = field.extract::<PyRef<'_, PyModelFieldRef>>()?;
            if field.exact_model_digest() != self.identity.model_digest() {
                return Err(PyValueError::new_err(
                    "FieldRef belongs to a different exact Model artifact",
                ));
            }
            let (dimension, values) = direction.extract::<(PyRef<'_, PyDimension>, Vec<f64>)>()?;
            let id = Id::<kinds::Field>::from_ulid(
                field
                    .exact_id()
                    .parse()
                    .map_err(|_| PyValueError::new_err("FieldRef has an invalid exact ID"))?,
            );
            fields.push((
                id,
                values
                    .into_iter()
                    .map(|value| DynQuantity::new(value, dimension.native()))
                    .collect(),
            ));
        }
        let native = self
            .native
            .observable_state_tangent(fields)
            .map_err(|error| diagnostic_error(py, &[error]))?;
        Ok(PyObservableStateTangent {
            native,
            result_identity: self.native.identity().to_owned(),
        })
    }

    pub(super) fn observe_jvp(
        &self,
        py: Python<'_>,
        observable: &PyObservableRef,
        tangent: &PyObservableStateTangent,
        points: usize,
    ) -> PyResult<PyObservation> {
        let rules = self.observation_rules(py, observable, Some(points))?;
        let value = self
            .native
            .observe_state_jvp(
                self.native.plan().model_artifact(),
                observable.id,
                &rules,
                &tangent.native,
            )
            .map_err(|error| diagnostic_error(py, &[error]))?;
        Ok(PyObservation {
            value,
            result_identity: self.native.identity().to_owned(),
            observable_id: observable.id.ulid().to_string(),
            evaluation_kind: "state-jvp",
            quadratures: quadrature_metadata(&rules, points),
        })
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyObservation>()?;
    module.add_class::<PyObservableStateTangent>()?;
    Ok(())
}

fn quadrature_metadata(
    rules: &HashMap<Id<kinds::Domain>, QuadratureRule>,
    points: usize,
) -> BTreeMap<String, (&'static str, usize, usize)> {
    rules
        .iter()
        .map(|(domain, rule)| {
            let dimension = rule.reference_cell().dimension();
            let (kind, points) = if dimension == 0 {
                ("Point", 1)
            } else {
                ("GaussLegendre", points)
            };
            (domain.ulid().to_string(), (kind, dimension, points))
        })
        .collect()
}
