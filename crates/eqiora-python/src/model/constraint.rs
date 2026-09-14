//! Exact immutable condition selection, distinct from enforcement policy.

use super::*;
use eqiora::entity::kinds;
use eqiora::kernel::{KernelNode, RelationConditionKind};
use eqiora_numerics::finite_constraints::ConstraintRef;

/// One inequality or complementarity condition bound to an exact Model artifact.
#[pyclass(
    name = "ConstraintRef",
    module = "eqiora._eqiora",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PyConstraintRef {
    pub(crate) model_digest: String,
    pub(crate) native: ConstraintRef,
    pub(crate) kind: &'static str,
}

impl Hash for PyConstraintRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.model_digest.hash(state);
        self.native.relation().ulid().hash(state);
        self.native.ordinal().hash(state);
    }
}

#[pymethods]
impl PyConstraintRef {
    fn __repr__(&self) -> String {
        format!(
            "ConstraintRef(relation_id={:?}, ordinal={}, kind={:?}, model_digest={:?})",
            self.relation_id(),
            self.native.ordinal(),
            self.kind,
            self.model_digest
        )
    }
    #[getter]
    fn model_digest(&self) -> &str {
        &self.model_digest
    }
    #[getter]
    fn relation_id(&self) -> String {
        self.native.relation().ulid().to_string()
    }
    #[getter]
    fn ordinal(&self) -> u32 {
        self.native.ordinal()
    }
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind
    }
}

pub(super) fn select(
    model: &PyModel,
    py: Python<'_>,
    selection: &str,
    ordinal: u32,
) -> PyResult<PyConstraintRef> {
    let relation = model
        .document
        .as_ref()
        .and_then(|document| document.aliases().get(selection).copied())
        .and_then(RawId::downcast::<kinds::Relation>)
        .or_else(|| {
            selection
                .parse::<ulid::Ulid>()
                .ok()
                .map(eqiora::Id::from_ulid)
        })
        .ok_or_else(|| {
            PyTypeError::new_err("constraint selection requires an exact Relation alias or ULID")
        })?;
    let kernel = model
        .artifact
        .to_program()
        .map_err(|errors| validation_error(py, &errors))?;
    let Some(KernelNode::Relation(definition)) = kernel.node(relation.erase()) else {
        return Err(PyTypeError::new_err(
            "constraint Relation is outside this exact Model",
        ));
    };
    let kind = match definition
        .conditions()
        .and_then(|conditions| conditions.get(ordinal as usize))
    {
        Some(RelationConditionKind::Inequality) => "inequality",
        Some(RelationConditionKind::Complementarity) => "complementarity",
        _ => {
            return Err(PyTypeError::new_err(
                "constraint selection requires an existing inequality or complementarity ordinal",
            ));
        }
    };
    let model_digest = model
        .artifact
        .artifact_reference()
        .map_err(|error| validation_error(py, &[error]))?
        .artifact()
        .to_string();
    Ok(PyConstraintRef {
        model_digest,
        native: ConstraintRef::new(relation, ordinal),
        kind,
    })
}
