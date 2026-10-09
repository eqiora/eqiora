//! Typed static Field output projected from one accepted Result.

use eqiora::DimExponents;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::array::PyArrayBuffer;
use crate::meshing::PyMesh;
use crate::model::PyModelFieldRef;

/// Project support-entity count without interpreting vector moments as nodal components.
pub(super) fn coefficient_count(
    space: eqiora::realization::Space,
    value_shape: &[usize],
    complex: bool,
    coordinate_count: usize,
    logical_shape: &[usize],
) -> PyResult<usize> {
    use eqiora::realization::SpaceFamily;
    let invalid = || {
        pyo3::exceptions::PyRuntimeError::new_err(
            "common Result Field block contradicts its coefficient layout",
        )
    };
    let width = match space.family() {
        SpaceFamily::TetrahedralEdge | SpaceFamily::TetrahedralFace => {
            if value_shape != [3] || logical_shape.len() != 1 {
                return Err(invalid());
            }
            1
        }
        _ => value_shape
            .iter()
            .try_fold(1usize, |n, extent| n.checked_mul(*extent))
            .filter(|width| *width > 0)
            .ok_or_else(invalid)?,
    };
    let count = logical_shape
        .iter()
        .try_fold(1usize, |n, extent| n.checked_mul(*extent))
        .ok_or_else(invalid)?;
    if logical_shape.is_empty()
        || count.checked_mul(if complex { 2 } else { 1 }) != Some(coordinate_count)
        || !count.is_multiple_of(width)
    {
        return Err(invalid());
    }
    Ok(count / width)
}

pub(crate) struct FieldOutputBlock {
    association: &'static str,
    values: Py<PyArrayBuffer>,
    coefficient_count: usize,
    logical_shape: Vec<usize>,
}

impl FieldOutputBlock {
    pub(super) const fn new(
        association: &'static str,
        values: Py<PyArrayBuffer>,
        coefficient_count: usize,
        logical_shape: Vec<usize>,
    ) -> Self {
        Self {
            association,
            values,
            coefficient_count,
            logical_shape,
        }
    }

    pub(crate) const fn association(&self) -> &'static str {
        self.association
    }

    pub(crate) const fn coefficient_count(&self) -> usize {
        self.coefficient_count
    }

    pub(crate) fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
    }

    pub(crate) fn snapshot(&self, py: Python<'_>) -> PyResult<Vec<f64>> {
        self.values.borrow(py).snapshot(py)
    }
}

/// Immutable coefficients for one exact Model Field on one exact Mesh.
#[pyclass(
    name = "FieldOutput",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object
)]
pub(crate) struct PyFieldOutput {
    field: Py<PyModelFieldRef>,
    mesh: Py<PyMesh>,
    dimension: DimExponents,
    coefficient_dimension: DimExponents,
    value_shape: Vec<usize>,
    space: &'static str,
    blocks: Vec<FieldOutputBlock>,
}

impl PyFieldOutput {
    pub(super) fn new(
        field: Py<PyModelFieldRef>,
        mesh: Py<PyMesh>,
        dimension: DimExponents,
        coefficient_dimension: DimExponents,
        value_shape: Vec<usize>,
        space: &'static str,
        blocks: Vec<FieldOutputBlock>,
    ) -> Self {
        Self {
            field,
            mesh,
            dimension,
            coefficient_dimension,
            value_shape,
            space,
            blocks,
        }
    }

    pub(crate) fn mesh_handle(&self, py: Python<'_>) -> Py<PyMesh> {
        self.mesh.clone_ref(py)
    }

    pub(crate) fn field_handle(&self, py: Python<'_>) -> Py<PyModelFieldRef> {
        self.field.clone_ref(py)
    }

    pub(crate) const fn dimension_value(&self) -> DimExponents {
        self.dimension
    }

    pub(crate) fn value_shape_value(&self) -> &[usize] {
        &self.value_shape
    }

    pub(crate) const fn space_value(&self) -> &'static str {
        self.space
    }

    pub(crate) fn blocks(&self) -> &[FieldOutputBlock] {
        &self.blocks
    }

    fn block(&self, association: &str) -> PyResult<&FieldOutputBlock> {
        self.blocks
            .iter()
            .find(|block| block.association == association)
            .ok_or_else(|| PyKeyError::new_err(association.to_owned()))
    }
}

#[pymethods]
impl PyFieldOutput {
    #[getter]
    fn field(&self, py: Python<'_>) -> Py<PyModelFieldRef> {
        self.field.clone_ref(py)
    }

    #[getter]
    fn mesh(&self, py: Python<'_>) -> Py<PyMesh> {
        self.mesh.clone_ref(py)
    }

    #[getter]
    fn dimension(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::modeling::dimension::exponents(py, self.dimension)
    }

    /// SI dimension of the stored coefficients, including the Space functional's measure.
    #[getter]
    fn coefficient_dimension(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        crate::modeling::dimension::exponents(py, self.coefficient_dimension)
    }

    /// Exact mathematical component shape; an empty tuple is scalar.
    #[getter]
    fn value_shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.value_shape.iter().copied())?.unbind())
    }

    /// Resolved discrete space or basis family for this Field.
    #[getter]
    const fn space(&self) -> &'static str {
        self.space
    }

    /// Coefficient associations in exact output-block order.
    #[getter]
    fn associations(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.blocks.iter().map(|block| block.association))?.unbind())
    }

    /// Read-only Eqiora-owned coefficients for one exact association.
    #[pyo3(signature = (association, /))]
    fn values(&self, py: Python<'_>, association: &str) -> PyResult<Py<PyArrayBuffer>> {
        Ok(self.block(association)?.values.clone_ref(py))
    }

    /// Number of support entities represented by one coefficient block.
    #[pyo3(signature = (association, /))]
    fn coefficient_count(&self, association: &str) -> PyResult<usize> {
        Ok(self.block(association)?.coefficient_count)
    }

    /// Logical array shape for one coefficient block.
    #[pyo3(signature = (association, /))]
    fn logical_shape(&self, py: Python<'_>, association: &str) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.block(association)?.logical_shape.iter().copied())?.unbind())
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "FieldOutput(field={:?}, space={:?}, associations={:?})",
            self.field.borrow(py).exact_id(),
            self.space,
            self.blocks
                .iter()
                .map(|block| block.association)
                .collect::<Vec<_>>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::coefficient_count;
    use eqiora::realization::Space;

    #[test]
    fn moment_counts_are_entities_even_for_complex_vectors() {
        for complex in [false, true] {
            let coordinates = if complex { 2 } else { 1 };
            // One tetrahedron has six edge integrals and four face fluxes.
            assert_eq!(
                coefficient_count(
                    Space::tetrahedral_edge(),
                    &[3],
                    complex,
                    6 * coordinates,
                    &[6]
                )
                .unwrap(),
                6
            );
            assert_eq!(
                coefficient_count(
                    Space::tetrahedral_face(),
                    &[3],
                    complex,
                    4 * coordinates,
                    &[4]
                )
                .unwrap(),
                4
            );
            // Replicated nodal vectors still carry three components at every vertex.
            assert_eq!(
                coefficient_count(
                    Space::continuous_lagrange(std::num::NonZeroU16::MIN),
                    &[3],
                    complex,
                    12 * coordinates,
                    &[4, 3]
                )
                .unwrap(),
                4
            );
        }
    }

    #[test]
    fn inconsistent_coefficient_layouts_reject() {
        let edge = Space::tetrahedral_edge();
        assert!(coefficient_count(edge, &[3], false, 18, &[6, 3]).is_err());
        assert!(coefficient_count(edge, &[], false, 6, &[6]).is_err());
        assert!(coefficient_count(edge, &[3], true, 11, &[6]).is_err());
        assert!(coefficient_count(edge, &[3], true, 0, &[usize::MAX]).is_err());
    }
}
