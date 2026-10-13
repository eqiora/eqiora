use crate::allocation::{allocate_vec, copy_slice, copy_string};
use eqiora_core::Diagnostic;
use eqiora_meshing::{DiscreteFieldPayload, MeshQualityGate, SimplicialMesh};

use super::plan::{VtuImportPlan, invalid_import};

/// One selected field reconstructed through the shared payload invariant.
#[derive(Debug, PartialEq)]
pub struct VtuImportedField {
    selector: Vec<u32>,
    name: Option<String>,
    raw_shape: Vec<u64>,
    payload: DiscreteFieldPayload,
}

impl VtuImportedField {
    /// Exact selected DataArray structural path.
    #[must_use]
    pub fn selector(&self) -> &[u32] {
        &self.selector
    }

    /// Optional display name; never a selector.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Source-logical shape, preserving scalar versus explicit vector shape.
    #[must_use]
    pub fn raw_shape(&self) -> &[u64] {
        &self.raw_shape
    }

    /// Invariant-checked mesh-associated values.
    #[must_use]
    pub const fn payload(&self) -> &DiscreteFieldPayload {
        &self.payload
    }
}

/// Shared invariant-checked mesh and caller-selected fields.
#[derive(Debug, PartialEq)]
pub struct VtuImport {
    mesh: SimplicialMesh,
    fields: Vec<VtuImportedField>,
}

impl VtuImport {
    /// Accepted affine-simplex mesh.
    #[must_use]
    pub const fn mesh(&self) -> &SimplicialMesh {
        &self.mesh
    }

    /// Selected fields in explicit caller order.
    #[must_use]
    pub fn fields(&self) -> &[VtuImportedField] {
        &self.fields
    }
}

pub(crate) fn accept_plan(
    plan: &VtuImportPlan,
    quality_gate: MeshQualityGate,
) -> Result<VtuImport, Diagnostic> {
    let dimension = plan.cell_kind.dimension();
    let vertex_count = plan.geometry.len() / dimension;
    let mut vertices = allocate_vec(vertex_count, "VTU mesh vertex table")?;
    for coordinates in plan.geometry.chunks_exact(dimension) {
        vertices.push(copy_slice(coordinates, "VTU mesh vertex coordinates")?);
    }
    let arity = plan.cell_kind.arity();
    let cell_count = plan.topology.len() / arity;
    let mut cells = allocate_vec(cell_count, "VTU mesh cell table")?;
    for topology in plan.topology.chunks_exact(arity) {
        let mut cell = allocate_vec(arity, "VTU mesh cell connectivity")?;
        for index in topology {
            cell.push(
                usize::try_from(*index)
                    .map_err(|_| invalid_import("VTU topology index exceeds local usize"))?,
            );
        }
        cells.push(cell);
    }
    let mesh = SimplicialMesh::new(dimension, vertices, cells, quality_gate).map_err(|error| {
        invalid_import(format!(
            "VTU mesh failed shared topology/geometry admission: {error}"
        ))
    })?;

    let mut fields = allocate_vec(plan.fields.len(), "VTU accepted field table")?;
    for field in &plan.fields {
        let payload = DiscreteFieldPayload::new(
            &mesh,
            field.association,
            field.shape,
            copy_slice(&field.values, "VTU accepted field values")?,
        )
        .map_err(|error| {
            invalid_import(format!(
                "VTU field failed shared payload admission: {error}"
            ))
        })?;
        fields.push(VtuImportedField {
            selector: copy_slice(&field.selector, "VTU accepted field selector")?,
            name: field
                .name
                .as_deref()
                .map(|name| copy_string(name, "VTU accepted field name"))
                .transpose()?,
            raw_shape: copy_slice(&field.raw_shape, "VTU accepted field shape")?,
            payload,
        });
    }
    Ok(VtuImport { mesh, fields })
}
