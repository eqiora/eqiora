use eqiora_core::Diagnostic;
use eqiora_meshing::{DiscreteFieldPayload, MeshQualityGate, SimplicialMesh};

use super::plan::{
    GeometryKind, XdmfArrayResponse, XdmfArrayRole, XdmfArrayValues, XdmfImportPlan,
    XdmfScalarType, invalid_import,
};

/// One selected field reconstructed through the shared payload invariant.
#[derive(Debug, Clone, PartialEq)]
pub struct XdmfImportedField {
    name: Option<String>,
    origin_selector: Vec<u32>,
    payload: DiscreteFieldPayload,
}

impl XdmfImportedField {
    /// Optional display name; never a content selector.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    /// Owning selected Attribute element path.
    #[must_use]
    pub fn origin_selector(&self) -> &[u32] {
        &self.origin_selector
    }
    /// Invariant-checked mesh-associated values.
    #[must_use]
    pub const fn payload(&self) -> &DiscreteFieldPayload {
        &self.payload
    }
}

/// Accepted content plus the exact resolver observations used to derive it.
#[derive(Debug, Clone, PartialEq)]
pub struct XdmfImport {
    mesh: SimplicialMesh,
    fields: Vec<XdmfImportedField>,
}

impl XdmfImport {
    /// Shared invariant-checked affine simplex mesh.
    #[must_use]
    pub const fn mesh(&self) -> &SimplicialMesh {
        &self.mesh
    }
    /// Selected fields in explicit caller order.
    #[must_use]
    pub fn fields(&self) -> &[XdmfImportedField] {
        &self.fields
    }
}

impl XdmfImportPlan {
    /// Admit one exact response per request and reconstruct shared mesh/field values.
    ///
    /// # Errors
    /// Returns `EQ0810` for a missing, extra, reordered, cross-wired, wrongly
    /// typed/shaped, non-finite, or over-budget response. Shared mesh and field
    /// invariants retain their own diagnostics.
    pub fn accept(
        &self,
        responses: &[XdmfArrayResponse],
        quality_gate: MeshQualityGate,
    ) -> Result<XdmfImport, Diagnostic> {
        accept_plan(self, responses, quality_gate)
    }
}

pub(crate) fn accept_plan(
    plan: &XdmfImportPlan,
    responses: &[XdmfArrayResponse],
    quality_gate: MeshQualityGate,
) -> Result<XdmfImport, Diagnostic> {
    if responses.len() != plan.requests.len() {
        return Err(invalid_import(
            "XDMF response count differs from the exact request count",
        ));
    }
    let mut total_source_bytes = 0_usize;
    let mut total_resolved_bytes = 0_usize;
    let mut work = 0_usize;
    for (request, response) in plan.requests.iter().zip(responses) {
        if response.request() != request {
            return Err(invalid_import(
                "XDMF response order or request identity differs from the plan",
            ));
        }
        if response.source_bytes().is_empty() {
            return Err(invalid_import(
                "XDMF resolved source occurrence bytes must not be empty",
            ));
        }
        if response.source_bytes().len() > plan.limits.max_source_bytes {
            return Err(invalid_import(
                "XDMF source occurrence exceeds the configured byte limit",
            ));
        }
        total_source_bytes = checked_add(
            total_source_bytes,
            response.source_bytes().len(),
            "XDMF aggregate source bytes",
        )?;
        if total_source_bytes > plan.limits.max_total_source_bytes {
            return Err(invalid_import(
                "XDMF aggregate source bytes exceed the configured limit",
            ));
        }
        let values = value_count(response.values());
        if values != shape_product(request.shape())? {
            return Err(invalid_import(
                "XDMF resolved value count differs from the declared shape",
            ));
        }
        if values > plan.limits.max_array_values {
            return Err(invalid_import(
                "XDMF resolved array exceeds the configured scalar-value limit",
            ));
        }
        let bytes = values
            .checked_mul(8)
            .ok_or_else(|| invalid_import("XDMF resolved array byte count overflows usize"))?;
        total_resolved_bytes =
            checked_add(total_resolved_bytes, bytes, "XDMF aggregate resolved bytes")?;
        if total_resolved_bytes > plan.limits.max_resolved_bytes {
            return Err(invalid_import(
                "XDMF aggregate resolved bytes exceed the configured limit",
            ));
        }
        work = checked_add(work, values, "XDMF resolution work")?;
        if work > plan.limits.max_resolution_work {
            return Err(invalid_import(
                "XDMF resolution work exceeds the configured limit",
            ));
        }
        match (request.scalar(), response.values()) {
            (XdmfScalarType::U64, XdmfArrayValues::U64(_)) => {}
            (XdmfScalarType::F64, XdmfArrayValues::F64(values)) => {
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(invalid_import(
                        "XDMF resolved f64 values must all be finite",
                    ));
                }
            }
            _ => {
                return Err(invalid_import(
                    "XDMF response scalar type differs from its request",
                ));
            }
        }
    }

    let geometry = f64_values(&responses[0], XdmfArrayRole::Geometry)?;
    let vertices = reconstruct_vertices(plan, geometry)?;
    let topology = u64_values(&responses[1], XdmfArrayRole::Topology)?;
    let cells = topology
        .chunks_exact(plan.dimension + 1)
        .map(|cell| {
            cell.iter()
                .map(|index| {
                    usize::try_from(*index)
                        .map_err(|_| invalid_import("XDMF topology index exceeds local usize"))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    if cells.len() != plan.cell_count {
        return Err(invalid_import(
            "XDMF reconstructed cell count differs from metadata",
        ));
    }
    let mesh = SimplicialMesh::new(plan.dimension, vertices, cells, quality_gate)?;

    let mut fields = Vec::new();
    fields
        .try_reserve_exact(plan.fields.len())
        .map_err(|_| invalid_import("XDMF imported field allocation failed"))?;
    for (field, response) in plan.fields.iter().zip(&responses[2..]) {
        let values = f64_values(response, XdmfArrayRole::Attribute)?.to_vec();
        let payload = DiscreteFieldPayload::new(&mesh, field.association, field.shape, values)?;
        fields.push(XdmfImportedField {
            name: field.name.clone(),
            origin_selector: field.origin_selector.clone(),
            payload,
        });
    }
    Ok(XdmfImport { mesh, fields })
}

fn reconstruct_vertices(
    plan: &XdmfImportPlan,
    values: &[f64],
) -> Result<Vec<Vec<f64>>, Diagnostic> {
    let source_width = match plan.geometry_kind {
        GeometryKind::Xy => 2,
        GeometryKind::Xyz => 3,
    };
    let mut vertices = Vec::new();
    vertices
        .try_reserve_exact(values.len() / source_width)
        .map_err(|_| invalid_import("XDMF vertex allocation failed"))?;
    for coordinate in values.chunks_exact(source_width) {
        let vertex = match (plan.dimension, plan.geometry_kind) {
            (2, GeometryKind::Xy) => vec![coordinate[0], coordinate[1]],
            (2, GeometryKind::Xyz) if coordinate[2] == 0.0 => vec![coordinate[0], coordinate[1]],
            (2, GeometryKind::Xyz) => {
                return Err(invalid_import(
                    "two-dimensional XDMF XYZ geometry requires zero Z",
                ));
            }
            (3, GeometryKind::Xyz) => coordinate.to_vec(),
            _ => {
                return Err(invalid_import(
                    "XDMF geometry is incompatible with topology dimension",
                ));
            }
        };
        vertices.push(vertex);
    }
    Ok(vertices)
}

fn f64_values(response: &XdmfArrayResponse, role: XdmfArrayRole) -> Result<&[f64], Diagnostic> {
    if response.request().role() != role {
        return Err(invalid_import(
            "XDMF response role differs from canonical plan order",
        ));
    }
    match response.values() {
        XdmfArrayValues::F64(values) => Ok(values),
        XdmfArrayValues::U64(_) => Err(invalid_import("XDMF response requires f64 values")),
    }
}

fn u64_values(response: &XdmfArrayResponse, role: XdmfArrayRole) -> Result<&[u64], Diagnostic> {
    if response.request().role() != role {
        return Err(invalid_import(
            "XDMF response role differs from canonical plan order",
        ));
    }
    match response.values() {
        XdmfArrayValues::U64(values) => Ok(values),
        XdmfArrayValues::F64(_) => Err(invalid_import("XDMF response requires u64 values")),
    }
}

fn value_count(values: &XdmfArrayValues) -> usize {
    match values {
        XdmfArrayValues::U64(values) => values.len(),
        XdmfArrayValues::F64(values) => values.len(),
    }
}

fn shape_product(shape: &[u64]) -> Result<usize, Diagnostic> {
    shape.iter().try_fold(1_usize, |product, dimension| {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| invalid_import("XDMF array extent exceeds local usize"))?;
        product
            .checked_mul(dimension)
            .ok_or_else(|| invalid_import("XDMF array shape product overflows usize"))
    })
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize, Diagnostic> {
    left.checked_add(right)
        .ok_or_else(|| invalid_import(format!("{label} overflows usize")))
}
