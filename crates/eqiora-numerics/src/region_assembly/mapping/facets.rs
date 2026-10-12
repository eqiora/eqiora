//! Parent-oriented facet geometry shared by exterior laws and trace quotients.
use super::*;
use crate::form_compiler::region::{BoundRegionForm, PreparedRegionFacet};
use eqiora_meshing::{AffineGeometryMap, MeshGeometry};

/// The caller supplies an incidence authenticated against exact Region ownership.
pub(crate) fn prepare_mesh_facet<'mesh, S: Coefficient>(
    mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
    form: &BoundRegionForm<S>,
    field: RawId,
    facet: MeshEntity,
    parent: EntityIncidence,
    datum: impl Fn(&[f64], &[f64]) -> Result<Vec<S>, Diagnostic>,
) -> Result<PreparedRegionFacet<S>, Diagnostic> {
    let cell = mesh
        .geometry_map(parent.entity)
        .ok_or_else(|| invalid("facet has no parent geometry"))?;
    let geometry = mesh
        .geometry_map(facet)
        .ok_or_else(|| invalid("facet has no geometry"))?;
    let cell_vertices = mesh
        .incidence(parent.entity, 0)
        .ok_or_else(|| invalid("facet parent has no vertices"))?;
    let vertices = mesh
        .incidence(facet, 0)
        .ok_or_else(|| invalid("facet has no vertices"))?;
    let positions = vertices
        .iter()
        .map(|vertex| {
            cell_vertices
                .iter()
                .position(|candidate| candidate.entity == vertex.entity)
                .ok_or_else(|| invalid("facet vertex is absent from its parent"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    form.prepare_natural_facet(
        field,
        &cell,
        (&geometry, parent, &positions),
        &form.natural_facet_quadrature(field)?,
        datum,
    )
}

/// Admitted constitutive tractions cancel in the shared test trace. The remaining
/// equation-derived transport flux must still be completed on each owned side;
/// different transport coefficients need not cancel across the interface.
pub(crate) fn prepare_interface_facets<'mesh, 'form, S: Coefficient + 'form>(
    mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
    forms: impl IntoIterator<Item = &'form BoundRegionForm<S>>,
    traces: &[TraceBinding],
) -> Result<Vec<(usize, PreparedRegionFacet<S>)>, Diagnostic> {
    let forms = forms
        .into_iter()
        .map(|form| (form.domain(), form))
        .collect::<BTreeMap<_, _>>();
    let mut result = Vec::new();
    for trace in traces {
        for (side, endpoint) in trace.quotient.endpoints().iter().enumerate() {
            let form = forms
                .get(&endpoint.domain().erase())
                .ok_or_else(|| invalid("interface facet has no exact Region form"))?;
            let field = endpoint.field().erase();
            let layout = form
                .fields()
                .iter()
                .find(|layout| layout.field == field)
                .ok_or_else(|| invalid("interface facet has no exact tested Field"))?;
            if !form.has_nonlinear_flux(field) {
                continue;
            }
            for witness in &trace.facets {
                let parent = witness.sides[side];
                let facet =
                    prepare_mesh_facet(mesh, form, field, witness.facet, parent, |_, _| {
                        Ok(vec![<S as From<f64>>::from(0.); layout.components])
                    })?;
                result.push((parent.entity.index(), facet));
            }
        }
    }
    Ok(result)
}
