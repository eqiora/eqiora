use super::*;
use crate::form_compiler::region::coordinates::{self, Real};
use crate::region_assembly::mapping::RegionDofMap;
use crate::spatial_expression::Coefficient;

impl CommonSpatialPlan {
    /// Construct initial coefficients in the selected Space. At time zero, source conditions
    /// fill unsupplied Fields; a Field cannot have both source and supplied data.
    /// At later times or for a stationary Plan, every Field requires explicit coefficients. Vertex values
    /// are nodal coefficients; cell values are cell-constant or hierarchical bubble
    /// coefficients according to the selected Space. Source data for cell constants
    /// is sampled at the cell centroid; bubble coefficients subtract the nodal interpolant.
    pub fn initial_state(
        &self,
        time_s: Option<f64>,
        fields: Vec<CommonInitialField>,
    ) -> Result<CommonState, Diagnostic> {
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        match self.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => {
                let (mapping, _) = self.state_assembly(equations, time_s)?;
                self.initial_state_from(equations, mapping, time_s, fields)
            }
            RecognizedNativeModel::ComplexSpatial(equations) => {
                let (mapping, _) = self.state_assembly(equations, time_s)?;
                self.initial_state_from(equations, mapping, time_s, fields)
            }
            _ => Err(invalid("missing spatial equations")),
        }
    }

    fn initial_state_from<S: Coefficient + Send + Sync>(
        &self,
        equations: &ExecutableSpatialEquations<S>,
        mapping: RegionDofMap<S>,
        time_s: Option<f64>,
        fields: Vec<CommonInitialField>,
    ) -> Result<CommonState, Diagnostic> {
        let inventory = self.physical_field_inventory(&mapping)?;
        let model = self.admission.model().digest()?;
        let mut seen = BTreeSet::new();
        let mut supplied = BTreeMap::new();
        for field in fields {
            if field.model() != &model || !seen.insert(field.field().erase()) {
                return Err(invalid(
                    "InitialField has foreign Model ownership or duplicates an exact Field",
                ));
            }
            let layout = inventory
                .get(&field.field().erase())
                .ok_or_else(|| invalid("InitialField is not an exact stored Plan Field"))?;
            if field.finite_value().is_some() {
                return Err(invalid(
                    "spatial InitialField cannot use finite coordinates",
                ));
            }
            let (vertices, cells): (Vec<_>, Vec<_>) = layout
                .coefficients
                .keys()
                .copied()
                .partition(|key| key.entity.dimension() == 0);
            if !cells.is_empty()
                && layout.space != Space::simplex_p1_bubble()
                && layout.space != Space::cell_constant()
            {
                return Err(invalid(
                    "cell initial coefficients require a selected cell-constant or bubble Space",
                ));
            }
            for (data, keys) in [(field.vertex(), vertices), (field.cell(), cells)] {
                if keys.is_empty() {
                    if data.is_some() {
                        return Err(invalid(
                            "InitialField contains an unselected coefficient association",
                        ));
                    }
                    continue;
                }
                let data = data.ok_or_else(|| {
                    invalid("InitialField omits a selected vertex or cell coefficient block")
                })?;
                if data.shape() != layout.value_type.shape() || data.values().len() != keys.len() {
                    return Err(invalid(
                        "InitialField shape or cardinality differs from its exact coefficient block",
                    ));
                }
                for (key, &(re, im)) in keys.into_iter().zip(data.values()) {
                    if layout.value_type.scalar_domain() == eqiora_core::ScalarDomain::Real
                        && im != 0.
                    {
                        return Err(invalid(
                            "real InitialField cannot contain imaginary coefficients",
                        ));
                    }
                    supplied.insert(
                        key,
                        S::from_parts(re, im).ok_or_else(|| {
                            invalid("InitialField scalar domain differs from its Plan")
                        })?,
                    );
                }
            }
        }
        let mut values = Vec::new();
        for key in inventory
            .values()
            .flat_map(|field| field.coefficients.keys())
            .copied()
        {
            let form = &equations
                .regions
                .iter()
                .find(|region| {
                    region
                        .form
                        .represented_fields()
                        .iter()
                        .any(|(field, _)| *field == key.field)
                })
                .ok_or_else(|| invalid("initial Field has no exact Region"))?
                .form;
            let source =
                if time_s == Some(0.0) {
                    let sample = |point: &[f64]| -> Result<Option<S>, Diagnostic> {
                        Ok(form
                            .initial_values_at(point)?
                            .get(&key.field)
                            .and_then(|components| components.get(key.component))
                            .copied())
                    };
                    match self.admission.resources() {
                        NativeMeshResources::Cartesian { mesh, .. }
                            if key.entity.dimension() == 0 =>
                        {
                            let point =
                                mesh.mesh().vertex_coordinates(key.entity).ok_or_else(|| {
                                    invalid("initial vertex is absent from its exact Mesh")
                                })?;
                            sample(&point)?
                        }
                        NativeMeshResources::Cartesian { mesh, .. }
                            if key.entity.dimension() == mesh.dimension()
                                && inventory[&key.field].space == Space::cell_constant() =>
                        {
                            use eqiora_meshing::MeshGeometry;
                            let map = mesh.mesh().geometry_map(key.entity).ok_or_else(|| {
                                invalid("initial cell is absent from its exact Mesh")
                            })?;
                            // Cell-constant initial coefficients sample the physical cell centroid.
                            sample(map.origin())?
                        }
                        NativeMeshResources::GmshSimplicial { mesh, .. }
                        | NativeMeshResources::AffineTriangleSimplicial { mesh, .. }
                            if key.entity.dimension() == 0 =>
                        {
                            let point = mesh.mesh().vertices().get(key.entity.index()).ok_or_else(
                                || invalid("initial vertex is absent from its exact Mesh"),
                            )?;
                            sample(point)?
                        }
                        NativeMeshResources::GmshSimplicial { mesh, .. }
                        | NativeMeshResources::AffineTriangleSimplicial { mesh, .. }
                            if key.entity.dimension() == mesh.dimension()
                                && inventory[&key.field].space == Space::simplex_p1_bubble() =>
                        {
                            let vertices =
                                mesh.mesh().entity_vertices(key.entity).ok_or_else(|| {
                                    invalid("initial bubble has no exact cell vertices")
                                })?;
                            let count = vertices.len() as f64;
                            let mut center = vec![0.0; mesh.dimension()];
                            for vertex in &vertices {
                                for (value, coordinate) in center
                                    .iter_mut()
                                    .zip(&mesh.mesh().vertices()[vertex.index()])
                                {
                                    *value += coordinate / count;
                                }
                            }
                            if let Some(center_value) = sample(&center)? {
                                // The normalized hierarchical bubble equals one at
                                // the barycenter; the P1 part already contributes
                                // the average of the vertex values there.
                                let mut nodal_value = S::zero();
                                for vertex in vertices {
                                    nodal_value += sample(&mesh.mesh().vertices()[vertex.index()])?
                                        .ok_or_else(|| {
                                            invalid("bubble source omitted a vertex value")
                                        })?
                                        / count;
                                }
                                Some(center_value - nodal_value)
                            } else {
                                None
                            }
                        }
                        _ => {
                            return Err(invalid(
                                "initial coefficient has an unsupported Mesh association",
                            ));
                        }
                    }
                } else {
                    None
                };
            let value = match (source, supplied.get(&key).copied()) {
                (Some(_), Some(_)) => {
                    return Err(invalid(
                        "InitialField duplicates a Model-owned initial condition",
                    ));
                }
                (Some(value), None) | (None, Some(value)) => value,
                (None, None) => return Err(invalid("initial State omits an exact stored Field")),
            };
            values.extend(coordinates::values::<S, Real>(
                &inventory[&key.field].value_type,
                &[value],
            )?);
        }
        self.scalar_state(time_s, values)
    }
}
