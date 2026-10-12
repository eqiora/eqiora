use super::*;

impl CommonSpatialPlan {
    /// Construct exact nodal initial assignments. At time zero, source conditions
    /// fill unsupplied Fields; a Field cannot have both source and supplied data.
    /// At later times, every Field requires explicit coefficients. Vertex values
    /// are nodal coefficients; cell values are the selected hierarchical bubble
    /// coefficients, not samples at cell centers.
    pub fn initial_state(
        &self,
        time_s: f64,
        fields: Vec<CommonInitialField>,
    ) -> Result<CommonState, Diagnostic> {
        if self.admission.temporal.is_none() || !time_s.is_finite() || time_s < 0.0 {
            return Err(invalid(
                "spatial initial State requires a transient Plan and finite nonnegative Time",
            ));
        }
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        let RecognizedNativeModel::Spatial(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing spatial equations"));
        };
        let (mapping, _) = self.scalar_assembly_at(time_s)?;
        let inventory = self.history_inventory(&mapping)?;
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
                return Err(invalid("nodal InitialField cannot use finite coordinates"));
            }
            let (vertices, cells): (Vec<_>, Vec<_>) = layout
                .coefficients
                .keys()
                .copied()
                .partition(|key| key.entity.dimension() == 0);
            if !cells.is_empty() && layout.space != Space::simplex_p1_bubble() {
                return Err(invalid(
                    "cell initial coefficients require the selected bubble Space",
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
                    invalid("InitialField omits a selected vertex or bubble coefficient block")
                })?;
                if data.shape() != layout.value_type.shape() || data.values().len() != keys.len() {
                    return Err(invalid(
                        "InitialField shape or cardinality differs from its exact coefficient block",
                    ));
                }
                supplied.extend(keys.into_iter().zip(data.values().iter().copied()));
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
            let source = if time_s == 0.0 {
                let sample = |point: &[f64]| -> Result<Option<f64>, Diagnostic> {
                    Ok(form
                        .initial_values_at(point)?
                        .get(&key.field)
                        .and_then(|components| components.get(key.component))
                        .copied())
                };
                match self.admission.resources() {
                    NativeMeshResources::Cartesian { mesh, .. } if key.entity.dimension() == 0 => {
                        let point =
                            mesh.mesh().vertex_coordinates(key.entity).ok_or_else(|| {
                                invalid("initial vertex is absent from its exact Mesh")
                            })?;
                        sample(&point)?
                    }
                    NativeMeshResources::GmshSimplicial { mesh, .. }
                    | NativeMeshResources::AdjacentPartitionSimplicial { mesh, .. }
                        if key.entity.dimension() == 0 =>
                    {
                        let point =
                            mesh.mesh()
                                .vertices()
                                .get(key.entity.index())
                                .ok_or_else(|| {
                                    invalid("initial vertex is absent from its exact Mesh")
                                })?;
                        sample(point)?
                    }
                    NativeMeshResources::GmshSimplicial { mesh, .. }
                    | NativeMeshResources::AdjacentPartitionSimplicial { mesh, .. }
                        if key.entity.dimension() == mesh.dimension()
                            && inventory[&key.field].space == Space::simplex_p1_bubble() =>
                    {
                        let vertices = mesh
                            .mesh()
                            .entity_vertices(key.entity)
                            .ok_or_else(|| invalid("initial bubble has no exact cell vertices"))?;
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
                            let mut nodal_value = 0.0;
                            for vertex in vertices {
                                nodal_value +=
                                    sample(&mesh.mesh().vertices()[vertex.index()])?.ok_or_else(
                                        || invalid("bubble source omitted a vertex value"),
                                    )? / count;
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
            values.push(value);
        }
        self.scalar_state(time_s, values)
    }
}
