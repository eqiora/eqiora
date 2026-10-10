use super::*;

impl CommonLinearPlan {
    /// Construct exact nodal initial assignments. At time zero, source conditions
    /// fill unsupplied Fields; a Field cannot have both source and supplied data.
    /// At later times, every Field requires explicit coefficients.
    pub fn initial_state(
        &self,
        time_s: f64,
        fields: Vec<CommonInitialField>,
    ) -> Result<CommonState, Diagnostic> {
        if self.admission.temporal.is_none() || !time_s.is_finite() || time_s < 0.0 {
            return Err(invalid(
                "linear initial State requires a transient Plan and finite nonnegative Time",
            ));
        }
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        let RecognizedNativeModel::Linear(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing linear equations"));
        };
        let (mapping, _) = self.scalar_assembly_at(time_s)?;
        let model = self.admission.model().digest()?;
        let mut seen = BTreeSet::new();
        let mut supplied = BTreeMap::new();
        for field in fields {
            if field.model() != &model || !seen.insert(field.field().erase()) {
                return Err(invalid(
                    "InitialField has foreign Model ownership or duplicates an exact Field",
                ));
            }
            let (_, layout) = mapping
                .field_layout(field.field().erase())
                .ok_or_else(|| invalid("InitialField is not an exact stored Plan Field"))?;
            let data = field
                .vertex()
                .ok_or_else(|| invalid("nodal InitialField requires vertex values"))?;
            let keys = mapping
                .keys()
                .filter(|key| key.field == field.field().erase())
                .collect::<Vec<_>>();
            if field.cell().is_some()
                || field.finite_value().is_some()
                || data.shape() != layout.value_type.shape()
                || data.values().len() != keys.len()
            {
                return Err(invalid(
                    "InitialField shape, association or cardinality differs from its exact nodal Field",
                ));
            }
            supplied.extend(keys.into_iter().zip(data.values().iter().copied()));
        }
        let mut values = Vec::new();
        for key in mapping.keys() {
            let form = &equations
                .regions
                .iter()
                .find(|region| {
                    region
                        .form
                        .fields()
                        .iter()
                        .any(|(field, _)| *field == key.field)
                })
                .ok_or_else(|| invalid("initial Field has no exact Region"))?
                .form;
            let source = if time_s == 0.0 {
                let point = match self.admission.resources() {
                    NativeMeshResources::Cartesian { mesh, .. } => {
                        mesh.mesh().vertex_coordinates(key.entity)
                    }
                    NativeMeshResources::GmshSimplicial { mesh, .. } => {
                        mesh.mesh().vertices().get(key.entity.index()).cloned()
                    }
                    _ => None,
                }
                .ok_or_else(|| invalid("initial coordinate is absent from its exact nodal Mesh"))?;
                form.initial_values_at(&point)?
                    .get(&key.field)
                    .and_then(|components| components.get(key.component))
                    .copied()
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
