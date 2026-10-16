use super::*;

impl CommonSpatialPlan {
    /// Identity of the exact stationary numerical request, including its seed.
    pub fn run_identity(&self, initial: Option<&CommonState>) -> Result<String, Diagnostic> {
        if self.admission.temporal.is_some() {
            return Err(invalid(
                "stationary request identity requires a stationary Plan",
            ));
        }
        let mut bytes = Vec::new();
        push_framed(&mut bytes, self.identity().as_bytes());
        match initial {
            Some(state) => {
                self.require_static_state(state)?;
                bytes.push(1);
                push_framed(&mut bytes, state.identity().as_bytes());
            }
            None => bytes.push(0),
        }
        Ok(domain_separated_identity(
            b"eqiora.common-static-run-request/v1\0",
            &bytes,
        ))
    }

    fn require_static_state(&self, state: &CommonState) -> Result<(), Diagnostic> {
        if self.admission.temporal.is_some()
            || state.time_s().is_some()
            || state.state_space_identity() != self.identity()
        {
            return Err(invalid(
                "stationary execution requires a timeless State owned by its exact Plan",
            ));
        }
        Ok(())
    }

    pub(crate) fn static_initial_fields(
        &self,
        state: &CommonState,
    ) -> Result<BTreeMap<eqiora_core::RawId, Vec<f64>>, Diagnostic> {
        self.require_static_state(state)?;
        self.state_fields(state)?
            .map(|(id, _, _, values)| Ok((id.erase(), values.to_vec())))
            .collect()
    }

    pub(super) fn state_assembly(
        &self,
        time_s: Option<f64>,
    ) -> Result<
        (
            crate::region_assembly::mapping::RegionDofMap<f64>,
            crate::region_assembly::mapping::RegionSolveInput<f64>,
        ),
        Diagnostic,
    > {
        match (self.admission.temporal, time_s) {
            (Some(_), Some(time)) if time.is_finite() && time >= 0.0 => {
                self.scalar_assembly_at(time)
            }
            (None, None) => {
                if self.admission.nonlinear.is_some()
                    || matches!(self.admission.spatial, NativeSpatialPolicy::ScalarTpfa(_))
                {
                    return Err(invalid(
                        "stationary spatial State requires a linear nodal Plan",
                    ));
                }
                let RecognizedNativeModel::Spatial(equations) = self.admission.recognized_model()
                else {
                    return Err(invalid(
                        "stationary spatial State requires real nodal equations",
                    ));
                };
                equations.static_assembly(
                    self.admission.resources(),
                    &self.admission.discretizations,
                    &self.admission.system_scaling,
                    self.admission.operator_properties,
                )
            }
            _ => Err(invalid(
                "State model Time must match the Plan: absent for stationary, finite and nonnegative for transient",
            )),
        }
    }

    pub(crate) fn scalar_state(
        &self,
        time_s: Option<f64>,
        values: Vec<f64>,
    ) -> Result<CommonState, Diagnostic> {
        let (mapping, input) = self.state_assembly(time_s)?;
        let history = self.physical_fields(&mapping, &values)?;
        let keys = mapping.keys().collect::<Vec<_>>();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "spatial State requires complete finite nodal coefficients",
            ));
        }
        let prescribed = mapping.lift(&vec![0.0; mapping.free_count()], false)?;
        for (key, expected) in &input.prescribed_states {
            if history
                .get(&key.field)
                .and_then(|field| field.coefficients.get(key))
                != Some(expected)
            {
                return Err(invalid(
                    "spatial State contradicts prescribed displacement boundary values",
                ));
            }
        }
        for key in &keys {
            let value = &history[&key.field].coefficients[key];
            if mapping.free_dof(*key).is_none() {
                let (_, layout) = mapping.field_layout(key.field).expect("exact Field");
                let expected =
                    prescribed[mapping.global_dof(*key).expect("exact coordinate")] * layout.scale;
                if *value != expected {
                    return Err(invalid(
                        "scalar State contradicts prescribed boundary values",
                    ));
                }
            }
        }
        let RecognizedNativeModel::Spatial(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing spatial equations"));
        };
        let algebraic = equations
            .regions
            .iter()
            .flat_map(|region| region.form.algebraic_fields())
            .collect::<BTreeSet<_>>();
        if time_s.is_some() && !algebraic.is_empty() {
            match self.admission.resources() {
                NativeMeshResources::Cartesian { mesh, .. } => mapping.validate_algebraic_state(
                    mesh.mesh(),
                    input,
                    &history,
                    &algebraic,
                    self.linear(),
                )?,
                NativeMeshResources::GmshSimplicial { mesh, .. }
                | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => mapping
                    .validate_algebraic_state(
                        mesh.mesh(),
                        input,
                        &history,
                        &algebraic,
                        self.linear(),
                    )?,
                _ => return Err(invalid("algebraic State requires its exact nodal Mesh")),
            }
        }
        CommonState::new(
            self.identity().to_owned(),
            time_s,
            Arc::new(self.admission.model().clone()),
            Arc::new(self.admission.resources().clone()),
            CommonStateKind::Spatial(values.into_boxed_slice()),
        )
    }
}
