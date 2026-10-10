use super::*;
use eqiora_meshing::MeshEntity;

impl CommonLinearPlan {
    /// Exact support of the admitted single scalar storage Field.
    pub fn storage_domain_id(&self) -> Result<String, Diagnostic> {
        let RecognizedNativeModel::Linear(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar equations"));
        };
        Ok(equations.single()?.form.domain().ulid().to_string())
    }

    pub(crate) fn scalar_state(
        &self,
        time_s: f64,
        values: Vec<f64>,
    ) -> Result<CommonState, Diagnostic> {
        let count = self
            .cartesian_cells()?
            .iter()
            .map(|n| n + 1)
            .product::<usize>()
            * self.fields.len();
        if self.admission.temporal.is_none()
            || values.len() != count
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "scalar State requires a transient Plan and complete finite Q1 coefficients",
            ));
        }
        let RecognizedNativeModel::Linear(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar equations"));
        };
        let region = equations.single()?;
        let NativeMeshResources::Cartesian { mesh, .. } = self.admission.resources() else {
            return Err(invalid("missing Cartesian Mesh"));
        };
        let field = region.form.fields()[0].0;
        for (&(axis, side), boundary) in &region.cartesian()?.boundaries {
            let law = &region.form.boundary_laws()[&field][boundary];
            if law.quantity != crate::canonical_boundary::PhysicalBoundaryQuantity::Trace {
                return Err(invalid(
                    "scalar storage requires complete essential boundary data",
                ));
            }
            let coordinate =
                region.cartesian()?.bounds[axis][usize::from(side == BoundarySide::Upper)];
            for (index, value) in values.iter().enumerate() {
                let point = mesh
                    .mesh()
                    .vertex_coordinates(MeshEntity::new(0, index))
                    .expect("validated vertex count");
                if point[axis] == coordinate && *value != law.evaluate(&point, &[])?[0] {
                    return Err(invalid(
                        "scalar State contradicts prescribed boundary values",
                    ));
                }
            }
        }
        CommonState::new(
            self.identity().to_owned(),
            time_s,
            Arc::new(self.admission.model().clone()),
            Arc::new(self.admission.resources().clone()),
            CommonStateKind::Scalar(values.into_boxed_slice()),
        )
    }

    /// Initialize the exact scalar storage from its consumed source conditions.
    pub fn initial_state(&self) -> Result<CommonState, Diagnostic> {
        if self.admission.temporal.is_none() {
            return Err(invalid("steady scalar Plan owns no initial State"));
        }
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        let RecognizedNativeModel::Linear(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar equations"));
        };
        let form = &equations.single()?.form;
        let NativeMeshResources::Cartesian { mesh, .. } = self.admission.resources() else {
            return Err(invalid(
                "scalar transient initialization requires its exact Cartesian Mesh",
            ));
        };
        let mesh = mesh.mesh();
        let vertex_count = mesh
            .entity_count(0)
            .ok_or_else(|| invalid("scalar Cartesian Mesh has no vertex inventory"))?;
        let mut values = Vec::with_capacity(vertex_count * self.fields.len());
        for (field, _) in self.fields.iter() {
            for index in 0..vertex_count {
                let point = mesh
                    .vertex_coordinates(MeshEntity::new(0, index))
                    .ok_or_else(|| invalid("scalar Cartesian vertex coordinate is absent"))?;
                let initial = form.initial_values_at(&point)?;
                let value = initial.get(&field.erase()).ok_or_else(|| {
                    invalid("scalar initial equation omits an exact stored Field")
                })?;
                values.push(*value);
            }
        }
        let state = self.scalar_state(0.0, values)?;
        Ok(state)
    }

    fn scalar_step_assembly(
        &self,
        state: &CommonState,
    ) -> Result<
        (
            crate::region_assembly::mapping::RegionDofMap<f64>,
            crate::region_assembly::mapping::RegionSolveInput<f64>,
        ),
        Diagnostic,
    > {
        if state.state_space_identity() != self.identity() {
            return Err(invalid("scalar State belongs to a foreign exact Plan"));
        }
        let CommonStateKind::Scalar(values) = &state.kind else {
            return Err(invalid("scalar Run requires scalar State"));
        };
        let temporal = self
            .admission
            .temporal
            .ok_or_else(|| invalid("scalar storage requires BackwardEuler"))?;
        let RecognizedNativeModel::Linear(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar equations"));
        };
        let NativeMeshResources::Cartesian { mesh, .. } = self.admission.resources() else {
            return Err(invalid("missing Cartesian Mesh"));
        };
        let mut bound = equations.clone();
        for region in &mut bound.regions {
            region.form = region.form.bind_backward_euler(temporal.step())?;
        }
        let (mapping, mut input) = bound.cartesian_assembly(mesh.mesh())?;
        let keys = mapping.keys().collect::<Vec<_>>();
        if keys.len() != values.len() {
            return Err(invalid(
                "scalar history differs from the exact mapped coefficient inventory",
            ));
        }
        let coefficients = keys
            .into_iter()
            .zip(values.iter().copied())
            .collect::<BTreeMap<_, _>>();
        input.previous = Some(
            bound
                .fields()
                .into_iter()
                .map(|(field, value_type)| {
                    let (domain, layout) = mapping.field_layout(field).expect("bound Field layout");
                    (
                        field,
                        crate::region_assembly::mapping::RecoveredRegionField {
                            domain,
                            value_type,
                            space: layout.space,
                            coefficients: coefficients
                                .iter()
                                .filter(|(key, _)| key.field == field)
                                .map(|(&key, &value)| (key, value))
                                .collect(),
                        },
                    )
                })
                .collect(),
        );
        Ok((mapping, input))
    }

    pub(in crate::numerical_admission) fn advance_scalar(
        &self,
        state: &CommonState,
        backend: &dyn LinearSolverBackend,
        next_time_s: f64,
    ) -> Result<CommonState, Diagnostic> {
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        let RecognizedNativeModel::Linear(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar inventory"));
        };
        let structure = equations.algebraic_structure(None)?;
        let checked = self
            .admission
            .linear
            .checked_backend(backend, Some(&structure))?;
        let (mapping, input) = self.scalar_step_assembly(state)?;
        let NativeMeshResources::Cartesian { mesh, .. } = self.admission.resources() else {
            return Err(invalid("missing Cartesian Mesh"));
        };
        let output = mapping.solve(
            mesh.mesh(),
            input,
            self.admission.linear.workers,
            LinearSolveRequest::new(&checked, self.admission.linear.solver),
            |reactions, values| reactions.recover(values),
        )?;
        let values = output
            .fields
            .into_values()
            .flat_map(|field| field.coefficients.into_values())
            .collect();
        self.scalar_state(next_time_s, values)
    }
}
impl ResolvedCommonPlan {
    pub(crate) fn spatial_state_space_identity(&self) -> Result<String, Diagnostic> {
        match self {
            Self::Linear(plan) if plan.admission.temporal.is_some() => {
                Ok(plan.identity().to_owned())
            }
            Self::TransientFlow(plan) => Ok(plan.state_space_identity()),
            _ => Err(invalid(
                "Plan does not own a scalar/flow transient state space",
            )),
        }
    }
}

impl CommonState {
    /// Scalar Q1 coefficients retained by this exact spatial State.
    pub fn scalar_values(&self) -> Option<&[f64]> {
        match &self.kind {
            CommonStateKind::Scalar(values) => Some(values),
            _ => None,
        }
    }
}
