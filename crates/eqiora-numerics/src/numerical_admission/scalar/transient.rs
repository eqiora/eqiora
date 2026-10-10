use super::*;

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
        let (mapping, _) = self.scalar_assembly()?;
        let keys = mapping.keys().collect::<Vec<_>>();
        if values.len() != keys.len() || values.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "scalar State requires a transient Plan and complete finite nodal coefficients",
            ));
        }
        let prescribed = mapping.lift(&vec![0.0; mapping.free_count()], false)?;
        for (key, value) in keys.iter().zip(&values) {
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
        let (mapping, _) = self.scalar_assembly()?;
        let mut values = Vec::new();
        for key in mapping.keys() {
            let point = match self.admission.resources() {
                NativeMeshResources::Cartesian { mesh, .. } => {
                    mesh.mesh().vertex_coordinates(key.entity)
                }
                NativeMeshResources::GmshSimplicial { mesh, .. } => {
                    mesh.mesh().vertices().get(key.entity.index()).cloned()
                }
                _ => None,
            }
            .ok_or_else(|| {
                invalid("scalar initial coordinate is absent from its exact nodal Mesh")
            })?;
            let initial = form.initial_values_at(&point)?;
            values.push(
                *initial.get(&key.field).ok_or_else(|| {
                    invalid("scalar initial equation omits an exact stored Field")
                })?,
            );
        }
        let state = self.scalar_state(0.0, values)?;
        Ok(state)
    }

    pub(super) fn scalar_assembly(
        &self,
    ) -> Result<
        (
            crate::region_assembly::mapping::RegionDofMap<f64>,
            crate::region_assembly::mapping::RegionSolveInput<f64>,
        ),
        Diagnostic,
    > {
        let temporal = self
            .admission
            .temporal
            .ok_or_else(|| invalid("scalar storage requires BackwardEuler"))?;
        let RecognizedNativeModel::Linear(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar equations"));
        };
        if equations
            .single()?
            .form
            .boundary_laws()
            .values()
            .flat_map(|laws| laws.values())
            .any(|law| law.quantity != crate::canonical_boundary::PhysicalBoundaryQuantity::Trace)
        {
            return Err(invalid(
                "scalar storage requires complete essential boundary data",
            ));
        }
        let mut bound = equations.clone();
        for region in &mut bound.regions {
            region.form = region.form.bind_backward_euler(temporal.step())?;
        }
        match self.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => bound.cartesian_assembly(mesh.mesh()),
            NativeMeshResources::GmshSimplicial { mesh, .. } => {
                let (mapping, forms, natural) = bound.simplicial_assembly(
                    mesh,
                    Space::continuous_lagrange(std::num::NonZeroU16::MIN),
                )?;
                Ok((
                    mapping,
                    crate::region_assembly::mapping::RegionSolveInput {
                        forms,
                        natural,
                        previous: None,
                    },
                ))
            }
            _ => Err(invalid("scalar storage requires its exact nodal Mesh")),
        }
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
        let (mapping, mut input) = self.scalar_assembly()?;
        let RecognizedNativeModel::Linear(bound) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar equations"));
        };
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
        let request = LinearSolveRequest::new(&checked, self.admission.linear.solver);
        let output = match self.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => mapping.solve(
                mesh.mesh(),
                input,
                self.admission.linear.workers,
                request,
                |reactions, values| reactions.recover(values),
            )?,
            NativeMeshResources::GmshSimplicial { mesh, .. } => mapping.solve(
                mesh.mesh(),
                input,
                self.admission.linear.workers,
                request,
                |reactions, values| reactions.recover(values),
            )?,
            _ => return Err(invalid("scalar storage requires its exact nodal Mesh")),
        };
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
    /// Scalar nodal coefficients retained by this exact spatial State.
    pub fn scalar_values(&self) -> Option<&[f64]> {
        match &self.kind {
            CommonStateKind::Scalar(values) => Some(values),
            _ => None,
        }
    }
}
