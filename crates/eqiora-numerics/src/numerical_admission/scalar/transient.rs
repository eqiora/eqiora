use super::*;
use eqiora_meshing::MeshEntity;

impl CommonScalarPlan {
    /// Exact support of the admitted single scalar storage Field.
    pub fn storage_domain_id(&self) -> Result<String, Diagnostic> {
        let RecognizedNativeModel::Scalar(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar equations"));
        };
        Ok(equations.single()?.form.domain().ulid().to_string())
    }

    pub(crate) fn scalar_state(
        &self,
        time_s: f64,
        values: Vec<f64>,
    ) -> Result<CommonState, Diagnostic> {
        let count = self.cells.iter().map(|n| n + 1).product::<usize>() * self.fields.len();
        if self.admission.temporal.is_none()
            || values.len() != count
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "scalar State requires a transient Plan and complete finite Q1 coefficients",
            ));
        }
        let RecognizedNativeModel::Scalar(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar equations"));
        };
        let region = equations.single()?;
        let NativeMeshResources::Cartesian { mesh, .. } = self.admission.resources() else {
            return Err(invalid("missing Cartesian Mesh"));
        };
        let field = region.form.fields()[0].0;
        for (&(axis, side), boundary) in &region.boundaries {
            let law = &region.form.boundary_laws()[&field][boundary];
            if law.quantity != crate::canonical_boundary::PhysicalBoundaryQuantity::Trace {
                return Err(invalid(
                    "scalar storage requires complete essential boundary data",
                ));
            }
            let coordinate = region.bounds[axis][usize::from(side == BoundarySide::Upper)];
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
        let RecognizedNativeModel::Scalar(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar equations"));
        };
        let initial = equations.single()?.form.initial_values()?;
        let count = self.cells.iter().map(|n| n + 1).product::<usize>();
        let values = self
            .fields
            .iter()
            .flat_map(|(field, _)| std::iter::repeat_n(initial[&field.erase()], count))
            .collect::<Vec<_>>();
        let state = self.scalar_state(0.0, values)?;
        Ok(state)
    }

    fn scalar_step_assembly(
        &self,
        state: &CommonState,
    ) -> Result<crate::cartesian_elliptic::linear::CartesianLinearAssembly, Diagnostic> {
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
        let RecognizedNativeModel::Scalar(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar equations"));
        };
        let region = equations.single()?;
        let NativeMeshResources::Cartesian { mesh, .. } = self.admission.resources() else {
            return Err(invalid("missing Cartesian Mesh"));
        };
        let form = region.form.bind_backward_euler(temporal.step())?;
        crate::cartesian_elliptic::linear::CartesianLinearAssembly::assemble_backward_euler(
            &form,
            mesh.mesh(),
            &QuadratureRule::tensor_product_gauss_legendre(mesh.dimension(), 2)?,
            &REFERENCE_ASSEMBLY_BACKEND,
            &region.boundaries,
            values,
        )
    }

    pub(in crate::numerical_admission) fn advance_scalar(
        &self,
        state: &CommonState,
        backend: &dyn LinearSolverBackend,
        next_time_s: f64,
    ) -> Result<CommonState, Diagnostic> {
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        let RecognizedNativeModel::Scalar(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar inventory"));
        };
        let structure = equations.algebraic_structure()?;
        let checked = self
            .admission
            .linear
            .checked_backend(backend, Some(&structure))?;
        let assembly = self.scalar_step_assembly(state)?;
        let canonical = Arc::new(eqiora_solver::CanonicalCsrSystemView::new(
            &assembly.system,
            LinearOperatorProperties::General,
        )?);
        let request = LinearSolveRequest::new(&checked, self.admission.linear.solver);
        let core = crate::finalized_spatial::FinalizedLinearCore::new(
            request.plan(),
            VectorLayoutKind::Replicated,
            Target::HostCpu {
                threads: self.admission.linear.workers,
            },
            canonical,
        );
        let solution = request.solve(&core.linear_problem()?)?;
        core.validate_solution(&solution)?;
        let (values, _) = solution.into_parts();
        let values = assembly.constraints.lift(&values)?;
        self.scalar_state(next_time_s, values)
    }
}
impl ResolvedCommonPlan {
    pub(crate) fn spatial_state_space_identity(&self) -> Result<String, Diagnostic> {
        match self {
            Self::Scalar(plan) if plan.admission.temporal.is_some() => {
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
