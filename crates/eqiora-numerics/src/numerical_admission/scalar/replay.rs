use super::*;

impl CommonSpatialPlan {
    pub(crate) fn reaccept_static_coefficients(
        &self,
        values: &BTreeMap<eqiora_core::RawId, Vec<f64>>,
        nullspace: Option<&crate::nullspace::NullspaceEvidence>,
        initial: Option<&CommonState>,
        initial_residual_norm: f64,
    ) -> Result<(), Diagnostic> {
        if self.admission.temporal.is_some()
            || values.keys().copied().collect::<BTreeSet<_>>()
                != self.fields().map(|(id, _)| id.erase()).collect()
        {
            return Err(invalid(
                "static Result requires its exact complete Field inventory",
            ));
        }
        let initial = initial
            .map(|state| self.static_initial_fields(state))
            .transpose()?;
        let first = self
            .fields()
            .next()
            .ok_or_else(|| invalid("static Result omits its Fields"))?
            .0
            .erase();
        self.check_nullspace_evidence(&values[&first], nullspace)?;
        // Constrained TPFA already rechecks the original equation, reference,
        // multiplier and gauge through its existing nullspace acceptance owner.
        if nullspace.is_some() {
            return Ok(());
        }
        let admission = &self.admission;
        match admission.recognized_model() {
            RecognizedNativeModel::Coordinates(equations) => {
                coordinate_grid::reaccept(admission, equations, values)
            }
            RecognizedNativeModel::Spatial(equations) => {
                let system = equations.finalize_static(
                    admission.linear.workers,
                    admission.linear.solver,
                    admission.resources(),
                    &admission.discretizations,
                    &admission.system_scaling,
                    admission.operator_properties,
                )?;
                system.validate_coefficients(values)?;
                system.validate_initial_residual(initial.as_ref(), initial_residual_norm)
            }
            RecognizedNativeModel::ComplexSpatial(equations) => {
                let values = values
                    .iter()
                    .map(|(id, values)| {
                        let (pairs, remainder) = values.as_chunks::<2>();
                        if !remainder.is_empty() {
                            return Err(invalid("complex Result omits a coefficient component"));
                        }
                        Ok((
                            *id,
                            pairs
                                .iter()
                                .map(|[re, im]| num_complex::Complex64::new(*re, *im))
                                .collect(),
                        ))
                    })
                    .collect::<Result<BTreeMap<_, _>, Diagnostic>>()?;
                let system = equations.finalize_complex_static(
                    admission.linear.workers,
                    admission.linear.solver,
                    admission.resources(),
                    &admission.discretizations,
                    &admission.system_scaling,
                    admission.operator_properties,
                )?;
                match system {
                    regions::ComplexSpatialSystem::Real(system) => {
                        system.validate_coefficients(&values)?;
                        system.validate_initial_residual(None, initial_residual_norm)
                    }
                    regions::ComplexSpatialSystem::Complex(system) => {
                        system.validate_coefficients(&values)?;
                        system.validate_initial_residual(None, initial_residual_norm)
                    }
                }
            }
            _ => Err(invalid(
                "static linear Result requires its exact admitted spatial equations",
            )),
        }
    }

    pub(crate) fn reaccept_spatial_step(
        &self,
        previous: &CommonState,
        current: &CommonState,
        nonlinear_iterations: Option<usize>,
    ) -> Result<(f64, f64, usize, usize, Option<f64>), Diagnostic> {
        if current.state_space_identity() != self.identity() {
            return Err(invalid(
                "step replay requires its exact current Plan checkpoint",
            ));
        }
        let (mapping, input) = self.scalar_step_assembly(previous, current.require_time_s()?)?;
        let values = current
            .spatial_values()
            .ok_or_else(|| invalid("step replay requires physical spatial coefficients"))?;
        let fields = self.physical_fields(&mapping, values)?;
        let solver = self.admission.linear.solver;
        let workers = self.admission.linear.workers;
        if self.admission.nonlinear.is_some() != nonlinear_iterations.is_some() {
            return Err(invalid(
                "spatial iteration history differs from its Plan solve strategy",
            ));
        }
        let validate = |mesh: &eqiora_meshing::SimplicialMesh| match self.admission.nonlinear {
            Some(nonlinear) => mapping.validate_nonlinear_step_coefficients(
                mesh,
                &input,
                &fields,
                nonlinear,
                nonlinear_iterations.expect("validated history"),
            ),
            None => mapping
                .validate_step_coefficients(mesh, &input, &fields, solver, workers)
                .map(|(n, t, p, a)| (n, t, p, a, None)),
        };
        match self.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => match self.admission.nonlinear {
                Some(nonlinear) => mapping.validate_nonlinear_step_coefficients(
                    mesh.mesh(),
                    &input,
                    &fields,
                    nonlinear,
                    nonlinear_iterations.expect("validated history"),
                ),
                None => mapping
                    .validate_step_coefficients(mesh.mesh(), &input, &fields, solver, workers)
                    .map(|(n, t, p, a)| (n, t, p, a, None)),
            },
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => {
                let current_mesh = input
                    .geometry_action
                    .as_ref()
                    .map(|action| action.current_mesh());
                validate(current_mesh.unwrap_or_else(|| mesh.mesh()))
            }
            _ => Err(invalid("step replay requires its exact affine mesh")),
        }
    }
}
