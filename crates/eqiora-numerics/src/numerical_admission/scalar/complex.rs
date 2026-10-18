//! Complex spatial Plan admission uses the shared equation and realization owners.
use super::*;

impl CommonSpatialPlan {
    pub(in crate::numerical_admission) fn from_complex_admission(
        model: &ModelEnvelope,
        admission: NativeNumericalAdmission,
        selection: FormulationSelectionMode,
        authored: Option<&AuthoredFormulationProjection>,
    ) -> Result<Self, Diagnostic> {
        let RecognizedNativeModel::ComplexSpatial(equations) = admission.recognized_model() else {
            return Err(invalid(
                "complex spatial Plan requires its exact typed equations",
            ));
        };
        let fields = equations
            .fields()
            .into_iter()
            .map(|(field, ty)| (field.downcast().expect("compiled Field identity"), ty))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let derived = equations.primal_form(admission.program())?;
        let formulation = if matches!(admission.spatial, NativeSpatialPolicy::ScalarTpfa(_)) {
            if authored.is_some() || selection != FormulationSelectionMode::Automatic {
                return Err(invalid(
                    "complex TPFA Formulation has no admitted authored source correspondence",
                ));
            }
            None
        } else if let Some(description) = equations.mixed_formulation(selection) {
            if authored.is_some() {
                return Err(invalid(
                    "authored mixed Region formulation requires an admitted source correspondence",
                ));
            }
            Some(description)
        } else {
            match derived {
                Some(derived) => {
                    if let Some(authored) = authored {
                        crate::form_compiler::admit_authored_scalar_primal_form(
                            authored,
                            admission.program(),
                            &derived,
                        )?;
                    }
                    let (kind, boundary, rules) = derived.formulation_description();
                    let mut description = describe_primal(
                        kind,
                        boundary,
                        rules,
                        if authored.is_some() {
                            FormulationSelectionMode::Authored
                        } else {
                            selection
                        },
                    );
                    description.requested_source_identity =
                        authored.map(|form| form.source_identity().to_owned());
                    Some(description)
                }
                None if authored.is_none() && selection == FormulationSelectionMode::Automatic => {
                    None
                }
                None => {
                    return Err(invalid(
                        "complex scalar Formulation has no admitted source correspondence",
                    ));
                }
            }
        };
        let portable = resolve_common_spatial_portable(&admission, equations)?;
        Self::finish_admission(
            model,
            admission,
            fields,
            portable,
            formulation,
            authored.cloned(),
        )
    }

    pub(super) fn reauthenticate_complex(&self) -> Result<(), Diagnostic> {
        if self.harmonic.is_some() {
            return self.reauthenticate_harmonic();
        }
        self.admission.revalidate()?;
        let replayed = Self::from_complex_admission(
            self.admission.model(),
            self.admission.clone(),
            self.formulation
                .as_ref()
                .map_or(FormulationSelectionMode::Automatic, |form| form.requested()),
            self.authored_formulation.as_ref(),
        )?;
        if replayed != *self {
            return Err(invalid(
                "complex linear Plan changed during exact internal replay",
            ));
        }
        Ok(())
    }
}

impl NativeNumericalAdmission {
    pub(in crate::numerical_admission) fn execute_complex_linear(
        &self,
        equations: &ExecutableSpatialEquations<num_complex::Complex64>,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonLinearRunOutput<f64>, Diagnostic> {
        self.revalidate()?;
        let structure = equations.algebraic_structure(None)?;
        let system = equations.finalize_complex_static(
            self.linear.workers,
            self.linear.solver,
            self.resources(),
            &self.discretizations,
            &self.system_scaling,
            self.operator_properties,
        )?;
        let output = match system {
            regions::ComplexSpatialSystem::Real(system) => {
                let checked = self.linear.checked_backend(backend, Some(&structure))?;
                system.solve(LinearSolveRequest::new(&checked, self.linear.solver), None)?
            }
            regions::ComplexSpatialSystem::Complex(system) => {
                let checked = self
                    .linear
                    .checked_complex_backend(backend, Some(&structure))?;
                system.solve(LinearSolveRequest::new(&checked, self.linear.solver), None)?
            }
        };
        Ok(CommonLinearRunOutput {
            fields: output
                .fields
                .into_iter()
                .map(|(field, ty, values, space)| {
                    (
                        field,
                        ty,
                        values
                            .into_iter()
                            .flat_map(|value| [value.re, value.im])
                            .collect(),
                        space,
                    )
                })
                .collect(),
            solve_report: output.solve_report,
            assembly_report: output.assembly_report,
            nullspace: output.nullspace,
        })
    }
}
