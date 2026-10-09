//! Complex spatial Plan admission uses the shared equation and realization owners.
use super::*;

impl CommonScalarPlan {
    pub(in crate::numerical_admission) fn from_complex_admission(
        model: &ModelEnvelope,
        admission: NativeNumericalAdmission,
        selection: FormulationSelectionMode,
        authored: Option<&AuthoredFormulationProjection>,
    ) -> Result<Self, Diagnostic> {
        let RecognizedNativeModel::ComplexScalar(equations) = admission.recognized_model() else {
            return Err(invalid(
                "complex scalar Plan requires its exact typed equations",
            ));
        };
        let NativeMeshResources::Cartesian {
            mesh, production, ..
        } = admission.resources()
        else {
            return Err(invalid(
                "complex scalar Plan requires authenticated Cartesian resources",
            ));
        };
        let cells = production
            .cartesian_cells()
            .ok_or_else(|| invalid("complex scalar Plan lost Cartesian production"))?
            .cells()
            .to_vec()
            .into_boxed_slice();
        let fields = equations
            .fields()
            .into_iter()
            .map(|(field, ty)| (field.downcast().expect("compiled Field identity"), ty))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let derived = equations.primal_form(admission.program())?;
        let formulation = match derived {
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
            None if authored.is_none() && selection == FormulationSelectionMode::Automatic => None,
            None => {
                return Err(invalid(
                    "complex scalar Formulation has no admitted source correspondence",
                ));
            }
        };
        let portable = resolve_common_scalar_portable(&admission, equations, mesh, &cells)?;
        Self::finish_admission(
            model,
            admission,
            cells,
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
                "complex scalar Plan changed during exact internal replay",
            ));
        }
        Ok(())
    }
}

impl NativeNumericalAdmission {
    pub(in crate::numerical_admission) fn execute_complex_scalar(
        &self,
        equations: &ExecutableScalarEquations<num_complex::Complex64>,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonScalarRunOutput<f64>, Diagnostic> {
        self.revalidate()?;
        let NativeMeshResources::Cartesian { mesh, .. } = self.resources() else {
            return Err(invalid("complex scalar Run lost Cartesian resources"));
        };
        let structure = equations.algebraic_structure(None)?;
        let checked = self
            .linear
            .checked_complex_backend(backend, Some(&structure))?;
        let output = equations.execute(
            self.linear.workers,
            LinearSolveRequest::new(&checked, self.linear.solver),
            mesh.mesh(),
            |reactions, values| reactions.recover(values),
        )?;
        Ok(CommonScalarRunOutput {
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
