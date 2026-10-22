use super::*;
use crate::region_assembly::mapping::{
    RecoveredRegionField, RegionAlgebraicCoordinates, RegionDofMap, RegionSolveInput,
};
use crate::spatial_expression::Coefficient;
use eqiora_meshing::{AffineGeometryMap, MeshGeometry};

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
                        let (_, ty) = self
                            .fields()
                            .find(|(field, _)| field.erase() == *id)
                            .ok_or_else(|| invalid("complex Result contains an unowned Field"))?;
                        Ok((
                            *id,
                            crate::form_compiler::region::coordinates::Real::decode::<
                                num_complex::Complex64,
                            >(ty, values)?,
                        ))
                    })
                    .collect::<Result<BTreeMap<_, _>, Diagnostic>>()?;
                let system = equations.finalize_complex_static(
                    admission.linear.workers,
                    admission.linear.solver,
                    admission.resources(),
                    &admission.discretizations,
                    &admission.system_scaling,
                )?;
                match system {
                    regions::StaticSpatialSystem::Real(system) => {
                        system.validate_coefficients(&values)?;
                        system.validate_initial_residual(None, initial_residual_norm)
                    }
                    regions::StaticSpatialSystem::Complex(system) => {
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
        let CommonStateKind::Spatial {
            previous: Some(history),
            ..
        } = &current.kind
        else {
            return Err(invalid(
                "step replay requires its retained mathematical predecessor",
            ));
        };
        if history.grid.previous()? != previous.time_grid()?
            || Some(history.values.as_ref()) != previous.spatial_values()
        {
            return Err(invalid(
                "step replay differs from the State's mathematical predecessor",
            ));
        }
        if self.admission.nonlinear.is_some() != nonlinear_iterations.is_some() {
            return Err(invalid(
                "spatial iteration history differs from its Plan solve strategy",
            ));
        }
        match self.admission.recognized_model() {
            RecognizedNativeModel::Spatial(equations) => {
                self.reaccept_typed_step(equations, previous, current, nonlinear_iterations)
            }
            RecognizedNativeModel::ComplexSpatial(equations) => {
                self.reaccept_typed_step(equations, previous, current, nonlinear_iterations)
            }
            _ => Err(invalid("step replay requires spatial equations")),
        }
    }

    fn reaccept_typed_step<S: Coefficient + Send + Sync>(
        &self,
        equations: &ExecutableSpatialEquations<S>,
        previous: &CommonState,
        current: &CommonState,
        nonlinear_iterations: Option<usize>,
    ) -> Result<(f64, f64, usize, usize, Option<f64>), Diagnostic>
    where
        RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
    {
        let (mapping, input) =
            self.scalar_step_assembly(equations, previous, current.time_grid()?)?;
        let values = current
            .spatial_values()
            .ok_or_else(|| invalid("step replay requires physical spatial coefficients"))?;
        let fields = self.physical_fields(&mapping, values)?;
        self.accept_step_fields(&mapping, &input, &fields, nonlinear_iterations)
    }

    /// Reuse the original step equation for live acceptance and standalone replay.
    /// Iteration metadata is checked only when replaying a producer's Result.
    pub(super) fn accept_step_fields<S: Coefficient + Send + Sync>(
        &self,
        mapping: &RegionDofMap<S>,
        input: &RegionSolveInput<S>,
        fields: &BTreeMap<eqiora_core::RawId, RecoveredRegionField<S>>,
        nonlinear_iterations: Option<usize>,
    ) -> Result<(f64, f64, usize, usize, Option<f64>), Diagnostic>
    where
        RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
    {
        match self.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => {
                self.accept_step_on_mesh(mesh.mesh(), mapping, input, fields, nonlinear_iterations)
            }
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => {
                let current_mesh = input
                    .geometry_action
                    .as_ref()
                    .map(|action| action.current_mesh());
                self.accept_step_on_mesh(
                    current_mesh.unwrap_or_else(|| mesh.mesh()),
                    mapping,
                    input,
                    fields,
                    nonlinear_iterations,
                )
            }
            _ => Err(invalid("step replay requires its exact affine mesh")),
        }
    }

    fn accept_step_on_mesh<'mesh, S: Coefficient + Send + Sync>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        mapping: &RegionDofMap<S>,
        input: &RegionSolveInput<S>,
        fields: &BTreeMap<eqiora_core::RawId, RecoveredRegionField<S>>,
        nonlinear_iterations: Option<usize>,
    ) -> Result<(f64, f64, usize, usize, Option<f64>), Diagnostic>
    where
        RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
    {
        match self.admission.nonlinear {
            Some(nonlinear) => {
                let initial = input.previous.as_ref().expect("validated step history");
                if let Some(iterations) = nonlinear_iterations {
                    mapping.validate_nonlinear_coefficients(
                        mesh, input, fields, initial, nonlinear, iterations,
                    )
                } else {
                    let accepted = mapping
                        .accept_nonlinear_coefficients(mesh, input, fields, initial, nonlinear)?;
                    Ok((
                        accepted.residual_norm,
                        accepted.residual_target,
                        accepted.assembly.report.packet_count(),
                        accepted.assembly.report.target_count(),
                        Some(accepted.initial.residual_norm),
                    ))
                }
            }
            None => mapping
                .validate_step_coefficients(
                    mesh,
                    input,
                    fields,
                    self.admission.linear.solver,
                    self.admission.linear.workers,
                )
                .map(|(n, t, p, a)| (n, t, p, a, None)),
        }
    }
}
