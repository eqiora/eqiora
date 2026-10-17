//! One exact spatial assembly and field reconstruction for execution and replay.
use super::*;
use crate::canonical::ScalarEllipticCartesianModel;
use crate::cartesian_elliptic::CartesianPoint;
use eqiora_core::{Id, entity::kinds};
use eqiora_meshing::CartesianMesh;

impl CommonSpatialPlan {
    /// Accept one selected Parameter point through this Plan's exact supplied Mesh and policies.
    /// `values=None` selects canonical Model values. Only selected Parameters vary;
    /// Model structure, numerical resources and dimensional scaling remain Plan-owned.
    pub fn differentiate(
        &self,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
    ) -> Result<CommonDifferentiationPoint, Diagnostic> {
        let (finalized, prepared) = self.prepare_differentiation(selected, values)?;
        let executor = HostExecutorDescriptor::new(
            self.admission.linear.provider,
            self.admission.linear.execution,
            self.admission.linear.workers,
            self.admission.linear.capabilities.clone(),
        );
        let binding = DeploymentBinding::bind_host(&self.portable, executor)?;
        let admitted = AdmittedExecution::admit_host_linear(
            &self.portable,
            finalized.canonical_csr_system_view(),
            binding,
        )?;
        let produced =
            REFERENCE_LINEAR_SOLVER.solve(&finalized.linear_problem()?, finalized.solver_plan())?;
        let accepted = admitted.accept(produced)?;
        let (solution, receipt) = accepted.into_parts();
        let point = finalized.finish(solution)?.into_point();
        let linearization = prepared.linearize(&point, receipt.report().residual_target())?;
        if linearization
            .relation()
            .state_jacobian()
            .agreement_fingerprint()
            != receipt.operator()
        {
            return Err(invalid(
                "common Plan solve receipt differs from its differentiated state system",
            ));
        }
        Ok(CommonDifferentiationPoint::from_linear(
            linearization,
            receipt,
        ))
    }

    /// Revalidate supplied algebraic coordinates at the exact selected Parameter point.
    ///
    /// Uses ordinary Plan admission, assembly, original-system residual acceptance,
    /// boundary lifting and partial actions. Executes no solve and constructs no
    /// producer report. This does not authenticate any claimed execution history.
    pub fn validate_differentiated_values(
        &self,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
        candidate: &[f64],
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let (finalized, prepared) = self.prepare_differentiation(selected, values)?;
        let (point, target) = finalized.accept_values(candidate.to_vec())?;
        prepared.linearize(&point, target)
    }

    fn prepare_differentiation(
        &self,
        selected: &[Id<kinds::Parameter>],
        values: Option<&[f64]>,
    ) -> Result<(FinalizedScalarEllipticCartesianProblem, PreparedPoint<'_>), Diagnostic> {
        if self.harmonic.is_some() {
            return Err(invalid(
                "harmonic solved-response derivatives require the retained reduction chain and are not admitted",
            ));
        }
        if !matches!(
            self.admission.system_scaling,
            eqiora_realization::SystemScaling::Dimensional
        ) {
            return Err(invalid(
                "selected-Parameter differentiation requires dimensional system scaling",
            ));
        }
        if self.admission.spatial.scalar_constraint().is_some() {
            return Err(invalid(
                "constrained TPFA differentiation has no admitted linearization",
            ));
        }

        if self.fields.len() != 1 {
            return Err(invalid(
                "selected-Parameter differentiation requires a single-field Plan",
            ));
        }
        if self.solver_provider() != REFERENCE_LINEAR_SOLVER.provider() {
            return Err(invalid(
                "selected-Parameter differentiation requires the reference solver provider",
            ));
        }
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        let RecognizedNativeModel::Spatial(template) = self.admission.recognized_model() else {
            return Err(invalid(
                "common spatial Plan lost its recognized mathematics",
            ));
        };
        let equations = template;
        let descriptor = template.conservation_descriptor(self.admission.program())?;
        let derived_form = template.primal_form(self.admission.program())?;
        let template =
            project_scalar_conservation_for_differentiation(&descriptor, derived_form.as_ref());
        let selected_values = selected
            .iter()
            .map(|field| {
                template
                    .parameter_fields()
                    .iter()
                    .position(|candidate| candidate == field)
                    .map(|index| template.parameter_values()[index])
                    .ok_or_else(|| {
                        invalid(
                            "selected differentiable Parameter is frozen or absent from this Plan",
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bound = template
            .bind_selected_parameters(selected, values.unwrap_or(selected_values.as_slice()))?;
        let NativeMeshResources::Cartesian { mesh, .. } = self.admission.resources() else {
            return Err(invalid(
                "common scalar differentiation requires exact Cartesian resources",
            ));
        };
        let dimension = self.cartesian_cells()?.len();
        let mesh = mesh.mesh();
        let source = |coordinates: &[f64]| bound.source().evaluate(coordinates).unwrap_or(f64::NAN);
        let coefficient = |coordinates: &[f64]| {
            bound
                .coefficient_expression()
                .evaluate(coordinates)
                .unwrap_or(f64::NAN)
        };
        let boundary = |axis: usize, side: BoundarySide, coordinates: &[f64]| {
            let condition = bound
                .boundary(axis, side)
                .expect("lowered Cartesian model owns every side");
            let value = condition.value().evaluate(coordinates).unwrap_or(f64::NAN);
            match condition {
                ScalarEllipticCartesianBoundary::Essential(_) => {
                    CartesianBoundaryValue::Essential(value)
                }
                ScalarEllipticCartesianBoundary::Natural(_) => {
                    CartesianBoundaryValue::Natural(value)
                }
            }
        };
        let solver = self.admission.linear.solver;
        let target = Target::HostCpu {
            threads: NonZeroUsize::MIN,
        };
        let (finalized, quadrature) = match self.admission.spatial {
            NativeSpatialPolicy::FiniteElement(_) => {
                let form = equations
                    .single()?
                    .form
                    .bind_parameter_point(bound.parameter_fields(), bound.parameter_values())?;
                let quadrature = QuadratureRule::tensor_product_gauss_legendre(
                    dimension,
                    form.volume()?.minimum_gauss_points(),
                )?;
                let assembly =
                    crate::cartesian_elliptic::linear::CartesianLinearAssembly::assemble(
                        &form,
                        mesh,
                        &quadrature,
                        &REFERENCE_ASSEMBLY_BACKEND,
                        &equations.single()?.cartesian()?.boundaries,
                    )?;
                (
                    FinalizedScalarEllipticCartesianProblem::finite_element_blocks(
                        self.portable.clone(),
                        solver,
                        VectorLayoutKind::Replicated,
                        target,
                        assembly,
                    )?,
                    quadrature,
                )
            }
            NativeSpatialPolicy::ScalarTpfa(_) => {
                let cell = QuadratureRule::tensor_product_gauss_legendre(dimension, 1)?;
                let facet = if dimension == 1 {
                    QuadratureRule::point()
                } else {
                    QuadratureRule::tensor_product_gauss_legendre(dimension - 1, 1)?
                };
                let assembly = finalize_scalar_elliptic_cartesian_fvm(
                    mesh,
                    &coefficient,
                    &source,
                    &boundary,
                    &cell,
                    &facet,
                    &REFERENCE_ASSEMBLY_BACKEND,
                )?;
                (
                    FinalizedScalarEllipticCartesianProblem::finite_volume(
                        self.portable.clone(),
                        solver,
                        VectorLayoutKind::Replicated,
                        target,
                        assembly,
                    )?,
                    cell,
                )
            }
            NativeSpatialPolicy::CoordinateCellConstant
            | NativeSpatialPolicy::StokesMiniP1(_)
            | NativeSpatialPolicy::TransientMiniP1(_)
            | NativeSpatialPolicy::TransientCellCentered(_) => {
                return Err(invalid(
                    "common scalar differentiation received a non-scalar spatial policy",
                ));
            }
        };
        Ok((
            finalized,
            PreparedPoint {
                bound,
                mesh,
                quadrature,
                coordinates: selected
                    .iter()
                    .copied()
                    .map(SpatialDesignCoordinate::from)
                    .collect(),
            },
        ))
    }
}

struct PreparedPoint<'a> {
    bound: ScalarEllipticCartesianModel,
    mesh: &'a CartesianMesh,
    quadrature: QuadratureRule,
    coordinates: Vec<SpatialDesignCoordinate>,
}

impl PreparedPoint<'_> {
    fn linearize(
        &self,
        point: &CartesianPoint,
        target: f64,
    ) -> Result<CommonDifferentiationLinearization, Diagnostic> {
        let (relation, output) =
            point.linearize(&self.bound, self.mesh, &self.quadrature, &self.coordinates)?;
        Ok(CommonDifferentiationLinearization::from_spatial(
            relation, output, target,
        ))
    }
}
