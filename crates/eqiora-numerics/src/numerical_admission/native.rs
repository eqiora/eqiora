use super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum NativeSpatialPolicy {
    ScalarQ1,
    ScalarTpfa,
    ElasticityQ1,
    StokesMiniP1(IncompressibleFlowScaleProfile2d),
    TransientMiniP1(IncompressibleFlowScaleProfile2d),
    TransientCellCentered(IncompressibleFlowScaleProfile2d),
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct NativeLinearPolicy {
    pub(super) solver: SolverPlan,
    pub(super) provider: SolverProvider,
    pub(super) capabilities: SolverCapabilities,
    pub(super) execution: ExecutionProvider,
    pub(super) workers: NonZeroUsize,
    pub(super) planning_objective: Option<SolverPlanningObjective>,
    pub(super) planning_profile: Option<eqiora_solver::HostSerialSolverProfile>,
    pub(super) planning_policy_id: Option<&'static str>,
    pub(super) selected_candidate_id: Option<&'static str>,
    pub(super) selected_evidence_case: Option<&'static str>,
    pub(super) planning_reasons: Vec<(&'static str, &'static str)>,
}

impl NativeLinearPolicy {
    pub(super) fn exact(
        solver: SolverPlan,
        backend: &dyn LinearSolverBackend,
    ) -> Result<Self, Diagnostic> {
        if solver.relative_tolerance().to_bits() == (-0.0_f64).to_bits()
            || solver.absolute_tolerance().to_bits() == (-0.0_f64).to_bits()
        {
            return Err(invalid(
                "linear policy contains signed-zero tolerance ambiguity",
            ));
        }
        let provider = backend.provider();
        provider.validate()?;
        SERIAL_EXECUTION_PROVIDER.validate()?;
        Ok(Self {
            solver,
            provider,
            capabilities: backend.capabilities(),
            execution: SERIAL_EXECUTION_PROVIDER,
            workers: NonZeroUsize::MIN,
            planning_objective: None,
            planning_profile: None,
            planning_policy_id: None,
            selected_candidate_id: None,
            selected_evidence_case: None,
            planning_reasons: Vec::new(),
        })
    }

    pub(super) fn with_planning(
        mut self,
        decision: &ResolvedHostSerialSolverPlan<'_>,
    ) -> Result<Self, Diagnostic> {
        if self.solver != decision.solver_plan()
            || self.provider != decision.solver_provider()
            || self.execution != decision.execution_provider()
        {
            return Err(invalid(
                "solver planning audit does not authenticate the effective linear policy",
            ));
        }
        self.planning_objective = Some(decision.objective());
        self.planning_profile = Some(decision.profile());
        self.planning_policy_id = Some(decision.policy_id());
        self.selected_candidate_id = Some(decision.selected_candidate_id());
        self.selected_evidence_case = Some(decision.selected_evidence_case());
        self.planning_reasons = decision.reasons().collect();
        Ok(self)
    }

    pub(super) fn checked_backend<'a>(
        &self,
        backend: &'a dyn LinearSolverBackend,
        structure: Option<&eqiora_solver::AlgebraicStructure>,
    ) -> Result<ProfileCheckedBackend<'a>, Diagnostic> {
        if backend.provider() != self.provider || backend.capabilities() != self.capabilities {
            return Err(invalid(
                "execution backend differs from admitted exact provider or capabilities",
            ));
        }
        if let Some(profile) = &self.planning_profile {
            profile.require_structure(structure)?;
        } else if structure.is_some() {
            return Err(invalid("typed algebraic structure lacks solver admission"));
        }
        Ok(ProfileCheckedBackend {
            backend,
            plan: self.solver,
            profile: self.planning_profile.clone(),
        })
    }

    pub(super) fn planning_audit_is_coherent(&self) -> bool {
        match self.planning_objective {
            None => {
                self.planning_policy_id.is_none()
                    && self.selected_candidate_id.is_none()
                    && self.selected_evidence_case.is_none()
                    && self.planning_reasons.is_empty()
            }
            Some(_) => {
                self.planning_profile.is_some()
                    && self.planning_policy_id.is_some()
                    && self.selected_candidate_id.is_some()
                    && self.selected_evidence_case.is_some()
                    && !self.planning_reasons.is_empty()
            }
        }
    }
}

/// The existing execution boundary rechecks admitted facts before forwarding
/// exactly one numerical attempt to the already selected backend.
#[derive(Debug)]
pub(super) struct ProfileCheckedBackend<'a> {
    backend: &'a dyn LinearSolverBackend,
    plan: SolverPlan,
    profile: Option<eqiora_solver::HostSerialSolverProfile>,
}

impl LinearSolverBackend for ProfileCheckedBackend<'_> {
    fn provider(&self) -> SolverProvider {
        self.backend.provider()
    }
    fn capabilities(&self) -> SolverCapabilities {
        self.backend.capabilities()
    }
    fn solve_with_execution(
        &self,
        problem: &eqiora_solver::LinearProblem<'_>,
        plan: SolverPlan,
        execution: &dyn eqiora_solver::ReplicatedLinearExecution,
    ) -> Result<eqiora_solver::LinearSolution, Diagnostic> {
        if plan != self.plan {
            return Err(invalid("execution changed the admitted exact solver plan"));
        }
        if let Some(profile) = &self.profile {
            profile.require_problem(problem)?;
        }
        self.backend.solve_with_execution(problem, plan, execution)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum NativeMeshResources {
    Cartesian {
        geometry: CanonicalGeometryV1,
        mesh: CartesianMeshEnvelopeV1,
        correspondence: GeometryMeshCorrespondenceEnvelopeV1,
        production: MeshProductionLineageEnvelopeV1,
    },
    AffineTriangleSimplicial {
        geometry: CanonicalGeometryV1,
        mesh: SimplicialMeshEnvelopeV1,
        correspondence: GeometryMeshCorrespondenceEnvelopeV1,
        production: MeshProductionLineageEnvelopeV1,
    },
    AdjacentPartitionSimplicial {
        geometry: CanonicalGeometryV1,
        mesh: SimplicialMeshEnvelopeV1,
        correspondence: GeometryMeshCorrespondenceEnvelopeV1,
        production: MeshProductionLineageEnvelopeV1,
    },
    GmshSimplicial {
        geometry: CanonicalGeometryV1,
        policy: eqiora_artifact::GmshMeshPolicyV1,
        provider_output: Box<[u8]>,
        mesh: SimplicialMeshEnvelopeV1,
        correspondence: GeometryMeshCorrespondenceEnvelopeV1,
        production: MeshProductionLineageEnvelopeV1,
    },
}

/// Authenticated in-process owner of one exact common Geometry/Mesh occurrence.
#[derive(Debug, Clone, PartialEq)]
pub struct AuthenticatedCommonMesh {
    pub(super) resources: NativeMeshResources,
}

impl AuthenticatedCommonMesh {
    /// Authenticate and own one structured-Cartesian rectangle occurrence.
    pub fn structured_cartesian(
        geometry: CanonicalGeometryV1,
        mesh: CartesianMeshEnvelopeV1,
        correspondence: GeometryMeshCorrespondenceEnvelopeV1,
        production: MeshProductionLineageEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        let resources = NativeMeshResources::Cartesian {
            geometry,
            mesh,
            correspondence,
            production,
        };
        validate_cartesian_resources(&resources)?;
        Ok(Self { resources })
    }

    /// Authenticate and own one fixed-diagonal affine-triangle rectangle occurrence.
    pub fn affine_triangle_rectangle(
        geometry: CanonicalGeometryV1,
        mesh: SimplicialMeshEnvelopeV1,
        correspondence: GeometryMeshCorrespondenceEnvelopeV1,
        production: MeshProductionLineageEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        let resources = NativeMeshResources::AffineTriangleSimplicial {
            geometry,
            mesh,
            correspondence,
            production,
        };
        validate_simplicial_resources(&resources)?;
        Ok(Self { resources })
    }

    /// Authenticate and own one fixed-diagonal adjacent-partition occurrence.
    pub fn adjacent_partition(
        geometry: CanonicalGeometryV1,
        mesh: SimplicialMeshEnvelopeV1,
        correspondence: GeometryMeshCorrespondenceEnvelopeV1,
        production: MeshProductionLineageEnvelopeV1,
    ) -> Result<Self, Diagnostic> {
        let resources = NativeMeshResources::AdjacentPartitionSimplicial {
            geometry,
            mesh,
            correspondence,
            production,
        };
        validate_simplicial_resources(&resources)?;
        Ok(Self { resources })
    }

    /// Re-import and own one exact bounded Gmsh 4.15.2 provider observation.
    pub fn gmsh_4152(
        geometry: CanonicalGeometryV1,
        policy: eqiora_artifact::GmshMeshPolicyV1,
        provider_output: Vec<u8>,
    ) -> Result<Self, Diagnostic> {
        let resources = derive_gmsh_resources(geometry, policy, provider_output)?;
        Ok(Self { resources })
    }
}

impl NativeMeshResources {
    pub(super) fn geometry(&self) -> &CanonicalGeometryV1 {
        match self {
            Self::Cartesian { geometry, .. }
            | Self::AffineTriangleSimplicial { geometry, .. }
            | Self::AdjacentPartitionSimplicial { geometry, .. }
            | Self::GmshSimplicial { geometry, .. } => geometry,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct NativeNumericalAdmission {
    recognition: RecognizedNativeAdmission,
    pub(super) spatial: NativeSpatialPolicy,
    pub(super) linear: NativeLinearPolicy,
    pub(super) policy_identity: String,
    pub(super) temporal: Option<CommonBackwardEuler>,
    pub(super) nonlinear: Option<NonlinearSolvePlan>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum RecognizedNativeModel {
    Scalar(Box<ExecutableScalarEquations>),
    Elasticity(Box<IsotropicElasticityContinuum<2>>),
    Stokes(Box<SteadyStokesGeometryBinding2d>),
    Transient(Box<TransientIncompressibleNavierStokesCartesianModel2d>),
    TransientGeometry(Box<TransientNavierStokesGeometryBinding2d>),
    Fsi(Box<FixedReferenceFsiCartesianModel2d>),
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct RecognizedNativeAdmission {
    pub(super) model: ModelEnvelope,
    pub(super) model_digest: String,
    pub(super) program: KernelProgram,
    pub(super) recognized: RecognizedNativeModel,
    pub(super) resources: NativeMeshResources,
}

impl RecognizedNativeAdmission {
    pub(super) fn recognize(
        model: &ModelEnvelope,
        owner: AuthenticatedCommonMesh,
    ) -> Result<Self, Diagnostic> {
        let resources = owner.resources;
        let program = replay_program(model, resources.geometry())?;
        let transient = lower_transient_incompressible_navier_stokes_cartesian_2d(&program);
        let transient_geometry =
            recognize_transient_incompressible_navier_stokes_geometry_mathematics(&program);
        let fsi = lower_fixed_reference_fsi_geometry_2d(&program, resources.geometry());
        let scalar = lower_scalar_candidate(&program, &resources);
        let recognized = recognize_exact_model(
            &program,
            &resources,
            scalar,
            transient,
            transient_geometry,
            fsi,
        )?;
        let model_digest = model.digest()?.to_string();
        Ok(Self {
            model: model.clone(),
            model_digest,
            program,
            recognized,
            resources,
        })
    }

    pub(super) fn complete(
        self,
        spatial: NativeSpatialPolicy,
        linear: NativeLinearPolicy,
        temporal: Option<CommonBackwardEuler>,
        nonlinear: Option<NonlinearSolvePlan>,
    ) -> Result<NativeNumericalAdmission, Diagnostic> {
        self.recognized.require_spatial_realization(spatial)?;
        require_policy_compatibility(spatial, &linear)?;
        validate_resources(spatial, &self.resources)?;
        if spatial == NativeSpatialPolicy::ScalarTpfa {
            let RecognizedNativeModel::Scalar(equations) = &self.recognized else {
                return Err(invalid("TPFA requires scalar equations"));
            };
            equations.conservation_descriptor(&self.program)?;
        }
        let policy_identity = policy_identity(spatial, &linear, temporal, nonlinear);
        Ok(NativeNumericalAdmission {
            recognition: self,
            spatial,
            linear,
            policy_identity,
            temporal,
            nonlinear,
        })
    }
}

impl RecognizedNativeModel {
    fn require_spatial_realization(&self, spatial: NativeSpatialPolicy) -> Result<(), Diagnostic> {
        let admitted = matches!(
            (self, spatial),
            (
                Self::Scalar(_),
                NativeSpatialPolicy::ScalarQ1 | NativeSpatialPolicy::ScalarTpfa
            ) | (Self::Elasticity(_), NativeSpatialPolicy::ElasticityQ1)
                | (Self::Stokes(_), NativeSpatialPolicy::StokesMiniP1(_))
                | (
                    Self::Transient(_) | Self::TransientGeometry(_),
                    NativeSpatialPolicy::TransientMiniP1(_)
                        | NativeSpatialPolicy::TransientCellCentered(_)
                )
        );
        if !admitted {
            return Err(invalid(
                "typed mathematical form and requested spatial realization are incompatible",
            ));
        }
        Ok(())
    }
}

impl NativeNumericalAdmission {
    #[cfg(test)]
    pub(super) fn admit(
        model: &ModelEnvelope,
        owner: AuthenticatedCommonMesh,
        spatial: NativeSpatialPolicy,
        linear: NativeLinearPolicy,
    ) -> Result<Self, Diagnostic> {
        RecognizedNativeAdmission::recognize(model, owner)?.complete(spatial, linear, None, None)
    }

    pub(super) fn revalidate(&self) -> Result<(), Diagnostic> {
        let replayed = RecognizedNativeAdmission::recognize(
            self.model(),
            AuthenticatedCommonMesh {
                resources: self.resources().clone(),
            },
        )?
        .complete(
            self.spatial,
            self.linear.clone(),
            self.temporal,
            self.nonlinear,
        )?;
        if &replayed != self {
            return Err(invalid(
                "native numerical admission changed during exact internal replay",
            ));
        }
        Ok(())
    }

    pub(super) const fn model(&self) -> &ModelEnvelope {
        &self.recognition.model
    }

    pub(super) fn model_digest(&self) -> &str {
        &self.recognition.model_digest
    }

    pub(super) const fn program(&self) -> &KernelProgram {
        &self.recognition.program
    }

    pub(super) const fn recognized_model(&self) -> &RecognizedNativeModel {
        &self.recognition.recognized
    }

    pub(super) fn policy_identity(&self) -> &str {
        &self.policy_identity
    }

    pub(super) const fn resources(&self) -> &NativeMeshResources {
        &self.recognition.resources
    }

    pub(super) fn stokes_binding(&self) -> Result<SteadyStokesGeometryBinding2d, Diagnostic> {
        let RecognizedNativeModel::Stokes(binding) = self.recognized_model() else {
            return Err(invalid(
                "native numerical admission does not own recognized steady-Stokes meaning",
            ));
        };
        Ok((**binding).clone())
    }

    pub(super) fn resolve_stokes(
        &self,
        binding: &SteadyStokesGeometryBinding2d,
    ) -> Result<
        (
            ResolvedFieldwiseRealization,
            PortableRealizationGraph,
            Space,
            Space,
        ),
        Diagnostic,
    > {
        let NativeSpatialPolicy::StokesMiniP1(scales) = self.spatial else {
            return Err(invalid(
                "steady-Stokes admission has a non-Stokes spatial policy",
            ));
        };
        let NativeMeshResources::GmshSimplicial { mesh, .. } = self.resources() else {
            return Err(invalid(
                "steady Stokes requires exact supplied simplicial resources",
            ));
        };
        let solver = self.linear.solver;
        let fieldwise = binding.mini_plan(mesh.artifact_reference()?, scales, solver)?;
        let selected_solver = SolverCapabilities::exact([SolverCapability {
            algorithm: solver.algorithm(),
            operator_properties: LinearOperatorProperties::SymmetricIndefinite,
            preconditioner: solver.preconditioner(),
            reduction: solver.reduction(),
            scalar_type: ScalarType::F64,
        }])?;
        let capabilities = RealizationCapabilities::cartesian_product(
            [DiscretizationMethod::ContinuousGalerkin],
            [(
                MeshKind::ImportedAffineSimplicial,
                SpatialDimensionSupport::exact(NonZeroUsize::new(2).expect("two is nonzero")),
            )],
            [VectorLayoutKind::Replicated],
            selected_solver,
            TargetCapabilities::none().with_host_cpu(self.linear.workers),
        )?;
        let resolved = resolve_fieldwise(
            &FieldwiseRealizationRequest::explicit(
                self.program().model(),
                SemanticRevision::new(self.program().revision().0),
                RealizationRevision::new(APPLICATION_REALIZATION_REVISION),
                fieldwise,
            ),
            binding.fieldwise_requirements(),
            &capabilities,
        )?;
        let portable = resolved.portable_graph()?;
        let mut velocity = None;
        let mut pressure = None;
        for field in resolved.plan().spatial().field_spaces() {
            match field.space().family() {
                SpaceFamily::SimplexP1Bubble if velocity.replace(field.space()).is_none() => {}
                SpaceFamily::ContinuousLagrange { order }
                    if order == std::num::NonZeroU16::MIN
                        && pressure.replace(field.space()).is_none() => {}
                _ => {
                    return Err(invalid(
                        "steady-Stokes resolved space inventory is not MINI/P1",
                    ));
                }
            }
        }
        let (velocity, pressure) = velocity
            .zip(pressure)
            .ok_or_else(|| invalid("steady-Stokes resolved space inventory is incomplete"))?;
        Ok((resolved, portable, velocity, pressure))
    }

    pub(super) fn execute_scalar(
        &self,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonScalarRunOutput, Diagnostic> {
        self.execute_scalar_with_completion(backend, |reactions, full| reactions.recover(full))
    }

    pub(super) fn execute_scalar_with_completion(
        &self,
        backend: &dyn LinearSolverBackend,
        complete: impl FnOnce(
            &crate::region_assembly::InterfaceReactions,
            &[f64],
        )
            -> Result<crate::region_assembly::RecoveredInterfaceReactions, Diagnostic>,
    ) -> Result<CommonScalarRunOutput, Diagnostic> {
        self.revalidate()?;
        if backend.provider() != self.linear.provider
            || backend.capabilities() != self.linear.capabilities
        {
            return Err(invalid(
                "scalar execution backend differs from admitted provider or capabilities",
            ));
        }
        let NativeMeshResources::Cartesian { mesh, .. } = self.resources() else {
            return Err(invalid(
                "scalar elliptic execution requires Cartesian resources",
            ));
        };
        let RecognizedNativeModel::Scalar(lowered) = self.recognized_model() else {
            return Err(invalid(
                "native numerical admission does not own recognized scalar-elliptic meaning",
            ));
        };
        let structure = lowered.algebraic_structure()?;
        let checked_backend = self.linear.checked_backend(backend, Some(&structure))?;
        let backend: &dyn LinearSolverBackend = &checked_backend;
        let solve = LinearSolveRequest::new(backend, self.linear.solver);
        if self.spatial == NativeSpatialPolicy::ScalarQ1 {
            return lowered.execute(self, solve, mesh.mesh(), complete);
        }
        let descriptor = lowered.conservation_descriptor(self.program())?;
        let region = descriptor
            .regions()
            .next()
            .expect("one admitted TPFA region");
        let source = |coordinates: &[f64]| {
            region.source().map_or(0.0, |source| {
                source
                    .expression()
                    .evaluate(coordinates)
                    .unwrap_or(f64::NAN)
            })
        };
        let coefficient = |coordinates: &[f64]| {
            region
                .flux()
                .coefficient()
                .evaluate(coordinates)
                .unwrap_or(f64::NAN)
        };
        let boundary = |axis: usize, side: BoundarySide, coordinates: &[f64]| {
            let law = region
                .exterior_at(axis, side)
                .expect("admitted scalar conservation owns every side")
                .law();
            match law {
                ScalarExteriorLaw::PrescribedTrace { value, .. } => {
                    CartesianBoundaryValue::Essential(
                        value.evaluate(coordinates).unwrap_or(f64::NAN),
                    )
                }
                ScalarExteriorLaw::PrescribedOutwardFlux { value, .. } => {
                    CartesianBoundaryValue::Natural(value.evaluate(coordinates).unwrap_or(f64::NAN))
                }
                ScalarExteriorLaw::ZeroOutwardFlux { .. } => CartesianBoundaryValue::Natural(0.0),
                ScalarExteriorLaw::Robin { .. } => {
                    unreachable!("steady scalar admission rejects Robin boundaries")
                }
            }
        };
        let solve = LinearSolveRequest::new(backend, self.linear.solver);
        match self.spatial {
            NativeSpatialPolicy::ScalarQ1 => {
                unreachable!("Q1 executed through linear block assembly")
            }
            NativeSpatialPolicy::ScalarTpfa => {
                let cell = QuadratureRule::tensor_product_gauss_legendre(mesh.dimension(), 1)?;
                let facet = if mesh.dimension() == 1 {
                    QuadratureRule::point()
                } else {
                    QuadratureRule::tensor_product_gauss_legendre(mesh.dimension() - 1, 1)?
                };
                let finalized = finalize_scalar_elliptic_cartesian_fvm(
                    mesh.mesh(),
                    &coefficient,
                    &source,
                    &boundary,
                    &cell,
                    &facet,
                    &REFERENCE_ASSEMBLY_BACKEND,
                )?;
                let (system, state) = finalized.into_canonical()?;
                let solved = solve.solve(&system.linear_problem()?)?;
                let solution = state.finish(solved, system)?;
                let [(field, value_type)] = lowered.single()?.form.fields() else {
                    return Err(invalid("TPFA requires one admitted Field"));
                };
                Ok(CommonScalarRunOutput {
                    fields: vec![(
                        field.downcast().expect("compiled Field identity"),
                        value_type.clone(),
                        solution.cell_values().to_vec(),
                    )],
                    solve_report: solution.solve_report().clone(),
                    assembly_report: solution.assembly_report().clone(),
                })
            }
            NativeSpatialPolicy::ElasticityQ1 => Err(invalid(
                "scalar execution received an elasticity spatial policy",
            )),
            NativeSpatialPolicy::StokesMiniP1(_) => Err(invalid(
                "scalar execution received a steady-Stokes spatial policy",
            )),
            NativeSpatialPolicy::TransientMiniP1(_)
            | NativeSpatialPolicy::TransientCellCentered(_) => Err(invalid(
                "scalar execution received a transient-flow spatial policy",
            )),
        }
    }

    pub(super) fn execute_elasticity(
        &self,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CartesianLinearElasticity2dSolution, Diagnostic> {
        self.revalidate()?;
        if backend.provider() != self.linear.provider
            || backend.capabilities() != self.linear.capabilities
        {
            return Err(invalid(
                "elasticity execution backend differs from admitted provider or capabilities",
            ));
        }
        let NativeMeshResources::Cartesian { mesh, .. } = self.resources() else {
            return Err(invalid("elasticity execution requires Cartesian resources"));
        };
        let RecognizedNativeModel::Elasticity(lowered) = self.recognized_model() else {
            return Err(invalid(
                "native numerical admission does not own recognized elasticity meaning",
            ));
        };
        let structure = super::elasticity::algebraic_structure(lowered)?;
        let checked_backend = self.linear.checked_backend(backend, Some(&structure))?;
        let backend: &dyn LinearSolverBackend = &checked_backend;
        let finalized = finalize_isotropic_elasticity_cartesian_q1_on_mesh(
            lowered,
            mesh.mesh(),
            self.linear.solver,
            &REFERENCE_ASSEMBLY_BACKEND,
        )?;
        let solved = backend.solve(&finalized.linear_problem()?, finalized.solver_plan())?;
        finalized.finish(solved)
    }
}

mod identity;
mod recognition;
mod resources;

pub(super) use identity::{
    domain_separated_identity, hex_bytes, invalid, policy_identity, push_framed, replay_program,
    require_portable_realization, space_identity, static_plan_identity_lineage,
};
pub(super) use recognition::{
    ResourceDigests, lower_scalar_candidate, recognize_exact_model, require_policy_compatibility,
    resource_artifact_digests, resource_digests,
};
pub(super) use resources::{
    derive_gmsh_resources, validate_cartesian_resources, validate_resources,
    validate_simplicial_resources,
};
