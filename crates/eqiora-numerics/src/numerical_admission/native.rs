use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum NativeSpatialPolicy {
    CoordinateCellConstant,
    FiniteElement(CommonSpatialRequest),
    ScalarTpfa(Option<eqiora_solver::AlgebraicConstraint>),
    StokesMiniP1(IncompressibleFlowScaleProfile2d),
    TransientMiniP1(IncompressibleFlowScaleProfile2d),
    TransientCellCentered(IncompressibleFlowScaleProfile2d),
}

impl NativeSpatialPolicy {
    pub(super) fn request(&self) -> CommonSpatialRequest {
        match self {
            Self::FiniteElement(request) => request.clone(),
            Self::ScalarTpfa(_) => {
                CommonSpatialRequest::Uniform(CommonSpatialPolicy::CellCenteredTpfa)
            }
            Self::CoordinateCellConstant | Self::TransientCellCentered(_) => {
                CommonSpatialRequest::Uniform(CommonSpatialPolicy::CellCentered)
            }
            Self::StokesMiniP1(_) | Self::TransientMiniP1(_) => {
                CommonSpatialRequest::Uniform(CommonSpatialPolicy::MiniP1)
            }
        }
    }

    pub(super) const fn scalar_constraint(&self) -> Option<eqiora_solver::AlgebraicConstraint> {
        match self {
            Self::ScalarTpfa(constraint) => *constraint,
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum NativeMeshResources {
    Coordinates(super::coordinate_grid::CoordinateGrid),
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
    pub(super) model_geometries: Vec<CanonicalGeometryV1>,
}

impl AuthenticatedCommonMesh {
    /// Retain additional exact Geometry artifacts referenced by the Model.
    /// These supply coordinate charts, not additional meshed regions. Resolution
    /// checks their exact closure against the Model's retained Geometry references.
    ///
    /// # Errors
    /// Rejects duplicates, repetition of the meshed Geometry, and use without a
    /// physical mesh. Unsupported numerical uses still reject during resolution.
    pub fn with_model_geometries(
        mut self,
        mut geometries: Vec<CanonicalGeometryV1>,
    ) -> Result<Self, Diagnostic> {
        if geometries.is_empty() {
            self.model_geometries.clear();
            return Ok(self);
        }
        let primary = self.resources.geometry()?.digest_bytes();
        geometries.sort_by_key(CanonicalGeometryV1::digest_bytes);
        if geometries
            .iter()
            .any(|geometry| geometry.digest_bytes() == primary)
            || geometries
                .windows(2)
                .any(|pair| pair[0].digest_bytes() == pair[1].digest_bytes())
        {
            return Err(invalid(
                "Model Geometry dependencies must be unique and distinct from the meshed Geometry",
            ));
        }
        self.model_geometries = geometries;
        Ok(self)
    }

    /// Bind a tensor grid to exact Model coordinate intervals, retaining each factor's units.
    /// This does not supply an ambient physical Geometry or a Field approximation.
    pub fn coordinate_factors(
        model: &ModelEnvelope,
        domain: eqiora_core::Id<eqiora_core::entity::kinds::Domain>,
        cells_per_factor: &[usize],
    ) -> Result<Self, Diagnostic> {
        let program = model
            .to_program()
            .map_err(|errors| errors.into_iter().next().expect("invalid Model"))?;
        Ok(Self {
            model_geometries: Vec::new(),
            resources: NativeMeshResources::Coordinates(
                super::coordinate_grid::CoordinateGrid::new(&program, domain, cells_per_factor)?,
            ),
        })
    }

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
        Ok(Self {
            resources,
            model_geometries: Vec::new(),
        })
    }

    /// Authenticate one source-owned affine-triangle occurrence from its exact production.
    pub fn affine_triangles(
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
        Ok(Self {
            resources,
            model_geometries: Vec::new(),
        })
    }

    /// Re-import and own one exact bounded Gmsh 4.15.2 provider observation.
    /// Affine tetrahedra require exact convex-polyhedral Geometry correspondence;
    /// accepting these resources does not admit a numerical space or flow solver.
    pub fn gmsh_4152(
        geometry: CanonicalGeometryV1,
        policy: eqiora_artifact::GmshMeshPolicyV1,
        provider_output: Vec<u8>,
    ) -> Result<Self, Diagnostic> {
        let resources = derive_gmsh_resources(geometry, policy, provider_output)?;
        Ok(Self {
            resources,
            model_geometries: Vec::new(),
        })
    }
}

impl NativeMeshResources {
    pub(super) fn cartesian_cells(&self) -> Result<Vec<usize>, Diagnostic> {
        let mesh = match self {
            Self::Cartesian { mesh, .. } => mesh.mesh(),
            Self::Coordinates(grid) => grid.mesh.mesh(),
            _ => {
                return Err(invalid(
                    "axis cell counts require an authenticated Cartesian mesh",
                ));
            }
        };
        (0..mesh.topological_dimension())
            .map(|axis| {
                mesh.axis_cell_count(axis)
                    .ok_or_else(|| invalid("Cartesian mesh omitted an axis"))
            })
            .collect()
    }

    pub(super) fn geometry(&self) -> Result<&CanonicalGeometryV1, Diagnostic> {
        match self {
            Self::Coordinates(_) => Err(invalid("coordinate-factor grid has no physical Geometry")),
            Self::Cartesian { geometry, .. }
            | Self::AffineTriangleSimplicial { geometry, .. }
            | Self::GmshSimplicial { geometry, .. } => Ok(geometry),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct NativeNumericalAdmission {
    recognition: RecognizedNativeAdmission,
    pub(super) spatial: NativeSpatialPolicy,
    pub(super) discretizations: Vec<eqiora_realization::DomainFieldDiscretization>,
    pub(super) system_scaling: eqiora_realization::SystemScaling,
    pub(super) linear: NativeLinearPolicy,
    pub(super) policy_identity: String,
    pub(super) temporal: Option<CommonTemporalPolicy>,
    pub(super) time_backend: Option<eqiora_time::TimeBackendCapabilities>,
    pub(super) nonlinear: Option<NonlinearSolvePlan>,
    pub(super) operator_properties: LinearOperatorProperties,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum RecognizedNativeModel {
    Coordinates(Box<super::coordinate_grid::CellEquations>),
    Spatial(Box<ExecutableSpatialEquations<f64>>),
    ComplexSpatial(Box<ExecutableSpatialEquations<num_complex::Complex64>>),
    Stokes(Box<SteadyStokesGeometryBinding2d>),
    Transient(Box<TransientIncompressibleNavierStokesModel<2>>),
    TransientGeometry(Box<TransientNavierStokesGeometryBinding2d>),
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct RecognizedNativeAdmission {
    /// Explicit formulation choices, replayed before automatic role derivation.
    pub(super) equation_tests: BTreeMap<eqiora_core::RawId, eqiora_core::RawId>,
    pub(super) model: ModelEnvelope,
    pub(super) model_digest: String,
    pub(super) program: KernelProgram,
    pub(super) recognized: RecognizedNativeModel,
    pub(super) elasticity: Option<Box<IsotropicElasticityContinuum<2>>>,
    pub(super) resources: NativeMeshResources,
    pub(super) model_geometries: Vec<CanonicalGeometryV1>,
}

impl RecognizedNativeAdmission {
    pub(super) fn form_operator_properties(
        &self,
        spatial: &NativeSpatialPolicy,
        nonlinear: Option<NonlinearSolvePlan>,
    ) -> Result<LinearOperatorProperties, Diagnostic> {
        if nonlinear.is_some() {
            return Ok(LinearOperatorProperties::General);
        }
        if let NativeSpatialPolicy::ScalarTpfa(constraint) = spatial {
            return match &self.recognized {
                RecognizedNativeModel::Spatial(equations) => {
                    equations.tpfa_operator_properties(*constraint)
                }
                RecognizedNativeModel::ComplexSpatial(equations) => {
                    equations.tpfa_operator_properties(*constraint)
                }
                _ => Err(invalid("TPFA requires scalar equations")),
            };
        }
        Ok(
            if matches!(&spatial, NativeSpatialPolicy::FiniteElement(request)
            if request.policies().all(|policy| matches!(policy, CommonSpatialPolicy::P1 | CommonSpatialPolicy::Q1)))
                && self.elasticity.is_some()
            {
                LinearOperatorProperties::SymmetricPositiveDefinite
            } else if matches!(spatial, NativeSpatialPolicy::FiniteElement(_))
                && let RecognizedNativeModel::Spatial(equations) = &self.recognized
            {
                equations.finite_element_operator_properties()
            } else {
                match spatial {
                    NativeSpatialPolicy::CoordinateCellConstant => {
                        LinearOperatorProperties::SymmetricPositiveDefinite
                    }
                    NativeSpatialPolicy::StokesMiniP1(_) => {
                        LinearOperatorProperties::SymmetricIndefinite
                    }
                    NativeSpatialPolicy::FiniteElement(_)
                    | NativeSpatialPolicy::TransientMiniP1(_)
                    | NativeSpatialPolicy::TransientCellCentered(_) => {
                        LinearOperatorProperties::General
                    }
                    NativeSpatialPolicy::ScalarTpfa(_) => {
                        unreachable!("TPFA properties derive from the compiled row above")
                    }
                }
            },
        )
    }

    pub(super) fn recognize(
        model: &ModelEnvelope,
        owner: AuthenticatedCommonMesh,
        request: &CommonSpatialRequest,
        initial_time_s: Option<f64>,
        equation_tests: &BTreeMap<eqiora_core::RawId, eqiora_core::RawId>,
    ) -> Result<Self, Diagnostic> {
        let resources = owner.resources;
        let program = if let NativeMeshResources::Coordinates(grid) = &resources {
            let program = model
                .to_program()
                .map_err(|errors| errors.into_iter().next().expect("invalid Model"))?;
            grid.source.require_program(&program)?;
            program
        } else {
            replay_program(model, resources.geometry()?, &owner.model_geometries)?
        };
        let recognized = if let NativeMeshResources::Coordinates(grid) = &resources {
            RecognizedNativeModel::Coordinates(Box::new(
                super::coordinate_grid::CellEquations::lower(&program, grid)?,
            ))
        } else {
            let transient = lower_transient_incompressible_navier_stokes_cartesian::<2>(&program);
            let transient_geometry =
                recognize_transient_incompressible_navier_stokes_geometry_mathematics(&program);
            let scalar =
                lower_scalar_candidate(&program, &resources, initial_time_s, equation_tests);
            select_exact_model(
                &program,
                &resources,
                request,
                scalar,
                transient,
                transient_geometry,
            )?
        };
        if !equation_tests.is_empty() {
            match &recognized {
                RecognizedNativeModel::Spatial(equations) => {
                    equations.require_test_assignments(equation_tests)?
                }
                RecognizedNativeModel::ComplexSpatial(equations) => {
                    equations.require_test_assignments(equation_tests)?
                }
                _ => {
                    return Err(invalid(
                        "explicit equation/test choices require the shared spatial equation owner",
                    ));
                }
            }
        }
        if !owner.model_geometries.is_empty()
            && !matches!(recognized, RecognizedNativeModel::Spatial(_))
        {
            return Err(invalid(
                "additional Model geometries require the common real linear path",
            ));
        }
        let elasticity = if matches!(recognized, RecognizedNativeModel::Spatial(_)) {
            super::elasticity::recognize(&program, &resources).map(Box::new)
        } else {
            None
        };
        let model_digest = model.digest()?.to_string();
        Ok(Self {
            equation_tests: equation_tests.clone(),
            model: model.clone(),
            model_digest,
            program,
            recognized,
            elasticity,
            resources,
            model_geometries: owner.model_geometries,
        })
    }

    pub(super) fn complete(
        self,
        spatial: NativeSpatialPolicy,
        linear: NativeLinearPolicy,
        temporal: Option<CommonTemporalPolicy>,
        time_backend: Option<eqiora_time::TimeBackendCapabilities>,
        nonlinear: Option<NonlinearSolvePlan>,
        system_scaling: eqiora_realization::SystemScaling,
    ) -> Result<NativeNumericalAdmission, Diagnostic> {
        match (
            temporal.as_ref().and_then(CommonTemporalPolicy::as_ode),
            time_backend,
        ) {
            (Some(policy), Some(backend)) => policy.validate_backend(backend)?,
            (Some(_), None) => {
                return Err(invalid(
                    "controlled time policy requires an available time backend",
                ));
            }
            (None, Some(_)) => {
                return Err(invalid("time backend requires a controlled time policy"));
            }
            (None, None) => {}
        }
        self.recognized.require_spatial_realization(&spatial)?;
        if !matches!(
            system_scaling,
            eqiora_realization::SystemScaling::Dimensional
        ) && !matches!(spatial, NativeSpatialPolicy::FiniteElement(_))
        {
            return Err(invalid(
                "congruence scaling requires the common finite-element path",
            ));
        }
        validate_resources(&spatial, &self.resources)?;
        let operator_properties =
            self.operator_properties(&spatial, &system_scaling, nonlinear, temporal.as_ref())?;
        require_policy_compatibility(
            operator_properties,
            &linear,
            self.numerical_scalar_domain(&spatial, &system_scaling, nonlinear, temporal.as_ref())?,
        )?;
        let discretizations = self.discretizations(&spatial)?;
        if matches!(spatial, NativeSpatialPolicy::FiniteElement(_)) {
            match &self.recognized {
                RecognizedNativeModel::Spatial(equations) => equations.validate_discretizations(
                    &discretizations,
                    &self.resources,
                    temporal.as_ref(),
                    nonlinear,
                    &system_scaling,
                )?,
                RecognizedNativeModel::ComplexSpatial(equations) => equations
                    .validate_discretizations(
                        &discretizations,
                        &self.resources,
                        temporal.as_ref(),
                        nonlinear,
                        &system_scaling,
                    )?,
                _ => {
                    return Err(invalid(
                        "finite-element admission requires linear equations",
                    ));
                }
            }
        }
        let policy_identity = policy_identity(
            &spatial,
            &linear,
            temporal.as_ref(),
            time_backend,
            nonlinear,
        )?;
        Ok(NativeNumericalAdmission {
            recognition: self,
            discretizations,
            system_scaling,
            operator_properties,
            spatial,
            linear,
            policy_identity,
            temporal,
            time_backend,
            nonlinear,
        })
    }
}

impl RecognizedNativeModel {
    fn require_spatial_realization(&self, spatial: &NativeSpatialPolicy) -> Result<(), Diagnostic> {
        let admitted = matches!(
            (self, spatial),
            (
                Self::Coordinates(_),
                NativeSpatialPolicy::CoordinateCellConstant
            ) | (
                Self::Spatial(_),
                NativeSpatialPolicy::FiniteElement(_) | NativeSpatialPolicy::ScalarTpfa(_)
            ) | (
                Self::ComplexSpatial(_),
                NativeSpatialPolicy::FiniteElement(_) | NativeSpatialPolicy::ScalarTpfa(None)
            ) | (Self::Stokes(_), NativeSpatialPolicy::StokesMiniP1(_))
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
    /// Rebuild derived admission and policy identity for a newly resolved exact tuple.
    /// Selection stays with the planner; this owner only rechecks its supplied result.
    pub(super) fn with_linear_policy(
        &self,
        linear: NativeLinearPolicy,
    ) -> Result<Self, Diagnostic> {
        self.recognition.clone().complete(
            self.spatial.clone(),
            linear,
            self.temporal.clone(),
            self.time_backend,
            self.nonlinear,
            self.system_scaling.clone(),
        )
    }

    pub(super) fn authenticated_mesh(&self) -> AuthenticatedCommonMesh {
        AuthenticatedCommonMesh {
            resources: self.resources().clone(),
            model_geometries: self.model_geometries().to_vec(),
        }
    }

    pub(super) fn elasticity_proof(&self) -> Option<&IsotropicElasticityContinuum<2>> {
        (matches!(self.spatial, NativeSpatialPolicy::FiniteElement(_))
            && self
                .discretizations
                .iter()
                .flat_map(|domain| domain.field_spaces())
                .all(|field| {
                    field.space() == Space::continuous_lagrange(std::num::NonZeroU16::MIN)
                }))
        .then_some(self.recognition.elasticity.as_deref())
        .flatten()
    }

    pub(super) fn model_geometries(&self) -> &[CanonicalGeometryV1] {
        &self.recognition.model_geometries
    }

    #[cfg(test)]
    pub(super) fn admit(
        model: &ModelEnvelope,
        owner: AuthenticatedCommonMesh,
        spatial: NativeSpatialPolicy,
        linear: NativeLinearPolicy,
    ) -> Result<Self, Diagnostic> {
        RecognizedNativeAdmission::recognize(
            model,
            owner,
            &spatial.request(),
            None,
            &BTreeMap::new(),
        )?
        .complete(
            spatial,
            linear,
            None,
            None,
            None,
            eqiora_realization::SystemScaling::Dimensional,
        )
    }

    pub(super) fn revalidate(&self) -> Result<(), Diagnostic> {
        let replayed = RecognizedNativeAdmission::recognize(
            self.model(),
            self.authenticated_mesh(),
            &self.spatial.request(),
            self.temporal.as_ref().map(|_| 0.),
            &self.recognition.equation_tests,
        )?
        .complete(
            self.spatial.clone(),
            self.linear.clone(),
            self.temporal.clone(),
            self.time_backend,
            self.nonlinear,
            self.system_scaling.clone(),
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
            scalar_domain: eqiora_core::ScalarDomain::Real,
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

    pub(super) fn execute_linear(
        &self,
        backend: &dyn LinearSolverBackend,
        initial: Option<&BTreeMap<eqiora_core::RawId, Vec<f64>>>,
    ) -> Result<CommonLinearRunOutput<f64>, Diagnostic> {
        if initial.is_some()
            && (!matches!(self.recognized_model(), RecognizedNativeModel::Spatial(_))
                || !matches!(self.spatial, NativeSpatialPolicy::FiniteElement(_)))
        {
            return Err(invalid(
                "initial spatial State requires a real nodal finite-element Plan",
            ));
        }
        if let RecognizedNativeModel::ComplexSpatial(equations) = self.recognized_model() {
            return self.execute_complex_linear(equations, backend);
        }
        self.revalidate()?;
        if backend.provider() != self.linear.provider
            || backend.capabilities() != self.linear.capabilities
        {
            return Err(invalid(
                "scalar execution backend differs from admitted provider or capabilities",
            ));
        }
        if let RecognizedNativeModel::Coordinates(projection) = self.recognized_model() {
            return super::coordinate_grid::execute(self, projection, backend);
        }
        let RecognizedNativeModel::Spatial(lowered) = self.recognized_model() else {
            return Err(invalid(
                "native numerical admission does not own recognized linear-region meaning",
            ));
        };
        let structure = lowered.algebraic_structure(self.spatial.scalar_constraint())?;
        let checked_backend = self.linear.checked_backend(backend, Some(&structure))?;
        let backend: &dyn LinearSolverBackend = &checked_backend;
        let solve = LinearSolveRequest::new(backend, self.linear.solver);
        if self.spatial.scalar_constraint().is_none() {
            return lowered
                .finalize_static(
                    self.linear.workers,
                    (solve).plan(),
                    self.resources(),
                    &self.discretizations,
                    &self.system_scaling,
                    self.operator_properties,
                )
                .and_then(|system| system.solve(solve, initial));
        }
        let system = lowered.finalize_static(
            self.linear.workers,
            self.linear.solver,
            self.resources(),
            &self.discretizations,
            &self.system_scaling,
            LinearOperatorProperties::Symmetric,
        )?;
        let reference = lowered.mean_reference(self.resources(), &system.mapping)?;
        system.solve_with_reference(solve, &reference)
    }
}

mod identity;
mod parameter_point;
pub(super) mod polyhedral;
mod recognition;
mod refinement;
mod resources;

pub(super) use identity::{
    domain_separated_identity, hex_bytes, invalid, policy_identity, push_framed,
    push_linear_policy_identity, replay_program, require_portable_realization, space_identity,
    static_plan_identity_lineage,
};
pub(super) use recognition::{
    ResourceDigests, lower_scalar_candidate, require_policy_compatibility,
    resource_artifact_digests, resource_digests, select_exact_model,
};
pub(super) use resources::{
    derive_gmsh_resources, validate_cartesian_resources, validate_resources,
    validate_simplicial_resources,
};

#[cfg(test)]
mod complex_profile;

mod coordinates;
