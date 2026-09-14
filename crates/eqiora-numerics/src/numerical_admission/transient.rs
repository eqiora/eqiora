use super::*;

pub(super) struct PreparedCommonTransientExecution<'a> {
    plan: &'a CommonTransientFlowPlan,
    backend: super::native::ProfileCheckedBackend<'a>,
    method: PreparedCommonTransientMethod<'a>,
}

enum PreparedCommonTransientMethod<'a> {
    MiniP1(Box<PreparedResolvedTransientMiniRun2d<'a>>),
    GeometryMiniP1(Box<PreparedResolvedTransientGeometryMiniRun2d<'a>>),
    CellCentered(Box<PreparedResolvedTransientCellCenteredRun2d<'a>>),
}

impl PreparedCommonTransientExecution<'_> {
    pub(super) fn advance(&self, state: &CommonState) -> Result<CommonState, Diagnostic> {
        let run = TransientNavierStokesRun2d::new(NonZeroStepCount::new(NonZeroUsize::MIN));
        let next_time = state.time_s() + self.plan.temporal().step().value();
        match &self.method {
            PreparedCommonTransientMethod::MiniP1(prepared) => {
                let CommonStateKind::MiniP1(initial) = &state.kind else {
                    return Err(invalid(
                        "prepared MINI Run received a non-MINI common State",
                    ));
                };
                let trajectory = prepared.advance(initial.as_ref().clone(), run, &self.backend)?;
                let accepted = trajectory
                    .states()
                    .last()
                    .ok_or_else(|| invalid("MINI transient step returned no accepted State"))?;
                let NativeMeshResources::AffineTriangleSimplicial { mesh, .. } =
                    self.plan.admission.resources()
                else {
                    unreachable!("prepared MINI Run owns affine-triangle resources")
                };
                self.accept_mini(mesh, state, next_time, accepted)
            }
            PreparedCommonTransientMethod::GeometryMiniP1(prepared) => {
                let CommonStateKind::MiniP1(initial) = &state.kind else {
                    return Err(invalid(
                        "prepared MINI Run received a non-MINI common State",
                    ));
                };
                let states = prepared.advance(initial.as_ref().clone(), run, &self.backend)?;
                let accepted = states.last().ok_or_else(|| {
                    invalid("Geometry MINI transient step returned no accepted State")
                })?;
                let NativeMeshResources::GmshSimplicial { mesh, .. } =
                    self.plan.admission.resources()
                else {
                    unreachable!("prepared Geometry MINI Run owns Gmsh resources")
                };
                self.accept_mini(mesh, state, next_time, accepted)
            }
            PreparedCommonTransientMethod::CellCentered(prepared) => {
                let CommonStateKind::CellCentered(initial) = &state.kind else {
                    return Err(invalid(
                        "prepared cell-centered Run received incompatible method history",
                    ));
                };
                let trajectory = prepared.advance(initial.as_ref().clone(), run, &self.backend)?;
                let accepted = trajectory.states().last().ok_or_else(|| {
                    invalid("cell-centered transient step returned no accepted State")
                })?;
                self.accept(
                    state,
                    next_time,
                    CommonStateKind::CellCentered(Box::new(
                        prepared.initial_from_accepted(accepted)?,
                    )),
                    Vec::new(),
                )
            }
        }
    }

    fn accept_mini(
        &self,
        mesh: &SimplicialMeshEnvelopeV1,
        previous: &CommonState,
        next_time: f64,
        accepted: &ResolvedTransientNavierStokesState2d,
    ) -> Result<CommonState, Diagnostic> {
        self.accept(
            previous,
            next_time,
            CommonStateKind::MiniP1(Box::new(mini_initial_from_resolved(
                self.plan, mesh, accepted,
            )?)),
            accepted.named_boundary_forces_on_domain().to_vec(),
        )
    }

    fn accept(
        &self,
        previous: &CommonState,
        next_time: f64,
        kind: CommonStateKind,
        named_boundary_forces_on_domain: Vec<(String, [f64; 2])>,
    ) -> Result<CommonState, Diagnostic> {
        CommonState::new_with_boundary_forces(
            self.plan.state_space_identity(),
            next_time,
            Arc::clone(&previous.model),
            Arc::clone(&previous.resources),
            kind,
            named_boundary_forces_on_domain,
        )
    }
}

impl CommonTransientFlowPlan {
    fn reauthenticate_portable_realization(&self) -> Result<(), Diagnostic> {
        let materialized = match &self.resolved {
            CommonTransientResolvedSpatial::MiniP1(resolved) => resolved.portable_graph()?,
            CommonTransientResolvedSpatial::CellCentered(resolved) => resolved.portable_graph()?,
        };
        require_portable_realization(&self.portable, materialized)
    }

    /// Effective Formulation and its automatic-selection audit.
    #[must_use]
    pub fn formulation(&self) -> CommonFormulationDescription {
        match &self.formulation {
            CommonTransientFormulation::MixedGalerkin(correspondence) => {
                CommonFormulationDescription::mixed(
                    correspondence,
                    self.formulation_selection,
                    match self.formulation_selection {
                        FormulationSelectionMode::Automatic => {
                            "eqiora.formulation.auto.mixed-galerkin-for-mini-p1/v1"
                        }
                        FormulationSelectionMode::Exact => {
                            "eqiora.formulation.exact.mixed-galerkin-admitted/v1"
                        }
                        FormulationSelectionMode::Authored => {
                            unreachable!("authored transient forms are not admitted")
                        }
                    },
                )
            }
            CommonTransientFormulation::IntegralConservative(correspondence) => {
                CommonFormulationDescription::integral(correspondence, self.formulation_selection)
            }
        }
    }

    pub(super) fn from_admission(
        model: &ModelEnvelope,
        admission: NativeNumericalAdmission,
        formulation_selection: FormulationSelectionMode,
        scaling: ResolvedIncompressibleScaling2d,
        temporal: CommonBackwardEuler,
        nonlinear: NonlinearSolvePlan,
    ) -> Result<Self, Diagnostic> {
        let model_reference = model.artifact_reference()?;
        let model_id = model_reference.model().ulid().to_string();
        let (velocity, pressure) = match admission.recognized_model() {
            RecognizedNativeModel::Transient(model) => (model.velocity(), model.pressure()),
            RecognizedNativeModel::TransientGeometry(binding) => {
                (binding.velocity(), binding.pressure())
            }
            _ => {
                return Err(invalid(
                    "transient Plan admission omitted recognized transient Model meaning",
                ));
            }
        };
        let velocity_field_id = velocity.ulid().to_string();
        let pressure_field_id = pressure.ulid().to_string();
        let solver = admission.linear.solver;
        let (resolved, formulation, velocity_space, pressure_space, gauge) =
            match (admission.spatial, admission.resources()) {
                (
                    NativeSpatialPolicy::TransientMiniP1(scales),
                    NativeMeshResources::AffineTriangleSimplicial { mesh, .. },
                ) => {
                    let RecognizedNativeModel::Transient(transient) = admission.recognized_model()
                    else {
                        return Err(invalid(
                            "affine transient Mesh requires Cartesian Model meaning",
                        ));
                    };
                    let plan = transient_navier_stokes_mini_plan_2d(
                        transient,
                        mesh.artifact_reference()?,
                        scales,
                        temporal.step(),
                        nonlinear,
                        solver,
                    )?;
                    let capabilities = transient_realization_capabilities(
                        DiscretizationMethod::ContinuousGalerkin,
                        MeshKind::ImportedAffineSimplicial,
                        solver,
                        &admission.linear.capabilities,
                    )?;
                    let resolved = resolve_transient_fieldwise(
                        &TransientFieldwiseRealizationRequest::explicit(
                            admission.program().model(),
                            SemanticRevision::new(admission.program().revision().0),
                            RealizationRevision::new(TRANSIENT_REALIZATION_REVISION),
                            plan,
                        ),
                        transient_navier_stokes_fieldwise_requirements_2d(transient),
                        &capabilities,
                    )?;
                    let gauge = if resolved
                        .plan()
                        .fieldwise()
                        .spatial()
                        .constraints()
                        .is_empty()
                    {
                        CommonPressureGauge2d::BoundaryTraction
                    } else {
                        CommonPressureGauge2d::ZeroIntegral
                    };
                    let common = transient.common_projection();
                    let formulation = common.mixed_galerkin_correspondence();
                    common
                        .replay_mixed_galerkin_correspondence(&formulation)
                        .map_err(invalid)?;
                    (
                        CommonTransientResolvedSpatial::MiniP1(resolved),
                        CommonTransientFormulation::MixedGalerkin(Box::new(formulation)),
                        Space::simplex_p1_bubble(),
                        Space::continuous_lagrange(std::num::NonZeroU16::MIN),
                        gauge,
                    )
                }
                (
                    NativeSpatialPolicy::TransientMiniP1(scales),
                    NativeMeshResources::GmshSimplicial { mesh, .. },
                ) => {
                    let RecognizedNativeModel::TransientGeometry(binding) =
                        admission.recognized_model()
                    else {
                        return Err(invalid(
                            "Gmsh transient Mesh requires Geometry-backed Model meaning",
                        ));
                    };
                    let plan = binding.mini_plan(
                        mesh.artifact_reference()?,
                        scales,
                        temporal.step(),
                        nonlinear,
                        solver,
                    )?;
                    let capabilities = transient_realization_capabilities(
                        DiscretizationMethod::ContinuousGalerkin,
                        MeshKind::ImportedAffineSimplicial,
                        solver,
                        &admission.linear.capabilities,
                    )?;
                    let resolved = resolve_transient_fieldwise(
                        &TransientFieldwiseRealizationRequest::explicit(
                            admission.program().model(),
                            SemanticRevision::new(admission.program().revision().0),
                            RealizationRevision::new(TRANSIENT_REALIZATION_REVISION),
                            plan,
                        ),
                        binding.fieldwise_requirements(),
                        &capabilities,
                    )?;
                    let gauge = if resolved
                        .plan()
                        .fieldwise()
                        .spatial()
                        .constraints()
                        .is_empty()
                    {
                        CommonPressureGauge2d::BoundaryTraction
                    } else {
                        CommonPressureGauge2d::ZeroIntegral
                    };
                    let formulation = binding.model().mixed_galerkin_correspondence();
                    binding
                        .model()
                        .replay_mixed_galerkin_correspondence(&formulation)
                        .map_err(invalid)?;
                    (
                        CommonTransientResolvedSpatial::MiniP1(resolved),
                        CommonTransientFormulation::MixedGalerkin(Box::new(formulation)),
                        Space::simplex_p1_bubble(),
                        Space::continuous_lagrange(std::num::NonZeroU16::MIN),
                        gauge,
                    )
                }
                (
                    NativeSpatialPolicy::TransientCellCentered(scales),
                    NativeMeshResources::Cartesian { mesh, .. },
                ) => {
                    let RecognizedNativeModel::Transient(transient) = admission.recognized_model()
                    else {
                        return Err(invalid(
                            "Cartesian transient Mesh requires Cartesian Model meaning",
                        ));
                    };
                    let artifact = mesh.artifact_reference()?;
                    let cells =
                        [
                            NonZeroUsize::new(mesh.mesh().axis_cell_count(0).ok_or_else(|| {
                                invalid("supplied Cartesian mesh omitted x cells")
                            })?)
                            .ok_or_else(|| invalid("supplied Cartesian x cell count is zero"))?,
                            NonZeroUsize::new(mesh.mesh().axis_cell_count(1).ok_or_else(|| {
                                invalid("supplied Cartesian mesh omitted y cells")
                            })?)
                            .ok_or_else(|| invalid("supplied Cartesian y cell count is zero"))?,
                        ];
                    let plan = transient_navier_stokes_cell_centered_plan_2d(
                        transient,
                        MeshPolicy::SuppliedCartesian { artifact, cells },
                        scales,
                        temporal.step(),
                        nonlinear,
                        solver,
                    )?;
                    let capabilities = transient_realization_capabilities(
                        DiscretizationMethod::CellCenteredFiniteVolume,
                        MeshKind::SuppliedCartesian,
                        solver,
                        &admission.linear.capabilities,
                    )?;
                    let resolved = resolve_transient_cell_centered_incompressible_flow(
                        &TransientCellCenteredIncompressibleFlowRealizationRequest::explicit(
                            admission.program().model(),
                            SemanticRevision::new(admission.program().revision().0),
                            RealizationRevision::new(TRANSIENT_REALIZATION_REVISION),
                            plan,
                        ),
                        transient_navier_stokes_cell_centered_requirements_2d(transient),
                        &TransientCellCenteredIncompressibleFlowCapabilities::new(capabilities),
                    )?;
                    (
                        CommonTransientResolvedSpatial::CellCentered(resolved),
                        CommonTransientFormulation::IntegralConservative(Box::new(
                            integral_conservative_correspondence(transient),
                        )),
                        Space::cell_constant(),
                        Space::cell_constant(),
                        CommonPressureGauge2d::ZeroIntegral,
                    )
                }
                _ => {
                    return Err(invalid(
                        "transient spatial policy and exact caller Mesh are cross-wired",
                    ));
                }
            };
        let portable = match &resolved {
            CommonTransientResolvedSpatial::MiniP1(resolved) => resolved.portable_graph()?,
            CommonTransientResolvedSpatial::CellCentered(resolved) => resolved.portable_graph()?,
        };
        let realization_digest = hex_bytes(&portable.digest()?);
        let digests = resource_digests(admission.resources())?;
        let receipt_digest = scaling.receipt().provenance_digest();
        let mut identity_bytes = Vec::new();
        for value in [
            admission.model_digest(),
            model_id.as_str(),
            digests.geometry.as_str(),
            digests.mesh.as_str(),
            digests.correspondence.as_str(),
            digests.production.as_str(),
            admission.policy_identity(),
            receipt_digest.as_str(),
            velocity_field_id.as_str(),
            pressure_field_id.as_str(),
            realization_digest.as_str(),
        ] {
            push_framed(&mut identity_bytes, value.as_bytes());
        }
        identity_bytes.extend_from_slice(&model_reference.semantic_revision().get().to_be_bytes());
        identity_bytes.extend_from_slice(&TRANSIENT_REALIZATION_REVISION.to_be_bytes());
        identity_bytes.extend_from_slice(&COMMON_TRANSIENT_RESOLVER_EPOCH.to_be_bytes());
        match admission.resources() {
            NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => {
                push_framed(&mut identity_bytes, b"imported-affine-simplicial");
                push_framed(
                    &mut identity_bytes,
                    mesh.artifact_reference()?.sha256().as_slice(),
                );
            }
            NativeMeshResources::GmshSimplicial { mesh, .. } => {
                push_framed(&mut identity_bytes, b"gmsh-simplicial");
                push_framed(
                    &mut identity_bytes,
                    mesh.artifact_reference()?.sha256().as_slice(),
                );
            }
            NativeMeshResources::Cartesian { mesh, .. } => {
                push_framed(&mut identity_bytes, b"supplied-cartesian");
                push_framed(
                    &mut identity_bytes,
                    mesh.artifact_reference()?.sha256().as_slice(),
                );
                for axis in 0..2 {
                    let count = mesh.mesh().axis_cell_count(axis).ok_or_else(|| {
                        invalid("supplied Cartesian transient Mesh omitted an axis")
                    })?;
                    identity_bytes.extend_from_slice(&count.to_be_bytes());
                }
            }
            NativeMeshResources::AdjacentPartitionSimplicial { .. } => {
                return Err(invalid(
                    "transient common Plan requires the exact caller affine-triangle or supplied-Cartesian envelope",
                ));
            }
        }
        push_framed(&mut identity_bytes, space_identity(velocity_space));
        push_framed(&mut identity_bytes, space_identity(pressure_space));
        push_framed(
            &mut identity_bytes,
            match gauge {
                CommonPressureGauge2d::ZeroIntegral => b"zero-integral",
                CommonPressureGauge2d::BoundaryTraction => b"boundary-traction",
            },
        );
        push_framed(&mut identity_bytes, formulation.identity());
        push_framed(&mut identity_bytes, formulation_selection.identity());
        for discriminant in [
            b"f64".as_slice(),
            b"replicated".as_slice(),
            b"general-operator".as_slice(),
            b"host-cpu".as_slice(),
            b"host-serial".as_slice(),
            b"offline".as_slice(),
        ] {
            push_framed(&mut identity_bytes, discriminant);
        }
        let identity =
            domain_separated_identity(b"eqiora.common-transient-flow-plan/v1\0", &identity_bytes);
        let lineage = CommonSpatialPlanLineage::new(
            identity,
            model_id,
            model_reference.semantic_revision().get(),
            digests,
            realization_digest,
        );
        Ok(Self {
            admission,
            resolved,
            portable,
            formulation,
            formulation_selection,
            scaling,
            temporal,
            nonlinear,
            lineage,
            velocity_field_id,
            pressure_field_id,
            velocity_space,
            pressure_space,
            gauge,
        })
    }

    #[must_use]
    pub fn identity(&self) -> &str {
        self.lineage.identity()
    }
    #[must_use]
    pub fn model_id(&self) -> &str {
        self.lineage.model_id()
    }
    #[must_use]
    pub const fn model_revision(&self) -> u64 {
        self.lineage.model_revision()
    }
    #[must_use]
    pub fn model_digest(&self) -> &str {
        self.admission.model_digest()
    }
    #[must_use]
    pub fn geometry_digest(&self) -> &str {
        self.lineage.geometry_digest()
    }
    #[must_use]
    pub fn mesh_digest(&self) -> &str {
        self.lineage.mesh_digest()
    }
    #[must_use]
    pub fn correspondence_digest(&self) -> &str {
        self.lineage.correspondence_digest()
    }
    #[must_use]
    pub fn production_digest(&self) -> &str {
        self.lineage.production_digest()
    }
    #[must_use]
    pub fn realization_digest(&self) -> &str {
        self.lineage.realization_digest()
    }
    /// Canonical portable numerical realization owned by this Plan.
    #[must_use]
    pub const fn portable_realization(&self) -> &PortableRealizationGraph {
        &self.portable
    }
    #[must_use]
    pub const fn scaling_receipt(&self) -> &IncompressibleScalingReceipt2d {
        self.scaling.receipt()
    }
    #[must_use]
    pub const fn scales(&self) -> IncompressibleFlowScaleProfile2d {
        self.scaling.scales()
    }
    #[must_use]
    pub const fn temporal(&self) -> CommonBackwardEuler {
        self.temporal
    }
    #[must_use]
    pub const fn nonlinear(&self) -> NonlinearSolvePlan {
        self.nonlinear
    }
    #[must_use]
    pub const fn linear(&self) -> SolverPlan {
        self.admission.linear.solver
    }
    #[must_use]
    pub const fn solver_provider(&self) -> SolverProvider {
        self.admission.linear.provider
    }
    #[must_use]
    pub const fn solver_planning_objective(&self) -> Option<SolverPlanningObjective> {
        self.admission.linear.planning_objective
    }
    #[must_use]
    pub const fn solver_planning_policy_id(&self) -> Option<&'static str> {
        self.admission.linear.planning_policy_id
    }
    #[must_use]
    pub const fn selected_solver_candidate_id(&self) -> Option<&'static str> {
        self.admission.linear.selected_candidate_id
    }
    #[must_use]
    pub const fn selected_solver_evidence_case(&self) -> Option<&'static str> {
        self.admission.linear.selected_evidence_case
    }
    #[must_use]
    pub fn solver_planning_reasons(&self) -> &[(&'static str, &'static str)] {
        &self.admission.linear.planning_reasons
    }
    #[must_use]
    pub fn velocity_field_id(&self) -> &str {
        &self.velocity_field_id
    }
    #[must_use]
    pub fn pressure_field_id(&self) -> &str {
        &self.pressure_field_id
    }
    #[must_use]
    pub fn domain_id(&self) -> String {
        match self.admission.recognized_model() {
            RecognizedNativeModel::Transient(model) => model.domain().ulid().to_string(),
            RecognizedNativeModel::TransientGeometry(binding) => {
                binding.domain().ulid().to_string()
            }
            _ => unreachable!("common transient Plan retains transient Model meaning"),
        }
    }
    #[must_use]
    pub const fn velocity_space(&self) -> Space {
        self.velocity_space
    }
    #[must_use]
    pub const fn pressure_space(&self) -> Space {
        self.pressure_space
    }
    #[must_use]
    pub const fn gauge(&self) -> CommonPressureGauge2d {
        self.gauge
    }
    #[must_use]
    pub const fn spatial(&self) -> CommonSpatialPolicy {
        match self.resolved {
            CommonTransientResolvedSpatial::MiniP1(_) => CommonSpatialPolicy::MiniP1,
            CommonTransientResolvedSpatial::CellCentered(_) => CommonSpatialPolicy::CellCentered,
        }
    }

    /// Identity of the complete restartable state space, excluding solve and Run controls.
    #[must_use]
    pub fn state_space_identity(&self) -> String {
        let mut bytes = Vec::new();
        for value in [
            self.model_digest(),
            self.geometry_digest(),
            self.mesh_digest(),
            self.correspondence_digest(),
            self.production_digest(),
            self.velocity_field_id(),
            self.pressure_field_id(),
            "f64",
            "replicated",
        ] {
            push_framed(&mut bytes, value.as_bytes());
        }
        push_framed(&mut bytes, space_identity(self.velocity_space()));
        push_framed(&mut bytes, space_identity(self.pressure_space()));
        push_framed(
            &mut bytes,
            match self.gauge() {
                CommonPressureGauge2d::ZeroIntegral => b"zero-integral",
                CommonPressureGauge2d::BoundaryTraction => b"boundary-traction",
            },
        );
        push_framed(&mut bytes, self.formulation.identity());
        push_framed(
            &mut bytes,
            match self.spatial() {
                CommonSpatialPolicy::MiniP1 => b"mini-p1/backward-euler/no-extra-history/v1",
                CommonSpatialPolicy::CellCentered => {
                    b"cell-centered/backward-euler/bdf1-previous-accepted-face-volume-flux/v1"
                }
                _ => unreachable!("closed transient spatial policy"),
            },
        );
        hex_bytes(&Sha256::digest(
            [
                b"eqiora.common-state-space/v1\0".as_slice(),
                bytes.as_slice(),
            ]
            .concat(),
        ))
    }

    /// Construct the sole explicit homogeneous-zero bootstrap for this transient Plan.
    pub fn zero_state(&self, time_s: f64) -> Result<CommonState, Diagnostic> {
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        let RecognizedNativeModel::Transient(model) = self.admission.recognized_model() else {
            return Err(invalid("State.zero requires a transient Plan"));
        };
        crate::canonical_stokes::require_complete_zero_trace(model)?;
        let time = DynQuantity::new(time_s, TIME);
        let kind = match (&self.resolved, self.admission.resources()) {
            (
                CommonTransientResolvedSpatial::MiniP1(_),
                NativeMeshResources::AffineTriangleSimplicial { mesh, .. },
            ) => {
                let mesh_data = mesh.mesh().clone();
                let vertex_count = mesh_data.vertices().len();
                let cell_count = mesh_data.entity_count(2).ok_or_else(|| {
                    invalid("affine-triangle transient Mesh omitted two-dimensional cells")
                })?;
                let reference = match self.gauge {
                    CommonPressureGauge2d::ZeroIntegral => {
                        SteadyStokesPressureReference2d::ZeroIntegral { multiplier: 0.0 }
                    }
                    CommonPressureGauge2d::BoundaryTraction => {
                        SteadyStokesPressureReference2d::BoundaryTraction
                    }
                };
                CommonStateKind::MiniP1(Box::new(TransientNavierStokesInitialState2d::new(
                    model,
                    time,
                    mesh.artifact_reference()?,
                    SimplicialMiniVelocityField2d::new(
                        mesh_data.clone(),
                        vec![[0.0; 2]; vertex_count],
                        vec![[0.0; 2]; cell_count],
                    )?,
                    SimplicialP1Field::new(mesh_data, vec![0.0; vertex_count])?,
                    reference,
                )?))
            }
            (
                CommonTransientResolvedSpatial::CellCentered(_),
                NativeMeshResources::Cartesian { mesh, .. },
            ) => {
                let mesh_data = mesh.mesh().clone();
                let cell_count = mesh_data.entity_count(2).ok_or_else(|| {
                    invalid("Cartesian transient Mesh omitted two-dimensional cells")
                })?;
                let facet_count =
                    crate::cartesian_fvm_geometry::cartesian_fvm_geometry::<2>(&mesh_data)?
                        .1
                        .len();
                CommonStateKind::CellCentered(Box::new(
                    CellCenteredNavierStokesInitialState2d::new(
                        model,
                        time,
                        CellCenteredVelocityField2d::new(
                            mesh_data.clone(),
                            vec![[0.0; 2]; cell_count],
                        )?,
                        CellCenteredPressureField2d::new(mesh_data, vec![0.0; cell_count])?,
                        0.0,
                        vec![0.0; facet_count],
                    )?,
                ))
            }
            _ => {
                return Err(invalid(
                    "transient Plan lost its exact caller Mesh envelope",
                ));
            }
        };
        CommonState::new(
            self.state_space_identity(),
            time_s,
            Arc::new(self.admission.model().clone()),
            Arc::new(self.admission.resources().clone()),
            kind,
        )
    }

    /// Admit complete coherent-SI MINI/P1 coefficients for this transient Plan.
    pub fn initial_state(
        &self,
        time_s: f64,
        fields: Vec<CommonInitialField>,
    ) -> Result<CommonState, Diagnostic> {
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        if !time_s.is_finite() || time_s < 0.0 {
            return Err(invalid(
                "transient State.initial time_s must be finite and non-negative",
            ));
        }
        if fields.len() != 2 {
            return Err(invalid(
                "transient State.initial requires exactly velocity and pressure InitialField assignments",
            ));
        }
        let expected_model = self.admission.model().digest()?;
        let mut by_field = BTreeMap::new();
        for field in fields {
            if field.model() != &expected_model {
                return Err(invalid(
                    "InitialField belongs to a foreign or stale exact Model",
                ));
            }
            if by_field
                .insert(field.field().ulid().to_string(), field)
                .is_some()
            {
                return Err(invalid("State.initial repeats one exact FieldRef"));
            }
        }
        if by_field.keys().cloned().collect::<BTreeSet<_>>()
            != BTreeSet::from([
                self.velocity_field_id.clone(),
                self.pressure_field_id.clone(),
            ])
        {
            return Err(invalid(
                "transient State.initial assignments are not complete and exclusive for Plan.fields",
            ));
        }
        let velocity = &by_field[&self.velocity_field_id];
        let pressure = &by_field[&self.pressure_field_id];
        let velocity_vertices = match velocity.vertex() {
            Some(CommonInitialValues::Vector2(values)) => values.to_vec(),
            _ => return Err(invalid("MINI velocity requires vector2 vertex_values")),
        };
        let velocity_bubbles = match velocity.cell() {
            Some(CommonInitialValues::Vector2(values)) => values.to_vec(),
            _ => return Err(invalid("MINI velocity requires vector2 cell_values")),
        };
        let pressure_vertices = match pressure.vertex() {
            Some(CommonInitialValues::Scalar(values)) => values.to_vec(),
            _ => return Err(invalid("P1 pressure requires scalar vertex_values")),
        };
        if pressure.cell().is_some() {
            return Err(invalid("P1 pressure rejects cell_values"));
        }
        let (mesh, model) = match (
            self.admission.resources(),
            self.admission.recognized_model(),
        ) {
            (
                NativeMeshResources::AffineTriangleSimplicial { mesh, .. },
                RecognizedNativeModel::Transient(model),
            ) => (mesh, model.common_projection()),
            (
                NativeMeshResources::GmshSimplicial { mesh, .. },
                RecognizedNativeModel::TransientGeometry(binding),
            ) => (mesh, binding.model().clone()),
            _ => {
                return Err(invalid(
                    "transient State.initial currently requires a MINI/P1 simplicial Plan",
                ));
            }
        };
        let mesh_data = mesh.mesh().clone();
        let cell_count = mesh_data
            .entity_count(2)
            .ok_or_else(|| invalid("transient simplicial Mesh omitted two-dimensional cells"))?;
        if velocity_vertices.len() != mesh_data.vertices().len()
            || velocity_bubbles.len() != cell_count
            || pressure_vertices.len() != mesh_data.vertices().len()
        {
            return Err(invalid(
                "transient InitialField cardinality differs from exact MINI/P1 associations",
            ));
        }
        let reference = match self.gauge {
            CommonPressureGauge2d::ZeroIntegral => {
                SteadyStokesPressureReference2d::ZeroIntegral { multiplier: 0.0 }
            }
            CommonPressureGauge2d::BoundaryTraction => {
                SteadyStokesPressureReference2d::BoundaryTraction
            }
        };
        let time = DynQuantity::new(time_s, TIME);
        let initial = TransientNavierStokesInitialState2d::new_for_model(
            &model,
            time,
            mesh.artifact_reference()?,
            SimplicialMiniVelocityField2d::new(
                mesh_data.clone(),
                velocity_vertices,
                velocity_bubbles,
            )?,
            SimplicialP1Field::new(mesh_data, pressure_vertices)?,
            reference,
        )?;
        CommonState::new(
            self.state_space_identity(),
            time_s,
            Arc::new(self.admission.model().clone()),
            Arc::new(self.admission.resources().clone()),
            CommonStateKind::MiniP1(Box::new(initial)),
        )
    }

    /// Advance exactly one accepted Backward-Euler step from a compatible complete State.
    pub fn advance_one(
        &self,
        state: &CommonState,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonState, Diagnostic> {
        self.prepare_execution(state, backend)?.advance(state)
    }

    pub(super) fn authenticate_execution(
        &self,
        state: &CommonState,
        backend: &dyn LinearSolverBackend,
    ) -> Result<(), Diagnostic> {
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        if state.state_space_identity != self.state_space_identity() {
            return Err(invalid(
                "State belongs to a different exact common state space",
            ));
        }
        if backend.provider() != self.admission.linear.provider
            || backend.capabilities() != self.admission.linear.capabilities
        {
            return Err(invalid(
                "transient execution backend differs from the admitted provider or capabilities",
            ));
        }
        Ok(())
    }

    pub(super) fn prepare_execution<'a>(
        &'a self,
        state: &CommonState,
        backend: &'a dyn LinearSolverBackend,
    ) -> Result<PreparedCommonTransientExecution<'a>, Diagnostic> {
        self.authenticate_execution(state, backend)?;
        let spatial = match &self.resolved {
            CommonTransientResolvedSpatial::MiniP1(resolved) => {
                resolved.plan().fieldwise().spatial()
            }
            CommonTransientResolvedSpatial::CellCentered(resolved) => {
                resolved.plan().fieldwise().spatial()
            }
        };
        let structure = eqiora_solver::AlgebraicStructure::new(
            spatial.field_spaces().iter().map(|binding| binding.field()),
            spatial.constraints().iter().copied(),
        )?;
        let checked_backend = self
            .admission
            .linear
            .checked_backend(backend, Some(&structure))?;
        let method = match (&self.resolved, self.admission.resources()) {
            (
                CommonTransientResolvedSpatial::MiniP1(resolved),
                NativeMeshResources::AffineTriangleSimplicial { mesh, .. },
            ) => PreparedCommonTransientMethod::MiniP1(Box::new(
                prepare_resolved_transient_navier_stokes_mini_run_2d(
                    self.admission.program(),
                    resolved,
                    mesh,
                )?,
            )),
            (
                CommonTransientResolvedSpatial::MiniP1(resolved),
                NativeMeshResources::GmshSimplicial { .. },
            ) => {
                let RecognizedNativeModel::TransientGeometry(binding) =
                    self.admission.recognized_model()
                else {
                    return Err(invalid(
                        "Gmsh transient Plan lost Geometry-backed Model meaning",
                    ));
                };
                PreparedCommonTransientMethod::GeometryMiniP1(Box::new(
                    prepare_resolved_transient_navier_stokes_geometry_mini_run_2d(
                        self.admission.program(),
                        resolved,
                        binding,
                    )?,
                ))
            }
            (
                CommonTransientResolvedSpatial::CellCentered(resolved),
                NativeMeshResources::Cartesian { mesh, .. },
            ) => PreparedCommonTransientMethod::CellCentered(Box::new(
                prepare_resolved_transient_navier_stokes_cell_centered_run_2d(
                    self.admission.program(),
                    resolved,
                    mesh,
                )?,
            )),
            _ => {
                return Err(invalid(
                    "transient Plan and admitted mesh resources have no prepared method binding",
                ));
            }
        };
        Ok(PreparedCommonTransientExecution {
            plan: self,
            backend: checked_backend,
            method,
        })
    }
}

pub(super) fn mini_initial_from_resolved(
    plan: &CommonTransientFlowPlan,
    mesh: &SimplicialMeshEnvelopeV1,
    state: &ResolvedTransientNavierStokesState2d,
) -> Result<TransientNavierStokesInitialState2d, Diagnostic> {
    let model = match plan.admission.recognized_model() {
        RecognizedNativeModel::Transient(model) => model.common_projection(),
        RecognizedNativeModel::TransientGeometry(binding) => binding.model().clone(),
        _ => return Err(invalid("transient Plan lost recognized Model meaning")),
    };
    TransientNavierStokesInitialState2d::new_for_model(
        &model,
        state.time(),
        mesh.artifact_reference()?,
        state.velocity().clone(),
        state.pressure().clone(),
        state.pressure_reference(),
    )
}

pub(super) fn transient_realization_capabilities(
    method: DiscretizationMethod,
    mesh: MeshKind,
    solver: SolverPlan,
    backend: &SolverCapabilities,
) -> Result<RealizationCapabilities, Diagnostic> {
    backend.require_problem(solver, ScalarType::F64, LinearOperatorProperties::General)?;
    RealizationCapabilities::cartesian_product(
        [method],
        [(
            mesh,
            SpatialDimensionSupport::exact(NonZeroUsize::new(2).expect("two")),
        )],
        [VectorLayoutKind::Replicated],
        SolverCapabilities::exact([SolverCapability {
            algorithm: solver.algorithm(),
            operator_properties: LinearOperatorProperties::General,
            preconditioner: solver.preconditioner(),
            reduction: solver.reduction(),
            scalar_type: ScalarType::F64,
        }])?,
        TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
    )
}
