use super::*;

impl CommonSteadyStokesPlan {
    pub(crate) fn boundary_observation_names(&self) -> (Vec<String>, Vec<String>) {
        self.binding.boundary_observation_names()
    }
    fn reauthenticate_portable_realization(&self) -> Result<(), Diagnostic> {
        require_portable_realization(&self.portable, self.resolved.portable_graph()?)
    }

    /// Effective mixed Galerkin Formulation and its automatic-selection audit.
    #[must_use]
    pub fn formulation(&self) -> CommonFormulationDescription {
        CommonFormulationDescription::mixed(
            self.binding.formulation_correspondence(),
            self.formulation_selection,
            match self.formulation_selection {
                FormulationSelectionMode::Automatic => {
                    "eqiora.formulation.auto.mixed-galerkin-for-mini-p1/v1"
                }
                FormulationSelectionMode::Exact => {
                    "eqiora.formulation.exact.mixed-galerkin-admitted/v1"
                }
                FormulationSelectionMode::Authored => {
                    unreachable!("authored steady-Stokes forms are not admitted")
                }
            },
        )
    }

    pub(super) fn from_admission(
        model: &ModelEnvelope,
        admission: NativeNumericalAdmission,
        formulation_selection: FormulationSelectionMode,
        scaling: ResolvedIncompressibleScaling2d,
    ) -> Result<Self, Diagnostic> {
        let model_reference = model.artifact_reference()?;
        let model_id = model_reference.model().ulid().to_string();
        let binding = admission.stokes_binding()?;
        let (resolved, portable, velocity_space, pressure_space) =
            admission.resolve_stokes(&binding)?;
        let mut velocity_field_id = None;
        let mut pressure_field_id = None;
        for field in resolved.plan().spatial().field_spaces() {
            match field.space().family() {
                SpaceFamily::SimplexP1Bubble => {
                    velocity_field_id = Some(field.field().ulid().to_string());
                }
                SpaceFamily::ContinuousLagrange { order } if order == std::num::NonZeroU16::MIN => {
                    pressure_field_id = Some(field.field().ulid().to_string());
                }
                _ => {}
            }
        }
        let (velocity_field_id, pressure_field_id) = velocity_field_id
            .zip(pressure_field_id)
            .ok_or_else(|| invalid("steady-Stokes Plan omitted its MINI/P1 Field identities"))?;
        let realization_digest = hex_bytes(&portable.digest()?);
        let scaling_provenance_digest = scaling.receipt().provenance_digest();
        let (digests, mut identity_bytes) =
            static_plan_identity_lineage(&admission, &realization_digest)?;
        push_framed(
            &mut identity_bytes,
            scaling_provenance_digest.as_str().as_bytes(),
        );
        push_framed(&mut identity_bytes, formulation_selection.identity());
        let identity =
            domain_separated_identity(b"eqiora.common-steady-stokes-plan/v1\0", &identity_bytes);
        let lineage = CommonSpatialPlanLineage::new(
            identity,
            model_id,
            model_reference.semantic_revision().get(),
            digests,
            realization_digest,
        );
        Ok(Self {
            admission,
            binding,
            resolved,
            portable,
            formulation_selection,
            scaling,
            lineage,
            velocity_field_id,
            pressure_field_id,
            velocity_space,
            pressure_space,
        })
    }

    /// Exact selected solver release and library inventory.
    #[must_use]
    pub const fn solver_provider(&self) -> SolverProvider {
        self.admission.linear.provider
    }

    /// Execute solely from the state retained by this Plan.
    fn run(
        &self,
        backend: &dyn LinearSolverBackend,
    ) -> Result<crate::fluid::SteadyStokesMiniSolution2d, Diagnostic> {
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        if backend.provider() != self.admission.linear.provider
            || backend.capabilities() != self.admission.linear.capabilities
        {
            return Err(invalid(
                "steady-Stokes execution backend differs from the admitted provider or capabilities",
            ));
        }
        let spatial = self.resolved.plan().spatial();
        let structure = eqiora_solver::AlgebraicStructure::new(
            spatial.field_spaces().iter().map(|binding| binding.field()),
            spatial.constraints().iter().copied(),
        )?;
        let checked_backend = self
            .admission
            .linear
            .checked_backend(backend, Some(&structure))?;
        let solution = solve_resolved_steady_stokes_geometry_mini_2d(
            self.admission.program(),
            &self.resolved,
            &self.binding,
            &checked_backend,
        )?;
        if solution.scales() != self.scaling.scales() {
            return Err(invalid(
                "steady-Stokes execution changed the Plan-owned effective scaling",
            ));
        }
        Ok(solution)
    }

    /// Project scientific observations only after reauthenticating this Plan's roles.
    pub(super) fn observe(
        &self,
        solution: &crate::fluid::SteadyStokesMiniSolution2d,
    ) -> Result<CommonSteadyStokesObservation, Diagnostic> {
        self.admission.revalidate()?;
        if solution.velocity_field().ulid().to_string() != self.velocity_field_id
            || solution.pressure_field().ulid().to_string() != self.pressure_field_id
            || solution.scales() != self.scaling.scales()
        {
            return Err(invalid(
                "steady-Stokes observation crossed a different Plan or effective scaling",
            ));
        }
        let bounds = self
            .admission
            .resources()
            .geometry()
            .circular_hole_bounds()
            .copied()
            .ok_or_else(|| invalid("steady-Stokes observation requires circular-hole Geometry"))?;
        let pressure = solution.pressure().vertex_values();
        let pressure_minimum = pressure
            .iter()
            .copied()
            .min_by(f64::total_cmp)
            .ok_or_else(|| invalid("steady-Stokes observation has no pressure values"))?;
        let pressure_maximum = pressure
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .ok_or_else(|| invalid("steady-Stokes observation has no pressure values"))?;
        let net_flux = self
            .binding
            .boundary_names()
            .filter_map(|name| solution.named_boundary_flux(name))
            .sum::<f64>();
        let constrained_reaction = solution.boundary_reaction();
        let integrated_body_force = solution.integrated_body_force();
        let integrated_boundary_traction = solution.integrated_boundary_traction();
        let momentum_closure = std::array::from_fn(|component| {
            constrained_reaction[component]
                + integrated_body_force[component]
                + integrated_boundary_traction[component]
        });
        let continuity_residual_norm = solution.dimensionless_solution().continuity_residual_norm();
        if bounds
            .iter()
            .flatten()
            .copied()
            .chain([pressure_minimum, pressure_maximum, net_flux])
            .chain(constrained_reaction)
            .chain(integrated_body_force)
            .chain(integrated_boundary_traction)
            .chain(momentum_closure)
            .chain([continuity_residual_norm])
            .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "steady-Stokes observation contains a non-finite value",
            ));
        }
        Ok(CommonSteadyStokesObservation {
            pressure_minimum,
            pressure_maximum,
            exact_bounds: bounds,
            net_flux,
            constrained_reaction,
            integrated_body_force,
            integrated_boundary_traction,
            momentum_closure,
            continuity_residual_norm,
        })
    }

    /// Execute and authenticate observations without exposing a re-pairing seam.
    pub(crate) fn run_observed(
        &self,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonSteadyStokesRunOutput, Diagnostic> {
        let solution = self.run(backend)?;
        let observation = self.observe(&solution)?;
        Ok(CommonSteadyStokesRunOutput {
            plan_identity: self.identity().to_owned(),
            solution,
            observation,
        })
    }

    /// Execute directly into the common producer-independent Result authority.
    pub fn run_result(
        &self,
        backend: &dyn LinearSolverBackend,
    ) -> Result<crate::CommonResult, Diagnostic> {
        crate::CommonResult::accept_steady_stokes(self.clone(), 0.0, self.run_observed(backend)?)
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
    pub fn velocity_field_id(&self) -> &str {
        &self.velocity_field_id
    }
    #[must_use]
    pub fn pressure_field_id(&self) -> &str {
        &self.pressure_field_id
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
    pub const fn scales(&self) -> IncompressibleFlowScaleProfile2d {
        self.scaling.scales()
    }
    #[must_use]
    pub const fn linear(&self) -> SolverPlan {
        self.admission.linear.solver
    }
}
