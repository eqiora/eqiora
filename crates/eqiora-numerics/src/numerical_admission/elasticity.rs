use super::*;

fn resolve_common_elasticity_portable(
    admission: &NativeNumericalAdmission,
    lowered: &IsotropicElasticityContinuum<2>,
    mesh: &CartesianMeshEnvelopeV1,
    cells: [usize; 2],
) -> Result<PortableRealizationGraph, Diagnostic> {
    if admission.spatial != NativeSpatialPolicy::ElasticityQ1 {
        return Err(invalid(
            "common elasticity portable graph received a non-elasticity spatial policy",
        ));
    }
    let domain = lowered
        .domain()
        .downcast::<eqiora_core::entity::kinds::Domain>()
        .ok_or_else(|| invalid("elasticity Domain lost its semantic kind"))?;
    let displacement = lowered
        .displacement()
        .downcast::<eqiora_core::entity::kinds::Field>()
        .ok_or_else(|| invalid("elasticity displacement lost its semantic Field kind"))?;
    let solver = admission.linear.solver;
    admission.linear.capabilities.require_problem(
        solver,
        ScalarType::F64,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )?;
    PortableRealizationGraph::linear_regions(
        RealizationLineage::explicit(
            admission.program().model(),
            SemanticRevision::new(admission.program().revision().0),
            RealizationRevision::new(COMMON_ELASTICITY_REALIZATION_REVISION),
        ),
        [eqiora_realization::DomainFieldDiscretization::new(
            domain,
            [eqiora_realization::FieldSpaceBinding::new(
                displacement,
                Space::continuous_lagrange(std::num::NonZeroU16::MIN),
            )],
            [],
        )?],
        [],
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::SuppliedCartesian {
                artifact: mesh.artifact_reference()?,
                cells: cells.map(|count| {
                    NonZeroUsize::new(count).expect("validated Cartesian cells are non-zero")
                }),
            },
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(2).expect("two is non-zero"),
            },
        ),
        LinearOperatorProperties::SymmetricPositiveDefinite,
        ScalarType::F64,
        VectorLayoutKind::Replicated,
        solver,
        Target::HostCpu {
            threads: admission.linear.workers,
        },
        ExecutionSchedule::Offline,
    )
}

impl CommonElasticityPlan {
    fn reauthenticate_portable_realization(&self) -> Result<(), Diagnostic> {
        let NativeMeshResources::Cartesian { mesh, .. } = self.admission.resources() else {
            return Err(invalid(
                "common elasticity Plan lost its exact Cartesian Mesh materialization",
            ));
        };
        let RecognizedNativeModel::Elasticity(lowered) = self.admission.recognized_model() else {
            return Err(invalid(
                "common elasticity Plan lost its recognized mathematical materialization",
            ));
        };
        require_portable_realization(
            &self.portable,
            resolve_common_elasticity_portable(&self.admission, lowered, mesh, self.cells)?,
        )
    }

    pub(super) fn from_admission(
        model: &ModelEnvelope,
        admission: NativeNumericalAdmission,
    ) -> Result<Self, Diagnostic> {
        let model_reference = model.artifact_reference()?;
        let NativeMeshResources::Cartesian { mesh, .. } = admission.resources() else {
            return Err(invalid(
                "linear-elasticity common Plan requires an authenticated Cartesian Mesh",
            ));
        };
        let cells = [
            mesh.mesh()
                .axis_cell_count(0)
                .ok_or_else(|| invalid("elasticity Plan Mesh omitted x-axis cells"))?,
            mesh.mesh()
                .axis_cell_count(1)
                .ok_or_else(|| invalid("elasticity Plan Mesh omitted y-axis cells"))?,
        ];
        let RecognizedNativeModel::Elasticity(lowered) = admission.recognized_model() else {
            return Err(invalid(
                "common elasticity Plan omitted recognized elasticity meaning",
            ));
        };
        let portable = resolve_common_elasticity_portable(&admission, lowered, mesh, cells)?;
        let realization_digest = hex_bytes(&portable.digest()?);
        let displacement_field_id = lowered.displacement().ulid().to_string();
        let (digests, identity_bytes) =
            static_plan_identity_lineage(&admission, &realization_digest)?;
        let identity = domain_separated_identity(
            b"eqiora.common-linear-elasticity-plan/v1\0",
            &identity_bytes,
        );
        let lineage = CommonSpatialPlanLineage::new(
            identity,
            model_reference.model().ulid().to_string(),
            model_reference.semantic_revision().get(),
            digests,
            realization_digest,
        );
        Ok(Self {
            admission,
            portable,
            lineage,
            displacement_field_id,
            cells,
        })
    }

    pub(crate) fn run(
        &self,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CartesianLinearElasticity2dSolution, Diagnostic> {
        self.reauthenticate_portable_realization()?;
        self.admission.execute_elasticity(backend)
    }

    /// Project scientific observations through this exact admitted Plan.
    pub(super) fn observe(
        &self,
        solution: &CartesianLinearElasticity2dSolution,
    ) -> Result<CommonElasticityObservation, Diagnostic> {
        self.admission.revalidate()?;
        let bounds = self
            .admission
            .resources()
            .geometry()
            .planar_rectangle_bounds()
            .copied()
            .ok_or_else(|| {
                invalid("linear-elasticity observation requires rectangular Geometry")
            })?;
        let constrained_reaction = solution.boundary_reaction();
        let integrated_body_force = solution.integrated_body_force();
        if bounds
            .iter()
            .flatten()
            .copied()
            .chain(constrained_reaction)
            .chain(integrated_body_force)
            .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "linear-elasticity observation contains a non-finite value",
            ));
        }
        Ok(CommonElasticityObservation {
            constrained_reaction,
            integrated_body_force,
            exact_bounds: bounds,
        })
    }

    /// Execute and authenticate observations without exposing a re-pairing seam.
    fn run_observed(
        &self,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonElasticityRunOutput, Diagnostic> {
        let solution = self.run(backend)?;
        let observation = self.observe(&solution)?;
        Ok(CommonElasticityRunOutput {
            plan_identity: self.identity().to_owned(),
            solution,
            observation,
        })
    }

    /// Execute solely from retained Plan state and publish one complete Result.
    pub fn run_result(
        &self,
        backend: &dyn LinearSolverBackend,
    ) -> Result<crate::CommonResult, Diagnostic> {
        crate::CommonResult::accept_elasticity(self.clone(), 0.0, self.run_observed(backend)?)
    }

    /// Exact selected solver release and library inventory.
    #[must_use]
    pub const fn solver_provider(&self) -> SolverProvider {
        self.admission.linear.provider
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
    pub fn displacement_field_id(&self) -> &str {
        &self.displacement_field_id
    }
    #[must_use]
    pub const fn cells(&self) -> [usize; 2] {
        self.cells
    }
    #[must_use]
    pub const fn linear(&self) -> SolverPlan {
        self.admission.linear.solver
    }
}
