use super::*;

pub(super) struct PreparedCommonFsiExecution<'a> {
    plan: &'a CommonFsiPlan,
    backend: super::native::ProfileCheckedBackend<'a>,
    prepared: PreparedResolvedFixedReferenceFsiRun2d<'a>,
}

impl PreparedCommonFsiExecution<'_> {
    pub(super) fn advance(&self, state: &CommonState) -> Result<CommonState, Diagnostic> {
        let CommonStateKind::Fsi {
            state: previous, ..
        } = &state.kind
        else {
            return Err(invalid("FSI Plan received a non-FSI common State"));
        };
        let solution = self.prepared.finalize(previous)?.solve(&self.backend)?;
        let next = FixedReferenceFsiState::<2>::new(
            self.plan.mesh(),
            &self.plan.partition,
            solution.vertex_velocity_coefficients().to_vec(),
            solution.fluid_velocity_bubble_coefficients().to_vec(),
            solution.solid_displacement_coefficients().to_vec(),
        )?;
        CommonState::new(
            self.plan.state_space_identity(),
            state.time_s + self.plan.temporal.step().value(),
            Arc::new(self.plan.model().clone()),
            Arc::new(self.plan.resources().clone()),
            CommonStateKind::Fsi {
                state: Box::new(next),
                pressure: solution
                    .fluid_pressure_coefficients()
                    .to_vec()
                    .into_boxed_slice(),
                accepted: Some(Box::new(solution)),
            },
        )
    }
}

impl CommonFsiPlan {
    pub(super) fn model(&self) -> &ModelEnvelope {
        &self.recognized.model
    }

    fn canonical(&self) -> &FixedReferenceFsiCartesianModel2d {
        match &self.recognized.recognized {
            RecognizedNativeModel::Fsi(canonical) => canonical,
            _ => unreachable!("CommonFsiPlan retains recognized FSI meaning"),
        }
    }

    pub(super) fn resources(&self) -> &NativeMeshResources {
        &self.recognized.resources
    }

    fn reauthenticate_portable_realization(&self) -> Result<(), Diagnostic> {
        let relation = self
            .canonical()
            .solid()
            .kinematic_relation()
            .downcast::<eqiora_core::entity::kinds::Relation>()
            .ok_or_else(|| invalid("FSI solid kinematic Relation lost its semantic kind"))?;
        require_portable_realization(&self.portable, self.resolved.portable_graph(relation)?)
    }

    pub(super) fn from_recognized(
        model: &ModelEnvelope,
        recognized: RecognizedNativeAdmission,
        scaling_request: Option<IncompressibleScalingRequest2d>,
        temporal: CommonBackwardEuler,
        linear: NativeLinearPolicy,
    ) -> Result<Self, Diagnostic> {
        let RecognizedNativeModel::Fsi(canonical) = &recognized.recognized else {
            return Err(invalid("native FSI Plan requires recognized FSI meaning"));
        };
        let NativeMeshResources::AdjacentPartitionSimplicial {
            geometry,
            mesh,
            correspondence,
            ..
        } = &recognized.resources
        else {
            return Err(invalid(
                "native FSI Plan requires authenticated adjacent-partition simplicial resources",
            ));
        };
        validate_simplicial_resources(&recognized.resources)?;
        let native_mesh = mesh.mesh().clone();
        let entities = |name: &str| {
            correspondence.adjacent_rectangle_partition_entity_set_entities(geometry, name)
        };
        let region_set = |domain: eqiora_core::RawId| -> Result<&str, Diagnostic> {
            match recognized.program.node(domain) {
                Some(eqiora_schema::kernel::KernelNode::Domain(definition)) => {
                    match definition.kind() {
                        eqiora_schema::kernel::DomainKind::GeometryRegion {
                            entity_set, ..
                        } => Ok(entity_set),
                        _ => Err(invalid("FSI canonical subdomain is not a GeometryRegion")),
                    }
                }
                _ => Err(invalid(
                    "FSI canonical subdomain identity is absent from the exact Model",
                )),
            }
        };
        let fluid_set = region_set(canonical.fluid().domain())?;
        let solid_set = region_set(canonical.solid().continuum().domain())?;
        let interface_set = match recognized
            .program
            .node(canonical.interface().fluid().boundary())
        {
            Some(eqiora_schema::kernel::KernelNode::Domain(definition)) => {
                match definition.kind() {
                    eqiora_schema::kernel::DomainKind::GeometryBoundary { entity_set } => {
                        entity_set.as_str()
                    }
                    _ => return Err(invalid("FSI fluid interface is not a GeometryBoundary")),
                }
            }
            _ => {
                return Err(invalid(
                    "FSI fluid interface boundary is absent from the exact Model",
                ));
            }
        };
        let fluid_cells = entities(fluid_set)?
            .into_iter()
            .filter(|entity| entity.dimension() == 2)
            .map(|entity| CellId::new(entity.index()))
            .collect();
        let solid_cells = entities(solid_set)?
            .into_iter()
            .filter(|entity| entity.dimension() == 2)
            .map(|entity| CellId::new(entity.index()))
            .collect();
        let interface_facets = entities(interface_set)?
            .into_iter()
            .filter(|entity| entity.dimension() == 1)
            .map(|entity| FacetId::new(entity.index()))
            .collect();
        let partition = FixedReferenceFsiPartition::<2>::new(
            &native_mesh,
            fluid_cells,
            solid_cells,
            interface_facets,
        )?;
        let (geometry_artifact, mesh_artifact, correspondence_artifact, production_artifact) =
            resource_artifact_digests(&recognized.resources)?;
        let (bounds, _) = geometry
            .planar_adjacent_rectangle_partition()
            .ok_or_else(|| invalid("FSI scaling requires exact adjacent bounds"))?;
        let resolved_scaling = resolve_fixed_reference_fsi_scaling_2d(
            scaling_request,
            model.digest()?,
            geometry_artifact,
            correspondence_artifact,
            mesh_artifact,
            production_artifact,
            bounds[0],
            canonical.solid().continuum().shear_modulus(),
            canonical.solid().mass_density(),
            canonical.fluid().mass_density(),
        )?;
        let flow_scales = resolved_scaling.scales();
        let scaling_receipt = resolved_scaling.receipt().clone();
        let scaling = FixedReferenceFsiScaleProfile2d::new(
            flow_scales.length(),
            flow_scales.velocity(),
            flow_scales.pressure(),
        )?;
        let mesh_reference =
            MeshArtifactReference::from_sha256(mesh.artifact_reference()?.sha256());
        let realization_plan = fixed_reference_fsi_plan_2d(
            canonical,
            mesh_reference,
            temporal.step(),
            scaling,
            linear.solver,
        )?;
        let resolved = resolve_coupled_fieldwise(
            &CoupledFieldwiseRealizationRequest::explicit(
                recognized.program.model(),
                SemanticRevision::new(canonical.semantic_revision()),
                RealizationRevision::new(177),
                realization_plan,
            ),
            fixed_reference_fsi_requirements_2d(canonical),
            &RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
        )?;
        let solid_kinematic_relation = canonical
            .solid()
            .kinematic_relation()
            .downcast::<eqiora_core::entity::kinds::Relation>()
            .ok_or_else(|| invalid("FSI solid kinematic Relation lost its semantic kind"))?;
        let portable = resolved.portable_graph(solid_kinematic_relation)?;
        let reference = model.artifact_reference()?;
        let solver_provider = linear.provider;
        let execution_provider = linear.execution;
        let model_id = reference.model().ulid().to_string();
        let model_revision = reference.semantic_revision().get();
        let model_digest = recognized.model_digest.as_str();
        let digests = resource_digests(&recognized.resources)?;
        let field_ids = [
            canonical.fluid().velocity().ulid().to_string(),
            canonical.fluid().pressure().ulid().to_string(),
            canonical.solid().velocity().ulid().to_string(),
            canonical
                .solid()
                .continuum()
                .displacement()
                .ulid()
                .to_string(),
        ];
        let domain_ids = [
            canonical.fluid().domain().ulid().to_string(),
            canonical.solid().continuum().domain().ulid().to_string(),
        ];
        let mut identity_bytes = Vec::new();
        let realization_digest = hex_bytes(&portable.digest()?);
        let scaling_provenance_digest = scaling_receipt.provenance_digest().to_string();
        for value in [
            model_digest,
            &digests.geometry,
            &digests.mesh,
            &digests.correspondence,
            &digests.production,
            &realization_digest,
            &scaling_provenance_digest,
            solver_provider.id().as_str(),
            solver_provider.implementation_version(),
            execution_provider.id().as_str(),
            execution_provider.implementation_version(),
        ] {
            push_framed(&mut identity_bytes, value.as_bytes());
        }
        for library in linear.provider.libraries() {
            push_framed(&mut identity_bytes, library.name().as_bytes());
            push_framed(&mut identity_bytes, library.version().as_bytes());
        }
        if let Some(objective) = linear.planning_objective {
            push_framed(
                &mut identity_bytes,
                match objective {
                    SolverPlanningObjective::Robust => b"robust",
                    SolverPlanningObjective::Fast => b"fast",
                    SolverPlanningObjective::LowMemory => b"low-memory",
                },
            );
            push_framed(
                &mut identity_bytes,
                linear
                    .planning_policy_id
                    .expect("ranked policy identity")
                    .as_bytes(),
            );
            push_framed(
                &mut identity_bytes,
                linear
                    .selected_candidate_id
                    .expect("ranked candidate identity")
                    .as_bytes(),
            );
        }
        identity_bytes.extend_from_slice(&temporal.step().value().to_bits().to_be_bytes());
        let digest: [u8; 32] = Sha256::digest(identity_bytes).into();
        let identity = format!("common-fsi:{}", hex_bytes(&digest));
        let lineage = CommonSpatialPlanLineage::new(
            identity,
            model_id,
            model_revision,
            digests,
            realization_digest,
        );
        Ok(Self {
            recognized,
            partition,
            resolved,
            portable,
            scaling,
            scaling_receipt,
            temporal,
            linear,
            lineage,
            field_ids,
            domain_ids,
        })
    }

    pub(super) fn mesh(&self) -> &SimplicialMesh {
        let NativeMeshResources::AdjacentPartitionSimplicial { mesh, .. } = self.resources() else {
            unreachable!("CommonFsiPlan owns adjacent simplicial resources")
        };
        mesh.mesh()
    }

    pub fn state_space_identity(&self) -> String {
        let mut bytes = Vec::new();
        for value in [
            "fixed-reference-fsi/f64/replicated/mini-p1-fluid+p1-solid/shared-trace-quotient/gauge-free-pressure/backward-euler-velocity-displacement-history/v1",
            self.model_digest(),
            self.lineage.geometry_digest(),
            self.lineage.mesh_digest(),
            self.lineage.correspondence_digest(),
            self.lineage.production_digest(),
        ] {
            push_framed(&mut bytes, value.as_bytes());
        }
        for value in self.field_ids.iter().chain(self.domain_ids.iter()) {
            push_framed(&mut bytes, value.as_bytes());
        }
        hex_bytes(&Sha256::digest(bytes))
    }

    /// Admit complete exact-Field assignments for all four FSI Fields.
    pub fn initial_state(
        &self,
        time_s: f64,
        fields: Vec<CommonInitialField>,
    ) -> Result<CommonState, Diagnostic> {
        self.reauthenticate_portable_realization()?;
        if fields.len() != 4 {
            return Err(invalid(
                "FSI State.initial requires exactly four complete InitialField assignments",
            ));
        }
        let expected_model = self.model().digest()?;
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
        if by_field.keys().cloned().collect::<Vec<_>>()
            != self
                .field_ids
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
        {
            return Err(invalid(
                "FSI State.initial assignments are not complete and exclusive for Plan.fields",
            ));
        }
        let take_vector =
            |field: &CommonInitialField, association: &str| -> Result<Vec<[f64; 2]>, Diagnostic> {
                let values = match association {
                    "vertex" => field.vertex(),
                    "cell" => field.cell(),
                    _ => unreachable!(),
                };
                match values {
                    Some(CommonInitialValues::Vector2(values)) => Ok(values.to_vec()),
                    Some(CommonInitialValues::Scalar(_)) => Err(invalid(format!(
                        "FSI vector Field has scalar {association}_values"
                    ))),
                    None => Err(invalid(format!(
                        "FSI vector Field omitted required {association}_values"
                    ))),
                }
            };
        let fluid_velocity = &by_field[&self.field_ids[0]];
        let fluid_pressure = &by_field[&self.field_ids[1]];
        let solid_velocity = &by_field[&self.field_ids[2]];
        let solid_displacement = &by_field[&self.field_ids[3]];
        let fluid_velocity_vertices = take_vector(fluid_velocity, "vertex")?;
        let fluid_velocity_bubbles = take_vector(fluid_velocity, "cell")?;
        let fluid_pressure_vertices = match fluid_pressure.vertex() {
            Some(CommonInitialValues::Scalar(values)) => values.to_vec(),
            _ => return Err(invalid("FSI pressure requires scalar vertex_values")),
        };
        if fluid_pressure.cell().is_some()
            || solid_velocity.cell().is_some()
            || solid_displacement.cell().is_some()
        {
            return Err(invalid(
                "FSI P1 pressure/solid velocity/displacement reject unexpected cell_values",
            ));
        }
        let solid_velocity_vertices = take_vector(solid_velocity, "vertex")?;
        let solid_displacement_vertices = take_vector(solid_displacement, "vertex")?;
        if fluid_velocity_vertices.len() != self.partition.fluid_vertices().len()
            || fluid_velocity_bubbles.len() != self.partition.fluid_cells().len()
            || fluid_pressure_vertices.len() != self.partition.fluid_vertices().len()
            || solid_velocity_vertices.len() != self.partition.solid_vertices().len()
            || solid_displacement_vertices.len() != self.partition.solid_vertices().len()
        {
            return Err(invalid(
                "FSI InitialField cardinality differs from exact Field support and association",
            ));
        }
        if fluid_velocity_vertices
            .iter()
            .flatten()
            .chain(fluid_velocity_bubbles.iter().flatten())
            .chain(fluid_pressure_vertices.iter())
            .chain(solid_velocity_vertices.iter().flatten())
            .chain(solid_displacement_vertices.iter().flatten())
            .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "FSI InitialField values must be finite coherent-SI numbers",
            ));
        }
        let mut velocity = vec![[f64::NAN; 2]; self.mesh().vertices().len()];
        for (vertex, value) in self
            .partition
            .fluid_vertices()
            .iter()
            .zip(fluid_velocity_vertices)
        {
            velocity[vertex.index()] = value;
        }
        for (vertex, value) in self
            .partition
            .solid_vertices()
            .iter()
            .zip(solid_velocity_vertices)
        {
            let slot = &mut velocity[vertex.index()];
            if slot[0].is_finite() && *slot != value {
                return Err(invalid(
                    "fluid and solid initial velocity traces disagree on the shared interface quotient",
                ));
            }
            *slot = value;
        }
        if velocity.iter().flatten().any(|value| !value.is_finite()) {
            return Err(invalid(
                "FSI velocity supports do not cover the complete shared vertex quotient",
            ));
        }
        let mut displacement = vec![[0.0; 2]; self.mesh().vertices().len()];
        for (vertex, value) in self
            .partition
            .solid_vertices()
            .iter()
            .zip(solid_displacement_vertices)
        {
            displacement[vertex.index()] = value;
        }
        let native = FixedReferenceFsiState::<2>::new(
            self.mesh(),
            &self.partition,
            velocity,
            fluid_velocity_bubbles,
            displacement,
        )?;
        CommonState::new(
            self.state_space_identity(),
            time_s,
            Arc::new(self.model().clone()),
            Arc::new(self.resources().clone()),
            CommonStateKind::Fsi {
                state: Box::new(native),
                pressure: fluid_pressure_vertices.into_boxed_slice(),
                accepted: None,
            },
        )
    }

    /// Advance one exact accepted monolithic Backward-Euler transition.
    pub fn advance(
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
        if state.state_space_identity() != self.state_space_identity() {
            return Err(invalid(
                "FSI State belongs to an incompatible common state space",
            ));
        }
        if backend.provider() != self.linear.provider
            || backend.capabilities() != self.linear.capabilities
        {
            return Err(invalid(
                "FSI execution backend differs from admitted MINRES provider/capabilities",
            ));
        }
        Ok(())
    }

    pub(super) fn prepare_execution<'a>(
        &'a self,
        state: &CommonState,
        backend: &'a dyn LinearSolverBackend,
    ) -> Result<PreparedCommonFsiExecution<'a>, Diagnostic> {
        self.authenticate_execution(state, backend)?;
        let NativeMeshResources::AdjacentPartitionSimplicial { mesh, .. } = self.resources() else {
            unreachable!("FSI Plan owns adjacent resources")
        };
        let mesh_reference =
            MeshArtifactReference::from_sha256(mesh.artifact_reference()?.sha256());
        let prepared = prepare_resolved_fixed_reference_fsi_run_2d(
            self.canonical(),
            &self.resolved,
            mesh_reference,
            mesh.mesh(),
            &self.partition,
        )?;
        let spatial = self.resolved.plan().spatial();
        let structure = eqiora_solver::AlgebraicStructure::new(
            spatial
                .domains()
                .iter()
                .flat_map(|domain| domain.field_spaces().iter().map(|binding| binding.field())),
            spatial
                .domains()
                .iter()
                .flat_map(|domain| domain.constraints().iter().copied()),
        )?;
        Ok(PreparedCommonFsiExecution {
            plan: self,
            backend: self.linear.checked_backend(backend, Some(&structure))?,
            prepared,
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
        &self.recognized.model_digest
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
    #[must_use]
    pub const fn linear(&self) -> SolverPlan {
        self.linear.solver
    }
    #[must_use]
    pub const fn solver_provider(&self) -> SolverProvider {
        self.linear.provider
    }
    #[must_use]
    pub const fn solver_capabilities(&self) -> &SolverCapabilities {
        &self.linear.capabilities
    }
    #[must_use]
    pub const fn execution_provider(&self) -> ExecutionProvider {
        self.linear.execution
    }
    #[must_use]
    pub const fn workers(&self) -> NonZeroUsize {
        self.linear.workers
    }
    #[must_use]
    pub const fn temporal(&self) -> CommonBackwardEuler {
        self.temporal
    }
    #[must_use]
    pub const fn scaling(&self) -> FixedReferenceFsiScaleProfile2d {
        self.scaling
    }
    #[must_use]
    pub const fn scaling_receipt(&self) -> &IncompressibleScalingReceipt2d {
        &self.scaling_receipt
    }
    #[must_use]
    pub fn field_ids(&self) -> &[String; 4] {
        &self.field_ids
    }
    #[must_use]
    pub fn domain_ids(&self) -> &[String; 2] {
        &self.domain_ids
    }
    #[must_use]
    pub const fn portable_realization(&self) -> &PortableRealizationGraph {
        &self.portable
    }
    #[must_use]
    pub fn fluid_vertex_indices(&self) -> Vec<usize> {
        self.partition
            .fluid_vertices()
            .iter()
            .map(|id| id.index())
            .collect()
    }
    #[must_use]
    pub fn fluid_cell_indices(&self) -> Vec<usize> {
        self.partition
            .fluid_cells()
            .iter()
            .map(|id| id.index())
            .collect()
    }
    #[must_use]
    pub fn solid_cell_indices(&self) -> Vec<usize> {
        self.partition
            .solid_cells()
            .iter()
            .map(|id| id.index())
            .collect()
    }
    #[must_use]
    pub fn solid_vertex_indices(&self) -> Vec<usize> {
        self.partition
            .solid_vertices()
            .iter()
            .map(|id| id.index())
            .collect()
    }
    #[must_use]
    pub fn interface_facet_vertices(&self) -> Vec<[usize; 2]> {
        self.partition
            .interface_facets()
            .iter()
            .map(|facet| {
                let vertices = self
                    .mesh()
                    .entity_vertices(MeshEntity::new(1, facet.index()))
                    .expect("accepted FSI interface facet owns exact connectivity");
                [vertices[0].index(), vertices[1].index()]
            })
            .collect()
    }
}
