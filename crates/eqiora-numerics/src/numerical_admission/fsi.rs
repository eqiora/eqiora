use super::*;
use eqiora_core::{Id, entity::kinds};

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
        CommonState::new(
            self.plan.state_space_identity(),
            state.time_s + self.plan.temporal.step().value(),
            Arc::new(self.plan.model().clone()),
            Arc::new(self.plan.resources().clone()),
            CommonStateKind::Fsi {
                state: Box::new(solution.state().clone()),
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
        require_portable_realization(&self.portable, self.resolved.portable_graph()?)
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
        let (geometry, mesh, correspondence) = match &recognized.resources {
            NativeMeshResources::AdjacentPartitionSimplicial {
                geometry,
                mesh,
                correspondence,
                ..
            }
            | NativeMeshResources::GmshSimplicial {
                geometry,
                mesh,
                correspondence,
                ..
            } => (geometry, mesh, correspondence),
            _ => {
                return Err(invalid(
                    "FSI Plan requires authenticated conforming region simplicial resources",
                ));
            }
        };
        validate_simplicial_resources(&recognized.resources)?;
        let native_mesh = mesh.mesh().clone();
        let entities = |name: &str| -> Result<Vec<MeshEntity>, Diagnostic> {
            match &recognized.resources {
                NativeMeshResources::AdjacentPartitionSimplicial { .. } => {
                    correspondence.adjacent_rectangle_partition_entity_set_entities(geometry, name)
                }
                NativeMeshResources::GmshSimplicial { .. } => correspondence
                    .region_entity_set_entities(
                        &eqiora_artifact::GeometryDefinitionV1::from_canonical(geometry)?,
                        name,
                    ),
                _ => unreachable!("authenticated simplicial region resources"),
            }
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
        let domain_cells = canonical
            .fluids()
            .map(|fluid| fluid.domain())
            .chain(canonical.solids().map(|solid| solid.continuum().domain()))
            .map(|domain| {
                let selected = entities(region_set(domain)?)?;
                if selected.iter().any(|entity| entity.dimension() != 2) {
                    return Err(invalid(
                        "Region correspondence includes an entity outside cell support",
                    ));
                }
                Ok((
                    domain.downcast().expect("Domain"),
                    selected
                        .into_iter()
                        .map(|entity| CellId::new(entity.index()))
                        .collect::<Vec<_>>(),
                ))
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let (geometry_artifact, mesh_artifact, correspondence_artifact, production_artifact) =
            resource_artifact_digests(&recognized.resources)?;
        let spans = canonical
            .fluids()
            .map(|fluid| fluid.bounds()[0])
            .chain(
                canonical
                    .solids()
                    .map(|solid| solid.continuum().bounds()[0]),
            )
            .collect::<Vec<_>>();
        let bounds = [
            spans
                .iter()
                .map(|span| span[0])
                .fold(f64::INFINITY, f64::min),
            spans
                .iter()
                .map(|span| span[1])
                .fold(f64::NEG_INFINITY, f64::max),
        ];
        let resolved_scaling = resolve_fixed_reference_fsi_scaling_2d(
            scaling_request,
            model.digest()?,
            geometry_artifact,
            correspondence_artifact,
            mesh_artifact,
            production_artifact,
            bounds,
            &canonical
                .solids()
                .map(|solid| (solid.continuum().shear_modulus(), solid.mass_density()))
                .collect::<Vec<_>>(),
            &canonical
                .fluids()
                .map(|fluid| fluid.mass_density())
                .collect::<Vec<_>>(),
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
        let partition = FixedReferenceFsiPartition::<2>::new(
            &native_mesh,
            domain_cells,
            realization_plan.spatial().trace_quotients(),
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
        let portable = resolved.portable_graph()?;
        let reference = model.artifact_reference()?;
        let solver_provider = linear.provider;
        let execution_provider = linear.execution;
        let model_id = reference.model().ulid().to_string();
        let model_revision = reference.semantic_revision().get();
        let model_digest = recognized.model_digest.as_str();
        let digests = resource_digests(&recognized.resources)?;
        let field_ids = resolved
            .plan()
            .spatial()
            .domains()
            .iter()
            .flat_map(|domain| domain.field_spaces().iter().map(|binding| binding.field()))
            .chain(
                resolved
                    .plan()
                    .time_step()
                    .eliminated_states()
                    .iter()
                    .map(|binding| binding.pair().state()),
            )
            .map(|field| field.ulid().to_string())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let domain_ids = resolved
            .plan()
            .spatial()
            .domains()
            .iter()
            .map(|domain| domain.domain().ulid().to_string())
            .collect();
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
        let (NativeMeshResources::AdjacentPartitionSimplicial { mesh, .. }
        | NativeMeshResources::GmshSimplicial { mesh, .. }) = self.resources()
        else {
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

    /// Admit complete exact-Field assignments for every represented Field.
    pub fn initial_state(
        &self,
        time_s: f64,
        fields: Vec<CommonInitialField>,
    ) -> Result<CommonState, Diagnostic> {
        self.reauthenticate_portable_realization()?;
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
        let mut values = Vec::new();
        for field in by_field.values() {
            let (domain, space) = self.field_support(field.field())?;
            let vertices = self
                .partition
                .domain_vertices(domain)
                .ok_or_else(|| invalid("missing exact Field Domain"))?;
            let cells = self
                .partition
                .domain_cells(domain)
                .ok_or_else(|| invalid("missing exact Field Domain"))?;
            let mut coefficients = Vec::new();
            let mut append = |entities: Vec<MeshEntity>,
                              input: Option<&CommonInitialValues>|
             -> Result<(), Diagnostic> {
                let input = input
                    .ok_or_else(|| invalid("InitialField omitted required entity association"))?;
                match input {
                    CommonInitialValues::Scalar(data) => {
                        if data.len() != entities.len() {
                            return Err(invalid(
                                "InitialField scalar cardinality differs from exact support",
                            ));
                        }
                        for (entity, &value) in entities.into_iter().zip(data.iter()) {
                            coefficients.push((entity, 0, 0, value));
                        }
                    }
                    CommonInitialValues::Vector2(data) => {
                        if data.len() != entities.len() {
                            return Err(invalid(
                                "InitialField vector cardinality differs from exact support",
                            ));
                        }
                        for (entity, vector) in entities.into_iter().zip(data.iter()) {
                            for (component, &value) in vector.iter().enumerate() {
                                coefficients.push((entity, 0, component, value));
                            }
                        }
                    }
                }
                Ok(())
            };
            append(
                vertices
                    .iter()
                    .map(|v| MeshEntity::new(0, v.index()))
                    .collect(),
                field.vertex(),
            )?;
            match space.family() {
                eqiora_realization::SpaceFamily::SimplexP1Bubble => append(
                    cells
                        .iter()
                        .map(|c| MeshEntity::new(2, c.index()))
                        .collect(),
                    field.cell(),
                )?,
                eqiora_realization::SpaceFamily::ContinuousLagrange { order }
                    if order.get() == 1 && field.cell().is_none() => {}
                _ => {
                    return Err(invalid(
                        "InitialField association differs from exact admitted space",
                    ));
                }
            }
            values.push((field.field(), coefficients));
        }
        let native = FixedReferenceFsiState::<2>::new(
            &self.recognized.program,
            self.resolved.plan(),
            self.mesh(),
            &self.partition,
            values,
        )?;
        CommonState::new(
            self.state_space_identity(),
            time_s,
            Arc::new(self.model().clone()),
            Arc::new(self.resources().clone()),
            CommonStateKind::Fsi {
                state: Box::new(native),
                accepted: None,
            },
        )
    }

    fn field_support(
        &self,
        field: Id<kinds::Field>,
    ) -> Result<(Id<kinds::Domain>, eqiora_realization::Space), Diagnostic> {
        for domain in self.resolved.plan().spatial().domains() {
            if let Some(binding) = domain
                .field_spaces()
                .iter()
                .find(|binding| binding.field() == field)
            {
                return Ok((domain.domain(), binding.space()));
            }
        }
        for binding in self.resolved.plan().time_step().eliminated_states() {
            if binding.pair().state() == field {
                let (domain, _) = self.field_support(binding.pair().rate())?;
                return Ok((domain, binding.state_space()));
            }
        }
        Err(invalid(
            "Field is absent from exact algebraic and eliminated-state inventory",
        ))
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
        let (NativeMeshResources::AdjacentPartitionSimplicial { mesh, .. }
        | NativeMeshResources::GmshSimplicial { mesh, .. }) = self.resources()
        else {
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
    pub fn field_ids(&self) -> &[String] {
        &self.field_ids
    }
    #[must_use]
    pub fn domain_ids(&self) -> &[String] {
        &self.domain_ids
    }
    pub(super) fn scoped_spatial_policies(&self) -> Vec<(Id<kinds::Domain>, CommonSpatialPolicy)> {
        self.resolved
            .plan()
            .spatial()
            .domains()
            .iter()
            .map(|domain| {
                let policy = if domain
                    .field_spaces()
                    .iter()
                    .any(|field| field.space() == Space::simplex_p1_bubble())
                {
                    CommonSpatialPolicy::MiniP1
                } else {
                    CommonSpatialPolicy::P1
                };
                (domain.domain(), policy)
            })
            .collect()
    }
    #[must_use]
    pub const fn portable_realization(&self) -> &PortableRealizationGraph {
        &self.portable
    }
    pub(crate) fn validate_interface_action(
        &self,
        action: &crate::simplicial_fsi::FixedReferenceFsiInterfaceAction<2>,
    ) -> Result<(), Diagnostic> {
        let endpoints = action.endpoints().map(|(domain, field, _)| (domain, field));
        let matched = self.partition.traces().iter().any(|trace| {
            trace.quotient.connection() == action.connection()
                && trace
                    .quotient
                    .endpoints()
                    .map(|endpoint| (endpoint.domain(), endpoint.field()))
                    == endpoints
                && action.slot() == 0
                && action.entity().dimension() == 0
                && trace.facets.iter().any(|facet| {
                    self.mesh()
                        .entity_vertices(facet.facet)
                        .is_some_and(|vertices| vertices.contains(&action.entity()))
                })
        });
        if !matched {
            return Err(invalid(
                "interface action differs from exact Connection endpoint and entity support",
            ));
        }
        Ok(())
    }

    /// Exact vertex support of one represented Field.
    pub fn field_vertex_indices(&self, field: Id<kinds::Field>) -> Result<Vec<usize>, Diagnostic> {
        let (domain, _) = self.field_support(field)?;
        Ok(self
            .partition
            .domain_vertices(domain)
            .ok_or_else(|| invalid("Field Domain has no exact support"))?
            .iter()
            .map(|id| id.index())
            .collect())
    }
    /// Exact cells in the support Domain of one represented Field.
    pub fn field_domain_cell_indices(
        &self,
        field: Id<kinds::Field>,
    ) -> Result<Vec<usize>, Diagnostic> {
        let (domain, _) = self.field_support(field)?;
        Ok(self
            .partition
            .domain_cells(domain)
            .ok_or_else(|| invalid("Field Domain has no exact support"))?
            .iter()
            .map(|id| id.index())
            .collect())
    }
    /// Exact oriented trace facet connectivity of one admitted Connection.
    pub fn interface_facet_vertices(
        &self,
        connection: Id<kinds::Connection>,
    ) -> Result<Vec<[usize; 2]>, Diagnostic> {
        let traces = self
            .partition
            .traces()
            .iter()
            .filter(|trace| trace.quotient.connection() == connection)
            .collect::<Vec<_>>();
        if traces.is_empty() {
            return Err(invalid("Connection is absent from exact trace inventory"));
        }
        let facets = traces
            .iter()
            .flat_map(|trace| trace.facets.iter().map(|facet| facet.facet))
            .collect::<BTreeSet<_>>();
        facets
            .into_iter()
            .map(|facet| {
                let vertices = self
                    .mesh()
                    .entity_vertices(MeshEntity::new(1, facet.index()))
                    .ok_or_else(|| invalid("trace facet has no exact connectivity"))?;
                Ok([vertices[0].index(), vertices[1].index()])
            })
            .collect()
    }
}
