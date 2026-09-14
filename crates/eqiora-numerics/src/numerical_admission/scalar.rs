use super::*;

mod interval;
mod regions;
mod transient;
pub(crate) use regions::ExecutableScalarEquations;

fn describe_primal(
    kind: FormulationKind,
    boundary_treatment: &'static str,
    rule_ids: [&'static str; 4],
    requested: FormulationSelectionMode,
) -> CommonFormulationDescription {
    CommonFormulationDescription {
        requested,
        kind,
        boundary_treatment,
        rule_ids: rule_ids.into(),
        selection_reason_codes: Box::new([match requested {
            FormulationSelectionMode::Automatic => {
                "eqiora.formulation.auto.primal-galerkin-for-q1/v1"
            }
            FormulationSelectionMode::Exact => {
                "eqiora.formulation.exact.primal-galerkin-admitted/v1"
            }
            FormulationSelectionMode::Authored => {
                "eqiora.formulation.authored.primal-galerkin-admitted/v1"
            }
        }]),
        requested_source_identity: None,
    }
}

pub(super) fn resolve_common_scalar_portable(
    admission: &NativeNumericalAdmission,
    lowered: &ExecutableScalarEquations,
    mesh: &CartesianMeshEnvelopeV1,
    cells: &[usize],
) -> Result<PortableRealizationGraph, Diagnostic> {
    let artifact = mesh.artifact_reference()?;
    let nonzero = |count| NonZeroUsize::new(count).expect("validated Cartesian cells are non-zero");
    let mesh = match cells {
        [x] => MeshPolicy::SuppliedCartesian1d {
            artifact,
            cells: [nonzero(*x)],
        },
        [x, y] => MeshPolicy::SuppliedCartesian {
            artifact,
            cells: [nonzero(*x), nonzero(*y)],
        },
        [x, y, z] => MeshPolicy::SuppliedCartesian3d {
            artifact,
            cells: [nonzero(*x), nonzero(*y), nonzero(*z)],
        },
        _ => {
            return Err(invalid(
                "common scalar Plan requires one to three Cartesian axes",
            ));
        }
    };
    let (method, space, quadrature) = match admission.spatial {
        NativeSpatialPolicy::ScalarQ1 => (
            DiscretizationMethod::ContinuousGalerkin,
            Space::continuous_lagrange(std::num::NonZeroU16::MIN),
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(2).expect("two is non-zero"),
            },
        ),
        NativeSpatialPolicy::ScalarTpfa => (
            DiscretizationMethod::CellCenteredFiniteVolume,
            Space::cell_constant(),
            QuadraturePolicy::CellCentroid,
        ),
        NativeSpatialPolicy::ElasticityQ1
        | NativeSpatialPolicy::StokesMiniP1(_)
        | NativeSpatialPolicy::TransientMiniP1(_)
        | NativeSpatialPolicy::TransientCellCentered(_) => {
            return Err(invalid(
                "common scalar portable graph received a non-scalar spatial policy",
            ));
        }
    };
    let solver = admission.linear.solver;
    admission.linear.capabilities.require_problem(
        solver,
        ScalarType::F64,
        scalar_operator_properties(admission.spatial),
    )?;
    PortableRealizationGraph::linear_regions(
        RealizationLineage::explicit(
            admission.program().model(),
            SemanticRevision::new(admission.program().revision().0),
            RealizationRevision::new(COMMON_SCALAR_REALIZATION_REVISION),
        ),
        lowered.discretizations(space)?,
        lowered.quotients()?,
        Discretization::new(method, mesh, quadrature),
        scalar_operator_properties(admission.spatial),
        ScalarType::F64,
        VectorLayoutKind::Replicated,
        solver,
        Target::HostCpu {
            threads: admission.linear.workers,
        },
        ExecutionSchedule::Offline,
    )
}

type ObservableSupport = (Vec<[f64; 2]>, Option<(usize, BoundarySide)>);

impl CommonScalarPlan {
    /// Exact linear solver provider selected by this Plan.
    #[must_use]
    pub const fn solver_provider(&self) -> eqiora_solver::SolverProvider {
        self.admission.linear.provider
    }

    fn reauthenticate_portable_realization(&self) -> Result<(), Diagnostic> {
        let NativeMeshResources::Cartesian { mesh, .. } = self.admission.resources() else {
            return Err(invalid(
                "common scalar Plan lost its exact Cartesian Mesh materialization",
            ));
        };
        let RecognizedNativeModel::Scalar(lowered) = self.admission.recognized_model() else {
            return Err(invalid(
                "common scalar Plan lost its recognized mathematical materialization",
            ));
        };
        if self.admission.spatial == NativeSpatialPolicy::ScalarTpfa {
            let requested = self
                .formulation
                .as_ref()
                .map_or(FormulationSelectionMode::Automatic, |form| form.requested());
            let description = interval::admit(
                &self.admission,
                lowered,
                requested,
                self.authored_formulation.as_ref(),
            )?;
            if description != self.formulation {
                return Err(invalid("TPFA interval formulation changed during replay"));
            }
        } else if let Some(authored) = &self.authored_formulation {
            let derived = lowered
                .primal_form(self.admission.program())?
                .ok_or_else(|| {
                    invalid("authored scalar-primal Plan lost its effective derived Formulation")
                })?;
            crate::form_compiler::admit_authored_scalar_primal_form(
                authored,
                self.admission.program(),
                &derived,
            )?;
        }
        require_portable_realization(
            &self.portable,
            resolve_common_scalar_portable(&self.admission, lowered, mesh, &self.cells)?,
        )
    }

    pub(super) fn from_admission(
        model: &ModelEnvelope,
        admission: NativeNumericalAdmission,
        formulation_selection: Option<FormulationSelectionMode>,
        authored_formulation: Option<&AuthoredFormulationProjection>,
    ) -> Result<Self, Diagnostic> {
        let model_reference = model.artifact_reference()?;
        let NativeMeshResources::Cartesian {
            mesh, production, ..
        } = admission.resources()
        else {
            return Err(invalid(
                "scalar Q1/TPFA common Plan requires an authenticated Cartesian Mesh",
            ));
        };
        let cells = production
            .cartesian_cells()
            .ok_or_else(|| invalid("common scalar Plan lost its Cartesian production policy"))?
            .cells()
            .to_vec()
            .into_boxed_slice();
        let RecognizedNativeModel::Scalar(lowered) = admission.recognized_model() else {
            return Err(invalid(
                "common scalar Plan admitted non-scalar mathematics",
            ));
        };
        if admission.temporal.is_some() {
            let region = lowered.single()?;
            if admission.spatial != NativeSpatialPolicy::ScalarQ1
                || region.form.fields().len() != 1
                || !region.form.is_transient()
                || !lowered.interfaces.is_empty()
            {
                return Err(invalid(
                    "scalar storage currently requires one Q1 Field on one complete Region",
                ));
            }
            region.form.initial_values()?;
        }
        let fields = lowered
            .fields()
            .iter()
            .map(|(field, value_type)| {
                (
                    field.downcast().expect("compiled Field identity"),
                    value_type.clone(),
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let is_interval = admission.spatial == NativeSpatialPolicy::ScalarTpfa;
        let derived_form = if formulation_selection.is_some() && !is_interval {
            lowered.primal_form(admission.program())?
        } else {
            None
        };
        let mut accepted_authored_formulation = None;
        let formulation = if is_interval {
            let description = interval::admit(
                &admission,
                lowered,
                formulation_selection.unwrap_or(FormulationSelectionMode::Automatic),
                authored_formulation,
            )?;
            if description.is_some() {
                accepted_authored_formulation = authored_formulation.cloned();
            }
            description
        } else {
            match formulation_selection {
                None => None,
                Some(selection) => match derived_form.as_ref() {
                    Some(derived) => {
                        if let Some(projection) = authored_formulation {
                            crate::form_compiler::admit_authored_scalar_primal_form(
                                projection,
                                admission.program(),
                                derived,
                            )?;
                            accepted_authored_formulation = Some(projection.clone());
                        }
                        let (kind, boundary_treatment, rule_ids) =
                            derived.formulation_description();
                        let mut description = describe_primal(
                            kind,
                            boundary_treatment,
                            rule_ids,
                            if authored_formulation.is_some() {
                                FormulationSelectionMode::Authored
                            } else {
                                selection
                            },
                        );
                        description.requested_source_identity = accepted_authored_formulation
                            .as_ref()
                            .map(|form| form.source_identity().to_owned());
                        Some(description)
                    }
                    None if authored_formulation.is_some() => {
                        return Err(invalid(
                            "authored scalar Q1 primal Formulation requires the admitted complete essential boundary class",
                        ));
                    }
                    None if selection == FormulationSelectionMode::Automatic => None,
                    None => {
                        return Err(invalid(
                            "exact scalar Q1 primal Formulation requires the admitted complete essential boundary class",
                        ));
                    }
                },
            }
        };
        let portable = resolve_common_scalar_portable(&admission, lowered, mesh, &cells)?;
        let realization_digest = hex_bytes(&portable.digest()?);
        let (digests, mut identity_bytes) =
            static_plan_identity_lineage(&admission, &realization_digest)?;
        push_framed(
            &mut identity_bytes,
            formulation
                .as_ref()
                .map(|description| description.requested().identity())
                .unwrap_or(b"no-proof-carrying-formulation"),
        );
        if let Some(description) = &formulation
            && description.kind == FormulationKind::IntegralConservative
        {
            // Bind derived mathematical rule meaning even without authored provenance.
            push_framed(
                &mut identity_bytes,
                description.boundary_treatment.as_bytes(),
            );
            for rule in &description.rule_ids {
                push_framed(&mut identity_bytes, rule.as_bytes());
            }
        }
        if let Some(authored) = &accepted_authored_formulation {
            push_framed(&mut identity_bytes, authored.source_identity().as_bytes());
            push_framed(&mut identity_bytes, authored.canonical_bytes());
        }
        let identity =
            domain_separated_identity(b"eqiora.common-scalar-plan/v2\0", &identity_bytes);
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
            formulation,
            authored_formulation: accepted_authored_formulation,
            lineage,
            fields,
            cells,
        })
    }

    pub(crate) fn run(
        &self,
        backend: &dyn LinearSolverBackend,
    ) -> Result<CommonScalarRunOutput, Diagnostic> {
        self.reauthenticate_portable_realization()?;
        self.admission.execute_scalar(backend)
    }

    /// Effective primal Galerkin Formulation for Q1, when one is admitted.
    #[must_use]
    pub fn formulation(&self) -> Option<CommonFormulationDescription> {
        self.formulation.clone()
    }

    pub(crate) fn authored_formulation_bytes(&self) -> Option<&[u8]> {
        self.authored_formulation
            .as_ref()
            .map(AuthoredFormulationProjection::canonical_bytes)
    }

    /// Execute solely from retained Plan state and publish one complete Result.
    pub fn run_result(
        &self,
        backend: &dyn LinearSolverBackend,
    ) -> Result<crate::CommonResult, Diagnostic> {
        if self.admission.temporal.is_some() {
            return Err(invalid(
                "scalar storage execution requires an exact State and Run schedule",
            ));
        }
        crate::CommonResult::accept_scalar(self.clone(), 0.0, self.run(backend)?)
    }

    /// Accept one selected Parameter point through this Plan's exact supplied Mesh and policies.
    ///
    /// `values=None` selects the Model's canonical values. Otherwise only the ordered selected
    /// Parameter values vary; Model structure and every numerical resource remain Plan-owned.
    pub fn differentiate(
        &self,
        selected: &[eqiora_core::Id<eqiora_core::entity::kinds::Parameter>],
        values: Option<&[f64]>,
    ) -> Result<CommonScalarDifferentiationPoint, Diagnostic> {
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
        let RecognizedNativeModel::Scalar(template) = self.admission.recognized_model() else {
            return Err(invalid(
                "common scalar Plan lost its recognized mathematics",
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
        let dimension = self.cells.len();
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
        let finalized = match self.admission.spatial {
            NativeSpatialPolicy::ScalarQ1 => {
                let quadrature = QuadratureRule::tensor_product_gauss_legendre(dimension, 2)?;
                let form = equations
                    .single()?
                    .form
                    .bind_parameter_point(bound.parameter_fields(), bound.parameter_values())?;
                let assembly =
                    crate::cartesian_elliptic::linear::CartesianLinearAssembly::assemble(
                        &form,
                        mesh,
                        &quadrature,
                        &REFERENCE_ASSEMBLY_BACKEND,
                        &equations.single()?.boundaries,
                    )?;
                FinalizedScalarEllipticCartesianProblem::finite_element_blocks(
                    self.portable.clone(),
                    solver,
                    VectorLayoutKind::Replicated,
                    target,
                    assembly,
                )?
            }
            NativeSpatialPolicy::ScalarTpfa => {
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
                FinalizedScalarEllipticCartesianProblem::finite_volume(
                    self.portable.clone(),
                    solver,
                    VectorLayoutKind::Replicated,
                    target,
                    assembly,
                )?
            }
            NativeSpatialPolicy::ElasticityQ1
            | NativeSpatialPolicy::StokesMiniP1(_)
            | NativeSpatialPolicy::TransientMiniP1(_)
            | NativeSpatialPolicy::TransientCellCentered(_) => {
                return Err(invalid(
                    "common scalar differentiation received a non-scalar spatial policy",
                ));
            }
        };
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
        let produced = REFERENCE_LINEAR_SOLVER.solve(&finalized.linear_problem()?, solver)?;
        let accepted = admitted.accept(produced)?;
        let (solution, receipt) = accepted.into_parts();
        let solution = finalized.finish(solution)?;
        let coordinates = selected
            .iter()
            .copied()
            .map(SpatialDesignCoordinate::ModelParameter)
            .collect::<Vec<_>>();
        let (relation, output) = match &solution {
            ResolvedScalarEllipticCartesianSolution::FiniteElement(solution) => {
                let quadrature = QuadratureRule::tensor_product_gauss_legendre(dimension, 2)?;
                (
                    linearize_scalar_elliptic_cartesian_fem(
                        &bound,
                        mesh,
                        solution,
                        &quadrature,
                        &coordinates,
                    )?,
                    linearize_scalar_elliptic_cartesian_fem_output(
                        &bound,
                        mesh,
                        solution,
                        &coordinates,
                    )?,
                )
            }
            ResolvedScalarEllipticCartesianSolution::FiniteVolume(solution) => {
                let cell = QuadratureRule::tensor_product_gauss_legendre(dimension, 1)?;
                let facet = if dimension == 1 {
                    QuadratureRule::point()
                } else {
                    QuadratureRule::tensor_product_gauss_legendre(dimension - 1, 1)?
                };
                (
                    linearize_scalar_elliptic_cartesian_fvm(
                        &bound,
                        mesh,
                        solution,
                        &cell,
                        &facet,
                        &coordinates,
                    )?,
                    linearize_scalar_elliptic_cartesian_fvm_output(
                        &bound,
                        mesh,
                        solution,
                        &coordinates,
                    )?,
                )
            }
        };
        if relation.state_jacobian().agreement_fingerprint() != receipt.operator() {
            return Err(invalid(
                "common Plan solve receipt differs from its differentiated state system",
            ));
        }
        Ok(CommonScalarDifferentiationPoint {
            relation,
            output,
            receipt,
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

    /// Exact canonical Model artifact selected by this Plan.
    pub fn model_reference(&self) -> Result<eqiora_artifact::ModelArtifactReference, Diagnostic> {
        self.admission.model().artifact_reference()
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

    /// Complete scalar-valued Field inventory in canonical identity order.
    #[must_use]
    pub fn fields(
        &self,
    ) -> impl ExactSizeIterator<
        Item = (
            eqiora_core::Id<eqiora_core::entity::kinds::Field>,
            &eqiora_core::ValueType,
        ),
    > + '_ {
        self.fields
            .iter()
            .map(|(field, value_type)| (*field, value_type))
    }

    #[must_use]
    pub fn cells(&self) -> &[usize] {
        &self.cells
    }

    #[must_use]
    pub fn spatial(&self) -> CommonSpatialPolicy {
        match self.admission.spatial {
            NativeSpatialPolicy::ScalarQ1 => CommonSpatialPolicy::Q1,
            NativeSpatialPolicy::ScalarTpfa => CommonSpatialPolicy::CellCenteredTpfa,
            NativeSpatialPolicy::ElasticityQ1 => {
                unreachable!("common scalar Plan cannot own elasticity policy")
            }
            NativeSpatialPolicy::StokesMiniP1(_) => {
                unreachable!("common scalar Plan cannot own Stokes policy")
            }
            NativeSpatialPolicy::TransientMiniP1(_)
            | NativeSpatialPolicy::TransientCellCentered(_) => {
                unreachable!("common scalar Plan cannot own transient-flow policy")
            }
        }
    }

    #[must_use]
    pub const fn linear(&self) -> SolverPlan {
        self.admission.linear.solver
    }
}

fn scalar_operator_properties(spatial: NativeSpatialPolicy) -> LinearOperatorProperties {
    match spatial {
        NativeSpatialPolicy::ScalarQ1 => LinearOperatorProperties::General,
        NativeSpatialPolicy::ScalarTpfa => LinearOperatorProperties::SymmetricPositiveDefinite,
        _ => unreachable!("scalar Plan owns a scalar discretization"),
    }
}

impl CommonScalarPlan {
    pub(crate) fn observation_program(&self) -> &KernelProgram {
        self.admission.program()
    }

    pub(crate) fn observation_support(
        &self,
        domain: eqiora_core::RawId,
    ) -> Result<ObservableSupport, Diagnostic> {
        let RecognizedNativeModel::Scalar(equations) = self.admission.recognized_model() else {
            return Err(invalid("Observable requires the exact scalar Plan support"));
        };
        let region = equations
            .regions
            .iter()
            .find(|region| {
                region.form.domain() == domain
                    || region
                        .boundaries
                        .values()
                        .any(|boundary| *boundary == domain)
            })
            .ok_or_else(|| invalid("Observable Domain is outside exact Region inventory"))?;
        let boundary = if domain == region.form.domain() {
            None
        } else {
            Some(region.boundaries.iter().find_map(|(side, id)| (*id == domain).then_some(*side)).ok_or_else(|| invalid("Observable Domain is not the exact volume or boundary realized by this Plan"))?)
        };
        Ok((region.bounds.clone(), boundary))
    }
}

impl CommonScalarPlan {
    pub(crate) fn field_support(
        &self,
        field: eqiora_core::RawId,
    ) -> Result<(Vec<usize>, Vec<usize>), Diagnostic> {
        let RecognizedNativeModel::Scalar(equations) = self.admission.recognized_model() else {
            return Err(invalid("missing scalar inventory"));
        };
        let NativeMeshResources::Cartesian { mesh, .. } = self.admission.resources() else {
            return Err(invalid("missing Cartesian mesh"));
        };
        let mesh = mesh.mesh();
        let region = equations
            .regions
            .iter()
            .find(|region| region.form.fields().iter().any(|(id, _)| *id == field))
            .ok_or_else(|| invalid("Field absent from exact Region inventory"))?;
        let mut shape = Vec::new();
        for (axis, bounds) in region.bounds.iter().enumerate() {
            let coordinates = mesh.axis_coordinates(axis).expect("axis");
            let start = coordinates
                .iter()
                .position(|x| *x == bounds[0])
                .ok_or_else(|| invalid("Field support lower bound absent"))?;
            let end = coordinates
                .iter()
                .position(|x| *x == bounds[1])
                .ok_or_else(|| invalid("Field support upper bound absent"))?;
            shape.push(end - start + usize::from(self.spatial() == CommonSpatialPolicy::Q1));
        }
        let domains = equations.cell_domains(mesh)?;
        let mut vertices = BTreeSet::new();
        for (index, domain) in domains.iter().enumerate() {
            if *domain == region.form.domain() {
                vertices.extend(
                    mesh.incidence(
                        eqiora_meshing::MeshEntity::new(mesh.topological_dimension(), index),
                        0,
                    )
                    .expect("cell closure")
                    .iter()
                    .map(|vertex| vertex.entity.index()),
                );
            }
        }
        Ok((shape, vertices.into_iter().collect()))
    }
}
