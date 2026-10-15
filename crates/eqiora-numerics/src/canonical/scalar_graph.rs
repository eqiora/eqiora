//! Scalar graph admission and the sole Cartesian finalization path.

use super::*;

impl ScalarEllipticCartesianModel {
    /// Project an authored scalar selection against this exact lowered Model.
    ///
    /// # Errors
    /// Rejects Model, revision, or spatial-dimension drift before constructing
    /// a graph with this Model's actual Domain, Field, and SPD operator claim.
    pub fn portable_realization(
        &self,
        resolved: &ResolvedRealization,
    ) -> Result<eqiora_realization::PortableRealizationGraph, Diagnostic> {
        if self.semantic_model != resolved.model()
            || self.semantic_revision != resolved.semantic_revision().get()
            || self.dimension() != resolved.requirements().spatial_dimension().get()
        {
            return Err(invalid_realization(
                "resolved realization differs from the lowered Model identity, revision, or spatial dimension",
            ));
        }
        resolved.portable_graph(
            self.domain_id(),
            self.field_id(),
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
    }
}

#[derive(Debug, Clone)]
pub(super) struct SingleFieldGraphSelection {
    pub(super) graph: eqiora_realization::PortableRealizationGraph,
    pub(super) discretization: Discretization,
    pub(super) space: Space,
    pub(super) solver: SolverPlan,
    pub(super) scalar_type: ScalarType,
    pub(super) vector_layout: VectorLayoutKind,
    pub(super) placement: PlacementRequirementNode,
}

pub(super) fn select_scalar_operator(
    graph: eqiora_realization::PortableRealizationGraph,
    domain: Id<kinds::Domain>,
    field: Id<kinds::Field>,
) -> Result<SingleFieldGraphSelection, Diagnostic> {
    let SolveRoot::Linear(root) = graph.root() else {
        return Err(invalid_realization(
            "scalar elliptic execution requires a linear portable Realization root",
        ));
    };
    let linear = graph
        .linear_solve(root)
        .ok_or_else(|| invalid_realization("scalar elliptic portable linear root is absent"))?;
    let system = graph.system(linear.system()).ok_or_else(|| {
        invalid_realization("scalar elliptic portable algebraic system is absent")
    })?;
    let domain_node = graph.domains().first().ok_or_else(|| {
        invalid_realization("scalar elliptic portable Domain selection is absent")
    })?;
    let field_node = graph.fields().first().ok_or_else(|| {
        invalid_realization("scalar elliptic portable Field representation is absent")
    })?;
    if graph.domains().len() != 1
        || graph.fields().len() != 1
        || domain_node.domain() != domain
        || field_node.field() != field
        || system.operator_properties() != LinearOperatorProperties::SymmetricPositiveDefinite
        || domain_node.coordinates() != eqiora_realization::CoordinateTreatment::Physical
        || domain_node.configuration() != eqiora_realization::DomainConfiguration::FixedGeometry
        || !matches!(
            system.scaling(),
            eqiora_realization::SystemScaling::Dimensional
        )
        || !graph.transformations().is_empty()
        || !graph.geometry_actions().is_empty()
        || system.blocks().len() != 1
        || linear.schedule() != eqiora_realization::ExecutionSchedule::Offline
    {
        return Err(invalid_realization(
            "scalar elliptic lowerer identities differ from the portable Realization graph",
        ));
    }
    let placement = graph
        .placement(linear.placement())
        .ok_or_else(|| invalid_realization("scalar elliptic portable placement is absent"))?;
    let discretization = domain_node.discretization();
    let space = field_node.space();
    let solver = linear.plan();
    let scalar_type = system.scalar_type();
    let vector_layout = system.partition();
    Ok(SingleFieldGraphSelection {
        graph,
        discretization,
        space,
        solver,
        scalar_type,
        vector_layout,
        placement,
    })
}

pub(super) fn require_deployment_target(
    placement: PlacementRequirementNode,
    target: Target,
) -> Result<(), Diagnostic> {
    let matches = match (placement, target) {
        (
            PlacementRequirementNode::HostWorkers {
                workers_per_partition,
            },
            Target::HostCpu { threads },
        ) => workers_per_partition == threads,
        (
            PlacementRequirementNode::CudaDevices {
                devices_per_partition,
            },
            Target::CudaGpu { .. },
        ) => devices_per_partition == NonZeroUsize::MIN,
        _ => false,
    };
    if matches {
        Ok(())
    } else {
        Err(invalid_realization(
            "portable placement requirement differs from its selected deployment target",
        ))
    }
}

/// Finalize one portable Cartesian scalar-elliptic realization for an
/// independently selected linear execution adapter.
///
/// The graph owns numerical intent. `target` is the selected runtime binding;
/// its host bound or device kind must match the graph, while a CUDA ordinal
/// remains outside persisted numerical identity.
///
/// This is the default-assembly form of the explicit-assembly finalizer. It
/// performs canonical lowering, plan validation, mesh construction,
/// local-operator evaluation, constraint handling, and deterministic sparse
/// finalization, but deliberately does not select or invoke a solver backend.
///
/// # Errors
/// Preserves exact lowering, Realization, discretization, and assembly
/// diagnostics.
pub fn finalize_scalar_elliptic_cartesian(
    program: &KernelProgram,
    graph: &eqiora_realization::PortableRealizationGraph,
    target: Target,
) -> Result<
    (
        ScalarEllipticCartesianModel,
        FinalizedScalarEllipticCartesianProblem,
    ),
    Diagnostic,
> {
    finalize_scalar_elliptic_cartesian_with_assembly(
        program,
        graph,
        target,
        &REFERENCE_ASSEMBLY_BACKEND,
    )
}

/// Finalize one portable Cartesian scalar-elliptic realization through an
/// explicit assembly adapter without selecting a solver backend.
///
/// The returned opaque problem is the only handoff: it owns the exact CSR and
/// right-hand side, the sole `SolverPlan`, mathematical properties, assembly
/// evidence, and private method-native reconstruction state.
///
/// # Errors
/// Returns `EQ0807` for revision, dimension, boundary, or policy mismatch and
/// preserves numerical diagnostics from assembly and sparse finalization.
pub fn finalize_scalar_elliptic_cartesian_with_assembly(
    program: &KernelProgram,
    graph: &eqiora_realization::PortableRealizationGraph,
    target: Target,
    assembly: &dyn AssemblyBackend<f64>,
) -> Result<
    (
        ScalarEllipticCartesianModel,
        FinalizedScalarEllipticCartesianProblem,
    ),
    Diagnostic,
> {
    if program.model() != graph.lineage().model()
        || program.revision().0 != graph.lineage().semantic_revision().get()
    {
        return Err(invalid_realization(
            "resolved realization does not reference this exact Semantic Model revision",
        ));
    }
    let model = lower_scalar_elliptic_cartesian(program)?;
    let finalized =
        finalize_lowered_scalar_elliptic_cartesian_with_assembly(&model, graph, target, assembly)?;
    Ok((model, finalized))
}

/// Finalize one already lowered Cartesian scalar-elliptic model without
/// returning to its canonical `KernelProgram`.
///
/// This entry point is shared by the canonical revision-local point and
/// immutable selected-Parameter bindings of that same lowered model. Exact
/// source Model and semantic-revision identity are retained privately by the
/// lowered value and revalidated against the Realization.
///
/// # Errors
/// Preserves the default finalizer's exact identity, dimension, boundary,
/// policy, discretization, and assembly diagnostics.
pub fn finalize_lowered_scalar_elliptic_cartesian(
    model: &ScalarEllipticCartesianModel,
    graph: &eqiora_realization::PortableRealizationGraph,
    target: Target,
) -> Result<FinalizedScalarEllipticCartesianProblem, Diagnostic> {
    finalize_lowered_scalar_elliptic_cartesian_with_assembly(
        model,
        graph,
        target,
        &REFERENCE_ASSEMBLY_BACKEND,
    )
}

/// Finalize one owned lowered Parameter point as an inseparable execution handoff.
///
/// This is the preferred application path for repeated evaluation of a static
/// Model/Realization program. The exact bound model survives solver execution
/// and is the only model that can later linearize the accepted solution.
///
/// # Errors
/// Preserves the lowered scalar-elliptic finalizer's diagnostics.
pub fn finalize_scalar_elliptic_parameter_point(
    model: ScalarEllipticCartesianModel,
    graph: &eqiora_realization::PortableRealizationGraph,
    target: Target,
) -> Result<FinalizedScalarEllipticParameterPoint, Diagnostic> {
    let problem = finalize_lowered_scalar_elliptic_cartesian(&model, graph, target)?;
    Ok(FinalizedScalarEllipticParameterPoint {
        model,
        realization: graph.clone(),
        problem,
    })
}

/// Finalize one already lowered Cartesian scalar-elliptic model through an
/// explicit assembly adapter.
///
/// # Errors
/// Preserves the lowered scalar-elliptic finalizer's diagnostics and the
/// selected assembly backend's complete-operation diagnostics.
pub(super) fn finalize_lowered_scalar_elliptic_cartesian_with_assembly(
    model: &ScalarEllipticCartesianModel,
    graph: &eqiora_realization::PortableRealizationGraph,
    target: Target,
    assembly: &dyn AssemblyBackend<f64>,
) -> Result<FinalizedScalarEllipticCartesianProblem, Diagnostic> {
    if model.semantic_model != graph.lineage().model()
        || model.semantic_revision != graph.lineage().semantic_revision().get()
    {
        return Err(invalid_realization(
            "resolved realization does not reference the exact Semantic Model revision retained by the lowered scalar elliptic model",
        ));
    }
    let selection = select_scalar_operator(graph.clone(), model.domain_id(), model.field_id())?;
    require_deployment_target(selection.placement, target)?;
    if selection.scalar_type != ScalarType::F64 {
        return Err(invalid_realization(
            "Cartesian scalar-elliptic finalization admits f64 scalar storage only",
        ));
    }
    validate_cartesian_layout_target(selection.vector_layout, target)?;

    let cells = match selection.discretization.mesh() {
        MeshPolicy::GeneratedUniform { cells_per_axis } => cells_per_axis.get(),
        MeshPolicy::ImportedSimplicial { .. } => {
            return Err(invalid_realization(
                "Cartesian v0 execution requires a generated uniform mesh; use the simplicial realization entry point for an imported mesh",
            ));
        }
        _ => {
            return Err(invalid_realization(
                "legacy Cartesian execution does not admit a supplied mesh",
            ));
        }
    };
    let mesh = CartesianMesh::uniform(model.bounds(), &vec![cells; model.dimension()])?;
    let source = |coordinates: &[f64]| model.source().evaluate(coordinates).unwrap_or(f64::NAN);
    let coefficient = |coordinates: &[f64]| {
        model
            .coefficient_expression()
            .evaluate(coordinates)
            .unwrap_or(f64::NAN)
    };
    let boundary = |axis: usize, side: BoundarySide, coordinates: &[f64]| {
        let condition = model
            .boundary(axis, side)
            .expect("lowered Cartesian model owns every side");
        let value = condition.value().evaluate(coordinates).unwrap_or(f64::NAN);
        match condition {
            ScalarEllipticCartesianBoundary::Essential(_) => {
                CartesianBoundaryValue::Essential(value)
            }
            ScalarEllipticCartesianBoundary::Natural(_) => CartesianBoundaryValue::Natural(value),
        }
    };
    let finalized = match selection.discretization.method() {
        DiscretizationMethod::ContinuousGalerkin => {
            if selection.space.family()
                != (SpaceFamily::ContinuousLagrange {
                    order: 1.try_into().expect("one is non-zero"),
                })
            {
                return Err(invalid_realization(
                    "Cartesian scalar elliptic FEM v0 lowers only continuous Q1 elements",
                ));
            }
            let QuadraturePolicy::GaussLegendre { points_per_axis } =
                selection.discretization.quadrature()
            else {
                return Err(invalid_realization(
                    "Cartesian Q1 FEM requires Gauss-Legendre quadrature",
                ));
            };
            let quadrature = QuadratureRule::tensor_product_gauss_legendre(
                model.dimension(),
                points_per_axis.get(),
            )?;
            FinalizedScalarEllipticCartesianProblem::finite_element(
                selection.graph,
                selection.solver,
                selection.vector_layout,
                target,
                finalize_scalar_elliptic_cartesian_fem(
                    &mesh,
                    &coefficient,
                    &source,
                    &boundary,
                    &quadrature,
                    assembly,
                    model.compiled_form.as_ref(),
                )?,
            )?
        }
        DiscretizationMethod::CellCenteredFiniteVolume => {
            if selection.space.family() != SpaceFamily::CellConstant
                || selection.discretization.quadrature() != QuadraturePolicy::CellCentroid
            {
                return Err(invalid_realization(
                    "Cartesian scalar elliptic FVM v0 requires cell-constant space and centroid quadrature",
                ));
            }
            let cell_quadrature =
                QuadratureRule::tensor_product_gauss_legendre(model.dimension(), 1)?;
            let facet_quadrature = if model.dimension() == 1 {
                QuadratureRule::point()
            } else {
                QuadratureRule::tensor_product_gauss_legendre(model.dimension() - 1, 1)?
            };
            FinalizedScalarEllipticCartesianProblem::finite_volume(
                selection.graph,
                selection.solver,
                selection.vector_layout,
                target,
                finalize_scalar_elliptic_cartesian_fvm(
                    &mesh,
                    &coefficient,
                    &source,
                    &boundary,
                    &cell_quadrature,
                    &facet_quadrature,
                    assembly,
                )?,
            )?
        }
    };
    Ok(finalized)
}

pub(super) fn validate_cartesian_layout_target(
    vector_layout: VectorLayoutKind,
    target: Target,
) -> Result<(), Diagnostic> {
    match (vector_layout, target) {
        (VectorLayoutKind::Replicated, Target::HostCpu { .. } | Target::CudaGpu { .. })
        | (
            VectorLayoutKind::Distributed,
            Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
        ) => Ok(()),
        (VectorLayoutKind::Distributed, Target::HostCpu { .. }) => Err(invalid_realization(
            "distributed Cartesian spatial execution admits exactly one host worker per partition in v0",
        )),
        (VectorLayoutKind::Distributed, Target::CudaGpu { .. }) => Err(invalid_realization(
            "distributed CUDA Cartesian spatial execution is not admitted in v0",
        )),
    }
}
