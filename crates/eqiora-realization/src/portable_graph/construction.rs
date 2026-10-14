//! Shared construction of a single-system portable solve graph.

use super::*;

impl PortableRealizationGraph {
    /// Connect the supplied semantic arenas to one system and its solve root.
    /// Callers retain arena ordering; all transformations belong to this system.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn single_system(
        lineage: RealizationLineage,
        domains: Vec<DomainDiscretizationNode>,
        fields: Vec<FieldRepresentationNode>,
        transformations: Vec<TransformationNode>,
        blocks: Vec<SystemBlock>,
        scaling: SystemScaling,
        operator_properties: LinearOperatorProperties,
        scalar_type: ScalarType,
        vector_layout: VectorLayoutKind,
        solver: SolverPlan,
        nonlinear: Option<NonlinearSolvePlan>,
        target: Target,
        schedule: ExecutionSchedule,
    ) -> Result<Self, Diagnostic> {
        let transformation_references = (0..transformations.len())
            .map(TransformationId::new)
            .collect();
        let graph = Self {
            lineage,
            domains,
            fields,
            geometry_actions: Vec::new(),
            transformations,
            systems: vec![AlgebraicSystemNode {
                blocks,
                transformations: transformation_references,
                scaling,
                operator_properties,
                scalar_type,
                partition: vector_layout,
            }],
            linear_solves: vec![LinearSolveNode {
                system: AlgebraicSystemId::new(0),
                plan: solver,
                placement: PlacementRequirementId::new(0),
                schedule,
            }],
            nonlinear_solves: nonlinear
                .map(|plan| NonlinearSolveNode {
                    residual_system: AlgebraicSystemId::new(0),
                    linearization: LinearSolveId::new(0),
                    plan,
                })
                .into_iter()
                .collect(),
            placements: vec![portable_placement(target)],
            root: if nonlinear.is_some() {
                SolveRoot::Nonlinear(NonlinearSolveId::new(0))
            } else {
                SolveRoot::Linear(LinearSolveId::new(0))
            },
        };
        graph.validate()?;
        Ok(graph)
    }
}

fn portable_placement(target: Target) -> PlacementRequirementNode {
    match target {
        Target::HostCpu { threads } => PlacementRequirementNode::HostWorkers {
            workers_per_partition: threads,
        },
        Target::CudaGpu { .. } => PlacementRequirementNode::CudaDevices {
            devices_per_partition: NonZeroUsize::MIN,
        },
    }
}
