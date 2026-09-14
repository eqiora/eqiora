//! Exact plural Region/Field graph construction, independent of physics families.
use super::*;
impl PortableRealizationGraph {
    /// Resolve one graph-native dimensional linear realization for exact Field bindings.
    /// The equation-aware caller supplies the exact Semantic identities and
    /// operator class. This constructor owns graph closure and solver
    /// compatibility; provider capability admission remains a separate step.
    /// # Errors
    /// Returns `EQ0807` when the supplied choices cannot form one connected
    /// portable linear-solve graph.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_regions(
        lineage: RealizationLineage,
        regions: impl IntoIterator<Item = crate::DomainFieldDiscretization>,
        quotients: impl IntoIterator<Item = crate::ConformingTraceQuotient>,
        discretization: Discretization,
        operator_properties: LinearOperatorProperties,
        scalar_type: ScalarType,
        vector_layout: VectorLayoutKind,
        solver: SolverPlan,
        target: Target,
        schedule: ExecutionSchedule,
    ) -> Result<Self, Diagnostic> {
        let mut regions = regions.into_iter().collect::<Vec<_>>();
        regions.sort_by_key(|region| region.domain().erase());
        if regions.is_empty()
            || regions
                .windows(2)
                .any(|pair| pair[0].domain() == pair[1].domain())
        {
            return Err(invalid_realization(
                "linear graph requires nonempty unique Regions",
            ));
        }
        let mut fields = Vec::new();
        for (index, region) in regions.iter().enumerate() {
            if !region.constraints().is_empty() {
                return Err(invalid_realization(
                    "dimensional linear Region graphs require explicitly supported gauges",
                ));
            }
            for binding in region.field_spaces() {
                discretization.validate_space(binding.space())?;
                fields.push(FieldRepresentationNode {
                    domain: DomainDiscretizationId::new(index),
                    field: binding.field(),
                    space: binding.space(),
                });
            }
        }
        fields.sort_by_key(|field| field.field.erase());
        if fields.is_empty() || fields.windows(2).any(|pair| pair[0].field == pair[1].field) {
            return Err(invalid_realization(
                "linear graph requires unique exact Field bindings",
            ));
        }
        let mut quotients = quotients.into_iter().collect::<Vec<_>>();
        quotients.sort_by_key(|quotient| {
            (
                quotient.connection().erase(),
                quotient
                    .endpoints()
                    .map(|endpoint| endpoint.field().erase()),
            )
        });
        let transformations = quotients.iter().map(|quotient| {
            let mut endpoints = Vec::new();
            for endpoint in quotient.endpoints() {
                let index = fields.iter().position(|field| field.field == endpoint.field() && regions[field.domain.index()].domain() == endpoint.domain())
                    .ok_or_else(|| invalid_realization("trace quotient endpoint is absent from exact Field/Region inventory"))?;
                endpoints.push(FieldRepresentationId::new(index));
            }
            Ok(TransformationNode::ConformingTraceQuotient { connection: quotient.connection(), endpoints: [endpoints[0],endpoints[1]] })
        }).collect::<Result<Vec<_>, Diagnostic>>()?;
        let transformation_references = (0..transformations.len())
            .map(TransformationId::new)
            .collect();
        let blocks = (0..fields.len())
            .map(|index| SystemBlock::Field(FieldRepresentationId::new(index)))
            .collect();
        crate::execution::validate_target_schedule(target, schedule)?;
        let graph = Self {
            lineage,
            domains: regions
                .into_iter()
                .map(|region| DomainDiscretizationNode {
                    domain: region.domain(),
                    coordinates: CoordinateTreatment::Physical,
                    configuration: DomainConfiguration::FixedGeometry,
                    discretization,
                })
                .collect(),
            fields,
            geometry_actions: Vec::new(),
            transformations,
            systems: vec![AlgebraicSystemNode {
                blocks,
                transformations: transformation_references,
                scaling: SystemScaling::Dimensional,
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
            nonlinear_solves: Vec::new(),
            placements: vec![portable_placement(target)],
            root: SolveRoot::Linear(LinearSolveId::new(0)),
        };
        graph.validate()?;
        Ok(graph)
    }
}
