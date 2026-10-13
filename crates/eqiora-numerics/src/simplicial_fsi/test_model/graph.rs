//! Resolve fixture choices through the ordinary common graph projection.
use super::*;

pub(crate) fn realization_graph(
    plan: &CoupledFieldwiseRealizationPlan,
) -> PortableRealizationGraph {
    let dimension = match plan.spatial().discretization().quadrature() {
        QuadraturePolicy::TriangleDuffyGaussLegendre { .. } => NonZeroUsize::new(2).unwrap(),
        QuadraturePolicy::SimplexDuffyGaussLegendre {
            spatial_dimension, ..
        } => spatial_dimension,
        _ => panic!("fixture requires explicit simplex quadrature"),
    };
    let represented = plan.represented_physical_fields().unwrap();
    let requirements = CoupledFieldwiseRealizationRequirements::new(
        plan.spatial().domains().iter().map(|domain| {
            DomainFieldInventory::new(
                domain.domain(),
                represented
                    .iter()
                    .filter(|field| field.domain() == domain.domain())
                    .map(|field| field.field()),
            )
            .unwrap()
        }),
        plan.spatial().trace_quotients(),
        plan.time_step()
            .eliminated_states()
            .iter()
            .map(|state| state.pair()),
        RealizationRequirements::new(
            dimension,
            eqiora_core::ScalarType::F64,
            VectorLayoutKind::Replicated,
        ),
    )
    .unwrap();
    let Target::HostCpu { threads } = plan.target() else {
        panic!("host fixture")
    };
    let capabilities = RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::ImportedAffineSimplicial,
            SpatialDimensionSupport::exact(dimension),
        )],
        [VectorLayoutKind::Replicated],
        eqiora_solver::SolverCapabilities::exact([eqiora_solver::SolverCapability {
            scalar_domain: eqiora_core::ScalarDomain::Real,
            algorithm: plan.solver().algorithm(),
            operator_properties: plan.operator_properties(),
            preconditioner: plan.solver().preconditioner(),
            reduction: plan.solver().reduction(),
            scalar_type: eqiora_core::ScalarType::F64,
        }])
        .unwrap(),
        TargetCapabilities::none().with_host_cpu(threads),
    )
    .unwrap();
    resolve_coupled_fieldwise(
        &CoupledFieldwiseRealizationRequest::explicit(
            eqiora_core::OntologyId::new(),
            SemanticRevision::new(1),
            RealizationRevision::new(1),
            plan.clone(),
        ),
        requirements,
        &capabilities,
    )
    .unwrap()
    .portable_graph()
    .unwrap()
}
