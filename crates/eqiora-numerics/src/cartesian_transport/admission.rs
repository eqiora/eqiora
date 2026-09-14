use eqiora_solver::AlgebraicBlock;
use std::num::NonZeroUsize;

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, DimExponents, GraphPath, ScalarType};
use eqiora_meshing::MeshTopology;
use eqiora_realization::{
    CellCenteredConvectionScheme, CoordinateTreatment, DiscretizationMethod, DomainConfiguration,
    ExecutionSchedule, MeshPolicy, PlacementRequirementNode, QuadraturePolicy, ResolutionSource,
    ResolvedTransientCellCenteredTransportRealization, SolveRoot, SpaceFamily, SystemBlock, Target,
    TransformationNode, VectorLayoutKind,
};
use eqiora_schema::kernel::KernelNode;
use eqiora_sem::KernelProgram;
use eqiora_solver::{LinearOperatorProperties, SolverPlan};

use super::api::ScalarTransportCellState2d;
use crate::canonical_transport::ScalarTransportCartesianModel2d;
use eqiora_meshing::CartesianMesh;

const DIMENSION: usize = 2;

#[derive(Debug, Clone, Copy)]
pub(super) struct ExactSelection {
    pub(super) cells: usize,
    pub(super) duration: f64,
    pub(super) solver_plan: SolverPlan,
    pub(super) target: Target,
    pub(super) state_scale: f64,
    pub(super) weak_scale: f64,
    pub(super) field_dimension: DimExponents,
    pub(super) convection_scheme: CellCenteredConvectionScheme,
}

pub(super) fn require_exact_realization(
    program: &KernelProgram,
    model: &ScalarTransportCartesianModel2d,
    resolved: &ResolvedTransientCellCenteredTransportRealization,
) -> Result<ExactSelection, Diagnostic> {
    if program.model() != resolved.model()
        || program.revision().0 != resolved.semantic_revision().get()
    {
        return Err(invalid_realization(
            "resolved transport realization does not reference this exact Semantic Model revision",
        ));
    }
    let requirements = resolved.requirements();
    let model_periodic_connections = model.spatial_periodic_connections().collect::<Vec<_>>();
    if requirements.relation() != model.transport_relation()
        || requirements.state() != model.state()
        || requirements.fieldwise().domain() != model.domain()
        || requirements.fieldwise().unknown_fields() != [model.state()]
        || requirements.spatial_periodic_connections() != model_periodic_connections
        || requirements
            .fieldwise()
            .execution()
            .spatial_dimension()
            .get()
            != DIMENSION
        || requirements.fieldwise().execution().scalar_type() != ScalarType::F64
        || requirements.fieldwise().execution().vector_layout() != VectorLayoutKind::Replicated
    {
        return Err(invalid_realization(
            "resolved transport requirements differ from the exact 2D Domain, Relation, state, spatial-periodic identifications, or replicated-f64 execution",
        ));
    }
    let graph = resolved.portable_graph()?;
    if graph.lineage().model() != resolved.model()
        || graph.lineage().semantic_revision() != resolved.semantic_revision()
        || graph.lineage().source() != ResolutionSource::Explicit(resolved.realization_revision())
        || graph.domains().len() != 1
        || graph.fields().len() != 1
        || graph.systems().len() != 1
        || graph.transformations().len() != 3
    {
        return Err(invalid_realization(
            "transport portable graph lineage or closed node inventory drifted",
        ));
    }
    let domain = &graph.domains()[0];
    let MeshPolicy::GeneratedUniform { cells_per_axis } = domain.discretization().mesh() else {
        return Err(invalid_realization(
            "transport portable graph mesh is not generated uniform Cartesian",
        ));
    };
    let CoordinateTreatment::Scaled(coordinate_scale) = domain.coordinates() else {
        return Err(invalid_realization(
            "transport portable graph requires one explicit coordinate scale",
        ));
    };
    if domain.domain() != model.domain()
        || domain.configuration() != DomainConfiguration::FixedGeometry
        || domain.discretization().method() != DiscretizationMethod::CellCenteredFiniteVolume
        || domain.discretization().quadrature() != QuadraturePolicy::CellCentroid
    {
        return Err(invalid_realization(
            "transport portable Domain node differs from the exact fixed Cartesian cell-centered tuple",
        ));
    }
    let field = graph.fields()[0];
    if field.domain().index() != 0
        || field.field() != model.state()
        || field.space().family() != SpaceFamily::CellConstant
    {
        return Err(invalid_realization(
            "transport portable Field node differs from the exact cell-constant state",
        ));
    }
    let (duration, convection_scheme) = match graph.transformations() {
        [
            TransformationNode::BackwardEulerDerivative {
                relation,
                state,
                duration,
            },
            TransformationNode::CellCenteredConvection {
                relation: upwind_relation,
                state: upwind_state,
                scheme,
            },
            TransformationNode::OrthogonalTwoPointDiffusion {
                relation: diffusion_relation,
                state: diffusion_state,
            },
        ] if *relation == model.transport_relation()
            && *upwind_relation == *relation
            && *diffusion_relation == *relation
            && state.index() == 0
            && upwind_state.index() == 0
            && diffusion_state.index() == 0 =>
        {
            (*duration, *scheme)
        }
        _ => {
            return Err(invalid_realization(
                "transport portable transformations differ from exact backward-difference, convection, and TPFA identities",
            ));
        }
    };
    if duration.dim() != time_dimension()
        || !duration.value().is_finite()
        || duration.value() <= 0.0
    {
        return Err(invalid_realization(
            "transport portable Backward Euler duration must be finite and positive",
        ));
    }
    let duration = duration.value();
    let system = &graph.systems()[0];
    if system.blocks().len() != 1
        || !matches!(system.blocks()[0], SystemBlock::Field(id) if id.index() == 0)
        || system
            .transformations()
            .iter()
            .map(|id| id.index())
            .ne([0, 1, 2])
        || system.operator_properties() != LinearOperatorProperties::General
        || system.scalar_type() != ScalarType::F64
        || system.partition() != VectorLayoutKind::Replicated
    {
        return Err(invalid_realization(
            "transport portable algebraic system differs from the exact one-block general f64 system",
        ));
    }
    let SolveRoot::Linear(root) = graph.root() else {
        return Err(invalid_realization(
            "transport portable graph requires one linear solve root",
        ));
    };
    let linear = graph
        .linear_solve(root)
        .ok_or_else(|| invalid_realization("transport portable graph linear root is absent"))?;
    let target = Target::HostCpu {
        threads: NonZeroUsize::MIN,
    };
    if linear.system().index() != 0
        || linear.schedule() != ExecutionSchedule::Offline
        || graph.placement(linear.placement())
            != Some(PlacementRequirementNode::HostWorkers {
                workers_per_partition: NonZeroUsize::MIN,
            })
    {
        return Err(invalid_realization(
            "transport portable graph topology or serial placement drifted",
        ));
    }
    let scaling = system.congruence_scaling().ok_or_else(|| {
        invalid_realization("transport portable system omitted symmetric congruence scaling")
    })?;
    let scales = scaling.block_scales();
    if scales.len() != 1 || scales[0].block() != AlgebraicBlock::Field(model.state()) {
        return Err(invalid_realization(
            "transport congruence scaling does not contain exactly the transported state block",
        ));
    }
    let state_scale = scales[0].scale().quantity().value();
    let weak_scale = scaling.weak_functional_scale().quantity().value();
    let field_dimension = match program.node(model.state().erase()) {
        Some(KernelNode::Field(field)) => field.dimension(),
        _ => {
            return Err(invalid_realization(
                "transported state Field definition is absent from the exact program",
            ));
        }
    };
    let length_dimension =
        DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
    let weak_dimension = field_dimension
        .mul(length_dimension)
        .and_then(|dimension| dimension.mul(length_dimension))
        .and_then(|dimension| dimension.div(time_dimension()))
        .ok_or_else(|| {
            invalid_realization(
                "transport weak-functional dimension exceeds rational exponent bounds",
            )
        })?;
    if coordinate_scale.quantity().dim() != length_dimension
        || scales[0].scale().quantity().dim() != field_dimension
        || scaling.weak_functional_scale().quantity().dim() != weak_dimension
    {
        return Err(invalid_realization(
            "transport coordinate, state, or weak-functional scale has the wrong physical dimension",
        ));
    }
    if !state_scale.is_finite()
        || state_scale <= 0.0
        || !weak_scale.is_finite()
        || weak_scale <= 0.0
    {
        return Err(invalid_realization(
            "transport physical scales must be finite and positive",
        ));
    }
    Ok(ExactSelection {
        cells: cells_per_axis.get(),
        duration,
        solver_plan: linear.plan(),
        target,
        state_scale,
        weak_scale,
        field_dimension,
        convection_scheme,
    })
}

pub(super) fn validate_previous(
    model: &ScalarTransportCartesianModel2d,
    resolved: &ResolvedTransientCellCenteredTransportRealization,
    mesh: &CartesianMesh,
    field_dimension: DimExponents,
    previous: &ScalarTransportCellState2d,
) -> Result<(), Diagnostic> {
    let expected = mesh
        .entity_count(DIMENSION)
        .expect("2D Cartesian mesh owns top cells");
    if previous.model != resolved.model()
        || previous.semantic_revision != resolved.semantic_revision()
        || previous.realization_revision != resolved.realization_revision()
        || previous.field != model.state()
        || &previous.mesh != mesh
        || previous.value_dimension != field_dimension
        || previous.values.len() != expected
        || previous.time.dim() != time_dimension()
        || !previous.time.value().is_finite()
        || previous.time.value() < 0.0
        || previous.values.iter().any(|value| !value.is_finite())
    {
        return Err(invalid_realization(
            "previous transport state differs from the exact Model, semantic/Realization revision, Field, generated mesh, value dimension, finite time, or cell shape",
        ));
    }
    Ok(())
}

pub(super) const fn time_dimension() -> DimExponents {
    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension")
}

fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message).with_graph_path(GraphPath::new([
        "realization".to_owned(),
        "scalar-transport-fvm-2d".to_owned(),
        "admission".to_owned(),
    ]))
}
