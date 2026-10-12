use super::*;
pub(super) use crate::canonical::{
    continuum_representation, has_edge, is_field, lowering_error, model_lowering_error,
    relation_expression, relations_on,
};
use crate::canonical_boundary::BoundaryRelationBinding;

pub(super) fn require_continuous_relation(
    program: &KernelProgram,
    relation: RawId,
) -> Result<(), Diagnostic> {
    crate::canonical::require_continuous_relation(
        program,
        relation,
        "elastic-solid Relations require exactly one continuous Activation",
    )
}

pub(super) fn require_closed_elasticity_models(
    program: &KernelProgram,
    subdomains: &[LoweredIsotropicElasticitySubdomain2d],
) -> Result<(), Diagnostic> {
    let closures = subdomains
        .iter()
        .map(|lowered| ElasticityClosure {
            domain: lowered.model.domain(),
            fields: vec![lowered.model.displacement(), lowered.model.load_potential()],
            volume_relations: vec![
                lowered.model.load_definition_relation(),
                lowered.model.equilibrium_relation(),
            ],
            boundary_relations: lowered.model.boundary_relations(),
            boundary: &lowered.boundary,
        })
        .collect::<Vec<_>>();
    require_closed_elasticity_parts(program, &closures)
}

pub(super) struct ElasticityClosure<'a, const D: usize> {
    pub(super) domain: RawId,
    pub(super) fields: Vec<RawId>,
    pub(super) volume_relations: Vec<RawId>,
    pub(super) boundary_relations: &'a [BoundaryRelationBinding],
    pub(super) boundary: &'a boundary::LoweredElasticityBoundary<D>,
}

pub(super) fn require_closed_elasticity_parts<const D: usize>(
    program: &KernelProgram,
    subdomains: &[ElasticityClosure<'_, D>],
) -> Result<(), Diagnostic> {
    let mut expected_domains = BTreeSet::new();
    let mut expected_fields = BTreeSet::new();
    let mut expected_relations = BTreeSet::new();
    let mut expected_representations = BTreeSet::new();
    let mut expected_ports = BTreeSet::new();
    let mut expected_connections = BTreeSet::new();

    for lowered in subdomains {
        expected_domains.insert(lowered.domain);
        expected_domains.extend(
            lowered
                .boundary_relations
                .iter()
                .map(|binding| binding.boundary()),
        );
        expected_domains.extend(lowered.boundary.connector_domains.iter().copied());
        expected_fields.extend(lowered.fields.iter().copied());
        expected_relations.extend(lowered.volume_relations.iter().copied());
        expected_relations.extend(
            lowered
                .boundary_relations
                .iter()
                .map(|binding| binding.relation()),
        );
        expected_representations.insert(
            continuum_representation(program, lowered.fields[0])
                .expect("field validation establishes one continuum Representation"),
        );
        expected_ports.extend(lowered.boundary.ports.iter().copied());
        expected_connections.extend(lowered.boundary.connections.iter().copied());
    }

    let expected_activations = program
        .edges()
        .iter()
        .filter(|edge| {
            edge.kind() == EdgeKind::Activates && expected_relations.contains(&edge.to())
        })
        .map(|edge| edge.from())
        .collect::<BTreeSet<_>>();
    let mut expected_parameters = expected_relations
        .iter()
        .copied()
        .flat_map(|relation| match program.node(relation) {
            Some(KernelNode::Relation(definition)) => definition.expression().nodes().iter(),
            _ => unreachable!("admitted Relations were already inspected"),
        })
        .filter_map(|node| match node {
            ExprNode::Symbol(SymbolRef::Parameter(parameter)) => Some(parameter.erase()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();

    // Observables are derived outputs, not extra equations or unknowns. Their
    // typed dependencies remain in this exact Model; execution checks the Result.
    for node in program.nodes() {
        if let KernelNode::Observable(value) = node {
            program.typed_observable(value.id()).map_err(|errors| {
                errors
                    .into_iter()
                    .next()
                    .expect("typing failure has diagnostic")
            })?;
            expected_parameters.extend(value.expression().nodes().iter().filter_map(|node| {
                match node {
                    ExprNode::Symbol(SymbolRef::Parameter(id)) => Some(id.erase()),
                    _ => None,
                }
            }));
        }
    }

    for node in program.nodes() {
        let admitted = match node {
            KernelNode::Domain(value) => expected_domains.contains(&value.id().erase()),
            KernelNode::Representation(value) => {
                expected_representations.contains(&value.id().erase())
            }
            KernelNode::Field(value) => expected_fields.contains(&value.id().erase()),
            KernelNode::Parameter(value) => expected_parameters.contains(&value.id().erase()),
            KernelNode::Relation(value) => expected_relations.contains(&value.id().erase()),
            KernelNode::Activation(value) => expected_activations.contains(&value.id().erase()),
            KernelNode::Port(value) => expected_ports.contains(&value.id().erase()),
            KernelNode::Connection(value) => expected_connections.contains(&value.id().erase()),
            KernelNode::Observable(_) => true,
            KernelNode::ClockDomain(_) => false,
            _ => false,
        };
        if !admitted {
            return Err(model_lowering_error(
                program,
                format!(
                    "closed 2D elasticity-family lowering would ignore unexpected {:?} node {}",
                    node.kind(),
                    node.id()
                ),
            ));
        }
    }
    Ok(())
}

pub(super) fn unique_root(expression: &ExprDag, owner: RawId) -> Result<ExprId, Diagnostic> {
    crate::canonical::require_unique_root(
        expression,
        owner,
        "elasticity Relation requires exactly one residual root",
    )
}

pub(super) fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}
