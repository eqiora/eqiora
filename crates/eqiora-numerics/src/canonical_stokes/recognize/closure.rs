use super::*;

pub(super) fn require_closed_model(
    program: &KernelProgram,
    model: &SteadyIncompressibleStokesModel2d,
    representation: RawId,
    boundary: &LoweredBoundaryProjection2d,
    coefficient_fields: &BTreeSet<RawId>,
    coefficient_definitions: &BTreeSet<RawId>,
    additional_parameters: &BTreeSet<RawId>,
) -> Result<(), Diagnostic> {
    let mut domains = BTreeSet::from([model.domain()]);
    let mut relations = BTreeSet::from([
        model.force_potential_definition(),
        model.momentum_relation(),
        model.incompressibility_relation(),
    ]);
    domains.extend(
        model
            .boundary_relations()
            .iter()
            .map(|binding| binding.boundary()),
    );
    domains.extend(boundary.connector_domains.iter().copied());
    relations.extend(
        model
            .boundary_relations()
            .iter()
            .map(|binding| binding.relation()),
    );
    relations.extend(coefficient_definitions.iter().copied());
    relations.extend(boundary.prescribed_velocity_definitions.iter().copied());
    debug_assert!(boundary.uninterpreted_live_relations.is_subset(&relations));
    let mut fields = BTreeSet::from([model.velocity(), model.pressure(), model.force_potential()]);
    fields.extend(coefficient_fields.iter().copied());
    fields.extend(boundary.prescribed_velocity_fields.iter().copied());
    crate::canonical_stokes::inventory::FlowClosure {
        domains: &domains,
        fields: &fields,
        relations: &relations,
        representation,
        ports: &boundary.ports,
        connections: &boundary.connections,
        additional_parameters,
    }
    .validate(program, "closed 2D steady Stokes lowering")
}
