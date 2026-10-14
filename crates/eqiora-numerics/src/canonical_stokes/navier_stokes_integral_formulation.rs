//! Integral-conservative Formulation for recognized transient flow Laws.

use crate::form_compiler::vocabulary::{
    IntegralConservativeCorrespondence, IntegralConservativeSource,
};

use super::TransientIncompressibleNavierStokesModel;

pub(crate) fn integral_conservative_correspondence(
    model: &TransientIncompressibleNavierStokesModel<2>,
) -> IntegralConservativeCorrespondence {
    let boundary_relations = model
        .boundary_relations()
        .iter()
        .map(|binding| binding.relation())
        .collect::<Vec<_>>();
    IntegralConservativeCorrespondence::derive(IntegralConservativeSource {
        domain: model.domain(),
        velocity: model.velocity(),
        pressure: model.pressure(),
        source: model.force_potential(),
        source_definition: model.force_potential_definition(),
        momentum_relation: model.momentum_relation(),
        incompressibility_relation: model.incompressibility_relation(),
        boundary_relations: &boundary_relations,
    })
}

pub(crate) fn replay_integral_conservative_correspondence(
    correspondence: &IntegralConservativeCorrespondence,
    model: &TransientIncompressibleNavierStokesModel<2>,
) -> Result<(), &'static str> {
    let boundary_relations = model
        .boundary_relations()
        .iter()
        .map(|binding| binding.relation())
        .collect::<Vec<_>>();
    correspondence.replay(IntegralConservativeSource {
        domain: model.domain(),
        velocity: model.velocity(),
        pressure: model.pressure(),
        source: model.force_potential(),
        source_definition: model.force_potential_definition(),
        momentum_relation: model.momentum_relation(),
        incompressibility_relation: model.incompressibility_relation(),
        boundary_relations: &boundary_relations,
    })
}
