//! Mixed Galerkin correspondence for recognized transient flow.

use crate::form_compiler::vocabulary::{MixedGalerkinCorrespondence, MixedGalerkinSource};

use super::navier_stokes::TransientIncompressibleNavierStokesModel;

impl TransientIncompressibleNavierStokesModel<2> {
    pub(crate) fn mixed_galerkin_correspondence(&self) -> MixedGalerkinCorrespondence {
        let boundary_relations = self
            .boundary_relations
            .iter()
            .map(|binding| binding.relation())
            .collect::<Vec<_>>();
        MixedGalerkinCorrespondence::derive(MixedGalerkinSource {
            domain: self.domain,
            velocity: self.velocity,
            pressure: self.pressure,
            source: self.force_source,
            momentum_relation: self.momentum_relation,
            incompressibility_relation: self.incompressibility_relation,
            boundary_relations: &boundary_relations,
        })
    }

    pub(crate) fn replay_mixed_galerkin_correspondence(
        &self,
        correspondence: &MixedGalerkinCorrespondence,
    ) -> Result<(), &'static str> {
        if correspondence != &self.mixed_galerkin_correspondence() {
            return Err("transient mixed Law identity or effective Formulation is stale");
        }
        Ok(())
    }
}
