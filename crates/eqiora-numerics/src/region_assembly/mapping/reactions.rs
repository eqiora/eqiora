//! Explicit weak-dual recovery from exact physical fields and their region forms.
use super::*;
use crate::region_assembly::{InterfaceReactions, RecoveredInterfaceReactions};
use eqiora_meshing::{AffineGeometryMap, MeshGeometry};

impl<S: Coefficient + crate::finalized_spatial::ResidualScalar + Send> RegionDofMap<S> {
    pub(crate) fn recover_reactions<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        input: RegionSolveInput<S>,
        fields: &BTreeMap<RawId, RecoveredRegionField<S>>,
    ) -> Result<RecoveredInterfaceReactions<S>, Diagnostic> {
        self.validate_solve_input(mesh, &input)?;
        self.validate_physical(fields)?;
        let physical_scales = input
            .forms
            .iter()
            .flat_map(|(form, _)| {
                form.fields().iter().map(move |field| {
                    form.physical_dual_scale(field.field)
                        .map(|scale| (field.field, scale))
                })
            })
            .collect::<Result<BTreeMap<_, _>, Diagnostic>>()?;
        let prepared = self.prepare_assembly(
            eqiora_assembly::AssemblyPacketSetIdentityV1::Unbound,
            mesh,
            input.forms,
            input.natural,
            input.previous.as_ref(),
            input.geometry_action.as_deref(),
        )?;
        let reactions = InterfaceReactions::prepare(
            &prepared.work,
            prepared.plan.target_id(1).expect("full target"),
            self,
            &prepared.packet_domains,
        )?;
        let full = self.normalized_coefficients(fields)?;
        let mut recovered = reactions.recover(&full)?;
        recovered.physical_scales = physical_scales;
        Ok(recovered)
    }
}
