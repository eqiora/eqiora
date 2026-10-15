//! Reconstruct weak reactions only when an accepted linear occurrence requests them.
use super::*;
use crate::region_assembly::mapping::RegionSolveInput;

impl CommonSpatialPlan {
    pub(crate) fn linear_reactions(
        &self,
        values: &[f64],
    ) -> Result<crate::region_assembly::RecoveredInterfaceReactions<f64>, Diagnostic> {
        self.admission.revalidate()?;
        let RecognizedNativeModel::Spatial(equations) = self.admission.recognized_model() else {
            return Err(invalid(
                "linear reactions require exact real spatial equations",
            ));
        };
        let selected = &self.admission.discretizations;
        let scaling = &self.admission.system_scaling;
        match self.admission.resources() {
            NativeMeshResources::Cartesian { mesh, .. } => {
                let (mapping, input) =
                    equations.cartesian_assembly(mesh.mesh(), selected, scaling)?;
                let fields = self.physical_fields(&mapping, values)?;
                mapping.recover_reactions(mesh.mesh(), input, &fields)
            }
            NativeMeshResources::GmshSimplicial { mesh, .. }
            | NativeMeshResources::AffineTriangleSimplicial { mesh, .. } => {
                let (mapping, forms, natural) =
                    equations.simplicial_assembly(mesh, selected, scaling)?;
                let fields = self.physical_fields(&mapping, values)?;
                mapping.recover_reactions(
                    mesh.mesh(),
                    RegionSolveInput {
                        operator_properties: self.admission.operator_properties,
                        forms,
                        natural,
                        previous: None,
                        prescribed_states: BTreeMap::new(),
                        geometry_action: None,
                    },
                    &fields,
                )
            }
            _ => Err(invalid("linear reactions require the exact affine mesh")),
        }
    }
}
