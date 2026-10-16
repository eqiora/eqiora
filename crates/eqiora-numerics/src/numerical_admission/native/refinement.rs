//! Immutable common-Mesh refinement through the source-owned production policy.

use super::*;

impl AuthenticatedCommonMesh {
    /// Refine selected source cells, retaining exact Geometry correspondence.
    ///
    /// The source remains immutable. The new occurrence carries the complete
    /// replayable production history and distinct Mesh/correspondence identities;
    /// callers must resolve their discretization and Plan on this new occurrence.
    /// Existing Results and preparation remain bound to their source revision.
    ///
    /// # Errors
    /// Requires the source-owned planar affine-triangle provider. Rejects empty
    /// or invalid marks, insufficient conformity budget, and unrepresentable or
    /// unacceptable target geometry. Does not transfer Fields or choose marks.
    pub fn refine(
        &self,
        marked: &[eqiora_meshing::CellId],
        maximum_cells: usize,
    ) -> Result<Self, Diagnostic> {
        let NativeMeshResources::AffineTriangleSimplicial {
            geometry,
            production,
            ..
        } = &self.resources
        else {
            return Err(invalid(
                "local refinement requires a source-owned affine-triangle Mesh",
            ));
        };
        let policy = production
            .affine_triangle_policy()
            .ok_or_else(|| invalid("affine-triangle Mesh omitted its production policy"))?
            .with_refinement(
                marked.iter().map(|cell| cell.index()).collect(),
                maximum_cells,
            )?;
        let (mesh, correspondence) =
            GeometryMeshCorrespondenceEnvelopeV1::from_affine_triangles(geometry, &policy)?;
        let production = MeshProductionLineageEnvelopeV1::from_affine_triangle_resources(
            &policy,
            geometry,
            &mesh,
            &correspondence,
        )?;
        Self::affine_triangles(geometry.clone(), mesh, correspondence, production)?
            .with_model_geometries(self.model_geometries.clone())
    }
}
