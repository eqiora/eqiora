//! Nodal interpolation through authenticated refinement ancestry.
use super::*;
use crate::field_sampling::{ContinuousP1Sampler2d, coordinates_2d};
use eqiora_meshing::{CellId, OverlapCoordinateChart2d, SimplicialMesh};

impl CommonSpatialPlan {
    /// Interpolate a stationary Result into this Plan's real continuous-P1 State.
    ///
    /// The target must retain the source's affine-triangle production history and
    /// exact Model/Geometry. Interior values are evaluations of the source finite
    /// element function; target essential traces are then lifted exactly. This is
    /// nodal interpolation, not an L2 projection, conservation correction or solve.
    ///
    /// # Errors
    /// Rejects unrelated revisions, changed Models, non-P1 Fields, moving or
    /// transient problems, and unsupported providers. A new Plan is required for
    /// a changed Mesh; neither the source Result nor its preparation is mutated.
    pub fn interpolate_state(
        &self,
        result: &crate::CommonResult,
    ) -> Result<CommonState, Diagnostic> {
        self.reauthenticate_portable_realization()?;
        self.admission.revalidate()?;
        let source = result
            .plan()
            .as_spatial()
            .ok_or_else(|| invalid("nodal interpolation requires a spatial Result"))?;
        if self.model_digest() != source.model_digest()
            || self.admission.temporal.is_some()
            || source.admission.temporal.is_some()
            || self.fields != source.fields
        {
            return Err(invalid(
                "nodal interpolation requires the same stationary Model and Fields",
            ));
        }
        self.require_fixed_observation_geometry()?;
        source.require_fixed_observation_geometry()?;
        let (
            RecognizedNativeModel::Spatial(source_equations),
            RecognizedNativeModel::Spatial(target_equations),
        ) = (
            source.admission.recognized_model(),
            self.admission.recognized_model(),
        )
        else {
            return Err(invalid(
                "nodal interpolation requires real spatial equations",
            ));
        };
        let (
            NativeMeshResources::AffineTriangleSimplicial {
                geometry: source_geometry,
                mesh: source_mesh,
                production: source_production,
                ..
            },
            NativeMeshResources::AffineTriangleSimplicial {
                geometry: target_geometry,
                mesh: target_mesh,
                production: target_production,
                ..
            },
        ) = (source.admission.resources(), self.admission.resources())
        else {
            return Err(invalid(
                "nodal interpolation requires source-owned affine-triangle Meshes",
            ));
        };
        if source_geometry.canonical_bytes() != target_geometry.canonical_bytes() {
            return Err(invalid(
                "nodal interpolation requires the exact same Geometry",
            ));
        }
        let source_policy = source_production
            .affine_triangle_policy()
            .ok_or_else(|| invalid("source Mesh omitted its refinement history"))?;
        let target_policy = target_production
            .affine_triangle_policy()
            .ok_or_else(|| invalid("target Mesh omitted its refinement history"))?;
        if source_policy.cells() != target_policy.cells()
            || !target_policy
                .refinements()
                .starts_with(source_policy.refinements())
        {
            return Err(invalid(
                "target Mesh does not retain the source refinement history",
            ));
        }
        let parents = refinement_parents(
            source_mesh.mesh(),
            target_mesh.mesh(),
            &target_policy.refinements()[source_policy.refinements().len()..],
        )?;
        let mut transferred = BTreeMap::new();
        for (index, (field, value_type)) in source.fields().enumerate() {
            let p1 = Space::continuous_lagrange(std::num::NonZeroU16::MIN);
            if source.field_space(field)? != p1 || self.field_space(field)? != p1 {
                return Err(invalid("nodal interpolation requires continuous-P1 Fields"));
            }
            let (owned, cells) = support::simplicial_topology(
                source_equations,
                source_mesh,
                field.erase(),
                &source.admission.discretizations,
            )?;
            let owned = owned.iter().map(|v| v.index()).collect::<Vec<_>>();
            let cells = cells.iter().map(|v| v.index()).collect::<Vec<_>>();
            let (_, coefficients, _) = result
                .field_block(index, 0)
                .ok_or_else(|| invalid("source Result omits its nodal Field"))?;
            let components = value_type
                .shape()
                .component_count()
                .ok_or_else(|| invalid("Field component count overflows"))?;
            let sampler = ContinuousP1Sampler2d::new(
                source_mesh.mesh(),
                &cells,
                &owned,
                coefficients,
                components,
            )?;
            let (target_vertices, target_cells) = support::simplicial_topology(
                target_equations,
                target_mesh,
                field.erase(),
                &self.admission.discretizations,
            )?;
            let mut candidates: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
            for cell in target_cells {
                let parent = parents[cell.index()];
                if cells.binary_search(&parent).is_err() {
                    return Err(invalid(
                        "refined Field cell has no parent in its exact source support",
                    ));
                }
                for &vertex in &target_mesh.mesh().cells()[cell.index()] {
                    candidates.entry(vertex).or_default().insert(parent);
                }
            }
            let mut values = Vec::new();
            for vertex in target_vertices {
                if let Ok(index) = owned.binary_search(&vertex.index()) {
                    // Bisection preserves source vertex indices and coordinates.
                    values.extend_from_slice(
                        &coefficients[index * components..(index + 1) * components],
                    );
                } else {
                    let candidates = candidates
                        .get(&vertex.index())
                        .ok_or_else(|| invalid("refined vertex has no exact source cell"))?
                        .iter()
                        .copied()
                        .collect::<Vec<_>>();
                    values.extend(sampler.at_in_cells(
                        coordinates_2d(&target_mesh.mesh().vertices()[vertex.index()])?,
                        &candidates,
                    )?);
                }
            }
            transferred.insert(field.erase(), values);
        }
        // Share the physical/free-coordinate owner with stepping: interpolate
        // interior values, then apply the target's exact prescribed traces.
        let (mapping, _) = self.state_assembly(None)?;
        let fields = mapping.physical_fields_from_coefficients(&transferred)?;
        let free = mapping.physical_predictor(&fields)?;
        let ids = transferred.keys().copied().collect::<Vec<_>>();
        let recovered = mapping.recover(&free, &ids)?;
        self.scalar_state(
            None,
            recovered
                .into_values()
                .flat_map(|field| field.coefficients.into_values())
                .collect(),
        )
    }
}

/// Compose existing overlap witnesses; no second geometric containment algorithm.
fn refinement_parents(
    source: &SimplicialMesh,
    target: &SimplicialMesh,
    history: &[(Vec<usize>, usize)],
) -> Result<Vec<usize>, Diagnostic> {
    let mut mesh = source.clone();
    let mut parents = (0..source.cells().len()).collect::<Vec<_>>();
    for (marked, maximum) in history {
        let marked = marked.iter().copied().map(CellId::new).collect::<Vec<_>>();
        let (next, overlap) =
            mesh.bisect_marked(OverlapCoordinateChart2d::Material, &marked, *maximum)?;
        let mut next_parents = vec![None; next.cells().len()];
        for fragment in overlap.cell_fragments() {
            let parent = parents[fragment.source_cell().index()];
            let entry = &mut next_parents[fragment.target_cell().index()];
            if entry.is_some_and(|previous| previous != parent) {
                return Err(invalid("refined cell has conflicting source ancestry"));
            }
            *entry = Some(parent);
        }
        parents = next_parents
            .into_iter()
            .map(|parent| parent.ok_or_else(|| invalid("refinement omitted a target cell")))
            .collect::<Result<_, _>>()?;
        mesh = next;
    }
    if &mesh != target {
        return Err(invalid(
            "target Mesh differs from its exact refinement history",
        ));
    }
    Ok(parents)
}
