//! Restrict an accepted overlap without repeating geometric intersections.

use super::*;

impl SimplicialRevisionOverlap2d {
    /// Select complete source and target cell regions from this overlap.
    ///
    /// Reuses admitted fragments and their exact revision coordinates, then
    /// proves complete coverage of both selected regions. This supports
    /// parent-relative interfaces within a whole-mesh refinement. No new
    /// geometric intersections are constructed.
    ///
    /// The returned overlap has no retained facets: selection changes which
    /// facets are frontiers. Attach the desired semantic boundaries with
    /// [`Self::with_retained_facets`] against the same Mesh revisions.
    ///
    /// # Errors
    /// Rejects empty, duplicate or unadmitted cells and any selection whose
    /// positive-area fragments do not cover both regions completely.
    pub fn select_cells(
        &self,
        source_cells: &[CellId],
        target_cells: &[CellId],
    ) -> Result<Self, Diagnostic> {
        let (source_cells, source_triangles) = select(
            source_cells,
            &self.source_cells,
            &self.source_triangles,
            "source",
        )?;
        let (target_cells, target_triangles) = select(
            target_cells,
            &self.target_cells,
            &self.target_triangles,
            "target",
        )?;
        let cell_fragments = self
            .cell_fragments
            .iter()
            .filter(|fragment| {
                source_cells.binary_search(&fragment.source_cell).is_ok()
                    && target_cells.binary_search(&fragment.target_cell).is_ok()
            })
            .cloned()
            .collect::<Vec<_>>();
        validate_cell_coverage(
            &source_cells,
            &source_triangles,
            &target_cells,
            &target_triangles,
            &cell_fragments,
        )?;
        Ok(Self {
            chart: self.chart,
            source_cells,
            target_cells,
            source_triangles,
            target_triangles,
            cell_fragments,
            source_retained_facets: Vec::new(),
            target_retained_facets: Vec::new(),
            retained_facet_fragments: Vec::new(),
        })
    }
}

fn select(
    cells: &[CellId],
    admitted: &[CellId],
    triangles: &[Triangle2d],
    label: &str,
) -> Result<(Vec<CellId>, Vec<Triangle2d>), Diagnostic> {
    let mut cells = cells.to_vec();
    cells.sort_unstable();
    if cells.is_empty() || cells.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid_overlap(format!(
            "{label} overlap selection requires nonempty, unique cells",
        )));
    }
    let triangles = cells
        .iter()
        .map(|cell| {
            admitted
                .binary_search(cell)
                .map(|index| triangles[index])
                .map_err(|_| invalid_overlap(format!("{label} cell is outside admitted overlap")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((cells, triangles))
}
