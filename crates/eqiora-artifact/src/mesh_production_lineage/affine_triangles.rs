//! One replayable policy for seed construction and local triangle refinement.

use eqiora_core::Diagnostic;
use serde::{Deserialize, Serialize};

use crate::invalid_artifact;

/// Initial subdivisions and ordered, source-cell-local refinement requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineTriangleMeshPolicy {
    cells: [usize; 2],
    refinements: Vec<(Vec<usize>, usize)>,
}

impl AffineTriangleMeshPolicy {
    /// Construct positive x/y subdivision counts whose analytic sizes fit `usize`.
    pub fn new(cells: [usize; 2]) -> Result<Self, Diagnostic> {
        if cells.contains(&0) {
            return Err(invalid_artifact(
                "affine-triangle cell counts must be positive",
            ));
        }
        let [nx, ny] = cells;
        let vertices = nx
            .checked_add(1)
            .and_then(|x| ny.checked_add(1).and_then(|y| x.checked_mul(y)));
        let triangles = nx.checked_mul(ny).and_then(|cells| cells.checked_mul(2));
        let boundary_facets = nx.checked_add(ny).and_then(|sum| sum.checked_mul(2));
        if vertices.is_none() || triangles.is_none() || boundary_facets.is_none() {
            return Err(invalid_artifact(
                "affine-triangle cell counts overflow analytic mesh counts",
            ));
        }
        Ok(Self {
            cells,
            refinements: Vec::new(),
        })
    }

    /// Append a local bisection request to the exact preceding mesh revision.
    ///
    /// Marks are source-cell indices in the mesh produced by the preceding
    /// policy. Duplicate marks and declaration order are nonsemantic. The
    /// explicit cell budget includes conformity closure. Generation validates
    /// the marks and budget against that exact source mesh.
    pub fn with_refinement(
        mut self,
        mut marked: Vec<usize>,
        maximum_cells: usize,
    ) -> Result<Self, Diagnostic> {
        marked.sort_unstable();
        marked.dedup();
        if marked.is_empty() || maximum_cells == 0 {
            return Err(invalid_artifact(
                "refinement requires marked cells and a positive cell budget",
            ));
        }
        self.refinements.push((marked, maximum_cells));
        Ok(self)
    }

    /// x/y cell counts before fixed-diagonal seed triangulation.
    #[must_use]
    pub const fn cells(&self) -> [usize; 2] {
        self.cells
    }

    /// Diagonal convention of the seed triangulation.
    #[must_use]
    pub const fn diagonal(&self) -> &'static str {
        "lower-left-to-upper-right"
    }

    /// Ordered source-cell marks and total cell budgets for exact replay.
    #[must_use]
    pub fn refinements(&self) -> &[(Vec<usize>, usize)] {
        &self.refinements
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireAffineTrianglePolicy {
    cells: [u64; 2],
    diagonal: WireDiagonal,
    refinements: Vec<WireRefinement>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WireDiagonal {
    LowerLeftToUpperRight,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRefinement {
    marked_cells: Vec<u64>,
    maximum_cells: u64,
}

impl TryFrom<&AffineTriangleMeshPolicy> for WireAffineTrianglePolicy {
    type Error = Diagnostic;

    fn try_from(policy: &AffineTriangleMeshPolicy) -> Result<Self, Self::Error> {
        Ok(Self {
            cells: [portable(policy.cells[0])?, portable(policy.cells[1])?],
            diagonal: WireDiagonal::LowerLeftToUpperRight,
            refinements: policy
                .refinements
                .iter()
                .map(|(marked, maximum)| {
                    Ok(WireRefinement {
                        marked_cells: marked
                            .iter()
                            .copied()
                            .map(portable)
                            .collect::<Result<_, _>>()?,
                        maximum_cells: portable(*maximum)?,
                    })
                })
                .collect::<Result<_, Diagnostic>>()?,
        })
    }
}

impl WireAffineTrianglePolicy {
    pub(super) fn to_policy(&self) -> Result<AffineTriangleMeshPolicy, Diagnostic> {
        let mut policy =
            AffineTriangleMeshPolicy::new([local(self.cells[0])?, local(self.cells[1])?])?;
        for step in &self.refinements {
            if !step.marked_cells.windows(2).all(|pair| pair[0] < pair[1]) {
                return Err(invalid_artifact(
                    "refinement marks must be unique and canonical",
                ));
            }
            policy = policy.with_refinement(
                step.marked_cells
                    .iter()
                    .copied()
                    .map(local)
                    .collect::<Result<_, _>>()?,
                local(step.maximum_cells)?,
            )?;
        }
        Ok(policy)
    }
}

fn portable(value: usize) -> Result<u64, Diagnostic> {
    u64::try_from(value)
        .map_err(|_| invalid_artifact("affine-triangle policy exceeds portable u64"))
}

fn local(value: u64) -> Result<usize, Diagnostic> {
    usize::try_from(value)
        .map_err(|_| invalid_artifact("affine-triangle policy exceeds local usize"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refinement_history_is_canonical_and_has_no_legacy_decoder() {
        let policy = AffineTriangleMeshPolicy::new([2, 3])
            .unwrap()
            .with_refinement(vec![3, 1, 3], 30)
            .unwrap();
        let wire = WireAffineTrianglePolicy::try_from(&policy).unwrap();
        assert_eq!(wire.to_policy().unwrap(), policy);
        assert_eq!(policy.refinements(), &[(vec![1, 3], 30)]);
        let json = serde_json::to_value(wire).unwrap();
        let mut missing_history = json.clone();
        missing_history
            .as_object_mut()
            .unwrap()
            .remove("refinements");
        assert!(serde_json::from_value::<WireAffineTrianglePolicy>(missing_history).is_err());
        for marks in [
            serde_json::json!([]),
            serde_json::json!([3, 1]),
            serde_json::json!([1, 1]),
        ] {
            let mut changed = json.clone();
            changed["refinements"][0]["marked_cells"] = marks;
            let changed: WireAffineTrianglePolicy = serde_json::from_value(changed).unwrap();
            assert!(changed.to_policy().is_err());
        }
    }
}
