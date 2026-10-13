mod compiled;
pub(crate) use compiled::CompiledProjection;
#[cfg(test)]
pub(crate) use compiled::compilation_count;

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, Scalar};
use num_complex::ComplexFloat;

use crate::{AssemblyMap, DofId, LocalContribution, LocalUnknown};

/// Canonical additive effect of one mapped local contribution on one row.
///
/// Duplicate local rows and columns have already been folded in the original
/// local row-major order. Entries are then exposed in ascending global-column
/// order so execution adapters can route the row without reimplementing
/// constraint elimination or local scatter semantics.
#[derive(Debug, Clone, PartialEq)]
pub struct AssemblyRowDelta<S> {
    row: DofId,
    entries: Vec<(DofId, S)>,
    rhs: S,
}

impl<S: Scalar + ComplexFloat> AssemblyRowDelta<S> {
    /// Global equation receiving this additive row.
    #[must_use]
    pub const fn row(&self) -> DofId {
        self.row
    }

    /// Canonically ordered global-column deltas.
    #[must_use]
    pub fn entries(&self) -> &[(DofId, S)] {
        &self.entries
    }

    /// Additive right-hand-side value for this row.
    #[must_use]
    pub const fn rhs(&self) -> S {
        self.rhs
    }
}

/// Canonical packet-local scatter delta for one square assembly target.
///
/// This is the sole lowering from anonymous local rows and columns into global
/// algebra. Distributed adapters may split its already-mapped rows by owner,
/// but must not repeat the mapping and fixed-column elimination themselves.
#[derive(Debug, Clone, PartialEq)]
pub struct AssemblyDelta<S> {
    target_size: usize,
    rows: Vec<AssemblyRowDelta<S>>,
}

impl<S: Scalar + ComplexFloat> AssemblyDelta<S> {
    pub(crate) fn empty(target_size: usize) -> Self {
        Self {
            target_size,
            rows: Vec::new(),
        }
    }

    /// Project one finite local contribution through its independent map.
    ///
    /// Local duplicates are accumulated in local row-major order before rows
    /// and columns are canonicalized. Exact zeros remain present until final
    /// COO compression so all backends share the reference structural gate.
    ///
    /// # Errors
    /// Returns `EQ0806` for a zero target, shape mismatch, an out-of-range
    /// global degree of freedom, or non-finite projected arithmetic.
    pub fn from_local(
        target_size: usize,
        map: &AssemblyMap<S>,
        local: &LocalContribution<S>,
    ) -> Result<Self, Diagnostic> {
        CompiledProjection::new(target_size, map)?.project(map, local)
    }

    /// Dimension of the square target addressed by these deltas.
    #[must_use]
    pub const fn target_size(&self) -> usize {
        self.target_size
    }

    /// Canonically ascending global rows.
    #[must_use]
    pub fn rows(&self) -> &[AssemblyRowDelta<S>] {
        &self.rows
    }
}

fn check_dof(target_size: usize, dof: DofId) -> Result<(), Diagnostic> {
    if dof.index() >= target_size {
        Err(assembly_failed(format!(
            "global degree of freedom {} is outside system size {target_size}",
            dof.index()
        )))
    } else {
        Ok(())
    }
}

fn assembly_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::ASSEMBLY_FAILED, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_projection_folds_duplicates_and_eliminates_fixed_values_without_conjugation() {
        use num_complex::Complex64 as C;
        let local = LocalContribution::new(
            2,
            3,
            vec![
                C::new(1., 2.),
                C::new(3., -1.),
                C::new(4., 3.),
                C::new(-1., 1.),
                C::new(2., 0.),
                C::new(1., -2.),
            ],
            vec![C::new(11., 13.), C::new(5., -7.)],
        )
        .unwrap();
        let map = AssemblyMap::new(
            vec![Some(DofId::new(0)); 2],
            vec![
                LocalUnknown::Free(DofId::new(0)),
                LocalUnknown::Free(DofId::new(0)),
                LocalUnknown::Fixed(C::new(2., -1.)),
            ],
        )
        .unwrap();
        let delta = AssemblyDelta::from_local(1, &map, &local).unwrap();
        // Free entries sum to 5+2i. Fixed entries sum to 5+i; their
        // ordinary product with 2-i is 11-3i, giving (16+6i)-(11-3i)=5+9i.
        assert_eq!(delta.rows().len(), 1);
        assert_eq!(
            delta.rows()[0].entries(),
            &[(DofId::new(0), C::new(5., 2.))]
        );
        assert_eq!(delta.rows()[0].rhs(), C::new(5., 9.));
        assert!(
            AssemblyMap::new(
                vec![Some(DofId::new(0))],
                vec![LocalUnknown::Fixed(C::new(0., f64::INFINITY))]
            )
            .is_err()
        );
        let huge =
            LocalContribution::new(1, 1, vec![C::new(f64::MAX, 0.)], vec![C::new(0., 0.)]).unwrap();
        let fixed = AssemblyMap::new(
            vec![Some(DofId::new(0))],
            vec![LocalUnknown::Fixed(C::new(2., 0.))],
        )
        .unwrap();
        assert!(AssemblyDelta::from_local(1, &fixed, &huge).is_err());
    }

    #[test]
    fn projection_preserves_local_fold_before_canonical_ordering() {
        let local = LocalContribution::new(
            2,
            4,
            vec![3.0, 1.0e16, 2.0, 3.0, 5.0, -1.0e16, 4.0, 5.0],
            vec![7.0, 11.0],
        )
        .unwrap();
        let map = AssemblyMap::new(
            vec![Some(DofId::new(1)), Some(DofId::new(1))],
            vec![
                LocalUnknown::Free(DofId::new(1)),
                LocalUnknown::Free(DofId::new(0)),
                LocalUnknown::Free(DofId::new(0)),
                LocalUnknown::Fixed(2.0),
            ],
        )
        .unwrap();

        let delta = AssemblyDelta::from_local(2, &map, &local).unwrap();
        assert_eq!(delta.target_size(), 2);
        assert_eq!(delta.rows().len(), 1);
        assert_eq!(delta.rows()[0].row(), DofId::new(1));
        assert_eq!(
            delta.rows()[0].entries(),
            &[(DofId::new(0), 6.0), (DofId::new(1), 8.0)]
        );
        assert_eq!(delta.rows()[0].rhs(), 2.0);
    }

    #[test]
    fn projection_checks_unknowns_even_when_every_equation_is_skipped() {
        let local = LocalContribution::new(1, 1, vec![1.0], vec![0.0]).unwrap();
        let map = AssemblyMap::new(vec![None], vec![LocalUnknown::Free(DofId::new(1))]).unwrap();
        assert_eq!(
            AssemblyDelta::from_local(1, &map, &local)
                .unwrap_err()
                .code(),
            codes::ASSEMBLY_FAILED
        );
    }
}
