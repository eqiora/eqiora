mod properties;

use eqiora_core::Scalar;
use num_complex::ComplexFloat;

use std::ops::Range;
#[cfg(test)]
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use sha2::{Digest, Sha256};

use crate::{
    DiagonalAvailability, LinearOperator, LinearOperatorProperties, LinearProblem,
    OrientedLinearOperator, RowLinearAction,
};

const AGREEMENT_DOMAIN_V1: &[u8] = b"eqiora.canonical-csr-agreement/v1\0";

/// Storage-only projection of one complete sparse linear system.
///
/// Implementors expose shape and immutable CSR/RHS storage only. In
/// particular, this trait deliberately has no operator action or identity
/// method: Eqiora captures and validates the data before defining either.
pub trait CompleteCsrStorage<S = f64> {
    /// Matrix row count.
    fn rows(&self) -> usize;

    /// Matrix column count.
    fn columns(&self) -> usize;

    /// CSR row offsets.
    fn row_offsets(&self) -> &[usize];

    /// CSR column indices.
    fn column_indices(&self) -> &[usize];

    /// CSR nonzero values.
    fn values(&self) -> &[S];

    /// Complete right-hand side.
    fn right_hand_side(&self) -> &[S];
}

/// Fixed-size L2 identity for exact canonical CSR algebraic agreement.
///
/// This identity is domain-separated from durable artifact digests. It covers
/// exact floating-point bits and an asserted property tag, but intentionally
/// carries no Semantic Model or Realization provenance. The v1 property tags
/// are frozen as `General = 0`, `SymmetricPositiveDefinite = 1`, and
/// `SymmetricIndefinite = 2`; adding the last tag preserved fingerprints
/// produced with either earlier property. Additional tags are symmetric = 3,
/// complex symmetric = 4, Hermitian = 5, and Hermitian positive definite = 6.
/// The scalar tags are binary64 real = 1
/// and binary64 complex = 2; complex entries bind real then imaginary bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CanonicalCsrAgreementFingerprintV1([u8; 32]);

impl CanonicalCsrAgreementFingerprintV1 {
    /// Raw SHA-256 bytes for fixed-size collective comparison.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Captured complete CSR system with one Eqiora-owned mathematical action.
///
/// Construction copies every exposed slice exactly once, validates the copy,
/// and thereafter derives the host action, linear problem, and agreement
/// fingerprint from those owned bytes. A third-party storage implementation
/// cannot substitute an unrelated virtual operator action.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalCsrSystemView<S = f64> {
    rows: usize,
    columns: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<S>,
    right_hand_side: Vec<S>,
    properties: LinearOperatorProperties,
    agreement_fingerprint: CanonicalCsrAgreementFingerprintV1,
    #[cfg(test)]
    operator_call_instrumentation: TestOperatorCallInstrumentation,
}

#[cfg(test)]
#[derive(Debug, Default)]
struct CanonicalCsrOperatorCallCounts {
    apply: AtomicUsize,
    diagonal: AtomicUsize,
}

/// Test-only, isolated call ledger for one owned canonical CSR operator.
///
/// Each ledger owns distinct atomic counters, so parallel tests cannot observe
/// one another. It does not exist in product builds.
#[cfg(test)]
#[derive(Debug, Clone, Default)]
pub(crate) struct CanonicalCsrOperatorCallLedger(Arc<CanonicalCsrOperatorCallCounts>);

#[cfg(test)]
impl CanonicalCsrOperatorCallLedger {
    pub(crate) fn apply_calls(&self) -> usize {
        self.0.apply.load(Ordering::SeqCst)
    }

    pub(crate) fn diagonal_calls(&self) -> usize {
        self.0.diagonal.load(Ordering::SeqCst)
    }

    pub(crate) fn reset(&self) {
        self.0.apply.store(0, Ordering::SeqCst);
        self.0.diagonal.store(0, Ordering::SeqCst);
    }
}

#[cfg(test)]
#[derive(Debug, Default)]
struct TestOperatorCallInstrumentation(Option<CanonicalCsrOperatorCallLedger>);

// A cloned canonical view is a distinct owned operator, so it must not inherit
// an observation ledger attached to the source object's identity.
#[cfg(test)]
impl Clone for TestOperatorCallInstrumentation {
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[cfg(test)]
impl TestOperatorCallInstrumentation {
    fn record_apply(&self) {
        if let Some(ledger) = &self.0 {
            ledger.0.apply.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn record_diagonal(&self) {
        if let Some(ledger) = &self.0 {
            ledger.0.diagonal.fetch_add(1, Ordering::SeqCst);
        }
    }
}

// Instrumentation is observational only: attaching a test ledger must not
// change the captured system's ordinary equality semantics.
#[cfg(test)]
impl PartialEq for TestOperatorCallInstrumentation {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl<S: Scalar + ComplexFloat<Real = f64> + Sync> CanonicalCsrSystemView<S> {
    /// Capture and validate one complete square finite binary64 real or complex CSR system.
    ///
    /// # Errors
    /// Returns `EQ0807` for an empty or non-square system, malformed CSR,
    /// non-finite values, an RHS mismatch, or a count outside portable `u64`.
    pub fn new(
        storage: &dyn CompleteCsrStorage<S>,
        properties: LinearOperatorProperties,
    ) -> Result<Self, Diagnostic> {
        if !properties.supports_domain(S::DOMAIN) {
            return Err(invalid_realization(
                "canonical CSR property is incompatible with its scalar domain",
            ));
        }
        let rows = storage.rows();
        let columns = storage.columns();
        let row_offsets = try_copy_slice(storage.row_offsets(), "CSR row offsets")?;
        let column_indices = try_copy_slice(storage.column_indices(), "CSR column indices")?;
        let values = try_copy_scalar_slice(storage.values(), "CSR values")?;
        let right_hand_side =
            try_copy_scalar_slice(storage.right_hand_side(), "CSR right-hand side")?;

        validate_complete_csr(
            rows,
            columns,
            &row_offsets,
            &column_indices,
            &values,
            &right_hand_side,
        )?;
        properties::validate_declared_symmetry(&row_offsets, &column_indices, &values, properties)?;
        let agreement_fingerprint = agreement_fingerprint(
            rows,
            columns,
            &row_offsets,
            &column_indices,
            &values,
            &right_hand_side,
            properties,
        )?;
        Ok(Self {
            rows,
            columns,
            row_offsets,
            column_indices,
            values,
            right_hand_side,
            properties,
            agreement_fingerprint,
            #[cfg(test)]
            operator_call_instrumentation: TestOperatorCallInstrumentation::default(),
        })
    }

    #[cfg(test)]
    pub(crate) fn attach_test_operator_call_ledger(
        &mut self,
        ledger: &CanonicalCsrOperatorCallLedger,
    ) {
        assert!(
            self.operator_call_instrumentation.0.is_none(),
            "a canonical CSR test call ledger may be attached only once"
        );
        self.operator_call_instrumentation = TestOperatorCallInstrumentation(Some(ledger.clone()));
    }

    /// Captured matrix row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Captured matrix column count.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// CSR row offsets captured at construction.
    #[must_use]
    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    /// CSR column indices captured at construction.
    #[must_use]
    pub fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }

    /// CSR values captured at construction.
    #[must_use]
    pub fn values(&self) -> &[S] {
        &self.values
    }

    /// Complete RHS captured at construction.
    #[must_use]
    pub fn right_hand_side(&self) -> &[S] {
        &self.right_hand_side
    }

    /// Realization-asserted mathematical properties.
    #[must_use]
    pub const fn properties(&self) -> LinearOperatorProperties {
        self.properties
    }

    /// Exact L2 algebraic agreement identity.
    #[must_use]
    pub const fn agreement_fingerprint(&self) -> CanonicalCsrAgreementFingerprintV1 {
        self.agreement_fingerprint
    }

    /// Borrow this same captured system as a host linear problem.
    ///
    /// # Errors
    /// Returns `EQ0802` only if the internal fixed action contradicts the
    /// invariants already checked at construction.
    pub fn linear_problem(&self) -> Result<LinearProblem<'_, S>, Diagnostic> {
        LinearProblem::from_canonical(self)
    }

    /// Borrow this captured coefficient system with a caller-supplied right-hand side.
    ///
    /// This preserves the exact canonical CSR coefficient source for direct
    /// backends while allowing correction, sensitivity, and adjoint solves to
    /// bind a right-hand side distinct from the captured primal one.
    ///
    /// # Errors
    /// Returns `EQ0802` when `right_hand_side` has the wrong length or contains
    /// a non-finite value.
    pub fn linear_problem_with_right_hand_side<'a>(
        &'a self,
        right_hand_side: &'a [S],
    ) -> Result<LinearProblem<'a, S>, Diagnostic> {
        LinearProblem::from_oriented_canonical(self, self, right_hand_side)
    }

    fn apply_range(
        &self,
        rows: Range<usize>,
        input: &[S],
        output: &mut [S],
    ) -> Result<(), Diagnostic> {
        let row_count = rows
            .end
            .checked_sub(rows.start)
            .ok_or_else(|| solve_failed("canonical CSR row range must be nondecreasing"))?;
        if rows.end > self.rows || input.len() != self.columns || output.len() != row_count {
            return Err(solve_failed(format!(
                "canonical CSR row action for {:?} of {}x{} has input/output sizes {}/{}",
                rows,
                self.rows,
                self.columns,
                input.len(),
                output.len()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(solve_failed("canonical CSR input must be finite"));
        }
        for (row, result) in rows.zip(output) {
            let mut sum = S::zero();
            for entry in self.row_offsets[row]..self.row_offsets[row + 1] {
                sum = sum + self.values[entry] * input[self.column_indices[entry]];
            }
            if !sum.is_finite() {
                return Err(solve_failed(format!(
                    "canonical CSR row {row} produced a non-finite value"
                )));
            }
            *result = sum;
        }
        Ok(())
    }
}

fn try_copy_slice<T: Copy>(source: &[T], name: &'static str) -> Result<Vec<T>, Diagnostic> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(source.len())
        .map_err(|_| invalid_realization(format!("could not reserve captured {name}")))?;
    copy.extend_from_slice(source);
    Ok(copy)
}

fn try_copy_scalar_slice<S: Scalar + ComplexFloat<Real = f64>>(
    source: &[S],
    name: &'static str,
) -> Result<Vec<S>, Diagnostic> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(source.len())
        .map_err(|_| invalid_realization(format!("could not reserve captured {name}")))?;
    copy.extend(source.iter().map(|value| *value + S::zero()));
    Ok(copy)
}

impl<S: Scalar + ComplexFloat<Real = f64> + Sync> LinearOperator for CanonicalCsrSystemView<S> {
    type Scalar = S;

    fn rows(&self) -> usize {
        self.rows
    }

    fn columns(&self) -> usize {
        self.columns
    }

    fn apply(&self, input: &[S], output: &mut [S]) -> Result<(), Diagnostic> {
        #[cfg(test)]
        self.operator_call_instrumentation.record_apply();
        self.apply_range(0..self.rows, input, output)
    }

    fn row_action(&self) -> Option<&dyn RowLinearAction<Scalar = S>> {
        Some(self)
    }

    fn diagonal(&self, output: &mut [S]) -> Result<DiagonalAvailability, Diagnostic> {
        #[cfg(test)]
        self.operator_call_instrumentation.record_diagonal();
        if output.len() != self.rows {
            return Err(solve_failed(
                "canonical CSR diagonal output must match its dimension",
            ));
        }
        for row in 0..self.rows {
            let columns = &self.column_indices[self.row_offsets[row]..self.row_offsets[row + 1]];
            if columns.binary_search(&row).is_err() {
                return Ok(DiagonalAvailability::Unavailable);
            }
        }
        for (row, value) in output.iter_mut().enumerate() {
            let columns = &self.column_indices[self.row_offsets[row]..self.row_offsets[row + 1]];
            let offset = columns
                .binary_search(&row)
                .expect("every canonical CSR diagonal position was checked");
            *value = self.values[self.row_offsets[row] + offset];
        }
        Ok(DiagonalAvailability::Available)
    }
}

impl<S: Scalar + ComplexFloat<Real = f64> + Sync> RowLinearAction for CanonicalCsrSystemView<S> {
    type Scalar = S;

    fn apply_rows(
        &self,
        rows: Range<usize>,
        input: &[S],
        output: &mut [S],
    ) -> Result<(), Diagnostic> {
        self.apply_range(rows, input, output)
    }
}

impl<S: Scalar + ComplexFloat<Real = f64> + Sync> OrientedLinearOperator
    for CanonicalCsrSystemView<S>
{
    fn supports_orientation(&self, _orientation: crate::LinearOperatorOrientation) -> bool {
        true
    }

    fn apply_oriented(
        &self,
        orientation: crate::LinearOperatorOrientation,
        input: &[S],
        output: &mut [S],
    ) -> Result<(), Diagnostic> {
        if orientation == crate::LinearOperatorOrientation::Normal {
            return self.apply(input, output);
        }
        if input.len() != self.rows || output.len() != self.columns {
            return Err(solve_failed(format!(
                "transposed canonical CSR is {}x{} but input/output have {}/{} values",
                self.columns,
                self.rows,
                input.len(),
                output.len()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(solve_failed(
                "transposed canonical CSR input must be finite",
            ));
        }
        output.fill(S::zero());
        for (row, input_value) in input.iter().enumerate() {
            for entry in self.row_offsets[row]..self.row_offsets[row + 1] {
                let value = if orientation == crate::LinearOperatorOrientation::ConjugateTransposed
                {
                    self.values[entry].conj()
                } else {
                    self.values[entry]
                };
                let column = self.column_indices[entry];
                output[column] = output[column] + value * *input_value;
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(solve_failed(
                "transposed canonical CSR action produced a non-finite value",
            ));
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_complete_csr<S: ComplexFloat<Real = f64>>(
    rows: usize,
    columns: usize,
    row_offsets: &[usize],
    column_indices: &[usize],
    values: &[S],
    right_hand_side: &[S],
) -> Result<(), Diagnostic> {
    if rows == 0 || rows != columns {
        return Err(invalid_realization(
            "canonical CSR requires a nonempty square matrix",
        ));
    }
    portable_count(rows, "row count")?;
    portable_count(columns, "column count")?;
    portable_count(row_offsets.len(), "row-offset count")?;
    portable_count(column_indices.len(), "column-index count")?;
    portable_count(values.len(), "value count")?;
    portable_count(right_hand_side.len(), "right-hand-side count")?;

    let expected_offsets = rows
        .checked_add(1)
        .ok_or_else(|| invalid_realization("canonical CSR row-offset extent overflowed"))?;
    if row_offsets.len() != expected_offsets
        || row_offsets.first() != Some(&0)
        || row_offsets.last() != Some(&column_indices.len())
        || column_indices.len() != values.len()
        || right_hand_side.len() != rows
        || row_offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(invalid_realization(
            "canonical CSR offsets, columns, values, and RHS have inconsistent shape",
        ));
    }
    if values
        .iter()
        .chain(right_hand_side)
        .any(|value| !value.is_finite())
    {
        return Err(invalid_realization(
            "canonical CSR values and RHS must be finite",
        ));
    }
    for row in 0..rows {
        let start = row_offsets[row];
        let end = row_offsets[row + 1];
        if end > column_indices.len() {
            return Err(invalid_realization(format!(
                "canonical CSR row {row} ends outside its column storage"
            )));
        }
        let columns_in_row = &column_indices[start..end];
        if columns_in_row.iter().any(|column| *column >= columns)
            || columns_in_row.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid_realization(format!(
                "canonical CSR row {row} columns must be in range and strictly ordered"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn agreement_fingerprint<S: Scalar + ComplexFloat<Real = f64>>(
    rows: usize,
    columns: usize,
    row_offsets: &[usize],
    column_indices: &[usize],
    values: &[S],
    right_hand_side: &[S],
    properties: LinearOperatorProperties,
) -> Result<CanonicalCsrAgreementFingerprintV1, Diagnostic> {
    let mut hash = Sha256::new();
    hash.update(AGREEMENT_DOMAIN_V1);
    hash.update([if S::DOMAIN == eqiora_core::ScalarDomain::Complex {
        2
    } else {
        1
    }]); // binary64 domain tag
    update_count(&mut hash, rows, "row count")?;
    update_count(&mut hash, columns, "column count")?;
    update_count(&mut hash, row_offsets.len(), "row-offset count")?;
    for &offset in row_offsets {
        update_count(&mut hash, offset, "row offset")?;
    }
    update_count(&mut hash, column_indices.len(), "column-index count")?;
    for &column in column_indices {
        update_count(&mut hash, column, "column index")?;
    }
    update_count(&mut hash, values.len(), "value count")?;
    for value in values {
        hash.update(value.re().to_bits().to_be_bytes());
        if S::DOMAIN == eqiora_core::ScalarDomain::Complex {
            hash.update(value.im().to_bits().to_be_bytes());
        }
    }
    update_count(&mut hash, right_hand_side.len(), "right-hand-side count")?;
    for value in right_hand_side {
        hash.update(value.re().to_bits().to_be_bytes());
        if S::DOMAIN == eqiora_core::ScalarDomain::Complex {
            hash.update(value.im().to_bits().to_be_bytes());
        }
    }
    hash.update([agreement_property_tag_v1(properties)]);
    Ok(CanonicalCsrAgreementFingerprintV1(hash.finalize().into()))
}

/// Frozen tags in the `eqiora.canonical-csr-agreement/v1` binary domain.
///
/// `SymmetricIndefinite` is an additive L2 agreement tag. This does not widen
/// any durable artifact-v1 schema, whose encoder rejects that property until
/// a separately versioned wire extension exists.
const fn agreement_property_tag_v1(properties: LinearOperatorProperties) -> u8 {
    match properties {
        LinearOperatorProperties::General => 0,
        LinearOperatorProperties::SymmetricPositiveDefinite => 1,
        LinearOperatorProperties::SymmetricIndefinite => 2,
        LinearOperatorProperties::Symmetric => 3,
        LinearOperatorProperties::ComplexSymmetric => 4,
        LinearOperatorProperties::Hermitian => 5,
        LinearOperatorProperties::HermitianPositiveDefinite => 6,
    }
}

fn update_count(hash: &mut Sha256, value: usize, name: &'static str) -> Result<(), Diagnostic> {
    hash.update(portable_count(value, name)?.to_be_bytes());
    Ok(())
}

fn portable_count(value: usize, name: &'static str) -> Result<u64, Diagnostic> {
    u64::try_from(value)
        .map_err(|_| invalid_realization(format!("canonical CSR {name} exceeds portable u64")))
}

fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct AdversarialStorage {
        values: Vec<f64>,
        hidden_action_scale: f64,
    }

    impl AdversarialStorage {
        fn unrelated_action(&self, input: &[f64]) -> Vec<f64> {
            input
                .iter()
                .map(|value| self.hidden_action_scale * value)
                .collect()
        }
    }

    impl CompleteCsrStorage for AdversarialStorage {
        fn rows(&self) -> usize {
            2
        }

        fn columns(&self) -> usize {
            2
        }

        fn row_offsets(&self) -> &[usize] {
            &[0, 1, 2]
        }

        fn column_indices(&self) -> &[usize] {
            &[0, 1]
        }

        fn values(&self) -> &[f64] {
            &self.values
        }

        fn right_hand_side(&self) -> &[f64] {
            &[2.0, 8.0]
        }
    }

    struct BitPatternStorage {
        values: [f64; 2],
        right_hand_side: [f64; 2],
    }

    impl CompleteCsrStorage for BitPatternStorage {
        fn rows(&self) -> usize {
            2
        }

        fn columns(&self) -> usize {
            2
        }

        fn row_offsets(&self) -> &[usize] {
            &[0, 1, 2]
        }

        fn column_indices(&self) -> &[usize] {
            &[0, 1]
        }

        fn values(&self) -> &[f64] {
            &self.values
        }

        fn right_hand_side(&self) -> &[f64] {
            &self.right_hand_side
        }
    }

    #[test]
    fn captured_view_owns_the_only_action_and_identity() {
        let mut storage = AdversarialStorage {
            values: vec![2.0, 4.0],
            hidden_action_scale: 99.0,
        };
        let view = CanonicalCsrSystemView::new(
            &storage,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
        let original = view.agreement_fingerprint();

        storage.values.fill(7.0);
        let mut actual = [0.0; 2];
        view.apply(&[1.0, 2.0], &mut actual).unwrap();

        assert_eq!(actual, [2.0, 8.0]);
        assert_ne!(&actual, storage.unrelated_action(&[1.0, 2.0]).as_slice());
        assert_eq!(view.agreement_fingerprint(), original);
        assert_eq!(view.linear_problem().unwrap().right_hand_side(), [2.0, 8.0]);
    }

    #[test]
    fn alternate_right_hand_side_preserves_the_exact_canonical_source() {
        let storage = AdversarialStorage {
            values: vec![2.0, 4.0],
            hidden_action_scale: 99.0,
        };
        let view = CanonicalCsrSystemView::new(
            &storage,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
        let correction = [-1.0, 3.0];
        let problem = view
            .linear_problem_with_right_hand_side(&correction)
            .unwrap();

        assert_eq!(problem.right_hand_side(), correction);
        assert!(std::ptr::eq(problem.canonical_csr_system().unwrap(), &view));
        assert!(view.linear_problem_with_right_hand_side(&[1.0]).is_err());
        assert!(
            view.linear_problem_with_right_hand_side(&[1.0, f64::NAN])
                .is_err()
        );
    }

    #[test]
    fn test_call_ledger_observes_the_exact_owned_problem_operator() {
        let storage = AdversarialStorage {
            values: vec![2.0, 4.0],
            hidden_action_scale: 99.0,
        };
        let mut view = CanonicalCsrSystemView::new(
            &storage,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
        let uninstrumented = view.clone();
        let ledger = CanonicalCsrOperatorCallLedger::default();
        view.attach_test_operator_call_ledger(&ledger);
        assert_eq!(view, uninstrumented);
        let cloned_view = view.clone();
        assert_eq!(cloned_view, view);

        let problem = view.linear_problem().unwrap();
        let mut applied = [0.0; 2];
        problem.operator().apply(&[1.0, 2.0], &mut applied).unwrap();
        let mut diagonal = [0.0; 2];
        assert_eq!(
            problem.operator().diagonal(&mut diagonal).unwrap(),
            DiagonalAvailability::Available
        );
        assert_eq!(ledger.apply_calls(), 1);
        assert_eq!(ledger.diagonal_calls(), 1);

        let mut cloned_applied = [0.0; 2];
        cloned_view
            .linear_problem()
            .unwrap()
            .operator()
            .apply(&[1.0, 2.0], &mut cloned_applied)
            .unwrap();
        assert_eq!(
            ledger.apply_calls(),
            1,
            "a cloned view has a distinct identity"
        );
        assert_eq!(ledger.diagonal_calls(), 1);

        ledger.reset();
        assert_eq!(ledger.apply_calls(), 0);
        assert_eq!(ledger.diagonal_calls(), 0);
    }

    #[test]
    fn captured_view_owns_the_transposed_action_too() {
        struct GeneralStorage;

        impl CompleteCsrStorage for GeneralStorage {
            fn rows(&self) -> usize {
                2
            }

            fn columns(&self) -> usize {
                2
            }

            fn row_offsets(&self) -> &[usize] {
                &[0, 2, 4]
            }

            fn column_indices(&self) -> &[usize] {
                &[0, 1, 0, 1]
            }

            fn values(&self) -> &[f64] {
                &[1.0, 2.0, 3.0, 4.0]
            }

            fn right_hand_side(&self) -> &[f64] {
                &[0.0, 0.0]
            }
        }

        let view = CanonicalCsrSystemView::new(&GeneralStorage, LinearOperatorProperties::General)
            .unwrap();
        let mut normal = [0.0; 2];
        let mut transposed = [0.0; 2];
        view.apply(&[5.0, 6.0], &mut normal).unwrap();
        view.apply_oriented(
            crate::LinearOperatorOrientation::Transposed,
            &[5.0, 6.0],
            &mut transposed,
        )
        .unwrap();

        assert_eq!(normal, [17.0, 39.0]);
        assert_eq!(transposed, [23.0, 34.0]);
    }

    #[test]
    fn property_tags_have_stable_distinct_v1_fingerprints() {
        let base = AdversarialStorage {
            values: vec![2.0, 4.0],
            hidden_action_scale: 0.0,
        };
        let general = CanonicalCsrSystemView::new(&base, LinearOperatorProperties::General)
            .unwrap()
            .agreement_fingerprint();
        let positive_definite =
            CanonicalCsrSystemView::new(&base, LinearOperatorProperties::SymmetricPositiveDefinite)
                .unwrap()
                .agreement_fingerprint();
        let indefinite =
            CanonicalCsrSystemView::new(&base, LinearOperatorProperties::SymmetricIndefinite)
                .unwrap()
                .agreement_fingerprint();

        // Fixed v1 binary-domain goldens: changing any value is a protocol change.
        assert_eq!(
            general.as_bytes(),
            [
                216, 219, 4, 82, 206, 193, 113, 167, 83, 207, 41, 176, 17, 222, 23, 17, 207, 234,
                69, 209, 9, 170, 71, 22, 175, 141, 110, 15, 252, 202, 164, 255,
            ]
        );
        assert_eq!(
            positive_definite.as_bytes(),
            [
                88, 186, 179, 102, 168, 55, 41, 198, 188, 115, 192, 186, 233, 223, 108, 253, 195,
                137, 115, 192, 200, 37, 142, 186, 226, 174, 112, 140, 30, 28, 57, 22,
            ]
        );
        assert_eq!(
            indefinite.as_bytes(),
            [
                87, 166, 128, 249, 195, 118, 13, 23, 215, 69, 253, 109, 152, 160, 12, 76, 97, 150,
                170, 92, 255, 216, 254, 41, 138, 159, 22, 36, 159, 126, 251, 198,
            ]
        );
        assert_ne!(general, positive_definite);
        assert_ne!(general, indefinite);
        assert_ne!(positive_definite, indefinite);
    }

    #[test]
    fn every_stored_algebra_axis_changes_the_fingerprint() {
        let base = AdversarialStorage {
            values: vec![2.0, 4.0],
            hidden_action_scale: 0.0,
        };
        let fingerprint =
            CanonicalCsrSystemView::new(&base, LinearOperatorProperties::SymmetricPositiveDefinite)
                .unwrap()
                .agreement_fingerprint();

        let changed = AdversarialStorage {
            values: vec![2.0, f64::from_bits(4.0_f64.to_bits() + 1)],
            hidden_action_scale: 0.0,
        };
        assert_ne!(
            CanonicalCsrSystemView::new(
                &changed,
                LinearOperatorProperties::SymmetricPositiveDefinite,
            )
            .unwrap()
            .agreement_fingerprint(),
            fingerprint
        );
    }

    #[test]
    fn canonical_capture_normalizes_only_signed_zero() {
        let unusual = f64::from_bits(0x000f_ffff_ffff_ffff);
        let negative_zero = BitPatternStorage {
            values: [-0.0, unusual],
            right_hand_side: [-0.0, unusual],
        };
        let positive_zero = BitPatternStorage {
            values: [0.0, unusual],
            right_hand_side: [0.0, unusual],
        };
        let negative =
            CanonicalCsrSystemView::new(&negative_zero, LinearOperatorProperties::General).unwrap();
        let positive =
            CanonicalCsrSystemView::new(&positive_zero, LinearOperatorProperties::General).unwrap();

        assert_eq!(negative.values()[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(negative.right_hand_side()[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(negative.values()[1].to_bits(), unusual.to_bits());
        assert_eq!(negative.right_hand_side()[1].to_bits(), unusual.to_bits());
        assert_eq!(
            negative.agreement_fingerprint(),
            positive.agreement_fingerprint()
        );

        let mut output = [f64::NAN; 2];
        negative.apply(&[3.0, 1.0], &mut output).unwrap();
        assert_eq!(output[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(output[1].to_bits(), unusual.to_bits());
    }

    #[test]
    fn malformed_complete_storage_fails_closed() {
        struct Broken;
        impl CompleteCsrStorage for Broken {
            fn rows(&self) -> usize {
                2
            }
            fn columns(&self) -> usize {
                2
            }
            fn row_offsets(&self) -> &[usize] {
                &[0, 2, 2]
            }
            fn column_indices(&self) -> &[usize] {
                &[1, 1]
            }
            fn values(&self) -> &[f64] {
                &[1.0, 2.0]
            }
            fn right_hand_side(&self) -> &[f64] {
                &[1.0, 2.0]
            }
        }
        assert_eq!(
            CanonicalCsrSystemView::new(&Broken, LinearOperatorProperties::General)
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
    }
}
