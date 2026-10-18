use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, Scalar};
use eqiora_solver::{CanonicalCsrSystemView, LinearOperatorOrientation};
use faer::complex::ComplexFloat;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::lu::{LuRef, NumericLu, SymbolicLu, factorize_symbolic_lu};
use faer::sparse::{SparseRowMat, SymbolicSparseRowMat};
use faer::traits::ComplexField;
use faer::{Conj, Mat, Par};

/// Scalars supported by both the typed Eqiora boundary and faer's native LU.
pub(super) trait SparseLuScalar:
    Scalar + ComplexFloat<Real = f64> + ComplexField<Real = f64> + Sync
{
}
impl<S: Scalar + ComplexFloat<Real = f64> + ComplexField<Real = f64> + Sync> SparseLuScalar for S {}

/// Owned faer symbolic state for one exact canonical CSR structure.
#[derive(Debug)]
pub(super) struct SparseLuSymbolicFactor {
    factor: SymbolicLu<usize>,
}

/// Owned faer numeric state produced under one compatible symbolic factor.
#[derive(Debug)]
pub(super) struct SparseLuNumericFactor<S> {
    factor: NumericLu<usize, S>,
}

fn symbolic_rows<S: SparseLuScalar>(
    system: &CanonicalCsrSystemView<S>,
) -> SymbolicSparseRowMat<usize> {
    SymbolicSparseRowMat::new_checked(
        system.rows(),
        system.columns(),
        system.row_offsets().to_vec(),
        None,
        system.column_indices().to_vec(),
    )
}

pub(super) fn factor_symbolic<S: SparseLuScalar>(
    system: &CanonicalCsrSystemView<S>,
) -> Result<SparseLuSymbolicFactor, Diagnostic> {
    let _phase =
        eqiora_execution::telemetry_span!(backend("symbolic_factorization", "faer")).entered();
    let symbolic_row = symbolic_rows(system);
    let symbolic_column = symbolic_row
        .to_col_major()
        .map_err(|error| solve_failed(format!("faer CSR structure conversion failed: {error}")))?;
    let factor = factorize_symbolic_lu(symbolic_column.as_ref(), Default::default())
        .map_err(|error| solve_failed(format!("faer symbolic LU failed: {error}")))?;
    Ok(SparseLuSymbolicFactor { factor })
}

pub(super) fn factor_numeric<S: SparseLuScalar>(
    symbolic: &SparseLuSymbolicFactor,
    system: &CanonicalCsrSystemView<S>,
) -> Result<SparseLuNumericFactor<S>, Diagnostic> {
    let _phase =
        eqiora_execution::telemetry_span!(backend("numeric_factorization", "faer")).entered();
    let symbolic_row = symbolic_rows(system);
    let row_matrix = SparseRowMat::<usize, S>::new(symbolic_row, system.values().to_vec());
    let column_matrix = row_matrix
        .to_col_major()
        .map_err(|error| solve_failed(format!("faer CSR conversion failed: {error}")))?;

    let parallelism = Par::Seq;
    let mut factor = NumericLu::<usize, S>::new();
    let scratch = symbolic
        .factor
        .factorize_numeric_lu_scratch::<S>(parallelism, Default::default());
    let mut buffer = MemBuffer::try_new(scratch)
        .map_err(|error| solve_failed(format!("faer numeric LU workspace failed: {error}")))?;
    symbolic
        .factor
        .factorize_numeric_lu(
            &mut factor,
            column_matrix.as_ref(),
            parallelism,
            MemStack::new(&mut buffer),
            Default::default(),
        )
        .map_err(|error| solve_failed(format!("faer numeric LU failed: {error}")))?;
    Ok(SparseLuNumericFactor { factor })
}

pub(super) fn solve_factored_oriented<S: SparseLuScalar>(
    symbolic: &SparseLuSymbolicFactor,
    numeric: &SparseLuNumericFactor<S>,
    right_hand_side: &[S],
    orientation: LinearOperatorOrientation,
) -> Result<Vec<S>, Diagnostic> {
    let _phase = eqiora_execution::telemetry_span!(backend("backsolve", "faer")).entered();
    if right_hand_side.len() != symbolic.factor.nrows()
        || symbolic.factor.nrows() != symbolic.factor.ncols()
    {
        return Err(solve_failed(
            "faer sparse LU factors and right-hand side have incompatible dimensions",
        ));
    }
    let parallelism = Par::Seq;
    let mut output = Mat::from_fn(right_hand_side.len(), 1, |row, _| right_hand_side[row]);
    let scratch = match orientation {
        LinearOperatorOrientation::Normal => {
            symbolic.factor.solve_in_place_scratch::<S>(1, parallelism)
        }
        LinearOperatorOrientation::Transposed | LinearOperatorOrientation::ConjugateTransposed => {
            symbolic
                .factor
                .solve_transpose_in_place_scratch::<S>(1, parallelism)
        }
    };
    let mut buffer = MemBuffer::try_new(scratch)
        .map_err(|error| solve_failed(format!("faer sparse LU solve workspace failed: {error}")))?;
    match orientation {
        LinearOperatorOrientation::Normal => {
            LuRef::new_unchecked(&symbolic.factor, &numeric.factor).solve_in_place_with_conj(
                Conj::No,
                output.as_mut(),
                parallelism,
                MemStack::new(&mut buffer),
            );
        }
        LinearOperatorOrientation::Transposed | LinearOperatorOrientation::ConjugateTransposed => {
            LuRef::new_unchecked(&symbolic.factor, &numeric.factor)
                .solve_transpose_in_place_with_conj(
                    if orientation == LinearOperatorOrientation::ConjugateTransposed {
                        Conj::Yes
                    } else {
                        Conj::No
                    },
                    output.as_mut(),
                    parallelism,
                    MemStack::new(&mut buffer),
                );
        }
    }
    let values = output.col_as_slice(0).to_vec();
    if values
        .iter()
        .any(|value| !value.re().is_finite() || !value.im().is_finite())
    {
        return Err(solve_failed(
            "faer sparse LU produced a non-finite solution",
        ));
    }
    Ok(values)
}
fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}
