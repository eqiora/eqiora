use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_solver::{CanonicalCsrSystemView, LinearOperatorOrientation};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::lu::{LuRef, NumericLu, SymbolicLu, factorize_symbolic_lu};
use faer::sparse::{SparseRowMat, SymbolicSparseRowMat};
use faer::{Conj, Mat, Par};

/// Owned faer symbolic state for one exact canonical CSR structure.
#[derive(Debug)]
pub(super) struct SparseLuSymbolicFactor {
    factor: SymbolicLu<usize>,
}

/// Owned faer numeric state produced under one compatible symbolic factor.
#[derive(Debug)]
pub(super) struct SparseLuNumericFactor {
    factor: NumericLu<usize, f64>,
}

pub(super) fn factor_symbolic(
    system: &CanonicalCsrSystemView,
) -> Result<SparseLuSymbolicFactor, Diagnostic> {
    let _phase =
        eqiora_execution::telemetry_span!(backend("symbolic_factorization", "faer")).entered();
    let symbolic_row = SymbolicSparseRowMat::<usize>::new_checked(
        system.rows(),
        system.columns(),
        system.row_offsets().to_vec(),
        None,
        system.column_indices().to_vec(),
    );
    let symbolic_column = symbolic_row
        .to_col_major()
        .map_err(|error| solve_failed(format!("faer CSR structure conversion failed: {error}")))?;
    let factor = factorize_symbolic_lu(symbolic_column.as_ref(), Default::default())
        .map_err(|error| solve_failed(format!("faer symbolic LU failed: {error}")))?;
    Ok(SparseLuSymbolicFactor { factor })
}

pub(super) fn factor_numeric(
    symbolic: &SparseLuSymbolicFactor,
    system: &CanonicalCsrSystemView,
) -> Result<SparseLuNumericFactor, Diagnostic> {
    let _phase =
        eqiora_execution::telemetry_span!(backend("numeric_factorization", "faer")).entered();
    let symbolic_row = SymbolicSparseRowMat::<usize>::new_checked(
        system.rows(),
        system.columns(),
        system.row_offsets().to_vec(),
        None,
        system.column_indices().to_vec(),
    );
    let row_matrix = SparseRowMat::<usize, f64>::new(symbolic_row, system.values().to_vec());
    let column_matrix = row_matrix
        .to_col_major()
        .map_err(|error| solve_failed(format!("faer CSR conversion failed: {error}")))?;

    let parallelism = Par::Seq;
    let mut factor = NumericLu::<usize, f64>::new();
    let scratch = symbolic
        .factor
        .factorize_numeric_lu_scratch::<f64>(parallelism, Default::default());
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

pub(super) fn solve_factored_oriented(
    symbolic: &SparseLuSymbolicFactor,
    numeric: &SparseLuNumericFactor,
    right_hand_side: &[f64],
    orientation: LinearOperatorOrientation,
) -> Result<Vec<f64>, Diagnostic> {
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
        LinearOperatorOrientation::Normal => symbolic
            .factor
            .solve_in_place_scratch::<f64>(1, parallelism),
        LinearOperatorOrientation::Transposed => symbolic
            .factor
            .solve_transpose_in_place_scratch::<f64>(1, parallelism),
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
        LinearOperatorOrientation::Transposed => {
            LuRef::new_unchecked(&symbolic.factor, &numeric.factor)
                .solve_transpose_in_place_with_conj(
                    Conj::No,
                    output.as_mut(),
                    parallelism,
                    MemStack::new(&mut buffer),
                );
        }
    }
    let values = output.col_as_slice(0).to_vec();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(solve_failed(
            "faer sparse LU produced a non-finite solution",
        ));
    }
    Ok(values)
}
fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}
