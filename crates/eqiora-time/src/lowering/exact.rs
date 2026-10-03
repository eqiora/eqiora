//! Shared exact elimination and orthogonal compatibility residuals.
use num_rational::BigRational;
use num_traits::Zero;

pub(super) fn eliminate(matrix: &mut [Vec<BigRational>], columns: usize) -> Vec<usize> {
    let mut pivots = Vec::new();
    for column in 0..columns {
        let rank = pivots.len();
        let Some(pivot) = (rank..matrix.len()).find(|&row| !matrix[row][column].is_zero()) else {
            continue;
        };
        matrix.swap(rank, pivot);
        for row in rank + 1..matrix.len() {
            let factor = &matrix[row][column] / &matrix[rank][column];
            if factor.is_zero() {
                continue;
            }
            for trailing in column..matrix[row].len() {
                let correction = &factor * &matrix[rank][trailing];
                matrix[row][trailing] -= correction;
            }
        }
        pivots.push(column);
        if pivots.len() == matrix.len() {
            break;
        }
    }
    pivots
}

pub(super) fn residual(matrix: &[Vec<BigRational>], rhs: &[BigRational]) -> Vec<BigRational> {
    let mut reduced = matrix.to_vec();
    let basis = eliminate(&mut reduced, matrix[0].len());
    let rank = basis.len();
    if rank == 0 {
        return rhs.to_vec();
    }
    // C contains independent original columns. Exact normal equations avoid
    // introducing the floating-point condition-number square of C^T C.
    let mut gram = basis
        .iter()
        .map(|&left| {
            let mut row = basis
                .iter()
                .map(|&right| matrix.iter().map(|row| &row[left] * &row[right]).sum())
                .collect::<Vec<BigRational>>();
            row.push(
                matrix
                    .iter()
                    .zip(rhs)
                    .map(|(row, value)| &row[left] * value)
                    .sum(),
            );
            row
        })
        .collect::<Vec<_>>();
    let pivots = eliminate(&mut gram, rank);
    debug_assert_eq!(pivots.len(), rank);
    let mut solution = vec![BigRational::zero(); rank];
    for row in (0..rank).rev() {
        let remainder: BigRational = ((row + 1)..rank)
            .map(|column| &gram[row][column] * &solution[column])
            .sum();
        solution[row] = (&gram[row][rank] - remainder) / &gram[row][row];
    }
    matrix
        .iter()
        .zip(rhs)
        .map(|(row, value)| {
            value
                - basis
                    .iter()
                    .zip(&solution)
                    .map(|(&column, weight)| &row[column] * weight)
                    .sum::<BigRational>()
        })
        .collect()
}
