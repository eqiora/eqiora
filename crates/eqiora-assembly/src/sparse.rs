use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_solver::{
    CompleteCsrStorage, DiagonalAvailability, LinearOperator, RowLinearAction,
    TransposeLinearOperator,
};

use crate::{AssemblyDelta, LocalContribution};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CsrTopology {
    pub(crate) size: usize,
    pub(crate) row_offsets: std::sync::Arc<[usize]>,
    pub(crate) column_indices: std::sync::Arc<[usize]>,
}

impl CsrTopology {
    pub(crate) fn new(
        size: usize,
        row_offsets: Vec<usize>,
        column_indices: Vec<usize>,
    ) -> Result<Self, Diagnostic> {
        CsrMatrix::validate_sorted_csr(
            size,
            size,
            &row_offsets,
            &column_indices,
            column_indices.len(),
        )?;
        Ok(Self {
            size,
            row_offsets: row_offsets.into(),
            column_indices: column_indices.into(),
        })
    }
}

#[derive(Debug)]
pub(crate) struct CsrValueAssembler {
    topology: CsrTopology,
    values: Vec<f64>,
    rhs: Vec<f64>,
}

impl CsrValueAssembler {
    pub(crate) fn new(topology: CsrTopology) -> Self {
        Self {
            values: vec![0.0; topology.column_indices.len()],
            rhs: vec![0.0; topology.size],
            topology,
        }
    }

    pub(crate) fn scatter_delta(&mut self, delta: &AssemblyDelta) -> Result<(), Diagnostic> {
        if delta.target_size() != self.topology.size {
            return Err(assembly_failed(
                "assembly delta differs from prepared CSR size",
            ));
        }
        for row in delta.rows() {
            let global_row = row.row().index();
            let range =
                self.topology.row_offsets[global_row]..self.topology.row_offsets[global_row + 1];
            let next_rhs = self.rhs[global_row] + row.rhs();
            if !next_rhs.is_finite() {
                return Err(assembly_failed(
                    "prepared CSR RHS accumulation is non-finite",
                ));
            }
            for &(column, value) in row.entries() {
                let offset = self.topology.column_indices[range.clone()]
                    .binary_search(&column.index())
                    .map_err(|_| assembly_failed(format!(
                        "packet introduces ({global_row}, {}) outside its prepared CSR topology",
                        column.index()
                    )))?;
                let slot = range.start + offset;
                let next = self.values[slot] + value;
                if !next.is_finite() {
                    return Err(assembly_failed(
                        "prepared CSR value accumulation is non-finite",
                    ));
                }
            }
        }
        for row in delta.rows() {
            let global_row = row.row().index();
            let range =
                self.topology.row_offsets[global_row]..self.topology.row_offsets[global_row + 1];
            self.rhs[global_row] += row.rhs();
            for &(column, value) in row.entries() {
                let offset = self.topology.column_indices[range.clone()]
                    .binary_search(&column.index())
                    .expect("prepared topology was checked above");
                self.values[range.start + offset] += value;
            }
        }
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<LinearSystem, Diagnostic> {
        let mut row_offsets = Vec::with_capacity(self.topology.size + 1);
        let mut column_indices = Vec::new();
        let mut values = Vec::new();
        row_offsets.push(0);
        for row in 0..self.topology.size {
            let range = self.topology.row_offsets[row]..self.topology.row_offsets[row + 1];
            for (&column, &value) in self.topology.column_indices[range.clone()]
                .iter()
                .zip(&self.values[range])
            {
                if value != 0.0 {
                    column_indices.push(column);
                    values.push(value);
                }
            }
            row_offsets.push(values.len());
            if row_offsets[row + 1] == row_offsets[row] {
                return Err(assembly_failed(format!(
                    "assembled global row {row} has no nonzero entries"
                )));
            }
        }
        LinearSystem::new(
            CsrMatrix::from_sorted_csr(
                self.topology.size,
                self.topology.size,
                row_offsets,
                column_indices,
                values,
            )?,
            self.rhs,
        )
    }
}

/// Dense global degree-of-freedom index in one assembled system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DofId(usize);

impl DofId {
    /// Construct a zero-based global degree-of-freedom index.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Zero-based index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// One local trial slot after essential constraints are resolved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LocalUnknown {
    /// Free global unknown.
    Free(DofId),
    /// Fixed affine value eliminated during assembly.
    Fixed(f64),
}

/// Mapping from anonymous local rows/columns to global algebra.
///
/// A missing equation row is skipped. This represents an essential boundary
/// equation without fabricating a global row. Fixed column values are moved
/// to the global right-hand side by the assembler.
#[derive(Debug, Clone, PartialEq)]
pub struct AssemblyMap {
    equations: Vec<Option<DofId>>,
    unknowns: Vec<LocalUnknown>,
}

impl AssemblyMap {
    /// Construct a local-to-global map.
    ///
    /// # Errors
    /// Returns `EQ0806` if a fixed value is non-finite. Shape compatibility
    /// with a contribution is checked at scatter time.
    pub fn new(
        equations: Vec<Option<DofId>>,
        unknowns: Vec<LocalUnknown>,
    ) -> Result<Self, Diagnostic> {
        if unknowns
            .iter()
            .any(|unknown| matches!(unknown, LocalUnknown::Fixed(value) if !value.is_finite()))
        {
            return Err(assembly_failed("fixed assembly values must be finite"));
        }
        Ok(Self {
            equations,
            unknowns,
        })
    }

    /// Global equation associated with each local row.
    #[must_use]
    pub fn equations(&self) -> &[Option<DofId>] {
        &self.equations
    }

    /// Free or fixed global unknown associated with each local column.
    #[must_use]
    pub fn unknowns(&self) -> &[LocalUnknown] {
        &self.unknowns
    }
}

/// Deterministic row-indexed coordinate accumulator for local contributions.
///
/// This v0 implementation prioritizes a small, inspectable contract. A
/// distributed or device assembler may consume the same `AssemblyMap` and
/// `LocalContribution` without changing local operators.
#[derive(Debug, Clone)]
pub struct CooAssembler {
    size: usize,
    rows: Vec<Vec<(usize, f64)>>,
    rhs: Vec<f64>,
}

impl CooAssembler {
    /// Construct a square global system.
    ///
    /// # Errors
    /// Returns `EQ0806` when the system has zero free equations.
    pub fn new(size: usize) -> Result<Self, Diagnostic> {
        if size == 0 {
            return Err(assembly_failed(
                "assembled system requires at least one free equation",
            ));
        }
        let mut rows = Vec::with_capacity(size);
        rows.resize_with(size, Vec::new);
        Ok(Self {
            size,
            rows,
            rhs: vec![0.0; size],
        })
    }

    /// Add one local contribution using its independent assembly map.
    ///
    /// # Errors
    /// Returns `EQ0806` for shape mismatch, an out-of-range global index, or
    /// a non-finite accumulated value.
    pub fn scatter(
        &mut self,
        map: &AssemblyMap,
        local: &LocalContribution,
    ) -> Result<(), Diagnostic> {
        let delta = AssemblyDelta::from_local(self.size, map, local)?;
        self.scatter_delta(&delta)
    }

    /// Add one already-mapped packet-local delta.
    ///
    /// This is the common accumulation boundary for direct and owner-routed
    /// assembly. The complete delta is checked against current state before
    /// any entry is changed.
    ///
    /// # Errors
    /// Returns `EQ0806` for a target-size mismatch or non-finite accumulated
    /// value. Failure leaves the assembler unchanged.
    pub fn scatter_delta(&mut self, delta: &AssemblyDelta) -> Result<(), Diagnostic> {
        if delta.target_size() != self.size {
            return Err(assembly_failed(format!(
                "assembly delta targets size {} but assembler size is {}",
                delta.target_size(),
                self.size
            )));
        }
        let finite_entries = delta
            .rows()
            .iter()
            .all(|row| row_accumulations_are_finite(&self.rows[row.row().index()], row.entries()));
        let finite_rhs = delta
            .rows()
            .iter()
            .all(|row| (self.rhs[row.row().index()] + row.rhs()).is_finite());
        if !finite_entries || !finite_rhs {
            return Err(assembly_failed(
                "sparse assembly produced a non-finite accumulated value",
            ));
        }
        for row in delta.rows() {
            accumulate_row(&mut self.rows[row.row().index()], row.entries());
            self.rhs[row.row().index()] += row.rhs();
        }
        Ok(())
    }

    /// Finalize deterministic compressed sparse row storage.
    ///
    /// # Errors
    /// Returns `EQ0806` if a global row has no nonzero entry.
    pub fn finish(self) -> Result<LinearSystem, Diagnostic> {
        let entry_capacity = self.rows.iter().map(Vec::len).sum();
        let mut row_offsets = Vec::with_capacity(self.size + 1);
        let mut column_indices = Vec::with_capacity(entry_capacity);
        let mut values = Vec::with_capacity(entry_capacity);
        row_offsets.push(0);
        for (row, entries) in self.rows.into_iter().enumerate() {
            for (column, value) in entries {
                if value != 0.0 {
                    column_indices.push(column);
                    values.push(value);
                }
            }
            row_offsets.push(values.len());
            if row_offsets[row + 1] == row_offsets[row] {
                return Err(assembly_failed(format!(
                    "assembled global row {row} has no nonzero entries"
                )));
            }
        }
        LinearSystem::new(
            CsrMatrix::from_sorted_csr(self.size, self.size, row_offsets, column_indices, values)?,
            self.rhs,
        )
    }
}

fn row_accumulations_are_finite(accumulated: &[(usize, f64)], delta: &[(DofId, f64)]) -> bool {
    let mut entry = 0;
    delta.iter().all(|(column, value)| {
        while entry < accumulated.len() && accumulated[entry].0 < column.index() {
            entry += 1;
        }
        let current = if entry < accumulated.len() && accumulated[entry].0 == column.index() {
            accumulated[entry].1
        } else {
            0.0
        };
        (current + value).is_finite()
    })
}

fn accumulate_row(accumulated: &mut Vec<(usize, f64)>, delta: &[(DofId, f64)]) {
    if accumulated.is_empty() {
        accumulated.extend(
            delta
                .iter()
                .map(|(column, value)| (column.index(), 0.0 + value)),
        );
        return;
    }

    let mut entry = 0;
    for &(column, value) in delta {
        while entry < accumulated.len() && accumulated[entry].0 < column.index() {
            entry += 1;
        }
        if entry < accumulated.len() && accumulated[entry].0 == column.index() {
            accumulated[entry].1 += value;
        } else {
            accumulated.insert(entry, (column.index(), 0.0 + value));
        }
        entry += 1;
    }
}

/// Immutable compressed sparse row matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    rows: usize,
    columns: usize,
    row_offsets: std::sync::Arc<[usize]>,
    column_indices: std::sync::Arc<[usize]>,
    values: Vec<f64>,
}

impl CsrMatrix {
    /// Construct validated CSR storage with strictly ordered columns per row.
    ///
    /// This is the single admission boundary for finalized sparse storage.
    /// Callers retain no mutable access to the admitted arrays, so row lookup
    /// and backend adapters may rely on the checked ordering thereafter.
    ///
    /// # Errors
    /// Returns `EQ0806` for zero or inconsistent shape, invalid row offsets,
    /// out-of-range or non-increasing columns, or a non-finite value.
    pub fn from_sorted_csr(
        rows: usize,
        columns: usize,
        row_offsets: Vec<usize>,
        column_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self, Diagnostic> {
        Self::validate_sorted_csr(rows, columns, &row_offsets, &column_indices, values.len())?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(assembly_failed("CSR values must all be finite"));
        }
        Ok(Self {
            rows,
            columns,
            row_offsets: row_offsets.into(),
            column_indices: column_indices.into(),
            values,
        })
    }

    fn validate_sorted_csr(
        rows: usize,
        columns: usize,
        row_offsets: &[usize],
        column_indices: &[usize],
        value_count: usize,
    ) -> Result<(), Diagnostic> {
        if rows == 0 || columns == 0 {
            return Err(assembly_failed(
                "CSR storage requires positive row and column counts",
            ));
        }
        let expected_offsets = rows
            .checked_add(1)
            .ok_or_else(|| assembly_failed("CSR row count overflows its offset table"))?;
        if row_offsets.len() != expected_offsets {
            return Err(assembly_failed(format!(
                "CSR with {rows} rows requires {expected_offsets} row offsets, found {}",
                row_offsets.len()
            )));
        }
        if column_indices.len() != value_count {
            return Err(assembly_failed(format!(
                "CSR column/value lengths differ: {} versus {}",
                column_indices.len(),
                value_count
            )));
        }
        if row_offsets.first() != Some(&0) || row_offsets.last() != Some(&value_count) {
            return Err(assembly_failed(
                "CSR row offsets must start at zero and end at the nonzero count",
            ));
        }
        for row in 0..rows {
            let start = row_offsets[row];
            let end = row_offsets[row + 1];
            if start > end || end > column_indices.len() {
                return Err(assembly_failed(format!(
                    "CSR row {row} has invalid offset range {start}..{end}"
                )));
            }
            let columns_in_row = &column_indices[start..end];
            if let Some(column) = columns_in_row.iter().find(|column| **column >= columns) {
                return Err(assembly_failed(format!(
                    "CSR row {row} contains column {column} outside 0..{columns}"
                )));
            }
            if columns_in_row.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(assembly_failed(format!(
                    "CSR row {row} columns must be strictly increasing"
                )));
            }
        }

        Ok(())
    }

    /// Row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Column count.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// CSR row offsets.
    #[must_use]
    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    /// CSR column indices.
    #[must_use]
    pub fn column_indices(&self) -> &[usize] {
        &self.column_indices
    }

    /// CSR nonzero values.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Multiply by a dense vector.
    ///
    /// # Errors
    /// Returns `EQ0802` for the wrong input size or a non-finite result.
    pub fn multiply(&self, input: &[f64]) -> Result<Vec<f64>, Diagnostic> {
        let mut output = vec![0.0; self.rows];
        self.multiply_into(input, &mut output)?;
        Ok(output)
    }

    /// Multiply into caller-owned storage without allocating.
    ///
    /// # Errors
    /// Returns `EQ0802` for a shape mismatch or non-finite result.
    pub fn multiply_into(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        if input.len() != self.columns || output.len() != self.rows {
            return Err(solve_failed(format!(
                "sparse matrix is {}x{} but input/output have {}/{} values",
                self.rows,
                self.columns,
                input.len(),
                output.len()
            )));
        }
        self.apply_rows(0..self.rows, input, output)
    }

    /// Read one matrix entry; structural zeros return zero.
    #[must_use]
    pub fn entry(&self, row: usize, column: usize) -> Option<f64> {
        if row >= self.rows || column >= self.columns {
            return None;
        }
        let range = self.row_offsets[row]..self.row_offsets[row + 1];
        match self.column_indices[range.clone()].binary_search(&column) {
            Ok(offset) => Some(self.values[range.start + offset]),
            Err(_) => Some(0.0),
        }
    }
}

impl LinearOperator for CsrMatrix {
    fn rows(&self) -> usize {
        self.rows
    }

    fn columns(&self) -> usize {
        self.columns
    }

    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        self.multiply_into(input, output)
    }

    fn row_action(&self) -> Option<&dyn RowLinearAction> {
        Some(self)
    }

    fn diagonal(&self, output: &mut [f64]) -> Result<DiagonalAvailability, Diagnostic> {
        if self.rows != self.columns || output.len() != self.rows {
            return Err(solve_failed(
                "CSR diagonal output must match a square matrix",
            ));
        }
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.entry(row, row).expect("row and column are in range");
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(solve_failed("CSR diagonal contains a non-finite value"));
        }
        Ok(DiagonalAvailability::Available)
    }
}

impl RowLinearAction for CsrMatrix {
    fn apply_rows(
        &self,
        rows: std::ops::Range<usize>,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        let row_count = rows
            .end
            .checked_sub(rows.start)
            .ok_or_else(|| solve_failed("sparse row action requires a nondecreasing row range"))?;
        if rows.end > self.rows || input.len() != self.columns || output.len() != row_count {
            return Err(solve_failed(format!(
                "sparse row action for {:?} of {}x{} has input/output sizes {}/{}",
                rows,
                self.rows,
                self.columns,
                input.len(),
                output.len()
            )));
        }
        for (row, output_value) in rows.zip(output.iter_mut()) {
            *output_value = 0.0;
            for entry in self.row_offsets[row]..self.row_offsets[row + 1] {
                *output_value += self.values[entry] * input[self.column_indices[entry]];
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(solve_failed(
                "sparse matrix-vector product produced a non-finite value",
            ));
        }
        Ok(())
    }
}

impl TransposeLinearOperator for CsrMatrix {
    fn apply_transpose(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        if input.len() != self.rows || output.len() != self.columns {
            return Err(solve_failed(format!(
                "transposed sparse matrix is {}x{} but input/output have {}/{} values",
                self.columns,
                self.rows,
                input.len(),
                output.len()
            )));
        }
        output.fill(0.0);
        for (row, input_value) in input.iter().enumerate() {
            for entry in self.row_offsets[row]..self.row_offsets[row + 1] {
                output[self.column_indices[entry]] += self.values[entry] * input_value;
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(solve_failed(
                "transposed sparse matrix-vector product produced a non-finite value",
            ));
        }
        Ok(())
    }
}

/// Sparse square matrix and matching right-hand side.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearSystem {
    matrix: CsrMatrix,
    rhs: Vec<f64>,
}

impl LinearSystem {
    /// Admit one canonical square sparse matrix and matching finite RHS.
    ///
    /// [`CsrMatrix`] deliberately supports rectangular operators and empty
    /// rows. A complete assembled linear system is narrower: it is square,
    /// omits explicit zeros, and has at least one nonzero in every row.
    ///
    /// # Errors
    /// Returns `EQ0806` for a nonsquare matrix, RHS shape/value mismatch,
    /// explicit stored zero, or structurally empty equation row.
    pub fn new(matrix: CsrMatrix, rhs: Vec<f64>) -> Result<Self, Diagnostic> {
        if matrix.rows != matrix.columns {
            return Err(assembly_failed(format!(
                "linear system matrix must be square, found {}x{}",
                matrix.rows, matrix.columns
            )));
        }
        if rhs.len() != matrix.rows || rhs.iter().any(|value| !value.is_finite()) {
            return Err(assembly_failed(format!(
                "linear system with {} rows requires {} finite right-hand-side values, found {}",
                matrix.rows,
                matrix.rows,
                rhs.len()
            )));
        }
        if matrix.values.contains(&0.0) {
            return Err(assembly_failed(
                "linear system CSR must omit explicit zero entries",
            ));
        }
        if let Some(row) =
            (0..matrix.rows).find(|row| matrix.row_offsets[*row] == matrix.row_offsets[*row + 1])
        {
            return Err(assembly_failed(format!(
                "assembled global row {row} has no nonzero entries"
            )));
        }
        Ok(Self { matrix, rhs })
    }

    /// Sparse operator.
    #[must_use]
    pub const fn matrix(&self) -> &CsrMatrix {
        &self.matrix
    }

    /// Right-hand side.
    #[must_use]
    pub fn rhs(&self) -> &[f64] {
        &self.rhs
    }
}

impl CompleteCsrStorage for LinearSystem {
    fn rows(&self) -> usize {
        self.matrix.rows()
    }

    fn columns(&self) -> usize {
        self.matrix.columns()
    }

    fn row_offsets(&self) -> &[usize] {
        self.matrix.row_offsets()
    }

    fn column_indices(&self) -> &[usize] {
        self.matrix.column_indices()
    }

    fn values(&self) -> &[f64] {
        self.matrix.values()
    }

    fn right_hand_side(&self) -> &[f64] {
        self.rhs()
    }
}

fn assembly_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::ASSEMBLY_FAILED, message)
}

fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

#[cfg(test)]
mod tests;
