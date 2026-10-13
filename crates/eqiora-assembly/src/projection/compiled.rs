//! Fixed duplicate groups; numerical folds retain local row-major order.
use super::*;
use std::collections::BTreeMap;

#[cfg(test)]
thread_local! {
    static COMPILATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[cfg(test)]
pub(crate) fn compilation_count() -> usize {
    COMPILATIONS.with(std::cell::Cell::get)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompiledProjection {
    target_size: usize,
    local_rows: usize,
    local_columns: usize,
    rows: Vec<(DofId, Vec<usize>)>,
    columns: Vec<(DofId, Vec<usize>)>,
    fixed: Vec<usize>,
}

impl CompiledProjection {
    pub(crate) fn new<S: Scalar + ComplexFloat>(
        target_size: usize,
        map: &AssemblyMap<S>,
    ) -> Result<Self, Diagnostic> {
        #[cfg(test)]
        COMPILATIONS.with(|count| count.set(count.get() + 1));
        if target_size == 0 {
            return Err(assembly_failed(
                "an assembly delta requires a nonempty target",
            ));
        }
        let mut rows = BTreeMap::<_, Vec<_>>::new();
        for (local, global) in map.equations().iter().enumerate() {
            if let Some(global) = global {
                check_dof(target_size, *global)?;
                rows.entry(*global).or_default().push(local);
            }
        }
        let mut columns = BTreeMap::<_, Vec<_>>::new();
        let mut fixed = Vec::new();
        for (local, unknown) in map.unknowns().iter().enumerate() {
            match unknown {
                LocalUnknown::Free(global) => {
                    check_dof(target_size, *global)?;
                    columns.entry(*global).or_default().push(local);
                }
                LocalUnknown::Fixed(_) => fixed.push(local),
            }
        }
        Ok(Self {
            target_size,
            local_rows: map.equations().len(),
            local_columns: map.unknowns().len(),
            rows: rows.into_iter().collect(),
            columns: columns.into_iter().collect(),
            fixed,
        })
    }

    pub(crate) fn rows(&self) -> impl Iterator<Item = DofId> + '_ {
        self.rows.iter().map(|(row, _)| *row)
    }

    pub(crate) fn columns(&self) -> impl Iterator<Item = DofId> + '_ {
        self.columns.iter().map(|(column, _)| *column)
    }

    // The caller owns exact map admission: either compilation from this map or
    // the prepared plan's exact packet binding, including fixed coefficient bits.
    pub(crate) fn project<S: Scalar + ComplexFloat>(
        &self,
        map: &AssemblyMap<S>,
        local: &LocalContribution<S>,
    ) -> Result<AssemblyDelta<S>, Diagnostic> {
        if local.rows() != self.local_rows || local.columns() != self.local_columns {
            return Err(assembly_failed(
                "local contribution differs from its compiled projection shape",
            ));
        }
        let mut rows = Vec::with_capacity(self.rows.len());
        for &(row, ref local_rows) in &self.rows {
            let mut rhs = S::zero();
            for &local_row in local_rows {
                rhs = rhs + local.rhs()[local_row];
                for &column in &self.fixed {
                    let LocalUnknown::Fixed(value) = map.unknowns()[column] else {
                        return Err(assembly_failed("compiled fixed column changed its binding"));
                    };
                    rhs = rhs - local.matrix()[local_row * self.local_columns + column] * value;
                }
            }
            let mut entries = Vec::with_capacity(self.columns.len());
            for &(column, ref local_columns) in &self.columns {
                let mut value = S::zero();
                for &local_row in local_rows {
                    for &local_column in local_columns {
                        value =
                            value + local.matrix()[local_row * self.local_columns + local_column];
                    }
                }
                if !value.is_finite() {
                    return Err(assembly_failed(
                        "sparse assembly produced a non-finite projected value",
                    ));
                }
                entries.push((column, value));
            }
            if !rhs.is_finite() {
                return Err(assembly_failed(
                    "sparse assembly produced a non-finite projected value",
                ));
            }
            rows.push(AssemblyRowDelta { row, entries, rhs });
        }
        Ok(AssemblyDelta {
            target_size: self.target_size,
            rows,
        })
    }
}
