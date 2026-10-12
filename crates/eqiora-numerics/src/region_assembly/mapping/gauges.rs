//! Method-selected multiplier coordinates, distinct from physical Field DOFs.
use super::*;
use eqiora_solver::AlgebraicConstraint;

impl<S: Coefficient + Send + Sync> RegionDofMap<S> {
    pub(crate) fn with_algebraic_constraints(
        mut self,
        constraints: &[AlgebraicConstraint],
    ) -> Result<Self, Diagnostic> {
        if !self.gauges.is_empty() {
            return Err(invalid(
                "algebraic constraints already have an exact binding",
            ));
        }
        let mut fields = BTreeSet::new();
        for constraint in constraints {
            let field = constraint.field().erase();
            let (_, layout) = self
                .fields
                .get(&field)
                .ok_or_else(|| invalid("algebraic constraint has a foreign Field"))?;
            if layout.components != 1
                || !layout.value_type.shape().is_scalar()
                || !fields.insert(field)
            {
                return Err(invalid(
                    "zero-integral constraints require unique exact scalar Fields",
                ));
            }
        }
        let full_count = self
            .full_count
            .checked_add(fields.len())
            .ok_or_else(|| invalid("algebraic constraint count overflows usize"))?;
        self.constraints = self.constraints.append_free(fields.len())?;
        self.gauges = fields
            .into_iter()
            .enumerate()
            .map(|(offset, field)| (field, self.full_count + offset))
            .collect();
        self.full_count = full_count;
        Ok(self)
    }

    pub(crate) fn constraint_global_dof(&self, constraint: AlgebraicConstraint) -> Option<usize> {
        self.gauges.get(&constraint.field().erase()).copied()
    }

    pub(crate) fn constraint_free_dof(&self, constraint: AlgebraicConstraint) -> Option<DofId> {
        self.constraint_global_dof(constraint)
            .and_then(|index| self.constraints.free_index(index))
    }

    /// One cell's scalar basis and the shared global multiplier. Essential
    /// elimination and full-row recovery use the same existing constrained map.
    pub(crate) fn constraint_map(
        &self,
        cell: usize,
        constraint: AlgebraicConstraint,
        reduced: bool,
    ) -> Result<AssemblyMap<S>, Diagnostic> {
        let gauge = self
            .constraint_global_dof(constraint)
            .ok_or_else(|| invalid("constraint packet has no exact multiplier binding"))?;
        let mut globals = self
            .cell_field_keys(cell, constraint.field().erase())?
            .iter()
            .map(|key| self.globals[key])
            .collect::<Vec<_>>();
        globals.push(gauge);
        if reduced {
            self.constraints.reduced_map(&globals)
        } else {
            self.constraints.full_map(&globals)
        }
    }
}

#[cfg(test)]
mod tests;
