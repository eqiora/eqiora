//! Source-preserving first-order coordinate selection, before equation classification.
use super::*;

pub(crate) struct StateOrder {
    pub(crate) state_coordinates: Vec<(Id<kinds::Field>, u32)>,
    pub(crate) coordinates: HashMap<(Id<kinds::Field>, u32), usize>,
    pub(crate) companions: Vec<(usize, usize)>,
    highest: Vec<usize>,
}

impl StateOrder {
    pub(crate) fn rate_symbols(&self) -> Vec<SymbolRef> {
        self.highest
            .iter()
            .map(|&index| {
                let (field, order) = self.state_coordinates[index];
                SymbolRef::Derivative(field, std::num::NonZeroU32::new(order + 1).unwrap())
            })
            .collect()
    }

    pub(crate) fn derivative_matrix(
        &self,
        relation: Id<kinds::Relation>,
        jacobian: &ConstantSymbolJacobian,
    ) -> Result<ConstantDerivativeMatrixProof, Diagnostic> {
        let source_count = self.highest.len();
        if jacobian.row_count() != source_count || jacobian.column_count() != source_count {
            return Err(invalid_time(
                relation,
                "time system requires one authored residual equation per source Field",
            ));
        }
        let n = self.state_coordinates.len();
        let size = n
            .checked_mul(n)
            .ok_or_else(|| invalid_time(relation, "first-order matrix cardinality overflows"))?;
        let mut coefficients = Vec::new();
        coefficients.try_reserve_exact(size).map_err(|_| {
            invalid_time(
                relation,
                "first-order derivative matrix exceeds available storage",
            )
        })?;
        coefficients.resize(size, 0.);
        for row in 0..source_count {
            for (source, &column) in self.highest.iter().enumerate() {
                coefficients[row * n + column] =
                    jacobian.coefficients()[row * source_count + source];
            }
        }
        for (row, &(coordinate, _)) in self.companions.iter().enumerate() {
            coefficients[(source_count + row) * n + coordinate] = 1.;
        }
        ConstantDerivativeMatrixProof::new(n, coefficients)
    }
}

pub(crate) fn state_order(
    relation: Id<kinds::Relation>,
    operator: &ScalarOperatorIr,
) -> Result<StateOrder, Diagnostic> {
    let mut fields = Vec::new();
    let mut orders = HashMap::new();
    for symbol in operator.symbols() {
        let (field, order) = match *symbol {
            SymbolRef::Field(field) => (field, 1),
            SymbolRef::Derivative(field, order) => (field, order.get()),
            _ => continue,
        };
        match orders.entry(field) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                fields.push(field);
                entry.insert(order);
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                *entry.get_mut() = (*entry.get()).max(order);
            }
        }
    }
    if fields.is_empty() || operator.residual_count() != fields.len() {
        return Err(invalid_time(
            relation,
            "time system requires one authored residual equation per source Field",
        ));
    }
    let count = orders
        .values()
        .try_fold(0_usize, |count, order| count.checked_add(*order as usize))
        .ok_or_else(|| invalid_time(relation, "first-order coordinate cardinality overflows"))?;
    let mut state_coordinates = Vec::new();
    state_coordinates
        .try_reserve_exact(count)
        .map_err(|_| invalid_time(relation, "first-order coordinates exceed available storage"))?;
    let mut coordinates = HashMap::new();
    let mut highest = Vec::new();
    let mut companions = Vec::new();
    for field in fields {
        for order in 0..orders[&field] {
            let index = state_coordinates.len();
            coordinates.insert((field, order), index);
            state_coordinates.push((field, order));
            if order + 1 < orders[&field] {
                companions.push((index, index + 1));
            } else {
                highest.push(index);
            }
        }
    }
    Ok(StateOrder {
        state_coordinates,
        coordinates,
        companions,
        highest,
    })
}
