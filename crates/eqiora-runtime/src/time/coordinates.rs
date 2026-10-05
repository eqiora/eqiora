//! Source-preserving first-order coordinate selection, before equation classification.
use super::*;

pub(crate) struct StateOrder {
    pub(crate) state_coordinates: Vec<eqiora_core::TimeStateCoordinate>,
    pub(crate) coordinates: HashMap<(Id<kinds::Field>, u32), usize>,
    pub(crate) companions: Vec<(usize, usize)>,
    highest: Vec<usize>,
}

impl StateOrder {
    pub(crate) fn rate_symbols(&self) -> Vec<SymbolRef> {
        self.highest
            .iter()
            .map(|&index| {
                let coordinate = self.state_coordinates[index];
                let (field, order) = (coordinate.field(), coordinate.derivative_order());
                SymbolRef::Derivative(field, std::num::NonZeroU32::new(order + 1).unwrap())
            })
            .collect()
    }

    pub(crate) fn derivative_matrix(
        &self,
        relation: Id<kinds::Relation>,
        jacobian: &ConstantSymbolJacobian,
    ) -> Result<ConstantDerivativeMatrixProof, Diagnostic> {
        derivative_matrix(
            relation,
            self.state_coordinates.len(),
            &self.highest,
            &self.companions,
            jacobian.coefficients(),
        )
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
            state_coordinates.push(eqiora_core::TimeStateCoordinate::new(
                field, order, 0, false,
            ));
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

fn derivative_matrix(
    relation: Id<kinds::Relation>,
    n: usize,
    highest: &[usize],
    companions: &[(usize, usize)],
    coefficients: &[f64],
) -> Result<ConstantDerivativeMatrixProof, Diagnostic> {
    let source_count = highest.len();
    if coefficients.len()
        != source_count
            .checked_mul(source_count)
            .ok_or_else(|| invalid_time(relation, "derivative matrix cardinality overflows"))?
    {
        return Err(invalid_time(
            relation,
            "time system requires one authored residual equation per source Field",
        ));
    }

    let size = n
        .checked_mul(n)
        .ok_or_else(|| invalid_time(relation, "first-order matrix cardinality overflows"))?;
    let source_coefficients = coefficients;
    let mut coefficients = Vec::new();
    coefficients.try_reserve_exact(size).map_err(|_| {
        invalid_time(
            relation,
            "first-order derivative matrix exceeds available storage",
        )
    })?;
    coefficients.resize(size, 0.);
    for row in 0..source_count {
        for (source, &column) in highest.iter().enumerate() {
            coefficients[row * n + column] = source_coefficients[row * source_count + source];
        }
    }
    for (row, &(coordinate, _)) in companions.iter().enumerate() {
        coefficients[(source_count + row) * n + coordinate] = 1.;
    }
    ConstantDerivativeMatrixProof::new(n, coefficients)
}

pub(super) struct ComponentStateOrder {
    pub(super) state_coordinates: Vec<eqiora_core::TimeStateCoordinate>,
    pub(super) coordinates: HashMap<eqiora_ir::ScalarSymbolCoordinate, usize>,
    pub(super) companions: Vec<(usize, usize)>,
    highest: Vec<usize>,
    rates: Vec<eqiora_ir::ScalarSymbolCoordinate>,
}
impl ComponentStateOrder {
    pub(super) fn rate_symbols(&self) -> Vec<eqiora_ir::ScalarSymbolCoordinate> {
        self.rates.clone()
    }
    pub(super) fn derivative_matrix(
        &self,
        relation: Id<kinds::Relation>,
        coefficients: &[f64],
    ) -> Result<ConstantDerivativeMatrixProof, Diagnostic> {
        derivative_matrix(
            relation,
            self.state_coordinates.len(),
            &self.highest,
            &self.companions,
            coefficients,
        )
    }
}

pub(super) fn component_state_order(
    program: &KernelProgram,
    relation: Id<kinds::Relation>,
    operator: &TimeOperator,
) -> Result<ComponentStateOrder, Diagnostic> {
    let mut fields = Vec::new();
    let mut orders = HashMap::new();
    for source in operator.symbols() {
        let (field, order) = match source.symbol() {
            SymbolRef::Field(field) => (field, 1),
            SymbolRef::Derivative(field, order) => (field, order.get()),
            _ => continue,
        };
        if let Some(highest) = orders.get_mut(&field) {
            *highest = order.max(*highest);
        } else {
            fields.push(field);
            orders.insert(field, order);
        }
    }
    let mut result = ComponentStateOrder {
        state_coordinates: Vec::new(),
        coordinates: HashMap::new(),
        companions: Vec::new(),
        highest: Vec::new(),
        rates: Vec::new(),
    };
    for field in fields {
        let Some(KernelNode::Field(definition)) = program.node(field.erase()) else {
            return Err(invalid_time(relation, "time Field has no value type"));
        };
        let ty = definition.value_type();
        let parts = if ty.scalar_domain() == eqiora_core::ScalarDomain::Complex {
            2
        } else {
            1
        };
        let sources = eqiora_ir::ScalarSymbolCoordinate::for_value(SymbolRef::Field(field), ty)?;
        let count = sources
            .len()
            .checked_mul(orders[&field] as usize)
            .ok_or_else(|| {
                invalid_time(relation, "first-order coordinate cardinality overflows")
            })?;
        result
            .state_coordinates
            .try_reserve_exact(count)
            .map_err(|_| {
                invalid_time(relation, "first-order coordinates exceed available storage")
            })?;
        for (component, source) in sources.into_iter().enumerate() {
            for order in 0..orders[&field] {
                let index = result.state_coordinates.len();
                let symbol = if order == 0 {
                    SymbolRef::Field(field)
                } else {
                    SymbolRef::Derivative(field, std::num::NonZeroU32::new(order).unwrap())
                };
                result.coordinates.insert(source.with_symbol(symbol), index);
                result
                    .state_coordinates
                    .push(eqiora_core::TimeStateCoordinate::new(
                        field,
                        order,
                        component / parts,
                        source.is_imaginary(),
                    ));
                if order + 1 < orders[&field] {
                    result.companions.push((index, index + 1));
                } else {
                    result.highest.push(index);
                    result.rates.push(source.with_symbol(SymbolRef::Derivative(
                        field,
                        std::num::NonZeroU32::new(order + 1).unwrap(),
                    )));
                }
            }
        }
    }
    if result.highest.is_empty() || operator.residual_count() != result.highest.len() {
        return Err(invalid_time(
            relation,
            "time system requires one scalar residual per source Field component and part",
        ));
    }
    Ok(result)
}
