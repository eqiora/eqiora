//! Model-derived validation of finite real/complex time witnesses.
use super::*;
use eqiora_ir::{ComponentScalarization, ScalarSymbolCoordinate};
use eqiora_schema::kernel::typing::TypedResidual;
use std::collections::HashMap;

pub(super) fn validate(
    proof: &TimeLoweringProof,
    program: &KernelProgram,
    typed: &TypedResidual<eqiora_core::RawId>,
) -> Result<(), Diagnostic> {
    let operator =
        ComponentScalarization::lower(typed).map_err(|error| invalid_artifact(error.message()))?;
    let mut fields = Vec::new();
    let mut orders = HashMap::<Id<kinds::Field>, u32>::new();
    for source in operator.rows().iter().flat_map(|row| row.symbols()) {
        let (field, order) = match source.symbol() {
            SymbolRef::Field(field) => (field, 1),
            SymbolRef::Derivative(field, order) => (field, order.get()),
            _ => continue,
        };
        match orders.get_mut(&field) {
            Some(highest) => *highest = (*highest).max(order),
            None => {
                fields.push(field);
                orders.insert(field, order);
            }
        }
    }
    let mut coordinates = Vec::new();
    let mut highest = Vec::new();
    let mut rates = Vec::new();
    let mut companions = Vec::new();
    for field in fields {
        let Some(KernelNode::Field(definition)) = program.node(field.erase()) else {
            return Err(invalid_artifact(
                "component time witness references an absent Field",
            ));
        };
        let ty = definition.value_type();
        let parts = if ty.scalar_domain() == eqiora_core::ScalarDomain::Complex {
            2
        } else {
            1
        };
        let sources = ScalarSymbolCoordinate::for_value(SymbolRef::Field(field), ty)?;
        let count = sources
            .len()
            .checked_mul(orders[&field] as usize)
            .and_then(|count| count.checked_add(coordinates.len()))
            .ok_or_else(|| invalid_artifact("component time witness cardinality overflows"))?;
        // Admission is bounded by the submitted proof before allocating its expansion.
        if count > proof.state_coordinates().len() {
            return Err(invalid_artifact(
                "component time witness omits Model coordinates",
            ));
        }
        for (channel, source) in sources.into_iter().enumerate() {
            for order in 0..orders[&field] {
                let column = coordinates.len();
                coordinates.push(TimeStateCoordinate::new(
                    field,
                    order,
                    channel / parts,
                    source.is_imaginary(),
                ));
                if order + 1 == orders[&field] {
                    highest.push(column);
                    rates.push(source.with_symbol(SymbolRef::Derivative(
                        field,
                        std::num::NonZeroU32::new(order + 1).expect("positive source order"),
                    )));
                } else {
                    companions.push(column);
                }
            }
        }
    }
    if coordinates != proof.state_coordinates() || operator.rows().len() != rates.len() {
        return Err(invalid_artifact(
            "component time witness differs from Model coordinate or residual order",
        ));
    }
    let n = coordinates.len();
    let witness = proof.derivative_matrix();
    if witness.dimension() != n {
        return Err(invalid_artifact(
            "component time witness matrix dimension differs",
        ));
    }
    for (row, source) in operator.rows().iter().enumerate() {
        let jacobian = source
            .constant_coordinate_jacobian(&rates)
            .map_err(|error| invalid_artifact(error.message()))?;
        for column in 0..n {
            let expected = highest
                .iter()
                .position(|&index| index == column)
                .map_or(0., |index| jacobian.coefficients()[index]);
            if witness.coefficients()[row * n + column] != expected {
                return Err(invalid_artifact(
                    "component time matrix differs from canonical derivative coefficients",
                ));
            }
        }
    }
    for (row, &state) in companions.iter().enumerate() {
        for column in 0..n {
            if witness.coefficients()[(rates.len() + row) * n + column]
                != f64::from(column == state)
            {
                return Err(invalid_artifact(
                    "component time matrix differs from companion equation",
                ));
            }
        }
    }
    Ok(())
}
