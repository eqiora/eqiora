//! Recover a scalar's units without demanding its numerical value.
use super::*;
use eqiora_core::DimExponents;

impl ScalarInputIrBuilder {
    pub(super) fn zero_like(
        &mut self,
        value: ScalarInputValueId,
        dimensions: &mut Vec<Option<DimExponents>>,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        while dimensions.len() <= value.0.0 as usize {
            let index = dimensions.len();
            let at = |id: ValueId| {
                dimensions[id.0 as usize]
                    .ok_or_else(|| ir_builder_error("scalar derivative requires numerical units"))
            };
            let units = |dimension: Option<DimExponents>| {
                dimension.ok_or_else(|| ir_builder_error("scalar derivative dimension overflow"))
            };
            let dimension = match self.instructions[index] {
                Instruction::Constant(value) => Some(value.dim()),
                Instruction::TypedConstant(value) => Some(
                    self.typed_constants[value as usize]
                        .value_type()
                        .dimension(),
                ),
                Instruction::Read(slot) => Some(match &self.slots[slot.0 as usize] {
                    BuilderInput::Bound(slot) => slot.dimension,
                    BuilderInput::Formal(dimension) => *dimension,
                }),
                Instruction::Neg(value)
                | Instruction::Require { value, .. }
                | Instruction::Demand { value, .. }
                | Instruction::DerivativeLimit { value, .. }
                | Instruction::Add(value, _)
                | Instruction::Sub(value, _)
                | Instruction::Select {
                    then_value: value, ..
                } => dimensions[value.0 as usize],
                Instruction::Mul(a, b) => Some(units(at(a)?.mul(at(b)?))?),
                Instruction::Div(a, b) => Some(units(at(a)?.div(at(b)?))?),
                Instruction::PowI(a, exponent) => Some(units(at(a)?.pow(exponent, 1))?),
                Instruction::ComplexDiv {
                    operands: [a, _, b, _],
                    ..
                } => Some(units(at(a)?.div(at(b)?))?),
                Instruction::Math(function, value) => {
                    let ty = eqiora_schema::kernel::typing::ExpressionType::<()>::scalar(
                        at(value)?,
                        None,
                    );
                    Some(
                        eqiora_schema::kernel::typing::unary_math(function, &ty)
                            .map_err(|error| ir_builder_error(error.to_string()))?
                            .dimension(),
                    )
                }
                Instruction::CoefficientJet { shape, start, len } => {
                    let operands =
                        coefficient_jet::operands(start, len, shape, &self.instructions[..index])?;
                    let (a, b) = operands.values(at)?;
                    Some(coefficient_jet::dimension(shape, &a, &b)?)
                }
                Instruction::MapInvariant {
                    start,
                    extent,
                    output,
                } => Some(units(at(start)?.pow(output.dimension_power(extent)?, 1))?),
                _ => None,
            };
            dimensions.push(dimension);
        }
        let dimension = dimensions[value.0.0 as usize]
            .ok_or_else(|| ir_builder_error("scalar derivative zero lacks numerical units"))?;
        self.constant(eqiora_core::DynQuantity::new(0., dimension))
    }
}
