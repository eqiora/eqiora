//! One source lowering path for real and complex scalar coefficient tapes.
use super::*;
use eqiora_core::{ScalarDomain, ValueLiteral};
use num_complex::Complex64;

// This private conversion boundary prevents imaginary channels from being
// silently discarded when the caller requests a real coefficient tape.
pub(crate) trait Coefficient: Scalar + ComplexFloat<Real = f64> + From<f64> {
    fn literal(value: &ValueLiteral) -> Option<Self>;
    fn imaginary_unit() -> Option<Self>;
}

impl Coefficient for f64 {
    fn literal(value: &ValueLiteral) -> Option<Self> {
        value.real_scalar_value().map(|value| value.value())
    }
    fn imaginary_unit() -> Option<Self> {
        None
    }
}

impl Coefficient for Complex64 {
    fn literal(value: &ValueLiteral) -> Option<Self> {
        if !value.value_type().shape().is_scalar()
            || !matches!(
                value.value_type().scalar_domain(),
                ScalarDomain::Real | ScalarDomain::Complex
            )
        {
            return None;
        }
        let (real, imag) = value.component(0)?;
        Some(Self::new(real, imag))
    }
    fn imaginary_unit() -> Option<Self> {
        Some(Self::new(0.0, 1.0))
    }
}

pub(crate) fn lower<S: Coefficient>(
    program: &KernelProgram,
    expression: &ExprDag,
    root: ExprId,
    owner: RawId,
    coordinate_dimension: usize,
) -> Result<ScalarSpatialExpression<S>, Diagnostic> {
    if coordinate_dimension == 0 {
        return Err(invalid(
            owner,
            "scalar spatial lowering requires a positive coordinate dimension",
        ));
    }
    let required = required_nodes(expression, root, owner)?;
    let mut remap = vec![None; expression.nodes().len()];
    let mut instructions = Vec::new();
    let mut coordinate_dependent = false;
    let mut parameter_fields = Vec::new();
    let mut parameter_values = Vec::new();

    for (index, node) in expression.nodes().iter().enumerate() {
        if !required[index] {
            continue;
        }
        let instruction = match node {
            ExprNode::Constant(value) => {
                Instruction::Constant(S::literal(value).ok_or_else(|| {
                    invalid(
                        owner,
                        "coefficient scalar domain or shape does not match the spatial tape",
                    )
                })?)
            }
            ExprNode::Complex { real, imag } => {
                let unit = S::imaginary_unit().ok_or_else(|| {
                    invalid(
                        owner,
                        "complex construction requires a complex spatial tape",
                    )
                })?;
                let real = remapped(&remap, *real, owner)?;
                let imag = remapped(&remap, *imag, owner)?;
                let unit_index = instructions.len();
                instructions.push(Instruction::Constant(unit));
                instructions.push(Instruction::Mul(unit_index, imag));
                Instruction::Add(real, unit_index + 1)
            }
            ExprNode::Symbol(SymbolRef::Parameter(parameter)) => {
                let value = program
                    .typed_value(parameter.erase())
                    .and_then(S::literal)
                    .ok_or_else(|| {
                        invalid(
                            owner,
                            "Parameter has no matching revision-local scalar value",
                        )
                    })?;
                let parameter = parameter_fields
                    .iter()
                    .position(|existing| existing == parameter)
                    .unwrap_or_else(|| {
                        let index = parameter_fields.len();
                        parameter_fields.push(*parameter);
                        parameter_values.push(value);
                        index
                    });
                Instruction::Parameter(parameter)
            }
            ExprNode::Symbol(SymbolRef::Coordinate {
                support,
                factor,
                axis,
            }) if *axis < coordinate_dimension
                && physical_coordinate(program, *support, *factor) =>
            {
                coordinate_dependent = true;
                Instruction::Coordinate(*axis)
            }
            ExprNode::Symbol(SymbolRef::Coordinate { axis, .. }) => {
                return Err(invalid(
                    owner,
                    format!(
                        "coordinate axis {axis} is unavailable in this physical spatial dimension {coordinate_dimension}"
                    ),
                ));
            }
            ExprNode::Neg(value) => Instruction::Neg(remapped(&remap, *value, owner)?),
            ExprNode::Add(left, right) => Instruction::Add(
                remapped(&remap, *left, owner)?,
                remapped(&remap, *right, owner)?,
            ),
            ExprNode::Sub(left, right) => Instruction::Sub(
                remapped(&remap, *left, owner)?,
                remapped(&remap, *right, owner)?,
            ),
            ExprNode::Mul(left, right) => Instruction::Mul(
                remapped(&remap, *left, owner)?,
                remapped(&remap, *right, owner)?,
            ),
            ExprNode::Div(left, right) => Instruction::Div(
                remapped(&remap, *left, owner)?,
                remapped(&remap, *right, owner)?,
            ),
            ExprNode::PowI(base, exponent) => {
                Instruction::PowI(remapped(&remap, *base, owner)?, *exponent)
            }
            ExprNode::UnaryMath(UnaryMathFunction::Conj, value) => {
                Instruction::Conjugate(remapped(&remap, *value, owner)?)
            }
            ExprNode::UnaryMath(UnaryMathFunction::Sin, value) => {
                Instruction::Sin(remapped(&remap, *value, owner)?)
            }
            ExprNode::UnaryMath(UnaryMathFunction::Sqrt, value) => {
                Instruction::Sqrt(remapped(&remap, *value, owner)?)
            }
            _ => {
                return Err(invalid(
                    owner,
                    "source must use constants, Parameters, in-domain coordinates, scalar arithmetic, and supported unary mathematics",
                ));
            }
        };
        let lowered = instructions.len();
        instructions.push(instruction);
        remap[index] = Some(lowered);
    }

    Ok(ScalarSpatialExpression {
        coordinate_dimension,
        instructions,
        root: remapped(&remap, root, owner)?,
        coordinate_dependent,
        parameter_fields,
        parameter_values,
    })
}
