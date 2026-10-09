//! Reauthenticate request coefficients against live original Parameters and supports.
use super::*;
use eqiora_compiler::AuthoredFormExpressionV1 as E;
use eqiora_core::{DimExponents, DynQuantity};
use eqiora_schema::kernel::typing::{ExpressionType, RootContract, SpatialSupport, TypedResidual};

pub(super) struct Coefficient {
    pub(super) expression: ExprDag,
    pub(super) value_type: ValueType,
}

pub(super) fn compile(
    program: &KernelProgram,
    expression: &E,
    support: Option<SpatialSupport<RawId>>,
) -> Result<Coefficient, Diagnostic> {
    let mut builder = ExprDagBuilder::new();
    let root = lower(program, expression, &mut builder, 0)?;
    let expression = builder.finish([root])?;
    let typed = TypedResidual::infer(
        expression.clone(),
        support.clone(),
        RootContract::ValueRoots,
        |symbol| {
            let SymbolRef::Coordinate {
                support: domain,
                factor,
                axis,
            } = symbol
            else {
                return Err(invalid(
                    "harmonic coefficient retains an unsupported dependency",
                ));
            };
            let actual = program
                .spatial_support(domain)
                .ok_or_else(|| invalid("harmonic coefficient has a foreign coordinate support"))?;
            if Some(actual) != support.as_ref() {
                return Err(invalid(
                    "harmonic coefficient coordinate differs from its original input support",
                ));
            }
            ExpressionType::coordinate(&factor.erase(), axis, Some(actual)).map_err(|_| {
                invalid("harmonic coefficient has an invalid coordinate factor or axis")
            })
        },
    )
    .map_err(|_| {
        invalid("harmonic coefficient has incompatible dimensions, scalar domains or exact support")
    })?;
    Ok(Coefficient {
        expression,
        value_type: typed.node_type(root).unwrap().value_type.clone(),
    })
}

fn lower(
    program: &KernelProgram,
    expression: &E,
    builder: &mut ExprDagBuilder,
    depth: usize,
) -> Result<ExprId, Diagnostic> {
    if depth > 128 {
        return Err(invalid(
            "harmonic coefficient exceeds the expression depth limit",
        ));
    }
    let node = match expression {
        E::Number { value } => {
            return builder.constant(DynQuantity::new(*value, DimExponents::DIMENSIONLESS));
        }
        E::Rational {
            numerator,
            denominator,
            dimension,
        } => {
            eqiora_schema::kernel::pure_operator::ExactRational::from_canonical_parts(
                *numerator,
                *denominator,
            )
            .map_err(|_| invalid("harmonic coefficient has a noncanonical rational"))?;
            let dimension = DimExponents::from_rationals(*dimension)
                .filter(|value| value.exponents() == *dimension)
                .ok_or_else(|| invalid("harmonic coefficient has noncanonical dimensions"))?;
            return builder.constant(DynQuantity::new(
                *numerator as f64 / *denominator as f64,
                dimension,
            ));
        }
        E::Parameter { ulid } => {
            let parameter = Id::<kinds::Parameter>::from_ulid(id(ulid)?);
            let value = program
                .typed_value(parameter.erase())
                .ok_or_else(|| invalid("harmonic coefficient is not a live Parameter"))?;
            return builder.constant(value.clone());
        }
        E::Coordinate {
            support_ulid,
            factor_ulid,
            axis,
        } => ExprNode::Symbol(SymbolRef::Coordinate {
            support: Id::from_ulid(id(support_ulid)?),
            factor: Id::from_ulid(id(factor_ulid)?),
            axis: *axis,
        }),
        E::Complex { real, imag } => ExprNode::Complex {
            real: lower(program, real, builder, depth + 1)?,
            imag: lower(program, imag, builder, depth + 1)?,
        },
        E::Neg { value } => ExprNode::Neg(lower(program, value, builder, depth + 1)?),
        E::Conjugate { value } => ExprNode::UnaryMath(
            UnaryMathFunction::Conj,
            lower(program, value, builder, depth + 1)?,
        ),
        E::Sin { value } => ExprNode::UnaryMath(
            UnaryMathFunction::Sin,
            lower(program, value, builder, depth + 1)?,
        ),
        E::Pow { base, exponent } => {
            ExprNode::PowI(lower(program, base, builder, depth + 1)?, *exponent)
        }
        E::Add { left, right }
        | E::Sub { left, right }
        | E::Mul { left, right }
        | E::Div { left, right } => {
            let left = lower(program, left, builder, depth + 1)?;
            let right = lower(program, right, builder, depth + 1)?;
            match expression {
                E::Add { .. } => ExprNode::Add(left, right),
                E::Sub { .. } => ExprNode::Sub(left, right),
                E::Mul { .. } => ExprNode::Mul(left, right),
                _ => ExprNode::Div(left, right),
            }
        }
        _ => {
            return Err(invalid(
                "harmonic excitation requires closed Parameters and fixed coordinates",
            ));
        }
    };
    builder.push(node)
}

pub(super) fn append(
    expression: &ExprDag,
    builder: &mut ExprDagBuilder,
) -> Result<ExprId, Diagnostic> {
    let mut mapped = Vec::new();
    for node in expression.nodes() {
        let get = |id: ExprId| mapped[id.index() as usize];
        let node = match node {
            ExprNode::Constant(_) | ExprNode::Symbol(SymbolRef::Coordinate { .. }) => node.clone(),
            ExprNode::Neg(a) => ExprNode::Neg(get(*a)),
            ExprNode::UnaryMath(op, a) => ExprNode::UnaryMath(*op, get(*a)),
            ExprNode::PowI(a, n) => ExprNode::PowI(get(*a), *n),
            ExprNode::Complex { real, imag } => ExprNode::Complex {
                real: get(*real),
                imag: get(*imag),
            },
            ExprNode::Add(a, b) => ExprNode::Add(get(*a), get(*b)),
            ExprNode::Sub(a, b) => ExprNode::Sub(get(*a), get(*b)),
            ExprNode::Mul(a, b) => ExprNode::Mul(get(*a), get(*b)),
            ExprNode::Div(a, b) => ExprNode::Div(get(*a), get(*b)),
            _ => return Err(invalid("unsupported harmonic coefficient DAG")),
        };
        mapped.push(builder.push(node)?);
    }
    Ok(mapped[expression.roots()[0].index() as usize])
}
