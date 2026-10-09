//! Complex-linear extension of admitted real expressions; no Fourier-mode guessing.
use super::*;
use eqiora_core::{DimExponents, ValueLiteral};
use eqiora_schema::kernel::typing::TypedResidual;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Dependence {
    Zero,
    Fixed,
    Harmonic,
}

pub(super) fn reduce(
    typed: &TypedResidual<RawId>,
    fields: &BTreeMap<RawId, Id<kinds::Field>>,
    inputs: &BTreeMap<RawId, ExprDag>,
    omega: f64,
) -> Result<ExprDag, Diagnostic> {
    use Dependence::*;
    let mut builder = ExprDagBuilder::new();
    let frequency = ValueType::scalar(
        ScalarDomain::Complex,
        DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap(),
    )
    .unwrap();
    let mut derivative = None;
    let mut mapped: Vec<(ExprId, Dependence)> = Vec::new();
    for (node, ty) in typed.expression().nodes().iter().zip(typed.node_types()) {
        if ty.value_type.scalar_domain() != ScalarDomain::Real {
            return Err(invalid(
                "harmonic source relations require real time-domain values",
            ));
        }
        let get = |id: ExprId| mapped[id.index() as usize];
        let (node, dependence) = match node {
            ExprNode::Constant(value) => (node.clone(), if value.is_zero() { Zero } else { Fixed }),
            ExprNode::Symbol(SymbolRef::Parameter(_) | SymbolRef::Coordinate { .. }) => {
                (node.clone(), Fixed)
            }
            ExprNode::Symbol(SymbolRef::Field(field)) => {
                let amplitude = *fields
                    .get(&field.erase())
                    .ok_or_else(|| invalid("unmapped harmonic Field"))?;
                (ExprNode::Symbol(SymbolRef::Field(amplitude)), Harmonic)
            }
            ExprNode::Symbol(SymbolRef::Derivative(field, order)) => {
                let amplitude = *fields
                    .get(&field.erase())
                    .ok_or_else(|| invalid("unmapped harmonic derivative"))?;
                let amplitude = builder.symbol(SymbolRef::Field(amplitude))?;
                let exponent = i32::try_from(order.get())
                    .map_err(|_| invalid("harmonic derivative order is not representable"))?;
                let frequency = match derivative {
                    Some(value) => value,
                    None => {
                        let value = builder.constant(
                            ValueLiteral::new(frequency.clone(), [(0.0, -omega)]).unwrap(),
                        )?;
                        derivative = Some(value);
                        value
                    }
                };
                let factor = builder.powi(frequency, exponent)?;
                (ExprNode::Mul(factor, amplitude), Harmonic)
            }
            ExprNode::Symbol(SymbolRef::Port(port)) => {
                let input = inputs
                    .get(&port.erase())
                    .ok_or_else(|| invalid("unmapped harmonic excitation"))?;
                let amplitude = coefficient::append(input, &mut builder)?;
                mapped.push((amplitude, Harmonic));
                continue;
            }
            ExprNode::Neg(a) => {
                let (a, d) = get(*a);
                (ExprNode::Neg(a), d)
            }
            ExprNode::Add(a, b) | ExprNode::Sub(a, b) => {
                let (a, da) = get(*a);
                let (b, db) = get(*b);
                let d = if da == Zero {
                    db
                } else if db == Zero || da == db {
                    da
                } else {
                    return Err(invalid(
                        "mixed DC and harmonic terms require a declared decomposition",
                    ));
                };
                (
                    if matches!(node, ExprNode::Add(..)) {
                        ExprNode::Add(a, b)
                    } else {
                        ExprNode::Sub(a, b)
                    },
                    d,
                )
            }
            ExprNode::Mul(a, b) => {
                let (a, da) = get(*a);
                let (b, db) = get(*b);
                if da == Harmonic && db == Harmonic {
                    return Err(invalid(
                        "nonlinear harmonic product generates additional frequencies",
                    ));
                }
                let d = if da == Harmonic || db == Harmonic {
                    Harmonic
                } else if da == Zero || db == Zero {
                    Zero
                } else {
                    Fixed
                };
                (ExprNode::Mul(a, b), d)
            }
            ExprNode::Div(a, b) => {
                let (a, da) = get(*a);
                let (b, db) = get(*b);
                if db != Fixed {
                    return Err(invalid(
                        "harmonic denominator must be a fixed nonzero coefficient",
                    ));
                }
                (ExprNode::Div(a, b), da)
            }
            ExprNode::PowI(a, exponent) => {
                let (a, d) = get(*a);
                if d == Harmonic && *exponent != 1 {
                    return Err(invalid(
                        "nonlinear harmonic power generates additional frequencies",
                    ));
                }
                (
                    ExprNode::PowI(a, *exponent),
                    if d == Harmonic { Harmonic } else { Fixed },
                )
            }
            // Conjugation of an original REAL value is identity. Conjugating its
            // complex amplitude instead would reverse the frequency and be wrong.
            ExprNode::UnaryMath(UnaryMathFunction::Conj | UnaryMathFunction::Real, a) => {
                mapped.push(get(*a));
                continue;
            }
            ExprNode::UnaryMath(op, a) => {
                let (a, d) = get(*a);
                if d == Harmonic {
                    return Err(invalid(
                        "nonlinear harmonic function generates additional frequencies",
                    ));
                }
                (ExprNode::UnaryMath(*op, a), Fixed)
            }
            ExprNode::Gradient(a)
            | ExprNode::Divergence(a)
            | ExprNode::Trace { value: a, .. }
            | ExprNode::NormalComponent { value: a, .. } => {
                let (a, d) = get(*a);
                let node = match node {
                    ExprNode::Gradient(_) => ExprNode::Gradient(a),
                    ExprNode::Divergence(_) => ExprNode::Divergence(a),
                    ExprNode::Trace { on, .. } => ExprNode::Trace { value: a, on: *on },
                    ExprNode::NormalComponent { on, .. } => {
                        ExprNode::NormalComponent { value: a, on: *on }
                    }
                    _ => unreachable!("spatial operator matched"),
                };
                (node, d)
            }
            _ => {
                return Err(invalid(
                    "harmonic LTI reduction does not admit this time-dependent, history-dependent or unsupported operator",
                ));
            }
        };
        mapped.push((builder.push(node)?, dependence));
    }
    let mut roots = Vec::new();
    for root in typed.expression().roots() {
        let (value, dependence) = mapped[root.index() as usize];
        if dependence == Fixed {
            return Err(invalid(
                "a fixed nonzero residual is not a harmonic excitation; DC needs a separate decomposition",
            ));
        }
        let ty = complex(&typed.node_type(*root).unwrap().value_type)?;
        let zero = builder.constant(
            ValueLiteral::from_real(ty, 0.0).map_err(|_| invalid("invalid harmonic zero type"))?,
        )?;
        roots.extend([value, zero]);
    }
    super::compact::compact(builder.finish(roots)?)
}
