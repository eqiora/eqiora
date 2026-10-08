//! Replay global weak expressions through the same value rules as original Relations.
use eqiora_compiler::{AuthoredFormExpressionV1 as E, AuthoredFormulationProjection};
use eqiora_core::{
    DimExponents, FiniteBasis, Id, RawId, ScalarDomain, ValueLiteral, ValueType, entity::kinds,
};
use eqiora_schema::kernel::typing::{self, ExpressionType};
use eqiora_schema::kernel::{FiniteBinaryOperation, FiniteUnaryOperation, KernelNode};
use eqiora_sem::KernelProgram;

type Ty = ExpressionType<RawId>;

pub(super) fn check(
    projection: &AuthoredFormulationProjection,
    program: &KernelProgram,
    mode: Id<kinds::Field>,
    residual: &ValueType,
) -> Option<()> {
    let [(_, _, _, test_dimension)] = projection.test_restrictions() else {
        return None;
    };
    let [(_, left, right)] = projection.equations() else {
        return None;
    };
    let test_dimension = dimension(test_dimension)?;
    let KernelNode::Field(field) = program.node(mode.erase())? else {
        return None;
    };
    let test = field
        .value_type()
        .clone()
        .with_dimension(test_dimension)
        .ok()?;
    let expected = ValueType::scalar(
        residual.scalar_domain(),
        residual.dimension().mul(test_dimension)?,
    )
    .ok()?;
    let mut context = Context {
        program,
        mode: mode.ulid().to_string(),
        test,
        remaining: 65536,
    };
    for value in [left, right] {
        let actual = context.expression(value, 0)?;
        if actual.value_type != expected && !matches!(value, E::Number { value: 0. }) {
            return None;
        }
    }
    Some(())
}

struct Context<'a> {
    program: &'a KernelProgram,
    mode: String,
    test: ValueType,
    remaining: usize,
}
impl Context<'_> {
    fn expression(&mut self, value: &E, depth: usize) -> Option<Ty> {
        if depth > 96 {
            return None;
        }
        self.remaining = self.remaining.checked_sub(1)?;
        let result = match value {
            E::Number { value } if value.is_finite() => {
                Ty::scalar(DimExponents::DIMENSIONLESS, None)
            }
            E::Rational {
                numerator,
                denominator,
                dimension: units,
            } => {
                eqiora_schema::kernel::pure_operator::ExactRational::from_canonical_parts(
                    *numerator,
                    *denominator,
                )
                .ok()?;
                Ty::scalar(dimension(units)?, None)
            }
            E::LinearMap {
                source_basis,
                target_basis,
                complex,
                dimension: units,
                values,
            } => {
                self.remaining = self.remaining.checked_sub(values.len())?;
                let ty = ValueType::linear_map(
                    self.basis(source_basis)?,
                    self.basis(target_basis)?,
                    if *complex {
                        ScalarDomain::Complex
                    } else {
                        ScalarDomain::Real
                    },
                    dimension(units)?,
                )
                .ok()?;
                ValueLiteral::new(ty.clone(), values.iter().copied()).ok()?;
                Ty::new(ty, None)
            }
            E::Field { ulid } => {
                let id = Id::<kinds::Field>::from_ulid(canonical_id(ulid)?);
                let KernelNode::Field(field) = self.program.node(id.erase())? else {
                    return None;
                };
                Ty::new(field.value_type().clone(), None)
            }
            E::Parameter { ulid } => {
                let id = Id::<kinds::Parameter>::from_ulid(canonical_id(ulid)?);
                let KernelNode::Parameter(parameter) = self.program.node(id.erase())? else {
                    return None;
                };
                Ty::new(parameter.value_type().clone(), None)
            }
            E::Test { field_ulid } if field_ulid == &self.mode => Ty::new(self.test.clone(), None),
            E::Neg { value } | E::Conjugate { value } => self.expression(value, depth + 1)?,
            E::Complex { real, imag } => self
                .expression(real, depth + 1)?
                .complex(self.expression(imag, depth + 1)?)
                .ok()?,
            E::Add { left, right } | E::Sub { left, right } => typing::additive(
                &self.expression(left, depth + 1)?,
                &self.expression(right, depth + 1)?,
            )
            .ok()?,
            E::Mul { left, right } => typing::multiply(
                &self.expression(left, depth + 1)?,
                &self.expression(right, depth + 1)?,
            )
            .ok()?,
            E::Div { left, right } => typing::divide(
                &self.expression(left, depth + 1)?,
                &self.expression(right, depth + 1)?,
            )
            .ok()?,
            E::Pow { base, exponent } => {
                typing::power(&self.expression(base, depth + 1)?, *exponent).ok()?
            }
            E::Apply { left, right } => self
                .expression(left, depth + 1)?
                .finite_binary(
                    FiniteBinaryOperation::Apply,
                    self.expression(right, depth + 1)?,
                )
                .ok()?,
            E::Inner { left, right } | E::Dot { left, right } => {
                let left = self.expression(left, depth + 1)?;
                let right = self.expression(right, depth + 1)?;
                if matches!(value, E::Inner { .. })
                    && left.shape().is_scalar()
                    && right.shape().is_scalar()
                {
                    typing::multiply(&left, &right).ok()?
                } else {
                    if left.value_type.coordinate_basis().is_none()
                        || right.value_type.coordinate_basis().is_none()
                    {
                        return None;
                    }
                    left.finite_unary(FiniteUnaryOperation::Adjoint)
                        .ok()?
                        .finite_binary(FiniteBinaryOperation::Pair, right)
                        .ok()?
                }
            }
            E::Component { value, indices } => {
                let value = self.expression(value, depth + 1)?;
                if indices.len() != value.shape().rank()
                    || indices
                        .iter()
                        .zip(value.shape().extents())
                        .any(|(i, n)| *i >= n.get())
                {
                    return None;
                }
                Ty::new(
                    ValueType::scalar(value.value_type.scalar_domain(), value.dimension()).ok()?,
                    None,
                )
            }
            _ => return None,
        };
        (result.value_type.array_rank() == 0
            && matches!(
                result.value_type.scalar_domain(),
                ScalarDomain::Real | ScalarDomain::Complex
            ))
        .then_some(result)
    }

    fn basis(&self, atoms: &[(String, u32, bool)]) -> Option<FiniteBasis> {
        let atomic = |(name, extent, dual): &(String, u32, bool)| {
            let id = Id::<kinds::FiniteSpace>::from_ulid(canonical_id(name)?);
            let KernelNode::FiniteSpace(space) = self.program.node(id.erase())? else {
                return None;
            };
            let basis = space.basis();
            if basis.space() != Some(id) || basis.extent() != *extent {
                return None;
            }
            Some(if *dual { basis.dual() } else { basis })
        };
        match atoms {
            [one] => atomic(one),
            [left, right] => FiniteBasis::product(atomic(left)?, atomic(right)?).ok(),
            _ => None,
        }
    }
}
fn canonical_id(text: &str) -> Option<ulid::Ulid> {
    text.parse::<ulid::Ulid>()
        .ok()
        .filter(|id| id.to_string() == text)
}
fn dimension(parts: &[(i32, i32); 7]) -> Option<DimExponents> {
    DimExponents::from_rationals(*parts).filter(|value| value.exponents() == *parts)
}
