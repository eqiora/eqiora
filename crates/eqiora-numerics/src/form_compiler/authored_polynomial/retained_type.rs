//! Reconstruct complete operand types with the Model's native typing rules.
use super::*;
use eqiora_core::{
    DimExponents, FiniteBasis, Id, RawId, ScalarDomain, ValueFrame, ValueLiteral, ValueShape,
    entity::kinds,
};
use eqiora_schema::kernel::{
    FiniteBinaryOperation, UnaryMathFunction,
    pure_operator::PureOperatorDefinition,
    typing::{self, ExpressionType},
};
type Ty = ExpressionType<RawId>;

impl Context<'_> {
    pub(super) fn component_type(
        &mut self,
        value: &E,
        indices: &[u32],
        depth: usize,
    ) -> Option<Ty> {
        let ty = self.retained_type(value, depth + 1)?;
        if ty.value_type.array_rank() != 0
            || indices.len() != ty.shape().rank()
            || indices
                .iter()
                .zip(ty.shape().extents())
                .any(|(i, n)| *i >= n.get())
        {
            return None;
        }
        Some(Ty::new(
            ValueType::scalar(ty.value_type.scalar_domain(), ty.dimension()).ok()?,
            ty.support,
        ))
    }

    pub(super) fn retained_type(&mut self, value: &E, depth: usize) -> Option<Ty> {
        self.step(depth)?;
        let unary = |context: &mut Self, value| context.retained_type(value, depth + 1);
        match value {
            E::Field { ulid } | E::Parameter { ulid } | E::TimeDerivative { field_ulid: ulid } => {
                let ty = Ty::new(
                    self.symbols.get(ulid)?.clone(),
                    self.supports.get(ulid).cloned(),
                );
                if matches!(value, E::TimeDerivative { .. }) {
                    typing::time_derivative(&ty, std::num::NonZeroU32::MIN).ok()
                } else {
                    Some(ty)
                }
            }
            E::Test { field_ulid } if field_ulid == self.field => unary(
                self,
                &E::Field {
                    ulid: field_ulid.clone(),
                },
            ),
            E::Direction { name, field_ulid } if name == self.name && field_ulid == self.field => {
                unary(
                    self,
                    &E::Field {
                        ulid: field_ulid.clone(),
                    },
                )
            }
            E::Time => Some(Ty::scalar(
                DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0])?,
                None,
            )),
            E::Trace { value, on_ulid } => {
                typing::trace(&unary(self, value)?, self.domains.get(on_ulid)).ok()
            }
            E::NormalTrace { value, on_ulid } => {
                typing::normal(&unary(self, value)?, self.domains.get(on_ulid)).ok()
            }
            E::TangentialTrace { value, on_ulid } => {
                let definition =
                    PureOperatorDefinition::tangential_lift(u32::try_from(self.dimensions).ok()?)
                        .ok()?;
                let operand = unary(self, value)?;
                let instance = definition.instantiate(&[operand]).ok()?;
                typing::normal(instance.result_type(), self.domains.get(on_ulid)).ok()
            }
            E::Number { value } if value.is_finite() => {
                Some(Ty::scalar(DimExponents::DIMENSIONLESS, None))
            }
            E::Rational { dimension, .. } => {
                Some(Ty::scalar(DimExponents::from_rationals(*dimension)?, None))
            }
            E::SpatialTensor {
                complex,
                dimension,
                shape,
                values,
            } => {
                self.remaining = self.remaining.checked_sub(values.len())?;
                let ty = ValueType::shaped(
                    domain(*complex),
                    DimExponents::from_rationals(*dimension)?,
                    ValueShape::new(shape.iter().copied()).ok()?,
                    ValueFrame::SpatialCartesian,
                )
                .ok()?;
                ValueLiteral::new(ty.clone(), values.iter().copied()).ok()?;
                Some(Ty::new(ty, None))
            }
            E::LinearMap {
                source_basis,
                target_basis,
                complex,
                dimension,
                values,
            } => {
                self.remaining = self.remaining.checked_sub(values.len())?;
                let ty = ValueType::linear_map(
                    basis(source_basis)?,
                    basis(target_basis)?,
                    domain(*complex),
                    DimExponents::from_rationals(*dimension)?,
                )
                .ok()?;
                ValueLiteral::new(ty.clone(), values.iter().copied()).ok()?;
                Some(Ty::new(ty, None))
            }
            E::Neg { value } => unary(self, value),
            E::Conjugate { value } => {
                typing::unary_math(UnaryMathFunction::Conj, &unary(self, value)?).ok()
            }
            E::Complex { real, imag } => unary(self, real)?.complex(unary(self, imag)?).ok(),
            E::Add { left, right } | E::Sub { left, right } => {
                typing::additive(&unary(self, left)?, &unary(self, right)?).ok()
            }
            E::Mul { left, right } => {
                typing::multiply(&unary(self, left)?, &unary(self, right)?).ok()
            }
            E::Div { left, right } => {
                typing::divide(&unary(self, left)?, &unary(self, right)?).ok()
            }
            E::Pow { base, exponent } => typing::power(&unary(self, base)?, *exponent).ok(),
            E::Gradient { value } => typing::gradient(&unary(self, value)?).ok(),
            E::Divergence { value } => typing::divergence(&unary(self, value)?).ok(),
            E::SymmetricPart { value } => typing::symmetric_part(&unary(self, value)?).ok(),
            E::IsotropicLift { value } => typing::isotropic_lift(&unary(self, value)?).ok(),
            E::Apply { left, right } => unary(self, left)?
                .finite_binary(FiniteBinaryOperation::Apply, unary(self, right)?)
                .ok(),
            E::Component { value, indices } => self.component_type(value, indices, depth + 1),
            E::Coordinate {
                support_ulid,
                factor_ulid,
                axis,
            } => {
                let support = self.domains.get(support_ulid)?;
                let factor = Id::<kinds::Domain>::from_ulid(factor_ulid.parse().ok()?).erase();
                Ty::coordinate(&factor, *axis, Some(support)).ok()
            }
            E::Curl { value } => {
                let value = unary(self, value)?;
                let gradient = typing::gradient(&value).ok()?;
                let definition = PureOperatorDefinition::curl_from_gradient(
                    u32::try_from(self.dimensions).ok()?,
                    u16::try_from(value.shape().rank()).ok()?,
                )
                .ok()?;
                Some(
                    definition
                        .instantiate(&[gradient])
                        .ok()?
                        .result_type()
                        .clone(),
                )
            }
            E::Cross { left, right } => Some(
                PureOperatorDefinition::cross_product()
                    .ok()?
                    .instantiate(&[unary(self, left)?, unary(self, right)?])
                    .ok()?
                    .result_type()
                    .clone(),
            ),
            E::Dot { left, right } | E::Frobenius { left, right } | E::Inner { left, right } => {
                let left = unary(self, left)?;
                let right = unary(self, right)?;
                let rank = u16::try_from(left.shape().rank()).ok()?;
                if rank == 0 {
                    return typing::multiply(&left, &right).ok();
                }
                let definition = PureOperatorDefinition::contract(
                    u32::try_from(self.dimensions).ok()?,
                    rank,
                    u16::try_from(right.shape().rank()).ok()?,
                    &(0..rank).map(|i| (i, i)).collect::<Vec<_>>(),
                )
                .ok()?;
                Some(
                    definition
                        .instantiate(&[left, right])
                        .ok()?
                        .result_type()
                        .clone(),
                )
            }
            _ => None,
        }
    }
}
fn domain(complex: bool) -> ScalarDomain {
    if complex {
        ScalarDomain::Complex
    } else {
        ScalarDomain::Real
    }
}
fn basis(atoms: &[(String, u32, bool)]) -> Option<FiniteBasis> {
    let atom = |(id, extent, dual): &(String, u32, bool)| {
        let id = Id::<kinds::FiniteSpace>::from_ulid(id.parse().ok()?);
        let basis = FiniteBasis::new(id, *extent).ok()?;
        Some(if *dual { basis.dual() } else { basis })
    };
    match atoms {
        [a] => atom(a),
        [a, b] => FiniteBasis::product(atom(a)?, atom(b)?).ok(),
        _ => None,
    }
}
