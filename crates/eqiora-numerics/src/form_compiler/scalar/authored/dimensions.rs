//! Check physical dimensions and coordinate support before numerical cancellation.
use eqiora_compiler::{AuthoredFormExpressionV1 as E, AuthoredFormulationProjection};
use eqiora_core::{DimExponents as D, Id, RawId, entity::kinds};
use eqiora_schema::kernel::{KernelNode, typing::TypedResidual};
use eqiora_sem::KernelProgram;

pub(super) fn check(
    projection: &AuthoredFormulationProjection,
    program: &KernelProgram,
    derived: &super::DerivedScalarGalerkinForm,
    typed: &TypedResidual<RawId>,
) -> Option<()> {
    check_pairing(
        projection,
        program,
        derived.domain,
        derived.dimension,
        &derived
            .boundary_roles
            .iter()
            .map(|boundary| boundary.domain)
            .collect::<Vec<_>>(),
        typed,
    )
}

pub(in crate::form_compiler) fn check_pairing(
    projection: &AuthoredFormulationProjection,
    program: &KernelProgram,
    domain: RawId,
    dimensions: usize,
    boundaries: &[RawId],
    typed: &TypedResidual<RawId>,
) -> Option<()> {
    let [(name, field, _, units)] = projection.test_restrictions() else {
        return None;
    };
    let [(_, left, right)] = projection.equations() else {
        return None;
    };
    let test = dimension(units)?;
    let measure = length().pow(i32::try_from(dimensions).ok()?, 1)?;
    let expected = typed
        .node_type(*typed.expression().roots().first()?)?
        .dimension()
        .mul(test)?
        .mul(measure)?;
    let mut context = Context {
        program,
        domain,
        dimensions,
        boundaries,
        name,
        field,
        test,
        remaining: 65536,
        integration_domain: None,
    };
    for value in [left, right] {
        let actual = context.expression(value, 0)?;
        if !zero(value) && actual != expected {
            return None;
        }
    }
    Some(())
}

struct Context<'a> {
    program: &'a KernelProgram,
    domain: RawId,
    dimensions: usize,
    boundaries: &'a [RawId],
    name: &'a str,
    field: &'a str,
    test: D,
    remaining: usize,
    integration_domain: Option<String>,
}
impl Context<'_> {
    fn expression(&mut self, value: &E, depth: usize) -> Option<D> {
        if depth > 96 {
            return None;
        }
        self.remaining = self.remaining.checked_sub(1)?;
        match value {
            E::Number { value } if value.is_finite() => Some(D::DIMENSIONLESS),
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
                dimension(units)
            }
            E::Field { ulid } => {
                let id = Id::<kinds::Field>::from_ulid(ulid.parse().ok()?);
                let KernelNode::Field(field) = self.program.node(id.erase())? else {
                    return None;
                };
                Some(field.dimension())
            }
            E::Parameter { ulid } => {
                let id = Id::<kinds::Parameter>::from_ulid(ulid.parse().ok()?);
                let KernelNode::Parameter(parameter) = self.program.node(id.erase())? else {
                    return None;
                };
                Some(parameter.value_type().dimension())
            }
            E::Test { field_ulid } if field_ulid == self.field => Some(self.test),
            E::Direction { name, field_ulid } if name == self.name && field_ulid == self.field => {
                Some(self.test)
            }
            E::Coordinate {
                support_ulid,
                factor_ulid,
                axis,
            } if self.integration_domain.as_deref() == Some(support_ulid.as_str())
                && factor_ulid == &self.domain.ulid().to_string()
                && *axis < self.dimensions =>
            {
                Some(length())
            }
            E::Neg { value }
            | E::Conjugate { value }
            | E::Trace { value }
            | E::Component { value, .. }
            | E::SymmetricPart { value }
            | E::Variation { value, .. } => self.expression(value, depth + 1),
            E::Gradient { value } | E::Divergence { value } | E::Curl { value } => {
                self.expression(value, depth + 1)?.div(length())
            }
            E::Add { left, right } | E::Sub { left, right } => {
                let l = self.expression(left, depth + 1)?;
                let r = self.expression(right, depth + 1)?;
                if zero(left) {
                    Some(r)
                } else if zero(right) || l == r {
                    Some(l)
                } else {
                    None
                }
            }
            E::Complex { real, imag } => {
                let real = self.expression(real, depth + 1)?;
                (real == self.expression(imag, depth + 1)?).then_some(real)
            }
            E::Mul { left, right }
            | E::Dot { left, right }
            | E::Inner { left, right }
            | E::Frobenius { left, right } => self
                .expression(left, depth + 1)?
                .mul(self.expression(right, depth + 1)?),
            E::Div { left, right } => self
                .expression(left, depth + 1)?
                .div(self.expression(right, depth + 1)?),
            E::Pow { base, exponent } => self.expression(base, depth + 1)?.pow(*exponent, 1),
            E::Sin { value } => {
                (self.expression(value, depth + 1)? == D::DIMENSIONLESS).then_some(D::DIMENSIONLESS)
            }
            E::Integrate {
                domain_ulid,
                integrand,
            } => {
                if self.integration_domain.is_some() {
                    return None;
                }
                let dimension = if domain_ulid == &self.domain.ulid().to_string() {
                    self.dimensions
                } else if self
                    .boundaries
                    .iter()
                    .any(|boundary| domain_ulid == &boundary.ulid().to_string())
                {
                    self.dimensions.checked_sub(1)?
                } else {
                    return None;
                };
                self.integration_domain = Some(domain_ulid.clone());
                let integrand = self.expression(integrand, depth + 1);
                self.integration_domain = None;
                integrand?.mul(length().pow(i32::try_from(dimension).ok()?, 1)?)
            }
            _ => None,
        }
    }
}
fn length() -> D {
    D::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("SI length")
}
fn zero(value: &E) -> bool {
    matches!(value, E::Number { value: 0. })
}
fn dimension(parts: &[(i32, i32); 7]) -> Option<D> {
    D::from_rationals(*parts).filter(|value| value.exponents() == *parts)
}
