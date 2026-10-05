use crate::{DynQuantity, ScalarDomain, ValueType};

/// A complete finite mathematical value, with domain-specific exact payloads.
///
/// Components follow the type's shape in row-major, last-axis-fastest order.
/// The type alone determines scalar domain, dimensions, shape, and frame.
#[derive(Debug, Clone, PartialEq)]
pub struct ValueLiteral {
    value_type: ValueType,
    payload: Payload,
}

#[derive(Debug, Clone, PartialEq)]
enum Payload {
    Boolean(bool),
    Enum(u32),
    Zero,
    Components(Box<[(f64, f64)]>),
    Integers(Box<[i64]>),
}

impl ValueLiteral {
    /// Construct a checked member of one exact enum type, without numeric coercion.
    pub fn enum_value(value_type: ValueType, tag: u32) -> Result<Self, InvalidValueLiteral> {
        let (Some(definition), Some(count)) =
            (value_type.enum_definition(), value_type.enum_member_count())
        else {
            return Err(InvalidValueLiteral::ScalarDomain);
        };
        if ValueType::enumeration(definition, count).ok().as_ref() != Some(&value_type) {
            return Err(InvalidValueLiteral::ScalarDomain);
        }
        if tag >= count {
            return Err(InvalidValueLiteral::NominalRange);
        }
        Ok(Self {
            value_type,
            payload: Payload::Enum(tag),
        })
    }
    /// Internal declared member selector; no integer or ordinal conversion is implied.
    pub const fn enum_tag(&self) -> Option<u32> {
        match self.payload {
            Payload::Enum(tag) => Some(tag),
            _ => None,
        }
    }

    /// Construct a logical truth value without integer or real coercion.
    #[must_use]
    pub fn boolean(value: bool) -> Self {
        Self {
            value_type: ValueType::boolean(),
            payload: Payload::Boolean(value),
        }
    }

    /// Extract only a logical truth value; numeric zero and one are not Booleans.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self.payload {
            Payload::Boolean(value) => Some(value),
            _ => None,
        }
    }

    /// Construct all components of the declared mathematical type.
    /// Signed zeros normalize to positive zero; all-zero values store no buffer.
    /// Inputs with unknown length are read at most one past the required count.
    ///
    /// # Errors
    /// Rejects incorrect cardinality, non-finite components, imaginary parts in
    /// real types, or component storage that cannot be allocated.
    pub fn new(
        value_type: ValueType,
        components: impl IntoIterator<Item = (f64, f64)>,
    ) -> Result<Self, InvalidValueLiteral> {
        if !matches!(
            value_type.scalar_domain(),
            ScalarDomain::Real | ScalarDomain::Complex
        ) {
            return Err(InvalidValueLiteral::ScalarDomain);
        }
        let count = value_type
            .shape()
            .component_count()
            .expect("checked ValueType");
        let mut input = components.into_iter();
        let (lower, upper) = input.size_hint();
        if lower > count || upper.is_some_and(|upper| upper < count) {
            return Err(InvalidValueLiteral::ComponentCount);
        }
        let mut values = Vec::new();
        for index in 0..count {
            let (real, imaginary) = input.next().ok_or(InvalidValueLiteral::ComponentCount)?;
            if !real.is_finite() || !imaginary.is_finite() {
                return Err(InvalidValueLiteral::NonFinite);
            }
            if value_type.scalar_domain() == ScalarDomain::Real && imaginary != 0.0 {
                return Err(InvalidValueLiteral::ImaginaryInReal);
            }
            let component = (normalize_zero(real), normalize_zero(imaginary));
            if !values.is_empty() || component != (0.0, 0.0) {
                let additional = index + 1 - values.len();
                values
                    .try_reserve(additional)
                    .map_err(|_| InvalidValueLiteral::Allocation)?;
                values.resize(index, (0.0, 0.0));
                values.push(component);
            }
        }
        if input.next().is_some() {
            return Err(InvalidValueLiteral::ComponentCount);
        }
        Ok(Self {
            value_type,
            payload: if values.is_empty() {
                Payload::Zero
            } else {
                Payload::Components(values.into_boxed_slice())
            },
        })
    }

    /// Embed a real scalar or contextual zero into the declared type.
    /// A shaped contextual zero requires no shape-sized allocation or traversal.
    ///
    /// # Errors
    /// Rejects non-finite values and nonzero scalar broadcasting to shaped types.
    pub fn from_real(value_type: ValueType, value: f64) -> Result<Self, InvalidValueLiteral> {
        if !matches!(
            value_type.scalar_domain(),
            ScalarDomain::Real | ScalarDomain::Complex
        ) {
            return Err(InvalidValueLiteral::ScalarDomain);
        }
        if !value.is_finite() {
            return Err(InvalidValueLiteral::NonFinite);
        }
        if value == 0.0 {
            return Ok(Self {
                value_type,
                payload: Payload::Zero,
            });
        }
        if !value_type.shape().is_scalar() {
            return Err(InvalidValueLiteral::NonzeroShape);
        }
        Self::new(value_type, [(value, 0.0)])
    }

    /// Complete mathematical type, including scalar domain and component roles.
    #[must_use]
    pub const fn value_type(&self) -> &ValueType {
        &self.value_type
    }

    /// Exact number of mathematical components, independent of storage.
    #[must_use]
    pub fn component_count(&self) -> usize {
        self.value_type
            .shape()
            .component_count()
            .expect("checked ValueType")
    }

    /// One ordered real/imaginary pair, or `None` outside the exact shape.
    #[must_use]
    pub fn component(&self, index: usize) -> Option<(f64, f64)> {
        if !matches!(
            self.value_type.scalar_domain(),
            ScalarDomain::Real | ScalarDomain::Complex
        ) {
            return None;
        }
        match &self.payload {
            Payload::Integers(_) | Payload::Boolean(_) | Payload::Enum(_) => None,
            Payload::Zero => (index < self.component_count()).then_some((0.0, 0.0)),
            Payload::Components(values) => values.get(index).copied(),
        }
    }

    /// Components in row-major, last-axis-fastest order, without materializing zero storage.
    pub fn components(
        &self,
    ) -> Option<impl ExactSizeIterator<Item = (f64, f64)> + DoubleEndedIterator + '_> {
        matches!(
            self.value_type.scalar_domain(),
            ScalarDomain::Real | ScalarDomain::Complex
        )
        .then(|| {
            (0..self.component_count())
                .map(|index| self.component(index).expect("in-range component"))
        })
    }

    /// Whether every numeric component is zero. Boolean false is not numeric zero.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        matches!(self.payload, Payload::Zero)
    }

    /// Extract a numerical quantity only for an invariant real scalar.
    #[must_use]
    pub fn real_scalar_value(&self) -> Option<DynQuantity> {
        if self.value_type.scalar_domain() == ScalarDomain::Real
            && self.value_type.shape().is_scalar()
        {
            Some(DynQuantity::new(
                self.component(0)?.0,
                self.value_type.dimension(),
            ))
        } else {
            None
        }
    }
}

fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

impl TryFrom<DynQuantity> for ValueLiteral {
    type Error = InvalidValueLiteral;
    fn try_from(value: DynQuantity) -> Result<Self, Self::Error> {
        Self::from_real(
            ValueType::scalar(ScalarDomain::Real, value.dim()).expect("checked scalar type"),
            value.value(),
        )
    }
}

/// A literal cannot initialize the requested mathematical type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidValueLiteral {
    /// The requested scalar domain, dimension or shape is not admitted.
    ScalarDomain,
    /// A count is negative or a nominal index is outside its exact bound.
    NominalRange,
    /// Exact integer arithmetic overflowed.
    IntegerOverflow,
    /// An integer quotient or remainder has a zero divisor.
    ZeroDivisor,
    /// A real value is fractional or outside the signed integer range.
    IntegerConversion,
    /// Mathematical components must be finite.
    NonFinite,
    /// Nonzero scalars cannot broadcast to a shape.
    NonzeroShape,
    /// Supplied components do not exactly fill the type's shape.
    ComponentCount,
    /// A real mathematical type cannot contain a nonzero imaginary component.
    ImaginaryInReal,
    /// Storage for supplied nonzero components cannot be allocated.
    Allocation,
}

impl core::fmt::Display for InvalidValueLiteral {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(match self {
            Self::ScalarDomain => {
                "operation requires the exact admitted scalar domain, dimension and shape"
            }
            Self::NominalRange => "count or index is outside its exact nonnegative range",
            Self::IntegerOverflow => "exact integer arithmetic overflow",
            Self::ZeroDivisor => "integer divisor must be nonzero",
            Self::IntegerConversion => {
                "real-to-integer conversion requires an in-range integral value"
            }
            Self::NonFinite => "mathematical components must be finite",
            Self::NonzeroShape => "a shaped value requires contextual zero or complete components",
            Self::ComponentCount => "component count must exactly match the mathematical type",
            Self::ImaginaryInReal => "a real mathematical type requires zero imaginary components",
            Self::Allocation => "mathematical component storage cannot be allocated",
        })
    }
}
impl std::error::Error for InvalidValueLiteral {}

mod array;
mod comparison;
mod integer;
mod projection;

#[cfg(test)]
mod tests;
