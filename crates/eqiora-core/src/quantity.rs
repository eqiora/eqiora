//! Two-layer unit system for compile-time and dynamic quantities.
//!
//! The soundness boundary is closed by three rules:
//!
//! 1. **Units inside Eqiora Language are always static.** The typeck layer
//!    resolves user-declared units to dimensions; `DynQuantity` never appears
//!    inside the language.
//! 2. **Dynamic dimensions enter only at external boundaries** (CSV import,
//!    experiment streams, foreign APIs) and are promoted into the static
//!    layer exclusively through [`DynQuantity::checked_cast`].
//! 3. **Demotion is always safe, promotion is always checked.**
//!    [`Quantity::into_dyn`] is infallible; the reverse returns `Result`.
//!    No implicit conversion exists in either direction.
//!
//! Consequence: code that passes static unit checking cannot raise a
//! dimension error at runtime, and every place where unit soundness *could*
//! break is greppable as a `checked_cast` call site.
//!
//! Type-level dimension *arithmetic* (e.g. `Velocity * Time = Length` in the
//! static layer) is deliberately deferred until `adt_const_params`
//! stabilizes; the dynamic layer already implements it. Migrating later must
//! not change the meaning of any public API.

use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::diagnostic::{Diagnostic, codes};

mod sealed {
    pub trait Sealed {}
}

/// Arithmetic scalar types admitted into quantities.
/// Ordering is a separate bound; complex quantities support equality only.
///
/// ```compile_fail
/// use eqiora_core::quantity::{Quantity, dim};
/// use num_complex::Complex64;
/// let z = Quantity::<Complex64, dim::LengthDim>::from_scalar(Complex64::new(1., 2.));
/// let _ = z < z;
/// ```
pub trait Scalar:
    Copy
    + PartialEq
    + fmt::Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + sealed::Sealed
{
}

impl sealed::Sealed for f32 {}
impl Scalar for f32 {}
impl sealed::Sealed for f64 {}
impl Scalar for f64 {}
impl sealed::Sealed for num_complex::Complex<f32> {}
impl Scalar for num_complex::Complex<f32> {}
impl sealed::Sealed for num_complex::Complex<f64> {}
impl Scalar for num_complex::Complex<f64> {}

mod exponents;
pub use exponents::DimExponents;

/// Compile-time dimension marker.
///
/// Sealed: user-declared units resolve to these through the typeck layer;
/// runtime-only dimensions stay in [`DynQuantity`].
pub trait Dimension: sealed::Sealed + 'static {
    /// The SI exponent vector of this dimension.
    const EXPONENTS: DimExponents;
}

macro_rules! define_dimension {
    ($(#[$doc:meta])* $name:ident $values:expr) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name;
        impl crate::quantity::sealed::Sealed for $name {}
        impl crate::quantity::Dimension for $name {
            const EXPONENTS: crate::quantity::DimExponents =
                crate::quantity::DimExponents::from_integers($values)
                    .expect("bounded standard dimension");
        }
    };
}

/// Standard dimension markers.
pub mod dim {
    define_dimension!(
        /// Dimensionless quantity.
        Dimensionless [0, 0, 0, 0, 0, 0, 0]
    );
    define_dimension!(
        /// Mass, kg.
        MassDim [1, 0, 0, 0, 0, 0, 0]
    );
    define_dimension!(
        /// Length, m.
        LengthDim [0, 1, 0, 0, 0, 0, 0]
    );
    define_dimension!(
        /// Time, s.
        TimeDim [0, 0, 1, 0, 0, 0, 0]
    );
    define_dimension!(
        /// Electric current, A.
        CurrentDim [0, 0, 0, 1, 0, 0, 0]
    );
    define_dimension!(
        /// Thermodynamic temperature, K.
        TemperatureDim [0, 0, 0, 0, 1, 0, 0]
    );
    define_dimension!(
        /// Amount of substance, mol.
        AmountDim [0, 0, 0, 0, 0, 1, 0]
    );
    define_dimension!(
        /// Velocity, m·s⁻¹.
        VelocityDim [0, 1, -1, 0, 0, 0, 0]
    );
    define_dimension!(
        /// Acceleration, m·s⁻².
        AccelerationDim [0, 1, -2, 0, 0, 0, 0]
    );
    define_dimension!(
        /// Frequency, s⁻¹.
        FrequencyDim [0, 0, -1, 0, 0, 0, 0]
    );
    define_dimension!(
        /// Force, kg·m·s⁻² (N).
        ForceDim [1, 1, -2, 0, 0, 0, 0]
    );
    define_dimension!(
        /// Pressure, kg·m⁻¹·s⁻² (Pa).
        PressureDim [1, -1, -2, 0, 0, 0, 0]
    );
    define_dimension!(
        /// Energy, kg·m²·s⁻² (J).
        EnergyDim [1, 2, -2, 0, 0, 0, 0]
    );
    define_dimension!(
        /// Power, kg·m²·s⁻³ (W).
        PowerDim [1, 2, -3, 0, 0, 0, 0]
    );
    define_dimension!(
        /// Density, kg·m⁻³.
        DensityDim [1, -3, 0, 0, 0, 0, 0]
    );
    define_dimension!(
        /// Dynamic viscosity, Pa·s.
        DynamicViscosityDim [1, -1, -1, 0, 0, 0, 0]
    );
    define_dimension!(
        /// Thermal conductivity, W·m⁻¹·K⁻¹.
        ThermalConductivityDim [1, 1, -3, 0, -1, 0, 0]
    );
}

/// Statically dimensioned quantity — the *only* representation of physical
/// values inside the platform (rule 1).
pub struct Quantity<S: Scalar, D: Dimension> {
    value: S,
    _dim: PhantomData<fn() -> D>,
}

impl<S: Scalar, D: Dimension> Quantity<S, D> {
    /// Wrap an explicitly typed scalar. Prefer [`Quantity::new`] for the
    /// default `f64` representation. External data goes through
    /// [`DynQuantity::checked_cast`] instead.
    #[must_use]
    pub const fn from_scalar(value: S) -> Self {
        Self {
            value,
            _dim: PhantomData,
        }
    }

    /// The raw scalar value in SI base units.
    #[must_use]
    pub fn value(self) -> S {
        self.value
    }
}

impl<D: Dimension> Quantity<f64, D> {
    /// Wrap an `f64` literal or trusted numerical value in a static
    /// dimension. This is intentionally the unqualified constructor so
    /// aliases such as `Velocity::new(3.0)` select the default scalar
    /// without an inference annotation.
    #[must_use]
    pub const fn new(value: f64) -> Self {
        Self::from_scalar(value)
    }

    /// Demote into the dynamic layer. Always safe (rule 3).
    #[must_use]
    pub fn into_dyn(self) -> DynQuantity {
        DynQuantity::new(self.value, D::EXPONENTS)
    }
}

impl<D: Dimension> Quantity<f32, D> {
    /// Wrap an `f32` literal or trusted numerical value.
    #[must_use]
    pub const fn new_f32(value: f32) -> Self {
        Self::from_scalar(value)
    }

    /// Demote into the dynamic layer (widening to `f64`). Always safe.
    #[must_use]
    pub fn into_dyn(self) -> DynQuantity {
        DynQuantity::new(f64::from(self.value), D::EXPONENTS)
    }
}

impl<D: Dimension> From<Quantity<f64, D>> for DynQuantity {
    fn from(q: Quantity<f64, D>) -> Self {
        q.into_dyn()
    }
}

// Manual impls: derives would add bounds on `D`.
impl<S: Scalar, D: Dimension> Clone for Quantity<S, D> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: Scalar, D: Dimension> Copy for Quantity<S, D> {}
impl<S: Scalar, D: Dimension> PartialEq for Quantity<S, D> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}
impl<S: Scalar + PartialOrd, D: Dimension> PartialOrd for Quantity<S, D> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}
impl<S: Scalar, D: Dimension> fmt::Debug for Quantity<S, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} [{}]", self.value, D::EXPONENTS)
    }
}

impl<S: Scalar, D: Dimension> Add for Quantity<S, D> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_scalar(self.value + rhs.value)
    }
}
impl<S: Scalar, D: Dimension> Sub for Quantity<S, D> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_scalar(self.value - rhs.value)
    }
}
impl<S: Scalar, D: Dimension> Neg for Quantity<S, D> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::from_scalar(-self.value)
    }
}
/// Scaling by a bare scalar preserves the dimension.
impl<S: Scalar, D: Dimension> Mul<S> for Quantity<S, D> {
    type Output = Self;
    fn mul(self, rhs: S) -> Self {
        Self::from_scalar(self.value * rhs)
    }
}
impl<S: Scalar, D: Dimension> Div<S> for Quantity<S, D> {
    type Output = Self;
    fn div(self, rhs: S) -> Self {
        Self::from_scalar(self.value / rhs)
    }
}
impl<D: Dimension> Mul<Quantity<f64, D>> for f64 {
    type Output = Quantity<f64, D>;
    fn mul(self, rhs: Quantity<f64, D>) -> Quantity<f64, D> {
        Quantity::from_scalar(self * rhs.value)
    }
}
impl<D: Dimension> Mul<Quantity<f32, D>> for f32 {
    type Output = Quantity<f32, D>;
    fn mul(self, rhs: Quantity<f32, D>) -> Quantity<f32, D> {
        Quantity::from_scalar(self * rhs.value)
    }
}

/// Convenient aliases for common quantities (`f64` by default).
pub mod aliases {
    use super::{Quantity, dim};

    /// Dimensionless quantity.
    pub type Dimensionless<S = f64> = Quantity<S, dim::Dimensionless>;
    /// Mass, kg.
    pub type Mass<S = f64> = Quantity<S, dim::MassDim>;
    /// Length, m.
    pub type Length<S = f64> = Quantity<S, dim::LengthDim>;
    /// Time, s.
    pub type Time<S = f64> = Quantity<S, dim::TimeDim>;
    /// Temperature, K.
    pub type Temperature<S = f64> = Quantity<S, dim::TemperatureDim>;
    /// Velocity, m/s.
    pub type Velocity<S = f64> = Quantity<S, dim::VelocityDim>;
    /// Force, N.
    pub type Force<S = f64> = Quantity<S, dim::ForceDim>;
    /// Pressure, Pa.
    pub type Pressure<S = f64> = Quantity<S, dim::PressureDim>;
    /// Energy, J.
    pub type Energy<S = f64> = Quantity<S, dim::EnergyDim>;
    /// Power, W.
    pub type Power<S = f64> = Quantity<S, dim::PowerDim>;
    /// Density, kg/m³.
    pub type Density<S = f64> = Quantity<S, dim::DensityDim>;
}

/// Runtime-dimensioned quantity — exists only at external-data boundaries
/// (rule 2). Never appears inside Eqiora Language semantics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynQuantity {
    value: f64,
    dim: DimExponents,
}

impl DynQuantity {
    /// Wrap a raw value with runtime dimension exponents.
    #[must_use]
    pub const fn new(value: f64, dim: DimExponents) -> Self {
        Self { value, dim }
    }

    /// The raw value in SI base units.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// The runtime dimension exponents.
    #[must_use]
    pub const fn dim(&self) -> DimExponents {
        self.dim
    }

    /// **The only promotion path into the static layer** (rule 2).
    ///
    /// # Errors
    /// Returns a `EQ0401` diagnostic when the runtime dimension does not
    /// match `D`.
    pub fn checked_cast<D: Dimension>(self) -> Result<Quantity<f64, D>, Diagnostic> {
        if self.dim == D::EXPONENTS {
            Ok(Quantity::new(self.value))
        } else {
            Err(Diagnostic::error(
                codes::DIMENSION_MISMATCH,
                format!(
                    "dimension mismatch: expected [{}], found [{}]",
                    D::EXPONENTS,
                    self.dim
                ),
            ))
        }
    }

    /// Addition requires equal dimensions.
    ///
    /// # Errors
    /// Returns a `EQ0401` diagnostic on a dimension mismatch.
    pub fn try_add(self, rhs: Self) -> Result<Self, Diagnostic> {
        if self.dim == rhs.dim {
            Ok(Self::new(self.value + rhs.value, self.dim))
        } else {
            Err(Diagnostic::error(
                codes::DIMENSION_MISMATCH,
                format!(
                    "cannot add [{}] to [{}]: dimensions differ",
                    rhs.dim, self.dim
                ),
            ))
        }
    }

    /// Subtraction requires equal dimensions.
    ///
    /// # Errors
    /// Returns a `EQ0401` diagnostic on a dimension mismatch.
    pub fn try_sub(self, rhs: Self) -> Result<Self, Diagnostic> {
        self.try_add(Self::new(-rhs.value, rhs.dim))
    }

    /// Multiply quantities, rejecting dimension-exponent overflow.
    pub fn try_mul(self, rhs: Self) -> Result<Self, Diagnostic> {
        let dimension = self.dim.mul(rhs.dim).ok_or_else(|| {
            Diagnostic::error(
                codes::DIMENSION_MISMATCH,
                "dimension product exceeds exponent bounds",
            )
        })?;
        Ok(Self::new(self.value * rhs.value, dimension))
    }

    /// Divide quantities, rejecting dimension-exponent overflow.
    pub fn try_div(self, rhs: Self) -> Result<Self, Diagnostic> {
        let dimension = self.dim.div(rhs.dim).ok_or_else(|| {
            Diagnostic::error(
                codes::DIMENSION_MISMATCH,
                "dimension quotient exceeds exponent bounds",
            )
        })?;
        Ok(Self::new(self.value / rhs.value, dimension))
    }
}
impl Mul<f64> for DynQuantity {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.value * rhs, self.dim)
    }
}
impl Neg for DynQuantity {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.value, self.dim)
    }
}

impl fmt::Display for DynQuantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}]", self.value, self.dim)
    }
}
