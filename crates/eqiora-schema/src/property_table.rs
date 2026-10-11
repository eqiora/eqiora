//! Verified resolved-array tables compiled to the shared exact calculus.
use crate::resolved_array::ResolvedF64Array;
fn invalid_artifact(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(eqiora_core::diagnostic::codes::INVALID_ARTIFACT, message)
}
use crate::kernel::{
    property_table::{MAX_TABLE_POINTS, exact_binary64, linear_table},
    pure_operator::{ExactRational, PureOperatorDefinition},
};
use eqiora_core::{Diagnostic, DimExponents};

/// Complete algorithm, layout and data-handling meaning of an admitted table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealTableProfile {
    /// Row-major binary64 `[N,2]`: axis then value in the specified SI dimensions.
    /// No preprocessing, missing values or extrapolation. Exact linear secants;
    /// values use the closed interval; first derivatives admit only open
    /// segment interiors, rejecting every knot and endpoint. The shared exact-rational representation bounds data
    /// and arithmetic. Evaluation uses the ordered calculus, not provider policy.
    PiecewiseAffineOpenIntervalsV1,
}

/// Accepted table metadata; the containing property release owns persistence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptedRealTable {
    array: ResolvedF64Array,
    axis_dimension: DimExponents,
    value_dimension: DimExponents,
    profile: RealTableProfile,
    validity: [ExactRational; 2],
    value: PureOperatorDefinition,
    derivative: PureOperatorDefinition,
}
impl AcceptedRealTable {
    /// Derive the table calculus from the retained immutable array.
    ///
    /// # Errors
    /// Returns `EQ0901` for invalid layout, axes or unsupported exact arithmetic.
    pub fn from_array(
        array: ResolvedF64Array,
        axis_dimension: DimExponents,
        value_dimension: DimExponents,
        profile: RealTableProfile,
        validity: [ExactRational; 2],
    ) -> Result<Self, Diagnostic> {
        let [rows, 2] = array.shape() else {
            return Err(invalid_artifact(
                "property table requires shape [N,2] with axis then value columns",
            ));
        };
        if !(2..=MAX_TABLE_POINTS as u64).contains(rows) {
            return Err(invalid_artifact(
                "property table row count is outside its exact calculus bounds",
            ));
        }
        let values = array.values();
        let mut axis = Vec::with_capacity(*rows as usize);
        let mut samples = Vec::with_capacity(*rows as usize);
        for pair in values.as_chunks::<2>().0 {
            axis.push(exact_binary64(pair[0]).map_err(|error| {
                invalid_artifact(format!(
                    "property table axis cannot be represented exactly: {error:?}"
                ))
            })?);
            samples.push(exact_binary64(pair[1]).map_err(|error| {
                invalid_artifact(format!(
                    "property table sample cannot be represented exactly: {error:?}"
                ))
            })?);
        }
        let compile = |derivative| {
            linear_table(
                &axis,
                &samples,
                axis_dimension,
                value_dimension,
                derivative,
                validity,
            )
            .map_err(|error| invalid_artifact(format!("invalid exact property table: {error:?}")))
        };
        // Match the profile explicitly so extending it cannot reuse this algorithm accidentally.
        let (value, derivative) = match profile {
            RealTableProfile::PiecewiseAffineOpenIntervalsV1 => (compile(false)?, compile(true)?),
        };
        Ok(Self {
            array,
            axis_dimension,
            value_dimension,
            profile,
            validity,
            value,
            derivative,
        })
    }
    /// Declared closed value interval; first derivatives exclude both boundaries.
    #[must_use]
    pub const fn validity(&self) -> [ExactRational; 2] {
        self.validity
    }

    /// Retained verified array closure used to generate this table.
    #[must_use]
    pub const fn array(&self) -> &ResolvedF64Array {
        &self.array
    }
    /// Existing domain-separated resolved-array identity.
    ///
    /// # Errors
    /// Returns `EQ0901` if canonical serialization fails.
    pub fn digest(&self) -> Result<[u8; 32], Diagnostic> {
        self.array.digest()
    }
    /// Exact independent-axis SI dimension.
    #[must_use]
    pub const fn axis_dimension(&self) -> DimExponents {
        self.axis_dimension
    }
    /// Exact sample SI dimension.
    #[must_use]
    pub const fn value_dimension(&self) -> DimExponents {
        self.value_dimension
    }
    /// Complete selected data and interpolation policy.
    #[must_use]
    pub const fn profile(&self) -> RealTableProfile {
        self.profile
    }
    /// Ordered shared calculus for property values.
    #[must_use]
    pub const fn value(&self) -> &PureOperatorDefinition {
        &self.value
    }
    /// Ordered shared calculus for first derivatives, including validity guards.
    #[must_use]
    pub const fn derivative(&self) -> &PureOperatorDefinition {
        &self.derivative
    }
}
