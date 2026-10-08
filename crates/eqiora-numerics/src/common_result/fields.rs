use super::{CommonFieldAssociation, invalid};
use eqiora_core::{Diagnostic, DimExponents, ScalarDomain};

/// Finite coefficients whose scalar-domain width is checked by the owning Field.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CommonResultFieldBlock {
    pub(super) association: CommonFieldAssociation,
    pub(super) values: Vec<f64>,
    pub(super) logical_shape: Vec<usize>,
}

impl CommonResultFieldBlock {
    pub(super) fn new(
        association: CommonFieldAssociation,
        values: Vec<f64>,
        logical_shape: Vec<usize>,
    ) -> Result<Self, Diagnostic> {
        let _count = logical_shape.iter().try_fold(1usize, |count, extent| {
            count
                .checked_mul(*extent)
                .ok_or_else(|| invalid("Result Field block shape overflows usize"))
        })?;
        if logical_shape.is_empty() || values.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "Result Field block requires a nonempty exact shape and finite coefficients",
            ));
        }
        Ok(Self {
            association,
            values,
            logical_shape,
        })
    }

    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    #[must_use]
    pub fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
    }
}

/// One exact semantic Field and its complete accepted coefficient blocks.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CommonResultField {
    pub(super) field_id: String,
    pub(super) scalar_domain: ScalarDomain,
    pub(super) dimension: DimExponents,
    pub(super) value_shape: Vec<usize>,
    pub(super) space: String,
    pub(super) blocks: Vec<CommonResultFieldBlock>,
}

impl CommonResultField {
    pub(super) fn new(
        field_id: String,
        scalar_domain: ScalarDomain,
        dimension: DimExponents,
        value_shape: Vec<usize>,
        space: impl Into<String>,
        blocks: Vec<CommonResultFieldBlock>,
    ) -> Result<Self, Diagnostic> {
        let space = space.into();
        if field_id.is_empty() || space.is_empty() || blocks.is_empty() {
            return Err(invalid(
                "Result Field requires exact identity, space, and coefficient blocks",
            ));
        }
        let width = match scalar_domain {
            ScalarDomain::Real => 1,
            ScalarDomain::Complex => 2,
            _ => {
                return Err(invalid(
                    "Result Field requires real or complex coefficients",
                ));
            }
        };
        for block in &blocks {
            let count = block
                .logical_shape
                .iter()
                .try_fold(width, |count: usize, extent| count.checked_mul(*extent))
                .ok_or_else(|| invalid("Result Field coordinate count overflows usize"))?;
            if count != block.values.len() {
                return Err(invalid(
                    "Result Field coefficient count differs from its scalar domain and logical shape",
                ));
            }
        }
        Ok(Self {
            field_id,
            scalar_domain,
            dimension,
            value_shape,
            space,
            blocks,
        })
    }

    #[must_use]
    pub fn field_id(&self) -> &str {
        &self.field_id
    }
    #[must_use]
    pub const fn dimension(&self) -> DimExponents {
        self.dimension
    }
    #[must_use]
    pub fn value_shape(&self) -> &[usize] {
        &self.value_shape
    }
    #[must_use]
    pub fn space(&self) -> &str {
        &self.space
    }
}
