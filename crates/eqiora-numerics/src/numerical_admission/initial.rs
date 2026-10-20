//! Exact Field-bound initial coefficients; physical meaning remains in the Model.
use super::*;

/// Immutable coherent-SI values for one supported exact Field association.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonInitialValues {
    shape: eqiora_core::ValueShape,
    values: Box<[(f64, f64)]>,
}

impl CommonInitialValues {
    /// Entity-major values, with row-major physical components within each entity.
    /// Real and complex inputs preserve both scalar parts; the owning Plan checks
    /// the exact Field scalar domain.
    pub fn new(
        shape: eqiora_core::ValueShape,
        values: Vec<impl Into<num_complex::Complex64>>,
    ) -> Result<Self, Diagnostic> {
        let values = values
            .into_iter()
            .map(|value| {
                let value = value.into();
                (value.re, value.im)
            })
            .collect::<Vec<_>>();
        let components = shape
            .component_count()
            .ok_or_else(|| invalid("InitialField component count overflows"))?;
        if !values.len().is_multiple_of(components)
            || values
                .iter()
                .any(|(re, im)| !re.is_finite() || !im.is_finite())
        {
            return Err(invalid(
                "InitialField requires complete finite coherent-SI components",
            ));
        }
        Ok(Self {
            shape,
            values: values.into_boxed_slice(),
        })
    }
    pub fn shape(&self) -> &eqiora_core::ValueShape {
        &self.shape
    }
    /// Row-major physical components as `(real, imaginary)` pairs.
    pub fn values(&self) -> &[(f64, f64)] {
        &self.values
    }

    pub(super) fn real_values(&self) -> Result<Vec<f64>, Diagnostic> {
        self.values
            .iter()
            .map(|&(re, im)| {
                if im != 0. {
                    return Err(invalid(
                        "real InitialField cannot contain imaginary coefficients",
                    ));
                }
                Ok(re)
            })
            .collect()
    }

    pub(super) fn vectors<const N: usize>(&self) -> Result<Vec<[f64; N]>, Diagnostic> {
        if self
            .shape
            .extents()
            .iter()
            .map(|extent| extent.get() as usize)
            .collect::<Vec<_>>()
            != [N]
        {
            return Err(invalid(
                "InitialField vector shape differs from exact Field",
            ));
        }
        Ok(self.real_values()?.as_chunks::<N>().0.to_vec())
    }
}

/// One exact Model/Field-bound initial assignment with bounded associations.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonInitialField {
    model: eqiora_artifact::ArtifactDigest,
    field: eqiora_core::Id<eqiora_core::entity::kinds::Field>,
    vertex: Option<CommonInitialValues>,
    cell: Option<CommonInitialValues>,
    finite: Option<CommonInitialValues>,
}

impl CommonInitialField {
    pub fn new(
        model: eqiora_artifact::ArtifactDigest,
        field: eqiora_core::Id<eqiora_core::entity::kinds::Field>,
        vertex: Option<CommonInitialValues>,
        cell: Option<CommonInitialValues>,
    ) -> Result<Self, Diagnostic> {
        if vertex.is_none() && cell.is_none() {
            return Err(invalid(
                "InitialField requires vertex_values or cell_values",
            ));
        }
        Ok(Self {
            model,
            field,
            vertex,
            cell,
            finite: None,
        })
    }
    /// One no-Mesh assignment in the exact Field's coherent-SI unit and basis.
    /// Components are row-major `(real, imaginary)` pairs. Plan admission checks
    /// the exact shape and scalar domain against the original Field.
    pub fn finite(
        model: eqiora_artifact::ArtifactDigest,
        field: eqiora_core::Id<eqiora_core::entity::kinds::Field>,
        shape: eqiora_core::ValueShape,
        components: Vec<(f64, f64)>,
    ) -> Result<Self, Diagnostic> {
        let value = CommonInitialValues::new(
            shape,
            components
                .into_iter()
                .map(|(re, im)| num_complex::Complex64::new(re, im))
                .collect(),
        )?;
        if value.shape().component_count() != Some(value.values().len()) {
            return Err(invalid(
                "finite InitialField requires exactly one complete physical value",
            ));
        }
        Ok(Self {
            model,
            field,
            vertex: None,
            cell: None,
            finite: Some(value),
        })
    }
    /// No-Mesh coefficients; absent for spatial assignments.
    #[must_use]
    pub fn finite_value(&self) -> Option<(&eqiora_core::ValueShape, &[(f64, f64)])> {
        self.finite
            .as_ref()
            .map(|value| (&value.shape, value.values()))
    }
    #[must_use]
    pub const fn model(&self) -> &eqiora_artifact::ArtifactDigest {
        &self.model
    }
    #[must_use]
    pub const fn field(&self) -> eqiora_core::Id<eqiora_core::entity::kinds::Field> {
        self.field
    }
    #[must_use]
    pub const fn vertex(&self) -> Option<&CommonInitialValues> {
        self.vertex.as_ref()
    }
    #[must_use]
    pub const fn cell(&self) -> Option<&CommonInitialValues> {
        self.cell.as_ref()
    }
}
