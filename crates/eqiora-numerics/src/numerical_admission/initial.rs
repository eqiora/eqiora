//! Exact Field-bound initial coefficients; physical meaning remains in the Model.
use super::*;

/// Immutable coherent-SI values for one supported exact Field association.
#[derive(Debug, Clone, PartialEq)]
pub enum CommonInitialValues {
    Scalar(Box<[f64]>),
    Vector2(Box<[[f64; 2]]>),
}

/// One exact Model/Field-bound initial assignment with bounded associations.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonInitialField {
    model: eqiora_artifact::ArtifactDigest,
    field: eqiora_core::Id<eqiora_core::entity::kinds::Field>,
    vertex: Option<CommonInitialValues>,
    cell: Option<CommonInitialValues>,
    finite: Option<FiniteInitialValue>,
}

#[derive(Debug, Clone, PartialEq)]
struct FiniteInitialValue {
    shape: eqiora_core::ValueShape,
    components: Box<[(f64, f64)]>,
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
        let finite = |values: &CommonInitialValues| match values {
            CommonInitialValues::Scalar(values) => values.iter().all(|value| value.is_finite()),
            CommonInitialValues::Vector2(values) => {
                values.iter().flatten().all(|value| value.is_finite())
            }
        };
        if vertex.as_ref().is_some_and(|values| !finite(values))
            || cell.as_ref().is_some_and(|values| !finite(values))
        {
            return Err(invalid(
                "InitialField values must be finite coherent-SI numbers",
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
        if shape.component_count() != Some(components.len())
            || components
                .iter()
                .any(|(re, im)| !re.is_finite() || !im.is_finite())
        {
            return Err(invalid(
                "finite InitialField requires its complete shape and finite coherent-SI components",
            ));
        }
        Ok(Self {
            model,
            field,
            vertex: None,
            cell: None,
            finite: Some(FiniteInitialValue {
                shape,
                components: components.into_boxed_slice(),
            }),
        })
    }
    /// No-Mesh coefficients; absent for spatial assignments.
    #[must_use]
    pub fn finite_value(&self) -> Option<(&eqiora_core::ValueShape, &[(f64, f64)])> {
        self.finite
            .as_ref()
            .map(|value| (&value.shape, value.components.as_ref()))
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
