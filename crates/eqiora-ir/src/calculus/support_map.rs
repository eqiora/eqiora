use std::fmt;

use eqiora_schema::kernel::typing::{SpatialSupport, TypedResidual};
use eqiora_schema::kernel::{ExprId, ExprNode};

use super::{CalculusError, expr_index};

/// Semantic intent carried by the first support-map vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportMapIntent {
    /// Restrict a volume-supported value to its exact boundary.
    TraceRestriction,
}

/// Orientation contract of a semantic support map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportMapOrientation {
    /// Target boundary uses the exact parent's outward orientation.
    ParentOutward,
}

/// Pairing contract independent of numerical transfer weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportMapPairing {
    /// Pointwise value restriction; no dual transfer is implied.
    PointwiseValue,
}

/// Typed semantic source-to-target support map.
///
/// Mesh entities, basis spaces, interpolation weights, mortar rules, and
/// quotient matrices are intentionally unrepresentable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupportMap<I> {
    source: SpatialSupport<I>,
    target: SpatialSupport<I>,
    orientation: SupportMapOrientation,
    pairing: SupportMapPairing,
    intent: SupportMapIntent,
}

impl<I: Clone + Eq> SupportMap<I> {
    /// Prove one exact parent-volume to boundary trace restriction.
    ///
    /// # Errors
    /// Rejects wrong support kinds, dimensions, or parent identity.
    pub fn trace(
        source: SpatialSupport<I>,
        target: SpatialSupport<I>,
    ) -> Result<Self, SupportMapViolation> {
        let SpatialSupport::Volume {
            domain: source_domain,
            dimensions: source_dimensions,
        } = &source
        else {
            return Err(SupportMapViolation::SourceMustBeVolume);
        };
        let SpatialSupport::Boundary {
            parent,
            dimensions: target_dimensions,
            ..
        } = &target
        else {
            return Err(SupportMapViolation::TargetMustBeBoundary);
        };
        if source_dimensions != target_dimensions {
            return Err(SupportMapViolation::DimensionMismatch);
        }
        if source_domain != parent {
            return Err(SupportMapViolation::ParentMismatch);
        }
        Ok(Self {
            source,
            target,
            orientation: SupportMapOrientation::ParentOutward,
            pairing: SupportMapPairing::PointwiseValue,
            intent: SupportMapIntent::TraceRestriction,
        })
    }

    /// Exact source support.
    #[must_use]
    pub const fn source(&self) -> &SpatialSupport<I> {
        &self.source
    }

    /// Exact target support.
    #[must_use]
    pub const fn target(&self) -> &SpatialSupport<I> {
        &self.target
    }

    /// Semantic orientation.
    #[must_use]
    pub const fn orientation(&self) -> SupportMapOrientation {
        self.orientation
    }

    /// Semantic pairing.
    #[must_use]
    pub const fn pairing(&self) -> SupportMapPairing {
        self.pairing
    }

    /// Closed semantic intent.
    #[must_use]
    pub const fn intent(&self) -> SupportMapIntent {
        self.intent
    }

    /// Derive the support map for one canonical `Trace` application in a
    /// typed residual.
    pub fn classify_trace(
        residual: &TypedResidual<I>,
        value: ExprId,
    ) -> Result<Option<Self>, CalculusError> {
        let Some(ExprNode::Trace { value: operand, .. }) = residual.expression().node(value) else {
            return Ok(None);
        };
        let source = residual.node_types()[expr_index(*operand, residual.node_types().len())?]
            .support
            .clone()
            .ok_or(CalculusError::SupportMap(
                SupportMapViolation::SourceMustBeVolume,
            ))?;
        let target = residual.node_types()[expr_index(value, residual.node_types().len())?]
            .support
            .clone()
            .ok_or(CalculusError::SupportMap(
                SupportMapViolation::TargetMustBeBoundary,
            ))?;
        Self::trace(source, target)
            .map(Some)
            .map_err(CalculusError::SupportMap)
    }
}

/// Closed semantic support-map failure set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportMapViolation {
    /// Source is not a volume.
    SourceMustBeVolume,
    /// Target is not a boundary.
    TargetMustBeBoundary,
    /// Ambient dimensions differ.
    DimensionMismatch,
    /// Boundary parent is not the exact source identity.
    ParentMismatch,
}

impl fmt::Display for SupportMapViolation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceMustBeVolume => formatter.write_str("support-map source must be a volume"),
            Self::TargetMustBeBoundary => {
                formatter.write_str("support-map target must be a boundary")
            }
            Self::DimensionMismatch => {
                formatter.write_str("support-map source and target dimensions differ")
            }
            Self::ParentMismatch => formatter
                .write_str("support-map target boundary does not belong to the exact source"),
        }
    }
}

#[cfg(test)]
mod tests {
    use eqiora_core::entity::kinds;
    use eqiora_core::{DimExponents, Id};
    use eqiora_schema::kernel::typing::{ExpressionType, RootContract};
    use eqiora_schema::kernel::{ExprDagBuilder, SymbolRef};

    use super::*;

    #[test]
    fn trace_support_map_requires_exact_parent_and_no_transfer_data() {
        let source = SpatialSupport::Volume {
            domain: "body",
            dimensions: 2,
        };
        let target = SpatialSupport::Boundary {
            domain: "wall",
            parent: "body",
            dimensions: 2,
        };
        let map = SupportMap::trace(source.clone(), target.clone()).unwrap();
        assert_eq!(map.source(), &source);
        assert_eq!(map.target(), &target);
        assert_eq!(map.orientation(), SupportMapOrientation::ParentOutward);
        assert_eq!(map.pairing(), SupportMapPairing::PointwiseValue);

        let foreign = SpatialSupport::Boundary {
            domain: "wall",
            parent: "other",
            dimensions: 2,
        };
        assert_eq!(
            SupportMap::trace(source, foreign),
            Err(SupportMapViolation::ParentMismatch)
        );
    }

    #[test]
    fn typed_trace_application_derives_the_same_exact_support_map() {
        let field = Id::<kinds::Field>::new();
        let volume = SpatialSupport::Volume {
            domain: "body",
            dimensions: 2,
        };
        let boundary = SpatialSupport::Boundary {
            domain: "wall",
            parent: "body",
            dimensions: 2,
        };
        let mut expression = ExprDagBuilder::new();
        let field_value = expression.symbol(SymbolRef::Field(field)).unwrap();
        let on = Id::<kinds::Domain>::new();
        let trace = expression.trace(field_value, on).unwrap();
        let dag = expression.finish([trace]).unwrap();
        let typed = TypedResidual::infer(
            dag,
            Some(boundary.clone()),
            |id| (id == on).then(|| boundary.clone()),
            RootContract::ComponentwiseResidual,
            |_| {
                Ok::<_, ()>(ExpressionType::scalar(
                    DimExponents::DIMENSIONLESS,
                    Some(volume.clone()),
                ))
            },
        )
        .unwrap();
        let map = SupportMap::classify_trace(&typed, trace).unwrap().unwrap();
        assert_eq!(map.source(), &volume);
        assert_eq!(map.target(), &boundary);
    }
}
