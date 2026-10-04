//! Derived Model quantities retain expression meaning without owning solve unknowns.

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, Id, RawId, ValueType};

use super::typing::{ExpressionType, SpatialSupport};
use super::{ExprDag, KernelNode};

/// Integration measure on one exact continuous support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservableMeasure {
    /// Cartesian product measure: physical volume or the product of abstract factor units.
    Volume,
    /// Exterior surface measure; orientation belongs to the exact boundary normal.
    Boundary,
}

impl ObservableMeasure {
    /// Infer an integral's exact output type using identity-parametric support.
    ///
    /// Source checking and Semantic Model admission use this same rule.
    /// # Errors
    /// Rejects a wrong measure, foreign support or unrepresentable dimension.
    pub fn output_type<I: PartialEq>(
        self,
        root: &ExpressionType<I>,
        support: &SpatialSupport<I>,
    ) -> Result<ValueType, Diagnostic> {
        if !matches!(
            root.value_type.scalar_domain(),
            eqiora_core::ScalarDomain::Real
                | eqiora_core::ScalarDomain::Complex
                | eqiora_core::ScalarDomain::Integer
        ) {
            return Err(invalid("Observable integral requires a numeric integrand"));
        }
        let measure_dimension = match (self, support) {
            (ObservableMeasure::Volume, SpatialSupport::Coordinates { factors, .. }) => factors
                .iter()
                .try_fold(DimExponents::DIMENSIONLESS, |product, (_, dimension)| {
                    product.mul(*dimension)
                })
                .ok_or_else(|| {
                    invalid("coordinate product measure dimension exceeds its exact representation")
                })?,
            (ObservableMeasure::Volume, SpatialSupport::Volume { dimensions, .. }) => {
                length_measure(*dimensions)?
            }
            (ObservableMeasure::Boundary, SpatialSupport::Boundary { dimensions, .. }) => {
                length_measure(
                    dimensions
                        .checked_sub(1)
                        .ok_or_else(|| invalid("Observable boundary has no ambient dimension"))?,
                )?
            }
            _ => return Err(invalid("Observable measure does not match its Domain kind")),
        };
        if root
            .support
            .as_ref()
            .is_some_and(|actual| actual != support)
        {
            return Err(invalid(
                "Observable integrand does not have the exact integration support; boundary traces must be explicit",
            ));
        }
        let dimension = root.dimension().mul(measure_dimension).ok_or_else(|| {
            invalid("Observable integral dimension exceeds its exact representation")
        })?;
        root.value_type
            .clone()
            .with_dimension(dimension)
            .map_err(|_| invalid("Observable integral requires a numeric result type"))
    }
}

/// Reduction meaning, independent of mesh and quadrature policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservableReduction {
    /// An instantaneous value with no spatial support.
    Value,
    /// An integral over exactly one declared spatial Domain.
    SpatialIntegral {
        /// Exact volume or boundary Domain, never a renderer selection.
        domain: Id<kinds::Domain>,
        /// Measure must agree with the selected Domain kind.
        measure: ObservableMeasure,
    },
}

impl ObservableReduction {
    /// Exact integration Domain, if present.
    #[must_use]
    pub const fn domain(self) -> Option<Id<kinds::Domain>> {
        match self {
            Self::Value => None,
            Self::SpatialIntegral { domain, .. } => Some(domain),
        }
    }
}

/// Named derived quantity in Model meaning, never a Field or solving Relation.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservableDef {
    id: Id<kinds::Observable>,
    value_type: ValueType,
    expression: ExprDag,
    reduction: ObservableReduction,
}

impl ObservableDef {
    /// Retain a single output expression and exact declared result type.
    ///
    /// Whole-Model admission resolves symbol types and validates the measure.
    /// # Errors
    /// Rejects expressions with other than one root.
    pub fn new(
        id: Id<kinds::Observable>,
        value_type: ValueType,
        expression: ExprDag,
        reduction: ObservableReduction,
    ) -> Result<Self, Diagnostic> {
        if expression.roots().len() != 1 {
            return Err(invalid("Observable requires exactly one expression root"));
        }
        Ok(Self {
            id,
            value_type,
            expression,
            reduction,
        })
    }

    /// Exact semantic identity.
    #[must_use]
    pub const fn id(&self) -> Id<kinds::Observable> {
        self.id
    }

    /// Complete declared result type after reduction.
    #[must_use]
    pub const fn value_type(&self) -> &ValueType {
        &self.value_type
    }

    /// Retained integrand or value expression.
    #[must_use]
    pub const fn expression(&self) -> &ExprDag {
        &self.expression
    }

    /// Mathematical reduction, without numerical approximation policy.
    #[must_use]
    pub const fn reduction(&self) -> ObservableReduction {
        self.reduction
    }

    /// Check the independently inferred root against exact support and measure.
    /// # Errors
    /// Rejects a wrong Domain, measure, expression support or resulting type.
    pub fn validate_type(
        &self,
        root: &ExpressionType<RawId>,
        integration_support: Option<&SpatialSupport<RawId>>,
    ) -> Result<(), Diagnostic> {
        let inferred = match self.reduction {
            ObservableReduction::Value => {
                if root.support.is_some() || integration_support.is_some() {
                    return Err(invalid(
                        "Observable value requires no spatial support; use an explicit integral",
                    ));
                }
                root.value_type.clone()
            }
            ObservableReduction::SpatialIntegral { domain, measure } => {
                let support = integration_support.ok_or_else(|| {
                    invalid("Observable integral requires an admitted spatial Domain")
                })?;
                if *support.domain() != domain.erase() {
                    return Err(invalid(
                        "Observable integral Domain differs from its exact support",
                    ));
                }
                measure.output_type(root, support)?
            }
        };
        if inferred != self.value_type {
            return Err(invalid(
                "Observable declared type differs from its expression and measure",
            ));
        }
        Ok(())
    }
}

impl From<ObservableDef> for KernelNode {
    fn from(value: ObservableDef) -> Self {
        Self::Observable(value)
    }
}

fn invalid(message: &str) -> Diagnostic {
    Diagnostic::error(codes::INVALID_KERNEL_DEFINITION, message)
}

fn length_measure(dimensions: usize) -> Result<DimExponents, Diagnostic> {
    let exponent = i32::try_from(dimensions)
        .map_err(|_| invalid("Observable measure dimension exceeds its exact representation"))?;
    DimExponents::from_integers([0, exponent, 0, 0, 0, 0, 0])
        .ok_or_else(|| invalid("Observable measure dimension exceeds its exact representation"))
}

#[cfg(test)]
mod tests {
    use super::super::ExprDagBuilder;
    use super::*;
    use eqiora_core::{DynQuantity, ScalarDomain};

    fn scalar(exponent: i32) -> ValueType {
        ValueType::scalar(
            ScalarDomain::Real,
            DimExponents::from_integers([0, exponent, 0, 0, 0, 0, 0]).unwrap(),
        )
        .unwrap()
    }

    fn integral(
        domain: Id<kinds::Domain>,
        measure: ObservableMeasure,
        exponent: i32,
    ) -> ObservableDef {
        let mut dag = ExprDagBuilder::new();
        let root = dag
            .constant(DynQuantity::new(3.0, DimExponents::DIMENSIONLESS))
            .unwrap();
        ObservableDef::new(
            Id::new(),
            scalar(exponent),
            dag.finish([root]).unwrap(),
            ObservableReduction::SpatialIntegral { domain, measure },
        )
        .unwrap()
    }

    #[test]
    fn measure_and_support_determine_dimensions_independently() {
        let domain = Id::new();
        let parent = Id::<kinds::Domain>::new();
        let constant = ExpressionType::new(scalar(0), None);
        let volume = SpatialSupport::Volume {
            domain: domain.erase(),
            dimensions: 3,
        };
        let boundary = SpatialSupport::Boundary {
            domain: domain.erase(),
            parent: parent.erase(),
            dimensions: 3,
        };
        let volume_integral = integral(domain, ObservableMeasure::Volume, 3);
        assert!(
            volume_integral
                .validate_type(&constant, Some(&volume))
                .is_ok()
        );
        assert!(
            volume_integral
                .validate_type(&constant, Some(&boundary))
                .is_err()
        );
        assert!(
            integral(domain, ObservableMeasure::Boundary, 2)
                .validate_type(&constant, Some(&boundary))
                .is_ok()
        );
        assert!(
            integral(domain, ObservableMeasure::Volume, 2)
                .validate_type(&constant, Some(&volume))
                .is_err()
        );
        let foreign = SpatialSupport::Volume {
            domain: parent.erase(),
            dimensions: 3,
        };
        assert!(
            volume_integral
                .validate_type(&constant, Some(&foreign))
                .is_err()
        );
        assert!(
            volume_integral
                .validate_type(
                    &ExpressionType::new(scalar(0), Some(foreign)),
                    Some(&volume)
                )
                .is_err()
        );
        let wrong_parent = SpatialSupport::Boundary {
            domain: domain.erase(),
            parent: Id::<kinds::Domain>::new().erase(),
            dimensions: 3,
        };
        assert!(
            integral(domain, ObservableMeasure::Boundary, 2)
                .validate_type(
                    &ExpressionType::new(scalar(0), Some(wrong_parent)),
                    Some(&boundary)
                )
                .is_err()
        );
    }

    #[test]
    fn zero_dimensional_measure_does_not_make_boolean_integrands_numeric() {
        let support = SpatialSupport::Boundary {
            domain: "wall",
            parent: "body",
            dimensions: 1,
        };
        let boolean = ExpressionType::new(ValueType::boolean(), None);
        assert!(
            ObservableMeasure::Boundary
                .output_type(&boolean, &support)
                .is_err()
        );
    }

    #[test]
    fn derived_value_does_not_accept_supported_field_without_reduction() {
        let mut dag = ExprDagBuilder::new();
        let root = dag
            .constant(DynQuantity::new(3.0, DimExponents::DIMENSIONLESS))
            .unwrap();
        let value = ObservableDef::new(
            Id::new(),
            scalar(0),
            dag.finish([root]).unwrap(),
            ObservableReduction::Value,
        )
        .unwrap();
        assert!(
            value
                .validate_type(&ExpressionType::new(scalar(0), None), None)
                .is_ok()
        );
        let support = SpatialSupport::Volume {
            domain: Id::<kinds::Domain>::new().erase(),
            dimensions: 1,
        };
        assert!(
            value
                .validate_type(&ExpressionType::new(scalar(0), Some(support)), None)
                .is_err()
        );
        assert!(matches!(KernelNode::from(value), KernelNode::Observable(_)));
    }
}
