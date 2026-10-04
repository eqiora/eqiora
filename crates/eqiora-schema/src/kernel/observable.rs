//! Derived Model quantities retain expression meaning without owning solve unknowns.

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, Id, RawId, ValueType};

use super::typing::{ExpressionType, SpatialSupport};
use super::{ExprDag, KernelNode};

mod measure;
pub use measure::ObservableMeasure;

/// Reduction meaning, independent of mesh and quadrature policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservableReduction {
    /// An instantaneous value on its declared output support.
    Value,
    /// An integral over exactly one declared spatial Domain.
    SpatialIntegral {
        /// Exact input support of the density, before removing selected factors.
        input: Id<kinds::Domain>,
        /// Exact volume or boundary Domain, never a renderer selection.
        domain: Id<kinds::Domain>,
        /// Measure must agree with the selected Domain kind.
        measure: ObservableMeasure,
    },
}

impl ObservableReduction {
    /// Exact density input Domain, if this is an integral.
    #[must_use]
    pub const fn input_domain(self) -> Option<Id<kinds::Domain>> {
        match self {
            Self::Value => None,
            Self::SpatialIntegral { input, .. } => Some(input),
        }
    }

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
        input_support: Option<&SpatialSupport<RawId>>,
        integration_support: Option<&SpatialSupport<RawId>>,
        output_support: Option<&SpatialSupport<RawId>>,
    ) -> Result<(), Diagnostic> {
        let inferred = match self.reduction {
            ObservableReduction::Value => {
                if input_support.is_some()
                    || integration_support.is_some()
                    || root
                        .support
                        .as_ref()
                        .is_some_and(|support| Some(support) != output_support)
                {
                    return Err(invalid(
                        "Observable value differs from its declared output support",
                    ));
                }
                root.value_type.clone()
            }
            ObservableReduction::SpatialIntegral {
                input,
                domain,
                measure,
            } => {
                let support = integration_support.ok_or_else(|| {
                    invalid("Observable integral requires an admitted spatial Domain")
                })?;
                let input_support = input_support.ok_or_else(|| {
                    invalid("Observable integral requires an admitted input Domain")
                })?;
                if *input_support.domain() != input.erase() || *support.domain() != domain.erase() {
                    return Err(invalid(
                        "Observable integral Domain differs from its exact support",
                    ));
                }
                measure
                    .output_type(root, input_support, support, output_support)?
                    .value_type
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
            ObservableReduction::SpatialIntegral {
                input: domain,
                domain,
                measure,
            },
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
                .validate_type(&constant, Some(&volume), Some(&volume), None)
                .is_ok()
        );
        assert!(
            volume_integral
                .validate_type(&constant, Some(&boundary), Some(&boundary), None)
                .is_err()
        );
        assert!(
            integral(domain, ObservableMeasure::Boundary, 2)
                .validate_type(&constant, Some(&boundary), Some(&boundary), None)
                .is_ok()
        );
        assert!(
            integral(domain, ObservableMeasure::Volume, 2)
                .validate_type(&constant, Some(&volume), Some(&volume), None)
                .is_err()
        );
        let foreign = SpatialSupport::Volume {
            domain: parent.erase(),
            dimensions: 3,
        };
        assert!(
            volume_integral
                .validate_type(&constant, Some(&foreign), Some(&foreign), None)
                .is_err()
        );
        assert!(
            volume_integral
                .validate_type(
                    &ExpressionType::new(scalar(0), Some(foreign)),
                    Some(&volume),
                    Some(&volume),
                    None
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
                    Some(&boundary),
                    Some(&boundary),
                    None
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
                .output_type(&boolean, &support, &support, None)
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
                .validate_type(&ExpressionType::new(scalar(0), None), None, None, None)
                .is_ok()
        );
        let support = SpatialSupport::Volume {
            domain: Id::<kinds::Domain>::new().erase(),
            dimensions: 1,
        };
        assert!(
            value
                .validate_type(
                    &ExpressionType::new(scalar(0), Some(support)),
                    None,
                    None,
                    None
                )
                .is_err()
        );
        assert!(matches!(KernelNode::from(value), KernelNode::Observable(_)));
    }
}
