//! Ordered scalar selection retains the complete operand type and compatible support.
use super::*;

impl<I: Clone + Eq> ExpressionType<I> {
    /// Type a finite minimum or maximum without numeric promotion or dimension changes.
    /// Only ordinary invariant real/integer scalars are ordered. Equal values select
    /// the first operand; both operands remain part of the expression dependency graph.
    pub fn ordered_selection(self, other: Self) -> Result<Self, TypeViolation<I>> {
        let condition = self
            .clone()
            .compare(crate::kernel::ComparisonOp::LessEqual, other.clone())?;
        condition.select(self, other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{ExprDagBuilder, ExprNode};
    use eqiora_core::{Id, ScalarDomain, ValueLiteral, ValueType};

    fn typed(value: ValueType) -> ExpressionType<u32> {
        ExpressionType::new(value, None)
    }

    #[test]
    fn ordered_selection_retains_dimension_and_exact_support_without_promotion() {
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        let real = ValueType::scalar(ScalarDomain::Real, length).expect("checked scalar type");
        let support = SpatialSupport::Volume {
            domain: 7,
            dimensions: 2,
        };
        let spatial = ExpressionType::new(real.clone(), Some(support.clone()));
        assert_eq!(
            typed(real.clone())
                .ordered_selection(spatial.clone())
                .unwrap(),
            spatial
        );
        assert_eq!(
            spatial
                .clone()
                .ordered_selection(typed(real.clone()))
                .unwrap()
                .support,
            Some(support)
        );
        let foreign = ExpressionType::new(
            real.clone(),
            Some(SpatialSupport::Volume {
                domain: 8,
                dimensions: 2,
            }),
        );
        assert!(spatial.ordered_selection(foreign).is_err());
        assert!(
            typed(real)
                .ordered_selection(typed(
                    ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                        .expect("checked scalar type")
                ))
                .is_err()
        );
        let integer = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("checked scalar type");
        assert_eq!(
            typed(integer.clone())
                .ordered_selection(typed(integer.clone()))
                .unwrap(),
            typed(integer.clone())
        );
        assert!(
            typed(integer)
                .ordered_selection(typed(
                    ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                        .expect("checked scalar type")
                ))
                .is_err()
        );
    }

    #[test]
    fn ordered_selection_rejects_unordered_and_shaped_types() {
        let integer = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("checked scalar type");
        let real = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .expect("checked scalar type");
        for value in [
            ValueType::boolean(),
            ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
                .expect("checked scalar type"),
            ValueType::index(Id::new(), 2).unwrap(),
            ValueType::counts(Id::new(), 2).unwrap(),
            ValueType::coordinates(
                eqiora_core::FiniteBasis::new(Id::new(), 2).unwrap(),
                eqiora_core::ScalarDomain::Integer,
                eqiora_core::DimExponents::DIMENSIONLESS,
            )
            .unwrap(),
            real.array(2).unwrap(),
            integer.clone().array(2).unwrap(),
            integer
                .with_dimension(DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap())
                .expect("checked numeric dimension"),
        ] {
            assert!(
                typed(value.clone())
                    .ordered_selection(typed(value))
                    .is_err()
            );
        }
    }

    #[test]
    fn ordered_selection_dag_retains_both_ordered_operands_and_checks_rhs() {
        let integer = ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
            .expect("checked scalar type");
        let mut builder = ExprDagBuilder::new();
        let left = builder
            .constant(ValueLiteral::from_integer(integer.clone(), 9_007_199_254_740_993).unwrap())
            .unwrap();
        let right = builder
            .constant(ValueLiteral::from_integer(integer.clone(), 9_007_199_254_740_992).unwrap())
            .unwrap();
        let minimum = builder.min(left, right).unwrap();
        let maximum = builder.max(right, left).unwrap();
        let dag = builder.finish([minimum, left, maximum, right]).unwrap();
        for (root, first, second, op) in [
            (minimum, left, right, crate::kernel::ComparisonOp::LessEqual),
            (
                maximum,
                right,
                left,
                crate::kernel::ComparisonOp::GreaterEqual,
            ),
        ] {
            let Some(ExprNode::Select {
                condition,
                then_value,
                else_value,
            }) = dag.node(root)
            else {
                panic!("selection must share the conditional owner")
            };
            assert_eq!((*then_value, *else_value), (first, second));
            assert_eq!(
                dag.node(*condition),
                Some(&ExprNode::Compare(op, first, second))
            );
        }
        assert_eq!(dag.nodes().len(), 6);
        let inferred = TypedResidual::<u32>::infer(
            dag,
            None,
            |_| None,
            RootContract::EquationSides,
            |_| Err::<ExpressionType<u32>, ()>(()),
        )
        .unwrap();
        assert_eq!(inferred.node_type(minimum), Some(&typed(integer)));

        let mut missing = ExprDagBuilder::new();
        let existing = missing.constant(ValueLiteral::boolean(false)).unwrap();
        assert!(missing.min(existing, right).is_err());
        assert!(missing.max(right, existing).is_err());

        for minimum in [true, false] {
            let mut incompatible = ExprDagBuilder::new();
            let left = incompatible
                .constant(
                    ValueLiteral::from_real(
                        ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                            .expect("checked scalar type"),
                        -1.0,
                    )
                    .unwrap(),
                )
                .unwrap();
            let right = incompatible.constant(ValueLiteral::boolean(false)).unwrap();
            let mut builder = incompatible;
            let root = if minimum {
                builder.min(left, right)
            } else {
                builder.max(left, right)
            }
            .unwrap();
            assert!(
                TypedResidual::<u32>::infer(
                    builder.finish([root, left]).unwrap(),
                    None,
                    |_| None,
                    RootContract::EquationSides,
                    |_| Err::<ExpressionType<u32>, ()>(())
                )
                .is_err()
            );
        }
    }
    #[test]
    fn finite_fold_reuses_prior_nodes_and_keeps_first_on_ties() {
        let mut builder = ExprDagBuilder::new();
        let value = ValueLiteral::from_integer(
            ValueType::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
                .expect("checked scalar type"),
            i64::MAX,
        )
        .unwrap();
        let first = builder.constant(value.clone()).unwrap();
        let mut root = first;
        for _ in 0..128 {
            let next = builder.constant(value.clone()).unwrap();
            let previous = root;
            root = builder.min(previous, next).unwrap();
        }
        let dag = builder.finish([root]).unwrap();
        assert_eq!(dag.nodes().len(), 1 + 3 * 128);
        let Some(ExprNode::Select {
            condition,
            then_value,
            else_value,
        }) = dag.node(root)
        else {
            panic!("expected shared selection")
        };
        assert_eq!(
            dag.node(*condition),
            Some(&ExprNode::Compare(
                crate::kernel::ComparisonOp::LessEqual,
                *then_value,
                *else_value
            ))
        );
        assert_eq!(
            value.checked_order(&value).unwrap(),
            std::cmp::Ordering::Equal
        );
    }
}
