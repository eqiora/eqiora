//! Coordinate binders preserve support rather than erasing it as point evaluation does.
use super::inference::{NodeInference, inferred_type};
use super::*;
use eqiora_core::ScalarDomain;
use std::collections::BTreeSet;

pub(super) fn infer_factor<I: Clone + Eq, E>(
    expression: &ExprDag,
    factor: super::super::CoordinateMapFactor,
    source: &[ExprId],
    at: &[(ExprId, ExprId)],
    inferred: &[Option<ExpressionType<I>>],
) -> NodeInference<I, E> {
    let invalid = || NodeInference::Type(TypeViolation::CoordinatePullbackRequiresExactMap);
    let Some((target, _)) = at.first() else {
        return invalid();
    };
    if source.len() != at.len() {
        return invalid();
    }
    let checked = infer(expression, *target, source, at, inferred);
    if !matches!(checked, NodeInference::Typed(_)) {
        return checked;
    }
    let source = source
        .iter()
        .map(|id| inferred_type(inferred, *id).expect("validated coordinate"))
        .collect::<Vec<_>>();
    let at = at
        .iter()
        .map(|(x, y)| {
            (
                inferred_type(inferred, *x).expect("validated coordinate"),
                inferred_type(inferred, *y).expect("validated map"),
            )
        })
        .collect::<Vec<_>>();
    match factor_result_type(factor, &source, &at) {
        Ok(result) => NodeInference::Typed(result),
        Err(error) => NodeInference::Type(error),
    }
}

/// Infer a local differential factor from resolved coordinate types.
/// The syntax/DAG owner proves exact unique coordinate selectors separately.
pub fn factor_result_type<I: Clone + Eq>(
    factor: super::super::CoordinateMapFactor,
    source: &[ExpressionType<I>],
    at: &[(ExpressionType<I>, ExpressionType<I>)],
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    let invalid = || TypeViolation::CoordinatePullbackRequiresExactMap;
    let (target, _) = at.first().ok_or_else(invalid)?;
    if source.len() != at.len() {
        return Err(invalid());
    }
    let mut result = result_type(target, source, at)?;
    let mut dimension = eqiora_core::DimExponents::DIMENSIONLESS;
    if factor != super::super::CoordinateMapFactor::Orientation {
        for (source, (target, _)) in source.iter().zip(at) {
            dimension = target
                .dimension()
                .div(source.dimension())
                .and_then(|ratio| dimension.mul(ratio))
                .ok_or(TypeViolation::DimensionOverflow {
                    operation: "coordinate Jacobian",
                })?;
        }
    }
    result.value_type =
        eqiora_core::ValueType::scalar(ScalarDomain::Real, dimension).expect("scalar dimension");
    Ok(result)
}

pub(super) fn infer<I: Clone + Eq, E>(
    expression: &ExprDag,
    value: ExprId,
    source: &[ExprId],
    at: &[(ExprId, ExprId)],
    inferred: &[Option<ExpressionType<I>>],
) -> NodeInference<I, E> {
    let invalid = || NodeInference::Type(TypeViolation::CoordinatePullbackRequiresExactMap);
    let Some(value) = inferred_type(inferred, value) else {
        return NodeInference::Unavailable;
    };
    if source
        .iter()
        .chain(at.iter().flat_map(|(x, y)| [x, y]))
        .any(|id| inferred_type(inferred, *id).is_none())
    {
        return NodeInference::Unavailable;
    }
    if inventory(expression, source.iter().copied(), inferred).is_none()
        || inventory(expression, at.iter().map(|(id, _)| *id), inferred).is_none()
    {
        return invalid();
    }
    let source = source
        .iter()
        .map(|id| inferred_type(inferred, *id).expect("checked operand"))
        .collect::<Vec<_>>();
    let at = at
        .iter()
        .map(|(selector, mapped)| {
            (
                inferred_type(inferred, *selector).expect("checked operand"),
                inferred_type(inferred, *mapped).expect("checked operand"),
            )
        })
        .collect::<Vec<_>>();
    match result_type(&value, &source, &at) {
        Ok(value) => NodeInference::Typed(value),
        Err(error) => NodeInference::Type(error),
    }
}

/// Infer scalar pullback units and support from resolved coordinate types.
/// The syntax or DAG owner separately proves that selectors are complete,
/// unique exact coordinate references, rather than arbitrary scalar expressions.
pub fn result_type<I: Clone + Eq>(
    value: &ExpressionType<I>,
    source: &[ExpressionType<I>],
    at: &[(ExpressionType<I>, ExpressionType<I>)],
) -> Result<ExpressionType<I>, TypeViolation<I>> {
    let invalid = || TypeViolation::CoordinatePullbackRequiresExactMap;
    let source_support = source
        .first()
        .and_then(|ty| ty.support.as_ref())
        .ok_or_else(invalid)?;
    let target = at
        .first()
        .and_then(|(ty, _)| ty.support.as_ref())
        .ok_or_else(invalid)?;
    if !matches!(
        source_support,
        SpatialSupport::Coordinates { .. } | SpatialSupport::Volume { .. }
    ) || !matches!(
        target,
        SpatialSupport::Coordinates { .. } | SpatialSupport::Volume { .. }
    ) || source.len() != source_support.intrinsic_dimensions()
        || at.len() != target.intrinsic_dimensions()
        || source
            .iter()
            .any(|ty| ty.support.as_ref() != Some(source_support))
        || !value.shape().is_scalar()
        || value.value_type.array_rank() != 0
        || value.value_type.frame() != ValueFrame::Invariant
        || !matches!(
            value.value_type.scalar_domain(),
            ScalarDomain::Real | ScalarDomain::Complex
        )
        || value
            .support
            .as_ref()
            .is_some_and(|support| support != target)
    {
        return Err(invalid());
    }
    for (selector, mapped) in at {
        if selector.support.as_ref() != Some(target)
            || !mapped.shape().is_scalar()
            || mapped.value_type.array_rank() != 0
            || mapped.value_type.scalar_domain() != ScalarDomain::Real
            || mapped.value_type.frame() != ValueFrame::Invariant
            || mapped.dimension() != selector.dimension()
            || mapped
                .support
                .as_ref()
                .is_some_and(|support| support != source_support)
        {
            return Err(invalid());
        }
    }
    Ok(ExpressionType::new(
        value.value_type.clone(),
        Some(source_support.clone()),
    ))
}

fn inventory<I: Clone + Eq>(
    expression: &ExprDag,
    selectors: impl Iterator<Item = ExprId>,
    inferred: &[Option<ExpressionType<I>>],
) -> Option<SpatialSupport<I>> {
    let mut axes = BTreeSet::new();
    let mut selected = None;
    for selector in selectors {
        let ExprNode::Symbol(SymbolRef::Coordinate { factor, axis, .. }) =
            expression.node(selector)?
        else {
            return None;
        };
        if !axes.insert((factor.erase(), *axis)) {
            return None;
        }
        let ty = inferred_type(inferred, selector)?;
        let support = ty.support?;
        if !matches!(
            support,
            SpatialSupport::Coordinates { .. } | SpatialSupport::Volume { .. }
        ) || selected
            .as_ref()
            .is_some_and(|previous| previous != &support)
        {
            return None;
        }
        selected = Some(support);
    }
    selected.filter(|support| axes.len() == support.intrinsic_dimensions())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::ExprDagBuilder;
    use eqiora_core::{DynQuantity, Id, RawId, ValueType, entity::kinds};

    fn selector(
        builder: &mut ExprDagBuilder,
        support: Id<kinds::Domain>,
        factor: Id<kinds::Domain>,
        axis: usize,
    ) -> ExprId {
        builder
            .symbol(SymbolRef::Coordinate {
                support,
                factor,
                axis,
            })
            .unwrap()
    }

    fn typed(
        expression: ExprDag,
        supports: &[SpatialSupport<RawId>],
    ) -> Result<TypedResidual<RawId>, Vec<TypedResidualError<RawId, ()>>> {
        TypedResidual::infer(expression, None, RootContract::ValueRoots, |symbol| {
            let SymbolRef::Coordinate {
                support,
                factor,
                axis,
            } = symbol
            else {
                return Err(());
            };
            let support = supports
                .iter()
                .find(|candidate| *candidate.domain() == support.erase())
                .ok_or(())?;
            coordinate(&factor.erase(), axis, Some(support)).map_err(|_| ())
        })
    }

    #[test]
    fn orientation_does_not_require_representing_the_determinant_unit() {
        use crate::kernel::CoordinateMapFactor::{Orientation, SignedJacobian};
        let dimension = |n| DimExponents::from_integers([0, n, 0, 0, 0, 0, 0]).unwrap();
        let support = |domain, unit| SpatialSupport::Coordinates {
            domain,
            factors: vec![(domain, unit, 1)],
        };
        let source_support = support(Id::<kinds::Domain>::new().erase(), dimension(-i32::MAX));
        let target_support = support(Id::<kinds::Domain>::new().erase(), dimension(i32::MAX));
        let source = coordinate(source_support.domain(), 0, Some(&source_support)).unwrap();
        let target = coordinate(target_support.domain(), 0, Some(&target_support)).unwrap();
        let mapped = ExpressionType::new(target.value_type.clone(), Some(source_support));
        let at = [(target, mapped)];
        // Orientation has unit1 even when the determinant would require m^(2*MAX).
        assert_eq!(
            factor_result_type(Orientation, std::slice::from_ref(&source), &at)
                .unwrap()
                .dimension(),
            DimExponents::DIMENSIONLESS
        );
        assert!(matches!(
            factor_result_type(SignedJacobian, &[source], &at),
            Err(TypeViolation::DimensionOverflow { .. })
        ));
    }

    #[test]
    fn phase_space_pullback_retains_source_support_and_individual_units() {
        let reference = Id::new();
        let physical = Id::new();
        let factors: [Id<kinds::Domain>; 4] = std::array::from_fn(|_| Id::new());
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        let speed = DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).unwrap();
        let time = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
        let supports = [
            SpatialSupport::Coordinates {
                domain: reference.erase(),
                factors: vec![
                    (factors[0].erase(), length, 1),
                    (factors[1].erase(), speed, 1),
                ],
            },
            SpatialSupport::Coordinates {
                domain: physical.erase(),
                factors: vec![
                    (factors[2].erase(), length, 1),
                    (factors[3].erase(), speed, 1),
                ],
            },
        ];
        let mut builder = ExprDagBuilder::new();
        let xi = selector(&mut builder, reference, factors[0], 0);
        let nu = selector(&mut builder, reference, factors[1], 0);
        let x = selector(&mut builder, physical, factors[2], 0);
        let v = selector(&mut builder, physical, factors[3], 0);
        let dt = builder.constant(DynQuantity::new(2.0, time)).unwrap();
        let displacement = builder.mul(dt, nu).unwrap();
        let moved = builder.add(xi, displacement).unwrap();
        let value = builder.mul(x, v).unwrap();
        let base = builder.finish([value]).unwrap();
        let mut builder = ExprDagBuilder::from_dag(&base);
        let pullback = builder
            .pullback(value, vec![xi, nu], vec![(x, moved), (v, nu)])
            .unwrap();
        let checked = typed(builder.finish([pullback]).unwrap(), &supports).unwrap();
        let output = checked.node_type(pullback).unwrap();
        assert_eq!(output.dimension(), length.mul(speed).unwrap());
        assert_eq!(output.support, Some(supports[0].clone()));

        // Missing, repeated, differently typed and foreign coordinate bindings all reject.
        for (source, at) in [
            (vec![xi], vec![(x, moved), (v, nu)]),
            (vec![xi, xi], vec![(x, moved), (v, nu)]),
            (vec![xi, nu], vec![(x, moved), (x, moved)]),
            (vec![xi, nu], vec![(x, nu), (v, moved)]),
            (vec![xi, nu], vec![(x, x), (v, nu)]),
        ] {
            let mut builder = ExprDagBuilder::from_dag(&base);
            let root = builder.pullback(value, source, at).unwrap();
            let errors = typed(builder.finish([root]).unwrap(), &supports).unwrap_err();
            assert!(errors.iter().any(|error| matches!(
                error,
                TypedResidualError::Type {
                    error: TypeViolation::CoordinatePullbackRequiresExactMap,
                    ..
                }
            )));
        }
    }

    #[test]
    fn maps_keep_expression_identity_and_have_no_physical_dimension_ceiling() {
        let reference = Id::new();
        let physical = Id::new();
        for n in [1, 2, 3, 5, 6, 9, 16] {
            let supports = [
                SpatialSupport::Volume {
                    domain: reference.erase(),
                    dimensions: n,
                },
                SpatialSupport::Volume {
                    domain: physical.erase(),
                    dimensions: n,
                },
            ];
            let mut builder = ExprDagBuilder::new();
            let source = (0..n)
                .map(|axis| selector(&mut builder, reference, reference, axis))
                .collect::<Vec<_>>();
            let target = (0..n)
                .map(|axis| selector(&mut builder, physical, physical, axis))
                .collect::<Vec<_>>();
            let two = builder
                .constant(DynQuantity::new(2.0, DimExponents::DIMENSIONLESS))
                .unwrap();
            let doubled = source
                .iter()
                .map(|x| builder.mul(two, *x).unwrap())
                .collect::<Vec<_>>();
            let identity = builder
                .pullback(
                    target[0],
                    source.clone(),
                    target.iter().copied().zip(source.iter().copied()).collect(),
                )
                .unwrap();
            let dilation = builder
                .pullback(
                    target[0],
                    source,
                    target.iter().copied().zip(doubled).collect(),
                )
                .unwrap();
            let checked = typed(builder.finish([identity, dilation]).unwrap(), &supports).unwrap();
            assert_ne!(
                checked.expression().node(identity),
                checked.expression().node(dilation)
            );
            assert_eq!(
                checked.node_type(dilation).unwrap().support,
                Some(supports[0].clone())
            );
        }
    }

    #[test]
    fn scalar_pullback_requires_explicit_vector_frame_conversion() {
        let reference = Id::<kinds::Domain>::new().erase();
        let physical = Id::<kinds::Domain>::new().erase();
        let source_support = SpatialSupport::Volume {
            domain: reference,
            dimensions: 1,
        };
        let target_support = SpatialSupport::Volume {
            domain: physical,
            dimensions: 1,
        };
        let source = coordinate(&reference, 0, Some(&source_support)).unwrap();
        let target = coordinate(&physical, 0, Some(&target_support)).unwrap();
        let value = ExpressionType::new(
            ValueType::shaped(
                ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
                ValueShape::new([1]).unwrap(),
                ValueFrame::SpatialCartesian,
            )
            .unwrap(),
            Some(target_support),
        );
        // An identity coordinate map still cannot silently reinterpret a vector.
        assert!(matches!(
            result_type(
                &value,
                std::slice::from_ref(&source),
                &[(target, source.clone())]
            ),
            Err(TypeViolation::CoordinatePullbackRequiresExactMap)
        ));
    }

    #[test]
    fn rectangular_map_typing_does_not_assert_a_scalar_volume_jacobian() {
        let reference = Id::new();
        let physical = Id::new();
        let supports = [
            SpatialSupport::Volume {
                domain: reference.erase(),
                dimensions: 1,
            },
            SpatialSupport::Volume {
                domain: physical.erase(),
                dimensions: 2,
            },
        ];
        let mut builder = ExprDagBuilder::new();
        let xi = selector(&mut builder, reference, reference, 0);
        let x = selector(&mut builder, physical, physical, 0);
        let y = selector(&mut builder, physical, physical, 1);
        let value = builder.add(x, y).unwrap();
        let root = builder
            .pullback(value, vec![xi], vec![(x, xi), (y, xi)])
            .unwrap();
        let checked = typed(builder.finish([root]).unwrap(), &supports).unwrap();
        assert_eq!(
            checked.node_type(root).unwrap().support,
            Some(supports[0].clone())
        );
        // This valid embedding needs a metric measure, not a square determinant.
        for factor in [
            crate::kernel::CoordinateMapFactor::SignedJacobian,
            crate::kernel::CoordinateMapFactor::VolumeScale,
            crate::kernel::CoordinateMapFactor::Orientation,
        ] {
            let mut builder = ExprDagBuilder::from_dag(checked.expression());
            let factor = builder
                .coordinate_map_factor(factor, vec![xi], vec![(x, xi), (y, xi)])
                .unwrap();
            let errors = typed(builder.finish([factor]).unwrap(), &supports).unwrap_err();
            assert!(errors.iter().any(|error| matches!(
                error,
                TypedResidualError::Type {
                    error: TypeViolation::CoordinatePullbackRequiresExactMap,
                    ..
                }
            )));
        }
    }
}
