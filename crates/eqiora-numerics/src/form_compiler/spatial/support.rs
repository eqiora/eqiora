//! Distinguish map binder selectors from values consumed by the equation.
use super::*;
use eqiora_schema::kernel::{ExprId, typing::TypedResidual};

pub(super) fn require_equation_support(
    typed: &TypedResidual<RawId>,
    domain: RawId,
    dimension: usize,
) -> Result<(), Diagnostic> {
    let dag = typed.expression();
    let mut selectors = BTreeSet::<ExprId>::new();
    let mut values = dag.roots().to_vec();
    for node in dag.nodes() {
        if let ExprNode::CoordinateMapFactor { source, at, .. } = node {
            // Semantic typing owns map shape, coordinate identity and the
            // source support of each mapped value. Only target binding slots
            // are exempt from the equation's support, never their values.
            values.extend(source.iter().copied());
            for (selector, value) in at {
                selectors.insert(*selector);
                values.push(*value);
            }
        } else {
            super::super::scalar::push_operands(node, &mut values);
        }
    }
    let values = values.into_iter().collect::<BTreeSet<_>>();
    for (index, node_type) in typed.node_types().iter().enumerate() {
        let Some(support) = &node_type.support else {
            continue;
        };
        let id = dag.node_id(index as u32).expect("typed node");
        let binder_only = selectors.contains(&id)
            && !values.contains(&id)
            && matches!(
                dag.node(id),
                Some(ExprNode::Symbol(SymbolRef::Coordinate { .. }))
            );
        if support.ambient_dimensions() != Some(dimension)
            || (*support.domain() != domain && !binder_only)
        {
            return Err(invalid(
                "linear equation support or coordinate dimension differs from its Domain",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_core::{DimExponents, Id};
    use eqiora_schema::kernel::{
        CoordinateMapFactor, ExprDagBuilder,
        typing::{ExpressionType, RootContract, SpatialSupport},
    };

    #[test]
    fn target_selector_is_exempt_only_in_its_binding_slot() {
        let reference = Id::new();
        let target = Id::new();
        for consume_target in [false, true] {
            let mut dag = ExprDagBuilder::new();
            let xi = dag.coordinate(reference, reference, 0).unwrap();
            let x = dag.coordinate(target, target, 0).unwrap();
            let factor = dag
                .coordinate_map_factor(CoordinateMapFactor::VolumeScale, vec![xi], vec![(x, xi)])
                .unwrap();
            let mut roots = vec![factor];
            if consume_target {
                // Reusing the exact same DAG node as a value cannot inherit
                // the exemption granted to its occurrence as a map selector.
                roots.push(dag.neg(x).unwrap());
            }
            let typed = TypedResidual::infer(
                dag.finish(roots).unwrap(),
                None,
                |_| None,
                RootContract::ValueRoots,
                |symbol| {
                    let SymbolRef::Coordinate { support, .. } = symbol else {
                        unreachable!()
                    };
                    Ok::<_, ()>(ExpressionType::scalar(
                        DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
                        Some(SpatialSupport::Volume {
                            domain: support.erase(),
                            dimensions: 1,
                        }),
                    ))
                },
            )
            .unwrap();
            let result = require_equation_support(&typed, reference.erase(), 1);
            if consume_target {
                assert!(
                    result
                        .unwrap_err()
                        .message()
                        .contains("support or coordinate")
                );
            } else {
                result.unwrap();
                assert!(require_equation_support(&typed, reference.erase(), 2).is_err());
                assert!(require_equation_support(&typed, target.erase(), 1).is_err());
            }
        }
    }
}
