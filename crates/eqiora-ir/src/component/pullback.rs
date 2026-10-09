//! Coordinate substitution shares scalar execution and its JVP/VJP calculus.
use super::*;

impl<I: Clone + Eq> ComponentDagLowering<'_, I> {
    pub(super) fn lower_pullback(
        &mut self,
        value: ExprId,
        at: &[(ExprId, ExprId)],
        part: ScalarPart,
    ) -> Result<ScalarInputValueId, Diagnostic> {
        *self.finite_products = self
            .finite_products
            .checked_add(at.len())
            .and_then(|work| work.checked_add(self.coordinate_bindings.len()))
            .filter(|work| *work <= 1_000_000)
            .ok_or_else(|| {
                invalid_component_ir("coordinate pullback exceeds component work budget")
            })?;
        if self.pullback_depth >= 128 {
            return Err(invalid_component_ir(
                "coordinate pullback exceeds expression binding depth",
            ));
        }
        // Every mapped expression is evaluated in the outer context, before
        // any new target binding is installed (including self-maps).
        let mut bindings = self.coordinate_bindings.clone();
        for (selector, mapped) in at {
            let Some(ExprNode::Symbol(symbol @ SymbolRef::Coordinate { .. })) =
                self.expression.node(*selector)
            else {
                return Err(invalid_component_ir(
                    "pullback target selector is not an exact coordinate",
                ));
            };
            let symbol = *symbol;
            let mapped = self.lower_part(*mapped, &[], ScalarPart::Real)?;
            bindings.insert(symbol, mapped);
        }
        let outer_bindings = std::mem::replace(&mut self.coordinate_bindings, bindings);
        let outer_remapped = std::mem::take(&mut self.remapped);
        let outer_maps = std::mem::take(&mut self.map_entries);
        self.pullback_depth += 1;
        let result = self.lower_part(value, &[], part);
        self.pullback_depth -= 1;
        self.coordinate_bindings = outer_bindings;
        self.remapped = outer_remapped;
        self.map_entries = outer_maps;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DifferentiationRole, LinearizedRelation, RelationCotangent, RelationTangent};
    use eqiora_core::{DimExponents, Id, entity::kinds};
    use eqiora_schema::kernel::{
        ExprDagBuilder,
        typing::{ExpressionType, RootContract, SpatialSupport},
    };

    fn polynomial(self_map: bool, add_identity: bool) -> ComponentScalarization {
        let source = Id::<kinds::Domain>::new();
        let target = if self_map { source } else { Id::new() };
        let mut builder = ExprDagBuilder::new();
        let mut axis = |support, axis| {
            builder
                .symbol(SymbolRef::Coordinate {
                    support,
                    factor: support,
                    axis,
                })
                .unwrap()
        };
        let xi = axis(source, 0);
        let eta = axis(source, 1);
        let x = axis(target, 0);
        let y = axis(target, 1);
        let xx = builder.mul(x, x).unwrap();
        let xy = builder.mul(x, y).unwrap();
        let u = builder.add(xx, xy).unwrap();
        let (mapped_x, mapped_y) = if self_map {
            (builder.add(xi, eta).unwrap(), builder.sub(xi, eta).unwrap())
        } else {
            let two = builder
                .constant(DynQuantity::new(2.0, DimExponents::DIMENSIONLESS))
                .unwrap();
            let three = builder
                .constant(DynQuantity::new(3.0, DimExponents::DIMENSIONLESS))
                .unwrap();
            let scaled = builder.mul(two, xi).unwrap();
            (
                builder.add(scaled, eta).unwrap(),
                builder.mul(three, eta).unwrap(),
            )
        };
        let pulled = builder
            .pullback(u, vec![xi, eta], vec![(x, mapped_x), (y, mapped_y)])
            .unwrap();
        let root = if add_identity {
            let identity = builder
                .pullback(u, vec![xi, eta], vec![(x, xi), (y, eta)])
                .unwrap();
            builder.add(pulled, identity).unwrap()
        } else {
            pulled
        };
        let typed = TypedResidual::infer(
            builder.finish([root]).unwrap(),
            None,
            |_| None,
            RootContract::ValueRoots,
            |symbol| {
                let SymbolRef::Coordinate {
                    support,
                    factor,
                    axis,
                } = symbol
                else {
                    return Err(());
                };
                ExpressionType::coordinate(
                    &factor.erase(),
                    axis,
                    Some(&SpatialSupport::Volume {
                        domain: support.erase(),
                        dimensions: 2,
                    }),
                )
                .map_err(|_| ())
            },
        )
        .unwrap();
        let lowered = ComponentScalarization::lower(&typed).unwrap();
        assert!(lowered.rows()[0].symbols().iter().all(
            |s| matches!(s.symbol(), SymbolRef::Coordinate { support, .. } if support == source)
        ));
        lowered
    }

    #[test]
    fn chain_rule_and_transpose_use_the_same_retained_map() {
        for (self_map, add_identity, expected, gradient) in [
            // 4xi²+10xi*eta+4eta²; gradient=(8xi+10eta,10xi+8eta).
            (false, false, 2.5, [7.0, 6.5]),
            // Add identity pullback xi²+xi*eta in a different binding context.
            (false, true, 2.6875, [8.0, 6.75]),
            // Self-map (xi+eta,xi-eta) yields 2xi²+2xi*eta.
            (true, false, 0.375, [2.0, 0.5]),
        ] {
            let lowered = polynomial(self_map, add_identity);
            let row = &lowered.rows()[0];
            let axes = row
                .symbols()
                .iter()
                .map(|s| {
                    let SymbolRef::Coordinate { axis, .. } = s.symbol() else {
                        panic!("coordinate input");
                    };
                    axis
                })
                .collect::<Vec<_>>();
            assert_eq!(axes.len(), 2);
            let values = axes
                .iter()
                .map(|axis| [0.25, 0.5][*axis])
                .collect::<Vec<_>>();
            assert_eq!(row.evaluate(&values).unwrap(), expected);
            let roles = vec![DifferentiationRole::Parameter; values.len()];
            let linearized = row.linearize(&values, &roles).unwrap();
            let direction = axes
                .iter()
                .map(|axis| [3.0, -2.0][*axis])
                .collect::<Vec<_>>();
            let mut tangent = [0.0];
            linearized
                .jvp(RelationTangent::Parameter(&direction), &mut tangent)
                .unwrap();
            assert_eq!(tangent[0], 3.0 * gradient[0] - 2.0 * gradient[1]);
            let mut adjoint = vec![0.0; axes.len()];
            linearized
                .vjp(&[2.0], RelationCotangent::Parameter(&mut adjoint))
                .unwrap();
            for (axis, value) in axes.iter().zip(adjoint) {
                assert_eq!(value, 2.0 * gradient[*axis]);
            }
        }
    }
}
