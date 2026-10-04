//! One derivation owner for source admission and exact live-Model replay.
use super::*;
use eqiora_schema::kernel::{ExprId, ExprNode, ObservableDef};
use std::collections::BTreeSet;

type Resolve<'a> = dyn FnMut(Id<kinds::Observable>) -> Result<(ObservableDef, TypedResidual<RawId>), Diagnostic>
    + 'a;

pub(super) struct Derived {
    pub(super) value: AuthoredFormExpression,
    pub(super) holding: BTreeSet<RawId>,
    pub(super) volume: RawId,
}

pub(super) fn derive(
    functional: Id<kinds::Observable>,
    wrt: Id<kinds::Field>,
    directions: &[String],
    resolve: &mut Resolve<'_>,
) -> Result<Derived, Diagnostic> {
    Context {
        wrt,
        directions,
        resolve,
        remaining: 65536,
    }
    .functional(functional, 0)
}

struct Context<'a, 'r> {
    wrt: Id<kinds::Field>,
    directions: &'a [String],
    resolve: &'a mut Resolve<'r>,
    remaining: usize,
}

impl Context<'_, '_> {
    fn step(&mut self, depth: usize) -> Result<(), Diagnostic> {
        if depth > 96 || self.remaining == 0 {
            return Err(wire::rejection(
                "functional composition exceeds its bounded derivation inventory",
            ));
        }
        self.remaining -= 1;
        Ok(())
    }

    fn functional(
        &mut self,
        id: Id<kinds::Observable>,
        depth: usize,
    ) -> Result<Derived, Diagnostic> {
        self.step(depth)?;
        let (functional, density) = (self.resolve)(id)?;
        if functional.id() != id || density.expression() != functional.expression() {
            return Err(wire::rejection(
                "variation energy differs from the exact live Observable",
            ));
        }
        self.remaining = self
            .remaining
            .checked_sub(density.expression().nodes().len())
            .ok_or_else(|| {
                wire::rejection("functional composition exceeds its expression work bound")
            })?;
        let root = density
            .node_type(density.expression().roots()[0])
            .ok_or_else(|| wire::rejection("functional density has no typed root"))?;
        if matches!(functional.reduction(), ObservableReduction::Value) {
            functional.validate_type(root, None, None, None, None)?;
            return self.expression(&density, density.expression().roots()[0], depth + 1);
        }
        let ObservableReduction::SpatialIntegral {
            limits: None,
            input,
            domain,
            measure,
        } = functional.reduction()
        else {
            return Err(wire::rejection(
                "local variation requires a full fixed-domain integral",
            ));
        };
        if input != domain {
            return Err(wire::rejection(
                "local variation requires a full fixed-domain integral",
            ));
        }
        let support = functional_support(&density, domain)?;
        if !matches!(
            (measure, support),
            (ObservableMeasure::Volume, SpatialSupport::Volume { .. })
                | (ObservableMeasure::Boundary, SpatialSupport::Boundary { .. })
        ) {
            return Err(wire::rejection(
                "variation measure differs from its live support",
            ));
        }
        functional.validate_type(root, None, Some(support), Some(support), None)?;
        let volume = support.parent().copied().unwrap_or(domain.erase());
        let holding = functional
            .expression()
            .nodes()
            .iter()
            .filter_map(|node| match node {
                ExprNode::Symbol(SymbolRef::Field(id)) if *id != self.wrt => Some(id.erase()),
                ExprNode::Symbol(SymbolRef::Parameter(id)) => Some(id.erase()),
                _ => None,
            })
            .collect();
        let value = derive_value(
            &density,
            self.wrt,
            self.directions,
            domain,
            functional.value_type().dimension(),
            &mut self.remaining,
        )?;
        Ok(Derived {
            value,
            holding,
            volume,
        })
    }

    fn expression(
        &mut self,
        density: &TypedResidual<RawId>,
        id: ExprId,
        depth: usize,
    ) -> Result<Derived, Diagnostic> {
        self.step(depth)?;
        let invalid = || {
            wire::rejection(
                "composite functional variation requires a sum or difference of fixed spatial Observables",
            )
        };
        let mut derived = match density.expression().node(id).ok_or_else(invalid)? {
            ExprNode::Symbol(SymbolRef::Observable(id)) => self.functional(*id, depth + 1)?,
            ExprNode::Neg(value) => {
                let mut result = self.expression(density, *value, depth + 1)?;
                let value = result.value;
                result.value = typed(
                    AuthoredFormExpressionKind::Neg(Box::new(value.clone())),
                    value.dimension,
                    value.shape,
                    None,
                );
                result
            }
            node @ (ExprNode::Add(left, right) | ExprNode::Sub(left, right)) => {
                let mut left = self.expression(density, *left, depth + 1)?;
                let right = self.expression(density, *right, depth + 1)?;
                if left.volume != right.volume || left.value.dimension != right.value.dimension {
                    return Err(wire::rejection(
                        "composite functional terms must share the exact parent volume and dimension",
                    ));
                }
                let dimension = left.value.dimension;
                left.holding.extend(right.holding);
                left.value = typed(
                    binary(
                        if matches!(node, ExprNode::Add(..)) {
                            BinaryOp::Add
                        } else {
                            BinaryOp::Sub
                        },
                        left.value,
                        right.value,
                    ),
                    dimension,
                    ValueShape::scalar(),
                    None,
                );
                left
            }
            _ => return Err(invalid()),
        };
        let ty = density.node_type(id).ok_or_else(invalid)?;
        if ty.support.is_some()
            || !ty.shape().is_scalar()
            || ty.dimension() != derived.value.dimension
        {
            return Err(wire::rejection(
                "composite functional value differs from its live reduced type",
            ));
        }
        // Every outer expression remains a scalar after all exact measure reductions.
        derived.value.support = None;
        Ok(derived)
    }
}
