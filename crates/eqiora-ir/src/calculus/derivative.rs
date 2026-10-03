//! Ordered formal differentiation delegates to the schema-owned transform.
use eqiora_core::{ScalarDomain, ValueLiteral, ValueType};
use eqiora_schema::kernel::typing::ExpressionType;
use eqiora_schema::kernel::{ExprDagBuilder, ExprId};

use super::{
    CalculusBuilder, CalculusError, CalculusNode, PureValueClass, ScalarCalculus,
    ScalarCalculusNode,
};

impl<I: Clone> ScalarCalculus<I> {
    /// Append an ordered first or second formal partial as an ordinary DAG.
    /// Other inputs are held fixed. Only reachable derivative work is appended;
    /// dimensions, product-rule order and input occurrence identities are retained.
    ///
    /// # Errors
    /// Rejects non-real/shaped inputs or results, unsupported orders, invalid
    /// argument IDs, and unrepresentable dimensions or calculus resource bounds.
    pub fn partial(
        &self,
        destination: &mut ExprDagBuilder,
        arguments: &[ExprId],
        formal: u16,
        order: u8,
    ) -> Result<(ExprId, ExpressionType<I>), CalculusError> {
        let selected = self
            .argument_types
            .get(usize::from(formal))
            .ok_or(CalculusError::InvalidFormal(formal))?;
        if !matches!(order, 1 | 2)
            || arguments.len() != self.argument_types.len()
            || self
                .argument_types
                .iter()
                .chain([&self.result_type])
                .any(|ty| {
                    ty.value_type.scalar_domain() != ScalarDomain::Real || !ty.shape().is_scalar()
                })
        {
            return Err(CalculusError::UnsupportedDerivative);
        }
        for argument in arguments {
            destination
                .validate_prior_operand(*argument)
                .map_err(projection)?;
        }
        let dimension = selected
            .dimension()
            .pow(i32::from(order), 1)
            .and_then(|denominator| self.result_type.dimension().div(denominator))
            .ok_or(CalculusError::UnsupportedDerivative)?;
        let class = |dimension| {
            PureValueClass::invariant_scalar()
                .with_dimension(dimension)
                .with_scalar_domain(ScalarDomain::Real)
        };
        let formals = self
            .argument_types
            .iter()
            .map(|ty| class(ty.dimension()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut builder = CalculusBuilder::new(formals, class(dimension)?)?;
        for node in &self.nodes {
            builder.push(match node {
                ScalarCalculusNode::Rational { value, dimension } => CalculusNode::Rational {
                    value: *value,
                    dimension: *dimension,
                },
                ScalarCalculusNode::FormalComponent(atom) => CalculusNode::FormalComponent {
                    formal: atom.formal(),
                    axes: Box::new([]),
                },
                ScalarCalculusNode::Differentiated { value, source, wrt } => {
                    CalculusNode::Differentiated {
                        value: *value,
                        source: *source,
                        wrt: *wrt,
                    }
                }
                ScalarCalculusNode::BoundInput(value) => CalculusNode::BoundInput(*value),
                ScalarCalculusNode::Neg(value) => CalculusNode::Neg(*value),
                ScalarCalculusNode::Add(left, right) => CalculusNode::Add(*left, *right),
                ScalarCalculusNode::Mul(left, right) => CalculusNode::Mul(*left, *right),
            })?;
        }
        let mut root = self.root;
        for _ in 0..order {
            root = builder.partial(root, formal)?;
        }
        let definition = builder.finish(root)?;
        let mut plan = PartialPlan::default();
        let mut mapped = Vec::with_capacity(definition.nodes().len());
        for node in definition.nodes() {
            let get = |id: super::CalculusNodeId| mapped[id.index() as usize];
            let next = match node {
                CalculusNode::Rational { value, dimension } => plan.push(PartialNode::Constant(
                    ValueLiteral::from_real(
                        ValueType::scalar(ScalarDomain::Real, *dimension)
                            .map_err(|_| CalculusError::UnsupportedDerivative)?,
                        value.as_f64(),
                    )
                    .map_err(|error| CalculusError::DerivativeProjection(error.to_string()))?,
                )),
                CalculusNode::FormalComponent { formal, .. } => {
                    plan.push(PartialNode::Argument(arguments[usize::from(*formal)]))
                }
                CalculusNode::BoundInput(value) | CalculusNode::Differentiated { value, .. } => {
                    get(*value)
                }
                CalculusNode::Neg(value) => plan.push(PartialNode::Neg(get(*value))),
                CalculusNode::Add(left, right) => {
                    plan.push(PartialNode::Add(get(*left), get(*right)))
                }
                CalculusNode::Mul(left, right) => {
                    plan.push(PartialNode::Mul(get(*left), get(*right)))
                }
                _ => return Err(CalculusError::UnsupportedDerivative),
            };
            mapped.push(next);
        }
        let root = mapped[definition.root().index() as usize];
        Ok((
            plan.append(root, destination)?,
            ExpressionType::scalar(dimension, self.result_type.support.clone()),
        ))
    }
}

fn projection(error: eqiora_core::Diagnostic) -> CalculusError {
    CalculusError::DerivativeProjection(error.to_string())
}

#[derive(Default)]
struct PartialPlan {
    nodes: Vec<PartialNode>,
}

enum PartialNode {
    Argument(ExprId),
    Constant(ValueLiteral),
    Neg(usize),
    Add(usize, usize),
    Mul(usize, usize),
}

impl PartialPlan {
    fn push(&mut self, node: PartialNode) -> usize {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }
    fn append(self, root: usize, builder: &mut ExprDagBuilder) -> Result<ExprId, CalculusError> {
        let mut reachable = vec![false; self.nodes.len()];
        let mut pending = vec![root];
        while let Some(id) = pending.pop() {
            if std::mem::replace(&mut reachable[id], true) {
                continue;
            }
            match self.nodes[id] {
                PartialNode::Neg(a) => pending.push(a),
                PartialNode::Add(a, b) | PartialNode::Mul(a, b) => pending.extend([a, b]),
                _ => {}
            }
        }
        let mut mapped = vec![None; self.nodes.len()];
        for (id, node) in self.nodes.into_iter().enumerate() {
            if !reachable[id] {
                continue;
            }
            let operand = |id: usize| mapped[id].expect("reachable prior derivative operand");
            mapped[id] = Some(match node {
                PartialNode::Argument(id) => id,
                PartialNode::Constant(value) => builder.constant(value).map_err(projection)?,
                PartialNode::Neg(a) => builder.neg(operand(a)).map_err(projection)?,
                PartialNode::Add(a, b) => {
                    builder.add(operand(a), operand(b)).map_err(projection)?
                }
                PartialNode::Mul(a, b) => {
                    builder.mul(operand(a), operand(b)).map_err(projection)?
                }
            });
        }
        Ok(mapped[root].expect("reachable derivative root"))
    }
}
