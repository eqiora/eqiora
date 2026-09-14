//! Checked structural projection at a typed point; Symbol dependencies stay live.
use super::*;
use eqiora_core::{ScalarDomain, ValueLiteral, ValueType};
use eqiora_schema::kernel::typing::{ExpressionType, RootContract, TypedResidual};
use eqiora_schema::kernel::{ExprDagBuilder, ExprId, UnaryMathFunction};

impl ScalarOperatorIr {
    pub(super) fn projection_cost(&self) -> Result<usize, Diagnostic> {
        self.instructions
            .iter()
            .try_fold(self.instructions.len(), |cost, node| {
                let extra = match node {
                    Instruction::PureOperator { definition, .. } => {
                        self.definitions[*definition as usize].1.nodes().len()
                    }
                    _ => 0,
                };
                cost.checked_add(extra)
                    .filter(|n| *n <= 1_000_000)
                    .ok_or_else(|| {
                        ir_builder_error("typed projection exceeds the scalar work budget")
                    })
            })
    }

    /// Project an admitted real scalar residual for numerical derivative products.
    /// Retained definition identity stays in the canonical typed residual; this
    /// derived graph uses the existing ordered scalar expansion and input slots.
    ///
    /// # Errors
    /// Rejects non-real or shaped values, invalid typed applications, and the
    /// existing scalar projection resource bounds.
    pub fn lower_typed_scalar<I>(typed: &TypedResidual<I>) -> Result<Self, Diagnostic> {
        let operator = Self::lower(typed.expression())?;
        if operator.definitions.is_empty() {
            return Ok(operator);
        }
        let types = typed
            .expression()
            .nodes()
            .iter()
            .zip(typed.node_types())
            .filter_map(|(node, ty)| match node {
                eqiora_schema::kernel::ExprNode::Symbol(symbol) => {
                    Some((*symbol, ty.value_type.clone()))
                }
                _ => None,
            })
            .collect::<HashMap<_, _>>();
        let inputs = operator
            .symbols
            .iter()
            .map(|symbol| {
                types
                    .get(symbol)
                    .cloned()
                    .ok_or_else(|| ir_builder_error("typed numerical symbol is unavailable"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        operator
            .project_types(&inputs, &operator.roots)
            .map(|(projected, _)| projected)
    }

    pub(super) fn project_point(
        &self,
        inputs: &[ValueLiteral],
        roots: &[ValueId],
    ) -> Result<(Self, Vec<ExprId>), Diagnostic> {
        let types = inputs
            .iter()
            .map(|value| value.value_type().clone())
            .collect::<Vec<_>>();
        self.project_types(&types, roots)
    }

    fn project_types(
        &self,
        inputs: &[ValueType],
        roots: &[ValueId],
    ) -> Result<(Self, Vec<ExprId>), Diagnostic> {
        self.projection_cost()?;
        if inputs.len() != self.symbols.len()
            || inputs.iter().any(|value| {
                value.scalar_domain() != ScalarDomain::Real || !value.shape().is_scalar()
            })
        {
            return Err(ir_builder_error(
                "typed scalar point requires one finite real scalar per symbol",
            ));
        }
        let mut builder = ExprDagBuilder::new();
        let mut source = Vec::with_capacity(self.instructions.len());
        for instruction in &self.instructions {
            source.push(self.append_instruction(&mut builder, *instruction, &source)?);
        }
        let dag = builder.finish(roots.iter().map(|id| source[id.0 as usize]))?;
        let types = self
            .symbols
            .iter()
            .copied()
            .zip(inputs)
            .map(|(symbol, value)| (symbol, ExpressionType::<()>::new(value.clone(), None)))
            .collect::<HashMap<_, _>>();
        let typed = TypedResidual::infer(dag, None, RootContract::InitialResiduals, |symbol| {
            types.get(&symbol).cloned().ok_or(())
        })
        .map_err(|_| {
            ir_builder_error(
                "typed scalar point has incompatible expression types or branch supports",
            )
        })?;
        for root in roots {
            if typed.node_type(source[root.0 as usize]).is_none_or(|ty| {
                ty.value_type.scalar_domain() != ScalarDomain::Real || !ty.shape().is_scalar()
            }) {
                return Err(ir_builder_error(
                    "typed scalar point requires real scalar outputs",
                ));
            }
        }
        let mut builder = ExprDagBuilder::new();
        let mut projected = Vec::with_capacity(self.instructions.len());
        for instruction in &self.instructions {
            let result = if let Instruction::PureOperator {
                definition,
                start,
                len,
            } = *instruction
            {
                let operands = &self.array_operands[start as usize..start as usize + len as usize];
                let args = operands
                    .iter()
                    .map(|id| {
                        typed
                            .node_type(source[id.0 as usize])
                            .expect("checked argument")
                            .clone()
                    })
                    .collect::<Vec<_>>();
                let instance = self.definitions[definition as usize]
                    .1
                    .instantiate(&args)
                    .map_err(|error| ir_builder_error(error.to_string()))?;
                let ids = operands
                    .iter()
                    .map(|id| projected[id.0 as usize])
                    .collect::<Vec<_>>();
                builder.project_scalar_operator(&instance, &ids, 1_000_000)?
            } else {
                self.append_instruction(&mut builder, *instruction, &projected)?
            };
            projected.push(result);
        }
        let roots = roots
            .iter()
            .map(|id| projected[id.0 as usize])
            .collect::<Vec<_>>();
        let dag = builder.finish(roots.iter().copied())?;
        let projected = Self::lower(&dag)?;
        if projected.symbols != self.symbols {
            return Err(ir_builder_error(
                "scalar projection changed input identity/order",
            ));
        }
        Ok((projected, roots))
    }

    fn append_instruction(
        &self,
        builder: &mut ExprDagBuilder,
        node: Instruction,
        ids: &[ExprId],
    ) -> Result<ExprId, Diagnostic> {
        let at = |id: ValueId| ids[id.0 as usize];
        match node {
            Instruction::Constant(value) => builder.constant(value),
            Instruction::TypedConstant(index) => {
                builder.constant(self.typed_constants[index as usize].clone())
            }
            Instruction::Read(slot) => builder.symbol(self.symbols[slot.0 as usize]),
            Instruction::Neg(a) => builder.neg(at(a)),
            Instruction::Add(a, b) => builder.add(at(a), at(b)),
            Instruction::Sub(a, b) => builder.sub(at(a), at(b)),
            Instruction::Mul(a, b) => builder.mul(at(a), at(b)),
            Instruction::Div(a, b) => builder.div(at(a), at(b)),
            Instruction::PowI(a, n) => builder.powi(at(a), n),
            Instruction::Sin(a) => builder.unary_math(UnaryMathFunction::Sin, at(a)),
            Instruction::Sqrt(a) => builder.unary_math(UnaryMathFunction::Sqrt, at(a)),
            Instruction::Compare(op, a, b) => builder.compare(op, at(a), at(b)),
            Instruction::Not(a) => builder.not(at(a)),
            Instruction::And(a, b) => builder.and(at(a), at(b)),
            Instruction::Or(a, b) => builder.or(at(a), at(b)),
            Instruction::Select {
                condition,
                then_value,
                else_value,
            } => builder.select(at(condition), at(then_value), at(else_value)),
            Instruction::Require { condition, value } => builder.require(at(condition), at(value)),
            Instruction::PureOperator {
                definition,
                start,
                len,
            } => builder.pure_operator(
                &self.definitions[definition as usize].1,
                self.array_operands[start as usize..start as usize + len as usize]
                    .iter()
                    .map(|id| at(*id)),
            ),
            _ => Err(ir_builder_error(
                "typed property projection requires scalar arithmetic, predicates and retained scalar operators",
            )),
        }
    }
}
