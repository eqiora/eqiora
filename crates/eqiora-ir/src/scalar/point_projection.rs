//! Checked structural projection at a typed point; Symbol dependencies stay live.
use super::*;
use eqiora_core::{ScalarDomain, ValueLiteral, ValueType};
use eqiora_schema::kernel::typing::{ExpressionType, RootContract, TypedResidual};
use eqiora_schema::kernel::{ExprDagBuilder, ExprId};

impl ScalarInputOperatorIr {
    /// Reuse a component program when each input is one whole real Semantic
    /// symbol. Coordinate pairs and shaped inputs keep their component owner.
    pub(crate) fn into_symbol_operator(
        self,
        source_values: Vec<Option<ScalarInputValueId>>,
    ) -> Result<ScalarOperatorIr, Diagnostic> {
        if !self.primal_checks.is_empty() {
            return Err(ir_builder_error(
                "scalar projection has unbound primal obligations",
            ));
        }
        let mut symbols = Vec::with_capacity(self.slots.len());
        let mut seen = std::collections::HashSet::with_capacity(self.slots.len());
        let mut inputs = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            let source = slot.source();
            if source.is_imaginary()
                || !source.component_index().is_empty()
                || !source.spatial_derivative_axes().is_empty()
                || !seen.insert(source.symbol())
            {
                return Err(ir_builder_error(
                    "scalar projection requires one whole real value per symbol",
                ));
            }
            symbols.push(source.symbol());
            inputs.push(
                ValueType::scalar(ScalarDomain::Real, slot.dimension())
                    .map_err(|error| ir_builder_error(error.to_string()))?,
            );
        }
        let operator = ScalarOperatorIr {
            typed_inputs: Some(inputs),
            source_values: source_values
                .into_iter()
                .map(|id| id.map(|id| id.0))
                .collect(),
            typed_constants: self.typed_constants,
            array_operands: Vec::new(),
            definitions: Vec::new(),
            symbols,
            instructions: self.instructions,
            roots: self.roots,
        };
        operator.projection_cost()?;
        Ok(operator)
    }
}

impl ScalarOperatorIr {
    pub(super) fn projection_cost(&self) -> Result<usize, Diagnostic> {
        self.instructions.iter().try_fold(
            slot_composition::relocation_work(&self.instructions)?,
            |cost, node| {
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
            },
        )
    }

    /// Project an admitted real scalar residual for numerical derivative products.
    /// Retained definition identity stays in the canonical typed residual; this
    /// derived graph uses the existing ordered scalar expansion and input slots.
    /// Conditional programs retain their input types so ordinary numerical
    /// evaluation and linearization use typed demand and active-branch rules.
    ///
    /// Complex intermediates use the shared component lowering; input symbols
    /// and roots remain real scalars.
    ///
    /// # Errors
    /// Rejects non-real or shaped inputs/roots, invalid typed applications, and the
    /// existing scalar projection resource bounds.
    pub fn lower_typed_scalar<I: Clone + Eq>(typed: &TypedResidual<I>) -> Result<Self, Diagnostic> {
        let mut pending = typed.expression().roots().to_vec();
        let mut visited = vec![false; typed.node_types().len()];
        while let Some(node) = pending.pop() {
            if std::mem::replace(&mut visited[node.index() as usize], true) {
                continue;
            }
            if typed.node_types()[node.index() as usize]
                .value_type
                .scalar_domain()
                == ScalarDomain::Complex
            {
                return crate::ComponentScalarization::lower_real_scalar(typed);
            }
            typed
                .expression()
                .node(node)
                .expect("typed node")
                .try_for_each_operand(|operand| {
                    pending.push(operand);
                    Ok::<(), Diagnostic>(())
                })?;
        }
        Self::project_typed_operator(typed, Self::lower(typed.expression())?)
    }

    pub(super) fn project_typed_operator<I>(
        typed: &TypedResidual<I>,
        operator: Self,
    ) -> Result<Self, Diagnostic> {
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
        let mut projected = if operator.definitions.is_empty() {
            operator
        } else {
            operator.project_types(&inputs, &operator.roots)?
        };
        if projected
            .instructions
            .iter()
            .any(Instruction::requires_typed_execution)
        {
            if inputs
                .iter()
                .any(|ty| ty.scalar_domain() != ScalarDomain::Real || !ty.shape().is_scalar())
            {
                return Err(ir_builder_error(
                    "typed scalar execution requires real scalar inputs",
                ));
            }
            projected.typed_inputs = Some(inputs);
        }
        Ok(projected)
    }

    pub(super) fn project_point(
        &self,
        inputs: &[ValueLiteral],
        roots: &[ValueId],
    ) -> Result<Self, Diagnostic> {
        let types = inputs
            .iter()
            .map(|value| value.value_type().clone())
            .collect::<Vec<_>>();
        self.project_types(&types, roots)
    }

    fn project_types(&self, inputs: &[ValueType], roots: &[ValueId]) -> Result<Self, Diagnostic> {
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
        if self.definitions.is_empty()
            && let Some(admitted) = &self.typed_inputs
        {
            if inputs != admitted
                || roots
                    .iter()
                    .any(|id| id.0 as usize >= self.instructions.len())
            {
                return Err(ir_builder_error(
                    "numeric projection differs from its admitted input types or roots",
                ));
            }
            let mut projected = self.clone();
            projected.roots = roots.to_vec();
            return Ok(projected);
        }
        let mut builder = ExprDagBuilder::new();
        let mut source = Vec::with_capacity(self.instructions.len());
        for instruction in &self.instructions {
            source.push(
                self.append_instruction(&mut builder, *instruction, |id| source[id.0 as usize])?,
            );
        }
        let dag = builder.finish(roots.iter().map(|id| source[id.0 as usize]))?;
        let types = self
            .symbols
            .iter()
            .copied()
            .zip(inputs)
            .map(|(symbol, value)| (symbol, ExpressionType::<()>::new(value.clone(), None)))
            .collect::<HashMap<_, _>>();
        let typed = TypedResidual::infer(
            dag,
            None,
            |_| None,
            RootContract::InitialResiduals,
            |symbol| types.get(&symbol).cloned().ok_or(()),
        )
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
        let mut builder = ScalarInputIrBuilder::new();
        builder.typed_constants = self.typed_constants.clone();
        let mut input_values = Vec::with_capacity(inputs.len());
        for (index, (&symbol, ty)) in self.symbols.iter().zip(inputs).enumerate() {
            let coordinate = ScalarSymbolCoordinate::for_value(symbol, ty)?.remove(0);
            input_values.push(builder.input(ScalarInputSlot::new(
                index as u32,
                coordinate,
                ty.dimension(),
            ))?);
        }
        let mut projected: Vec<ScalarInputValueId> = Vec::with_capacity(self.instructions.len());
        let mut array_operands = Vec::new();
        let mut work = self.projection_cost()?;
        let mut map_entries = HashMap::new();
        for (index, instruction) in self.instructions.iter().copied().enumerate() {
            let value = match instruction {
                Instruction::Read(slot) => input_values[slot.0 as usize],
                Instruction::PureOperator {
                    definition,
                    start,
                    len,
                } => {
                    let operands =
                        &self.array_operands[start as usize..start as usize + len as usize];
                    let arguments = operands
                        .iter()
                        .map(|id| source[id.0 as usize])
                        .collect::<Vec<_>>();
                    let expected = typed
                        .node_type(source[index])
                        .expect("checked application type");
                    let (scope, root) = crate::ComponentScalarization::lower_scalar_definition(
                        &self.definitions[definition as usize].1,
                        &typed,
                        &arguments,
                        expected,
                        &mut work,
                    )?;
                    let bindings = operands
                        .iter()
                        .zip(&arguments)
                        .map(|(id, argument)| {
                            (
                                projected[id.0 as usize],
                                typed
                                    .node_type(*argument)
                                    .expect("checked argument")
                                    .dimension(),
                            )
                        })
                        .collect::<Vec<_>>();
                    builder.compose_scoped(scope, &[root], &bindings, |cost| {
                        work = work
                            .checked_add(cost)
                            .filter(|work| *work <= 1_000_000)
                            .ok_or_else(|| {
                                ir_builder_error("typed projection exceeds the scalar work budget")
                            })?;
                        Ok(())
                    })?[0]
                }
                Instruction::Array { start, len } => {
                    let next = u32::try_from(array_operands.len()).map_err(|_| ir_size_error())?;
                    array_operands.extend(
                        self.array_operands[start as usize..start as usize + len as usize]
                            .iter()
                            .map(|id| projected[id.0 as usize].0),
                    );
                    builder.push(Instruction::Array { start: next, len })?
                }
                _ => {
                    let instruction = slot_composition::remap_instruction(
                        &self.instructions,
                        index,
                        &mut builder.instructions,
                        &mut map_entries,
                        |id| projected[id.0 as usize].0,
                    )?;
                    builder.push(instruction)?
                }
            };
            projected.push(value);
        }
        if !builder.primal_checks.is_empty() {
            return Err(ir_builder_error(
                "pure scalar projection has unbound primal obligations",
            ));
        }
        Ok(Self {
            typed_inputs: Some(inputs.to_vec()),
            source_values: self
                .source_values
                .iter()
                .map(|id| id.map(|id| projected[id.0 as usize].0))
                .collect(),
            typed_constants: builder.typed_constants,
            array_operands,
            definitions: Vec::new(),
            symbols: self.symbols.clone(),
            instructions: builder.instructions,
            roots: roots.iter().map(|id| projected[id.0 as usize].0).collect(),
        })
    }

    pub(super) fn append_instruction(
        &self,
        builder: &mut ExprDagBuilder,
        node: Instruction,
        at: impl Fn(ValueId) -> ExprId,
    ) -> Result<ExprId, Diagnostic> {
        match node {
            Instruction::Constant(value) => builder.constant(value),
            Instruction::TypedConstant(index) => {
                builder.constant(self.typed_constants[index as usize].clone())
            }
            Instruction::Read(slot) => builder.symbol(self.symbols[slot.0 as usize]),
            Instruction::Array { start, len } => builder.array(
                self.array_operands[start as usize..start as usize + len as usize]
                    .iter()
                    .map(|id| at(*id)),
            ),
            Instruction::Index(a, index) => builder.index(at(a), index),
            Instruction::Quotient(a, b) => builder.quotient(at(a), at(b)),
            Instruction::Remainder(a, b) => builder.remainder(at(a), at(b)),
            Instruction::ToReal(a) => builder.to_real(at(a)),
            Instruction::ToInteger(a) => builder.to_integer(at(a)),
            Instruction::Ordinal(a) => builder.ordinal(at(a)),
            Instruction::MapInvariant { .. }
            | Instruction::ComplexDiv { .. }
            | Instruction::CoefficientJet { .. } => Err(ir_builder_error(
                "numerical map admission cannot be projected into Semantic expressions",
            )),
            Instruction::Neg(a) => builder.neg(at(a)),
            Instruction::Add(a, b) => builder.add(at(a), at(b)),
            Instruction::Sub(a, b) => builder.sub(at(a), at(b)),
            Instruction::Mul(a, b) => builder.mul(at(a), at(b)),
            Instruction::Div(a, b) => builder.div(at(a), at(b)),
            Instruction::PowI(a, n) => builder.powi(at(a), n),
            Instruction::Math(function, a) => builder.unary_math(function, at(a)),
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
            Instruction::DerivativeLimit {
                start,
                len,
                value,
                order,
            } => builder.derivative_limit(
                derivative_limit::inputs(start, len, self.instructions.len())?
                    .map(|input| at(ValueId(input as u32)))
                    .collect(),
                at(value),
                order,
            ),
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
        }
    }
}

#[cfg(test)]
mod tests;
