//! Demand-driven evaluation; each coordinate binding owns a fresh value cache.
use super::*;
mod jacobian;

pub(super) fn evaluate_selected(
    owner: RawId,
    expression: &ExprDag,
    roots: &[ExprId],
    resolve: &mut dyn FnMut(SymbolRef) -> Option<ValueLiteral>,
) -> Result<Vec<ValueLiteral>, Diagnostic> {
    let mut resolve = |input, _: Option<&EvaluationPoint>| {
        let EvaluationInput::Value(symbol) = input else {
            return Err(Diagnostic::error(
                codes::NOT_IMPLEMENTED,
                "coordinate partial requires an admitted point reconstruction",
            ));
        };
        resolve(symbol).ok_or_else(|| {
            Diagnostic::error(
                codes::MISSING_EXECUTION_INPUT,
                format!("no reference-execution value is available for {symbol:?}"),
            )
        })
    };
    Evaluator {
        owner,
        expression,
        program: None,
        resolve: &mut resolve,
        component_work: 0,
        point_work: 0,
    }
    .selected(roots, None, 0)
}

pub(crate) fn evaluate_with_points(
    program: &KernelProgram,
    owner: RawId,
    expression: &ExprDag,
    point: Option<&EvaluationPoint>,
    resolve: &mut Resolver<'_>,
) -> Result<Vec<ValueLiteral>, Diagnostic> {
    Evaluator {
        owner,
        expression,
        program: (point.is_some()
            || expression.nodes().iter().any(|node| {
                matches!(
                    node,
                    ExprNode::Evaluate { .. }
                        | ExprNode::Pullback { .. }
                        | ExprNode::CoordinateMapFactor { .. }
                )
            }))
        .then_some(program),
        resolve,
        component_work: 0,
        point_work: 0,
    }
    .selected(expression.roots(), point, 0)
}

struct Evaluator<'a, 'r> {
    owner: RawId,
    expression: &'a ExprDag,
    program: Option<&'a KernelProgram>,
    resolve: &'r mut Resolver<'r>,
    component_work: usize,
    point_work: usize,
}

impl Evaluator<'_, '_> {
    fn selected(
        &mut self,
        roots: &[ExprId],
        point: Option<&EvaluationPoint>,
        depth: usize,
    ) -> Result<Vec<ValueLiteral>, Diagnostic> {
        let owner = self.owner;
        let expression = self.expression;
        if self.program.is_some() {
            self.point_work = self
                .point_work
                .checked_add(expression.nodes().len())
                .filter(|work| *work <= 1_000_000)
                .ok_or_else(component_budget_error)?;
            if depth > 128 {
                return Err(component_budget_error());
            }
        }
        enum Frame {
            Demand(ExprId),
            Apply(ExprId),
            Logical(ExprId),
            Branch(ExprId),
        }
        let mut values = vec![None; expression.nodes().len()];
        for &root in roots {
            let mut pending = vec![Frame::Demand(root)];
            while let Some(frame) = pending.pop() {
                let id = match frame {
                    Frame::Demand(id)
                    | Frame::Apply(id)
                    | Frame::Logical(id)
                    | Frame::Branch(id) => id,
                };
                let index = id.index() as usize;
                let Some(node) = expression.nodes().get(index) else {
                    return Err(Diagnostic::error(
                        codes::INVALID_EXPRESSION_DAG,
                        "requested expression root is unavailable",
                    ));
                };
                if values[index].is_some() {
                    continue;
                }
                if point.is_some_and(|point| point.side().is_some()) {
                    super::point::require_side_regularity(expression, node)?;
                }
                if matches!(
                    node,
                    ExprNode::Gradient(_)
                        | ExprNode::Divergence(_)
                        | ExprNode::Trace { .. }
                        | ExprNode::NormalComponent { .. }
                ) {
                    let program = self.program.ok_or_else(|| {
                        Diagnostic::error(
                            codes::NOT_IMPLEMENTED,
                            "analytic spatial evaluation requires its exact KernelProgram",
                        )
                    })?;
                    let point = point.ok_or_else(|| {
                        Diagnostic::error(
                            codes::NOT_IMPLEMENTED,
                            "analytic spatial evaluation requires an exact point",
                        )
                    })?;
                    values[index] = Some(super::spatial::evaluate(
                        program,
                        owner,
                        expression,
                        id,
                        point,
                        &mut self.component_work,
                        self.resolve,
                    )?);
                    continue;
                }
                if let ExprNode::CoordinateMapFactor { factor, source, at } = node {
                    values[index] = Some(self.map_factor(id, *factor, source, at, point)?);
                    continue;
                }
                let binding = match node {
                    ExprNode::Evaluate { value, at, side } => Some((*value, at, *side)),
                    ExprNode::Pullback { value, source, at } => {
                        let source_point = point.ok_or_else(|| {
                            Diagnostic::error(
                                codes::MISSING_EXECUTION_INPUT,
                                "coordinate pullback requires an exact source point",
                            )
                        })?;
                        if source_point.side().is_some() {
                            return Err(Diagnostic::error(
                                codes::NOT_IMPLEMENTED,
                                "one-sided coordinate pullback requires an admitted orientation transform",
                            ));
                        }
                        for selector in source {
                            let Some(ExprNode::Symbol(SymbolRef::Coordinate {
                                support,
                                factor,
                                axis,
                            })) = expression.node(*selector)
                            else {
                                return Err(Diagnostic::error(
                                    codes::INVALID_EXPRESSION_DAG,
                                    "coordinate pullback source selector is not a coordinate",
                                ));
                            };
                            source_point.coordinate(*support, *factor, *axis)?;
                        }
                        Some((*value, at, None))
                    }
                    _ => None,
                };
                if let Some((value, at, side)) = binding {
                    let program = self.program.ok_or_else(|| {
                        Diagnostic::error(
                            codes::NOT_IMPLEMENTED,
                            "point evaluation requires its exact KernelProgram",
                        )
                    })?;
                    let roots = at.iter().map(|(_, value)| *value).collect::<Vec<_>>();
                    let points = self.selected(&roots, point, depth + 1)?;
                    let bound = EvaluationPoint::bind(program, expression, at, &points, side)?;
                    let mut evaluated = self.selected(&[value], Some(&bound), depth + 1)?;
                    values[index] = evaluated.pop();
                    continue;
                }
                if let ExprNode::CoordinatePartial { value, wrt } = node {
                    let Some(point) = point else {
                        return Err(Diagnostic::error(
                            codes::NOT_IMPLEMENTED,
                            "coordinate partial requires an admitted point reconstruction",
                        ));
                    };
                    let (
                        Some(ExprNode::Symbol(SymbolRef::Field(field))),
                        Some(ExprNode::Symbol(SymbolRef::Coordinate {
                            support,
                            factor,
                            axis,
                        })),
                    ) = (expression.node(*value), expression.node(*wrt))
                    else {
                        return Err(Diagnostic::error(
                            codes::NOT_IMPLEMENTED,
                            "point partial requires a Field and exact coordinate selector",
                        ));
                    };
                    point.coordinate(*support, *factor, *axis)?;
                    values[index] = Some((self.resolve)(
                        EvaluationInput::CoordinatePartial {
                            field: *field,
                            factor: *factor,
                            axis: *axis,
                        },
                        Some(point),
                    )?);
                    continue;
                }
                if let ExprNode::Select { condition, .. } | ExprNode::Require { condition, .. } =
                    node
                {
                    if matches!(frame, Frame::Demand(_)) {
                        pending.push(Frame::Branch(id));
                        pending.push(Frame::Demand(*condition));
                        continue;
                    }
                    if matches!(frame, Frame::Branch(_)) {
                        let condition = boolean(operand(&values, *condition, owner)?)?;
                        let selected = match node {
                            ExprNode::Select {
                                then_value,
                                else_value,
                                ..
                            } => {
                                if condition {
                                    *then_value
                                } else {
                                    *else_value
                                }
                            }
                            ExprNode::Require { value, .. } if condition => *value,
                            _ => {
                                return Err(Diagnostic::error(
                                    codes::NONFINITE_EVALUATION,
                                    "required expression domain condition is false",
                                ));
                            }
                        };
                        pending.push(Frame::Apply(id));
                        pending.push(Frame::Demand(selected));
                        continue;
                    }
                }
                if let ExprNode::And(left, right) | ExprNode::Or(left, right) = node {
                    if matches!(frame, Frame::Demand(_)) {
                        pending.push(Frame::Logical(id));
                        pending.push(Frame::Demand(*left));
                        continue;
                    }
                    if matches!(frame, Frame::Logical(_)) {
                        let left = boolean(operand(&values, *left, owner)?)?;
                        if matches!(node, ExprNode::And(_, _)) && !left
                            || matches!(node, ExprNode::Or(_, _)) && left
                        {
                            values[index] = Some(ValueLiteral::boolean(left));
                        } else {
                            pending.push(Frame::Apply(id));
                            pending.push(Frame::Demand(*right));
                        }
                        continue;
                    }
                }
                if matches!(frame, Frame::Demand(_)) {
                    pending.push(Frame::Apply(id));
                    match node {
                        ExprNode::Constant(_) | ExprNode::Symbol(_) => {}
                        ExprNode::PureOperatorApplication(application) => {
                            check_component_work(
                                self.component_work,
                                application.arguments().len(),
                            )?;
                            pending.extend(
                                application
                                    .arguments()
                                    .iter()
                                    .rev()
                                    .copied()
                                    .map(Frame::Demand),
                            );
                        }
                        ExprNode::Array { elements } => {
                            check_component_work(self.component_work, elements.len())?;
                            pending.extend(elements.iter().rev().copied().map(Frame::Demand));
                        }
                        ExprNode::UnaryMath(_, value)
                        | ExprNode::FiniteUnary(_, value)
                        | ExprNode::Index { value, .. }
                        | ExprNode::Sample { value, .. }
                        | ExprNode::Hold(value)
                        | ExprNode::Neg(value)
                        | ExprNode::PowI(value, _)
                        | ExprNode::ToReal(value)
                        | ExprNode::ToInteger(value)
                        | ExprNode::Ordinal(value)
                        | ExprNode::Not(value) => pending.push(Frame::Demand(*value)),
                        ExprNode::Complex { real: a, imag: b }
                        | ExprNode::Compare(_, a, b)
                        | ExprNode::FiniteBinary(_, a, b)
                        | ExprNode::Add(a, b)
                        | ExprNode::Sub(a, b)
                        | ExprNode::Mul(a, b)
                        | ExprNode::Div(a, b)
                        | ExprNode::Quotient(a, b)
                        | ExprNode::Remainder(a, b) => {
                            pending.push(Frame::Demand(*b));
                            pending.push(Frame::Demand(*a));
                        }
                        _ => {
                            return Err(Diagnostic::error(
                                codes::NOT_IMPLEMENTED,
                                "expression node is outside the reference execution profile",
                            ));
                        }
                    }
                    continue;
                }
                let value = match node {
                    ExprNode::Select {
                        condition,
                        then_value,
                        else_value,
                    } => {
                        let selected = if boolean(operand(&values, *condition, owner)?)? {
                            then_value
                        } else {
                            else_value
                        };
                        operand(&values, *selected, owner)?.clone()
                    }
                    ExprNode::Require { value, .. } => operand(&values, *value, owner)?.clone(),
                    ExprNode::UnaryMath(function, value) => {
                        function.evaluate(operand(&values, *value, owner)?)?
                    }

                    ExprNode::FiniteUnary(operation, value) => {
                        finite::unary(*operation, operand(&values, *value, owner)?)?
                    }
                    ExprNode::FiniteBinary(operation, left, right) => finite::binary(
                        *operation,
                        operand(&values, *left, owner)?,
                        operand(&values, *right, owner)?,
                        &mut self.component_work,
                    )?,
                    ExprNode::PureOperatorApplication(application) => {
                        let definition = expression
                            .definition(application.definition())
                            .expect("checked definition");
                        let arguments = application
                            .arguments()
                            .iter()
                            .map(|id| operand(&values, *id, owner))
                            .collect::<Result<Vec<_>, _>>()?;
                        pure::evaluate(owner, definition, &arguments, &mut self.component_work)?
                    }
                    ExprNode::Array { elements } => {
                        let elements = elements
                            .iter()
                            .map(|id| operand(&values, *id, owner))
                            .collect::<Result<Vec<_>, _>>()?;
                        for element in &elements {
                            require_channels(element.value_type())?;
                        }
                        let types = elements
                            .iter()
                            .map(|value| {
                                eqiora_schema::kernel::typing::ExpressionType::<()>::new(
                                    value.value_type().clone(),
                                    None,
                                )
                            })
                            .collect::<Vec<_>>();
                        let ty = eqiora_schema::kernel::typing::ExpressionType::array(&types)
                            .map_err(|error| {
                                Diagnostic::error(codes::INVALID_EXPRESSION_DAG, error.to_string())
                            })?;
                        let count = ty
                            .value_type
                            .shape()
                            .component_count()
                            .expect("checked type");
                        check_component_work(self.component_work, count)?;
                        ValueLiteral::array(&elements).map_err(discrete_error)?
                    }
                    ExprNode::Index { value, index } => {
                        let value = operand(&values, *value, owner)?;
                        require_channels(value.value_type())?;
                        let ty = eqiora_schema::kernel::typing::ExpressionType::<()>::new(
                            value.value_type().clone(),
                            None,
                        )
                        .index(*index)
                        .map_err(|error| {
                            Diagnostic::error(codes::INVALID_EXPRESSION_DAG, error.to_string())
                        })?;
                        check_component_work(
                            self.component_work,
                            ty.value_type
                                .shape()
                                .component_count()
                                .expect("checked type"),
                        )?;
                        value.index(*index).map_err(discrete_error)?
                    }
                    ExprNode::Not(value) => {
                        ValueLiteral::boolean(!boolean(operand(&values, *value, owner)?)?)
                    }
                    ExprNode::And(_, right) | ExprNode::Or(_, right) => {
                        ValueLiteral::boolean(boolean(operand(&values, *right, owner)?)?)
                    }
                    ExprNode::Compare(op, left, right) => compare(
                        *op,
                        operand(&values, *left, owner)?,
                        operand(&values, *right, owner)?,
                    )?,
                    ExprNode::Constant(value) => {
                        check_component_work(self.component_work, value.component_count())?;
                        value.clone()
                    }
                    ExprNode::Symbol(symbol) => {
                        if let (
                            Some(point),
                            SymbolRef::Coordinate {
                                support,
                                factor,
                                axis,
                            },
                        ) = (point, symbol)
                        {
                            point.coordinate(*support, *factor, *axis)?
                        } else {
                            (self.resolve)(EvaluationInput::Value(*symbol), point).map_err(
                                |error| error.with_graph_path(expression_path(owner, index)),
                            )?
                        }
                    }
                    ExprNode::Sample { value, .. } | ExprNode::Hold(value) => {
                        operand(&values, *value, owner)?.clone()
                    }
                    ExprNode::Neg(value) => {
                        let value = operand(&values, *value, owner)?;
                        if value.value_type().scalar_domain() == ScalarDomain::Integer {
                            require_scalar_arithmetic(value)?;
                            value.checked_neg().map_err(discrete_error)?
                        } else {
                            numeric::unary(node, value)?
                        }
                    }
                    ExprNode::Add(left, right)
                    | ExprNode::Sub(left, right)
                    | ExprNode::Mul(left, right)
                    | ExprNode::Div(left, right)
                    | ExprNode::Complex {
                        real: left,
                        imag: right,
                    } => {
                        let left = operand(&values, *left, owner)?;
                        let right = operand(&values, *right, owner)?;
                        if left.value_type().scalar_domain() == ScalarDomain::Integer {
                            require_scalar_arithmetic(left)?;
                            require_scalar_arithmetic(right)?;
                            match node {
                                ExprNode::Add(..) => left.checked_add(right),
                                ExprNode::Sub(..) => left.checked_sub(right),
                                ExprNode::Mul(..) => left.checked_mul(right),
                                _ => Err(eqiora_core::InvalidValueLiteral::ScalarDomain),
                            }
                            .map_err(discrete_error)?
                        } else {
                            numeric::binary(node, left, right)?
                        }
                    }
                    ExprNode::Quotient(a, b) => operand(&values, *a, owner)?
                        .checked_quotient(operand(&values, *b, owner)?)
                        .map_err(discrete_error)?,
                    ExprNode::Remainder(a, b) => operand(&values, *a, owner)?
                        .checked_remainder(operand(&values, *b, owner)?)
                        .map_err(discrete_error)?,
                    ExprNode::ToReal(value) => operand(&values, *value, owner)?
                        .to_real()
                        .map_err(discrete_error)?,
                    ExprNode::Ordinal(value) => operand(&values, *value, owner)?
                        .ordinal()
                        .map_err(discrete_error)?,
                    ExprNode::ToInteger(value) => operand(&values, *value, owner)?
                        .to_integer()
                        .map_err(discrete_error)?,
                    ExprNode::PowI(base, _) => {
                        numeric::unary(node, operand(&values, *base, owner)?)?
                    }
                    _ => {
                        return Err(Diagnostic::error(
                            codes::NOT_IMPLEMENTED,
                            "expression node is newer than this reference interpreter",
                        )
                        .with_graph_path(expression_path(owner, index)));
                    }
                };
                check_component_work(self.component_work, value.component_count())?;
                self.component_work += value.component_count();
                values[index] = Some(value);
            }
        }

        roots
            .iter()
            .map(|root| operand(&values, *root, owner).cloned())
            .collect()
    }
}
