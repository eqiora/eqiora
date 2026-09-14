mod property;
mod pure_operator;
mod record;
pub(super) use record::lower_record;
mod source;
pub(super) use source::from_source;
mod contextual;
mod enumeration;
mod event;
mod law;
mod observable;
mod partial;
pub(super) use observable::lower_observable;
mod physical_accessors;
pub(crate) use partial::result_type as partial_result_type;
mod piecewise;
use super::*;
pub(super) use event::lower_event_guard;
pub(super) use law::lower_law;

use eqiora_schema::kernel::typing::{self, ExpressionType, SpatialSupport};

pub(super) struct LoweredRelation {
    pub(super) expression: ExprDag,
    pub(super) dependencies: BTreeSet<RawId>,
    pub(super) ports: BTreeSet<RawId>,
}

pub(super) fn lower_relation(
    file: &str,
    range: TextRange,
    activation: &ActivationSyntax,
    domain: Option<&str>,
    equations: &[LoweringEquation],
    initial: bool,
    bindings: &BTreeMap<String, Binding>,
) -> Result<LoweredRelation, Diagnostic> {
    if let Some(domain) = domain {
        match bindings.get(domain) {
            Some(Binding::Domain(_, DomainContract::Spatial { .. })) => {}
            Some(Binding::Domain(_, DomainContract::ScalarPhysical { .. })) => {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    range,
                    "spatial Relation scope cannot be a scalar physical Domain",
                ));
            }
            Some(_) | None => {
                return Err(unresolved(file, range, domain, "Relation Domain"));
            }
        }
    }
    if let ActivationSyntax::Named(clock) = activation {
        if !matches!(
            bindings.get(clock),
            Some(Binding::Clock(_, _) | Binding::Event(_))
        ) {
            return Err(unresolved(file, range, clock, "ClockDomain or Event"));
        }
    } else if !matches!(activation, ActivationSyntax::Continuous) {
        return Err(source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            file,
            range,
            "Activation syntax is newer than this compiler",
        ));
    }

    let support = domain
        .map(|name| relation_support(file, range, name, bindings))
        .transpose()?;
    let discrete = matches!(activation, ActivationSyntax::Named(_));
    let mut lowerer = ExpressionLowerer {
        file,
        bindings,
        builder: ExprDagBuilder::new(),
        dependencies: BTreeSet::new(),
        ports: BTreeSet::new(),
        cache: HashMap::new(),
        sampling: false,
        allow_discrete_symbols: discrete || initial,
        activation,
        initial,
    };
    let mut normalized = Vec::with_capacity(equations.len());
    for equation in equations {
        if initial && equation.kind != eqiora_schema::kernel::RelationConditionKind::Equality {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                equation.range,
                "initial blocks admit only equality conditions",
            ));
        }
        if equation.kind == eqiora_schema::kernel::RelationConditionKind::Complementarity {
            let mut operands = Vec::new();
            let mut types = Vec::new();
            for predicate in [&equation.left, &equation.right] {
                let operand = constraints::lowered_operand(predicate).map_err(|message| {
                    source_error(codes::LANGUAGE_TYPE_ERROR, file, predicate.range(), message)
                })?;
                expression_type(file, predicate, bindings, support.as_ref())?;
                types.push(expression_type(file, &operand, bindings, support.as_ref())?);
                operands.push(operand);
            }
            let value = eqiora_schema::kernel::RelationConditionKind::Complementarity
                .check_operands(&types[0], &types[1])
                .map_err(|error| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        file,
                        equation.range,
                        error.to_string(),
                    )
                })?;
            typing::residual(&value, support.as_ref()).map_err(|error| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    equation.range,
                    error.to_string(),
                )
            })?;
            normalized.extend(operands);
            continue;
        }

        let (left_expression, right_expression) = contextual::equation(
            file,
            &equation.left,
            &equation.right,
            bindings,
            support.as_ref(),
        )?;
        let left_type = expression_type(file, &left_expression, bindings, support.as_ref())?;
        let right_type = expression_type(file, &right_expression, bindings, support.as_ref())?;
        let checked = equality::check(
            left_type,
            right_type,
            equation.contextual_left_zero,
            equation.contextual_right_zero,
        )
        .map_err(|error| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                equation.range,
                error.to_string(),
            )
        })?;
        equation
            .kind
            .check_operands(&checked.left, &checked.right)
            .map_err(|error| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    equation.range,
                    error.to_string(),
                )
            })?;
        typing::residual(
            &checked.equation_type,
            if initial {
                checked.equation_type.support.as_ref()
            } else {
                support.as_ref()
            },
        )
        .map_err(|error| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                equation.range,
                error.to_string(),
            )
        })?;
        let contextual =
            |expression: &LoweringExpression, is_zero, value_type: &eqiora_core::ValueType| {
                if is_zero {
                    LoweringExpression::literal(
                        if value_type.scalar_domain() == eqiora_core::ScalarDomain::Integer {
                            eqiora_core::ValueLiteral::from_integer(value_type.clone(), 0)
                        } else {
                            eqiora_core::ValueLiteral::from_real(value_type.clone(), 0.0)
                        }
                        .expect("zero inhabits every checked mathematical type"),
                        expression.range(),
                    )
                } else {
                    expression.clone()
                }
            };
        let left = contextual(
            &left_expression,
            equation.contextual_left_zero,
            &checked.left.value_type,
        );
        let right = contextual(
            &right_expression,
            equation.contextual_right_zero,
            &checked.right.value_type,
        );
        normalized.extend([left, right]);
    }
    // Keep all normalized nodes alive for the pointer-keyed lowering cache.
    let roots = normalized
        .iter()
        .map(|residual| lowerer.lower(residual).map(|value| value.id))
        .collect::<Result<Vec<_>, _>>()?;
    let expression = lowerer.builder.finish(roots).map_err(|diagnostic| {
        source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            file,
            range,
            diagnostic.message(),
        )
    })?;
    Ok(LoweredRelation {
        expression,
        dependencies: lowerer.dependencies,
        ports: lowerer.ports,
    })
}

mod types;
use types::expression_type;
pub(super) use types::relation_support;

fn spatial_type_error(
    file: &str,
    expression: &LoweringExpression,
    error: impl std::fmt::Display,
) -> Diagnostic {
    source_error(
        codes::LANGUAGE_TYPE_ERROR,
        file,
        expression.range(),
        error.to_string(),
    )
}

struct ExpressionLowerer<'a> {
    file: &'a str,
    bindings: &'a BTreeMap<String, Binding>,
    builder: ExprDagBuilder,
    dependencies: BTreeSet<RawId>,
    ports: BTreeSet<RawId>,
    cache: HashMap<(usize, bool), TypedExpression>,
    sampling: bool,
    allow_discrete_symbols: bool,
    activation: &'a ActivationSyntax,
    initial: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TypedExpression {
    id: ExprId,
    pub(super) dimension: DimExponents,
}

impl ExpressionLowerer<'_> {
    fn lower(&mut self, expression: &LoweringExpression) -> Result<TypedExpression, Diagnostic> {
        let key = (Arc::as_ptr(&expression.node) as usize, self.sampling);
        if let Some(lowered) = self.cache.get(&key) {
            return Ok(*lowered);
        }
        let lowered = match expression.node.as_ref() {
            LoweringExpressionNode::Partial { value, wrt } => {
                self.lower_partial(expression, value, wrt)
            }
            LoweringExpressionNode::Number(_) => {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    expression.range(),
                    "unresolved contextual number",
                ));
            }
            LoweringExpressionNode::IntegerCall {
                operator,
                arguments,
            } => {
                let operands = arguments
                    .iter()
                    .map(|argument| self.lower(argument))
                    .collect::<Result<Vec<_>, _>>()?;
                let result = match operator {
                    super::IntegerBuiltin::Ordinal => self.builder.ordinal(operands[0].id),
                    super::IntegerBuiltin::Quotient => {
                        self.builder.quotient(operands[0].id, operands[1].id)
                    }
                    super::IntegerBuiltin::Remainder => {
                        self.builder.remainder(operands[0].id, operands[1].id)
                    }
                    super::IntegerBuiltin::ToReal => self.builder.to_real(operands[0].id),
                    super::IntegerBuiltin::ToInteger => self.builder.to_integer(operands[0].id),
                };
                result
                    .map(|id| TypedExpression {
                        id,
                        dimension: DimExponents::DIMENSIONLESS,
                    })
                    .map_err(|error| self.builder_error(expression, error))
            }
            LoweringExpressionNode::Sample { value, clock } => {
                let Some(Binding::Clock(id, _)) = self.bindings.get(clock) else {
                    return Err(unresolved(
                        self.file,
                        expression.range(),
                        clock,
                        "sample ClockDomain",
                    ));
                };
                let id = *id;
                if self.sampling
                    || self.initial
                    || self.activation != &ActivationSyntax::Named(clock.clone())
                {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.file,
                        expression.range(),
                        "sample requires its exact clock's update relation",
                    ));
                }
                self.sampling = true;
                let operand = self.lower(value);
                self.sampling = false;
                let operand = operand?;
                self.dependencies.insert(id.erase());
                self.builder
                    .sample(operand.id, id)
                    .map(|id| TypedExpression {
                        id,
                        dimension: operand.dimension,
                    })
                    .map_err(|error| self.builder_error(expression, error))
            }
            LoweringExpressionNode::Array(elements) => {
                let elements = elements
                    .iter()
                    .map(|value| self.lower(value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.builder
                    .array(elements.iter().map(|element| element.id))
                    .map(|id| TypedExpression {
                        id,
                        dimension: elements[0].dimension,
                    })
                    .map_err(|diagnostic| self.builder_error(expression, diagnostic))
            }
            LoweringExpressionNode::Index { value, index } => {
                let value = self.lower(value)?;
                self.builder
                    .index(value.id, *index)
                    .map(|id| TypedExpression {
                        id,
                        dimension: value.dimension,
                    })
                    .map_err(|diagnostic| self.builder_error(expression, diagnostic))
            }
            LoweringExpressionNode::Complex { real, imag } => {
                let real = self.lower(real)?;
                let imag = self.lower(imag)?;
                self.builder
                    .complex(real.id, imag.id)
                    .map(|id| TypedExpression {
                        id,
                        dimension: real.dimension,
                    })
                    .map_err(|diagnostic| self.builder_error(expression, diagnostic))
            }
            LoweringExpressionNode::InvalidValue(message) => Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                expression.range(),
                *message,
            )),
            LoweringExpressionNode::Literal(value) => self
                .builder
                .constant(value.clone())
                .map(|id| TypedExpression {
                    id,
                    dimension: value.value_type().dimension(),
                })
                .map_err(|diagnostic| self.builder_error(expression, diagnostic)),
            LoweringExpressionNode::Name(name) if name == "time" => self
                .builder
                .symbol(SymbolRef::Time)
                .map(|id| TypedExpression {
                    id,
                    dimension: time_dimension(),
                })
                .map_err(|diagnostic| self.builder_error(expression, diagnostic)),
            LoweringExpressionNode::Name(name) => self.lower_name(expression, name),
            LoweringExpressionNode::Not(value) => {
                let value = self.lower(value)?;
                self.builder
                    .not(value.id)
                    .map(|id| TypedExpression {
                        id,
                        dimension: DimExponents::DIMENSIONLESS,
                    })
                    .map_err(|diagnostic| self.builder_error(expression, diagnostic))
            }
            LoweringExpressionNode::Neg(value) => {
                let value = self.lower(value)?;
                self.builder
                    .neg(value.id)
                    .map(|id| TypedExpression {
                        id,
                        dimension: value.dimension,
                    })
                    .map_err(|diagnostic| self.builder_error(expression, diagnostic))
            }
            LoweringExpressionNode::Case { value, arms } => {
                self.lower_case(expression, value, arms)
            }
            LoweringExpressionNode::Select {
                condition,
                then_value,
                else_value,
            } => {
                let condition = self.lower(condition)?;
                let then_value = self.lower(then_value)?;
                let else_value = self.lower(else_value)?;
                self.builder
                    .select(condition.id, then_value.id, else_value.id)
                    .map(|id| TypedExpression {
                        id,
                        dimension: then_value.dimension,
                    })
                    .map_err(|diagnostic| self.builder_error(expression, diagnostic))
            }
            LoweringExpressionNode::Require { condition, value } => {
                let condition = self.lower(condition)?;
                let value = self.lower(value)?;
                self.builder
                    .require(condition.id, value.id)
                    .map(|id| TypedExpression {
                        id,
                        dimension: value.dimension,
                    })
                    .map_err(|diagnostic| self.builder_error(expression, diagnostic))
            }
            LoweringExpressionNode::Piecewise { name, arguments } => {
                self.lower_piecewise(expression, name, arguments)
            }
            LoweringExpressionNode::Extremum {
                minimum,
                left,
                right,
            } => {
                let left = self.lower(left)?;
                let right = self.lower(right)?;
                let result = if *minimum {
                    self.builder.min(left.id, right.id)
                } else {
                    self.builder.max(left.id, right.id)
                };
                result
                    .map(|id| TypedExpression {
                        id,
                        dimension: left.dimension,
                    })
                    .map_err(|diagnostic| self.builder_error(expression, diagnostic))
            }
            LoweringExpressionNode::Binary {
                operator,
                left,
                right,
            } => self.lower_binary(expression, *operator, left, right),
            LoweringExpressionNode::Call { callee, argument } => {
                self.lower_call(expression, callee, argument)
            }
            LoweringExpressionNode::Property { release, arguments } => {
                self.lower_property(expression, release, arguments)
            }
            LoweringExpressionNode::PureOperator {
                definition,
                arguments,
            } => self.lower_pure_operator(expression, definition, arguments),
            LoweringExpressionNode::UnknownMath(path) => Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                expression.range(),
                format!("unknown compiler-owned scalar mathematics member `{path}`"),
            )),
            LoweringExpressionNode::Unsupported => Err(source_error(
                codes::LANGUAGE_LOWERING_ERROR,
                self.file,
                expression.range(),
                "expression syntax is newer than this compiler",
            )),
        }?;
        self.cache.insert(key, lowered);
        Ok(lowered)
    }

    fn lower_name(
        &mut self,
        expression: &LoweringExpression,
        name: &str,
    ) -> Result<TypedExpression, Diagnostic> {
        let Some(binding) = self.bindings.get(name).cloned() else {
            return Err(unresolved(
                self.file,
                expression.range(),
                name,
                "expression symbol",
            ));
        };
        let (symbol, id, dimension) = match binding {
            Binding::Field(id, contract) => {
                if self.sampling && !matches!(contract.activation, ActivationSyntax::Continuous) {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.file,
                        expression.range(),
                        "sample operand must be continuous",
                    ));
                }
                if contract.role == eqiora_lang::FieldRoleSyntax::Variable
                    && matches!(contract.activation, ActivationSyntax::Named(_))
                    && (self.initial || &contract.activation != self.activation)
                {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.file,
                        expression.range(),
                        "clocked Variable read requires its exact declared activation",
                    ));
                }
                (SymbolRef::Field(id), id.erase(), contract.dimension)
            }
            Binding::Parameter(id, value_type) => {
                (SymbolRef::Parameter(id), id.erase(), value_type.dimension())
            }
            Binding::Port(id, contract) => match resolve_port_contract(
                self.file,
                expression.range(),
                &contract,
                self.bindings,
            )? {
                ResolvedPortContract::Signal {
                    value_type, clock, ..
                } => {
                    let expected = if self.sampling {
                        None
                    } else {
                        match self.activation {
                            ActivationSyntax::Named(name) => match self.bindings.get(name) {
                                Some(Binding::Clock(id, _)) => Some(*id),
                                _ => None,
                            },
                            _ => None,
                        }
                    };
                    if clock != expected {
                        return Err(source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            self.file,
                            expression.range(),
                            "signal Port read requires the exact declared activation; use an explicit transition",
                        ));
                    }
                    self.ports.insert(id.erase());
                    (SymbolRef::Port(id), id.erase(), value_type.dimension())
                }
                ResolvedPortContract::ScalarPhysical { .. } => {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.file,
                        expression.range(),
                        format!(
                            "scalar physical Port `{name}` requires a declared quantity member (`port.member`)"
                        ),
                    ));
                }
                ResolvedPortContract::BoundaryPhysical { .. } => {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.file,
                        expression.range(),
                        format!(
                            "field-physical Port `{name}` requires a declared quantity member (`port.member`)"
                        ),
                    ));
                }
            },
            Binding::Domain(_, _)
            | Binding::Representation(_)
            | Binding::Clock(_, _)
            | Binding::Observable(_)
            | Binding::Event(_)
            | Binding::Relation { .. } => {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    expression.range(),
                    format!("`{name}` is not a scalar Field, Parameter, or Port"),
                ));
            }
        };
        self.dependencies.insert(id);
        self.builder
            .symbol(symbol)
            .map(|id| TypedExpression { id, dimension })
            .map_err(|diagnostic| self.builder_error(expression, diagnostic))
    }

    fn lower_call(
        &mut self,
        expression: &LoweringExpression,
        callee: &str,
        argument: &LoweringExpression,
    ) -> Result<TypedExpression, Diagnostic> {
        if callee == "period" {
            let LoweringExpressionNode::Name(name) = argument.node.as_ref() else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    argument.range(),
                    "period requires one clock name",
                ));
            };
            let Some(Binding::Clock(_, period)) = self.bindings.get(name) else {
                return Err(unresolved(
                    self.file,
                    argument.range(),
                    name,
                    "period ClockDomain",
                ));
            };
            let dimension = crate::dimensions::time_dimension();
            let literal = eqiora_core::ValueLiteral::from_real(
                eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, dimension)
                    .expect("admitted numeric scalar type"),
                period.as_seconds_f64(),
            )
            .expect("bounded positive period");
            return self
                .builder
                .constant(literal)
                .map(|id| TypedExpression { id, dimension })
                .map_err(|error| self.builder_error(expression, error));
        }
        if callee == "sin" {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                expression.range(),
                "bare `sin` is not language vocabulary; use compiler-owned `math.sin`",
            ));
        }
        if callee.starts_with("math.") && !matches!(callee, "math.sin" | "math.sqrt") {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                expression.range(),
                format!("unknown compiler-owned scalar mathematics member `{callee}`"),
            ));
        }
        let boundary_trace = callee == "trace"
            && matches!(
                argument.node.as_ref(),
                LoweringExpressionNode::Name(name)
                    if matches!(
                        self.bindings.get(name),
                        Some(Binding::Port(_, PortContract::BoundaryPhysical { .. }))
                    )
            );
        if matches!(callee, "across" | "through" | "flux") || boundary_trace {
            return self.lower_physical_accessor(expression, callee, argument);
        }
        if callee == "coordinate" {
            let axis = lowering_integer_literal(argument)
                .and_then(|axis| usize::try_from(axis).ok())
                .ok_or_else(|| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.file,
                        argument.range(),
                        "coordinate(...) requires a non-negative integer literal axis",
                    )
                })?;
            return self
                .builder
                .spatial_coordinate(axis)
                .map(|id| TypedExpression {
                    id,
                    dimension: length_dimension(),
                })
                .map_err(|diagnostic| self.builder_error(expression, diagnostic));
        }
        if matches!(callee, "math.sin" | "math.sqrt") {
            let operand = self.lower(argument)?;
            if callee == "math.sin" && operand.dimension != DimExponents::DIMENSIONLESS {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    argument.range(),
                    format!(
                        "math.sin(...) requires a dimensionless scalar, received [{}]",
                        operand.dimension
                    ),
                ));
            }
            let (function, dimension) = if callee == "math.sqrt" {
                (
                    UnaryMathFunction::Sqrt,
                    operand
                        .dimension
                        .pow(1, 2)
                        .ok_or_else(|| dimension_overflow(self.file, expression.range()))?,
                )
            } else {
                (UnaryMathFunction::Sin, DimExponents::DIMENSIONLESS)
            };
            return self
                .builder
                .unary_math(function, operand.id)
                .map(|id| TypedExpression { id, dimension })
                .map_err(|diagnostic| self.builder_error(expression, diagnostic));
        }
        if matches!(
            callee,
            "grad" | "div" | "symmetric_part" | "isotropic_lift" | "trace" | "normal"
        ) {
            let operand = self.lower(argument)?;
            let (result, dimension) = match callee {
                "grad" => (
                    self.builder.gradient(operand.id),
                    operand
                        .dimension
                        .div(length_dimension())
                        .ok_or_else(|| dimension_overflow(self.file, expression.range()))?,
                ),
                "div" => (
                    self.builder.divergence(operand.id),
                    operand
                        .dimension
                        .div(length_dimension())
                        .ok_or_else(|| dimension_overflow(self.file, expression.range()))?,
                ),
                "symmetric_part" => (self.builder.symmetric_part(operand.id), operand.dimension),
                "isotropic_lift" => (self.builder.isotropic_lift(operand.id), operand.dimension),
                "trace" => (self.builder.trace(operand.id), operand.dimension),
                "normal" => (self.builder.normal_component(operand.id), operand.dimension),
                _ => unreachable!("spatial operator was matched"),
            };
            return result
                .map(|id| TypedExpression { id, dimension })
                .map_err(|diagnostic| self.builder_error(expression, diagnostic));
        }
        let LoweringExpressionNode::Name(name) = argument.node.as_ref() else {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                argument.range(),
                format!("{callee}(...) requires one Field name"),
            ));
        };
        let Some(Binding::Field(field, contract)) = self.bindings.get(name).cloned() else {
            return Err(unresolved(
                self.file,
                argument.range(),
                name,
                "Field operator argument",
            ));
        };
        if callee == "hold" {
            if contract.role != eqiora_lang::FieldRoleSyntax::State
                || !matches!(contract.activation, ActivationSyntax::Named(_))
            {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    expression.range(),
                    "hold requires one periodic State",
                ));
            }
            self.dependencies.insert(field.erase());
            let symbol = self
                .builder
                .symbol(SymbolRef::Field(field))
                .map_err(|error| self.builder_error(expression, error))?;
            return self
                .builder
                .hold(symbol)
                .map(|id| TypedExpression {
                    id,
                    dimension: contract.dimension,
                })
                .map_err(|error| self.builder_error(expression, error));
        }
        if self.sampling {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                expression.range(),
                "sample operand cannot contain an evolution operator",
            ));
        }
        if matches!(callee, "derivative" | "pre" | "next") {
            let eligible = self.eligible_evolution(callee, &contract);
            if !eligible {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    expression.range(),
                    "evolution operator requires an eligible declared state at the exact clock",
                ));
            }
        }
        if matches!(callee, "pre" | "next") && !self.allow_discrete_symbols {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                expression.range(),
                format!("continuous Relation cannot use `{callee}`"),
            ));
        }
        if callee == "derivative" && self.allow_discrete_symbols && !self.initial {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.file,
                expression.range(),
                "clocked Relation cannot use `derivative`",
            ));
        }
        let (symbol, dimension) = match callee {
            "derivative" => (
                SymbolRef::Derivative(field),
                contract.dimension.div(time_dimension()).ok_or_else(|| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.file,
                        expression.range(),
                        "derivative dimension exponent exceeds rational exponent bounds",
                    )
                })?,
            ),
            "pre" => (SymbolRef::Pre(field), contract.dimension),
            "next" => (SymbolRef::Next(field), contract.dimension),
            _ => {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    expression.range(),
                    format!("unknown scalar operator `{callee}`"),
                ));
            }
        };
        self.dependencies.insert(field.erase());
        self.builder
            .symbol(symbol)
            .map(|id| TypedExpression { id, dimension })
            .map_err(|diagnostic| self.builder_error(expression, diagnostic))
    }

    fn lower_binary(
        &mut self,
        expression: &LoweringExpression,
        operator: BinaryOp,
        left: &LoweringExpression,
        right: &LoweringExpression,
    ) -> Result<TypedExpression, Diagnostic> {
        if operator == BinaryOp::Pow {
            let base = self.lower(left)?;
            let exponent = lowering_integer_literal(right).ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    right.range(),
                    "power exponent must be an i32 integer literal",
                )
            })?;
            let dimension = base.dimension.pow(exponent, 1).ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    expression.range(),
                    "power dimension exponent exceeds rational exponent bounds",
                )
            })?;
            return self
                .builder
                .powi(base.id, exponent)
                .map(|id| TypedExpression { id, dimension })
                .map_err(|diagnostic| self.builder_error(expression, diagnostic));
        }

        let left = self.lower(left)?;
        let right = self.lower(right)?;
        let dimension = match operator {
            BinaryOp::Add | BinaryOp::Sub if left.dimension == right.dimension => left.dimension,
            BinaryOp::Add | BinaryOp::Sub => {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.file,
                    expression.range(),
                    format!(
                        "addition/subtraction combines dimensions [{}] and [{}]",
                        left.dimension, right.dimension
                    ),
                ));
            }
            BinaryOp::Mul => left
                .dimension
                .mul(right.dimension)
                .ok_or_else(|| dimension_overflow(self.file, expression.range()))?,
            BinaryOp::Div => left
                .dimension
                .div(right.dimension)
                .ok_or_else(|| dimension_overflow(self.file, expression.range()))?,
            BinaryOp::Pow => unreachable!("power handled above"),
            _ => DimExponents::DIMENSIONLESS,
        };
        let result = match operator {
            BinaryOp::Add => self.builder.add(left.id, right.id),
            BinaryOp::Sub => self.builder.sub(left.id, right.id),
            BinaryOp::Mul => self.builder.mul(left.id, right.id),
            BinaryOp::Div => self.builder.div(left.id, right.id),
            BinaryOp::Pow => unreachable!("power handled above"),
            BinaryOp::And => self.builder.and(left.id, right.id),
            BinaryOp::Or => self.builder.or(left.id, right.id),
            op => self.builder.compare(
                super::comparison_operator(op).expect("comparison"),
                left.id,
                right.id,
            ),
        };
        result
            .map(|id| TypedExpression { id, dimension })
            .map_err(|diagnostic| self.builder_error(expression, diagnostic))
    }

    fn builder_error(&self, expression: &LoweringExpression, diagnostic: Diagnostic) -> Diagnostic {
        source_error(
            codes::LANGUAGE_LOWERING_ERROR,
            self.file,
            expression.range(),
            diagnostic.message(),
        )
    }
}

pub(super) fn lowering_integer_literal(expression: &LoweringExpression) -> Option<i32> {
    let value = match expression.node.as_ref() {
        LoweringExpressionNode::Literal(value)
            if value.value_type().dimension() == DimExponents::DIMENSIONLESS =>
        {
            value.real_scalar_value()?.value()
        }
        LoweringExpressionNode::Neg(value) => match value.node.as_ref() {
            LoweringExpressionNode::Literal(value)
                if value.value_type().dimension() == DimExponents::DIMENSIONLESS =>
            {
                -value.real_scalar_value()?.value()
            }
            _ => return None,
        },
        _ => return None,
    };
    (value.fract() == 0.0 && value >= f64::from(i32::MIN) && value <= f64::from(i32::MAX))
        .then_some(value as i32)
}

fn instantiate_pure_dimension(
    definition: &PureOperatorDefinition,
    arguments: &[TypedExpression],
) -> Option<DimExponents> {
    if arguments.len() != definition.formals().len() {
        return None;
    }
    arguments
        .iter()
        .zip(definition.dimension_monomial().exponents())
        .try_fold(
            definition.dimension_monomial().fixed_dimension(),
            |result, (argument, exponent)| {
                let term = argument.dimension.pow(
                    i32::try_from(exponent.numerator()).ok()?,
                    i32::try_from(exponent.denominator()).ok()?,
                )?;
                result.mul(term)
            },
        )
}
