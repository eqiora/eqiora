mod relation;
pub(super) use relation::{validate_relation_expression, validate_relation_family_expression};
mod aliases;
mod observable;
pub(super) use observable::validate_observable;
mod channels;
mod conditions;
mod enumeration;
mod integer;
mod law;
mod partial;
mod reductions;
mod time_derivative;
mod transitions;
pub(in crate::hierarchy) use aliases::DependencyActivation;
pub(super) use aliases::{AliasContract, validate_aliases};

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, DimExponents};
use eqiora_lang::{
    ActivationSyntax, BinaryOp, Expr, ExprKind, RelationDecl, RelationFamilyDecl, UnaryOp,
};
use eqiora_schema::kernel::typing::{self, ExpressionType, SpatialSupport, TypeViolation};

use crate::diagnostics::source_error;
use crate::dimensions::{integer_literal, time_dimension};
use crate::pure_operator::is_builtin_operator;

use super::PhysicalEndpointSelections;
use super::scope::{
    BoundaryFamilyScope, DefinitionScope, PortContract, SymbolContract, unresolved,
};

pub(super) fn validate_initial_expression(
    scope: &DefinitionScope<'_, '_>,
    declaration: &eqiora_lang::InitialDecl,
) -> Result<(), Vec<Diagnostic>> {
    let mut checker = ExpressionChecker {
        scope,
        relation_support: None,
        family_scope: None,
        allow_discrete_symbols: true,
        initial: true,
        activation: &ActivationSyntax::Continuous,
        physical_endpoints: PhysicalEndpointSelections::new(),
        intrinsic: false,
        alias_dependencies: Vec::new(),
        evolution: Vec::new(),
        contextual: Vec::new(),
        sampling: false,
    };
    let errors: Vec<_> = declaration
        .equations()
        .iter()
        .filter_map(|equation| checker.check_equation(equation).err())
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

pub(super) fn validate_event_guard(
    scope: &DefinitionScope<'_, '_>,
    guard: &Expr,
) -> Result<(), Diagnostic> {
    let mut checker = ExpressionChecker {
        scope,
        relation_support: None,
        family_scope: None,
        allow_discrete_symbols: false,
        initial: false,
        activation: &ActivationSyntax::Continuous,
        physical_endpoints: PhysicalEndpointSelections::new(),
        intrinsic: false,
        alias_dependencies: Vec::new(),
        evolution: Vec::new(),
        contextual: Vec::new(),
        sampling: false,
    };
    let inferred = checker.check(guard)?;
    if inferred.value_type.scalar_domain() != eqiora_core::ScalarDomain::Real
        || !inferred.shape().is_scalar()
        || inferred.frame() != eqiora_core::ValueFrame::Invariant
        || inferred.support.is_some()
    {
        return Err(source_error(
            codes::LANGUAGE_TYPE_ERROR,
            scope.file,
            guard.range(),
            "event crossing guard requires one real invariant scalar quantity",
        ));
    }
    Ok(())
}

struct ExpressionChecker<'a, 'e, 'd> {
    scope: &'a DefinitionScope<'e, 'd>,
    relation_support: Option<SpatialSupport<String>>,
    family_scope: Option<&'a BoundaryFamilyScope>,
    allow_discrete_symbols: bool,
    initial: bool,
    activation: &'a ActivationSyntax,
    physical_endpoints: PhysicalEndpointSelections,
    intrinsic: bool,
    alias_dependencies: Vec<std::sync::Arc<AliasContract>>,
    evolution: Vec<aliases::EvolutionRequirement>,
    contextual: Vec<Expr>,
    sampling: bool,
}

impl ExpressionChecker<'_, '_, '_> {
    fn check(&mut self, expression: &Expr) -> Result<ExpressionType<String>, Diagnostic> {
        if let Some(value) = expression.resolved_enum() {
            return Ok(ExpressionType::new(value.value_type().clone(), None));
        }
        if let Some((port, member)) = super::super::quantity_member::split(expression)
            && (matches!(port.kind(), ExprKind::BoundaryPortSelection { .. })
                || self.physical_port_contract(&port).is_some())
        {
            return self.check_physical_member(member, &port);
        }
        match expression.kind() {
            ExprKind::Partial {
                value,
                wrt,
                holding,
            } => {
                let selected = self.partial_binding(wrt)?;
                let mut seen = std::collections::BTreeSet::from([wrt.as_str()]);
                for binding in holding {
                    self.partial_binding(binding)?;
                    if !seen.insert(binding.as_str()) {
                        return Err(source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            self.scope.file,
                            binding.range(),
                            "partial holding requires distinct other independent bindings",
                        ));
                    }
                }
                let value = self.check(value)?;
                crate::lower::partial_result_type(&value, &selected).map_err(|message| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.scope.file,
                        expression.range(),
                        message,
                    )
                })
            }
            ExprKind::Reduction { .. } => self.reduction(expression),
            ExprKind::Array(_) | ExprKind::Index { .. } | ExprKind::Slice { .. } => {
                self.channels(expression)
            }
            ExprKind::Path(path) if path.as_str() == "math.i" => Ok(ExpressionType::new(
                eqiora_core::ValueType::scalar(
                    eqiora_core::ScalarDomain::Complex,
                    DimExponents::DIMENSIONLESS,
                )
                .expect("admitted numeric scalar type"),
                None,
            )),
            ExprKind::Call { callee, .. } if callee.as_str() == "tensor_value" => {
                let value = crate::hierarchy::parameters::frames::literal(
                    self.scope.file,
                    expression,
                    &self.scope.static_values,
                    &mut |name| self.scope.spatial_support(name),
                )?;
                Ok(ExpressionType::new(value.value_type().clone(), None))
            }
            ExprKind::Call {
                callee,
                arguments: eqiora_lang::CallArguments::Positional(arguments),
            } if callee.as_str() == "math.complex" => {
                let [real, imag] = arguments.as_slice() else {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.scope.file,
                        expression.range(),
                        "math.complex requires exactly two real scalar arguments",
                    ));
                };
                ExpressionType::complex(self.check(real)?, self.check(imag)?)
                    .map_err(|error| type_error(self.scope.file, expression, error))
            }
            ExprKind::Case { value, arms } => self.check_case(expression, value, arms, None),
            ExprKind::Select {
                condition,
                then_value,
                else_value,
            } => {
                let condition = self.check(condition)?;
                let (then_value, else_value) = self.check_pair(then_value, else_value)?;
                condition
                    .select(then_value, else_value)
                    .map_err(|error| type_error(self.scope.file, expression, error))
            }
            ExprKind::Boolean(_) => {
                Ok(ExpressionType::new(eqiora_core::ValueType::boolean(), None))
            }
            ExprKind::Number(_) => Ok(ExpressionType::scalar(DimExponents::DIMENSIONLESS, None)),
            ExprKind::Quantity { value, unit } => {
                let quantity = crate::units::quantity(value, unit).map_err(|message| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.scope.file,
                        expression.range(),
                        message,
                    )
                })?;
                Ok(ExpressionType::scalar(quantity.dim(), None))
            }
            ExprKind::Name(name) if name == "time" => Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                expression.range(),
                "the continuous coordinate is spelled time()",
            )),
            ExprKind::Name(name) => self.scalar_local_symbol(expression, name),
            ExprKind::Path(path) => match crate::math::constant(path) {
                Some(_) => Ok(ExpressionType::scalar(DimExponents::DIMENSIONLESS, None)),
                None if crate::math::is_namespaced(path) => Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    path.range(),
                    format!("unknown compiler-owned scalar mathematics member `{path}`"),
                )),
                None => self.scalar_contract(
                    expression,
                    path.as_str(),
                    self.scope.resolve_symbol(path)?,
                ),
            },
            ExprKind::Member { .. } => {
                let (path, key) = self.scope.indexed_member(expression)?;
                self.scalar_contract(
                    expression,
                    path.as_str(),
                    self.scope.resolve_symbol_at(&path, key[1].parse().ok())?,
                )
            }
            ExprKind::BoundaryPortSelection { port, selector } => {
                let Some(family_scope) = self.family_scope else {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.scope.file,
                        expression.range(),
                        "boundary Port selector is valid only inside a boundary family scope",
                    ));
                };
                let contract =
                    self.scope
                        .resolve_boundary_port_selection(port, selector, family_scope)?;
                self.scalar_contract(expression, port.as_str(), SymbolContract::Port(contract))
            }
            ExprKind::Unary {
                op: UnaryOp::Neg,
                value,
            } => {
                let inferred = self.check(value)?;
                if inferred.value_type.is_count()
                    || inferred.value_type.index_set().is_some()
                    || inferred.value_type.scalar_domain() == eqiora_core::ScalarDomain::Boolean
                {
                    return Err(type_error(
                        self.scope.file,
                        expression,
                        TypeViolation::ScalarDomainMismatch,
                    ));
                }
                Ok(inferred)
            }
            ExprKind::Unary {
                op: UnaryOp::Not,
                value,
            } => self
                .check(value)?
                .logical_not()
                .map_err(|error| type_error(self.scope.file, expression, error)),
            ExprKind::Binary { op, left, right } => self.check_binary(expression, *op, left, right),
            ExprKind::Call { callee, arguments } => self.check_call(expression, callee, arguments),
            _ => Err(source_error(
                codes::LANGUAGE_LOWERING_ERROR,
                self.scope.file,
                expression.range(),
                "expression syntax is newer than definition-body validation",
            )),
        }
    }

    fn scalar_local_symbol(
        &mut self,
        expression: &Expr,
        name: &str,
    ) -> Result<ExpressionType<String>, Diagnostic> {
        if let Some(contract) = self.scope.properties.get(name) {
            return super::property::bare(self.scope.file, expression.range(), contract);
        }
        let contract = self.scope.symbols.get(name).cloned().ok_or_else(|| {
            unresolved(
                self.scope.file,
                expression.range(),
                name,
                "expression symbol",
            )
        })?;
        self.scalar_contract(expression, name, contract)
    }

    fn scalar_contract(
        &mut self,
        expression: &Expr,
        display: &str,
        contract: SymbolContract,
    ) -> Result<ExpressionType<String>, Diagnostic> {
        match contract {
            SymbolContract::Field(inferred, role, activation) => {
                if self.sampling && activation != ActivationSyntax::Continuous {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.scope.file,
                        expression.range(),
                        "sample operand must be continuous",
                    ));
                }
                if role == eqiora_lang::FieldRoleSyntax::Variable
                    && matches!(activation, ActivationSyntax::Named(_))
                {
                    if self.intrinsic {
                        self.contextual.push(expression.clone());
                    } else if self.initial
                        || !self.scope.activation_matches(&activation, self.activation)
                    {
                        return Err(source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            self.scope.file,
                            expression.range(),
                            "clocked Variable read requires its exact declared activation",
                        ));
                    }
                }
                Ok(inferred)
            }
            SymbolContract::Parameter(inferred) => Ok(inferred),
            SymbolContract::Alias(alias) => self.use_alias(alias),
            SymbolContract::Port(contract) => {
                if let PortContract::Signal { activation, .. } = &contract {
                    if self.intrinsic {
                        self.contextual.push(expression.clone());
                    } else if !self.scope.activation_matches(activation, self.activation) {
                        return Err(source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            self.scope.file,
                            expression.range(),
                            "signal Port read requires its exact declared activation; use an explicit transition",
                        ));
                    }
                }
                contract.expression_type().ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    expression.range(),
                    format!(
                        "physical Port `{display}` requires a declared quantity member (`port.member`)"
                    ),
                )
            })
            }
            SymbolContract::PortFamily(_) => Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                expression.range(),
                format!(
                    "boundary Port family `{display}` requires an exact `[member = target]` selector"
                ),
            )),
            SymbolContract::Domain(_)
            | SymbolContract::Support(_)
            | SymbolContract::CompleteExterior { .. }
            | SymbolContract::Clock
            | SymbolContract::Event
            | SymbolContract::Observable
            | SymbolContract::Relation => Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                expression.range(),
                format!("`{display}` is not a scalar Field, Parameter, or Port"),
            )),
        }
    }

    fn check_call(
        &mut self,
        expression: &Expr,
        callee: &eqiora_lang::NamePath,
        arguments: &eqiora_lang::CallArguments,
    ) -> Result<ExpressionType<String>, Diagnostic> {
        let callee_name = callee.as_str();
        if let Some(contract) = self.scope.properties.get(callee_name).cloned() {
            return super::property::application(
                self.scope.file,
                expression.range(),
                &contract,
                arguments,
                |argument| self.check(argument),
            );
        }
        if !is_builtin_operator(callee)
            && !matches!(callee_name, "counts" | "coordinates" | "index" | "sin")
            && crate::lower::IntegerBuiltin::named(callee_name).is_none()
            && !crate::math::is_namespaced(callee)
        {
            let definition = self.scope.elaborator.resolve_pure_operator(
                &self.scope.namespace,
                callee,
                self.scope.file,
                callee.range(),
            )?;
            let inferred = crate::pure_operator::ordered_arguments(
                self.scope.file,
                expression.range(),
                definition
                    .declaration
                    .formals()
                    .iter()
                    .map(|formal| formal.name()),
                arguments,
            )?
            .into_iter()
            .map(|argument| self.check(argument))
            .collect::<Result<Vec<_>, _>>()?;
            return definition
                .definition
                .instantiate(&inferred)
                .map(|application| application.result_type().clone())
                .map_err(|error| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.scope.file,
                        expression.range(),
                        format!("invalid application of pure operator `{callee}`: {error}"),
                    )
                });
        }
        let arguments = arguments.positional().ok_or_else(|| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                expression.range(),
                "builtin calls require positional arguments",
            )
        })?;
        if callee_name == "time" {
            if !arguments.is_empty()
                || (!self.intrinsic && !matches!(self.activation, ActivationSyntax::Continuous))
            {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    expression.range(),
                    "time() requires no arguments and the enclosing continuous timeline",
                ));
            }
            if self.intrinsic {
                self.contextual.push(expression.clone());
            }
            return Ok(ExpressionType::scalar(time_dimension(), None));
        }
        if matches!(callee_name, "counts" | "coordinates" | "index") {
            return expression
                .resolved_nominal()
                .cloned()
                .map(|value_type| ExpressionType::new(value_type, None))
                .ok_or_else(|| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.scope.file,
                        expression.range(),
                        "nominal constructor requires its resolved lexical declaration",
                    )
                });
        }
        if crate::math::piecewise::arity(callee_name).is_some() {
            let operands = arguments
                .iter()
                .map(|value| self.check(value))
                .collect::<Result<Vec<_>, _>>()?;
            return crate::math::piecewise::result_type(callee_name, &operands)
                .map_err(|error| type_error(self.scope.file, expression, error));
        }
        if let Some(operator) = crate::lower::IntegerBuiltin::named(callee_name) {
            if arguments.len() != operator.arity() {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    expression.range(),
                    "invalid integer operation arity",
                ));
            }
            let operands = arguments
                .iter()
                .map(|value| self.check_numeric_context(value, operator.operand_domain()))
                .collect::<Result<Vec<_>, _>>()?;
            return operator
                .infer(&operands)
                .map_err(|error| type_error(self.scope.file, expression, error));
        }

        if callee_name == "period" {
            let [argument] = arguments else {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    expression.range(),
                    "period requires one clock name",
                ));
            };
            if !matches!(argument.kind(),ExprKind::Name(name) if matches!(self.scope.symbols.get(name),Some(SymbolContract::Clock)))
            {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    argument.range(),
                    "period requires one declared periodic clock",
                ));
            }
            return Ok(ExpressionType::new(
                eqiora_core::ValueType::scalar(eqiora_core::ScalarDomain::Real, time_dimension())
                    .expect("admitted numeric scalar type"),
                None,
            ));
        }
        if matches!(callee_name, "sample" | "hold") {
            return self.check_transition(expression, callee_name, arguments);
        }
        if callee_name == "sin" {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                callee.range(),
                "bare `sin` is not language vocabulary; use compiler-owned `math.sin`",
            ));
        }
        if crate::math::is_namespaced(callee) && !crate::math::is_function(callee) {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                callee.range(),
                format!("unknown compiler-owned scalar mathematics member `{callee}`"),
            ));
        }
        let [argument] = arguments else {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                expression.range(),
                format!("builtin operator `{callee_name}` requires exactly one argument"),
            ));
        };
        if callee_name == "trace" && self.is_boundary_port_selection(argument) {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                expression.range(),
                "physical Port quantities require their declared `port.member` name",
            ));
        }
        if self.intrinsic && matches!(callee_name, "coordinate" | "trace" | "normal") {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                expression.range(),
                "let alias has no support context for this operator; write it directly in its Relation",
            ));
        }
        if callee_name == "coordinate" {
            let axis = integer_literal(argument)
                .and_then(|axis| usize::try_from(axis).ok())
                .ok_or_else(|| {
                    source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.scope.file,
                        argument.range(),
                        "coordinate(...) requires a non-negative integer literal axis",
                    )
                })?;
            return typing::coordinate(axis, self.relation_support.as_ref())
                .map_err(|error| type_error(self.scope.file, expression, error));
        }
        if matches!(callee_name, "math.sin" | "math.sqrt") {
            let function = if callee_name == "math.sqrt" {
                eqiora_schema::kernel::UnaryMathFunction::Sqrt
            } else {
                eqiora_schema::kernel::UnaryMathFunction::Sin
            };
            return typing::unary_math(function, &self.check(argument)?)
                .map_err(|error| type_error(self.scope.file, expression, error));
        }
        if matches!(
            callee_name,
            "grad" | "div" | "symmetric_part" | "isotropic_lift" | "trace" | "normal"
        ) {
            let operand = self.check(argument)?;
            let result = match callee_name {
                "grad" => typing::gradient(&operand),
                "div" => typing::divergence(&operand),
                "symmetric_part" => typing::symmetric_part(&operand),
                "isotropic_lift" => typing::isotropic_lift(&operand),
                "trace" => typing::trace(&operand, self.relation_support.as_ref()),
                "normal" => typing::normal(&operand, self.relation_support.as_ref()),
                _ => unreachable!("spatial operator was matched"),
            };
            return result.map_err(|error| type_error(self.scope.file, expression, error));
        }

        if callee_name == "derivative"
            && !matches!(argument.kind(), ExprKind::Name(name)
                if matches!(self.scope.symbols.get(name), Some(SymbolContract::Field(..))))
        {
            return self.check_time_derivative(expression, argument);
        }
        self.check_evolution(callee_name, expression, argument)
    }

    fn is_boundary_port_selection(&self, expression: &Expr) -> bool {
        matches!(
            self.physical_port_contract(expression),
            Some(PortContract::BoundaryPhysical { .. })
        )
    }

    fn physical_port_contract(&self, expression: &Expr) -> Option<PortContract> {
        let contract = match expression.kind() {
            ExprKind::Name(name) => self.scope.symbols.get(name).cloned(),
            ExprKind::Path(path) => self.scope.resolve_symbol(path).ok(),
            ExprKind::Member { .. } => self
                .scope
                .indexed_member(expression)
                .ok()
                .and_then(|(path, _)| self.scope.resolve_symbol(&path).ok()),
            ExprKind::BoundaryPortSelection { port, selector } => self
                .family_scope
                .and_then(|family_scope| {
                    self.scope
                        .resolve_boundary_port_selection(port, selector, family_scope)
                        .ok()
                })
                .map(SymbolContract::Port),
            _ => None,
        };
        match contract {
            Some(SymbolContract::Port(value)) if value.is_physical() => Some(value),
            _ => None,
        }
    }

    fn check_physical_member(
        &mut self,
        member: &str,
        argument: &Expr,
    ) -> Result<ExpressionType<String>, Diagnostic> {
        let (display, contract, endpoint) = match argument.kind() {
            ExprKind::Name(name) => (
                name.clone(),
                self.scope.symbols.get(name).cloned().ok_or_else(|| {
                    unresolved(
                        self.scope.file,
                        argument.range(),
                        name,
                        "scalar physical Port",
                    )
                })?,
                super::ResolvedPhysicalEndpoint::from_expression(argument),
            ),
            ExprKind::Path(path) => (
                path.as_str().to_owned(),
                self.scope.resolve_symbol(path)?,
                super::ResolvedPhysicalEndpoint::from_expression(argument),
            ),
            ExprKind::Member { .. } => {
                let (path, key) = self.scope.indexed_member(argument)?;
                let endpoint =
                    super::ResolvedPhysicalEndpoint::from_key(&key).ok_or_else(|| {
                        source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            self.scope.file,
                            argument.range(),
                            "physical accessor requires an exact static indexed occurrence",
                        )
                    })?;
                (
                    key.join("."),
                    self.scope.resolve_symbol_at(&path, key[1].parse().ok())?,
                    Some(endpoint),
                )
            }
            ExprKind::BoundaryPortSelection { port, selector } => {
                let Some(family_scope) = self.family_scope else {
                    return Err(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.scope.file,
                        argument.range(),
                        "boundary Port selector is valid only inside a boundary family scope",
                    ));
                };
                (
                    format!("{}[{} = {}]", port, selector.member(), selector.target()),
                    SymbolContract::Port(self.scope.resolve_boundary_port_selection(
                        port,
                        selector,
                        family_scope,
                    )?),
                    None,
                )
            }
            _ => {
                return Err(source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    argument.range(),
                    format!("quantity `{member}` requires one exact physical Port selection"),
                ));
            }
        };
        if let Some(endpoint) = endpoint {
            self.physical_endpoints.insert(endpoint);
        }
        let role = match &contract {
            SymbolContract::Port(
                PortContract::Physical { quantities, .. }
                | PortContract::BoundaryPhysical { quantities, .. },
            ) => quantities.role(member),
            _ => None,
        }
        .ok_or_else(|| {
            source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                argument.range(),
                format!("Port `{display}` has no declared quantity `{member}`"),
            )
        })?;
        match contract {
            SymbolContract::Port(PortContract::Physical {
                across_type,
                through_type,
                ..
            }) if matches!(role, "across" | "through") => Ok(ExpressionType::new(
                if role == "across" {
                    across_type
                } else {
                    through_type
                },
                None,
            )),
            SymbolContract::Port(PortContract::BoundaryPhysical {
                connector, support, ..
            }) if matches!(role, "trace" | "flux") => Ok(ExpressionType::new(
                if role == "trace" {
                    connector.trace_type().clone()
                } else {
                    connector.flux_type().clone()
                },
                Some(support),
            )),
            SymbolContract::Port(PortContract::BoundaryPhysical { .. }) => Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                argument.range(),
                format!(
                    "field-physical Port `{display}` requires its declared trace or flux quantity"
                ),
            )),
            _ => Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.scope.file,
                argument.range(),
                format!("`{display}` is not compatible with quantity `{member}`"),
            )),
        }
    }

    fn check_binary(
        &mut self,
        expression: &Expr,
        operator: BinaryOp,
        left: &Expr,
        right: &Expr,
    ) -> Result<ExpressionType<String>, Diagnostic> {
        if operator == BinaryOp::Pow {
            let base = self.check(left)?;
            let exponent = integer_literal(right).ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    self.scope.file,
                    right.range(),
                    "power exponent must be an i32 integer literal",
                )
            })?;
            return typing::power(&base, exponent)
                .map_err(|error| type_error(self.scope.file, expression, error));
        }
        let (left, right) = self.check_pair(left, right)?;
        let result = match operator {
            BinaryOp::And => left.and(right),
            BinaryOp::Or => left.or(right),
            op if crate::lower::comparison_operator(op).is_some() => {
                left.compare(crate::lower::comparison_operator(op).unwrap(), right)
            }
            BinaryOp::Add => left.sum(right),
            BinaryOp::Sub
                if left.value_type.is_count() || left.value_type.index_set().is_some() =>
            {
                Err(TypeViolation::ScalarDomainMismatch)
            }
            BinaryOp::Sub => typing::additive(&left, &right),
            BinaryOp::Mul => typing::multiply(&left, &right),
            BinaryOp::Div => typing::divide(&left, &right),
            BinaryOp::Pow => unreachable!("power handled above"),
            _ => unreachable!("comparison handled above"),
        };
        result.map_err(|error| type_error(self.scope.file, expression, error))
    }
}

fn type_error(file: &str, expression: &Expr, error: TypeViolation<String>) -> Diagnostic {
    let message = match &error {
        TypeViolation::AdditiveTypeMismatch { left, right } if left.shape() == right.shape() => {
            format!(
                "addition/subtraction combines dimensions [{}] and [{}]",
                left.dimension(),
                right.dimension()
            )
        }
        TypeViolation::SinRequiresDimensionlessScalar => {
            "math.sin(...) requires a dimensionless scalar".to_owned()
        }
        _ => error.to_string(),
    };
    source_error(
        codes::LANGUAGE_TYPE_ERROR,
        file,
        expression.range(),
        message,
    )
}
