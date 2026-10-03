use std::collections::{BTreeMap, BTreeSet, VecDeque};

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, ScalarDomain, ValueLiteral, ValueType};
use eqiora_lang::{
    BinaryOp, ComponentDecl, ComponentItem, ComponentParameterDecl, Expr, ExprKind, InstanceDecl,
    TextRange, UnaryOp, VisibilitySyntax,
};

use crate::diagnostics::{source_error, stable_sort};
use crate::identity::FullElaborationIdentity;
use crate::lower::LoweringExpression;

use super::hierarchy_error;

mod array_types;
mod records;
pub(in crate::hierarchy) use records::RecordContext;
mod bindings;
mod dependencies;
mod selected;
pub(in crate::hierarchy) use array_types::{extent_expressions, specialize_type};
pub(in crate::hierarchy) use selected::resolve_selected_parameters;
mod expression_eval;
pub(in crate::hierarchy) mod frames;
mod predicates;
mod tensor_values;
mod value_expressions;
use dependencies::{
    ExpressionDefinition, collect_expression_dependencies, expression_cycles,
    expression_evaluation_order,
};
use eqiora_schema::kernel::typing::SpatialSupport;
mod model_lets;
mod model_parameters;
use expression_eval::{
    ExpressionContext, coerce_parameter, coerce_parameter_with_label, evaluate_initializer,
};
pub(super) use model_lets::{alias_order, resolve_component_lets, resolve_model_lets};
pub(super) use model_parameters::{
    resolve_model_parameters, resolve_model_parameters_symbolically,
};

#[derive(Debug, Clone)]
pub(super) struct ResolvedParameter {
    pub(super) value: ValueLiteral,
    pub(super) expression: LoweringExpression,
    pub(super) lineage: ParameterLineage,
}

/// One static value after definition-time symbolic resolution.
///
/// The checked type is retained through symbolic resolution. `value`
/// is absent exactly when the expression depends on at least one required
/// public Parameter or zero-input property whose value belongs to a future occurrence.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct SymbolicParameterValue {
    pub(super) value: Option<ValueLiteral>,
    pub(super) value_type: ValueType,
    pub(super) expression: Option<LoweringExpression>,
    pub(super) lineage: Option<ParameterLineage>,
}

pub(super) type SymbolicParameterMap = BTreeMap<String, SymbolicParameterValue>;

fn component_parameter_type(
    file: &str,
    declaration: &ComponentParameterDecl,
    frames: &BTreeMap<String, SpatialSupport<String>>,
    values: &SymbolicParameterMap,
) -> Result<ValueType, Diagnostic> {
    let syntax = specialize_type(file, declaration.value_type(), values)?;
    frames::parameter_type(file, &syntax, declaration.default(), frames)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EvaluatedType {
    Known(ValueType),
    /// Only dimension is unresolved. The dimensionless projection retains
    /// scalar domain, component roles and frame for shared type checking.
    Deferred(ValueType),
}

impl EvaluatedType {
    fn dimension(&self) -> Option<DimExponents> {
        match self {
            Self::Known(value) => Some(value.dimension()),
            Self::Deferred(_) => None,
        }
    }

    fn value_type(&self) -> &ValueType {
        match self {
            Self::Known(value) | Self::Deferred(value) => value,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct EvaluatedParameter {
    value: Option<ValueLiteral>,
    value_type: EvaluatedType,
    bare_literal: bool,
    expression: Option<LoweringExpression>,
    lineage: Option<ParameterLineage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ParameterLineage {
    Constant,
    Parameter(FullElaborationIdentity),
    Derived,
}

impl ResolvedParameter {
    pub(super) fn model_parameter(
        value: ValueLiteral,
        identity: FullElaborationIdentity,
        internal_name: String,
        range: TextRange,
    ) -> Self {
        Self {
            value,
            expression: LoweringExpression::name(internal_name, range),
            lineage: ParameterLineage::Parameter(identity),
        }
    }
}

impl From<ResolvedParameter> for SymbolicParameterValue {
    fn from(parameter: ResolvedParameter) -> Self {
        Self {
            value: Some(parameter.value.clone()),
            value_type: parameter.value.value_type().clone(),
            expression: Some(parameter.expression),
            lineage: Some(parameter.lineage),
        }
    }
}

impl From<SymbolicParameterValue> for EvaluatedParameter {
    fn from(value: SymbolicParameterValue) -> Self {
        Self {
            value: value.value,
            value_type: EvaluatedType::Known(value.value_type),
            bare_literal: false,
            expression: value.expression,
            lineage: value.lineage,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequiredParameterPolicy {
    RejectUnbound,
    PublicIsFree,
}

type StaticContexts<'a> = (
    &'a mut dyn FnMut(&str) -> Option<Option<eqiora_schema::kernel::RationalTime>>,
    &'a mut dyn FnMut(&str) -> Option<SpatialSupport<String>>,
);

struct SymbolicParameterResolver<'a> {
    declaration_file: &'a str,
    declarations: BTreeMap<String, ComponentParameterDecl>,
    bindings: Option<bindings::Bindings>,
    bound_values: SymbolicParameterMap,
    resolved: SymbolicParameterMap,
    required_policy: RequiredParameterPolicy,
    frames: BTreeMap<String, SpatialSupport<String>>,
}

impl<'a> SymbolicParameterResolver<'a> {
    fn component_interface(
        declaration_file: &'a str,
        component: &'a ComponentDecl,
        records: &RecordContext,
    ) -> Result<Self, Vec<Diagnostic>> {
        Ok(Self {
            declaration_file,
            declarations: records::expand(
                declaration_file,
                parameter_declarations(component),
                records,
            )?,
            bindings: None,
            bound_values: BTreeMap::new(),
            resolved: BTreeMap::new(),
            required_policy: RequiredParameterPolicy::PublicIsFree,
            frames: super::supports::component_spatial_supports(declaration_file, component)?,
        })
    }

    fn instance(
        (declaration_file, binding_file): (&'a str, &str),
        component: &'a ComponentDecl,
        instance: &InstanceDecl,
        resolve_parent: impl FnMut(&str) -> Option<SymbolicParameterValue>,
        resolve_clock: &mut dyn FnMut(&str) -> Option<Option<eqiora_schema::kernel::RationalTime>>,
        resolve_frame: &mut dyn FnMut(&str) -> Option<SpatialSupport<String>>,
        record_contexts: (&RecordContext, &RecordContext),
    ) -> Result<Self, Vec<Diagnostic>> {
        let declarations =
            parameter_declaration_leaves(declaration_file, component, record_contexts.0)?;
        let frames = frames::instance_frames(declaration_file, component, instance, resolve_frame)?;
        let bindings = bindings::Bindings::freeze(
            binding_file,
            component,
            instance,
            resolve_parent,
            (&mut *resolve_clock, &mut *resolve_frame),
            record_contexts,
        )?;
        Ok(Self {
            declaration_file,
            declarations,
            bindings: Some(bindings),
            bound_values: BTreeMap::new(),
            resolved: BTreeMap::new(),
            required_policy: RequiredParameterPolicy::RejectUnbound,
            frames,
        })
    }

    fn resolve_all(
        mut self,
        resolve_clock: &mut dyn FnMut(&str) -> Option<Option<eqiora_schema::kernel::RationalTime>>,
    ) -> Result<SymbolicParameterMap, Vec<Diagnostic>> {
        let mut diagnostics = Vec::new();
        let mut definitions = BTreeMap::new();
        for (name, declaration) in &self.declarations {
            let binding = self
                .bindings
                .as_ref()
                .and_then(|bindings| bindings.expression(name));
            let mut dependencies = BTreeMap::new();
            let mut errors = Vec::new();
            // Call-site values belong to the parent. Only an unbound default and
            // the authored declaration type have dependencies in this child scope.
            let expressions = declaration
                .default()
                .filter(|_| binding.is_none() && !self.bound_values.contains_key(name))
                .into_iter()
                .chain(extent_expressions(declaration.value_type()));
            for expression in expressions {
                let (found, more) = collect_expression_dependencies(
                    self.declaration_file,
                    expression,
                    |name| self.declarations.contains_key(name),
                    ExpressionContext::Default,
                );
                dependencies.extend(found);
                errors.extend(more);
            }
            definitions.insert(
                name.clone(),
                ExpressionDefinition {
                    expression: binding.or(declaration.default()),
                    dependencies,
                    valid: errors.is_empty(),
                },
            );
            diagnostics.extend(errors);
        }
        let mut cyclic = BTreeSet::new();
        for cycle in expression_cycles(&definitions) {
            cyclic.extend(cycle.members);
            diagnostics.push(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                self.declaration_file,
                cycle.range,
                format!(
                    "component Parameter dependency cycle: {}",
                    cycle.path.join(" -> ")
                ),
            ));
        }
        let mut expression_work = 0usize;
        let expression_limit =
            crate::source_identity::LocalSourceIdentityLimits::default().max_expression_nodes;
        for name in expression_evaluation_order(&definitions, &cyclic) {
            let definition = &definitions[&name];
            if !definition.valid
                || definition
                    .dependencies
                    .keys()
                    .any(|name| !self.resolved.contains_key(name))
            {
                continue;
            }
            let declaration = &self.declarations[&name];
            for expression in definition
                .expression
                .into_iter()
                .chain(extent_expressions(declaration.value_type()))
            {
                let values = if self
                    .bindings
                    .as_ref()
                    .is_some_and(|bindings| bindings.expression(&name) == Some(expression))
                {
                    self.bindings.as_ref().unwrap().values()
                } else {
                    &self.resolved
                };
                let work = super::reductions::expanded_nodes(
                    self.declaration_file,
                    expression,
                    &mut |_| None,
                    values,
                    expression_limit.saturating_sub(expression_work),
                )
                .map_err(|error| vec![error])?;
                expression_work = expression_work
                    .checked_add(work)
                    .filter(|work| *work <= expression_limit)
                    .ok_or_else(|| {
                        vec![source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            self.declaration_file,
                            expression.range(),
                            "Parameter specialization exceeds the aggregate expression work limit",
                        )]
                    })?;
            }
            if self.required_policy == RequiredParameterPolicy::PublicIsFree {
                let mut deferred = false;
                for extent in extent_expressions(declaration.value_type()) {
                    match structural_extent(self.declaration_file, extent, &self.resolved) {
                        Ok(None) => deferred = true,
                        Ok(Some(_)) => {}
                        Err(error) => {
                            diagnostics.push(error);
                            deferred = true;
                        }
                    }
                }
                // No invented array shape enters the typed value map. The source
                // declaration remains pending until an actual occurrence supplies
                // every structural Parameter; an uninstantiated body fails closed.
                if deferred {
                    continue;
                }
            }
            let target = match component_parameter_type(
                self.declaration_file,
                declaration,
                &self.frames,
                &self.resolved,
            ) {
                Ok(target) => target,
                Err(error) => {
                    diagnostics.push(error);
                    continue;
                }
            };
            let mut structural = BTreeSet::new();
            for extent in extent_expressions(declaration.value_type()) {
                if let Some((_, dependencies)) =
                    structural_extent(self.declaration_file, extent, &self.resolved)
                        .map_err(|error| vec![error])?
                {
                    structural.extend(dependencies);
                }
            }
            if let Some(value) = self.bound_values.get(&name) {
                if value.value_type != target {
                    diagnostics.push(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.declaration_file,
                        declaration.range(),
                        "external checked value requires the exact specialized declared type",
                    ));
                } else {
                    let mut value = value.clone();
                    value.expression = value
                        .expression
                        .map(|expression| expression.with_structural_parameters(structural));
                    self.resolved.insert(name, value);
                }
                continue;
            }
            let Some(expression) = definition.expression else {
                if self.required_policy == RequiredParameterPolicy::PublicIsFree
                    && declaration.visibility() == VisibilitySyntax::Public
                {
                    self.resolved.insert(
                        name,
                        SymbolicParameterValue {
                            value: None,
                            value_type: target,
                            expression: None,
                            lineage: None,
                        },
                    );
                } else {
                    diagnostics.push(source_error(
                        codes::LANGUAGE_TYPE_ERROR,
                        self.declaration_file,
                        declaration.range(),
                        if declaration.visibility() == VisibilitySyntax::Private {
                            format!("required private Parameter `{name}` has no default")
                        } else {
                            format!("required Parameter `{name}` has no instance binding")
                        },
                    ));
                }
                continue;
            };
            let evaluated = if let Some(bindings) = &self.bindings
                && bindings.expression(&name).is_some()
            {
                bindings.evaluate(expression, target)
            } else {
                evaluate_initializer(
                    self.declaration_file,
                    expression,
                    ExpressionContext::Default,
                    &mut |dependency, range| {
                        self.resolved.get(dependency).cloned().ok_or_else(|| {
                            source_error(
                                codes::LANGUAGE_TYPE_ERROR,
                                self.declaration_file,
                                range,
                                format!("unknown component Parameter `{dependency}`"),
                            )
                        })
                    },
                    target.clone(),
                    "Parameter initializer",
                    (&mut *resolve_clock, &mut |name| {
                        self.frames.get(name).cloned()
                    }),
                )
                .and_then(|value| {
                    coerce_parameter_with_label(
                        self.declaration_file,
                        expression.range(),
                        value,
                        target,
                        "Parameter initializer",
                        true,
                    )
                })
            };
            match evaluated {
                Ok(mut value) => {
                    value.expression = value
                        .expression
                        .map(|expression| expression.with_structural_parameters(structural));
                    self.resolved.insert(name, value);
                }
                Err(error) => diagnostics.push(error),
            }
        }
        stable_sort(&mut diagnostics);
        if diagnostics.is_empty() {
            Ok(self.resolved)
        } else {
            Err(diagnostics)
        }
    }
}

fn parameter_declarations(component: &ComponentDecl) -> BTreeMap<String, ComponentParameterDecl> {
    component
        .signature()
        .iter()
        .filter_map(|item| match item {
            eqiora_lang::SignatureItem::Parameter(value) => {
                Some((value.name().to_owned(), value.clone()))
            }
            _ => None,
        })
        .chain(component.items().iter().filter_map(|item| match item {
            ComponentItem::Parameter(value) => Some((value.name().to_owned(), value.clone())),
            _ => None,
        }))
        .collect()
}

pub(in crate::hierarchy) fn parameter_declaration_leaves(
    file: &str,
    component: &ComponentDecl,
    records: &RecordContext,
) -> Result<BTreeMap<String, ComponentParameterDecl>, Vec<Diagnostic>> {
    records::expand(file, parameter_declarations(component), records)
}

/// Resolve a reusable Component's Parameter interface without inventing an
/// occurrence value for any required public Parameter.
pub(super) fn resolve_component_parameters_symbolically(
    declaration_file: &str,
    component: &ComponentDecl,
    mut resolve_clock: impl FnMut(&str) -> Option<Option<eqiora_schema::kernel::RationalTime>>,
    records: &RecordContext,
) -> Result<SymbolicParameterMap, Vec<Diagnostic>> {
    SymbolicParameterResolver::component_interface(declaration_file, component, records)?
        .resolve_all(&mut resolve_clock)
}

pub(in crate::hierarchy) fn resolve_external_parameters(
    file: &str,
    component: &ComponentDecl,
    bindings: &[super::ExternalParameterBinding],
    records: &RecordContext,
) -> Result<SymbolicParameterMap, Vec<Diagnostic>> {
    let mut resolver = SymbolicParameterResolver::component_interface(file, component, records)?;
    resolver.required_policy = RequiredParameterPolicy::RejectUnbound;
    resolver.bound_values = bindings
        .iter()
        .map(|binding| {
            (
                binding.parameter().to_owned(),
                SymbolicParameterValue {
                    value: Some(binding.value().clone()),
                    value_type: binding.value().value_type().clone(),
                    expression: Some(LoweringExpression::literal(
                        binding.value().clone(),
                        component.range(),
                    )),
                    lineage: Some(ParameterLineage::Constant),
                },
            )
        })
        .collect();
    resolver.resolve_all(&mut |name| super::clocks::component(file, component, name))
}

pub(super) struct ParameterResolver<'a> {
    inner: SymbolicParameterResolver<'a>,
}

impl<'a> ParameterResolver<'a> {
    pub(super) fn new(
        (declaration_file, binding_file): (&'a str, &str),
        component: &'a ComponentDecl,
        instance: &InstanceDecl,
        mut resolve_parent: impl FnMut(&str) -> Option<ResolvedParameter>,
        mut resolve_clock: impl FnMut(&str) -> Option<Option<eqiora_schema::kernel::RationalTime>>,
        mut resolve_frame: impl FnMut(&str) -> Option<SpatialSupport<String>>,
        record_contexts: (&RecordContext, &RecordContext),
    ) -> Result<Self, Vec<Diagnostic>> {
        SymbolicParameterResolver::instance(
            (declaration_file, binding_file),
            component,
            instance,
            |name| resolve_parent(name).map(SymbolicParameterValue::from),
            &mut resolve_clock,
            &mut resolve_frame,
            record_contexts,
        )
        .map(|inner| Self { inner })
    }

    pub(super) fn resolve_all(
        self,
        mut resolve_clock: impl FnMut(&str) -> Option<Option<eqiora_schema::kernel::RationalTime>>,
    ) -> Result<BTreeMap<String, ResolvedParameter>, Vec<Diagnostic>> {
        self.inner
            .resolve_all(&mut resolve_clock)
            .and_then(concrete_parameters)
    }
}

fn concrete_parameters(
    parameters: SymbolicParameterMap,
) -> Result<BTreeMap<String, ResolvedParameter>, Vec<Diagnostic>> {
    parameters
        .into_iter()
        .map(|(name, parameter)| {
            let value = parameter.value.ok_or_else(|| {
                vec![hierarchy_error(format!(
                    "concrete instance Parameter `{name}` remained symbolic"
                ))]
            })?;
            let expression = parameter.expression.ok_or_else(|| {
                vec![hierarchy_error(format!(
                    "concrete instance Parameter `{name}` has no resolved expression"
                ))]
            })?;
            let lineage = parameter.lineage.ok_or_else(|| {
                vec![hierarchy_error(format!(
                    "concrete instance Parameter `{name}` has no exact binding lineage"
                ))]
            })?;
            Ok((
                name,
                ResolvedParameter {
                    value,
                    expression,
                    lineage,
                },
            ))
        })
        .collect()
}

fn combine_parameters(
    file: &str,
    range: TextRange,
    operator: BinaryOp,
    left: EvaluatedParameter,
    right: EvaluatedParameter,
) -> Result<EvaluatedParameter, Diagnostic> {
    let lineage = combine_lineages(left.lineage.clone(), right.lineage.clone());
    let exponent = if operator == BinaryOp::Pow {
        require_dimensionless_exponent(file, range, &right.value_type)?;
        if matches!(
            right.lineage,
            Some(ParameterLineage::Parameter(_) | ParameterLineage::Derived)
        ) {
            return Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                range,
                "compile-time power exponent cannot depend on a live Parameter",
            ));
        }
        right
            .value
            .as_ref()
            .map(|value| {
                value
                    .real_scalar_value()
                    .and_then(|quantity| exact_i32(quantity.value()))
                    .ok_or_else(|| {
                        source_error(
                            codes::LANGUAGE_TYPE_ERROR,
                            file,
                            range,
                            "compile-time power exponent must be an exact i32 integer",
                        )
                    })
            })
            .transpose()?
    } else {
        None
    };
    let value_type = combine_types(
        file,
        range,
        operator,
        left.value_type,
        right.value_type,
        exponent,
    )?;
    let expression = match (left.expression, right.expression) {
        (Some(left), Some(right)) => Some(LoweringExpression::binary(operator, left, right, range)),
        _ => None,
    };
    let value = match (left.value, right.value) {
        (Some(left), Some(right)) => Some(
            crate::typed_values::binary(
                operator,
                &left,
                &right,
                value_type.value_type().clone(),
                exponent,
            )
            .map_err(|message| source_error(codes::LANGUAGE_TYPE_ERROR, file, range, message))?,
        ),
        _ => None,
    };
    Ok(EvaluatedParameter {
        value,
        value_type,
        bare_literal: false,
        expression,
        lineage,
    })
}

fn transform_lineage(lineage: Option<ParameterLineage>) -> Option<ParameterLineage> {
    match lineage {
        Some(ParameterLineage::Constant) => Some(ParameterLineage::Constant),
        Some(ParameterLineage::Parameter(_) | ParameterLineage::Derived) => {
            Some(ParameterLineage::Derived)
        }
        None => None,
    }
}

fn combine_lineages(
    left: Option<ParameterLineage>,
    right: Option<ParameterLineage>,
) -> Option<ParameterLineage> {
    match (left, right) {
        (Some(ParameterLineage::Constant), Some(ParameterLineage::Constant)) => {
            Some(ParameterLineage::Constant)
        }
        (Some(ParameterLineage::Parameter(_) | ParameterLineage::Derived), _)
        | (_, Some(ParameterLineage::Parameter(_) | ParameterLineage::Derived)) => {
            Some(ParameterLineage::Derived)
        }
        _ => None,
    }
}

fn combine_types(
    file: &str,
    range: TextRange,
    operator: BinaryOp,
    left: EvaluatedType,
    right: EvaluatedType,
    exponent: Option<i32>,
) -> Result<EvaluatedType, Diagnostic> {
    use eqiora_schema::kernel::typing::{self, ExpressionType};
    let left_dimension = left.dimension();
    let right_dimension = right.dimension();
    let dimension_known = match operator {
        BinaryOp::Add | BinaryOp::Sub => left_dimension.or(right_dimension).is_some(),
        BinaryOp::Mul | BinaryOp::Div => left_dimension.is_some() && right_dimension.is_some(),
        _ if crate::lower::comparison_operator(operator).is_some()
            || matches!(operator, BinaryOp::And | BinaryOp::Or) =>
        {
            unreachable!("predicates use checked scalar typing")
        }
        BinaryOp::Pow => {
            exponent == Some(0)
                || (exponent.is_some() && left_dimension.is_some())
                || left_dimension == Some(DimExponents::DIMENSIONLESS)
        }
        _ => unreachable!("predicate operator"),
    };
    let projected = |value: &EvaluatedType, dimension| {
        value
            .value_type()
            .clone()
            .with_dimension(dimension)
            .map(|value| ExpressionType::<()>::new(value, None))
            .map_err(|error| {
                source_error(codes::LANGUAGE_TYPE_ERROR, file, range, error.to_string())
            })
    };
    let fallback = DimExponents::DIMENSIONLESS;
    let result = match operator {
        BinaryOp::Add | BinaryOp::Sub => {
            let left = projected(
                &left,
                left_dimension.or(right_dimension).unwrap_or(fallback),
            )?;
            let right = projected(
                &right,
                right_dimension.or(left_dimension).unwrap_or(fallback),
            )?;
            if operator == BinaryOp::Add {
                left.sum(right)
            } else {
                typing::additive(&left, &right)
            }
        }
        BinaryOp::Mul | BinaryOp::Div => {
            let left = projected(
                &left,
                if dimension_known {
                    left_dimension.unwrap()
                } else {
                    fallback
                },
            )?;
            let right = projected(
                &right,
                if dimension_known {
                    right_dimension.unwrap()
                } else {
                    fallback
                },
            )?;
            if operator == BinaryOp::Mul {
                typing::multiply(&left, &right)
            } else {
                typing::divide(&left, &right)
            }
        }
        _ if crate::lower::comparison_operator(operator).is_some()
            || matches!(operator, BinaryOp::And | BinaryOp::Or) =>
        {
            unreachable!("predicates use checked scalar typing")
        }
        BinaryOp::Pow => typing::power(
            &projected(&left, left_dimension.unwrap_or(fallback))?,
            exponent.unwrap_or(1),
        ),
        _ => unreachable!("predicate operator"),
    };
    result
        .map(|value| {
            if dimension_known {
                EvaluatedType::Known(value.value_type)
            } else {
                EvaluatedType::Deferred(
                    value
                        .value_type
                        .with_dimension(fallback)
                        .expect("dimensionless projection of checked arithmetic type"),
                )
            }
        })
        .map_err(|error| {
            if matches!(error, typing::TypeViolation::DimensionOverflow { .. }) {
                constant_dimension_overflow(file, range)
            } else {
                source_error(codes::LANGUAGE_TYPE_ERROR, file, range, error.to_string())
            }
        })
}

fn require_dimensionless_exponent(
    file: &str,
    range: TextRange,
    dimension: &EvaluatedType,
) -> Result<(), Diagnostic> {
    match dimension {
        EvaluatedType::Known(value_type) | EvaluatedType::Deferred(value_type)
            if value_type.dimension() != DimExponents::DIMENSIONLESS
                || value_type.scalar_domain() != ScalarDomain::Real
                || !value_type.shape().is_scalar() =>
        {
            Err(source_error(
                codes::LANGUAGE_TYPE_ERROR,
                file,
                range,
                "compile-time power exponent must be a real dimensionless scalar",
            ))
        }
        EvaluatedType::Known(_) | EvaluatedType::Deferred(_) => Ok(()),
    }
}

fn exact_i32(value: f64) -> Option<i32> {
    (value.fract() == 0.0 && value >= f64::from(i32::MIN) && value <= f64::from(i32::MAX))
        .then_some(value as i32)
}

pub(super) fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn constant_dimension_overflow(file: &str, range: TextRange) -> Diagnostic {
    source_error(
        codes::LANGUAGE_TYPE_ERROR,
        file,
        range,
        "compile-time dimension arithmetic exceeds rational exponent bounds",
    )
}

#[cfg(test)]
mod tests;

mod static_values;
pub(crate) use expression_eval::exact_signed_literal;
pub(in crate::hierarchy) use static_values::index_set_extent;
pub(crate) use static_values::{closed_value, infer_closed_value};
pub(in crate::hierarchy) use static_values::{
    static_index, static_slice, structural_extent, structural_index,
};

pub(in crate::hierarchy) fn resolve_instance_parameters_symbolically(
    (declaration_file, binding_file): (&str, &str),
    component: &ComponentDecl,
    instance: &InstanceDecl,
    parent: &SymbolicParameterMap,
    resolve_clock: &mut dyn FnMut(&str) -> Option<Option<eqiora_schema::kernel::RationalTime>>,
    resolve_frame: &mut dyn FnMut(&str) -> Option<SpatialSupport<String>>,
    record_contexts: (&RecordContext, &RecordContext),
) -> Result<SymbolicParameterMap, Vec<Diagnostic>> {
    SymbolicParameterResolver::instance(
        (declaration_file, binding_file),
        component,
        instance,
        |name| parent.get(name).cloned(),
        resolve_clock,
        resolve_frame,
        record_contexts,
    )?
    .resolve_all(resolve_clock)
}
