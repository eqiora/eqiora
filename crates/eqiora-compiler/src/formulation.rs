//! Typed, compiler-owned projections of authored mathematical formulations.

use std::collections::BTreeMap;

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, Id, RawId, ValueShape};
use eqiora_graph::{EdgeKind, Op, Transaction};
use eqiora_lang::{BinaryOp, ComponentDecl, Expr, ExprKind, NamePath, TextRange, UnaryOp};
use eqiora_schema::kernel::{KernelNode, ParameterDef};

use crate::diagnostics::source_error;
use crate::dimensions::length_dimension;
use crate::lower::ModelSymbols;
use crate::source_identity::formulation::AuthoredFormSourceIdentity;

mod expression;
mod index;
mod interval;
use index::KernelIndex;
mod restriction;
mod typing;
mod wire;

pub use interval::check_derived_interval_conservation;
pub use wire::{AuthoredFormExpressionV1, AuthoredFormulationProjection};

/// One typed expression in an authored Formulation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AuthoredFormExpression {
    kind: AuthoredFormExpressionKind,
    dimension: DimExponents,
    shape: ValueShape,
    support: Option<Id<kinds::Domain>>,
}

/// Closed expression vocabulary accepted by the first scalar-primal compiler.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub(crate) enum AuthoredFormExpressionKind {
    /// Dimensionless scalar literal.
    Number(f64),
    /// Scalar Field value.
    Field(Id<kinds::Field>),
    /// Scalar Parameter value.
    Parameter(Id<kinds::Parameter>),
    /// One physical Cartesian coordinate in the Relation Domain.
    Coordinate(usize),
    /// Scalar test function associated with one trial Field.
    Test(Id<kinds::Field>),
    /// Arithmetic negation.
    Neg(Box<AuthoredFormExpression>),
    /// One typed binary operation.
    Binary {
        /// Mathematical operation.
        operator: BinaryOp,
        /// Left operand.
        left: Box<AuthoredFormExpression>,
        /// Right operand.
        right: Box<AuthoredFormExpression>,
    },
    /// Integer power of a scalar.
    Pow(Box<AuthoredFormExpression>, i32),
    /// Spatial gradient.
    Gradient(Box<AuthoredFormExpression>),
    Divergence(Box<AuthoredFormExpression>),
    SymmetricPart(Box<AuthoredFormExpression>),
    Frobenius(Box<AuthoredFormExpression>, Box<AuthoredFormExpression>),
    /// Scalar sine.
    Sin(Box<AuthoredFormExpression>),
    /// Euclidean inner product of equal vectors.
    Dot(Box<AuthoredFormExpression>, Box<AuthoredFormExpression>),
    /// Volume integral over one exact Domain.
    Integrate {
        /// Integration Domain.
        domain: Id<kinds::Domain>,
        /// Scalar integrand on that Domain.
        integrand: Box<AuthoredFormExpression>,
    },
}

/// One compiler-owned authored Formulation retained beside a freshly compiled Model.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledAuthoredFormulation {
    relations: Vec<Id<kinds::Relation>>,
    domain: Id<kinds::Domain>,
    trials: Vec<Id<kinds::Field>>,
    projection: AuthoredFormulationProjection,
    file: String,
    range: TextRange,
}

impl CompiledAuthoredFormulation {
    /// Identity of the Component's canonical authored-form source.
    #[must_use]
    pub fn source_identity(&self) -> &str {
        self.projection.source_identity()
    }

    /// Exact Relations represented by the explicitly owned equations.
    #[must_use]
    pub fn relations(&self) -> &[Id<kinds::Relation>] {
        &self.relations
    }

    /// Exact integration and Relation Domain.
    #[must_use]
    pub const fn domain(&self) -> Id<kinds::Domain> {
        self.domain
    }

    /// Exact trial Fields associated with the named test inventory.
    #[must_use]
    pub fn trials(&self) -> &[Id<kinds::Field>] {
        &self.trials
    }

    /// Exact canonical projection consumed by resolution and Plan replay.
    #[must_use]
    pub const fn projection(&self) -> &AuthoredFormulationProjection {
        &self.projection
    }

    /// Source filename used for diagnostics and inspection.
    #[must_use]
    pub fn file(&self) -> &str {
        &self.file
    }

    /// Exact source range of the Formulation declaration.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }
}

pub(crate) fn compile_component_formulations(
    file: &str,
    component: &ComponentDecl,
    symbols: &ModelSymbols,
    transaction: &Transaction,
    geometry: &eqiora_geometry::CanonicalGeometryV1,
    supports: &[crate::external::ExternalGeometrySupportBinding],
) -> Result<Vec<CompiledAuthoredFormulation>, Vec<Diagnostic>> {
    if component.formulations().len() == 0 {
        return Ok(Vec::new());
    }
    if component.formulations().len() != 1 {
        return Err(vec![source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            component.range(),
            "the Formulation compiler accepts exactly one named form per Component",
        )]);
    }
    let source_identity = AuthoredFormSourceIdentity::from_component(component)
        .map_err(|diagnostic| vec![diagnostic])?;
    let index = KernelIndex::new(transaction);
    component
        .formulations()
        .map(|(name, relations, equations, range)| {
            let binding = component
                .formulation_binding(name)
                .expect("retained binder");
            if matches!(binding, eqiora_lang::FormulationBinding::Interval { .. }) {
                let ([relation], [(left, right)]) = (relations, equations) else {
                    return Err(vec![error(
                        file,
                        range,
                        "interval requires exactly one Law and equation",
                    )]);
                };
                return interval::compile(
                    file,
                    (name, relation, left, right, range),
                    binding,
                    source_identity,
                    symbols,
                    &index,
                    geometry,
                )
                .map_err(|e| vec![e]);
            }
            let eqiora_lang::FormulationBinding::WeakTests { tests } = binding else {
                unreachable!()
            };
            compile_weak(
                file,
                name,
                relations,
                equations,
                tests,
                range,
                source_identity,
                symbols,
                &index,
                geometry,
                supports,
            )
            .map_err(|e| vec![e])
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn compile_weak(
    file: &str,
    name: &str,
    relation_names: &[String],
    equations: &[(Expr, Expr)],
    tests: &[(String, String, Vec<String>)],
    range: TextRange,
    source_identity: AuthoredFormSourceIdentity,
    symbols: &ModelSymbols,
    index: &KernelIndex<'_>,
    geometry: &eqiora_geometry::CanonicalGeometryV1,
    supports: &[crate::external::ExternalGeometrySupportBinding],
) -> Result<CompiledAuthoredFormulation, Diagnostic> {
    if relation_names.len() != equations.len()
        || tests.len() != equations.len()
        || equations.is_empty()
        || equations.len() > 8
    {
        return Err(error(
            file,
            range,
            "form requires one explicit Relation and test per equation",
        ));
    }
    let relations = relation_names
        .iter()
        .map(|name| {
            resolve_symbol(file, range, name, symbols)?
                .downcast::<kinds::Relation>()
                .ok_or_else(|| error(file, range, "form owner is not a Relation"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let domain = index
        .applies_on
        .get(&relations[0].erase())
        .copied()
        .and_then(RawId::downcast::<kinds::Domain>)
        .ok_or_else(|| error(file, range, "form has no exact Domain"))?;
    if relations
        .iter()
        .any(|r| index.applies_on.get(&r.erase()) != Some(&domain.erase()))
    {
        return Err(error(file, range, "form Relations have foreign support"));
    }
    let mut named_tests = BTreeMap::new();
    let mut restrictions = Vec::new();
    let mut trials = Vec::new();
    for (test, trial, boundaries) in tests {
        let declaration_suffix = format!(".{test}");
        if symbols.get(test).is_some()
            || symbols
                .iter()
                .any(|(candidate, _)| candidate.ends_with(&declaration_suffix))
            || named_tests.insert(test.as_str(), trial.as_str()).is_some()
        {
            return Err(error(
                file,
                range,
                "test name must not shadow a declaration or another test",
            ));
        }
        let trial = resolve_symbol(file, range, trial, symbols)?
            .downcast::<kinds::Field>()
            .ok_or_else(|| error(file, range, "test trial is not a Field"))?;
        if trials.contains(&trial) {
            return Err(error(file, range, "duplicate trial test"));
        }
        let zero_on = if tests.len() > 1 && boundaries.is_empty() {
            vec![]
        } else {
            restriction::resolve(
                file,
                (range, boundaries),
                domain.erase(),
                symbols,
                index,
                supports,
            )?
        };
        restrictions.push((test.clone(), trial.ulid().to_string(), zero_on));
        trials.push(trial);
    }
    let mut context = ExpressionContext {
        file,
        symbols,
        index,
        ambient_dimension: geometry.ambient_dimension(),
        topological_dimension: geometry.topological_dimension(),
        relation_domain: domain,
        tests: named_tests,
        used_tests: std::collections::BTreeSet::new(),
    };
    let mut compiled = Vec::new();
    for ((left, right), relation) in equations.iter().zip(&relations) {
        let left = context.compile_root(left)?;
        let right = context.compile_root(right)?;
        let zero =
            |v: &AuthoredFormExpression| matches!(v.kind, AuthoredFormExpressionKind::Number(0.0));
        if (left.dimension != right.dimension && !zero(&left) && !zero(&right))
            || left.shape != right.shape
        {
            return Err(error(
                file,
                range,
                "Formulation equality sides must have identical dimension and shape",
            ));
        }
        compiled.push((
            relation.ulid().to_string(),
            wire::expression(&left),
            wire::expression(&right),
        ));
    }
    if context.used_tests.len() != tests.len() {
        return Err(error(file, range, "form must consume every declared test"));
    }
    let projection = AuthoredFormulationProjection::encode_weak(
        source_identity.to_string(),
        name.into(),
        domain.erase(),
        restrictions,
        compiled,
    )?;
    Ok(CompiledAuthoredFormulation {
        relations,
        domain,
        trials,
        projection,
        file: file.into(),
        range,
    })
}

struct ExpressionContext<'a> {
    file: &'a str,
    symbols: &'a ModelSymbols,
    index: &'a KernelIndex<'a>,
    ambient_dimension: usize,
    topological_dimension: usize,
    relation_domain: Id<kinds::Domain>,
    tests: BTreeMap<&'a str, &'a str>,
    used_tests: std::collections::BTreeSet<String>,
}

fn parameter_expression(parameter: &ParameterDef) -> AuthoredFormExpression {
    typed(
        AuthoredFormExpressionKind::Parameter(parameter.id()),
        parameter.value_type().dimension(),
        ValueShape::scalar(),
        None,
    )
}

fn typed(
    kind: AuthoredFormExpressionKind,
    dimension: DimExponents,
    shape: ValueShape,
    support: Option<Id<kinds::Domain>>,
) -> AuthoredFormExpression {
    AuthoredFormExpression {
        kind,
        dimension,
        shape,
        support,
    }
}

fn binary(
    operator: BinaryOp,
    left: AuthoredFormExpression,
    right: AuthoredFormExpression,
) -> AuthoredFormExpressionKind {
    AuthoredFormExpressionKind::Binary {
        operator,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn resolve_symbol(
    file: &str,
    range: TextRange,
    name: &str,
    symbols: &ModelSymbols,
) -> Result<RawId, Diagnostic> {
    if let Some(id) = symbols.get(name) {
        return Ok(id);
    }
    let suffix = format!(".{name}");
    let mut matches = symbols
        .iter()
        .filter_map(|(candidate, id)| candidate.ends_with(&suffix).then_some(id));
    let Some(id) = matches.next() else {
        return Err(error(
            file,
            range,
            format!("unknown Formulation symbol `{name}`"),
        ));
    };
    if matches.next().is_some() {
        return Err(error(
            file,
            range,
            format!("ambiguous Formulation symbol `{name}`"),
        ));
    }
    Ok(id)
}

fn unqualified_callee<'a>(
    file: &str,
    range: TextRange,
    callee: &'a NamePath,
) -> Result<&'a str, Diagnostic> {
    if crate::math::is_namespaced(callee) && !crate::math::is_function(callee) {
        Err(error(
            file,
            range,
            format!("unknown compiler-owned scalar mathematics member `{callee}`"),
        ))
    } else if callee.is_qualified() && !crate::math::is_function(callee) {
        Err(error(
            file,
            range,
            "Formulation operators must be unqualified",
        ))
    } else {
        Ok(callee.as_str())
    }
}

fn merge_support(
    file: &str,
    range: TextRange,
    left: Option<Id<kinds::Domain>>,
    right: Option<Id<kinds::Domain>>,
) -> Result<Option<Id<kinds::Domain>>, Diagnostic> {
    match (left, right) {
        (Some(left), Some(right)) if left != right => Err(error(
            file,
            range,
            "expression operands have different spatial supports",
        )),
        (Some(value), _) | (_, Some(value)) => Ok(Some(value)),
        (None, None) => Ok(None),
    }
}

fn require_scalar(
    file: &str,
    range: TextRange,
    value: &AuthoredFormExpression,
) -> Result<(), Diagnostic> {
    if value.shape.is_scalar() {
        Ok(())
    } else {
        Err(error(file, range, "operator requires a scalar expression"))
    }
}

fn integer_literal(expression: &Expr) -> Option<i32> {
    crate::dimensions::integer_literal(expression)
}

fn error(file: &str, range: TextRange, message: impl Into<String>) -> Diagnostic {
    source_error(codes::LANGUAGE_TYPE_ERROR, file, range, message)
}
