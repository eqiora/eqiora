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

mod restriction;
mod wire;

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
    relation: Id<kinds::Relation>,
    domain: Id<kinds::Domain>,
    trial: Id<kinds::Field>,
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

    /// Relation represented by this form.
    #[must_use]
    pub const fn relation(&self) -> Id<kinds::Relation> {
        self.relation
    }

    /// Exact integration and Relation Domain.
    #[must_use]
    pub const fn domain(&self) -> Id<kinds::Domain> {
        self.domain
    }

    /// Scalar trial Field named by every test function.
    #[must_use]
    pub const fn trial(&self) -> Id<kinds::Field> {
        self.trial
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
    geometry_dimensions: (usize, usize),
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
            "the scalar-primal Formulation compiler accepts exactly one form per Component",
        )]);
    }
    let source_identity = AuthoredFormSourceIdentity::from_component(component)
        .map_err(|diagnostic| vec![diagnostic])?;
    let index = KernelIndex::new(transaction);
    component
        .formulations()
        .map(|(name, relation, left, right, range)| {
            let (test, trial, zero_on) = component
                .formulation_test(name)
                .expect("retained form test");
            let form = FormSource {
                name,
                relation,
                test,
                trial,
                zero_on,
                left,
                right,
                range,
            };
            compile_formulation(
                file,
                form,
                source_identity,
                symbols,
                &index,
                geometry_dimensions,
                supports,
            )
            .map_err(|diagnostic| vec![diagnostic])
        })
        .collect()
}

struct KernelIndex<'a> {
    nodes: BTreeMap<RawId, &'a KernelNode>,
    applies_on: BTreeMap<RawId, RawId>,
    defined_on: BTreeMap<RawId, RawId>,
    boundary_of: BTreeMap<RawId, RawId>,
}

impl<'a> KernelIndex<'a> {
    fn new(transaction: &'a Transaction) -> Self {
        let mut nodes = BTreeMap::new();
        let mut applies_on = BTreeMap::new();
        let mut defined_on = BTreeMap::new();
        let mut boundary_of = BTreeMap::new();
        for op in transaction.ops() {
            match op {
                Op::DefineKernelNode { node } => {
                    nodes.insert(node.id(), node);
                }
                Op::Connect {
                    from,
                    to,
                    edge: EdgeKind::AppliesOn,
                } => {
                    applies_on.insert(*from, *to);
                }
                Op::Connect {
                    from,
                    to,
                    edge: EdgeKind::DefinedOn,
                } if to.downcast::<kinds::Domain>().is_some() => {
                    defined_on.insert(*from, *to);
                }
                Op::Connect {
                    from,
                    to,
                    edge: EdgeKind::BoundaryOf,
                } => {
                    boundary_of.insert(*from, *to);
                }
                _ => {}
            }
        }
        Self {
            nodes,
            applies_on,
            defined_on,
            boundary_of,
        }
    }
}

struct FormSource<'a> {
    name: &'a str,
    relation: &'a str,
    test: &'a str,
    trial: &'a str,
    zero_on: &'a [String],
    left: &'a Expr,
    right: &'a Expr,
    range: TextRange,
}

fn compile_formulation(
    file: &str,
    form: FormSource<'_>,
    source_identity: AuthoredFormSourceIdentity,
    symbols: &ModelSymbols,
    index: &KernelIndex<'_>,
    geometry_dimensions: (usize, usize),
    supports: &[crate::external::ExternalGeometrySupportBinding],
) -> Result<CompiledAuthoredFormulation, Diagnostic> {
    let FormSource {
        name,
        relation: relation_name,
        test: test_name,
        trial: trial_name,
        zero_on: boundaries,
        left: left_source,
        right: right_source,
        range,
    } = form;
    let (ambient_dimension, topological_dimension) = geometry_dimensions;
    if resolve_symbol(file, range, test_name, symbols).is_ok() {
        return Err(error(
            file,
            range,
            "test name must not shadow a Model declaration",
        ));
    }
    let relation_raw = resolve_symbol(file, range, relation_name, symbols)?;
    let relation = relation_raw
        .downcast::<kinds::Relation>()
        .ok_or_else(|| error(file, range, format!("`{relation_name}` is not a Relation")))?;
    let domain = index
        .applies_on
        .get(&relation_raw)
        .copied()
        .and_then(RawId::downcast::<kinds::Domain>)
        .ok_or_else(|| {
            error(
                file,
                range,
                "Formulation Relation has no exact AppliesOn Domain",
            )
        })?;
    let mut context = ExpressionContext {
        file,
        symbols,
        index,
        ambient_dimension,
        topological_dimension,
        relation_domain: domain,
        trial: None,
        test_name,
        trial_name,
    };
    let left = context.compile_root(left_source)?;
    let right = context.compile_root(right_source)?;
    if left.dimension != right.dimension || left.shape != right.shape {
        return Err(error(
            file,
            range,
            "Formulation equality sides must have identical dimension and shape",
        ));
    }
    let trial = context.trial.ok_or_else(|| {
        error(
            file,
            range,
            "scalar weak Formulation must use its declared test function",
        )
    })?;
    let zero_on = restriction::resolve(
        file,
        (range, boundaries),
        domain.erase(),
        symbols,
        index,
        supports,
    )?;
    let projection = AuthoredFormulationProjection::encode(
        source_identity.to_string(),
        relation.erase(),
        domain.erase(),
        trial.erase(),
        (name.to_owned(), test_name.to_owned(), zero_on),
        &left,
        &right,
    );
    Ok(CompiledAuthoredFormulation {
        relation,
        domain,
        trial,
        projection,
        file: file.to_owned(),
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
    trial: Option<Id<kinds::Field>>,
    test_name: &'a str,
    trial_name: &'a str,
}

impl ExpressionContext<'_> {
    fn compile_root(&mut self, expression: &Expr) -> Result<AuthoredFormExpression, Diagnostic> {
        let value = self.compile(expression)?;
        match value.kind {
            AuthoredFormExpressionKind::Integrate { .. } => Ok(value),
            _ => Err(error(
                self.file,
                expression.range(),
                "each scalar-primal equality side must be one integrate(domain, expression) call",
            )),
        }
    }

    fn compile(&mut self, expression: &Expr) -> Result<AuthoredFormExpression, Diagnostic> {
        match expression.kind() {
            ExprKind::Number(value) => Ok(typed(
                AuthoredFormExpressionKind::Number(
                    value
                        .to_f64()
                        .map_err(|e| error(self.file, expression.range(), e.message()))?,
                ),
                DimExponents::DIMENSIONLESS,
                ValueShape::scalar(),
                None,
            )),
            ExprKind::Name(name) if name == self.test_name => self.compile_test(expression),
            ExprKind::Name(name) => self.compile_name(expression, name),
            ExprKind::Path(path) => match crate::math::constant(path) {
                Some(value) => Ok(typed(
                    AuthoredFormExpressionKind::Number(value),
                    DimExponents::DIMENSIONLESS,
                    ValueShape::scalar(),
                    None,
                )),
                None if crate::math::is_namespaced(path) => Err(error(
                    self.file,
                    expression.range(),
                    format!("unknown compiler-owned scalar mathematics member `{path}`"),
                )),
                None => Err(error(
                    self.file,
                    expression.range(),
                    "qualified names are not accepted in scalar-primal forms",
                )),
            },
            ExprKind::BoundaryPortSelection { .. } => Err(error(
                self.file,
                expression.range(),
                "boundary-selected names are not accepted in scalar-primal forms",
            )),
            ExprKind::Unary {
                op: UnaryOp::Neg,
                value,
            } => {
                let value = self.compile(value)?;
                Ok(typed(
                    AuthoredFormExpressionKind::Neg(Box::new(value.clone())),
                    value.dimension,
                    value.shape.clone(),
                    value.support,
                ))
            }
            ExprKind::Binary { op, left, right } => {
                self.compile_binary(expression, *op, left, right)
            }
            ExprKind::Call {
                callee,
                arguments: eqiora_lang::CallArguments::Positional(arguments),
            } => self.compile_call(expression, callee, arguments),
            _ => Err(error(
                self.file,
                expression.range(),
                "unsupported scalar-primal expression",
            )),
        }
    }

    fn compile_name(
        &self,
        expression: &Expr,
        name: &str,
    ) -> Result<AuthoredFormExpression, Diagnostic> {
        let raw = resolve_symbol(self.file, expression.range(), name, self.symbols)?;
        match self.index.nodes.get(&raw).copied() {
            Some(KernelNode::Field(field))
                if field.shape().is_scalar()
                    && field.value_type().scalar_domain() == eqiora_core::ScalarDomain::Real =>
            {
                let support = self.field_support(expression, raw)?;
                Ok(typed(
                    AuthoredFormExpressionKind::Field(field.id()),
                    field.dimension(),
                    ValueShape::scalar(),
                    Some(support),
                ))
            }
            Some(KernelNode::Field(_)) => Err(error(
                self.file,
                expression.range(),
                "scalar-primal forms accept only real scalar Fields",
            )),
            Some(KernelNode::Parameter(parameter)) if parameter.real_scalar_value().is_some() => {
                Ok(parameter_expression(parameter))
            }
            Some(KernelNode::Parameter(_)) => Err(error(
                self.file,
                expression.range(),
                "scalar-primal forms accept only real scalar Parameters",
            )),
            _ => Err(error(
                self.file,
                expression.range(),
                format!("`{name}` is not a scalar Field or Parameter"),
            )),
        }
    }

    fn field_support(
        &self,
        expression: &Expr,
        field: RawId,
    ) -> Result<Id<kinds::Domain>, Diagnostic> {
        let support = self
            .index
            .defined_on
            .get(&field)
            .copied()
            .and_then(RawId::downcast::<kinds::Domain>)
            .ok_or_else(|| {
                error(
                    self.file,
                    expression.range(),
                    "Formulation Field has no exact DefinedOn Domain",
                )
            })?;
        if support != self.relation_domain {
            return Err(error(
                self.file,
                expression.range(),
                "Formulation Field support differs from the Relation Domain",
            ));
        }
        Ok(support)
    }

    fn compile_binary(
        &mut self,
        expression: &Expr,
        op: BinaryOp,
        left: &Expr,
        right: &Expr,
    ) -> Result<AuthoredFormExpression, Diagnostic> {
        if op == BinaryOp::Pow {
            let base = self.compile(left)?;
            require_scalar(self.file, left.range(), &base)?;
            let exponent = integer_literal(right).ok_or_else(|| {
                error(
                    self.file,
                    right.range(),
                    "Formulation power requires an integer literal",
                )
            })?;
            let dimension = base.dimension.pow(exponent, 1).ok_or_else(|| {
                error(
                    self.file,
                    expression.range(),
                    "Formulation dimension exponent overflows",
                )
            })?;
            return Ok(typed(
                AuthoredFormExpressionKind::Pow(Box::new(base.clone()), exponent),
                dimension,
                ValueShape::scalar(),
                base.support,
            ));
        }
        let left_value = self.compile(left)?;
        let right_value = self.compile(right)?;
        let support = merge_support(
            self.file,
            expression.range(),
            left_value.support,
            right_value.support,
        )?;
        let (kind, dimension, shape) = match op {
            BinaryOp::Add | BinaryOp::Sub => {
                if left_value.dimension != right_value.dimension
                    || left_value.shape != right_value.shape
                {
                    return Err(error(
                        self.file,
                        expression.range(),
                        "addition and subtraction require identical dimension and shape",
                    ));
                }
                let operator = if op == BinaryOp::Add {
                    BinaryOp::Add
                } else {
                    BinaryOp::Sub
                };
                let kind = binary(operator, left_value.clone(), right_value);
                (kind, left_value.dimension, left_value.shape.clone())
            }
            BinaryOp::Mul => {
                if !left_value.shape.is_scalar() && !right_value.shape.is_scalar() {
                    return Err(error(
                        self.file,
                        expression.range(),
                        "multiplication accepts at most one non-scalar operand",
                    ));
                }
                let dimension =
                    left_value
                        .dimension
                        .mul(right_value.dimension)
                        .ok_or_else(|| {
                            error(
                                self.file,
                                expression.range(),
                                "Formulation dimension multiplication overflows",
                            )
                        })?;
                let shape = if left_value.shape.is_scalar() {
                    right_value.shape.clone()
                } else {
                    left_value.shape.clone()
                };
                (
                    binary(BinaryOp::Mul, left_value, right_value),
                    dimension,
                    shape,
                )
            }
            BinaryOp::Div => {
                require_scalar(self.file, right.range(), &right_value)?;
                let dimension =
                    left_value
                        .dimension
                        .div(right_value.dimension)
                        .ok_or_else(|| {
                            error(
                                self.file,
                                expression.range(),
                                "Formulation dimension division overflows",
                            )
                        })?;
                let shape = left_value.shape.clone();
                (
                    binary(BinaryOp::Div, left_value, right_value),
                    dimension,
                    shape,
                )
            }
            BinaryOp::Pow => unreachable!(),
            _ => {
                return Err(error(
                    self.file,
                    expression.range(),
                    "Boolean predicates are not admitted in mathematical forms",
                ));
            }
        };
        Ok(typed(kind, dimension, shape, support))
    }

    fn compile_call(
        &mut self,
        expression: &Expr,
        callee: &NamePath,
        arguments: &[Expr],
    ) -> Result<AuthoredFormExpression, Diagnostic> {
        let name = unqualified_callee(self.file, expression.range(), callee)?;
        match (name, arguments) {
            ("coordinate", [axis]) => self.compile_coordinate(expression, axis),
            ("math.sin", [argument]) => {
                let argument = self.compile(argument)?;
                require_scalar(self.file, expression.range(), &argument)?;
                if argument.dimension != DimExponents::DIMENSIONLESS {
                    return Err(error(
                        self.file,
                        expression.range(),
                        "math.sin requires a dimensionless argument",
                    ));
                }
                Ok(typed(
                    AuthoredFormExpressionKind::Sin(Box::new(argument.clone())),
                    DimExponents::DIMENSIONLESS,
                    ValueShape::scalar(),
                    argument.support,
                ))
            }
            ("grad", [argument]) => {
                let argument = self.compile(argument)?;
                require_scalar(self.file, expression.range(), &argument)?;
                let support = argument.support.ok_or_else(|| {
                    error(
                        self.file,
                        expression.range(),
                        "grad requires a spatially supported expression",
                    )
                })?;
                let dimension = argument.dimension.div(length_dimension()).ok_or_else(|| {
                    error(
                        self.file,
                        expression.range(),
                        "gradient dimension overflows",
                    )
                })?;
                let extent = u32::try_from(self.ambient_dimension)
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or_else(|| {
                        error(
                            self.file,
                            expression.range(),
                            "Geometry ambient dimension is not representable",
                        )
                    })?;
                let shape = ValueShape::new([extent]).map_err(|_| {
                    error(
                        self.file,
                        expression.range(),
                        "Geometry ambient dimension is not representable",
                    )
                })?;
                Ok(typed(
                    AuthoredFormExpressionKind::Gradient(Box::new(argument)),
                    dimension,
                    shape,
                    Some(support),
                ))
            }
            ("dot", [left, right]) => {
                let left = self.compile(left)?;
                let right = self.compile(right)?;
                if left.shape.is_scalar() || left.shape != right.shape {
                    return Err(error(
                        self.file,
                        expression.range(),
                        "dot requires equal non-scalar vector shapes",
                    ));
                }
                let support =
                    merge_support(self.file, expression.range(), left.support, right.support)?;
                let dimension = left.dimension.mul(right.dimension).ok_or_else(|| {
                    error(
                        self.file,
                        expression.range(),
                        "dot-product dimension overflows",
                    )
                })?;
                Ok(typed(
                    AuthoredFormExpressionKind::Dot(Box::new(left), Box::new(right)),
                    dimension,
                    ValueShape::scalar(),
                    support,
                ))
            }
            ("integrate", [domain, integrand]) => {
                self.compile_integral(expression, domain, integrand)
            }
            _ => Err(error(
                self.file,
                expression.range(),
                format!("unsupported scalar-primal operator `{name}` or arity"),
            )),
        }
    }

    fn compile_test(&mut self, expression: &Expr) -> Result<AuthoredFormExpression, Diagnostic> {
        let raw = resolve_symbol(self.file, expression.range(), self.trial_name, self.symbols)?;
        let Some(KernelNode::Field(field)) = self.index.nodes.get(&raw).copied() else {
            return Err(error(
                self.file,
                expression.range(),
                "test argument is not a Field",
            ));
        };
        if !field.shape().is_scalar()
            || field.value_type().scalar_domain() != eqiora_core::ScalarDomain::Real
        {
            return Err(error(
                self.file,
                expression.range(),
                "test requires a scalar Field",
            ));
        }
        let support = self.field_support(expression, raw)?;
        if self
            .trial
            .replace(field.id())
            .is_some_and(|trial| trial != field.id())
        {
            return Err(error(
                self.file,
                expression.range(),
                "one scalar-primal form cannot mix test functions from different Fields",
            ));
        }
        Ok(typed(
            AuthoredFormExpressionKind::Test(field.id()),
            DimExponents::DIMENSIONLESS,
            ValueShape::scalar(),
            Some(support),
        ))
    }

    fn compile_coordinate(
        &self,
        expression: &Expr,
        axis: &Expr,
    ) -> Result<AuthoredFormExpression, Diagnostic> {
        let axis = integer_literal(axis)
            .and_then(|axis| usize::try_from(axis).ok())
            .filter(|axis| *axis < self.ambient_dimension)
            .ok_or_else(|| {
                error(
                    self.file,
                    expression.range(),
                    "coordinate axis must be a nonnegative literal below the Geometry ambient dimension",
                )
            })?;
        Ok(typed(
            AuthoredFormExpressionKind::Coordinate(axis),
            length_dimension(),
            ValueShape::scalar(),
            Some(self.relation_domain),
        ))
    }

    fn compile_integral(
        &mut self,
        expression: &Expr,
        domain: &Expr,
        integrand: &Expr,
    ) -> Result<AuthoredFormExpression, Diagnostic> {
        let ExprKind::Name(name) = domain.kind() else {
            return Err(error(
                self.file,
                domain.range(),
                "integrate Domain must be one unqualified name",
            ));
        };
        let raw = resolve_symbol(self.file, domain.range(), name, self.symbols)?;
        let domain_id = raw.downcast::<kinds::Domain>().ok_or_else(|| {
            error(
                self.file,
                domain.range(),
                "integrate first argument is not a Domain",
            )
        })?;
        if !matches!(self.index.nodes.get(&raw), Some(KernelNode::Domain(_)))
            || domain_id != self.relation_domain
        {
            return Err(error(
                self.file,
                domain.range(),
                "integrate Domain must equal the Formulation Relation Domain",
            ));
        }
        let integrand_range = integrand.range();
        let integrand = self.compile(integrand)?;
        require_scalar(self.file, integrand_range, &integrand)?;
        if integrand.support != Some(domain_id) {
            return Err(error(
                self.file,
                expression.range(),
                "integrand support must equal its integration Domain",
            ));
        }
        let topological_dimension = i32::try_from(self.topological_dimension).map_err(|_| {
            error(
                self.file,
                expression.range(),
                "Geometry dimension is not representable",
            )
        })?;
        let measure_dimension = length_dimension()
            .pow(topological_dimension, 1)
            .ok_or_else(|| {
                error(
                    self.file,
                    expression.range(),
                    "integration-measure dimension overflows",
                )
            })?;
        let dimension = integrand.dimension.mul(measure_dimension).ok_or_else(|| {
            error(
                self.file,
                expression.range(),
                "integral dimension overflows",
            )
        })?;
        Ok(typed(
            AuthoredFormExpressionKind::Integrate {
                domain: domain_id,
                integrand: Box::new(integrand),
            },
            dimension,
            ValueShape::scalar(),
            None,
        ))
    }
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
