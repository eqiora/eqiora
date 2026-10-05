use super::*;

/// Source expression with its exact byte range.
#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub(crate) resolved_enum: Option<Box<eqiora_core::ValueLiteral>>,
    pub(crate) resolved_nominal: Option<Box<eqiora_core::ValueType>>,
    pub(crate) kind: ExprKind,
    pub(crate) range: TextRange,
}

impl Expr {
    /// Render this expression with the canonical module formatter.
    #[must_use]
    pub fn to_source(&self) -> String {
        crate::formatter::expression_source(self)
    }

    /// Checked nominal constructor type supplied by lexical declaration resolution.
    #[must_use]
    pub fn resolved_nominal(&self) -> Option<&eqiora_core::ValueType> {
        self.resolved_enum
            .as_deref()
            .map(eqiora_core::ValueLiteral::value_type)
            .or(self.resolved_nominal.as_deref())
    }

    /// Checked enum member supplied by lexical declaration resolution.
    #[must_use]
    pub fn resolved_enum(&self) -> Option<&eqiora_core::ValueLiteral> {
        self.resolved_enum.as_deref()
    }

    /// Expression form.
    #[must_use]
    pub const fn kind(&self) -> &ExprKind {
        &self.kind
    }

    /// Full expression range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }

    /// Rewrite structurally named references without reparsing source text.
    ///
    /// The callback visits bare [`ExprKind::Name`] and qualified
    /// [`ExprKind::Path`] occurrences as one [`NamePath`] abstraction. Returning
    /// `None` retains the occurrence; returning a path replaces it. Expression
    /// topology and every expression/source range are preserved. Call callees
    /// are structural [`NamePath`] values and are visited before their ordered
    /// arguments.
    /// Reduction set names are visited, while occurrences rooted in the local
    /// binder are protected from outer rewrites. Unit-catalog names inside
    /// quantity literals are not value-name occurrences.
    #[must_use]
    pub fn rewrite_name_paths(
        &self,
        mut rewrite: impl FnMut(&NamePath) -> Option<NamePath>,
    ) -> Self {
        self.rewrite_name_paths_with(&mut rewrite)
    }

    fn rewrite_name_paths_with(
        &self,
        rewrite: &mut dyn FnMut(&NamePath) -> Option<NamePath>,
    ) -> Self {
        let kind = match &self.kind {
            ExprKind::CoordinateMapFactor { factor, source, at } => ExprKind::CoordinateMapFactor {
                factor: *factor,
                source: source
                    .iter()
                    .map(|name| {
                        rewrite(name)
                            .unwrap_or_else(|| name.clone())
                            .with_range(name.range())
                    })
                    .collect(),
                at: at
                    .iter()
                    .map(|(name, value)| {
                        (
                            rewrite(name)
                                .unwrap_or_else(|| name.clone())
                                .with_range(name.range()),
                            value.rewrite_name_paths_with(rewrite),
                        )
                    })
                    .collect(),
            },
            ExprKind::Pullback { value, source, at } => ExprKind::Pullback {
                value: Box::new(value.rewrite_name_paths_with(rewrite)),
                source: source
                    .iter()
                    .map(|name| {
                        rewrite(name)
                            .unwrap_or_else(|| name.clone())
                            .with_range(name.range())
                    })
                    .collect(),
                at: at
                    .iter()
                    .map(|(name, value)| {
                        (
                            rewrite(name)
                                .unwrap_or_else(|| name.clone())
                                .with_range(name.range()),
                            value.rewrite_name_paths_with(rewrite),
                        )
                    })
                    .collect(),
            },
            ExprKind::Evaluate { value, at, side } => ExprKind::Evaluate {
                value: Box::new(value.rewrite_name_paths_with(rewrite)),
                at: at
                    .iter()
                    .map(|(coordinate, value)| {
                        (
                            rewrite(coordinate)
                                .unwrap_or_else(|| coordinate.clone())
                                .with_range(coordinate.range()),
                            value.rewrite_name_paths_with(rewrite),
                        )
                    })
                    .collect(),
                side: *side,
            },
            ExprKind::Partial {
                value,
                wrt,
                holding,
            } => ExprKind::Partial {
                value: Box::new(value.rewrite_name_paths_with(rewrite)),
                wrt: rewrite(wrt)
                    .unwrap_or_else(|| wrt.clone())
                    .with_range(wrt.range()),
                holding: holding
                    .iter()
                    .map(|name| {
                        rewrite(name)
                            .unwrap_or_else(|| name.clone())
                            .with_range(name.range())
                    })
                    .collect(),
            },
            ExprKind::Member { value, member } => ExprKind::Member {
                value: Box::new(value.rewrite_name_paths_with(rewrite)),
                member: member.clone(),
            },
            ExprKind::Boolean(value) => ExprKind::Boolean(*value),
            ExprKind::Number(value) => ExprKind::Number(value.clone()),
            ExprKind::Quantity { value, unit } => ExprKind::Quantity {
                value: value.clone(),
                unit: unit.clone(),
            },
            ExprKind::Name(name) => {
                let path = NamePath::single(name.clone(), self.range);
                rewrite(&path).map_or_else(
                    || ExprKind::Name(name.clone()),
                    |replacement| expression_name(replacement.with_range(self.range)),
                )
            }
            ExprKind::Path(path) => rewrite(path).map_or_else(
                || ExprKind::Path(path.clone()),
                |replacement| expression_name(replacement.with_range(self.range)),
            ),
            ExprKind::BoundaryPortSelection { port, selector } => ExprKind::BoundaryPortSelection {
                port: Box::new(rewrite(port).unwrap_or_else(|| port.as_ref().clone())),
                selector: selector.clone(),
            },
            ExprKind::Unary { op, value } => ExprKind::Unary {
                op: *op,
                value: Box::new(value.rewrite_name_paths_with(rewrite)),
            },
            ExprKind::Binary { op, left, right } => ExprKind::Binary {
                op: *op,
                left: Box::new(left.rewrite_name_paths_with(rewrite)),
                right: Box::new(right.rewrite_name_paths_with(rewrite)),
            },
            ExprKind::Array(elements) => ExprKind::Array(
                elements
                    .iter()
                    .map(|element| element.rewrite_name_paths_with(rewrite))
                    .collect(),
            ),
            ExprKind::Tuple(elements) => ExprKind::Tuple(
                elements
                    .iter()
                    .map(|element| element.rewrite_name_paths_with(rewrite))
                    .collect(),
            ),
            ExprKind::Index { value, index } => ExprKind::Index {
                value: Box::new(value.rewrite_name_paths_with(rewrite)),
                index: Box::new(index.rewrite_name_paths_with(rewrite)),
            },
            ExprKind::Slice {
                value,
                lower,
                upper,
            } => ExprKind::Slice {
                value: Box::new(value.rewrite_name_paths_with(rewrite)),
                lower: Box::new(lower.rewrite_name_paths_with(rewrite)),
                upper: Box::new(upper.rewrite_name_paths_with(rewrite)),
            },
            ExprKind::Case { value, arms } => ExprKind::Case {
                value: Box::new(value.rewrite_name_paths_with(rewrite)),
                arms: arms
                    .iter()
                    .map(|arm| CaseArm {
                        pattern: rewrite(&arm.pattern).map_or_else(
                            || arm.pattern.clone(),
                            |path| path.with_range(arm.pattern.range()),
                        ),
                        value: arm.value.rewrite_name_paths_with(rewrite),
                        resolved_pattern: arm.resolved_pattern.clone(),
                        range: arm.range,
                    })
                    .collect(),
            },
            ExprKind::Select {
                condition,
                then_value,
                else_value,
            } => ExprKind::Select {
                condition: Box::new(condition.rewrite_name_paths_with(rewrite)),
                then_value: Box::new(then_value.rewrite_name_paths_with(rewrite)),
                else_value: Box::new(else_value.rewrite_name_paths_with(rewrite)),
            },
            ExprKind::Reduction {
                operation,
                binder,
                value,
            } => {
                let set = rewrite(&binder.set)
                    .unwrap_or_else(|| binder.set.clone())
                    .with_range(binder.set.range());
                let mut scoped = |path: &NamePath| {
                    if path.segments().next() == Some(binder.member.as_str()) {
                        None
                    } else {
                        rewrite(path)
                    }
                };
                ExprKind::Reduction {
                    operation: *operation,
                    binder: FamilyBinderSyntax {
                        member: binder.member.clone(),
                        set,
                        range: binder.range,
                    },
                    value: Box::new(value.rewrite_name_paths_with(&mut scoped)),
                }
            }
            ExprKind::Call { callee, arguments } => ExprKind::Call {
                callee: rewrite(callee).map_or_else(
                    || callee.clone(),
                    |replacement| replacement.with_range(callee.range()),
                ),
                arguments: match arguments {
                    CallArguments::Mixed { positional, named } => CallArguments::Mixed {
                        positional: positional
                            .iter()
                            .map(|value| value.rewrite_name_paths_with(rewrite))
                            .collect(),
                        named: named
                            .iter()
                            .map(|binding| NamedBindingDecl {
                                value: binding.value.rewrite_name_paths_with(rewrite),
                                ..binding.clone()
                            })
                            .collect(),
                    },
                    CallArguments::Positional(values) => CallArguments::Positional(
                        values
                            .iter()
                            .map(|value| value.rewrite_name_paths_with(rewrite))
                            .collect(),
                    ),
                    CallArguments::Named(bindings) => CallArguments::Named(
                        bindings
                            .iter()
                            .map(|binding| NamedBindingDecl {
                                value: binding.value.rewrite_name_paths_with(rewrite),
                                ..binding.clone()
                            })
                            .collect(),
                    ),
                },
            },
        };
        Self {
            resolved_enum: self.resolved_enum.clone(),
            resolved_nominal: self.resolved_nominal.clone(),
            kind,
            range: self.range,
        }
    }
}

fn expression_name(path: NamePath) -> ExprKind {
    if path.is_qualified() {
        ExprKind::Path(path)
    } else {
        ExprKind::Name(path.as_str().to_owned())
    }
}

/// Recursive parser AST. Canonical residual storage is the lowered DAG.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ExprKind {
    /// Differential factor of an explicit map between ordered coordinates.
    CoordinateMapFactor {
        /// Signed Jacobian, volume scale, or orientation.
        factor: eqiora_schema::kernel::CoordinateMapFactor,
        /// Complete source coordinate inventory.
        source: Vec<NamePath>,
        /// Complete target coordinate expressions.
        at: Vec<(NamePath, Expr)>,
    },
    /// Pull an invariant scalar along a map between exact coordinate supports.
    Pullback {
        /// Expression on the target support.
        value: Box<Expr>,
        /// Complete source coordinate inventory.
        source: Vec<NamePath>,
        /// Target coordinates paired with expressions on the source support.
        at: Vec<(NamePath, Expr)>,
    },
    /// Bind exact coordinate projections to dimensioned points, without choosing a reconstruction.
    Evaluate {
        /// Analytic expression or represented Field to evaluate.
        value: Box<Expr>,
        /// Exact coordinate bindings and their point expressions.
        at: Vec<(NamePath, Expr)>,
        /// Requested one-sided value for a one-dimensional discontinuous representation.
        side: Option<BoundarySideSyntax>,
    },
    /// An explicit first partial with compile-time independent-input bindings.
    Partial {
        /// Explicit expression; aliases retain their ordinary dependencies.
        value: Box<Expr>,
        /// Declared independent value or lexical operator formal.
        wrt: NamePath,
        /// Additional held-fixed binding assertions, never runtime arguments.
        holding: Vec<NamePath>,
    },
    /// Boolean truth value, distinct from numeric literals.
    Boolean(bool),
    /// Exact decimal literal, interpreted in its required value-domain context.
    Number(crate::DecimalLiteral),
    /// Numeric literal with an explicit input-unit expression.
    Quantity {
        /// Exact decimal value before unit conversion.
        value: crate::DecimalLiteral,
        /// Unit-catalog expression, independent of value names and dimension aliases.
        unit: Box<Expr>,
    },
    /// Nonempty ordered channel-array elements; rectangularity is checked during lowering.
    Array(Vec<Expr>),
    /// Compile-time structural tuple; never implicitly a value array or tensor.
    Tuple(Vec<Expr>),
    /// Select one channel from a value; index legality is checked during lowering.
    Index {
        /// Value being indexed.
        value: Box<Expr>,
        /// Authored index expression.
        index: Box<Expr>,
    },
    /// Nonempty immutable half-open channel slice with explicit static bounds.
    Slice {
        /// Complete array value, never a spatial-axis slice.
        value: Box<Expr>,
        /// Inclusive nonnegative lower bound.
        lower: Box<Expr>,
        /// Exclusive upper bound, no larger than the channel extent.
        upper: Box<Expr>,
    },
    /// Member of a statically selected indexed component occurrence.
    Member { value: Box<Expr>, member: String },
    /// Source identifier.
    Name(String),
    /// Qualified lexical or instance-member name.
    Path(NamePath),
    /// One boundary-family Port selected by an exact boundary spelling.
    BoundaryPortSelection {
        /// Port family path.
        port: Box<NamePath>,
        /// Closed boundary selector.
        selector: Box<BoundaryPortSelectorSyntax>,
    },
    /// Prefix operator.
    Unary {
        /// Operator.
        op: UnaryOp,
        /// Operand.
        value: Box<Expr>,
    },
    /// Infix operator.
    Binary {
        /// Operator.
        op: BinaryOp,
        /// Left operand.
        left: Box<Expr>,
        /// Right operand.
        right: Box<Expr>,
    },
    /// Exhaustive value selection over one nominal enumeration.
    Case {
        /// Value whose exact enum determines the required patterns.
        value: Box<Expr>,
        /// Authored-order explicit tag arms; no wildcard or pattern binding.
        arms: Vec<CaseArm>,
    },
    /// Lazy value selection; both branches remain authored expressions.
    Select {
        /// Boolean scalar predicate.
        condition: Box<Expr>,
        /// Value selected when the predicate is true.
        then_value: Box<Expr>,
        /// Value selected when the predicate is false.
        else_value: Box<Expr>,
    },
    /// Ordered reduction over one nonempty bounded nominal index set.
    Reduction {
        /// Scalar reduction operation, evaluated in index-set order.
        operation: ReductionOp,
        /// Lexical member binding and exact set name.
        binder: FamilyBinderSyntax,
        /// Scalar expression evaluated for each member.
        value: Box<Expr>,
    },
    /// Qualified named operator with one or more ordered arguments.
    Call {
        /// Structurally qualified operator name.
        callee: NamePath,
        /// Nonempty ordered arguments.
        arguments: CallArguments,
    },
}

/// Scalar operation for an ordered finite reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionOp {
    /// Ordered addition.
    Sum,
    /// Ordered multiplication.
    Product,
    /// Ordered minimum, retaining the first equal operand.
    Min,
    /// Ordered maximum, retaining the first equal operand.
    Max,
}

/// Prefix expression operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// Arithmetic negation.
    Neg,
    /// Boolean negation.
    Not,
}

/// Infix expression operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Integer power (validated during lowering).
    Pow,
    /// Equality comparison.
    Equal,
    /// Inequality comparison.
    NotEqual,
    /// Strict order comparison.
    Less,
    /// Inclusive order comparison.
    LessEqual,
    /// Strict reverse order comparison.
    Greater,
    /// Inclusive reverse order comparison.
    GreaterEqual,
    /// Short-circuit Boolean conjunction.
    And,
    /// Short-circuit Boolean disjunction.
    Or,
}

/// One authored-order argument list for a shared expression call.
#[derive(Debug, Clone, PartialEq)]
pub enum CallArguments {
    /// Ordered arguments of a positional primitive.
    Positional(Vec<Expr>),
    /// Explicit bindings to the target signature, retaining authored order.
    Named(Vec<NamedBindingDecl>),
    /// Positional operands followed by explicit named options.
    Mixed {
        /// Nonempty positional prefix.
        positional: Vec<Expr>,
        /// Nonempty named suffix.
        named: Vec<NamedBindingDecl>,
    },
}

impl CallArguments {
    /// Expression values in authored argument order, without erasing binding names.
    pub fn expressions(&self) -> impl ExactSizeIterator<Item = &Expr> {
        let (positional, named) = self.parts();
        (0..positional.len() + named.len()).map(move |index| {
            if index < positional.len() {
                &positional[index]
            } else {
                named[index - positional.len()].value()
            }
        })
    }
    /// Positional prefix and named suffix, retaining authored order within each.
    #[must_use]
    pub fn parts(&self) -> (&[Expr], &[NamedBindingDecl]) {
        match self {
            Self::Positional(values) => (values, &[]),
            Self::Named(bindings) => (&[], bindings),
            Self::Mixed { positional, named } => (positional, named),
        }
    }
    /// Positional values, only when this call uses positional syntax.
    #[must_use]
    pub fn positional(&self) -> Option<&[Expr]> {
        match self {
            Self::Positional(values) => Some(values),
            Self::Named(_) | Self::Mixed { .. } => None,
        }
    }
    /// Named bindings, only when this call uses named syntax.
    #[must_use]
    pub fn named(&self) -> Option<&[NamedBindingDecl]> {
        match self {
            Self::Named(bindings) => Some(bindings),
            Self::Positional(_) | Self::Mixed { .. } => None,
        }
    }
}
