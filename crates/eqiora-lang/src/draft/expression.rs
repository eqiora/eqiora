//! Native handles construct the same Expr algebra as parsed and Python modules.
use super::*;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(super) enum NativeReference {
    EnumValue(eqiora_core::ValueLiteral),
    Value(DraftReference),
    Port(DraftPortReference),
    Domain(DraftSpatialDomain),
}
#[derive(Debug, Clone, Copy)]
pub(super) enum DraftExpressionReference<'a> {
    EnumValue(&'a eqiora_core::ValueLiteral),
    Value(&'a DraftReference),
    Port(&'a DraftPortReference),
    Domain(&'a DraftSpatialDomain),
}
impl DraftExpression {
    pub(super) fn failed(message: &str) -> Self {
        Self {
            syntax: Err(crate::AstConstructionError::new(message)),
            references: Arc::new(Vec::new()),
            depth: 0,
            nodes: 0,
        }
    }
    pub(super) fn leaf(kind: ExprKind) -> Self {
        Self {
            syntax: Ok(Arc::new(Expr {
                kind,
                range: TextRange::default(),
                resolved_enum: None,
                resolved_nominal: None,
            })),
            references: Arc::new(Vec::new()),
            depth: 1,
            nodes: 1,
        }
    }
    pub(super) fn compose(children: Vec<Self>, make: impl FnOnce(Vec<Expr>) -> ExprKind) -> Self {
        let mut depth = 1;
        let mut nodes = 1usize;
        for child in &children {
            if let Err(error) = &child.syntax {
                return Self::failed(error.message());
            }
            depth = depth.max(child.depth.saturating_add(1));
            let Some(count) = nodes.checked_add(child.nodes) else {
                return Self::failed("native expression node count overflows");
            };
            nodes = count;
            if depth > crate::SourceAstFactory::MAX_EXPRESSION_DEPTH
                || nodes > crate::SourceAstFactory::MAX_EXPRESSION_NODES
            {
                return Self::failed(
                    "native expression exceeds the shared AST depth or node limit",
                );
            }
        }
        // Bounds precede subtree copies and metadata concatenation. A handle
        // clone itself only clones Arcs, including for shared subexpressions.
        let mut references = Vec::new();
        let values = children
            .into_iter()
            .map(|child| {
                references.extend(child.references.iter().cloned());
                Arc::unwrap_or_clone(child.syntax.expect("checked child"))
            })
            .collect();
        let mut value = Self::leaf(make(values));
        value.references = Arc::new(references);
        value.depth = depth;
        value.nodes = nodes;
        value
    }
    pub(super) fn call(name: &str, children: Vec<Self>) -> Self {
        Self::compose(children, |values| ExprKind::Call {
            callee: NamePath::from_parsed_segments(
                name.split('.').map(str::to_owned),
                TextRange::default(),
            ),
            arguments: crate::CallArguments::Positional(values),
        })
    }
    pub(super) fn unary(self, op: UnaryOp) -> Self {
        Self::compose(vec![self], |mut values| ExprKind::Unary {
            op,
            value: Box::new(values.pop().expect("one operand")),
        })
    }
    /// Construct an ordered channel-array expression.
    #[must_use]
    pub fn array(values: impl IntoIterator<Item = Self>) -> Self {
        let mut children = Vec::new();
        let mut nodes = 1usize;
        for value in values {
            if let Err(error) = &value.syntax {
                return Self::failed(error.message());
            }
            let Some(count) = nodes.checked_add(value.nodes) else {
                return Self::failed("native array node count overflows");
            };
            nodes = count;
            if nodes > crate::SourceAstFactory::MAX_EXPRESSION_NODES {
                return Self::failed("native array exceeds the shared AST node limit");
            }
            children.push(value);
        }
        if children.is_empty() {
            return Self::failed("native array literal must not be empty");
        }
        Self::compose(children, ExprKind::Array)
    }
    /// Select a static channel index. The compiler checks type and bounds.
    #[must_use]
    pub fn index(self, index: u32) -> Self {
        Self::compose(
            vec![
                self,
                Self::constant(
                    crate::DecimalLiteral::parse(&index.to_string()).expect("u32 ordinal"),
                ),
            ],
            |mut values| {
                let index = values.pop().expect("index");
                let value = values.pop().expect("array");
                ExprKind::Index {
                    value: Box::new(value),
                    index: Box::new(index),
                }
            },
        )
    }
    /// Select an immutable nonempty half-open channel slice.
    #[must_use]
    pub fn slice(self, lower: u32, upper: u32) -> Self {
        Self::compose(
            vec![
                self,
                Self::constant(
                    crate::DecimalLiteral::parse(&lower.to_string()).expect("u32 ordinal"),
                ),
                Self::constant(
                    crate::DecimalLiteral::parse(&upper.to_string()).expect("u32 ordinal"),
                ),
            ],
            |mut values| {
                let upper = values.pop().expect("upper");
                let lower = values.pop().expect("lower");
                let value = values.pop().expect("array");
                ExprKind::Slice {
                    value: Box::new(value),
                    lower: Box::new(lower),
                    upper: Box::new(upper),
                }
            },
        )
    }
    /// Construct one checked enum member, retaining its exact declaration identity.
    ///
    /// # Errors
    /// Rejects literals that are not enum members.
    pub fn enum_value(
        value: eqiora_core::ValueLiteral,
    ) -> Result<Self, crate::AstConstructionError> {
        if value.enum_tag().is_none() {
            return Err(crate::AstConstructionError::new(
                "enum expression requires an exact enum member",
            ));
        }
        // This existing nominal annotation is authoritative until Module closure
        // supplies lexical spelling. source_ast must resolve it before emission.
        let mut result = Self::leaf(ExprKind::Path(NamePath::from_parsed_segments(
            vec!["_native_enum".into(), "_member".into()],
            TextRange::default(),
        )));
        Arc::make_mut(result.syntax.as_mut().expect("leaf")).resolved_enum =
            Some(Box::new(value.clone()));
        result.references = Arc::new(vec![NativeReference::EnumValue(value)]);
        Ok(result)
    }
    /// Boolean literal with no numeric coercion.
    #[must_use]
    pub fn boolean(value: bool) -> Self {
        Self::leaf(ExprKind::Boolean(value))
    }
    /// Select one value lazily from a Boolean condition and two branches.
    #[must_use]
    pub fn select(condition: Self, then_value: Self, else_value: Self) -> Self {
        Self::compose(vec![condition, then_value, else_value], |mut values| {
            let else_value = values.pop().expect("else");
            let then_value = values.pop().expect("then");
            let condition = values.pop().expect("condition");
            ExprKind::Select {
                condition: Box::new(condition),
                then_value: Box::new(then_value),
                else_value: Box::new(else_value),
            }
        })
    }
    /// Boolean negation.
    #[must_use]
    pub fn logical_not(self) -> Self {
        self.unary(UnaryOp::Not)
    }
    /// Construct a `equal` predicate.
    #[must_use]
    pub fn equal(self, right: Self) -> Self {
        self.binary(BinaryOp::Equal, right)
    }
    /// Construct a `not_equal` predicate.
    #[must_use]
    pub fn not_equal(self, right: Self) -> Self {
        self.binary(BinaryOp::NotEqual, right)
    }
    /// Construct a `less` predicate.
    #[must_use]
    pub fn less(self, right: Self) -> Self {
        self.binary(BinaryOp::Less, right)
    }
    /// Construct a `less_equal` predicate.
    #[must_use]
    pub fn less_equal(self, right: Self) -> Self {
        self.binary(BinaryOp::LessEqual, right)
    }
    /// Construct a `greater` predicate.
    #[must_use]
    pub fn greater(self, right: Self) -> Self {
        self.binary(BinaryOp::Greater, right)
    }
    /// Construct a `greater_equal` predicate.
    #[must_use]
    pub fn greater_equal(self, right: Self) -> Self {
        self.binary(BinaryOp::GreaterEqual, right)
    }
    /// Construct a `logical_and` predicate.
    #[must_use]
    pub fn logical_and(self, right: Self) -> Self {
        self.binary(BinaryOp::And, right)
    }
    /// Construct a `logical_or` predicate.
    #[must_use]
    pub fn logical_or(self, right: Self) -> Self {
        self.binary(BinaryOp::Or, right)
    }
    pub(super) fn construction_error(&self) -> Option<&crate::AstConstructionError> {
        self.syntax.as_ref().err()
    }
    pub(super) fn references<'a>(&'a self, output: &mut Vec<DraftExpressionReference<'a>>) {
        output.extend(self.references.iter().map(|reference| match reference {
            NativeReference::EnumValue(value) => DraftExpressionReference::EnumValue(value),
            NativeReference::Value(value) => DraftExpressionReference::Value(value),
            NativeReference::Port(value) => DraftExpressionReference::Port(value),
            NativeReference::Domain(value) => DraftExpressionReference::Domain(value),
        }));
    }
    /// Resolve native annotations into the shared lexical Expr graph.
    #[doc(hidden)]
    /// # Errors
    /// Rejects failed construction and enum members without exact declarations.
    pub fn source_ast<'a>(
        &self,
        mut resolve: impl FnMut(eqiora_core::RawId) -> Option<NamePath>,
        mut resolve_enum: impl FnMut(eqiora_core::RawId) -> Option<&'a eqiora_schema::kernel::EnumDef>,
    ) -> Result<Expr, crate::AstConstructionError> {
        self.ast(
            &GraphPath::new(["argument".to_owned()]),
            &mut RangeAllocator::default(),
            &mut HashMap::new(),
            &mut resolve,
            &mut resolve_enum,
        )
    }
    pub(super) fn ast<'a>(
        &self,
        path: &GraphPath,
        ranges: &mut RangeAllocator,
        paths: &mut HashMap<TextRange, GraphPath>,
        resolve: &mut dyn FnMut(eqiora_core::RawId) -> Option<NamePath>,
        resolve_enum: &mut dyn FnMut(
            eqiora_core::RawId,
        ) -> Option<&'a eqiora_schema::kernel::EnumDef>,
    ) -> Result<Expr, crate::AstConstructionError> {
        let mut expression = self.syntax.as_ref().map_err(Clone::clone)?.as_ref().clone();
        let mut error = None;
        crate::factory::expression_visit::expression(None, &mut expression, &mut |_, node| {
            if error.is_some() {
                return;
            }
            let range = ranges.allocate(path, paths);
            if let Some(value) = node.resolved_enum() {
                match crate::SourceAstFactory::value_literal(
                    value,
                    None,
                    range,
                    &mut *resolve,
                    &mut *resolve_enum,
                    |_| None,
                ) {
                    Ok(value) => *node = value,
                    Err(failure) => {
                        error = Some(failure);
                        return;
                    }
                }
            }
            node.range = range;
            match &mut node.kind {
                ExprKind::Path(path) => *path = path.clone().with_range(range),
                ExprKind::Call { callee, .. } => *callee = callee.clone().with_range(range),
                _ => {}
            }
        });
        if let Some(error) = error {
            return Err(error);
        }
        crate::factory::validate_expression(&expression)?;
        Ok(expression)
    }
}
impl From<&DraftConservingPort> for DraftPortReference {
    fn from(port: &DraftConservingPort) -> Self {
        Self {
            symbol: port.symbol.clone(),
            name: port.name.clone(),
        }
    }
}

impl DraftExpression {
    /// Dimensionless numeric literal.
    #[must_use]
    pub fn constant(value: crate::DecimalLiteral) -> Self {
        Self::leaf(ExprKind::Number(value))
    }

    /// Construct a complex scalar without discarding either component.
    #[must_use]
    pub fn complex(real: f64, imaginary: f64) -> Self {
        let values = [real, imaginary].map(crate::DecimalLiteral::from_f64);
        match values {
            [Ok(real), Ok(imaginary)] => Self::call(
                "math.complex",
                vec![Self::constant(real), Self::constant(imaginary)],
            ),
            _ => Self::failed("native expression contains a non-finite numeric literal"),
        }
    }

    pub(super) fn reference(symbol: DraftSymbol, name: String, kind: DraftSymbolKind) -> Self {
        let mut value = Self::leaf(ExprKind::Name(name.clone()));
        value.references =
            std::sync::Arc::new(vec![expression::NativeReference::Value(DraftReference {
                symbol,
                name,
                kind,
            })]);
        value
    }

    /// Time derivative of one Field.
    #[must_use]
    pub fn derivative(field: &DraftField) -> Self {
        Self::call(
            "derivative",
            vec![Self::reference(
                field.symbol.clone(),
                field.name.clone(),
                DraftSymbolKind::Field,
            )],
        )
    }

    /// Read the across variable of one scalar conserving Port.
    #[must_use]
    pub fn across(port: &DraftConservingPort) -> Self {
        Self::port_reference(port, &port.domain.across_name)
    }

    /// Read the through variable of one scalar conserving Port.
    #[must_use]
    pub fn through(port: &DraftConservingPort) -> Self {
        Self::port_reference(port, &port.domain.through_name)
    }

    fn port_reference(port: &DraftConservingPort, member: &str) -> Self {
        let reference = DraftPortReference::from(port);
        let mut value = Self::leaf(ExprKind::Path(NamePath::from_parsed_segments(
            vec![reference.name.clone(), member.to_owned()],
            TextRange::default(),
        )));
        value.references = std::sync::Arc::new(vec![expression::NativeReference::Port(reference)]);
        value
    }

    /// Spatial gradient of one expression.
    #[must_use]
    pub fn gradient(value: Self) -> Self {
        Self::call("grad", vec![value])
    }

    /// Spatial divergence of one expression.
    #[must_use]
    pub fn divergence(value: Self) -> Self {
        Self::call("div", vec![value])
    }

    /// Boundary trace of one expression.
    #[must_use]
    pub fn trace(
        value: Self,
        on: Option<&DraftSpatialDomain>,
        from: Option<&DraftSpatialDomain>,
    ) -> Self {
        let mut children = vec![value];
        let mut names = Vec::new();
        for (name, domain) in [("on", on), ("from", from)] {
            if let Some(domain) = domain {
                let mut reference = Self::leaf(ExprKind::Name(domain.name().to_owned()));
                reference.references = Arc::new(vec![NativeReference::Domain(domain.clone())]);
                names.push(name);
                children.push(reference);
            }
        }
        Self::compose(children, |values| {
            let mut values = values.into_iter();
            let positional = vec![values.next().expect("trace operand")];
            let arguments = if names.is_empty() {
                crate::CallArguments::Positional(positional)
            } else {
                crate::CallArguments::Mixed {
                    positional,
                    named: names
                        .into_iter()
                        .zip(values)
                        .map(|(name, value)| {
                            crate::SourceAstFactory::named_binding(
                                name,
                                value,
                                TextRange::default(),
                            )
                            .expect("fixed selector name")
                        })
                        .collect(),
                }
            };
            ExprKind::Call {
                callee: NamePath::from_parsed_segments(vec!["trace".into()], TextRange::default()),
                arguments,
            }
        })
    }

    pub(super) fn binary(self, operator: BinaryOp, right: Self) -> Self {
        Self::compose(vec![self, right], |mut values| {
            let right = values.pop().expect("two operands");
            let left = values.pop().expect("two operands");
            ExprKind::Binary {
                op: operator,
                left: Box::new(left),
                right: Box::new(right),
            }
        })
    }
}
