use eqiora_core::ScalarDomain;

use super::{Expr, TextRange};

/// Source-level mathematical type, before dimensions and support are resolved.
#[derive(Debug, Clone, PartialEq)]
pub struct ValueTypeSyntax {
    pub(crate) kind: Box<ValueTypeSyntaxKind>,
    pub(crate) range: TextRange,
    pub(crate) resolved_nominal: Option<Box<eqiora_core::ValueType>>,
}

/// One declared finite basis or its algebraic dual in source syntax.
#[derive(Debug, Clone, PartialEq)]
pub struct FiniteBasisSyntax {
    /// Exact lexical declaration path.
    pub name: super::NamePath,
    /// Dual basis; this does not imply complex conjugation or a metric conversion.
    pub dual: bool,
}

impl FiniteBasisSyntax {
    /// Read an exact declaration name or explicit `dual(Name)` value argument.
    pub fn from_expression(expression: &Expr) -> Option<Self> {
        let (expression, dual) = match expression.kind() {
            super::ExprKind::Call { callee, arguments } if callee.as_str() == "dual" => {
                let [name] = arguments.positional()? else {
                    return None;
                };
                (name, true)
            }
            _ => (expression, false),
        };
        let name = match expression.kind() {
            super::ExprKind::Name(name) => {
                super::NamePath::single(name.clone(), expression.range())
            }
            super::ExprKind::Path(name) => name.clone(),
            _ => return None,
        };
        Some(Self { name, dual })
    }

    /// Project one exact basis argument without inferring a metric or conversion.
    pub fn to_expression(&self, range: TextRange) -> Result<Expr, crate::AstConstructionError> {
        let name =
            crate::SourceAstFactory::expression(super::ExprKind::Path(self.name.clone()), range)?;
        if self.dual {
            crate::SourceAstFactory::expression(
                super::ExprKind::Call {
                    callee: super::NamePath::single("dual".into(), range),
                    arguments: super::CallArguments::Positional(vec![name]),
                },
                range,
            )
        } else {
            Ok(name)
        }
    }
}

/// Closed mathematical type constructors. Arrays retain their element type.
#[derive(Debug, Clone, PartialEq)]
pub enum ValueTypeSyntaxKind {
    /// Unresolved type or dimension name, classified only by lexical resolution.
    Named(super::NamePath),
    /// Numeric coordinates in an exact declared finite basis.
    Coordinates {
        /// Scalar component domain and physical dimension.
        scalar: Box<ValueTypeSyntax>,
        /// Exact primal or dual basis.
        basis: FiniteBasisSyntax,
    },
    /// Linear map retaining both ordered basis identities.
    LinearMap {
        /// Scalar component domain and physical dimension.
        scalar: Box<ValueTypeSyntax>,
        /// Input coordinate basis.
        source: FiniteBasisSyntax,
        /// Output coordinate basis.
        target: FiniteBasisSyntax,
    },
    /// Nonnegative exact counts indexed by an exact atomic finite space.
    Counts(super::NamePath),
    /// One bounded ordinal carrying its exact index-set identity.
    Index(super::NamePath),
    /// A real or complex scalar with a structural dimension expression.
    Scalar {
        /// Mathematical domain, unrelated to numerical precision.
        domain: ScalarDomain,
        /// Physical dimension expression.
        dimension: Expr,
    },
    /// Spatial vector; its scalar element cannot itself be an array or tensor.
    Vector {
        /// Scalar component type.
        scalar: Box<ValueTypeSyntax>,
        /// Exact positive spatial extent.
        extent: u32,
    },
    /// Spatial tensor with ordered axes and scalar components.
    Tensor {
        /// Scalar component type.
        scalar: Box<ValueTypeSyntax>,
        /// Exact ordered spatial extents.
        extents: Vec<u32>,
    },
    /// Indexed channels, not a spatial frame or finite component basis.
    Array {
        /// Complete checked element type.
        element: Box<ValueTypeSyntax>,
        /// Exact static positive channel count, resolved in the declaration scope.
        extent: Expr,
    },
}

impl ValueTypeSyntax {
    /// Exact source type constructor.
    #[must_use]
    pub const fn kind(&self) -> &ValueTypeSyntaxKind {
        &self.kind
    }

    /// Complete source type range.
    #[must_use]
    pub const fn range(&self) -> TextRange {
        self.range
    }

    /// Whether syntax alone proves a scalar type, before resolving named types.
    #[must_use]
    pub const fn is_scalar(&self) -> bool {
        matches!(
            *self.kind,
            ValueTypeSyntaxKind::Scalar { .. } | ValueTypeSyntaxKind::Index(_)
        )
    }

    /// Explicit physical dimension, absent while a type name remains unresolved.
    #[must_use]
    pub fn dimension(&self) -> Option<&Expr> {
        match self.kind.as_ref() {
            ValueTypeSyntaxKind::Named(_) => None,
            ValueTypeSyntaxKind::Counts(_) | ValueTypeSyntaxKind::Index(_) => {
                Some(dimensionless_syntax())
            }
            ValueTypeSyntaxKind::Scalar { dimension, .. } => Some(dimension),
            ValueTypeSyntaxKind::Coordinates { scalar, .. }
            | ValueTypeSyntaxKind::LinearMap { scalar, .. }
            | ValueTypeSyntaxKind::Vector { scalar, .. }
            | ValueTypeSyntaxKind::Tensor { scalar, .. } => scalar.dimension(),
            ValueTypeSyntaxKind::Array { element, .. } => element.dimension(),
        }
    }

    /// Explicit scalar domain, absent while a type name remains unresolved.
    #[must_use]
    pub fn scalar_domain(&self) -> Option<ScalarDomain> {
        match self.kind.as_ref() {
            ValueTypeSyntaxKind::Named(_) => None,
            ValueTypeSyntaxKind::Counts(_) | ValueTypeSyntaxKind::Index(_) => {
                Some(ScalarDomain::Integer)
            }
            ValueTypeSyntaxKind::Scalar { domain, .. } => Some(*domain),
            ValueTypeSyntaxKind::Coordinates { scalar, .. }
            | ValueTypeSyntaxKind::LinearMap { scalar, .. }
            | ValueTypeSyntaxKind::Vector { scalar, .. }
            | ValueTypeSyntaxKind::Tensor { scalar, .. } => scalar.scalar_domain(),
            ValueTypeSyntaxKind::Array { element, .. } => element.scalar_domain(),
        }
    }

    pub(crate) fn real(dimension: Expr) -> Self {
        let range = dimension.range();
        let kind = match dimension.kind() {
            super::ExprKind::Name(name) => {
                ValueTypeSyntaxKind::Named(super::NamePath::single(name.clone(), range))
            }
            super::ExprKind::Path(name) => ValueTypeSyntaxKind::Named(name.clone()),
            _ => ValueTypeSyntaxKind::Scalar {
                domain: ScalarDomain::Real,
                dimension,
            },
        };
        Self {
            resolved_nominal: None,
            range,
            kind: Box::new(kind),
        }
    }

    /// Checked nominal binding attached by the lexical source-type resolution pass.
    #[must_use]
    pub fn resolved_nominal(&self) -> Option<&eqiora_core::ValueType> {
        self.resolved_nominal.as_deref()
    }

    pub(crate) fn rewrite_dimension(&mut self, rewrite: &mut impl FnMut(&Expr) -> Expr) {
        if self.resolved_nominal.is_some() {
            return;
        }
        match self.kind.as_mut() {
            ValueTypeSyntaxKind::Named(name) => {
                let expression = Expr {
                    resolved_enum: None,
                    resolved_nominal: None,
                    kind: if name.is_qualified() {
                        super::ExprKind::Path(name.clone())
                    } else {
                        super::ExprKind::Name(name.as_str().to_owned())
                    },
                    range: self.range,
                };
                let rewritten = rewrite(&expression);
                if rewritten != expression {
                    self.kind = Self::real(rewritten).kind;
                }
            }
            ValueTypeSyntaxKind::Scalar { dimension, .. } => *dimension = rewrite(dimension),
            ValueTypeSyntaxKind::Coordinates { scalar, .. }
            | ValueTypeSyntaxKind::LinearMap { scalar, .. }
            | ValueTypeSyntaxKind::Vector { scalar, .. }
            | ValueTypeSyntaxKind::Tensor { scalar, .. } => scalar.rewrite_dimension(rewrite),
            ValueTypeSyntaxKind::Array { element, .. } => element.rewrite_dimension(rewrite),
            ValueTypeSyntaxKind::Counts(_) | ValueTypeSyntaxKind::Index(_) => {}
        }
    }
}

fn dimensionless_syntax() -> &'static Expr {
    static ONE: std::sync::LazyLock<Expr> = std::sync::LazyLock::new(|| Expr {
        resolved_enum: None,
        resolved_nominal: None,
        kind: super::ExprKind::Number(crate::DecimalLiteral::parse("1").expect("one")),
        range: TextRange::new(0, 0),
    });
    &ONE
}
