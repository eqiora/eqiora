//! Type authored mathematical expressions without choosing a numerical realization.
use super::*;

impl ExpressionContext<'_> {
    pub(super) fn compile_root(
        &mut self,
        expression: &Expr,
    ) -> Result<AuthoredFormExpression, Diagnostic> {
        let value = self.compile(expression)?;
        if value.shape.is_scalar()
            && (value.support.is_none()
                || matches!(value.kind, AuthoredFormExpressionKind::Number(0.0)))
        {
            Ok(value)
        } else {
            Err(error(
                self.file,
                expression.range(),
                "form equality sides must be scalar integrals or zero",
            ))
        }
    }

    pub(super) fn compile(
        &mut self,
        expression: &Expr,
    ) -> Result<AuthoredFormExpression, Diagnostic> {
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
            ExprKind::Name(name) if self.tests.contains_key(name.as_str()) => {
                self.compile_test(expression, name)
            }
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
                if field.shape().rank() <= 1
                    && (self.tests.len() > 1 || field.shape().is_scalar())
                    && field.value_type().scalar_domain() == eqiora_core::ScalarDomain::Real =>
            {
                let support = self.field_support(expression, raw)?;
                Ok(typed(
                    AuthoredFormExpressionKind::Field(field.id()),
                    field.dimension(),
                    field.shape().clone(),
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
                if argument.shape.rank() > 1 {
                    return Err(error(
                        self.file,
                        expression.range(),
                        "gradient supports scalar and vector forms",
                    ));
                }
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
                let mut axes = argument
                    .shape
                    .extents()
                    .iter()
                    .map(|n| n.get())
                    .collect::<Vec<_>>();
                axes.push(extent);
                let shape = ValueShape::new(axes).map_err(|_| {
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
            ("dot" | "frobenius", [left, right]) => {
                let left = self.compile(left)?;
                let right = self.compile(right)?;
                if left.shape.rank() != if name == "dot" { 1 } else { 2 }
                    || left.shape != right.shape
                {
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
                    if name == "dot" {
                        AuthoredFormExpressionKind::Dot(Box::new(left), Box::new(right))
                    } else {
                        AuthoredFormExpressionKind::Frobenius(Box::new(left), Box::new(right))
                    },
                    dimension,
                    ValueShape::scalar(),
                    support,
                ))
            }
            ("div" | "symmetric_part", [argument]) => {
                let argument = self.compile(argument)?;
                let expected = if name == "div" { 1 } else { 2 };
                let n = self.ambient_dimension as u32;
                if argument.shape.rank() != expected
                    || argument.shape.extents().iter().any(|axis| axis.get() != n)
                {
                    return Err(error(
                        self.file,
                        expression.range(),
                        "mixed operator requires exact spatial vector/tensor axes",
                    ));
                }
                let (kind, dimension, shape) = if name == "div" {
                    (
                        AuthoredFormExpressionKind::Divergence(Box::new(argument.clone())),
                        argument.dimension.div(length_dimension()).ok_or_else(|| {
                            error(
                                self.file,
                                expression.range(),
                                "divergence dimension overflow",
                            )
                        })?,
                        ValueShape::scalar(),
                    )
                } else {
                    (
                        AuthoredFormExpressionKind::SymmetricPart(Box::new(argument.clone())),
                        argument.dimension,
                        argument.shape.clone(),
                    )
                };
                Ok(typed(kind, dimension, shape, argument.support))
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

    fn compile_test(
        &mut self,
        expression: &Expr,
        name: &str,
    ) -> Result<AuthoredFormExpression, Diagnostic> {
        let raw = resolve_symbol(
            self.file,
            expression.range(),
            self.tests[name],
            self.symbols,
        )?;
        let Some(KernelNode::Field(field)) = self.index.nodes.get(&raw).copied() else {
            return Err(error(
                self.file,
                expression.range(),
                "test trial is not a Field",
            ));
        };
        if (self.tests.len() == 1 && !field.shape().is_scalar())
            || field.shape().rank() > 1
            || field.value_type().scalar_domain() != eqiora_core::ScalarDomain::Real
        {
            return Err(error(
                self.file,
                expression.range(),
                "test requires a real scalar/vector Field",
            ));
        }
        let support = self.field_support(expression, raw)?;
        self.used_tests.insert(name.into());
        Ok(typed(
            AuthoredFormExpressionKind::Test(field.id()),
            DimExponents::DIMENSIONLESS,
            field.shape().clone(),
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
