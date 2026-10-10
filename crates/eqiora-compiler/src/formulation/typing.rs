//! Type authored mathematical expressions without choosing a numerical realization.
use super::*;

mod contraction;
mod oriented;

impl ExpressionContext<'_> {
    pub(super) fn compile_root(
        &mut self,
        expression: &Expr,
    ) -> Result<AuthoredFormExpression, Diagnostic> {
        let value = self.compile(expression)?;
        if value.value_type.shape().is_scalar()
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
                ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                    .expect("real scalar type"),
                None,
            )),
            ExprKind::Quantity { .. } => {
                let (dag, value_type) = crate::lower::lower_parameter_coefficient(
                    self.file,
                    &crate::lower::LoweringExpression::from_source(expression),
                    &BTreeMap::new(),
                )?;
                coefficient_alias(&dag, value_type)
            }
            ExprKind::Name(name) if self.tests.contains_key(name.as_str()) => {
                self.compile_test(expression, name)
            }
            ExprKind::Name(name) => self.compile_name(expression, name),
            ExprKind::Path(path) => match crate::math::constant(path) {
                Some(value) => Ok(typed(
                    AuthoredFormExpressionKind::Number(value),
                    ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
                        .expect("real scalar type"),
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
                    value.value_type.clone(),
                    value.support,
                ))
            }
            ExprKind::Binary { op, left, right } => {
                self.compile_binary(expression, *op, left, right)
            }
            ExprKind::Call { callee, arguments }
                if matches!(callee.as_str(), "trace" | "normal" | "tangential_trace") =>
            {
                self.compile_boundary(expression, callee.as_str(), arguments)
            }
            ExprKind::Call { callee, arguments } if callee.as_str() == "variation" => {
                self.compile_variation(expression, arguments)
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
        if let Some(value) = self.index.coefficients.get(name) {
            if (value.value_type.shape().is_scalar()
                || (self.relation_domain.is_none() && value.value_type.map_bases().is_some()))
                && matches!(
                    value.value_type.scalar_domain(),
                    ScalarDomain::Real | ScalarDomain::Complex
                )
            {
                return Ok(value.clone());
            }
            return Err(error(
                self.file,
                expression.range(),
                "weak forms require real or complex scalar coefficient aliases or global finite maps",
            ));
        }
        let raw = resolve_symbol(self.file, expression.range(), name, self.symbols)?;
        match self.index.nodes.get(&raw).copied() {
            Some(KernelNode::Field(field))
                if self.admits_field_type(field.value_type())
                    && matches!(
                        field.value_type().scalar_domain(),
                        ScalarDomain::Real | ScalarDomain::Complex
                    ) =>
            {
                let support = self.field_support(expression, raw)?;
                Ok(typed(
                    AuthoredFormExpressionKind::Field(field.id()),
                    field.value_type().clone(),
                    support,
                ))
            }
            Some(KernelNode::Field(_)) => Err(error(
                self.file,
                expression.range(),
                "weak forms require admitted real or complex Field shapes",
            )),
            Some(KernelNode::Parameter(parameter))
                if (parameter.value_type().shape().is_scalar()
                    || (self.relation_domain.is_none()
                        && parameter.value_type().map_bases().is_some()))
                    && matches!(
                        parameter.value_type().scalar_domain(),
                        ScalarDomain::Real | ScalarDomain::Complex
                    ) =>
            {
                Ok(parameter_expression(parameter))
            }
            Some(KernelNode::Parameter(_)) => Err(error(
                self.file,
                expression.range(),
                "weak forms require real or complex scalar Parameters or global finite maps",
            )),
            _ => Err(error(
                self.file,
                expression.range(),
                format!("`{name}` is not a scalar Field or Parameter"),
            )),
        }
    }

    fn admits_field_type(&self, value: &ValueType) -> bool {
        value.array_rank() == 0
            && (value.shape().is_scalar()
                || (self.relation_domain.is_some()
                    && value.frame() == ValueFrame::SpatialCartesian
                    && value.shape().rank() == 1
                    && value.shape().extents()[0].get() as usize == self.ambient_dimension)
                || (self.relation_domain.is_none()
                    && value.shape().rank() <= 1
                    && value.coordinate_basis().is_some()))
    }

    fn field_support(
        &self,
        expression: &Expr,
        field: RawId,
    ) -> Result<Option<Id<kinds::Domain>>, Diagnostic> {
        let support = match self.index.defined_on.get(&field).copied() {
            Some(raw) => Some(raw.downcast::<kinds::Domain>().ok_or_else(|| {
                error(
                    self.file,
                    expression.range(),
                    "Formulation Field support is not a Domain",
                )
            })?),
            None => None,
        };
        if !self.is_relation_field_support(support) {
            return Err(error(
                self.file,
                expression.range(),
                "Formulation Field support differs from the Relation Domain",
            ));
        }
        Ok(support)
    }

    fn is_relation_field_support(&self, support: Option<Id<kinds::Domain>>) -> bool {
        support == self.relation_domain
            || self
                .relation_domain
                .and_then(|domain| self.index.interface_sides(domain.erase()))
                .is_some_and(|(_, parents)| {
                    support.is_some_and(|domain| parents.contains(&domain.erase()))
                })
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
            let dimension = base
                .value_type
                .dimension()
                .pow(exponent, 1)
                .ok_or_else(|| {
                    error(
                        self.file,
                        expression.range(),
                        "Formulation dimension exponent overflows",
                    )
                })?;
            return Ok(typed(
                AuthoredFormExpressionKind::Pow(Box::new(base.clone()), exponent),
                base.value_type
                    .clone()
                    .with_dimension(dimension)
                    .map_err(|_| wire::rejection("invalid power dimension"))?,
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
        let (kind, value_type) = match op {
            BinaryOp::Add | BinaryOp::Sub => {
                let value_type = left_value
                    .value_type
                    .clone()
                    .with_common_scalar_domain(&right_value.value_type)
                    .ok_or_else(|| {
                        wire::rejection("incompatible additive scalar domains or roles")
                    })?;
                let right_type = right_value
                    .value_type
                    .clone()
                    .with_common_scalar_domain(&left_value.value_type)
                    .ok_or_else(|| {
                        wire::rejection("incompatible additive scalar domains or roles")
                    })?;
                if value_type != right_type {
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
                (kind, value_type)
            }
            BinaryOp::Mul | BinaryOp::Div => {
                use eqiora_schema::kernel::typing::{self, ExpressionType};
                // Support was checked above; delegate complete value roles and
                // dimensions to the same rules used for the original Relation.
                let left_type = ExpressionType::<RawId>::new(left_value.value_type.clone(), None);
                let right_type = ExpressionType::<RawId>::new(right_value.value_type.clone(), None);
                let result = if op == BinaryOp::Mul {
                    typing::multiply(&left_type, &right_type)
                } else {
                    typing::divide(&left_type, &right_type)
                }
                .map_err(|violation| {
                    error(
                        self.file,
                        expression.range(),
                        format!("invalid weak-form arithmetic: {violation}"),
                    )
                })?;
                (binary(op, left_value, right_value), result.value_type)
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
        Ok(typed(kind, value_type, support))
    }

    fn compile_call(
        &mut self,
        expression: &Expr,
        callee: &NamePath,
        arguments: &[Expr],
    ) -> Result<AuthoredFormExpression, Diagnostic> {
        let name = unqualified_callee(self.file, expression.range(), callee)?;
        if self.relation_domain.is_none()
            && matches!(
                name,
                "coordinate"
                    | "trace"
                    | "grad"
                    | "div"
                    | "curl"
                    | "tangential_trace"
                    | "symmetric_part"
                    | "integrate"
            )
        {
            return Err(error(
                self.file,
                expression.range(),
                "global weak forms cannot contain spatial operators or measures",
            ));
        }
        match (name, arguments) {
            ("coordinate", [axis]) => self.compile_coordinate(expression, axis),
            ("curl", [argument]) => self.compile_curl(expression, argument),
            ("cross", [left, right]) => self.compile_cross(expression, left, right),
            ("math.sin", [argument]) => {
                let argument = self.compile(argument)?;
                require_scalar(self.file, expression.range(), &argument)?;
                if argument.value_type.dimension() != DimExponents::DIMENSIONLESS {
                    return Err(error(
                        self.file,
                        expression.range(),
                        "math.sin requires a dimensionless argument",
                    ));
                }
                Ok(typed(
                    AuthoredFormExpressionKind::Sin(Box::new(argument.clone())),
                    argument.value_type.clone(),
                    argument.support,
                ))
            }
            ("grad", [argument]) => {
                let argument = self.compile(argument)?;
                if argument.value_type.shape().rank() > 1 {
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
                if !self.is_relation_field_support(Some(support))
                    || !matches!(
                        self.physical_support(support),
                        eqiora_schema::kernel::typing::SpatialSupport::Volume { .. }
                    )
                {
                    return Err(error(
                        self.file,
                        expression.range(),
                        "grad requires a parent-volume expression; boundary derivatives need an explicit tangential operator",
                    ));
                }
                let dimension = argument
                    .value_type
                    .dimension()
                    .div(length_dimension())
                    .ok_or_else(|| {
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
                    .value_type
                    .shape()
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
                let value_type = ValueType::shaped(
                    argument.value_type.scalar_domain(),
                    dimension,
                    shape,
                    ValueFrame::SpatialCartesian,
                )
                .map_err(|_| wire::rejection("invalid spatial form type"))?;
                Ok(typed(
                    AuthoredFormExpressionKind::Gradient(Box::new(argument)),
                    value_type,
                    Some(support),
                ))
            }
            ("math.complex", [real, imag]) => {
                let real = self.compile(real)?;
                let imag = self.compile(imag)?;
                if !real.value_type.shape().is_scalar()
                    || real.value_type.scalar_domain() != ScalarDomain::Real
                    || real.value_type != imag.value_type
                {
                    return Err(error(
                        self.file,
                        expression.range(),
                        "math.complex requires two equally dimensioned real scalars",
                    ));
                }
                let value_type =
                    ValueType::scalar(ScalarDomain::Complex, real.value_type.dimension())
                        .map_err(|_| wire::rejection("invalid complex scalar type"))?;
                let support =
                    merge_support(self.file, expression.range(), real.support, imag.support)?;
                Ok(typed(
                    AuthoredFormExpressionKind::Complex(Box::new(real), Box::new(imag)),
                    value_type,
                    support,
                ))
            }
            ("math.conj", [argument]) => {
                let argument = self.compile(argument)?;
                let value_type = argument.value_type.clone();
                let support = argument.support;
                Ok(typed(
                    AuthoredFormExpressionKind::Conjugate(Box::new(argument)),
                    value_type,
                    support,
                ))
            }
            ("apply", [left, right]) if self.relation_domain.is_none() => {
                self.compile_apply(expression, left, right)
            }
            ("dot" | "frobenius" | "inner", [left, right]) => {
                self.compile_contraction(expression, name, left, right)
            }
            ("div" | "symmetric_part", [argument]) => {
                let argument = self.compile(argument)?;
                let expected = if name == "div" { 1 } else { 2 };
                let n = self.ambient_dimension as u32;
                if argument.value_type.shape().rank() != expected
                    || argument
                        .value_type
                        .shape()
                        .extents()
                        .iter()
                        .any(|axis| axis.get() != n)
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
                        argument
                            .value_type
                            .dimension()
                            .div(length_dimension())
                            .ok_or_else(|| {
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
                        argument.value_type.dimension(),
                        argument.value_type.shape().clone(),
                    )
                };
                Ok(typed(
                    kind,
                    ValueType::shaped(
                        argument.value_type.scalar_domain(),
                        dimension,
                        shape.clone(),
                        if shape.is_scalar() {
                            ValueFrame::Invariant
                        } else {
                            argument.value_type.frame()
                        },
                    )
                    .map_err(|_| wire::rejection("invalid spatial form type"))?,
                    argument.support,
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

    fn compile_test(
        &mut self,
        expression: &Expr,
        name: &str,
    ) -> Result<AuthoredFormExpression, Diagnostic> {
        let raw = resolve_symbol(
            self.file,
            expression.range(),
            self.tests[name].0,
            self.symbols,
        )?;
        let Some(KernelNode::Field(field)) = self.index.nodes.get(&raw).copied() else {
            return Err(error(
                self.file,
                expression.range(),
                "test trial is not a Field",
            ));
        };
        if !self.admits_field_type(field.value_type())
            || !matches!(
                field.value_type().scalar_domain(),
                ScalarDomain::Real | ScalarDomain::Complex
            )
        {
            return Err(error(
                self.file,
                expression.range(),
                "test requires an admitted real or complex Field shape",
            ));
        }
        let support = self.field_support(expression, raw)?;
        self.used_tests.insert(name.into());
        let trial_count = self
            .tests
            .values()
            .map(|(trial, _)| resolve_symbol(self.file, expression.range(), trial, self.symbols))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|trial| *trial == raw)
            .count();
        Ok(typed(
            if trial_count > 1 {
                AuthoredFormExpressionKind::Direction {
                    name: name.into(),
                    trial: field.id(),
                }
            } else {
                AuthoredFormExpressionKind::Test(field.id())
            },
            field
                .value_type()
                .clone()
                .with_dimension(self.tests[name].1)
                .map_err(|_| wire::rejection("invalid test dimension"))?,
            support,
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
        let support = self
            .integration_domain
            .or(self.relation_domain)
            .ok_or_else(|| {
                error(
                    self.file,
                    expression.range(),
                    "coordinate requires an exact support",
                )
            })?;
        let factor = self
            .index
            .boundary_of
            .get(&support.erase())
            .copied()
            .unwrap_or(support.erase())
            .downcast::<kinds::Domain>()
            .ok_or_else(|| {
                error(
                    self.file,
                    expression.range(),
                    "coordinate factor is not a Domain",
                )
            })?;
        Ok(typed(
            AuthoredFormExpressionKind::Coordinate {
                support,
                factor,
                axis,
            },
            ValueType::scalar(ScalarDomain::Real, length_dimension()).expect("real scalar type"),
            self.integration_domain.or(self.relation_domain),
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
        let boundary = self
            .index
            .boundary_of
            .get(&raw)
            .copied()
            .is_some_and(|parent| Some(parent) == self.relation_domain.map(Id::erase));
        if self.integration_domain.is_some()
            || !matches!(self.index.nodes.get(&raw), Some(KernelNode::Domain(_)))
            || (Some(domain_id) != self.relation_domain && !boundary)
        {
            return Err(error(
                self.file,
                domain.range(),
                "integrate requires the Relation Domain or its exact boundary, without nested integrals",
            ));
        }
        let integrand_range = integrand.range();
        self.integration_domain = Some(domain_id);
        let compiled = self.compile(integrand);
        self.integration_domain = None;
        let integrand = compiled?;
        require_scalar(self.file, integrand_range, &integrand)?;
        if integrand
            .support
            .is_some_and(|support| support != domain_id)
        {
            return Err(error(
                self.file,
                expression.range(),
                "integrand support must equal its integration Domain",
            ));
        }
        let dimension = self
            .topological_dimension
            .checked_sub(usize::from(boundary))
            .ok_or_else(|| {
                error(
                    self.file,
                    expression.range(),
                    "boundary measure requires a positive-dimensional parent",
                )
            })?;
        let topological_dimension = i32::try_from(dimension).map_err(|_| {
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
        let dimension = integrand
            .value_type
            .dimension()
            .mul(measure_dimension)
            .ok_or_else(|| {
                error(
                    self.file,
                    expression.range(),
                    "integral dimension overflows",
                )
            })?;
        let value_type = integrand
            .value_type
            .clone()
            .with_dimension(dimension)
            .map_err(|_| wire::rejection("invalid integral dimension"))?;
        Ok(typed(
            AuthoredFormExpressionKind::Integrate {
                domain: domain_id,
                integrand: Box::new(integrand),
            },
            value_type,
            None,
        ))
    }
}
