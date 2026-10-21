//! Combined Boolean and numeric type check for calculus nodes and their dimensions.
use super::*;
use eqiora_core::{ScalarDomain, ValueType};

#[derive(Clone)]
struct Proof {
    domain: Option<ScalarDomain>,
    dimension: FormalDimensionMonomial,
}

fn dimensionless(count: usize) -> FormalDimensionMonomial {
    FormalDimensionMonomial {
        fixed_dimension: DimExponents::DIMENSIONLESS,
        exponents: vec![ExactRational::integer(0); count].into_boxed_slice(),
    }
}
fn numeric(proof: &Proof) -> Result<(), PureOperatorError> {
    if proof.domain == Some(ScalarDomain::Boolean) {
        Err(PureOperatorError::FormalTypeMismatch)
    } else {
        Ok(())
    }
}
fn boolean(proof: &Proof) -> Result<(), PureOperatorError> {
    if proof.domain == Some(ScalarDomain::Boolean) {
        Ok(())
    } else {
        Err(PureOperatorError::FormalTypeMismatch)
    }
}
fn common(left: Option<ScalarDomain>, right: Option<ScalarDomain>) -> Option<ScalarDomain> {
    if left == Some(ScalarDomain::Complex) || right == Some(ScalarDomain::Complex) {
        Some(ScalarDomain::Complex)
    } else if left == Some(ScalarDomain::Real) && right == Some(ScalarDomain::Real) {
        Some(ScalarDomain::Real)
    } else {
        None
    }
}
fn same_dimension(
    formals: &[PureValueClass],
    left: &Proof,
    right: &Proof,
) -> Result<(), PureOperatorError> {
    if normalized(formals, &left.dimension)? != normalized(formals, &right.dimension)? {
        Err(PureOperatorError::AdditiveDimensionMismatch)
    } else {
        Ok(())
    }
}

fn combine_dimension(
    left: &mut FormalDimensionMonomial,
    right: &FormalDimensionMonomial,
    divide: bool,
) -> Result<(), PureOperatorError> {
    left.fixed_dimension = if divide {
        left.fixed_dimension.div(right.fixed_dimension)
    } else {
        left.fixed_dimension.mul(right.fixed_dimension)
    }
    .ok_or(PureOperatorError::ResultDimensionOverflow)?;
    for (a, b) in left.exponents.iter_mut().zip(right.exponents.iter()) {
        let b = if divide { b.checked_neg()? } else { b.clone() };
        *a = bounded(a.checked_add(b)?)?;
    }
    Ok(())
}

fn prove(
    formals: &[PureValueClass],
    nodes: &[CalculusNode],
) -> Result<Vec<Proof>, PureOperatorError> {
    let mut proofs: Vec<Proof> = Vec::with_capacity(nodes.len());
    for node in nodes {
        let get = |id| {
            proofs
                .get(definition_index(id, proofs.len())?)
                .cloned()
                .ok_or(PureOperatorError::InvalidNode)
        };
        let proof = match node {
            CalculusNode::Demand {
                prerequisite,
                value,
            } => {
                get(*prerequisite)?;
                get(*value)?
            }
            CalculusNode::Require { condition, value } => {
                boolean(&get(*condition)?)?;
                get(*value)?
            }
            CalculusNode::Rational { dimension, .. } => {
                let mut proof = Proof {
                    domain: Some(ScalarDomain::Real),
                    dimension: dimensionless(formals.len()),
                };
                proof.dimension.fixed_dimension = *dimension;
                proof
            }
            CalculusNode::Boolean(_) => Proof {
                domain: Some(ScalarDomain::Boolean),
                dimension: dimensionless(formals.len()),
            },
            CalculusNode::KroneckerDelta(..) => Proof {
                domain: Some(ScalarDomain::Real),
                dimension: dimensionless(formals.len()),
            },
            CalculusNode::FormalComponent { formal, .. } => {
                let index = usize::from(*formal);
                let class = formals
                    .get(index)
                    .ok_or(PureOperatorError::InvalidFormal(*formal))?;
                let mut dimension = dimensionless(formals.len());
                dimension.exponents[index] = ExactRational::integer(1);
                Proof {
                    domain: class.scalar_domain(),
                    dimension,
                }
            }
            CalculusNode::Differentiated { source, wrt } => {
                let mut source = get(*source)?;
                let wrt = get(*wrt)?;
                numeric(&source)?;
                numeric(&wrt)?;
                combine_dimension(&mut source.dimension, &wrt.dimension, true)?;
                source
            }
            CalculusNode::BoundInput(value) => get(*value)?,
            CalculusNode::Neg(value) => {
                let value = get(*value)?;
                numeric(&value)?;
                value
            }
            CalculusNode::Not(value) => {
                let value = get(*value)?;
                boolean(&value)?;
                value
            }
            CalculusNode::And(left, right) | CalculusNode::Or(left, right) => {
                let left = get(*left)?;
                let right = get(*right)?;
                boolean(&left)?;
                boolean(&right)?;
                left
            }
            CalculusNode::Compare(op, left, right) => {
                let left = get(*left)?;
                let right = get(*right)?;
                if left.domain == Some(ScalarDomain::Boolean)
                    || right.domain == Some(ScalarDomain::Boolean)
                {
                    boolean(&left)?;
                    boolean(&right)?;
                    if !matches!(
                        op,
                        super::super::ComparisonOp::Equal | super::super::ComparisonOp::NotEqual
                    ) {
                        return Err(PureOperatorError::FormalTypeMismatch);
                    }
                } else {
                    if left.domain != Some(ScalarDomain::Real)
                        || right.domain != Some(ScalarDomain::Real)
                    {
                        return Err(PureOperatorError::FormalTypeMismatch);
                    }
                    same_dimension(formals, &left, &right)?;
                }
                Proof {
                    domain: Some(ScalarDomain::Boolean),
                    dimension: dimensionless(formals.len()),
                }
            }
            CalculusNode::Select {
                condition,
                then_value,
                else_value,
            } => {
                boolean(&get(*condition)?)?;
                let left = get(*then_value)?;
                let right = get(*else_value)?;
                if left.domain != right.domain {
                    return Err(PureOperatorError::FormalTypeMismatch);
                }
                same_dimension(formals, &left, &right)?;
                left
            }
            CalculusNode::Add(left, right) => {
                let mut left = get(*left)?;
                let right = get(*right)?;
                numeric(&left)?;
                numeric(&right)?;
                same_dimension(formals, &left, &right)?;
                left.domain = common(left.domain, right.domain);
                left
            }
            CalculusNode::Mul(left, right) | CalculusNode::Div(left, right) => {
                let mut left = get(*left)?;
                let right = get(*right)?;
                numeric(&left)?;
                numeric(&right)?;
                combine_dimension(
                    &mut left.dimension,
                    &right.dimension,
                    matches!(node, CalculusNode::Div(..)),
                )?;
                left.domain = common(left.domain, right.domain);
                left
            }
            CalculusNode::Complex { real, imag } => {
                let mut real = get(*real)?;
                let imag = get(*imag)?;
                if real.domain != Some(ScalarDomain::Real)
                    || imag.domain != Some(ScalarDomain::Real)
                {
                    return Err(PureOperatorError::FormalTypeMismatch);
                }
                same_dimension(formals, &real, &imag)?;
                real.domain = Some(ScalarDomain::Complex);
                real
            }
            CalculusNode::UnaryMath(function, operand) => {
                let operand = get(*operand)?;
                let (dimension, free) = normalized(formals, &operand.dimension)?;
                if function.requires_dimensionless_input()
                    && free.iter().any(|power| !power.is_zero())
                {
                    return Err(PureOperatorError::FormalTypeMismatch);
                }
                let ty = ExpressionType::<()>::new(
                    ValueType::scalar(
                        operand
                            .domain
                            .ok_or(PureOperatorError::FormalTypeMismatch)?,
                        dimension,
                    )
                    .map_err(|_| PureOperatorError::FormalTypeMismatch)?,
                    None,
                );
                let ty =
                    super::super::typing::unary_math(*function, &ty).map_err(
                        |error| match error {
                            super::super::typing::TypeViolation::DimensionOverflow { .. } => {
                                PureOperatorError::ResultDimensionOverflow
                            }
                            _ => PureOperatorError::FormalTypeMismatch,
                        },
                    )?;
                let (numerator, denominator) = function.dimension_power();
                let mut dimension = operand.dimension;
                dimension.fixed_dimension = dimension
                    .fixed_dimension
                    .pow(numerator, denominator)
                    .ok_or(PureOperatorError::ResultDimensionOverflow)?;
                let exponent = ExactRational::new(numerator.into(), denominator.into())?;
                for power in &mut dimension.exponents {
                    *power = bounded(power.checked_mul(exponent.clone())?)?;
                }
                Proof {
                    domain: Some(ty.value_type.scalar_domain()),
                    dimension,
                }
            }
        };
        // Check fixed formal dimensions before any downstream component expansion.
        normalized(formals, &proof.dimension)?;
        proofs.push(proof);
    }
    Ok(proofs)
}

/// Concrete component types are computed once for the exact argument types.
pub(super) fn component_types(
    formals: &[PureValueClass],
    nodes: &[CalculusNode],
) -> Result<Vec<ValueType>, PureOperatorError> {
    prove(formals, nodes)?
        .into_iter()
        .map(|proof| {
            let (dimension, free) = normalized(formals, &proof.dimension)?;
            if free.iter().any(|power| !power.is_zero()) {
                return Err(PureOperatorError::FormalTypeMismatch);
            }
            ValueType::scalar(
                proof.domain.ok_or(PureOperatorError::FormalTypeMismatch)?,
                dimension,
            )
            .map_err(|_| PureOperatorError::FormalTypeMismatch)
        })
        .collect()
}

fn bounded(exponent: ExactRational) -> Result<ExactRational, PureOperatorError> {
    if exponent < ExactRational::integer(-i64::from(MAX_FORMAL_EXPONENT))
        || exponent > ExactRational::integer(i64::from(MAX_FORMAL_EXPONENT))
    {
        return Err(PureOperatorError::FormalExponentLimit);
    }
    fraction(&exponent)?;
    Ok(exponent)
}
fn fraction(exponent: &ExactRational) -> Result<(i32, i32), PureOperatorError> {
    exponent
        .to_i32_ratio()
        .ok_or(PureOperatorError::ResultDimensionOverflow)
}
fn power(
    dimension: DimExponents,
    exponent: ExactRational,
) -> Result<DimExponents, PureOperatorError> {
    let (n, d) = fraction(&exponent)?;
    dimension
        .pow(n, d)
        .ok_or(PureOperatorError::ResultDimensionOverflow)
}
pub(super) fn derive_symbolic_dimension(
    formals: &[PureValueClass],
    nodes: &[CalculusNode],
    root: CalculusNodeId,
) -> Result<FormalDimensionMonomial, PureOperatorError> {
    let proofs = prove(formals, nodes)?;
    let proof = &proofs[definition_index(root, proofs.len())?];
    numeric(proof)?;
    Ok(proof.dimension.clone())
}
pub(super) fn expression_domain(
    formals: &[PureValueClass],
    nodes: &[CalculusNode],
    root: CalculusNodeId,
) -> Result<Option<ScalarDomain>, PureOperatorError> {
    let proofs = prove(formals, nodes)?;
    Ok(proofs[definition_index(root, proofs.len())?].domain)
}
impl PureOperatorDefinition {
    /// Instantiate the body's physical dimension from ordered input dimensions.
    /// Full value contracts are checked separately by [`Self::instantiate`].
    ///
    /// # Errors
    /// Rejects wrong arity or dimensions outside the portable SI exponent range.
    pub fn result_dimension(
        &self,
        arguments: &[DimExponents],
    ) -> Result<DimExponents, PureOperatorError> {
        if arguments.len() != self.formals.len() {
            return Err(PureOperatorError::ArityMismatch);
        }
        let mut result = self.dimension.fixed_dimension;
        for (argument, exponent) in arguments.iter().zip(self.dimension.exponents.iter()) {
            result = result
                .mul(power(*argument, exponent.clone())?)
                .ok_or(PureOperatorError::ResultDimensionOverflow)?;
        }
        Ok(result)
    }
}
fn normalized(
    formals: &[PureValueClass],
    monomial: &FormalDimensionMonomial,
) -> Result<(DimExponents, Vec<ExactRational>), PureOperatorError> {
    let mut fixed = monomial.fixed_dimension;
    let mut free = Vec::with_capacity(formals.len());
    for (formal, exponent) in formals.iter().zip(monomial.exponents.iter()) {
        if let Some(dimension) = formal.dimension() {
            fixed = fixed
                .mul(power(dimension, exponent.clone())?)
                .ok_or(PureOperatorError::ResultDimensionOverflow)?;
            free.push(ExactRational::integer(0));
        } else {
            free.push(exponent.clone());
        }
    }
    Ok((fixed, free))
}
pub(super) fn validate_result_dimension(
    formals: &[PureValueClass],
    result: &PureValueClass,
    monomial: &FormalDimensionMonomial,
) -> Result<(), PureOperatorError> {
    if let Some(expected) = result.dimension() {
        let (fixed, free) = normalized(formals, monomial)?;
        if fixed != expected || free.iter().any(|power| !power.is_zero()) {
            return Err(PureOperatorError::AdditiveDimensionMismatch);
        }
    }
    Ok(())
}

impl CalculusBuilder {
    /// Resolve the concrete scalar type of a component from the calculus type check.
    /// Unconstrained generic dimensions or scalar domains cannot give a concrete type.
    pub fn value_type(&self, node: CalculusNodeId) -> Result<ValueType, PureOperatorError> {
        concrete_value_type(&self.formals, &self.nodes, node)
    }
}

impl PureOperatorDefinition {
    /// Resolve a concrete component type with the same proof used during construction.
    pub fn value_type(&self, node: CalculusNodeId) -> Result<ValueType, PureOperatorError> {
        concrete_value_type(&self.formals, &self.nodes, node)
    }
}

fn concrete_value_type(
    formals: &[PureValueClass],
    nodes: &[CalculusNode],
    node: CalculusNodeId,
) -> Result<ValueType, PureOperatorError> {
    let proofs = prove(formals, nodes)?;
    let proof = &proofs[definition_index(node, proofs.len())?];
    let domain = proof.domain.ok_or(PureOperatorError::FormalTypeMismatch)?;
    let (dimension, free) = normalized(formals, &proof.dimension)?;
    if free.iter().any(|power| !power.is_zero()) {
        return Err(PureOperatorError::FormalTypeMismatch);
    }
    ValueType::scalar(domain, dimension).map_err(|_| PureOperatorError::FormalTypeMismatch)
}
