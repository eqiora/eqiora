//! One mixed Boolean/numeric proof for retained calculus nodes and exact dimensions.
use super::*;
use eqiora_core::{ScalarDomain, ValueType};

#[derive(Clone)]
struct Proof {
    domain: Option<ScalarDomain>,
    dimension: FormalDimensionMonomial,
}

pub(super) fn validate_profile(
    formals: &[PureValueClass],
    result: PureValueClass,
    nodes: &[CalculusNode],
) -> Result<(), PureOperatorError> {
    if nodes.iter().any(|node| {
        matches!(
            node,
            CalculusNode::Differentiated { .. }
                | CalculusNode::Require { .. }
                | CalculusNode::Boolean(_)
                | CalculusNode::Compare(..)
                | CalculusNode::Not(_)
                | CalculusNode::And(..)
                | CalculusNode::Or(..)
                | CalculusNode::Select { .. }
                | CalculusNode::UnaryMath(..)
        )
    }) && formals.iter().chain([&result]).any(|class| {
        !class.is_invariant_scalar()
            || class.dimension().is_none()
            || class.scalar_domain() != Some(ScalarDomain::Real)
    }) {
        return Err(PureOperatorError::FormalTypeMismatch);
    }
    Ok(())
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
            CalculusNode::Differentiated { value, source, wrt } => {
                let value = get(*value)?;
                let source = get(*source)?;
                let wrt = get(*wrt)?;
                numeric(&value)?;
                numeric(&source)?;
                numeric(&wrt)?;
                let mut quotient = source;
                quotient.dimension.fixed_dimension = quotient
                    .dimension
                    .fixed_dimension
                    .div(wrt.dimension.fixed_dimension)
                    .ok_or(PureOperatorError::ResultDimensionOverflow)?;
                for (exponent, divisor) in quotient
                    .dimension
                    .exponents
                    .iter_mut()
                    .zip(wrt.dimension.exponents.iter())
                {
                    *exponent = bounded(exponent.checked_add(divisor.checked_neg()?)?)?;
                }
                same_dimension(formals, &value, &quotient)?;
                value
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
            CalculusNode::Mul(left, right) => {
                let mut left = get(*left)?;
                let right = get(*right)?;
                numeric(&left)?;
                numeric(&right)?;
                left.dimension.fixed_dimension = left
                    .dimension
                    .fixed_dimension
                    .mul(right.dimension.fixed_dimension)
                    .ok_or(PureOperatorError::ResultDimensionOverflow)?;
                for (left, right) in left
                    .dimension
                    .exponents
                    .iter_mut()
                    .zip(right.dimension.exponents.iter())
                {
                    *left = bounded(left.checked_add(*right)?)?;
                }
                left.domain = common(left.domain, right.domain);
                left
            }
            CalculusNode::UnaryMath(function, value) => {
                let mut value = get(*value)?;
                if value.domain != Some(ScalarDomain::Real) {
                    return Err(PureOperatorError::FormalTypeMismatch);
                }
                if matches!(
                    function,
                    super::super::UnaryMathFunction::Sin
                        | super::super::UnaryMathFunction::Cos
                        | super::super::UnaryMathFunction::Exp
                        | super::super::UnaryMathFunction::Log
                ) {
                    if normalized(formals, &value.dimension)?
                        != normalized(formals, &dimensionless(formals.len()))?
                    {
                        return Err(PureOperatorError::FormalTypeMismatch);
                    }
                    value.dimension = dimensionless(formals.len());
                    proofs.push(value);
                    continue;
                }
                use super::super::UnaryMathFunction;
                let (numerator, denominator) = match function {
                    UnaryMathFunction::Sqrt => (1, 2),
                    UnaryMathFunction::Abs2 => (2, 1),
                    UnaryMathFunction::Arg => {
                        value.dimension = dimensionless(formals.len());
                        proofs.push(value);
                        continue;
                    }
                    _ => {
                        proofs.push(value);
                        continue;
                    }
                };
                value.dimension.fixed_dimension = value
                    .dimension
                    .fixed_dimension
                    .pow(numerator, denominator)
                    .ok_or(PureOperatorError::ResultDimensionOverflow)?;
                for exponent in &mut value.dimension.exponents {
                    *exponent =
                        bounded(exponent.checked_mul(ExactRational::new(
                            numerator.into(),
                            denominator.into(),
                        )?)?)?;
                }
                value
            }
        };
        // Check fixed formal dimensions before any downstream component expansion.
        normalized(formals, &proof.dimension)?;
        proofs.push(proof);
    }
    Ok(proofs)
}

fn bounded(exponent: ExactRational) -> Result<ExactRational, PureOperatorError> {
    if i128::from(exponent.numerator()).abs()
        > i128::from(MAX_FORMAL_EXPONENT) * i128::from(exponent.denominator())
    {
        return Err(PureOperatorError::FormalExponentLimit);
    }
    fraction(exponent)?;
    Ok(exponent)
}
fn fraction(exponent: ExactRational) -> Result<(i32, i32), PureOperatorError> {
    Ok((
        i32::try_from(exponent.numerator())
            .map_err(|_| PureOperatorError::ResultDimensionOverflow)?,
        i32::try_from(exponent.denominator())
            .map_err(|_| PureOperatorError::ResultDimensionOverflow)?,
    ))
}
fn power(
    dimension: DimExponents,
    exponent: ExactRational,
) -> Result<DimExponents, PureOperatorError> {
    let (n, d) = fraction(exponent)?;
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
pub(super) fn instantiate_dimension<I>(
    monomial: &FormalDimensionMonomial,
    arguments: &[ExpressionType<I>],
) -> Result<DimExponents, PureOperatorError> {
    let mut result = monomial.fixed_dimension;
    for (argument, exponent) in arguments.iter().zip(monomial.exponents()) {
        result = result
            .mul(power(argument.dimension(), *exponent)?)
            .ok_or(PureOperatorError::ResultDimensionOverflow)?;
    }
    Ok(result)
}
fn normalized(
    formals: &[PureValueClass],
    monomial: &FormalDimensionMonomial,
) -> Result<(DimExponents, Vec<ExactRational>), PureOperatorError> {
    let mut fixed = monomial.fixed_dimension;
    let mut free = Vec::with_capacity(formals.len());
    for (formal, exponent) in formals.iter().zip(monomial.exponents()) {
        if let Some(dimension) = formal.dimension() {
            fixed = fixed
                .mul(power(dimension, *exponent)?)
                .ok_or(PureOperatorError::ResultDimensionOverflow)?;
            free.push(ExactRational::integer(0));
        } else {
            free.push(*exponent);
        }
    }
    Ok((fixed, free))
}
pub(super) fn validate_result_dimension(
    formals: &[PureValueClass],
    result: PureValueClass,
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
    /// Resolves a concrete scalar component type using the retained calculus proof.
    /// Unconstrained generic dimensions or scalar domains cannot supply a concrete type.
    pub fn value_type(&self, node: CalculusNodeId) -> Result<ValueType, PureOperatorError> {
        let proofs = prove(&self.formals, &self.nodes)?;
        let proof = &proofs[definition_index(node, proofs.len())?];
        let domain = proof.domain.ok_or(PureOperatorError::FormalTypeMismatch)?;
        let (dimension, free) = normalized(&self.formals, &proof.dimension)?;
        if free.iter().any(|power| !power.is_zero()) {
            return Err(PureOperatorError::FormalTypeMismatch);
        }
        ValueType::scalar(domain, dimension).map_err(|_| PureOperatorError::FormalTypeMismatch)
    }
}
