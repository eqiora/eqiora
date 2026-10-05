//! Structural argument grades for the admitted complex linear weak-form profile.
//! Coefficients are not sampled, and cancellation is never assumed.
use std::collections::{BTreeMap, BTreeSet};

use super::*;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Argument {
    Trial(RawId),
    Test(RawId, bool),
    Direction(String, bool),
}

// (linear, antilinear) degree per independently named argument. Degrees
// saturate at two because the admitted profile cannot consume a higher degree.
type Term = BTreeMap<Argument, (u8, u8)>;
type Terms = BTreeSet<Term>;

pub(super) fn check_complex_form(value: &AuthoredFormExpression) -> Result<(), Diagnostic> {
    let mut budget = 65536;
    for term in classify(value, false, &mut budget, 0)? {
        let (mut tests, mut trials) = (0_u32, 0_u32);
        for (argument, (linear, antilinear)) in term {
            match argument {
                Argument::Trial(_) => {
                    if antilinear != 0 {
                        return Err(rejection());
                    }
                    trials += u32::from(linear);
                }
                Argument::Test(_, complex) | Argument::Direction(_, complex) => {
                    if (complex && linear != 0) || (!complex && antilinear != 0) {
                        return Err(rejection());
                    }
                    tests += u32::from(linear) + u32::from(antilinear);
                }
            }
        }
        if tests != 1 || trials > 1 {
            return Err(rejection());
        }
    }
    Ok(())
}

fn rejection() -> Diagnostic {
    wire::rejection(
        "complex weak forms require conjugate-linear test dependence and linear trial dependence",
    )
}

fn spend(budget: &mut usize) -> Result<(), Diagnostic> {
    *budget = budget.checked_sub(1).ok_or_else(|| {
        wire::rejection("form dependence expansion exceeds its expression budget")
    })?;
    Ok(())
}

fn constant() -> Terms {
    BTreeSet::from([BTreeMap::new()])
}

fn product(left: &Terms, right: &Terms, budget: &mut usize) -> Result<Terms, Diagnostic> {
    let mut result = BTreeSet::new();
    for left in left {
        for right in right {
            spend(budget)?;
            let mut term = left.clone();
            for (argument, (linear, antilinear)) in right {
                let degree = term.entry(argument.clone()).or_default();
                degree.0 = degree.0.saturating_add(*linear).min(2);
                degree.1 = degree.1.saturating_add(*antilinear).min(2);
            }
            result.insert(term);
        }
    }
    Ok(result)
}

fn classify(
    value: &AuthoredFormExpression,
    conjugated: bool,
    budget: &mut usize,
    depth: usize,
) -> Result<Terms, Diagnostic> {
    use AuthoredFormExpressionKind as E;
    spend(budget)?;
    if depth > 128 {
        return Err(wire::rejection(
            "form dependence exceeds its expression depth",
        ));
    }
    let complex = value.value_type.scalar_domain() == ScalarDomain::Complex;
    let mut child = |v, conjugated| classify(v, conjugated, budget, depth + 1);
    let atom = |argument| {
        let anti = conjugated && complex;
        BTreeSet::from([BTreeMap::from([(
            argument,
            if anti { (0, 1) } else { (1, 0) },
        )])])
    };
    Ok(match &value.kind {
        E::Number(x) if *x == 0.0 => Terms::new(),
        E::Number(_) | E::Rational(_) | E::Parameter(_) | E::Coordinate { .. } => constant(),
        E::Field(id) => atom(Argument::Trial(id.erase())),
        E::Test(id) => atom(Argument::Test(id.erase(), complex)),
        E::Direction { name, .. } => atom(Argument::Direction(name.clone(), complex)),
        E::Conjugate(v) => child(v, !conjugated)?,
        E::Neg(v)
        | E::Trace(v)
        | E::Gradient(v)
        | E::Divergence(v)
        | E::SymmetricPart(v)
        | E::Component { value: v, .. }
        | E::Integrate { integrand: v, .. } => child(v, conjugated)?,
        E::Inner(left, right) | E::Dot(left, right) | E::Frobenius(left, right) => {
            let left = child(left, conjugated ^ matches!(&value.kind, E::Inner(..)))?;
            let right = child(right, conjugated)?;
            product(&left, &right, budget)?
        }
        E::Complex(left, right)
        | E::Binary {
            operator: BinaryOp::Add | BinaryOp::Sub,
            left,
            right,
        } => {
            let mut result = child(left, conjugated)?;
            result.extend(child(right, conjugated)?);
            result
        }
        E::Binary {
            operator: BinaryOp::Mul,
            left,
            right,
        } => {
            let left = child(left, conjugated)?;
            let right = child(right, conjugated)?;
            product(&left, &right, budget)?
        }
        E::Binary {
            operator: BinaryOp::Div,
            left,
            right,
        } => {
            let left = child(left, conjugated)?;
            if child(right, conjugated)?
                .iter()
                .any(|term| !term.is_empty())
            {
                return Err(rejection());
            }
            left
        }
        E::Pow(base, exponent) => {
            let base = child(base, conjugated)?;
            if *exponent == 0 || base.iter().all(BTreeMap::is_empty) {
                constant()
            } else if *exponent == 1 {
                base
            } else {
                return Err(rejection());
            }
        }
        E::Sin(v) => {
            if child(v, conjugated)?.iter().any(|term| !term.is_empty()) {
                return Err(rejection());
            }
            constant()
        }
        E::Variation { .. } | E::Binary { .. } => return Err(rejection()),
    })
}
