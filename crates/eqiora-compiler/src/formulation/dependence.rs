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

fn check_complex_form(
    value: &AuthoredFormExpressionV1,
    domains: &BTreeMap<RawId, ScalarDomain>,
    trials: &BTreeSet<RawId>,
) -> Result<(), Diagnostic> {
    let mut budget = 65536;
    for term in classify(value, false, domains, trials, &mut budget, 0)? {
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
    value: &AuthoredFormExpressionV1,
    conjugated: bool,
    domains: &BTreeMap<RawId, ScalarDomain>,
    trials: &BTreeSet<RawId>,
    budget: &mut usize,
    depth: usize,
) -> Result<Terms, Diagnostic> {
    use AuthoredFormExpressionV1 as E;
    spend(budget)?;
    if depth > 128 {
        return Err(wire::rejection(
            "form dependence exceeds its expression depth",
        ));
    }
    let mut child = |v, conjugated| classify(v, conjugated, domains, trials, budget, depth + 1);
    let atom = |argument, complex| {
        let anti = conjugated && complex;
        BTreeSet::from([BTreeMap::from([(
            argument,
            if anti { (0, 1) } else { (1, 0) },
        )])])
    };
    Ok(match value {
        E::Number { value } if *value == 0.0 => Terms::new(),
        E::Rational { numerator: 0, .. } => Terms::new(),
        E::Number { .. }
        | E::Rational { .. }
        | E::Parameter { .. }
        | E::Coordinate { .. }
        | E::LinearMap { .. } => constant(),
        E::Field { ulid } => {
            let id = field_id(ulid)?;
            if trials.contains(&id) {
                atom(Argument::Trial(id), domains[&id] == ScalarDomain::Complex)
            } else {
                constant()
            }
        }
        E::Test { field_ulid } | E::Direction { field_ulid, .. } => {
            let id = field_id(field_ulid)?;
            if !trials.contains(&id) {
                return Err(wire::rejection(
                    "test or direction is outside the declared trial inventory",
                ));
            }
            let complex = domains[&id] == ScalarDomain::Complex;
            let argument = if let E::Direction { name, .. } = value {
                Argument::Direction(name.clone(), complex)
            } else {
                Argument::Test(id, complex)
            };
            atom(argument, complex)
        }
        E::Conjugate { value } => child(value, !conjugated)?,
        E::Neg { value }
        | E::Trace { value }
        | E::Gradient { value }
        | E::Curl { value }
        | E::TangentialTrace { value }
        | E::Divergence { value }
        | E::SymmetricPart { value }
        | E::Component { value, .. }
        | E::Integrate {
            integrand: value, ..
        } => child(value, conjugated)?,
        E::Inner { left, right }
        | E::Cross { left, right }
        | E::Dot { left, right }
        | E::Frobenius { left, right } => {
            let left = child(left, conjugated ^ matches!(value, E::Inner { .. }))?;
            let right = child(right, conjugated)?;
            product(&left, &right, budget)?
        }
        E::Complex {
            real: left,
            imag: right,
        }
        | E::Add { left, right }
        | E::Sub { left, right } => {
            let mut result = child(left, conjugated)?;
            result.extend(child(right, conjugated)?);
            result
        }
        E::Mul { left, right } | E::Apply { left, right } => {
            let left = child(left, conjugated)?;
            let right = child(right, conjugated)?;
            product(&left, &right, budget)?
        }
        E::Div { left, right } => {
            let left = child(left, conjugated)?;
            if child(right, conjugated)?
                .iter()
                .any(|term| !term.is_empty())
            {
                return Err(rejection());
            }
            left
        }
        E::Pow { base, exponent } => {
            let base = child(base, conjugated)?;
            if *exponent == 0 || base.iter().all(BTreeMap::is_empty) {
                constant()
            } else if *exponent == 1 {
                base
            } else {
                return Err(rejection());
            }
        }
        E::Sin { value } => {
            if child(value, conjugated)?
                .iter()
                .any(|term| !term.is_empty())
            {
                return Err(rejection());
            }
            constant()
        }
        E::Variation { .. }
        | E::EndpointFlux { .. }
        | E::IntervalIntegral { .. }
        | E::CoordinatePartial { .. } => return Err(rejection()),
    })
}

fn field_id(text: &str) -> Result<RawId, Diagnostic> {
    canonical_id(text).map(|id| Id::<kinds::Field>::from_ulid(id).erase())
}

fn canonical_id(text: &str) -> Result<ulid::Ulid, Diagnostic> {
    text.parse::<ulid::Ulid>()
        .ok()
        .filter(|id| id.to_string() == text)
        .ok_or_else(|| wire::rejection("form dependence requires a canonical live identity"))
}

impl AuthoredFormulationProjection {
    /// Recheck complex weak-form argument dependence against live definitions.
    ///
    /// The resolver must obtain each Field/Parameter type from the admitted Model.
    /// Declared trial Fields are arguments; other live Fields are coefficients.
    /// This does not authenticate supports, dimensions, restrictions, correspondence,
    /// Hermitian symmetry, or solver suitability. Real-only and non-weak profiles
    /// retain their separate admission checks. Decoding alone never calls this checker.
    /// # Errors
    /// Rejects missing live identities, non-sesquilinear complex argument dependence,
    /// and expressions exceeding the bounded structural classification inventory.
    pub fn check_complex_dependence(
        &self,
        resolve: &mut dyn FnMut(RawId) -> Result<ValueType, Diagnostic>,
    ) -> Result<(), Diagnostic> {
        if self.test_restrictions().is_empty() {
            return Ok(());
        }
        let trials = self
            .trial_ulids()
            .iter()
            .map(|id| field_id(id))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let mut domains = BTreeMap::new();
        for id in &trials {
            domains.insert(*id, resolve(*id)?.scalar_domain());
        }
        let mut complex = domains
            .values()
            .any(|domain| *domain == ScalarDomain::Complex);
        let mut pending = self
            .equations()
            .iter()
            .flat_map(|(_, a, b)| [(a, 0), (b, 0)])
            .collect::<Vec<_>>();
        let mut budget = 65536;
        use AuthoredFormExpressionV1 as E;
        while let Some((value, depth)) = pending.pop() {
            spend(&mut budget)?;
            if depth > 128 {
                return Err(wire::rejection(
                    "form dependence exceeds its expression depth",
                ));
            }
            let mut push = |value| pending.push((value, depth + 1));
            let id = match value {
                E::Field { ulid } => Some(field_id(ulid)?),
                E::Test { field_ulid } | E::Direction { field_ulid, .. } => {
                    Some(field_id(field_ulid)?)
                }
                E::Parameter { ulid } => {
                    Some(Id::<kinds::Parameter>::from_ulid(canonical_id(ulid)?).erase())
                }
                E::Complex { real, imag } => {
                    complex = true;
                    push(real.as_ref());
                    push(imag.as_ref());
                    None
                }
                E::Add { left, right }
                | E::Sub { left, right }
                | E::Mul { left, right }
                | E::Apply { left, right }
                | E::Div { left, right }
                | E::Inner { left, right }
                | E::Cross { left, right }
                | E::Dot { left, right }
                | E::Frobenius { left, right } => {
                    push(left.as_ref());
                    push(right.as_ref());
                    None
                }
                E::CoordinatePartial { value, wrt } => {
                    push(value.as_ref());
                    push(wrt.as_ref());
                    None
                }
                E::Neg { value }
                | E::Trace { value }
                | E::Gradient { value }
                | E::Curl { value }
                | E::TangentialTrace { value }
                | E::Divergence { value }
                | E::SymmetricPart { value }
                | E::Sin { value }
                | E::Conjugate { value }
                | E::Component { value, .. }
                | E::Variation { value, .. }
                | E::Integrate {
                    integrand: value, ..
                }
                | E::IntervalIntegral {
                    integrand: value, ..
                }
                | E::EndpointFlux { flux: value, .. }
                | E::Pow { base: value, .. } => {
                    push(value.as_ref());
                    None
                }
                E::LinearMap {
                    complex: domain, ..
                } => {
                    complex |= *domain;
                    None
                }
                E::Number { .. } | E::Rational { .. } | E::Coordinate { .. } => None,
            };
            if let Some(id) = id {
                let domain = match domains.entry(id) {
                    std::collections::btree_map::Entry::Occupied(entry) => *entry.get(),
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        *entry.insert(resolve(id)?.scalar_domain())
                    }
                };
                complex |= domain == ScalarDomain::Complex;
            }
        }
        if complex {
            for (_, left, right) in self.equations() {
                check_complex_form(left, &domains, &trials)?;
                check_complex_form(right, &domains, &trials)?;
            }
        }
        Ok(())
    }
}
