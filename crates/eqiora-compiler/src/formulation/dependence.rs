//! Structural test grades for complex weak residuals, independently of State affinity.
//! Coefficients are not sampled, and cancellation is never assumed.
use std::collections::{BTreeMap, BTreeSet};

use super::*;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Argument {
    Test(RawId, bool),
    Direction(String, bool),
}

// (linear, antilinear) degree per independently named argument. Degrees
// saturate at two because a weak residual must be linear in its test argument.
type Term = BTreeMap<Argument, (u8, u8)>;
type Terms = BTreeSet<Term>;

fn check_complex_form(
    value: &AuthoredFormExpressionV1,
    domains: &BTreeMap<RawId, ScalarDomain>,
    trials: &BTreeSet<RawId>,
) -> Result<(), Diagnostic> {
    let mut budget = 65536;
    for term in classify(value, false, domains, trials, &mut budget, 0)? {
        let mut tests = 0_u32;
        for (argument, (linear, antilinear)) in term {
            match argument {
                Argument::Test(_, complex) | Argument::Direction(_, complex) => {
                    if (complex && linear != 0) || (!complex && antilinear != 0) {
                        return Err(rejection());
                    }
                    tests += u32::from(linear) + u32::from(antilinear);
                }
            }
        }
        if tests != 1 {
            return Err(rejection());
        }
    }
    Ok(())
}

fn rejection() -> Diagnostic {
    wire::rejection("complex weak residuals require conjugate-linear test dependence")
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
        E::Rational { value, .. } if value.is_zero() => Terms::new(),
        E::Number { .. }
        | E::Rational { .. }
        | E::Parameter { .. }
        | E::Coordinate { .. }
        | E::LinearMap { .. } => constant(),
        // Unknown State may enter nonlinearly or antilinearly. Its affinity,
        // derivative actions and solver suitability belong to numerical admission.
        // This traversal certifies only dependence on the independent test.
        E::Field { .. } | E::TimeDerivative { .. } => constant(),
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
        | E::Trace { value, .. }
        | E::NormalTrace { value, .. }
        | E::Gradient { value }
        | E::Curl { value }
        | E::TangentialTrace { value, .. }
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
    /// Test and variation-direction binders are the graded arguments. State and
    /// coefficient dependence may be nonlinear or nonholomorphic.
    /// This does not authenticate supports, dimensions, restrictions, correspondence,
    /// State affinity, Hermitian symmetry, or solver suitability. Real-only and non-weak profiles
    /// retain their separate admission checks. Decoding alone never calls this checker.
    /// # Errors
    /// Rejects missing live identities, non-conjugate-linear complex test dependence,
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
        walk(
            self.equations().iter().flat_map(|(_, a, b)| [a, b]),
            |value| {
                use AuthoredFormExpressionV1 as E;
                let id = match value {
                    E::Field { ulid } | E::TimeDerivative { field_ulid: ulid } => {
                        Some(field_id(ulid)?)
                    }
                    E::Test { field_ulid } | E::Direction { field_ulid, .. } => {
                        Some(field_id(field_ulid)?)
                    }
                    E::Parameter { ulid } => {
                        Some(Id::<kinds::Parameter>::from_ulid(canonical_id(ulid)?).erase())
                    }
                    E::Complex { .. } => {
                        complex = true;
                        None
                    }
                    E::LinearMap {
                        complex: domain, ..
                    } => {
                        complex |= *domain;
                        None
                    }
                    _ => None,
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
                Ok(())
            },
        )?;
        if complex {
            for (_, left, right) in self.equations() {
                check_complex_form(left, &domains, &trials)?;
                check_complex_form(right, &domains, &trials)?;
            }
        }
        Ok(())
    }
}

impl AuthoredFormulationProjection {
    /// Recover each residual's declared test Field from its actual arguments.
    /// Declaration order and principal differential operands never choose the row.
    /// This does not establish source correspondence or numerical admissibility.
    /// # Errors
    /// Rejects missing, foreign or multiple test arguments in an equation, repeated
    /// test rows, and direction binders outside the declared test inventory.
    pub fn equation_test_fields(&self) -> Result<BTreeMap<RawId, RawId>, Diagnostic> {
        let requested = self.requested_test_fields()?;
        if self.equations().is_empty() && !requested.is_empty() {
            return Ok(requested);
        }
        let declared = self
            .test_restrictions()
            .iter()
            .map(|(_, field, _, _, _)| field_id(field))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let mut rows = BTreeMap::new();
        let mut used = BTreeSet::new();
        for (relation, left, right) in self.equations() {
            let mut tests = BTreeSet::new();
            walk([left, right], |value| {
                match value {
                    AuthoredFormExpressionV1::Test { field_ulid } => {
                        tests.insert(field_id(field_ulid)?);
                    }
                    AuthoredFormExpressionV1::Direction { name, field_ulid } => {
                        if !self
                            .test_restrictions()
                            .iter()
                            .any(|(test, field, _, _, _)| test == name && field == field_ulid)
                        {
                            return Err(wire::rejection(
                                "variation direction is not a declared test argument",
                            ));
                        }
                        tests.insert(field_id(field_ulid)?);
                    }
                    _ => {}
                }
                Ok(())
            })?;
            if tests.len() != 1 || !tests.is_subset(&declared) {
                return Err(wire::rejection(
                    "each residual requires exactly one declared test Field",
                ));
            }
            let tested = *tests.first().expect("one test");
            if !used.insert(tested) {
                return Err(wire::rejection("residual rows repeat a test Field"));
            }
            rows.insert(
                Id::<kinds::Relation>::from_ulid(canonical_id(relation)?).erase(),
                tested,
            );
        }
        if used != declared {
            return Err(wire::rejection("residual rows omit a declared test Field"));
        }
        if !requested.is_empty() && rows != requested {
            return Err(wire::rejection(
                "derived equation test arguments differ from requested associations",
            ));
        }
        Ok(rows)
    }
}

/// Shared bounded traversal for semantic queries over the existing expression vocabulary.
fn walk<'a>(
    roots: impl IntoIterator<Item = &'a AuthoredFormExpressionV1>,
    mut visit: impl FnMut(&AuthoredFormExpressionV1) -> Result<(), Diagnostic>,
) -> Result<(), Diagnostic> {
    use AuthoredFormExpressionV1 as E;
    let mut pending = roots
        .into_iter()
        .map(|value| (value, 0))
        .collect::<Vec<_>>();
    let mut budget = 65536;
    while let Some((value, depth)) = pending.pop() {
        spend(&mut budget)?;
        if depth > 128 {
            return Err(wire::rejection(
                "form dependence exceeds its expression depth",
            ));
        }
        visit(value)?;
        let mut push = |value| pending.push((value, depth + 1));
        match value {
            E::Complex { real, imag } => {
                push(real.as_ref());
                push(imag.as_ref());
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
            }
            E::CoordinatePartial { value, wrt } => {
                push(value.as_ref());
                push(wrt.as_ref());
            }
            E::Neg { value }
            | E::Trace { value, .. }
            | E::NormalTrace { value, .. }
            | E::Gradient { value }
            | E::Curl { value }
            | E::TangentialTrace { value, .. }
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
            | E::Pow { base: value, .. } => push(value.as_ref()),
            E::Field { .. }
            | E::TimeDerivative { .. }
            | E::Test { .. }
            | E::Direction { .. }
            | E::Parameter { .. }
            | E::LinearMap { .. }
            | E::Number { .. }
            | E::Rational { .. }
            | E::Coordinate { .. } => {}
        }
    }
    Ok(())
}
