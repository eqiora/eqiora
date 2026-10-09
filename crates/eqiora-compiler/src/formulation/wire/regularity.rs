//! Conditional continuum test spaces, independent of any numerical element.
use super::{AuthoredFormExpressionV1 as E, Diagnostic, WireBinding, WireForm, rejection};
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Space {
    H1,
    Curl,
    Div,
    L2,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Demand {
    Value,
    Gradient,
    Curl,
    Div,
    Trace,
    Tangential,
    Unsupported,
}

pub(super) fn check(wire: &WireForm) -> Result<(), Diagnostic> {
    let WireBinding::WeakTests { tests } = &wire.binding else {
        return Ok(());
    };
    let mut spaces = BTreeMap::new();
    for (_, trial, zero_on, _, regularity) in tests {
        if wire.domain_ulid.is_none() {
            if regularity.is_some() {
                return Err(rejection(
                    "global weak tests cannot declare spatial regularity",
                ));
            }
            continue;
        }
        let space = match regularity.as_deref() {
            Some("h1") => Space::H1,
            Some("hcurl") => Space::Curl,
            Some("hdiv") => Space::Div,
            Some("l2") => Space::L2,
            _ => {
                return Err(rejection(
                    "spatial test regularity must be h1, hcurl, hdiv or l2",
                ));
            }
        };
        if space != Space::H1 && !zero_on.is_empty() {
            return Err(rejection("full zero_on test traces require h1 regularity"));
        }
        spaces.insert(trial.as_str(), space);
    }
    for (_, left, right) in &wire.equations {
        visit(left, Demand::Value, &spaces)?;
        visit(right, Demand::Value, &spaces)?;
    }
    Ok(())
}

fn visit(value: &E, demand: Demand, spaces: &BTreeMap<&str, Space>) -> Result<(), Diagnostic> {
    // The existing H1/classical profile stays available. We admit only direct
    // first-order graph derivatives for the new, weaker test-space declarations;
    // this is not a general regularity inference or distribution-product engine.
    let composed = if demand == Demand::Value {
        demand
    } else {
        Demand::Unsupported
    };
    let derivative = |next| {
        if demand == Demand::Value {
            next
        } else {
            Demand::Unsupported
        }
    };
    match value {
        E::Test { field_ulid } | E::Direction { field_ulid, .. } => {
            let admitted = match spaces.get(field_ulid.as_str()) {
                None | Some(Space::H1) => true,
                Some(Space::Curl) => {
                    matches!(demand, Demand::Value | Demand::Curl | Demand::Tangential)
                }
                Some(Space::Div) => matches!(demand, Demand::Value | Demand::Div),
                Some(Space::L2) => demand == Demand::Value,
            };
            if !admitted {
                return Err(rejection(
                    "test derivative or trace exceeds its declared regularity profile",
                ));
            }
        }
        E::Gradient { value } => visit(value, derivative(Demand::Gradient), spaces)?,
        E::Curl { value } => visit(value, derivative(Demand::Curl), spaces)?,
        E::Divergence { value } => visit(value, derivative(Demand::Div), spaces)?,
        E::Trace { value } => visit(value, derivative(Demand::Trace), spaces)?,
        E::TangentialTrace { value } => visit(value, derivative(Demand::Tangential), spaces)?,
        E::CoordinatePartial { value, wrt } => {
            visit(value, derivative(Demand::Gradient), spaces)?;
            visit(wrt, Demand::Unsupported, spaces)?;
        }
        E::Neg { value } | E::Conjugate { value } => visit(value, demand, spaces)?,
        E::Add { left, right } | E::Sub { left, right } => {
            visit(left, demand, spaces)?;
            visit(right, demand, spaces)?;
        }
        E::Complex {
            real: left,
            imag: right,
        }
        | E::Mul { left, right }
        | E::Div { left, right }
        | E::Cross { left, right }
        | E::Frobenius { left, right }
        | E::Inner { left, right }
        | E::Apply { left, right }
        | E::Dot { left, right } => {
            visit(left, composed, spaces)?;
            visit(right, composed, spaces)?;
        }
        E::Component { value, .. }
        | E::Variation { value, .. }
        | E::SymmetricPart { value }
        | E::Sin { value }
        | E::Pow { base: value, .. }
        | E::Integrate {
            integrand: value, ..
        }
        | E::IntervalIntegral {
            integrand: value, ..
        }
        | E::EndpointFlux { flux: value, .. } => {
            visit(value, composed, spaces)?;
        }
        E::Number { .. }
        | E::Rational { .. }
        | E::Field { .. }
        | E::Parameter { .. }
        | E::Coordinate { .. }
        | E::LinearMap { .. } => {}
    }
    Ok(())
}
