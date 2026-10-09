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
    Normal,
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
        visit(left, Demand::Value, &spaces, None)?;
        visit(right, Demand::Value, &spaces, None)?;
    }
    Ok(())
}

fn visit(
    value: &E,
    demand: Demand,
    spaces: &BTreeMap<&str, Space>,
    integration_domain: Option<&str>,
) -> Result<(), Diagnostic> {
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
                Some(Space::Div) => matches!(demand, Demand::Value | Demand::Div | Demand::Normal),
                Some(Space::L2) => demand == Demand::Value,
            };
            if !admitted {
                return Err(rejection(
                    "test derivative or trace exceeds its declared regularity profile",
                ));
            }
        }
        E::Gradient { value } => visit(
            value,
            derivative(Demand::Gradient),
            spaces,
            integration_domain,
        )?,
        E::Curl { value } => visit(value, derivative(Demand::Curl), spaces, integration_domain)?,
        E::Divergence { value } => {
            visit(value, derivative(Demand::Div), spaces, integration_domain)?
        }
        E::Trace {
            value: operand,
            on_ulid,
        }
        | E::NormalTrace {
            value: operand,
            on_ulid,
        }
        | E::TangentialTrace {
            value: operand,
            on_ulid,
        } => {
            if integration_domain != Some(on_ulid.as_str())
                || on_ulid
                    .parse::<ulid::Ulid>()
                    .ok()
                    .map(|id| id.to_string())
                    .as_ref()
                    != Some(on_ulid)
            {
                return Err(rejection(
                    "trace target must be the exact canonical integration boundary",
                ));
            }
            let requested = if matches!(value, E::Trace { .. }) {
                Demand::Trace
            } else if matches!(value, E::NormalTrace { .. }) {
                Demand::Normal
            } else {
                Demand::Tangential
            };
            visit(operand, derivative(requested), spaces, integration_domain)?;
        }
        E::CoordinatePartial { value, wrt } => {
            visit(
                value,
                derivative(Demand::Gradient),
                spaces,
                integration_domain,
            )?;
            visit(wrt, Demand::Unsupported, spaces, integration_domain)?;
        }
        E::Neg { value } | E::Conjugate { value } => {
            visit(value, demand, spaces, integration_domain)?
        }
        E::Add { left, right } | E::Sub { left, right } => {
            visit(left, demand, spaces, integration_domain)?;
            visit(right, demand, spaces, integration_domain)?;
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
            visit(left, composed, spaces, integration_domain)?;
            visit(right, composed, spaces, integration_domain)?;
        }
        E::Integrate {
            domain_ulid,
            integrand,
        } => {
            if integration_domain.is_some() {
                return Err(rejection("weak forms cannot contain nested integrals"));
            }
            visit(integrand, composed, spaces, Some(domain_ulid))?;
        }
        E::Component { value, .. }
        | E::Variation { value, .. }
        | E::SymmetricPart { value }
        | E::Sin { value }
        | E::Pow { base: value, .. }
        | E::IntervalIntegral {
            integrand: value, ..
        }
        | E::EndpointFlux { flux: value, .. } => {
            visit(value, composed, spaces, integration_domain)?;
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
