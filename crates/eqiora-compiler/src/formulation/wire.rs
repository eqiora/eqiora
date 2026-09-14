use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, RawId};
use eqiora_lang::BinaryOp;
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use super::{AuthoredFormExpression, AuthoredFormExpressionKind};

const SCHEMA: &str = "eqiora.authored-scalar-form/v3";
const MAX_BYTES: usize = 1024 * 1024;

/// Exact compiler-owned projection of one authored scalar Formulation.
///
/// This is a source-compilation sidecar rather than Model meaning. Its
/// canonical bytes are retained in resolved Plan identity and may be decoded
/// during Plan replay; callers cannot construct a projection without passing
/// the closed canonical decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct AuthoredFormulationProjection {
    wire: WireForm,
    canonical_bytes: Box<[u8]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireForm {
    schema: String,
    source_identity: String,
    relation_ulid: String,
    domain_ulid: String,
    trial_ulid: String,
    name: String,
    binding: WireBinding,
    implication: String,
    assumptions: Vec<String>,
    left: AuthoredFormExpressionV1,
    right: AuthoredFormExpressionV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum WireBinding {
    WeakTest {
        test_name: String,
        zero_on: Vec<String>,
    },
    Interval {
        name: String,
        lower: String,
        upper: String,
    },
}

/// Closed expression vocabulary persisted by
/// [`AuthoredFormulationProjection`].
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
#[non_exhaustive]
pub enum AuthoredFormExpressionV1 {
    EndpointFlux {
        interval: String,
        endpoint: String,
        normal: i8,
        flux: Box<Self>,
    },
    IntervalIntegral {
        interval: String,
        integrand: Box<Self>,
    },
    Number {
        value: f64,
    },
    Field {
        ulid: String,
    },
    Parameter {
        ulid: String,
    },
    Coordinate {
        axis: usize,
    },
    Test {
        field_ulid: String,
    },
    Neg {
        value: Box<Self>,
    },
    Add {
        left: Box<Self>,
        right: Box<Self>,
    },
    Sub {
        left: Box<Self>,
        right: Box<Self>,
    },
    Mul {
        left: Box<Self>,
        right: Box<Self>,
    },
    Div {
        left: Box<Self>,
        right: Box<Self>,
    },
    Pow {
        base: Box<Self>,
        exponent: i32,
    },
    Gradient {
        value: Box<Self>,
    },
    Sin {
        value: Box<Self>,
    },
    Dot {
        left: Box<Self>,
        right: Box<Self>,
    },
    Integrate {
        domain_ulid: String,
        integrand: Box<Self>,
    },
}

impl AuthoredFormulationProjection {
    pub(super) fn encode(
        source_identity: String,
        relation: RawId,
        domain: RawId,
        trial: RawId,
        restriction: (String, String, Vec<String>),
        left: &AuthoredFormExpression,
        right: &AuthoredFormExpression,
    ) -> Self {
        let wire = WireForm {
            schema: SCHEMA.to_owned(),
            source_identity,
            relation_ulid: ulid(relation),
            domain_ulid: ulid(domain),
            trial_ulid: ulid(trial),
            name: restriction.0,
            binding: WireBinding::WeakTest {
                test_name: restriction.1,
                zero_on: restriction.2,
            },
            implication: "strong-implies-weak".into(),
            assumptions: Self::required_assumptions()
                .iter()
                .map(|s| (*s).into())
                .collect(),
            left: expression(left),
            right: expression(right),
        };
        let canonical_bytes = serde_json::to_vec(&wire)
            .expect("typed authored Formulation is canonical JSON")
            .into_boxed_slice();
        Self {
            wire,
            canonical_bytes,
        }
    }

    pub(super) fn encode_interval(
        source_identity: String,
        relation: RawId,
        domain: RawId,
        trial: RawId,
        name: String,
        interval: (String, String, String),
        equality: (AuthoredFormExpressionV1, AuthoredFormExpressionV1),
    ) -> Result<Self, Diagnostic> {
        let wire = WireForm {
            schema: SCHEMA.into(),
            source_identity,
            relation_ulid: ulid(relation),
            domain_ulid: ulid(domain),
            trial_ulid: ulid(trial),
            name,
            binding: WireBinding::Interval {
                name: interval.0,
                lower: interval.1,
                upper: interval.2,
            },
            implication: "strong-implies-interval-conservation".into(),
            assumptions: super::interval::ASSUMPTIONS
                .iter()
                .map(|s| (*s).into())
                .collect(),
            left: equality.0,
            right: equality.1,
        };
        let bytes = serde_json::to_vec(&wire)
            .map_err(|_| rejection("interval form is not finite canonical JSON"))?;
        Self::decode(&bytes)
    }

    /// Decode exactly one bounded canonical v3 projection.
    ///
    /// # Errors
    /// Returns a diagnostic for an oversized, malformed, noncanonical, or
    /// identity-malformed projection.
    pub fn decode(bytes: &[u8]) -> Result<Self, Diagnostic> {
        if bytes.len() > MAX_BYTES {
            return Err(rejection("projection exceeds the decoder limit"));
        }
        let wire: WireForm = serde_json::from_slice(bytes)
            .map_err(|_| rejection("projection is not the closed canonical wire"))?;
        if wire.schema != SCHEMA || serde_json::to_vec(&wire).ok().as_deref() != Some(bytes) {
            return Err(rejection(
                "projection schema or canonical encoding is invalid",
            ));
        }
        if wire.source_identity.len() != 64
            || !wire
                .source_identity
                .bytes()
                .all(|value| value.is_ascii_digit() || (b'a'..=b'f').contains(&value))
        {
            return Err(rejection(
                "source identity is not one canonical SHA-256 digest",
            ));
        }
        for (label, value) in [
            ("Relation", wire.relation_ulid.as_str()),
            ("Domain", wire.domain_ulid.as_str()),
            ("trial Field", wire.trial_ulid.as_str()),
        ] {
            let parsed = value
                .parse::<Ulid>()
                .map_err(|_| rejection(&format!("{label} identity is not one canonical ULID")))?;
            if parsed.to_string() != value {
                return Err(rejection(&format!(
                    "{label} identity is not one canonical ULID"
                )));
            }
        }
        if wire.implication
            != match wire.binding {
                WireBinding::WeakTest { .. } => "strong-implies-weak",
                WireBinding::Interval { .. } => "strong-implies-interval-conservation",
            }
            || !wire
                .assumptions
                .iter()
                .map(String::as_str)
                .eq(match wire.binding {
                    WireBinding::WeakTest { .. } => Self::required_assumptions(),
                    WireBinding::Interval { .. } => super::interval::ASSUMPTIONS,
                }
                .iter()
                .copied())
        {
            return Err(rejection(
                "scalar implication or required hypotheses differ from the admitted profile",
            ));
        }
        let mut names = vec![wire.name.as_str()];
        match &wire.binding {
            WireBinding::WeakTest { test_name, zero_on } => {
                names.push(test_name);
                if zero_on.is_empty() || zero_on.windows(2).any(|pair| pair[0] >= pair[1]) {
                    return Err(rejection(
                        "test boundaries must be nonempty, sorted and unique",
                    ));
                }
                for boundary in zero_on {
                    if boundary
                        .parse::<Ulid>()
                        .ok()
                        .map(|id| id.to_string())
                        .as_ref()
                        != Some(boundary)
                    {
                        return Err(rejection("test boundary is not one canonical ULID"));
                    }
                }
            }
            WireBinding::Interval { name, lower, upper } => {
                if name == lower || name == upper || lower == upper {
                    return Err(rejection("interval binders must be distinct"));
                }
                names.extend([name.as_str(), lower.as_str(), upper.as_str()]);
            }
        }
        for name in names {
            if name.is_empty()
                || !name.bytes().enumerate().all(|(i, c)| {
                    c.is_ascii_alphabetic() || c == b'_' || (i > 0 && c.is_ascii_digit())
                })
            {
                return Err(rejection(
                    "form and binder names must be source identifiers",
                ));
            }
        }
        Ok(Self {
            wire,
            canonical_bytes: bytes.into(),
        })
    }

    /// Conditional scalar implication; the reverse implication is not admitted.
    #[must_use]
    pub fn implication(&self) -> &str {
        &self.wire.implication
    }
    /// Closed hypotheses retained in the scalar projection, not proved by numerical execution.
    #[must_use]
    pub const fn required_assumptions() -> &'static [&'static str] {
        &[
            "fixed-domain",
            "classical-divergence-and-boundary-trace",
            "admissible-h1-test-with-zero-essential-trace",
        ]
    }
    /// Exact hypotheses bound by this projection's identity.
    #[must_use]
    pub fn assumptions(&self) -> &[String] {
        &self.wire.assumptions
    }

    /// Authored form name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.wire.name
    }
    /// Test name and exact zero-trace boundaries, only for a weak form.
    #[must_use]
    pub fn test_restriction(&self) -> Option<(&str, &[String])> {
        match &self.wire.binding {
            WireBinding::WeakTest { test_name, zero_on } => Some((test_name, zero_on)),
            _ => None,
        }
    }
    /// Universally quantified interval and its ordered endpoint binders.
    #[must_use]
    pub fn interval(&self) -> Option<(&str, &str, &str)> {
        match &self.wire.binding {
            WireBinding::Interval { name, lower, upper } => Some((name, lower, upper)),
            _ => None,
        }
    }

    #[must_use]
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    #[must_use]
    pub fn source_identity(&self) -> &str {
        &self.wire.source_identity
    }

    #[must_use]
    pub fn relation_ulid(&self) -> &str {
        &self.wire.relation_ulid
    }

    #[must_use]
    pub fn domain_ulid(&self) -> &str {
        &self.wire.domain_ulid
    }

    #[must_use]
    pub fn trial_ulid(&self) -> &str {
        &self.wire.trial_ulid
    }

    #[must_use]
    pub const fn left(&self) -> &AuthoredFormExpressionV1 {
        &self.wire.left
    }

    #[must_use]
    pub const fn right(&self) -> &AuthoredFormExpressionV1 {
        &self.wire.right
    }
}

pub(super) fn expression(value: &AuthoredFormExpression) -> AuthoredFormExpressionV1 {
    match &value.kind {
        AuthoredFormExpressionKind::Number(value) => {
            AuthoredFormExpressionV1::Number { value: *value }
        }
        AuthoredFormExpressionKind::Field(id) => AuthoredFormExpressionV1::Field {
            ulid: ulid(id.erase()),
        },
        AuthoredFormExpressionKind::Parameter(id) => AuthoredFormExpressionV1::Parameter {
            ulid: ulid(id.erase()),
        },
        AuthoredFormExpressionKind::Coordinate(axis) => {
            AuthoredFormExpressionV1::Coordinate { axis: *axis }
        }
        AuthoredFormExpressionKind::Test(id) => AuthoredFormExpressionV1::Test {
            field_ulid: ulid(id.erase()),
        },
        AuthoredFormExpressionKind::Neg(value) => AuthoredFormExpressionV1::Neg {
            value: Box::new(expression(value)),
        },
        AuthoredFormExpressionKind::Binary {
            operator,
            left,
            right,
        } => {
            let left = Box::new(expression(left));
            let right = Box::new(expression(right));
            match operator {
                BinaryOp::Add => AuthoredFormExpressionV1::Add { left, right },
                BinaryOp::Sub => AuthoredFormExpressionV1::Sub { left, right },
                BinaryOp::Mul => AuthoredFormExpressionV1::Mul { left, right },
                BinaryOp::Div => AuthoredFormExpressionV1::Div { left, right },
                BinaryOp::Pow => unreachable!("power is represented by the typed Pow node"),
                _ => unreachable!("form checking rejects Boolean predicates"),
            }
        }
        AuthoredFormExpressionKind::Pow(base, exponent) => AuthoredFormExpressionV1::Pow {
            base: Box::new(expression(base)),
            exponent: *exponent,
        },
        AuthoredFormExpressionKind::Gradient(value) => AuthoredFormExpressionV1::Gradient {
            value: Box::new(expression(value)),
        },
        AuthoredFormExpressionKind::Sin(value) => AuthoredFormExpressionV1::Sin {
            value: Box::new(expression(value)),
        },
        AuthoredFormExpressionKind::Dot(left, right) => AuthoredFormExpressionV1::Dot {
            left: Box::new(expression(left)),
            right: Box::new(expression(right)),
        },
        AuthoredFormExpressionKind::Integrate { domain, integrand } => {
            AuthoredFormExpressionV1::Integrate {
                domain_ulid: ulid(domain.erase()),
                integrand: Box::new(expression(integrand)),
            }
        }
    }
}

fn ulid(id: RawId) -> String {
    id.ulid().to_string()
}

fn rejection(message: &str) -> Diagnostic {
    Diagnostic::error(
        codes::INVALID_DISCRETIZATION,
        format!("authored scalar Formulation rejected: {message}"),
    )
}

#[cfg(test)]
mod tests {
    use eqiora_core::entity::kinds;
    use eqiora_core::{DimExponents, Id, ValueShape};

    use super::*;

    fn projection() -> AuthoredFormulationProjection {
        let id = |value: &str| value.parse::<Ulid>().expect("fixed ULID");
        let expression = AuthoredFormExpression {
            kind: AuthoredFormExpressionKind::Number(1.0),
            dimension: DimExponents::DIMENSIONLESS,
            shape: ValueShape::scalar(),
            support: None,
        };
        AuthoredFormulationProjection::encode(
            "a".repeat(64),
            Id::<kinds::Relation>::from_ulid(id("01ARZ3NDEKTSV4RRFFQ69G5FAV")).erase(),
            Id::<kinds::Domain>::from_ulid(id("01ARZ3NDEKTSV4RRFFQ69G5FAW")).erase(),
            Id::<kinds::Field>::from_ulid(id("01ARZ3NDEKTSV4RRFFQ69G5FAX")).erase(),
            (
                "weak".into(),
                "w".into(),
                vec!["01ARZ3NDEKTSV4RRFFQ69G5FAY".into()],
            ),
            &expression,
            &expression,
        )
    }

    #[test]
    fn conditional_implication_rejects_changed_or_missing_hypotheses() {
        let projection = projection();
        let text = String::from_utf8(projection.canonical_bytes().to_vec()).unwrap();
        for changed in [
            text.replace("strong-implies-weak", "weak-implies-strong"),
            text.replace("fixed-domain", "moving-domain"),
        ] {
            assert!(
                AuthoredFormulationProjection::decode(changed.as_bytes())
                    .unwrap_err()
                    .message()
                    .contains("hypotheses")
            );
        }
    }

    #[test]
    fn one_codec_owns_canonical_round_trip_and_fail_closed_decode() {
        let projection = projection();
        let bytes = projection.canonical_bytes();
        assert_eq!(
            AuthoredFormulationProjection::decode(bytes).unwrap(),
            projection
        );

        let mut trailing = bytes.to_vec();
        trailing.push(b' ');
        assert!(AuthoredFormulationProjection::decode(&trailing).is_err());

        let unknown = String::from_utf8(bytes.to_vec())
            .unwrap()
            .replace("\"relation_ulid\"", "\"unknown\":0,\"relation_ulid\"");
        assert!(AuthoredFormulationProjection::decode(unknown.as_bytes()).is_err());

        let malformed_identity = String::from_utf8(bytes.to_vec())
            .unwrap()
            .replace("01ARZ3NDEKTSV4RRFFQ69G5FAV", "not-a-canonical-ulid-value");
        assert!(AuthoredFormulationProjection::decode(malformed_identity.as_bytes()).is_err());

        assert!(AuthoredFormulationProjection::decode(&vec![b' '; MAX_BYTES + 1]).is_err());
    }
}
