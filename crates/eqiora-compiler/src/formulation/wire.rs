use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, RawId};
use eqiora_lang::BinaryOp;
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use super::{AuthoredFormExpression, AuthoredFormExpressionKind};

const SCHEMA: &str = "eqiora.authored-form/v7";
const MAX_BYTES: usize = 1024 * 1024;

/// Ordered test name, trial Field, zero-trace boundaries and canonical SI dimension.
pub type AuthoredTestRestriction = (String, String, Vec<String>, [(i32, i32); 7]);

/// Exact compiler-owned projection of one authored Formulation.
///
/// This is a source-compilation sidecar rather than Model meaning. Its
/// canonical bytes are retained in resolved Plan identity and may be decoded
/// during Plan replay. Canonical decoding validates the bounded structural
/// representation; mathematical correspondence requires its live-source checker.
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
    domain_ulid: Option<String>,
    trial_ulids: Vec<String>,
    name: String,
    binding: WireBinding,
    gauge: Option<WireGauge>,
    implication: String,
    assumptions: Vec<String>,
    equations: Vec<(String, AuthoredFormExpressionV1, AuthoredFormExpressionV1)>,
}

/// One declared constant scalar gauge; numerical admission checks its actual
/// nullspace and the original load independently of this authored condition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireGauge {
    pub(super) field_ulids: Vec<String>,
    pub(super) reference: (AuthoredFormExpressionV1, AuthoredFormExpressionV1),
    pub(super) compatibility: (AuthoredFormExpressionV1, AuthoredFormExpressionV1),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub(super) enum WireBinding {
    Finite {
        name: String,
    },
    WeakTests {
        tests: Vec<AuthoredTestRestriction>,
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
    Rational {
        numerator: i64,
        denominator: u64,
        dimension: [(i32, i32); 7],
    },
    Direction {
        name: String,
        field_ulid: String,
    },
    Component {
        value: Box<Self>,
        indices: Vec<u32>,
    },
    Variation {
        functional_ulid: String,
        wrt_ulid: String,
        directions: Vec<String>,
        holding: Vec<String>,
        value: Box<Self>,
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
    Divergence {
        value: Box<Self>,
    },
    SymmetricPart {
        value: Box<Self>,
    },
    Frobenius {
        left: Box<Self>,
        right: Box<Self>,
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
    pub(super) fn encode_finite(
        source_identity: String,
        name: String,
        space: String,
        trials: Vec<String>,
        equations: Vec<(String, AuthoredFormExpressionV1, AuthoredFormExpressionV1)>,
        gauge: Option<WireGauge>,
    ) -> Result<Self, Diagnostic> {
        let wire = WireForm {
            schema: SCHEMA.into(),
            source_identity,
            name,
            domain_ulid: None,
            trial_ulids: trials,
            binding: WireBinding::Finite { name: space },
            gauge,
            implication: "strong-equivalent-finite".into(),
            assumptions: super::finite::ASSUMPTIONS
                .iter()
                .map(|s| (*s).into())
                .collect(),
            equations,
        };
        Self::decode(&serde_json::to_vec(&wire).map_err(|_| rejection("nonfinite form"))?)
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
            domain_ulid: Some(ulid(domain)),
            trial_ulids: vec![ulid(trial)],
            name,
            binding: WireBinding::Interval {
                name: interval.0,
                lower: interval.1,
                upper: interval.2,
            },
            gauge: None,
            implication: "strong-implies-interval-conservation".into(),
            assumptions: super::interval::ASSUMPTIONS
                .iter()
                .map(|s| (*s).into())
                .collect(),
            equations: vec![(ulid(relation), equality.0, equality.1)],
        };
        let bytes = serde_json::to_vec(&wire)
            .map_err(|_| rejection("interval form is not finite canonical JSON"))?;
        Self::decode(&bytes)
    }

    pub(super) fn encode_weak(
        source_identity: String,
        name: String,
        domain: RawId,
        tests: Vec<AuthoredTestRestriction>,
        equations: Vec<(String, AuthoredFormExpressionV1, AuthoredFormExpressionV1)>,
    ) -> Result<Self, Diagnostic> {
        let mut trial_ulids = Vec::new();
        for (_, trial, _, _) in &tests {
            if !trial_ulids.contains(trial) {
                trial_ulids.push(trial.clone());
            }
        }
        let assumptions = if trial_ulids.len() > 1 {
            Self::mixed_assumptions()
        } else {
            Self::required_assumptions()
        };
        let wire = WireForm {
            schema: SCHEMA.into(),
            source_identity,
            domain_ulid: Some(ulid(domain)),
            trial_ulids,
            name,
            binding: WireBinding::WeakTests { tests },
            gauge: None,
            implication: "strong-implies-weak".into(),
            assumptions: assumptions.iter().map(|s| (*s).into()).collect(),
            equations,
        };
        Self::decode(&serde_json::to_vec(&wire).map_err(|_| rejection("nonfinite form"))?)
    }

    pub(super) fn with_gauge(mut self, gauge: WireGauge) -> Result<Self, Diagnostic> {
        self.wire.gauge = Some(gauge);
        Self::decode(&serde_json::to_vec(&self.wire).map_err(|_| rejection("nonfinite gauge"))?)
    }

    /// Exact Field carrying the declared constant scalar shift freedom.
    #[must_use]
    pub fn gauge_field_ulids(&self) -> Option<&[String]> {
        self.wire
            .gauge
            .as_ref()
            .map(|gauge| gauge.field_ulids.as_slice())
    }

    /// Authored reference equality, separate from the original Model equations.
    #[must_use]
    pub fn gauge_reference(
        &self,
    ) -> Option<(&AuthoredFormExpressionV1, &AuthoredFormExpressionV1)> {
        self.wire
            .gauge
            .as_ref()
            .map(|gauge| (&gauge.reference.0, &gauge.reference.1))
    }

    /// Authored load compatibility equality, never a permission to project a load.
    #[must_use]
    pub fn gauge_compatibility(
        &self,
    ) -> Option<(&AuthoredFormExpressionV1, &AuthoredFormExpressionV1)> {
        self.wire
            .gauge
            .as_ref()
            .map(|gauge| (&gauge.compatibility.0, &gauge.compatibility.1))
    }

    /// Decode exactly one bounded canonical v7 projection.
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
        if wire.equations.is_empty()
            || wire.equations.len() > 8
            || wire.trial_ulids.is_empty()
            || wire.trial_ulids.len() > 8
        {
            return Err(rejection(
                "form requires bounded nonempty equation and trial inventories",
            ));
        }
        let mut relations = std::collections::BTreeSet::new();
        for (relation, _, _) in &wire.equations {
            if !relations.insert(relation) {
                return Err(rejection("repeated equation Relation"));
            }
        }
        let mut trials = std::collections::BTreeSet::new();
        for trial in &wire.trial_ulids {
            if !trials.insert(trial) {
                return Err(rejection("repeated trial Field"));
            }
        }
        for value in wire.domain_ulid.iter().chain(relations).chain(trials) {
            if value.parse::<Ulid>().ok().map(|id| id.to_string()).as_ref() != Some(value) {
                return Err(rejection("form identity is not one canonical ULID"));
            }
        }
        if wire.implication
            != match wire.binding {
                WireBinding::Finite { .. } => "strong-equivalent-finite",
                WireBinding::WeakTests { .. } => "strong-implies-weak",
                WireBinding::Interval { .. } => "strong-implies-interval-conservation",
            }
            || !wire
                .assumptions
                .iter()
                .map(String::as_str)
                .eq(match wire.binding {
                    WireBinding::WeakTests { .. } if wire.trial_ulids.len() > 1 => {
                        Self::mixed_assumptions()
                    }
                    WireBinding::Finite { .. } => super::finite::ASSUMPTIONS,
                    WireBinding::WeakTests { .. } => Self::required_assumptions(),
                    WireBinding::Interval { .. } => super::interval::ASSUMPTIONS,
                }
                .iter()
                .copied())
        {
            return Err(rejection(
                "scalar implication or required hypotheses differ from the admitted profile",
            ));
        }
        if matches!(wire.binding, WireBinding::Finite { .. }) != wire.domain_ulid.is_none() {
            return Err(rejection(
                "finite forms have no spatial Domain; spatial forms require one",
            ));
        }
        if let Some(gauge) = &wire.gauge
            && (!matches!(
                wire.binding,
                WireBinding::Interval { .. } | WireBinding::Finite { .. }
            ) || wire.trial_ulids != gauge.field_ulids)
        {
            return Err(rejection(
                "constant gauge requires the exact ordered trial Fields",
            ));
        }
        let mut names = vec![wire.name.as_str()];
        match &wire.binding {
            WireBinding::Finite { name } => {
                names.push(name);
                if wire.equations.len() != wire.trial_ulids.len() {
                    return Err(rejection("finite equation and trial inventories differ"));
                }
            }
            WireBinding::WeakTests { tests } => {
                let second_direction =
                    second_direction_profile(tests, &wire.trial_ulids, &wire.equations);
                if !second_direction
                    && (tests.len() != wire.trial_ulids.len()
                        || tests.len() != wire.equations.len())
                {
                    return Err(rejection("equations and test/trial inventories differ"));
                }
                if second_direction && (tests[0].2 != tests[1].2 || tests[0].3 != tests[1].3) {
                    return Err(rejection(
                        "second variation directions require identical dimension and boundary restrictions",
                    ));
                }
                let mut seen = std::collections::BTreeSet::new();
                let mut test_names = std::collections::BTreeSet::new();
                for (test_name, trial, zero_on, dimension) in tests {
                    if eqiora_core::DimExponents::from_rationals(*dimension)
                        .is_none_or(|value| value.exponents() != *dimension)
                    {
                        return Err(rejection(
                            "test dimension must be canonical rational SI exponents",
                        ));
                    }
                    names.push(test_name);
                    if !test_names.insert(test_name) {
                        return Err(rejection("test names must be unique"));
                    }
                    if (!seen.insert(trial) && !second_direction)
                        || !wire.trial_ulids.contains(trial)
                    {
                        return Err(rejection("test has repeated or foreign trial"));
                    }
                    if zero_on.windows(2).any(|p| p[0] >= p[1]) {
                        return Err(rejection("test boundaries must be sorted and unique"));
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
            }
            WireBinding::Interval { name, lower, upper } => {
                if wire.equations.len() != 1 || wire.trial_ulids.len() != 1 {
                    return Err(rejection("interval requires one Law and Field"));
                }
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
    /// Named tests with their exact trial Field and zero-trace supports.
    #[must_use]
    pub fn test_restrictions(&self) -> &[AuthoredTestRestriction] {
        match &self.wire.binding {
            WireBinding::WeakTests { tests } => tests,
            _ => &[],
        }
    }
    /// Conditional hypotheses for the bounded real steady mixed form.
    #[must_use]
    pub const fn mixed_assumptions() -> &'static [&'static str] {
        &[
            "fixed-domain",
            "classical-divergence-and-boundary-trace",
            "admissible-h1-velocity-test-with-zero-essential-trace",
            "admissible-l2-pressure-test",
        ]
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
    pub fn domain_ulid(&self) -> Option<&str> {
        self.wire.domain_ulid.as_deref()
    }
    /// Ordered finite coordinate binder; no spatial Domain is fabricated.
    #[must_use]
    pub fn finite_space(&self) -> Option<&str> {
        match &self.wire.binding {
            WireBinding::Finite { name } => Some(name),
            _ => None,
        }
    }
    #[must_use]
    pub fn trial_ulids(&self) -> &[String] {
        &self.wire.trial_ulids
    }
    /// Exact Relation ownership and both mathematical sides of each equation.
    #[must_use]
    pub fn equations(&self) -> &[(String, AuthoredFormExpressionV1, AuthoredFormExpressionV1)] {
        &self.wire.equations
    }
}

pub(super) fn expression(value: &AuthoredFormExpression) -> AuthoredFormExpressionV1 {
    match &value.kind {
        AuthoredFormExpressionKind::Rational(rational) => AuthoredFormExpressionV1::Rational {
            numerator: rational.numerator(),
            denominator: rational.denominator(),
            dimension: value.dimension.exponents(),
        },
        AuthoredFormExpressionKind::Direction { name, trial } => {
            AuthoredFormExpressionV1::Direction {
                name: name.clone(),
                field_ulid: ulid(trial.erase()),
            }
        }
        AuthoredFormExpressionKind::Component { value, indices } => {
            AuthoredFormExpressionV1::Component {
                value: Box::new(expression(value)),
                indices: indices.clone(),
            }
        }
        AuthoredFormExpressionKind::Variation {
            functional,
            wrt,
            directions,
            holding,
            value,
        } => AuthoredFormExpressionV1::Variation {
            functional_ulid: ulid(functional.erase()),
            wrt_ulid: ulid(wrt.erase()),
            directions: directions.clone(),
            holding: holding.iter().map(|id| ulid(*id)).collect(),
            value: Box::new(expression(value)),
        },
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
        AuthoredFormExpressionKind::Divergence(value) => AuthoredFormExpressionV1::Divergence {
            value: Box::new(expression(value)),
        },
        AuthoredFormExpressionKind::SymmetricPart(value) => {
            AuthoredFormExpressionV1::SymmetricPart {
                value: Box::new(expression(value)),
            }
        }
        AuthoredFormExpressionKind::Frobenius(left, right) => AuthoredFormExpressionV1::Frobenius {
            left: Box::new(expression(left)),
            right: Box::new(expression(right)),
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

fn second_direction_profile(
    tests: &[AuthoredTestRestriction],
    trials: &[String],
    equations: &[(String, AuthoredFormExpressionV1, AuthoredFormExpressionV1)],
) -> bool {
    let ([first, second], [trial], [(_, left, right)]) = (tests, trials, equations) else {
        return false;
    };
    if first.1 != *trial || second.1 != *trial || first.0 == second.0 {
        return false;
    }
    [left, right].iter().any(|expression| {
        matches!(expression,
        AuthoredFormExpressionV1::Variation { wrt_ulid, directions, .. }
        if wrt_ulid == trial && directions.len() == 2 && directions[0] != directions[1]
            && directions.contains(&first.0) && directions.contains(&second.0))
    })
}

pub(super) fn rejection(message: &str) -> Diagnostic {
    Diagnostic::error(
        codes::INVALID_DISCRETIZATION,
        format!("authored Formulation rejected: {message}"),
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
        AuthoredFormulationProjection::encode_weak(
            "a".repeat(64),
            "weak".into(),
            Id::<kinds::Domain>::from_ulid(id("01ARZ3NDEKTSV4RRFFQ69G5FAW")).erase(),
            vec![(
                "w".into(),
                "01ARZ3NDEKTSV4RRFFQ69G5FAX".into(),
                vec!["01ARZ3NDEKTSV4RRFFQ69G5FAY".into()],
                DimExponents::DIMENSIONLESS.exponents(),
            )],
            vec![(
                "01ARZ3NDEKTSV4RRFFQ69G5FAV".into(),
                super::expression(&expression),
                super::expression(&expression),
            )],
        )
        .unwrap()
    }

    #[test]
    fn test_dimension_round_trip_rejects_noncanonical_exponents() {
        let mut wire = projection().wire;
        let length = [(0, 1), (1, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)];
        if let WireBinding::WeakTests { tests } = &mut wire.binding {
            tests[0].3 = length;
        }
        let admitted =
            AuthoredFormulationProjection::decode(&serde_json::to_vec(&wire).unwrap()).unwrap();
        assert_eq!(admitted.test_restrictions()[0].3, length);
        for exponent in [(2, 2), (1, 0), (-1, -1)] {
            if let WireBinding::WeakTests { tests } = &mut wire.binding {
                tests[0].3[1] = exponent;
            }
            assert!(
                AuthoredFormulationProjection::decode(&serde_json::to_vec(&wire).unwrap())
                    .unwrap_err()
                    .message()
                    .contains("canonical rational SI")
            );
        }
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
    fn plural_codec_rejects_duplicate_test_names_and_old_schema() {
        let mut wire = projection().wire;
        wire.trial_ulids.push("01ARZ3NDEKTSV4RRFFQ69G5FAZ".into());
        wire.equations.push((
            "01ARZ3NDEKTSV4RRFFQ69G5FB0".into(),
            AuthoredFormExpressionV1::Number { value: 0.0 },
            AuthoredFormExpressionV1::Number { value: 0.0 },
        ));
        if let WireBinding::WeakTests { tests } = &mut wire.binding {
            tests.push((
                "q".into(),
                "01ARZ3NDEKTSV4RRFFQ69G5FAZ".into(),
                vec![],
                DimExponents::DIMENSIONLESS.exponents(),
            ));
        }
        wire.assumptions = AuthoredFormulationProjection::mixed_assumptions()
            .iter()
            .map(|s| (*s).into())
            .collect();
        assert!(AuthoredFormulationProjection::decode(&serde_json::to_vec(&wire).unwrap()).is_ok());
        if let WireBinding::WeakTests { tests } = &mut wire.binding {
            tests[1].0 = tests[0].0.clone();
        }
        assert!(
            AuthoredFormulationProjection::decode(&serde_json::to_vec(&wire).unwrap())
                .unwrap_err()
                .message()
                .contains("unique")
        );
        let bytes = projection().canonical_bytes().to_vec();
        let old = String::from_utf8(bytes)
            .unwrap()
            .replace("eqiora.authored-form/v7", "eqiora.authored-scalar-form/v3");
        assert!(AuthoredFormulationProjection::decode(old.as_bytes()).is_err());
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
            .replace("\"equations\"", "\"unknown\":0,\"equations\"");
        assert!(AuthoredFormulationProjection::decode(unknown.as_bytes()).is_err());

        let malformed_identity = String::from_utf8(bytes.to_vec())
            .unwrap()
            .replace("01ARZ3NDEKTSV4RRFFQ69G5FAV", "not-a-canonical-ulid-value");
        assert!(AuthoredFormulationProjection::decode(malformed_identity.as_bytes()).is_err());

        assert!(AuthoredFormulationProjection::decode(&vec![b' '; MAX_BYTES + 1]).is_err());
    }
}
