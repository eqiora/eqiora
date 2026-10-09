//! One retained request; live-source admission must establish its hypotheses.
use super::*;

pub(super) const IMPLICATION: &str = "harmonic-response-satisfies-original-relations";
pub(super) const ASSUMPTIONS: &[&str] = &[
    "fixed-domain",
    "linear-time-invariant",
    "positive-angular-frequency",
    "explicit-harmonic-excitation",
];

/// A harmonic response request, distinct from an initial-value equivalence claim.
/// Types and exact supports are inherited from the referenced original Fields;
/// source compilation checks any explicit amplitude type or support assertion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HarmonicFormulationRequest {
    convention: String,
    normalization: String,
    angular_frequency: AuthoredFormExpressionV1,
    relations: Vec<String>,
    excitations: Vec<(String, AuthoredFormExpressionV1)>,
    amplitudes: Vec<(String, String)>,
}

impl HarmonicFormulationRequest {
    pub(in crate::formulation) fn new(
        angular_frequency: AuthoredFormExpressionV1,
        relations: Vec<String>,
        excitations: Vec<(String, AuthoredFormExpressionV1)>,
        amplitudes: Vec<(String, String)>,
    ) -> Self {
        Self {
            convention: "negative-exponential".into(),
            normalization: "peak".into(),
            angular_frequency,
            relations,
            excitations,
            amplitudes,
        }
    }
    /// Exact original Relations, including every transformed source and boundary.
    pub fn relations(&self) -> &[String] {
        &self.relations
    }
    /// Ordered amplitude names and exact original Field identities.
    pub fn amplitudes(&self) -> &[(String, String)] {
        &self.amplitudes
    }
    /// Exact original signal input Port identities and their peak amplitude expressions.
    pub fn excitations(&self) -> &[(String, AuthoredFormExpressionV1)] {
        &self.excitations
    }
    /// Positive real angular frequency; physical units alone do not imply this role.
    pub fn angular_frequency(&self) -> &AuthoredFormExpressionV1 {
        &self.angular_frequency
    }
    /// The retained phase convention, never inferred from a backend.
    pub fn convention(&self) -> &str {
        &self.convention
    }
    /// The retained amplitude normalization.
    pub fn normalization(&self) -> &str {
        &self.normalization
    }

    pub(super) fn validate(&self, form: &WireForm) -> Result<(), Diagnostic> {
        if self.convention != "negative-exponential"
            || self.normalization != "peak"
            || form.implication != IMPLICATION
            || !form
                .assumptions
                .iter()
                .map(String::as_str)
                .eq(ASSUMPTIONS.iter().copied())
            || !form.equations.is_empty()
            || form.gauge.is_some()
            || form.domain_ulid.is_some()
            || self.relations.is_empty()
            || self.amplitudes.is_empty()
            || !form
                .trial_ulids
                .iter()
                .eq(self.amplitudes.iter().map(|(_, field)| field))
        {
            return Err(rejection(
                "harmonic request differs from its declared response restriction",
            ));
        }
        let canonical_id = |id: &str| {
            id.parse::<Ulid>()
                .ok()
                .is_some_and(|value| value.to_string() == id)
        };
        let mut relations = std::collections::BTreeSet::new();
        for relation in &self.relations {
            if !canonical_id(relation) || !relations.insert(relation) {
                return Err(rejection(
                    "harmonic Relations must be distinct exact identities",
                ));
            }
        }
        let mut fields = std::collections::BTreeSet::new();
        let mut names = std::collections::BTreeSet::new();
        for name in std::iter::once(&form.name).chain(self.amplitudes.iter().map(|(name, _)| name))
        {
            if name.is_empty()
                || !name.bytes().enumerate().all(|(i, c)| {
                    c.is_ascii_alphabetic() || c == b'_' || (i > 0 && c.is_ascii_digit())
                })
            {
                return Err(rejection(
                    "harmonic form and amplitude names must be identifiers",
                ));
            }
        }
        for (name, field) in &self.amplitudes {
            if !names.insert(name) {
                return Err(rejection("harmonic amplitude names must be unique"));
            }
            if !canonical_id(field) || !fields.insert(field) {
                return Err(rejection(
                    "harmonic unknown mappings require distinct exact Fields",
                ));
            }
        }
        let mut inputs = std::collections::BTreeSet::new();
        for (input, _) in &self.excitations {
            if !canonical_id(input) || !inputs.insert(input) {
                return Err(rejection(
                    "harmonic excitations require distinct exact input Ports",
                ));
            }
        }
        Ok(())
    }
}

impl AuthoredFormulationProjection {
    pub(in crate::formulation) fn encode_harmonic(
        source_identity: String,
        name: String,
        request: HarmonicFormulationRequest,
    ) -> Result<Self, Diagnostic> {
        let wire = WireForm {
            schema: SCHEMA.into(),
            source_identity,
            name,
            domain_ulid: None,
            trial_ulids: request
                .amplitudes
                .iter()
                .map(|(_, id)| id.clone())
                .collect(),
            binding: WireBinding::Harmonic { request },
            gauge: None,
            implication: IMPLICATION.into(),
            assumptions: ASSUMPTIONS.iter().map(|s| (*s).into()).collect(),
            equations: Vec::new(),
        };
        Self::decode(
            &serde_json::to_vec(&wire).map_err(|_| rejection("nonfinite harmonic request"))?,
        )
    }
    /// Retained harmonic candidate. Decoding alone never proves its LTI assumptions.
    pub fn harmonic_request(&self) -> Option<&HarmonicFormulationRequest> {
        match &self.wire.binding {
            WireBinding::Harmonic { request } => Some(request),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_request_rejects_changed_phase_scale_and_equivalence_claims() {
        let id = |value| Ulid::from(value).to_string();
        let request = HarmonicFormulationRequest::new(
            AuthoredFormExpressionV1::Rational {
                numerator: 1000,
                denominator: 1,
                dimension: [(0, 1), (0, 1), (-1, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
            },
            vec![id(1u128)],
            Vec::new(),
            vec![("u_hat".into(), id(2u128))],
        );
        let valid = AuthoredFormulationProjection::encode_harmonic(
            "0".repeat(64),
            "response".into(),
            request,
        )
        .unwrap();
        assert_eq!(
            AuthoredFormulationProjection::decode(valid.canonical_bytes()).unwrap(),
            valid
        );
        for mutation in 0..4 {
            let mut wire = valid.wire.clone();
            let WireBinding::Harmonic { request } = &mut wire.binding else {
                unreachable!()
            };
            match mutation {
                0 => request.convention = "positive-exponential".into(),
                1 => request.normalization = "rms".into(),
                2 => wire.implication = "equivalent-initial-value-problem".into(),
                3 => wire.assumptions.clear(),
                _ => unreachable!(),
            }
            let bytes = serde_json::to_vec(&wire).unwrap();
            let error = AuthoredFormulationProjection::decode(&bytes).unwrap_err();
            assert!(
                error.message().contains("response restriction"),
                "{error:?}"
            );
        }
    }
}
