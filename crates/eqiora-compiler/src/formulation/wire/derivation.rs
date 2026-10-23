//! Explicit derivation requests retain their row choices beside effective equations.
use super::*;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn requested(wire: &WireForm) -> Result<BTreeMap<String, String>, Diagnostic> {
    let WireBinding::WeakTests { tests, derivations } = &wire.binding else {
        return Ok(BTreeMap::new());
    };
    if derivations.is_empty() {
        return Ok(BTreeMap::new());
    }
    let mut rows = BTreeMap::new();
    let mut used = BTreeSet::new();
    for (relation, field) in derivations {
        for value in [relation, field] {
            if value.parse::<Ulid>().ok().map(|id| id.to_string()).as_ref() != Some(value) {
                return Err(rejection(
                    "derivation association requires canonical identities",
                ));
            }
        }
        if rows.insert(relation.clone(), field.clone()).is_some() || !used.insert(field) {
            return Err(rejection("derivation repeats a Relation or test Field"));
        }
    }
    if used != tests.iter().map(|(_, field, _, _, _)| field).collect() {
        return Err(rejection(
            "derivation must consume exactly the declared tests",
        ));
    }
    Ok(rows)
}

impl AuthoredFormulationProjection {
    /// Whether this input requests derivation from explicit equation/test choices.
    #[must_use]
    pub fn requests_weak_derivation(&self) -> bool {
        matches!(&self.wire.binding, WireBinding::WeakTests { derivations, .. } if !derivations.is_empty())
    }

    pub(in crate::formulation) fn requested_test_fields(
        &self,
    ) -> Result<BTreeMap<RawId, RawId>, Diagnostic> {
        requested(&self.wire)?
            .into_iter()
            .map(|(relation, field)| {
                Ok((
                    eqiora_core::Id::<eqiora_core::entity::kinds::Relation>::from_ulid(
                        relation
                            .parse()
                            .map_err(|_| rejection("invalid Relation"))?,
                    )
                    .erase(),
                    eqiora_core::Id::<eqiora_core::entity::kinds::Field>::from_ulid(
                        field.parse().map_err(|_| rejection("invalid Field"))?,
                    )
                    .erase(),
                ))
            })
            .collect()
    }

    /// Attach effective equations to a derivation request without changing its choices.
    /// The numerical correspondence owner must still authenticate the derived expressions.
    /// # Errors
    /// Rejects non-request inputs, malformed expressions and changed equation/test choices.
    pub fn with_derived_equations(
        &self,
        equations: Vec<(String, AuthoredFormExpressionV1, AuthoredFormExpressionV1)>,
    ) -> Result<Self, Diagnostic> {
        if !self.requests_weak_derivation() || equations.is_empty() {
            return Err(rejection(
                "effective equations require a nonempty weak derivation request",
            ));
        }
        let mut wire = self.wire.clone();
        wire.equations = equations;
        let result = Self::decode(
            &serde_json::to_vec(&wire).map_err(|_| rejection("nonfinite derived form"))?,
        )?;
        result.equation_test_fields()?;
        Ok(result)
    }
}
