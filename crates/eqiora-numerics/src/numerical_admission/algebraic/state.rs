//! One exact numerical seed; source initial equations remain mathematical constraints.
use super::*;

/// Initial numerical values bound to one exact finite Plan.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonAlgebraicState {
    pub(super) plan_identity: String,
    pub(super) identity: String,
    pub(super) values: Vec<f64>,
}

impl CommonAlgebraicState {
    pub fn to_bytes(&self) -> Result<Vec<u8>, Diagnostic> {
        serde_json::to_vec(&(
            "eqiora.common-algebraic-state/v2",
            &self.plan_identity,
            &self.identity,
            &self.values,
        ))
        .map_err(|e| invalid(format!("cannot encode finite State: {e}")))
    }
    /// Reconstruct through the same Plan-bound seed admission, then require canonical bytes.
    pub fn from_bytes(bytes: &[u8], plan: &CommonAlgebraicPlan) -> Result<Self, Diagnostic> {
        if bytes.len() > 1024 * 1024 {
            return Err(invalid("finite State exceeds its byte bound"));
        }
        let (schema, owner, identity, values): (String, String, String, Vec<f64>) =
            serde_json::from_slice(bytes)
                .map_err(|error| invalid(format!("invalid finite State: {error}")))?;
        let state = plan.state_from_values(values)?;
        if schema != "eqiora.common-algebraic-state/v2"
            || owner != plan.identity()
            || identity != state.identity
            || bytes != state.to_bytes()?
        {
            return Err(invalid(
                "finite State differs from its canonical exact Plan-bound seed",
            ));
        }
        Ok(state)
    }
    /// Exact numerical Plan that owns this seed.
    #[must_use]
    pub fn plan_identity(&self) -> &str {
        &self.plan_identity
    }
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

impl CommonAlgebraicPlan {
    /// Complete exact Field assignments for nonlinear seeds; affine execution uses no assignments.
    pub fn initial_state(
        &self,
        fields: &[CommonInitialField],
    ) -> Result<CommonAlgebraicState, Diagnostic> {
        if !self
            .enforcement()
            .is_some_and(FiniteConstraintEnforcement::is_strict_interior)
        {
            if !fields.is_empty() {
                return Err(invalid("affine finite State requires no seed assignments"));
            }
            return self.state_from_values(vec![0.0; self.symbols.len()]);
        }
        if fields.len() != self.symbols.len() {
            return Err(invalid(
                "nonlinear finite State requires exactly one scalar assignment per Field",
            ));
        }
        let mut values = vec![None; self.symbols.len()];
        for field in fields {
            if field.model().as_str() != self.model_digest() {
                return Err(invalid("finite seed Field belongs to another exact Model"));
            }
            let index = self
                .symbols
                .iter()
                .position(|symbol| *symbol == SymbolRef::Field(field.field()))
                .ok_or_else(|| invalid("finite seed Field is outside the Plan"))?;
            let value = field
                .scalar_value()
                .ok_or_else(|| invalid("finite seed requires a no-Mesh scalar association"))?;
            if values[index].replace(value).is_some() {
                return Err(invalid("finite seed repeats a Field assignment"));
            }
        }
        self.state_from_values(
            values
                .into_iter()
                .map(|value| value.expect("complete unique field coverage"))
                .collect(),
        )
    }

    pub(super) fn state_from_values(
        &self,
        mut values: Vec<f64>,
    ) -> Result<CommonAlgebraicState, Diagnostic> {
        if values.len() != self.symbols.len() || values.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "finite seed requires the complete finite Plan coordinate vector",
            ));
        }
        if self
            .enforcement()
            .is_some_and(FiniteConstraintEnforcement::is_strict_interior)
        {
            self.problem.validate_seed(&values)?;
        } else if values.iter().any(|value| *value != 0.0) {
            return Err(invalid("affine finite State has a canonical zero seed"));
        }
        for value in &mut values {
            if *value == 0.0 {
                *value = 0.0;
            }
        }
        let bytes = serde_json::to_vec(&(self.identity(), &values))
            .map_err(|error| invalid(format!("cannot identify finite seed: {error}")))?;
        Ok(CommonAlgebraicState {
            plan_identity: self.identity.clone(),
            identity: finite_digest(b"eqiora.common-algebraic-initial-state/v2\0", &bytes),
            values,
        })
    }
}
