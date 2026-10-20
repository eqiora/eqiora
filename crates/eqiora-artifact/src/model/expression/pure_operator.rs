//! Definition table admission and aggregate pure-calculus resource bounds.
use super::*;

impl WireExpression {
    pub(crate) fn pure_operator_counts(&self) -> Result<PureOperatorWireCounts, Diagnostic> {
        let formals = checked_count_sum(
            self.definitions
                .iter()
                .map(|definition| definition.formals.len()),
            "pure-operator formal count",
        )?;
        let calculus_nodes = checked_count_sum(
            self.definitions
                .iter()
                .map(|definition| definition.nodes.len()),
            "pure-operator calculus-node count",
        )?;
        let application_arguments = checked_count_sum(
            self.nodes.iter().map(WireExpressionNode::application_arity),
            "pure-operator application-argument count",
        )?;
        Ok(PureOperatorWireCounts {
            definitions: self.definitions.len(),
            formals,
            calculus_nodes,
            application_arguments,
        })
    }

    pub(crate) fn validate_pure_operator_features(&self) -> Result<(), Diagnostic> {
        for definition in &self.definitions {
            definition.validate_features()?;
        }
        Ok(())
    }

    pub(crate) fn canonicalize_pure_operator_definitions(&mut self) -> Result<(), Diagnostic> {
        for definition in &self.definitions {
            definition.rebuild_and_validate_digest()?;
        }
        self.definitions
            .sort_by(|left, right| left.digest.cmp(&right.digest));
        if self
            .definitions
            .windows(2)
            .any(|pair| pair[0].digest == pair[1].digest)
        {
            return Err(invalid_artifact(
                "wire expression contains a duplicate pure-operator definition digest",
            ));
        }
        Ok(())
    }

    pub(crate) fn decode_definitions(
        &self,
    ) -> Result<BTreeMap<String, PureOperatorDefinition>, Diagnostic> {
        let mut definitions = BTreeMap::new();
        for definition in &self.definitions {
            let rebuilt = definition.rebuild_and_validate_digest()?;
            if definitions
                .insert(definition.digest.clone(), rebuilt)
                .is_some()
            {
                return Err(invalid_artifact(
                    "wire expression contains a duplicate pure-operator definition digest",
                ));
            }
        }
        Ok(definitions)
    }
}

/// Aggregate resource counters for the current pure-operator vocabulary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PureOperatorWireCounts {
    pub(crate) definitions: usize,
    pub(crate) formals: usize,
    pub(crate) calculus_nodes: usize,
    pub(crate) application_arguments: usize,
}

impl PureOperatorWireCounts {
    pub(crate) fn checked_add(self, other: Self) -> Result<Self, Diagnostic> {
        Ok(Self {
            definitions: self
                .definitions
                .checked_add(other.definitions)
                .ok_or_else(|| {
                    invalid_artifact("pure-operator definition count overflows usize")
                })?,
            formals: self
                .formals
                .checked_add(other.formals)
                .ok_or_else(|| invalid_artifact("pure-operator formal count overflows usize"))?,
            calculus_nodes: self
                .calculus_nodes
                .checked_add(other.calculus_nodes)
                .ok_or_else(|| {
                    invalid_artifact("pure-operator calculus-node count overflows usize")
                })?,
            application_arguments: self
                .application_arguments
                .checked_add(other.application_arguments)
                .ok_or_else(|| {
                    invalid_artifact("pure-operator application-argument count overflows usize")
                })?,
        })
    }

    pub(crate) fn ensure_limits(
        self,
        limits: ModelDecoderLimits,
        label: &str,
    ) -> Result<(), Diagnostic> {
        require_decoder_count(
            &format!("{label} pure-operator definitions"),
            self.definitions,
            limits.max_pure_operator_definitions,
        )?;
        require_decoder_count(
            &format!("{label} pure-operator formals"),
            self.formals,
            limits.max_pure_operator_formals,
        )?;
        require_decoder_count(
            &format!("{label} pure-operator calculus nodes"),
            self.calculus_nodes,
            limits.max_pure_operator_calculus_nodes,
        )?;
        require_decoder_count(
            &format!("{label} pure-operator application arguments"),
            self.application_arguments,
            limits.max_pure_operator_application_arguments,
        )
    }
}
