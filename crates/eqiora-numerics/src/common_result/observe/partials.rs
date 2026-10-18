//! Real-coordinate covectors of reduced observations before the implicit solve.
use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, RawId};

use super::invalid;

#[derive(Clone)]
pub(crate) struct Covector {
    pub(crate) fields: BTreeMap<RawId, Vec<f64>>,
    pub(crate) parameters: Vec<f64>,
}

impl Covector {
    pub(super) fn zero(parameters: usize) -> Self {
        Self {
            fields: BTreeMap::new(),
            parameters: vec![0.; parameters],
        }
    }

    pub(super) fn add_scaled(&mut self, other: &Self, scale: f64) -> Result<(), Diagnostic> {
        fn add(target: &mut [f64], source: &[f64], scale: f64) -> Result<(), Diagnostic> {
            if target.len() != source.len() || !scale.is_finite() {
                return Err(invalid("Observable covector has inconsistent coordinates"));
            }
            for (target, source) in target.iter_mut().zip(source) {
                *target += scale * source;
                if !target.is_finite() {
                    return Err(invalid("Observable covector is non-finite"));
                }
            }
            Ok(())
        }
        add(&mut self.parameters, &other.parameters, scale)?;
        for (id, source) in &other.fields {
            let target = self
                .fields
                .entry(*id)
                .or_insert_with(|| vec![0.; source.len()]);
            add(target, source, scale)?;
        }
        Ok(())
    }
}
