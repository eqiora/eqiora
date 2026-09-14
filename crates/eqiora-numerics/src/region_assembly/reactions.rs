//! Exact Domain residuals, accumulated once before Connection projection.
use super::invalid;
use eqiora_assembly::{AssemblyDelta, AssemblyRowDelta, AssemblyTargetId, AssemblyWork};
use eqiora_core::{Diagnostic, RawId};
use std::collections::{BTreeMap, BTreeSet};

mod interfaces;
pub(crate) use interfaces::{InterfaceReactions, RecoveredInterfaceReactions};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DomainReactions {
    size: usize,
    rows: BTreeMap<RawId, Vec<AssemblyRowDelta>>,
}

impl DomainReactions {
    pub(crate) fn prepare(
        work: &dyn AssemblyWork,
        source_target: AssemblyTargetId,
        size: usize,
        packet_domains: &[RawId],
        rows: &BTreeSet<usize>,
    ) -> Result<Self, Diagnostic> {
        if size == 0
            || packet_domains.len() != work.packet_count()
            || rows.iter().any(|&row| row >= size)
            || packet_domains.iter().any(|domain| {
                domain
                    .downcast::<eqiora_core::entity::kinds::Domain>()
                    .is_none()
            })
        {
            return Err(invalid(
                "reaction recovery requires exact packet/Domain and full-row inventory",
            ));
        }
        let mut selected = BTreeMap::<_, Vec<_>>::new();
        for (index, domain) in packet_domains.iter().copied().enumerate() {
            let packet = work.evaluate(index)?;
            let mapping = packet
                .mappings()
                .iter()
                .find(|mapping| mapping.target() == source_target)
                .ok_or_else(|| invalid("reaction packet omits its full-system map"))?;
            let delta = AssemblyDelta::from_local(size, mapping.map(), packet.local())?;
            selected.entry(domain).or_default().extend(
                delta
                    .rows()
                    .iter()
                    .filter(|row| rows.contains(&row.row().index()))
                    .cloned(),
            );
        }
        Ok(Self {
            size,
            rows: selected,
        })
    }

    pub(crate) fn recover(&self, values: &[f64]) -> Result<RecoveredDomainReactions, Diagnostic> {
        if values.len() != self.size || values.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "reaction values differ from the exact finite full solution",
            ));
        }
        let mut recovered = BTreeMap::new();
        for (&domain, rows) in &self.rows {
            let mut residual = vec![0.0; self.size];
            for row in rows {
                let mut product_sum = 0.0;
                for (column, coefficient) in row.entries() {
                    let product = coefficient * values[column.index()];
                    product_sum += product;
                }
                residual[row.row().index()] += product_sum - row.rhs();
            }
            if residual.iter().any(|value| !value.is_finite()) {
                return Err(invalid("Domain reaction accumulation is non-finite"));
            }
            recovered.insert(domain, residual);
        }
        Ok(RecoveredDomainReactions { values: recovered })
    }
}

pub(crate) struct RecoveredDomainReactions {
    pub(crate) values: BTreeMap<RawId, Vec<f64>>,
}
