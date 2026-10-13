use super::*;
mod projection;
use crate::region_assembly::mapping::{FieldDof, RegionDofMap};
pub(crate) use projection::InterfaceReactionProjection;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InterfaceReactions<S> {
    domains: DomainReactions<S>,
    projection: InterfaceReactionProjection,
    volume_loads: BTreeMap<FieldDof, S>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RecoveredInterfaceReactions<S> {
    actions: BTreeMap<(RawId, FieldDof), S>,
    pub(crate) imbalance_norm: f64,
    /// Normalized weak dual coordinates, retaining exact Field/Domain ownership.
    pub(crate) constrained_actions: BTreeMap<FieldDof, S>,
    /// Volume-packet RHS coordinates, excluding exterior traction packets.
    pub(crate) volume_loads: BTreeMap<FieldDof, S>,
    /// Exact inverse row factors for physical weak duals produced by shared forms.
    pub(crate) physical_scales: BTreeMap<RawId, eqiora_core::DynQuantity>,
}

impl<S: Coefficient> RecoveredInterfaceReactions<S> {
    pub(crate) fn physical_value(&self, key: FieldDof, normalized: S) -> Result<S, Diagnostic> {
        let scale = self.physical_scales.get(&key.field).ok_or_else(|| {
            invalid("physical reaction lacks its exact residual-row normalization")
        })?;
        let value = <S as From<f64>>::from(scale.value()) * normalized;
        if !value.is_finite() {
            return Err(invalid("physical reaction is non-finite"));
        }
        Ok(value)
    }
    #[cfg(test)]
    pub(crate) fn connections(&self) -> BTreeSet<RawId> {
        self.actions
            .keys()
            .map(|(connection, _)| *connection)
            .collect()
    }

    pub(crate) fn action(&self, connection: RawId, key: FieldDof) -> Result<S, Diagnostic> {
        self.actions
            .get(&(connection, key))
            .copied()
            .ok_or_else(|| invalid("reaction request is outside exact Connection/Field support"))
    }
}

impl<S: Coefficient + Send + Sync> InterfaceReactions<S> {
    pub(crate) fn prepare(
        work: &dyn AssemblyWork<S>,
        target: AssemblyTargetId,
        mapping: &RegionDofMap<S>,
        packet_domains: &[RawId],
    ) -> Result<Self, Diagnostic> {
        let known = mapping
            .cell_domains()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if !packet_domains.starts_with(mapping.cell_domains())
            || packet_domains.iter().any(|domain| !known.contains(domain))
        {
            return Err(invalid(
                "reaction packet ownership differs from exact mapped Domains",
            ));
        }
        let projection = InterfaceReactionProjection::new(mapping)?;
        let domains = DomainReactions::prepare(
            work,
            target,
            mapping.full_count(),
            packet_domains,
            mapping.cell_domains().len(),
            &projection.rows,
        )?;
        let volume_loads = mapping
            .keys()
            .map(|key| {
                let domain = mapping.field_layout(key.field).expect("mapped Field").0;
                let row = mapping.global_dof(key).expect("mapped coordinate");
                (key, domains.volume_loads[&domain][row])
            })
            .collect();
        Ok(Self {
            domains,
            projection,
            volume_loads,
        })
    }

    pub(crate) fn recover(&self, full: &[S]) -> Result<RecoveredInterfaceReactions<S>, Diagnostic> {
        let domains = self.domains.recover(full)?;
        let mut recovered = self.projection.recover(&domains)?;
        recovered.volume_loads = self.volume_loads.clone();
        if recovered
            .constrained_actions
            .values()
            .chain(recovered.volume_loads.values())
            .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "recovered constrained action or volume load is non-finite",
            ));
        }
        Ok(recovered)
    }
}
