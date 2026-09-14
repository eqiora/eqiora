use super::*;
use crate::region_assembly::mapping::{FieldDof, RegionDofMap};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InterfaceReactions {
    domains: DomainReactions,
    projections: BTreeMap<(RawId, FieldDof), (RawId, usize)>,
    pairs: Vec<[(RawId, FieldDof); 2]>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RecoveredInterfaceReactions {
    actions: BTreeMap<(RawId, FieldDof), f64>,
    pub(crate) imbalance_norm: f64,
}

impl RecoveredInterfaceReactions {
    #[cfg(test)]
    pub(crate) fn connections(&self) -> BTreeSet<RawId> {
        self.actions
            .keys()
            .map(|(connection, _)| *connection)
            .collect()
    }

    pub(crate) fn action(&self, connection: RawId, key: FieldDof) -> Result<f64, Diagnostic> {
        self.actions
            .get(&(connection, key))
            .copied()
            .ok_or_else(|| invalid("reaction request is outside exact Connection/Field support"))
    }
}

impl InterfaceReactions {
    pub(crate) fn prepare(
        work: &dyn AssemblyWork,
        target: AssemblyTargetId,
        mapping: &RegionDofMap,
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
        let mut projections = BTreeMap::new();
        let mut pairs = Vec::new();
        let mut owners = BTreeMap::new();
        let mut rows = BTreeSet::new();
        for (quotient, keys) in mapping.traces() {
            let connection = quotient.connection().erase();
            let endpoints = quotient.endpoints();
            for key in keys
                .iter()
                .filter(|key| key.field == endpoints[0].field().erase())
            {
                if mapping.free_dof(*key).is_none() {
                    continue;
                }
                let other = FieldDof {
                    field: endpoints[1].field().erase(),
                    ..*key
                };
                if !keys.contains(&other) {
                    return Err(invalid(
                        "reaction trace omits its paired exact Field coordinate",
                    ));
                }
                let row = mapping.global_dof(*key).expect("mapped trace coordinate");
                if owners
                    .insert(row, connection)
                    .is_some_and(|owner| owner != connection)
                {
                    return Err(invalid(
                        "reaction recovery at a shared Connection junction requires an exact facet dual",
                    ));
                }
                let pair = [(connection, *key), (connection, other)];
                for (selected, endpoint) in pair.into_iter().zip(endpoints) {
                    if mapping.global_dof(selected.1) != Some(row)
                        || projections
                            .insert(selected, (endpoint.domain().erase(), row))
                            .is_some()
                    {
                        return Err(invalid(
                            "reaction trace repeats or misbinds an exact quotient coordinate",
                        ));
                    }
                }
                rows.insert(row);
                pairs.push(pair);
            }
        }
        Ok(Self {
            domains: DomainReactions::prepare(
                work,
                target,
                mapping.full_count(),
                packet_domains,
                &rows,
            )?,
            projections,
            pairs,
        })
    }

    pub(crate) fn recover(&self, full: &[f64]) -> Result<RecoveredInterfaceReactions, Diagnostic> {
        let domains = self.domains.recover(full)?;
        let actions = self
            .projections
            .iter()
            .map(|(&key, &(domain, row))| {
                domains
                    .values
                    .get(&domain)
                    .and_then(|values| values.get(row))
                    .copied()
                    .map(|value| (key, value))
                    .ok_or_else(|| {
                        invalid("reaction recovery omits an exact Domain/Connection owner")
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let mut imbalance_norm = 0.0_f64;
        for [first, second] in &self.pairs {
            imbalance_norm = imbalance_norm.hypot(actions[first] + actions[second]);
        }
        if !imbalance_norm.is_finite() {
            return Err(invalid("interface reaction imbalance is non-finite"));
        }
        Ok(RecoveredInterfaceReactions {
            actions,
            imbalance_norm,
        })
    }
}
