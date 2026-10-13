//! Project exact Domain primal residuals onto trace and constrained Field coordinates.
use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InterfaceReactionProjection {
    size: usize,
    domains: BTreeSet<RawId>,
    projections: BTreeMap<(RawId, FieldDof), (RawId, usize)>,
    pairs: Vec<[(RawId, FieldDof); 2]>,
    constrained: BTreeMap<FieldDof, (RawId, usize)>,
    pub(super) rows: BTreeSet<usize>,
}

impl InterfaceReactionProjection {
    pub(crate) fn new<S: Coefficient + Send + Sync>(
        mapping: &RegionDofMap<S>,
    ) -> Result<Self, Diagnostic> {
        let mut projections = BTreeMap::new();
        let mut pairs = Vec::new();
        let mut owners = BTreeMap::new();
        let mut rows = BTreeSet::new();
        let mut constrained = BTreeMap::new();
        for key in mapping
            .keys()
            .filter(|key| mapping.free_dof(*key).is_none())
        {
            let row = mapping
                .global_dof(key)
                .expect("mapped constrained coordinate");
            let domain = mapping.field_layout(key.field).expect("mapped Field").0;
            rows.insert(row);
            constrained.insert(key, (domain, row));
        }
        for (quotient, keys) in mapping.traces() {
            let connection = quotient.source().owner();
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
            size: mapping.full_count(),
            domains: mapping.cell_domains().iter().copied().collect(),
            projections,
            pairs,
            constrained,
            rows,
        })
    }

    /// Consume the original primal residual, never a tangent-system reconstruction.
    pub(crate) fn recover<S: Coefficient>(
        &self,
        domains: &RecoveredDomainReactions<S>,
    ) -> Result<RecoveredInterfaceReactions<S>, Diagnostic> {
        if domains.values.keys().copied().collect::<BTreeSet<_>>() != self.domains
            || domains.values.values().any(|values| {
                values.len() != self.size || values.iter().any(|value| !value.is_finite())
            })
        {
            return Err(invalid(
                "reaction residuals differ from exact finite Domain/full-row inventories",
            ));
        }
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
            imbalance_norm = imbalance_norm.hypot((actions[first] + actions[second]).abs());
        }
        if !imbalance_norm.is_finite() {
            return Err(invalid("interface reaction imbalance is non-finite"));
        }
        let constrained_actions = self
            .constrained
            .iter()
            .map(|(&key, &(domain, row))| (key, domains.values[&domain][row]))
            .collect();
        Ok(RecoveredInterfaceReactions {
            actions,
            imbalance_norm,
            constrained_actions,
            volume_loads: BTreeMap::new(),
            physical_scales: BTreeMap::new(),
        })
    }
}
