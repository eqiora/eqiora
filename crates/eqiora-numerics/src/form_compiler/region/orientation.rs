use super::*;

impl<S: Coefficient> CompiledRegionForm<S> {
    pub(in crate::form_compiler) fn has_zero_multiplier_block(&self) -> bool {
        let multipliers = self
            .constraint_pairs()
            .map(|(_, field)| field)
            .collect::<BTreeSet<_>>();
        let zero = Data::constant(self.dimension, <S as From<f64>>::from(0.0));
        !multipliers.is_empty()
            && self
                .rows
                .iter()
                .filter(|row| multipliers.contains(&row.tested))
                .all(|row| {
                    row.dyadics.is_empty()
                        && row
                            .terms
                            .iter()
                            .filter(|term| multipliers.contains(&term.trial))
                            .all(|term| term.coefficient.same_coefficient(&zero))
                })
    }

    /// Orient paired real terms after exact backward-Euler state elimination.
    /// Matching timestep powers proves symmetry for every positive step; boundary
    /// terms and global constraints still require their own admission.
    pub(in crate::form_compiler) fn volume_test_orientations(
        &self,
    ) -> Option<BTreeMap<RawId, f64>> {
        if S::DOMAIN != ScalarDomain::Real || self.rows.iter().any(|row| !row.dyadics.is_empty()) {
            return None;
        }
        let zero = Data::constant(self.dimension, <S as From<f64>>::from(0.0));
        let negative = Data::constant(self.dimension, <S as From<f64>>::from(-1.0));
        let rates = self
            .roles
            .relations
            .values()
            .filter_map(|role| match role.kind {
                Role::Kinematic { state, rate } => Some((state, rate)),
                _ => None,
            })
            .collect::<BTreeMap<_, _>>();
        let effective = |term: &Term<S>| {
            if let Some(&rate) = rates.get(&term.trial) {
                (rate, if term.derivative { 0_i8 } else { 1 })
            } else {
                (term.trial, if term.derivative { -1_i8 } else { 0 })
            }
        };
        let mut edges = Vec::new();
        for row in &self.rows {
            for term in &row.terms {
                if term.coefficient.same_coefficient(&zero) {
                    continue;
                }
                let transpose = term.pairing.transpose()?;
                let (trial, time_power) = effective(term);
                let other = self.rows.iter().find(|other| other.tested == trial)?;
                let matching = other
                    .terms
                    .iter()
                    .filter(|candidate| {
                        effective(candidate) == (row.tested, time_power)
                            && candidate.pairing == transpose
                    })
                    .collect::<Vec<_>>();
                // Repeated terms require coefficient aggregation before a proof;
                // declining here preserves the original residual orientation.
                if matching.len() != 1
                    || row
                        .terms
                        .iter()
                        .filter(|candidate| {
                            effective(candidate) == (trial, time_power)
                                && candidate.pairing == term.pairing
                        })
                        .count()
                        != 1
                {
                    return None;
                }
                let coefficient = &matching[0].coefficient;
                let relative = if term.coefficient.same_coefficient(coefficient) {
                    1.0
                } else if term
                    .coefficient
                    .same_coefficient(&coefficient.clone().multiply(negative.clone()))
                {
                    -1.0
                } else {
                    return None;
                };
                edges.push((row.tested, trial, relative));
            }
        }
        let multipliers = self
            .constraint_pairs()
            .map(|(_, multiplier)| multiplier)
            .collect::<BTreeSet<_>>();
        let mut signs = BTreeMap::new();
        // Prefer a primal row as each connected component's positive anchor.
        for row in self
            .rows
            .iter()
            .filter(|row| !multipliers.contains(&row.tested))
            .chain(
                self.rows
                    .iter()
                    .filter(|row| multipliers.contains(&row.tested)),
            )
        {
            if signs.contains_key(&row.tested) {
                continue;
            }
            signs.insert(row.tested, 1.0);
            let mut pending = vec![row.tested];
            while let Some(field) = pending.pop() {
                for &(a, b, relative) in &edges {
                    let other = if a == field {
                        b
                    } else if b == field {
                        a
                    } else {
                        continue;
                    };
                    let expected = signs[&field] * relative;
                    if let Some(&actual) = signs.get(&other) {
                        if actual != expected {
                            return None;
                        }
                    } else {
                        signs.insert(other, expected);
                        pending.push(other);
                    }
                }
            }
        }
        Some(signs)
    }
}
