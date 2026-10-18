use super::*;
use crate::form_compiler::bilinear::BasisComponent;
use eqiora_schema::kernel::{ExprId, ExprNode};

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

impl<S: Coefficient> Context<'_, S> {
    pub(in crate::form_compiler) fn diffusion_orientation(
        &self,
        id: ExprId,
        depth: usize,
    ) -> Result<Option<i8>, Diagnostic> {
        if depth > 128 {
            return Err(super::invalid("linear expression nesting exceeds 128"));
        }
        let orientation = |id| self.diffusion_orientation(id, depth + 1);
        let merge = |left: Option<i8>, right: Option<i8>| match (left, right) {
            (Some(a), Some(b)) if a != b => Err(super::invalid(
                "linear diffusion terms have conflicting additive orientations",
            )),
            (Some(a), _) | (_, Some(a)) => Ok(Some(a)),
            _ => Ok(None),
        };
        match self.dag.node(id) {
            Some(ExprNode::Divergence(_)) => Ok(Some(1)),
            Some(ExprNode::PureOperatorApplication(_))
                if self.dimension == 2
                    && super::super::planar_curl::gradient(self.dag, id).is_some() =>
            {
                Ok(Some(-1))
            }
            Some(ExprNode::Neg(a)) => Ok(orientation(*a)?.map(|sign| -sign)),
            Some(ExprNode::Add(a, b)) => merge(orientation(*a)?, orientation(*b)?),
            Some(ExprNode::Sub(a, b)) => {
                merge(orientation(*a)?, orientation(*b)?.map(|sign| -sign))
            }
            Some(ExprNode::Mul(a, b)) if self.data(*a, depth + 1).is_ok() => orientation(*b),
            Some(ExprNode::Mul(a, b)) if self.data(*b, depth + 1).is_ok() => orientation(*a),
            Some(ExprNode::Div(a, _)) => orientation(*a),
            _ => Ok(None),
        }
    }
}

impl<S: Coefficient> Row<S> {
    pub(super) fn orient_scalar_diffusion(
        &mut self,
        context: &Context<'_, S>,
        root: ExprId,
    ) -> Result<(), Diagnostic> {
        // A scalar multiplier constraint such as div(velocity)=0 has no
        // principal gradient of its own Field and must retain its authored sign.
        if self.value_type.shape().is_scalar()
            && self.terms.iter().any(|term| {
                term.trial == self.tested
                    && matches!(
                        term.pairing,
                        Pairing::Gradient
                            | Pairing::Components {
                                test: BasisComponent::Gradient { .. },
                                trial: BasisComponent::Gradient { .. },
                            }
                    )
            })
            && context.diffusion_orientation(root, 0)? == Some(1)
        {
            self.map_data(&|value| {
                Ok(value.clone().multiply(Data::constant(
                    context.dimension,
                    <S as From<f64>>::from(-1.0),
                )))
            })?;
        }
        Ok(())
    }

    pub(super) fn require_positive_scalar_diffusion(&mut self) {
        if S::DOMAIN == ScalarDomain::Real && self.value_type.shape().is_scalar() {
            for term in self
                .terms
                .iter_mut()
                .chain(self.flux.iter_mut().filter_map(|flux| match flux {
                    flux::FluxTerm::Trial(term) => Some(term),
                    _ => None,
                }))
            {
                if term.trial == self.tested && term.pairing == Pairing::Gradient {
                    term.positive_diffusion = true;
                }
            }
        }
    }
}
