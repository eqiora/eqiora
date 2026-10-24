//! Source rate coordinates and sufficient constant-capacity certificates.
use super::*;
use eqiora_core::TimeStateCoordinate;
use std::collections::HashSet;

/// Derived from the retained equations, never a second coefficient inventory.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TemporalStructure {
    pub(crate) differential: HashSet<TimeStateCoordinate>,
    pub(crate) positive_own_capacity: bool,
}

impl<S: Coefficient> CompiledRegionForm<S> {
    pub(in crate::form_compiler) fn temporal_structure(
        &self,
    ) -> Result<TemporalStructure, Diagnostic> {
        let mut structure = TemporalStructure {
            differential: HashSet::new(),
            positive_own_capacity: true,
        };
        let kinematic = |field| {
            self.roles
                .relations
                .values()
                .any(|role| matches!(role.kind, Role::Kinematic { state, .. } if state == field))
        };
        for row in &self.rows {
            for term in &row.components {
                let rates = term.value.rate_coordinates()?;
                if !rates.is_empty() {
                    structure.positive_own_capacity = false;
                }
                structure.differential.extend(rates);
            }
            let mut capacity: Option<Data<S>> = None;
            for term in row
                .terms
                .iter()
                .filter(|term| term.derivative && !kinematic(term.trial))
            {
                let ty = &self.roles.fields[&term.trial].1;
                let components = match term.pairing {
                    Pairing::Components { trial, .. } => {
                        let (super::super::bilinear::BasisComponent::Value(component)
                        | super::super::bilinear::BasisComponent::Gradient {
                            component, ..
                        }) = trial;
                        component..component + 1
                    }
                    _ => 0..components(ty, self.dimension)?,
                };
                let images = term.coefficient.coordinate_images(self.dimension)?;
                for component in components {
                    for (part, (_, active)) in images
                        .iter()
                        .enumerate()
                        .take(1 + usize::from(ty.scalar_domain() == ScalarDomain::Complex))
                    {
                        if !active {
                            continue;
                        }
                        let imaginary = part != 0;
                        structure.differential.insert(TimeStateCoordinate::new(
                            term.trial.downcast().expect("rate Field"),
                            0,
                            component,
                            imaginary,
                        ));
                    }
                }
                if term.trial != row.tested || term.pairing != Pairing::Value {
                    structure.positive_own_capacity = false;
                }
                capacity = Some(match capacity {
                    Some(value) => value.add(term.coefficient.clone()),
                    None => term.coefficient.clone(),
                });
            }
            if let Some(capacity) = capacity {
                // A spatially uniform coefficient may still depend on Time.
                // This certificate is sufficient for constant capacity, so it
                // must not evaluate an unbound clock or certify one time sample.
                if capacity.spatial()
                    || !capacity.constant_in_direction(&|input| {
                        f64::from(input.symbol() == eqiora_schema::kernel::SymbolRef::Time)
                    })?
                {
                    structure.positive_own_capacity = false;
                } else {
                    let images = capacity.coordinate_images(self.dimension)?;
                    let value = images[0].0.evaluate(&vec![0.; self.dimension])?;
                    let mut positive = value.is_finite() && value.im() == 0. && value.re() > 0.;
                    if row.value_type.scalar_domain() == ScalarDomain::Complex {
                        let imaginary = images[1].0.evaluate(&vec![0.; self.dimension])?;
                        positive &= imaginary.re() == 0. && imaginary.im() == value.re();
                    }
                    structure.positive_own_capacity &= positive;
                }
            }
        }
        Ok(structure)
    }
}

impl<S: Coefficient> BoundRegionForm<S> {
    /// Derive the snapshot certificate from its current prescribed input values.
    pub(crate) fn temporal_structure(&self) -> Result<TemporalStructure, Diagnostic> {
        self.form.temporal_structure()
    }
}
