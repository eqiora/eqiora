//! Borrowed conservative blocks from the existing compiled equation rows.
use super::coordinates::{Native, Projection};
use super::*;
use crate::form_compiler::bilinear::BasisComponent;
use num_complex::ComplexFloat;
use std::marker::PhantomData;

pub(crate) struct ConservativeSystem<'a, S: Coefficient, P: Projection<S> = Native> {
    rows: BTreeMap<RawId, &'a Row<S>>,
    coordinates: Vec<(RawId, usize, usize)>,
    dimension: usize,
    projection: PhantomData<fn() -> P>,
}

impl<S: Coefficient> CompiledRegionForm<S> {
    pub(in crate::form_compiler) fn conservative_system(
        &self,
    ) -> Result<ConservativeSystem<'_, S>, Diagnostic> {
        let rows = self
            .rows
            .iter()
            .map(|row| (row.tested, row))
            .collect::<BTreeMap<_, _>>();
        if rows.is_empty() || rows.len() != self.rows.len() {
            return Err(invalid(
                "conservative assembly requires distinct tested Fields",
            ));
        }
        for row in rows.values() {
            let field_type = &self.roles.fields[&row.tested].1;
            if field_type.scalar_domain() != S::DOMAIN
                || !same_component_type(field_type, &row.value_type)
                || row.forcing.len() != components(field_type, self.dimension)?
                || row.is_nonlinear()
                || row.terms.iter().any(|term| {
                    let Some(trial) = rows.get(&term.trial) else {
                        return true;
                    };
                    term.derivative
                        || match term.pairing {
                            Pairing::Value | Pairing::Gradient => {
                                row.forcing.len() != trial.forcing.len()
                            }
                            Pairing::Components {
                                test,
                                trial: component,
                            } => match (test, component) {
                                (BasisComponent::Value(test), BasisComponent::Value(component)) => {
                                    test >= row.forcing.len() || component >= trial.forcing.len()
                                }
                                (
                                    BasisComponent::Gradient {
                                        component: test,
                                        axis,
                                    },
                                    BasisComponent::Gradient {
                                        component,
                                        axis: trial_axis,
                                    },
                                ) => {
                                    test >= row.forcing.len()
                                        || component >= trial.forcing.len()
                                        || axis >= self.dimension
                                        || trial_axis != axis
                                }
                                _ => true,
                            },
                            _ => true,
                        }
                })
            {
                return Err(invalid(
                    "conservative assembly requires steady axis-aligned-diffusion/component-reaction blocks in their exact coefficient domain",
                ));
            }
            let gradients = row
                .terms
                .iter()
                .filter(|term| gradient_pairing(term.pairing));
            if !gradients.eq(row.flux.iter().filter_map(|flux| match flux {
                flux::FluxTerm::Trial(term) => Some(term),
                flux::FluxTerm::Component(_)
                | flux::FluxTerm::Isotropic(_)
                | flux::FluxTerm::Prescribed { .. } => None,
            })) || row.flux.iter().any(|flux| {
                matches!(flux, flux::FluxTerm::Trial(term) if !gradient_pairing(term.pairing))
            }) {
                return Err(invalid("conservative assembly requires its complete axis-aligned gradient flux"));
            }
            for flux in &row.flux {
                flux.prescribed_components(self.dimension, row.forcing.len())?;
            }
        }
        ConservativeSystem::new(rows, self.dimension)
    }
}

impl<'a, S: Coefficient, P: Projection<S>> ConservativeSystem<'a, S, P> {
    fn new(rows: BTreeMap<RawId, &'a Row<S>>, dimension: usize) -> Result<Self, Diagnostic> {
        let mut coordinates = Vec::new();
        for (&field, row) in &rows {
            let width = P::width(&row.value_type)?;
            coordinates.extend(
                (0..row.forcing.len())
                    .flat_map(|component| (0..width).map(move |part| (field, component, part))),
            );
        }
        Ok(Self {
            rows,
            coordinates,
            dimension,
            projection: PhantomData,
        })
    }

    pub(crate) fn project<Q: Projection<S>>(
        self,
    ) -> Result<ConservativeSystem<'a, S, Q>, Diagnostic> {
        ConservativeSystem::new(self.rows, self.dimension)
    }

    pub(crate) fn fields(&self) -> impl Iterator<Item = (RawId, usize)> + '_ {
        self.rows
            .iter()
            .map(|(field, row)| (*field, row.forcing.len()))
    }

    pub(crate) fn coordinate(&self, component: usize) -> Result<(RawId, usize, usize), Diagnostic> {
        self.coordinates
            .get(component)
            .copied()
            .ok_or_else(|| invalid("conservative component is outside its exact Field layout"))
    }

    pub(crate) fn components(&self) -> usize {
        self.coordinates.len()
    }

    pub(crate) fn has_flux(&self) -> bool {
        self.rows.values().any(|row| !row.flux.is_empty())
    }

    pub(crate) fn has_reaction(&self) -> bool {
        self.rows.values().any(|row| {
            row.terms.iter().any(|term| {
                matches!(
                    term.pairing,
                    Pairing::Value
                        | Pairing::Components {
                            test: BasisComponent::Value(_),
                            trial: BasisComponent::Value(_)
                        }
                )
            })
        })
    }

    /// A diagonal positive diffusion proof; coupled blocks retain General properties.
    pub(crate) fn has_positive_diffusion(&self) -> bool {
        self.rows.values().all(|row| {
            let gradients = row
                .terms
                .iter()
                .filter(|term| gradient_pairing(term.pairing))
                .collect::<Vec<_>>();
            !gradients.is_empty()
                && gradients.iter().all(|term| {
                    term.pairing == Pairing::Gradient
                        && term.trial == row.tested
                        && term.positive_diffusion
                })
        })
    }

    pub(crate) fn diffusion(
        &self,
        test: usize,
        trial: usize,
        axis: usize,
        point: &[f64],
    ) -> Result<P::Scalar, Diagnostic> {
        if axis >= self.dimension {
            return Err(invalid("conservative normal axis is out of bounds"));
        }
        self.coefficient(
            test,
            trial,
            |pairing, test, trial| {
                (test == trial && pairing == Pairing::Gradient)
                    || pairing
                        == Pairing::Components {
                            test: BasisComponent::Gradient {
                                component: test,
                                axis,
                            },
                            trial: BasisComponent::Gradient {
                                component: trial,
                                axis,
                            },
                        }
            },
            point,
        )
    }

    pub(crate) fn prescribed_flux(
        &self,
        test: usize,
        axis: usize,
        point: &[f64],
    ) -> Result<P::Scalar, Diagnostic> {
        if axis >= self.dimension {
            return Err(invalid("conservative normal axis is out of bounds"));
        }
        let (field, component, part) = self.coordinate(test)?;
        let row = self.rows[&field];
        let mut value = <P::Scalar as From<f64>>::from(0.0);
        for flux in &row.flux {
            if let Some((data, mut coordinates)) =
                flux.prescribed_components(self.dimension, row.forcing.len())?
                && coordinates.any(|coordinate| coordinate == component * self.dimension + axis)
            {
                value += P::component(data.evaluate(point)?, part);
            }
        }
        if !value.is_finite() {
            return Err(invalid("conservative prescribed flux sum is non-finite"));
        }
        Ok(value)
    }

    pub(crate) fn reaction(
        &self,
        test: usize,
        trial: usize,
        point: &[f64],
    ) -> Result<P::Scalar, Diagnostic> {
        self.coefficient(
            test,
            trial,
            |pairing, test, trial| {
                (test == trial && pairing == Pairing::Value)
                    || pairing
                        == Pairing::Components {
                            test: BasisComponent::Value(test),
                            trial: BasisComponent::Value(trial),
                        }
            },
            point,
        )
    }

    pub(crate) fn source(&self, component: usize, point: &[f64]) -> Result<P::Scalar, Diagnostic> {
        let (field, component, part) = self.coordinate(component)?;
        self.rows[&field].forcing[component]
            .evaluate(point)
            .map(|value| P::component(value, part))
    }

    fn coefficient(
        &self,
        test: usize,
        trial: usize,
        select: impl Fn(Pairing, usize, usize) -> bool,
        point: &[f64],
    ) -> Result<P::Scalar, Diagnostic> {
        let (field, test, test_part) = self.coordinate(test)?;
        let (trial_field, trial, trial_part) = self.coordinate(trial)?;
        let width = P::width(&self.rows[&trial_field].value_type)?;
        let mut value = <P::Scalar as From<f64>>::from(0.0);
        for term in self.rows[&field]
            .terms
            .iter()
            .filter(|term| term.trial == trial_field && select(term.pairing, test, trial))
        {
            let columns = P::columns(&term.coefficient, self.dimension, width)?;
            let coefficient = columns[trial_part].evaluate(point)?;
            P::validate(term, coefficient, point)?;
            value += P::component(coefficient, test_part);
        }
        if !value.is_finite() {
            return Err(invalid("conservative coefficient sum is non-finite"));
        }
        Ok(value)
    }
}

fn gradient_pairing(pairing: Pairing) -> bool {
    matches!(
        pairing,
        Pairing::Gradient
            | Pairing::Components {
                test: BasisComponent::Gradient { .. },
                trial: BasisComponent::Gradient { .. },
            }
    )
}
