use eqiora_core::{DimExponents, DynQuantity};
use eqiora_realization::Space;

use super::*;

/// Transient admission result, consumed into the sole executable region representation.
pub(in crate::form_compiler) struct ScalarRow<S: Coefficient> {
    pub relation: RawId,
    pub field: RawId,
    pub residual_type: ValueType,
    pub diffusion: Data<S>,
    pub reaction: BTreeMap<RawId, Data<S>>,
    pub storage: BTreeMap<RawId, Data<S>>,
    pub transport: BTreeMap<(RawId, usize), Data<S>>,
    pub forcing: Data<S>,
}

impl<S: Coefficient> CompiledRegionForm<S> {
    pub(in crate::form_compiler) fn scalar(
        domain: RawId,
        dimension: usize,
        roles: EquationRoles,
        rows: Vec<ScalarRow<S>>,
    ) -> Result<Self, Diagnostic> {
        let rows = rows
            .into_iter()
            .map(|row| {
                let mut terms = vec![Term {
                    trial: row.field,
                    derivative: false,
                    pairing: Pairing::Gradient,
                    coefficient: row.diffusion,
                    positive_diffusion: false,
                }];
                terms.extend(
                    row.transport
                        .into_iter()
                        .map(|((trial, axis), coefficient)| Term {
                            trial,
                            derivative: false,
                            pairing: Pairing::TestGradientTrialValue(axis),
                            coefficient,
                            positive_diffusion: false,
                        }),
                );
                let flux = terms
                    .iter()
                    .cloned()
                    .map(super::flux::FluxTerm::Trial)
                    .collect();
                terms.extend(row.reaction.into_iter().map(|(trial, coefficient)| Term {
                    trial,
                    derivative: false,
                    pairing: Pairing::Value,
                    coefficient,
                    positive_diffusion: false,
                }));
                terms.extend(row.storage.into_iter().map(|(trial, coefficient)| Term {
                    trial,
                    derivative: true,
                    pairing: Pairing::Value,
                    coefficient,
                    positive_diffusion: false,
                }));
                let mut row = Row {
                    relation: row.relation,
                    tested: row.field,
                    value_type: row.residual_type,
                    flux,
                    terms,
                    dyadics: Vec::new(),
                    components: Vec::new(),
                    forcing: vec![row.forcing],
                };
                row.require_positive_scalar_diffusion();
                row
            })
            .collect();
        let form = Self {
            domain,
            dimension,
            roles,
            rows,
        };
        Ok(form)
    }

    /// Bind an existing congruence selection as S A S / W. Row orientation
    /// comes from the admitted equations and applies to forcing and history too.
    pub(crate) fn congruence_bindings(
        &self,
        space: impl Fn(RawId) -> Result<Space, Diagnostic>,
        scaling: &eqiora_realization::SymmetricCongruenceScaling,
        orientations: &BTreeMap<RawId, f64>,
    ) -> Result<(Vec<RegionFieldBinding>, BTreeMap<RawId, DynQuantity>), Diagnostic> {
        let fields = self.fields().map(|(field, _)| {
            let scale = scaling.block_scales().iter().find(|scale|
                matches!(scale.block(), eqiora_solver::AlgebraicBlock::Field(id) if id.erase() == field)
            ).ok_or_else(|| invalid("region Field has no exact congruence scale"))?;
            Ok(RegionFieldBinding {
                field,
                space: space(field)?,
                scale: scale.scale().quantity(),
            })
        }).collect::<Result<Vec<_>, Diagnostic>>()?;
        let functional = scaling.weak_functional_scale().quantity();
        let rows = self
            .rows()
            .map(|(relation, tested, _)| {
                let scale = fields
                    .iter()
                    .find(|field| field.field == tested)
                    .ok_or_else(|| invalid("region test Field has no congruence binding"))?
                    .scale
                    .try_div(functional)?;
                let sign = orientations
                    .get(&tested)
                    .copied()
                    .filter(|sign| *sign == 1.0 || *sign == -1.0)
                    .ok_or_else(|| invalid("region row has no exact admitted test orientation"))?;
                Ok((
                    relation,
                    DynQuantity::new(sign * scale.value(), scale.dim()),
                ))
            })
            .collect::<Result<BTreeMap<_, _>, Diagnostic>>()?;
        Ok((fields, rows))
    }

    pub(in crate::form_compiler) fn si_bindings(
        &self,
        space: impl Fn(RawId) -> Result<Space, Diagnostic>,
    ) -> Result<(Vec<RegionFieldBinding>, BTreeMap<RawId, DynQuantity>), Diagnostic> {
        let form = self;
        let dimension = self.dimension;
        let fields = form
            .fields()
            .map(|(field, value_type)| {
                let space = space(field)?;
                Ok(RegionFieldBinding {
                    field,
                    space,
                    scale: DynQuantity::new(
                        1.0,
                        space
                            .coefficient_dimension(value_type.dimension())
                            .ok_or_else(|| invalid("coefficient functional dimension overflow"))?,
                    ),
                })
            })
            .collect::<Result<Vec<_>, Diagnostic>>()?;
        let measure = DimExponents::from_integers([0, dimension as i32, 0, 0, 0, 0, 0])
            .ok_or_else(|| invalid("region measure dimension overflow"))?;
        let orientations = self
            .constraint_pairs()
            .next()
            .and_then(|_| self.volume_test_orientations());
        let multipliers = form
            .rows()
            .map(|(relation, tested, value_type)| {
                let space = space(tested)?;
                let dimension = value_type
                    .dimension()
                    .mul(measure)
                    .and_then(|dim| {
                        dim.div(space.coefficient_dimension(DimExponents::DIMENSIONLESS)?)
                    })
                    .and_then(|dim| dim.pow(-1, 1))
                    .ok_or_else(|| invalid("region row normalization dimension overflow"))?;
                let sign = orientations.as_ref().map_or(1.0, |signs| signs[&tested]);
                Ok((relation, DynQuantity::new(sign, dimension)))
            })
            .collect::<Result<BTreeMap<_, _>, Diagnostic>>()?;
        Ok((fields, multipliers))
    }
}

impl<S: Coefficient> CompiledRegionForm<S> {
    pub(in crate::form_compiler) fn bind_inputs<'a>(
        &self,
        point: impl Into<crate::form_compiler::spatial::InputPoint<'a>>,
    ) -> Result<Self, Diagnostic> {
        let point = point.into();
        let mut bound = self.map_data(&|value| value.bind_inputs(point))?;
        for row in &mut bound.rows {
            for term in &mut row.components {
                term.value = std::sync::Arc::new(term.value.bind_inputs(point)?);
            }
            for flux in &mut row.flux {
                if let super::flux::FluxTerm::Component(term) = flux {
                    term.value = std::sync::Arc::new(term.value.bind_inputs(point)?);
                }
            }
        }
        Ok(bound)
    }

    pub(super) fn map_data(
        &self,
        map: &impl Fn(&Data<S>) -> Result<Data<S>, Diagnostic>,
    ) -> Result<Self, Diagnostic> {
        let mut bound = self.clone();
        for row in &mut bound.rows {
            row.map_data(map)?;
        }
        Ok(bound)
    }

    /// Prescribed-input partial at fixed physical Field values and geometry.
    /// Parameter and model-Time coordinates use the same coefficient product.
    pub(in crate::form_compiler) fn input_partial(
        &self,
        coordinate: &eqiora_ir::ScalarSymbolCoordinate,
    ) -> Result<Self, Diagnostic> {
        if self
            .rows
            .iter()
            .flat_map(|row| &row.components)
            .any(|term| term.input_partial.is_some())
        {
            return Err(invalid(
                "repeated component input partials require a higher-order input product",
            ));
        }
        let mut partial = self.map_data(&|value| {
            value.derivative(&|input| f64::from(input == coordinate), self.dimension)
        })?;
        // d(c*f)/dp = c_p*f + c*f_p. Retain both original operands, including
        // their domains, even when a sampled factor or tangent vanishes.
        for (row, source) in partial.rows.iter_mut().zip(&self.rows) {
            row.components.extend(source.components.iter().map(|term| {
                let mut partial = term.clone();
                partial.input_partial = Some(coordinate.clone());
                partial
            }));
        }
        // Positivity belongs to the already assembled primal constitutive law.
        // Its derivative can have either sign, including zero.
        for row in &mut partial.rows {
            for term in &mut row.terms {
                term.positive_diffusion = false;
            }
            for flux in &mut row.flux {
                if let super::flux::FluxTerm::Trial(term) = flux {
                    term.positive_diffusion = false;
                }
            }
        }
        Ok(partial)
    }
}

impl<S: Coefficient> BoundRegionForm<S> {
    pub(crate) fn input_partial(
        &self,
        coordinate: &eqiora_ir::ScalarSymbolCoordinate,
    ) -> Result<Self, Diagnostic> {
        let mut partial = self.clone();
        partial.form = self.form.input_partial(coordinate)?;
        partial.conservative_flux = self
            .conservative_flux
            .as_ref()
            .map(|flux| flux.input_partial(coordinate).map(Box::new))
            .transpose()?;
        Ok(partial)
    }
}

impl<S: Coefficient> Row<S> {
    pub(super) fn map_data(
        &mut self,
        map: &impl Fn(&Data<S>) -> Result<Data<S>, Diagnostic>,
    ) -> Result<(), Diagnostic> {
        for flux in &mut self.flux {
            flux.map_data(map)?;
        }
        for component in &mut self.forcing {
            *component = map(component)?;
        }
        for term in &mut self.dyadics {
            term.coefficient = map(&term.coefficient)?;
        }
        for term in &mut self.components {
            term.coefficient = map(&term.coefficient)?;
        }
        for term in &mut self.terms {
            term.coefficient = map(&term.coefficient)?;
        }
        Ok(())
    }
}
