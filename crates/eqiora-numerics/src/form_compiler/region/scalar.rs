use eqiora_core::{DimExponents, DynQuantity, Id, entity::kinds};
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
                    positive_diffusion: S::DOMAIN == ScalarDomain::Real,
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
                Row {
                    relation: row.relation,
                    tested: row.field,
                    value_type: row.residual_type,
                    flux,
                    terms,
                    dyadics: Vec::new(),
                    forcing: vec![row.forcing],
                }
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
    pub(in crate::form_compiler) fn bind_parameter_point(
        &self,
        fields: &[Id<kinds::Parameter>],
        values: &[S],
    ) -> Result<Self, Diagnostic> {
        let mut bound = self.clone();
        for row in &mut bound.rows {
            for flux in &mut row.flux {
                flux.bind_parameter_point(fields, values)?;
            }
            for component in &mut row.forcing {
                *component = component.bind_parameter_point(fields, values)?;
            }
            for term in &mut row.dyadics {
                term.coefficient = term.coefficient.bind_parameter_point(fields, values)?;
            }
            for term in &mut row.terms {
                term.coefficient = term.coefficient.bind_parameter_point(fields, values)?;
            }
        }
        Ok(bound)
    }
}
