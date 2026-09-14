use eqiora_core::{DimExponents, DynQuantity, Id, entity::kinds};
use eqiora_meshing::ReferenceCell;
use eqiora_realization::Space;

use super::*;

/// Transient admission result, consumed into the sole executable region representation.
pub(in crate::form_compiler) struct ScalarRow {
    pub relation: RawId,
    pub field: RawId,
    pub residual_type: ValueType,
    pub diffusion: Data,
    pub reaction: BTreeMap<RawId, Data>,
    pub storage: BTreeMap<RawId, Data>,
    pub forcing: Data,
}

impl CompiledRegionForm {
    pub(in crate::form_compiler) fn scalar(
        domain: RawId,
        dimension: usize,
        roles: EquationRoles,
        rows: Vec<ScalarRow>,
    ) -> Result<Self, Diagnostic> {
        let rows = rows
            .into_iter()
            .map(|row| {
                let mut terms = vec![Term {
                    trial: row.field,
                    derivative: false,
                    pairing: Pairing::Gradient,
                    coefficient: row.diffusion,
                    positive_diffusion: true,
                }];
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
                    flux: vec![super::flux::FluxTerm::Trial(terms[0].clone())],
                    terms,
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

    pub(in crate::form_compiler) fn bind_scalar(
        &self,
        time: Option<&RegionTimeBinding>,
    ) -> Result<BoundRegionForm, Diagnostic> {
        let form = self;
        let dimension = self.dimension;
        let fields = form
            .fields()
            .map(|(field, value_type)| RegionFieldBinding {
                field,
                space: Space::continuous_lagrange(std::num::NonZeroU16::MIN),
                scale: DynQuantity::new(1.0, value_type.dimension()),
            })
            .collect::<Vec<_>>();
        let measure = DimExponents::from_integers([0, dimension as i32, 0, 0, 0, 0, 0])
            .ok_or_else(|| invalid("scalar measure dimension overflow"))?;
        let multipliers = form
            .rows()
            .map(|(relation, _, value_type)| {
                let dimension = value_type
                    .dimension()
                    .mul(measure)
                    .and_then(|dim| dim.pow(-1, 1))
                    .ok_or_else(|| invalid("scalar row normalization dimension overflow"))?;
                Ok((relation, DynQuantity::new(1.0, dimension)))
            })
            .collect::<Result<BTreeMap<_, _>, Diagnostic>>()?;
        form.bind(
            ReferenceCell::hypercube(dimension)?,
            &fields,
            &multipliers,
            time,
        )
    }
}

impl CompiledRegionForm {
    pub(in crate::form_compiler) fn bind_parameter_point(
        &self,
        fields: &[Id<kinds::Parameter>],
        values: &[f64],
    ) -> Result<Self, Diagnostic> {
        let mut bound = self.clone();
        for row in &mut bound.rows {
            for flux in &mut row.flux {
                flux.bind_parameter_point(fields, values)?;
            }
            for component in &mut row.forcing {
                *component = component.bind_parameter_point(fields, values)?;
            }
            for term in &mut row.terms {
                term.coefficient = term.coefficient.bind_parameter_point(fields, values)?;
            }
        }
        Ok(bound)
    }
}
