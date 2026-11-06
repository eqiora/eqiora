//! Affine scalar conormal laws reuse the conservation equation decomposition.
use super::*;
use crate::form_compiler::region::flux::FluxTerm;
use eqiora_schema::kernel::typing::TypedResidual;

pub(super) fn validate_coefficient<S: Coefficient>(
    coefficient: &Data<S>,
    dimension: usize,
) -> Result<(), Diagnostic> {
    if coefficient.spatial() {
        return Err(invalid(
            "Robin trace coefficient must be spatially constant",
        ));
    }
    let value = coefficient.evaluate(&vec![0.; dimension])?;
    if !value.is_finite() || value.im() != 0. || value.re() < 0. {
        return Err(invalid(
            "Robin trace coefficient must be finite, real and nonnegative",
        ));
    }
    Ok(())
}

pub(super) fn validate_diffusion<S: Coefficient>(
    coefficient: &Data<S>,
    dimension: usize,
) -> Result<(), Diagnostic> {
    validate_coefficient(coefficient, dimension)?;
    if coefficient.evaluate(&vec![0.; dimension])?.re() <= 0. {
        return Err(invalid("Robin diffusion coefficient must be positive"));
    }
    Ok(())
}

impl<S: Coefficient> RegionBoundaryLaw<S> {
    pub(crate) fn facet_datum_degree(
        &self,
        form: &BoundRegionForm<S>,
        field: RawId,
    ) -> Option<usize> {
        // natural_trace_degree adds the test degree; Robin also needs its trial trace.
        if self.robin_coefficient.is_some() {
            Some(
                self.polynomial_degree()
                    .unwrap_or(0)
                    .max(form.field_basis_degree(field)),
            )
        } else {
            self.polynomial_degree()
        }
    }

    pub(crate) fn robin_coefficient(&self) -> Option<&Data<S>> {
        self.robin_coefficient.as_ref()
    }
}

impl<S: Coefficient> CompiledRegionForm<S> {
    pub(super) fn robin_boundary_law(
        &self,
        context: &Context<'_, S>,
        typed: &TypedResidual<RawId>,
        view: &AdditiveResidualView,
        boundary: RawId,
        relation: RawId,
        dependencies: &BTreeSet<RawId>,
    ) -> Result<Option<RegionBoundaryLaw<S>>, Diagnostic> {
        for row in &self.rows {
            let Some(terms) = crate::scalar_conservation::affine_robin_terms(
                typed.expression(),
                view,
                row.tested,
                false,
            )?
            else {
                continue;
            };
            if !matches!(typed.expression().node(terms.trace),
                Some(ExprNode::Trace {on, ..}) if on.erase() == boundary)
            {
                return Err(invalid("Robin trial trace has a foreign exact support"));
            }
            let field_type = &self.roles.fields[&row.tested].1;
            if !field_type.shape().is_scalar() || field_type.scalar_domain() != ScalarDomain::Real {
                return Err(invalid("Robin requires one real scalar Field"));
            }
            let [FluxTerm::Trial(diffusion)] = row.flux.as_slice() else {
                return Err(invalid(
                    "Robin requires one isotropic scalar diffusion flux",
                ));
            };
            if diffusion.trial != row.tested
                || diffusion.derivative
                || diffusion.pairing != Pairing::Gradient
                || diffusion.coefficient.spatial()
            {
                return Err(invalid(
                    "Robin requires the same Field and constant diffusion coefficient",
                ));
            }
            let k = diffusion.coefficient.evaluate(&vec![0.; self.dimension])?;
            if !k.is_finite() || k.im() != 0. || k.re() <= 0. {
                return Err(invalid(
                    "Robin diffusion coefficient must be finite, real and positive",
                ));
            }
            self.require_boundary_flux(
                context.program,
                BoundaryRelationBinding::new(boundary, relation),
                row.tested,
                terms.normal,
                false,
                context.time_s,
            )?;
            let coefficient = context.data(terms.coefficient, 0)?;
            validate_coefficient(&coefficient, self.dimension)?;
            let mut datum = PrescribedDatum::derive(context, typed, &row.value_type, terms.datum)?;
            if terms.datum_negative {
                datum = PrescribedDatum::Neg(Box::new(datum));
            }
            return Ok(Some(RegionBoundaryLaw {
                binding: BoundaryRelationBinding::new(boundary, relation),
                tested: row.tested,
                trace_field: None,
                quantity: PhysicalBoundaryQuantity::Flux,
                dependencies: dependencies.clone(),
                operator: terms.normal,
                datum_expression: terms.datum,
                datum_negative: terms.datum_negative,
                datum,
                robin_coefficient_expression: Some(terms.coefficient),
                robin_trace_expression: Some(terms.trace),
                robin_primal_coefficient: Some(coefficient.clone()),
                robin_diffusion: Some(diffusion.coefficient.clone()),
                robin_coefficient: Some(coefficient),
                dimension: self.dimension,
            }));
        }
        Ok(None)
    }
}
