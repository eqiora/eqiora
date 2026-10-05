//! Exact constitutive flux retained separately from its volume divergence.

use eqiora_core::{Id, entity::kinds};
use eqiora_schema::kernel::{ExprId, ExprNode};

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum FluxTerm {
    Trial(Term),
    // A uniform isotropic tensor has zero volume divergence, but its outward
    // traction is its scalar coefficient times the exact parent normal.
    Isotropic(Data<f64>),
}

impl FluxTerm {
    fn same_coefficient(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Trial(a), Self::Trial(b)) => {
                a.trial == b.trial
                    && a.derivative == b.derivative
                    && a.pairing == b.pairing
                    && a.coefficient.same_coefficient(&b.coefficient)
            }
            (Self::Isotropic(a), Self::Isotropic(b)) => a.same_coefficient(b),
            _ => false,
        }
    }

    pub(super) fn bind_parameter_point(
        &mut self,
        fields: &[Id<kinds::Parameter>],
        values: &[f64],
    ) -> Result<(), Diagnostic> {
        let coefficient = match self {
            Self::Trial(term) => &mut term.coefficient,
            Self::Isotropic(coefficient) => coefficient,
        };
        *coefficient = coefficient.bind_parameter_point(fields, values)?;
        Ok(())
    }
}

impl CompiledRegionForm {
    pub(super) fn require_boundary_flux(
        &self,
        program: &KernelProgram,
        boundary: RawId,
        relation: RawId,
        field: RawId,
        normal: ExprId,
    ) -> Result<(), Diagnostic> {
        if crate::canonical::boundary_parent(program, boundary) != Some(self.domain)
            || !crate::canonical::relations_on(program, boundary).contains(&relation)
        {
            return Err(invalid(
                "flux witness has a foreign Boundary or Relation support",
            ));
        }
        let row = self
            .rows
            .iter()
            .find(|row| row.tested == field)
            .ok_or_else(|| invalid("flux witness has a foreign tested Field"))?;
        if row.flux.is_empty() {
            return Err(invalid("tested row has no constitutive boundary flux"));
        }
        let typed = typed_relation(program, relation)?;
        let Some(ExprNode::NormalComponent(flux)) = typed.expression().node(normal) else {
            return Err(invalid(
                "flux witness requires an exact normal-component operator",
            ));
        };
        let coefficients =
            super::super::linear::coefficients(program, self.dimension, &self.roles)?;
        let context = Context {
            program,
            dag: typed.expression(),
            owner: relation,
            dimension: self.dimension,
            coefficients: &coefficients,
        };
        let mut candidate = super::lowering::boundary_flux(&context, *flux, row)?;
        if candidate.len() != row.flux.len() {
            return Err(invalid(
                "boundary flux differs from the complete volume constitutive flux",
            ));
        }
        for expected in &row.flux {
            let Some(index) = candidate
                .iter()
                .position(|term| expected.same_coefficient(term))
            else {
                return Err(invalid(
                    "boundary flux differs from the exact Field/operator/coefficient identity",
                ));
            };
            candidate.remove(index);
        }
        Ok(())
    }
}
