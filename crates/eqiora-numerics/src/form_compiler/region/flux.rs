//! Exact constitutive flux retained separately from its volume divergence.

use eqiora_schema::kernel::{DomainKind, ExprId, ExprNode, KernelNode};

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum FluxTerm<S: Coefficient> {
    Trial(Term<S>),
    // A uniform isotropic tensor has zero volume divergence, but its outward
    // traction is its scalar coefficient times the exact parent normal.
    Isotropic(Data<S>),
    Prescribed {
        component: usize,
        axis: usize,
        value: Data<S>,
    },
}

impl<S: Coefficient> FluxTerm<S> {
    /// Exact flat gradient coordinates paired with one prescribed flux datum.
    /// Isotropic stress keeps its divergence-conforming meaning while sharing
    /// the same component contraction as a general prescribed tensor.
    pub(super) fn prescribed_components(
        &self,
        dimension: usize,
        components: usize,
    ) -> Result<Option<(&Data<S>, impl Iterator<Item = usize>)>, Diagnostic> {
        let (data, start, count, stride) = match self {
            Self::Trial(_) => return Ok(None),
            Self::Isotropic(value) if components == dimension => {
                (value, 0, dimension, dimension + 1)
            }
            Self::Prescribed {
                component,
                axis,
                value,
            } if *component < components && *axis < dimension => {
                (value, component * dimension + axis, 1, 1)
            }
            _ => {
                return Err(invalid(
                    "prescribed flux differs from its exact Field gradient shape",
                ));
            }
        };
        Ok(Some((
            data,
            (0..count).map(move |index| start + index * stride),
        )))
    }

    fn same_coefficient(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Trial(a), Self::Trial(b)) => {
                a.trial == b.trial
                    && a.derivative == b.derivative
                    && a.pairing == b.pairing
                    && a.coefficient.same_coefficient(&b.coefficient)
            }
            (Self::Isotropic(a), Self::Isotropic(b)) => a.same_coefficient(b),
            (
                Self::Prescribed {
                    component: ac,
                    axis: aa,
                    value: a,
                },
                Self::Prescribed {
                    component: bc,
                    axis: ba,
                    value: b,
                },
            ) => ac == bc && aa == ba && a.same_coefficient(b),
            _ => false,
        }
    }

    pub(super) fn bind_parameter_point(
        &mut self,
        point: &crate::parameter_point::ParameterPoint,
    ) -> Result<(), Diagnostic> {
        let coefficient = match self {
            Self::Trial(term) => &mut term.coefficient,
            Self::Isotropic(coefficient)
            | Self::Prescribed {
                value: coefficient, ..
            } => coefficient,
        };
        *coefficient = coefficient.bind_parameter_point(point)?;
        Ok(())
    }
}

impl<S: Coefficient> CompiledRegionForm<S> {
    pub(crate) fn require_boundary_flux(
        &self,
        program: &KernelProgram,
        boundary: RawId,
        relation: RawId,
        field: RawId,
        normal: ExprId,
        homogeneous: bool,
    ) -> Result<(), Diagnostic> {
        let owns_support = crate::canonical::boundary_parent(program, boundary)
            == Some(self.domain)
            || matches!(program.node(boundary), Some(KernelNode::Domain(definition))
                if matches!(definition.kind(), DomainKind::PhysicalInterface { boundaries }
                    if boundaries.iter().any(|side|
                        crate::canonical::boundary_parent(program, side.erase()) == Some(self.domain))));
        if !owns_support || !crate::canonical::relations_on(program, boundary).contains(&relation) {
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
        let Some(ExprNode::NormalComponent { value: flux, on }) = typed.expression().node(normal)
        else {
            return Err(invalid(
                "flux witness requires an exact normal-component operator",
            ));
        };
        if on.erase() != boundary {
            return Err(invalid("flux witness normal has a foreign exact support"));
        }
        let coefficients =
            super::super::spatial::coefficients(program, self.dimension, &self.roles)?;
        let context = Context {
            time_s: None,
            program,
            dag: typed.expression(),
            owner: relation,
            dimension: self.dimension,
            coefficients: &coefficients,
        };
        let candidate = super::lowering::boundary_flux(&context, *flux, row)?;
        if candidate.len() != row.flux.len() {
            return Err(invalid(
                "boundary flux differs from the complete volume constitutive flux",
            ));
        }
        if same_flux(&row.flux, candidate.clone()) {
            return Ok(());
        }
        // Only a homogeneous equation may reverse its complete constitutive
        // flux without changing the prescribed datum. Individual terms cannot
        // change sign independently; all operator/Field identities stay exact.
        if homogeneous {
            let reversed = candidate
                .into_iter()
                .map(|mut term| {
                    let coefficient = match &mut term {
                        FluxTerm::Trial(term) => &mut term.coefficient,
                        FluxTerm::Isotropic(value) | FluxTerm::Prescribed { value, .. } => value,
                    };
                    *coefficient = coefficient
                        .clone()
                        .multiply(Data::constant(self.dimension, <S as From<f64>>::from(-1.0)));
                    term
                })
                .collect();
            if same_flux(&row.flux, reversed) {
                return Ok(());
            }
        }
        Err(invalid(
            "boundary flux differs from the exact Field/operator/coefficient identity",
        ))
    }
}

fn same_flux<S: Coefficient>(expected: &[FluxTerm<S>], mut candidate: Vec<FluxTerm<S>>) -> bool {
    if expected.len() != candidate.len() {
        return false;
    }
    for expected in expected {
        let Some(index) = candidate
            .iter()
            .position(|term| expected.same_coefficient(term))
        else {
            return false;
        };
        candidate.remove(index);
    }
    true
}
