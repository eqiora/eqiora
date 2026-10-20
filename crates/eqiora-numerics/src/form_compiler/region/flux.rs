//! Exact constitutive flux retained separately from its volume divergence.

use eqiora_schema::kernel::{DomainKind, ExprId, ExprNode, KernelNode};

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum FluxTerm<S: Coefficient> {
    Trial(Term<S>),
    Component(nonlinear::ComponentTerm<S>),
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
    pub(super) fn coefficient(&self) -> &Data<S> {
        match self {
            Self::Trial(term) => &term.coefficient,
            Self::Component(term) => &term.coefficient,
            Self::Isotropic(value) | Self::Prescribed { value, .. } => value,
        }
    }

    fn add_coefficient(&mut self, other: &Self) -> bool {
        let (left, right) = match (self, other) {
            (Self::Trial(a), Self::Trial(b)) => return a.add_coefficient(b),
            (Self::Component(a), Self::Component(b)) => return a.add_coefficient(b),
            (Self::Isotropic(a), Self::Isotropic(b)) => (a, b),
            (
                Self::Prescribed {
                    component: a,
                    axis: c,
                    value: e,
                },
                Self::Prescribed {
                    component: b,
                    axis: d,
                    value: f,
                },
            ) if a == b && c == d => (e, f),
            _ => return false,
        };
        *left = left.clone().add(right.clone());
        true
    }

    /// Exact flat gradient coordinates paired with one prescribed flux datum.
    /// Isotropic stress keeps its divergence-conforming meaning while sharing
    /// the same component contraction as a general prescribed tensor.
    pub(super) fn prescribed_components(
        &self,
        dimension: usize,
        components: usize,
    ) -> Result<Option<(&Data<S>, impl Iterator<Item = usize>)>, Diagnostic> {
        let (data, start, count, stride) = match self {
            Self::Trial(_) | Self::Component(_) => return Ok(None),
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
            (Self::Component(a), Self::Component(b)) => {
                a.same_component(b) && a.coefficient.same_coefficient(&b.coefficient)
            }
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

    pub(super) fn map_data(
        &mut self,
        map: &impl Fn(&Data<S>) -> Result<Data<S>, Diagnostic>,
    ) -> Result<(), Diagnostic> {
        let coefficient = match self {
            Self::Trial(term) => &mut term.coefficient,
            Self::Component(term) => &mut term.coefficient,
            Self::Isotropic(coefficient)
            | Self::Prescribed {
                value: coefficient, ..
            } => coefficient,
        };
        *coefficient = map(coefficient)?;
        Ok(())
    }
}

impl<S: Coefficient> Row<S> {
    /// Collect by exact trial/derivative/pairing identity before coefficient
    /// validation. Keep the full Data expression, including cancelled domains.
    pub(super) fn collect_coefficients(&mut self) {
        collect(&mut self.terms, Term::add_coefficient);
        collect(
            &mut self.components,
            nonlinear::ComponentTerm::add_coefficient,
        );
        collect(&mut self.flux, FluxTerm::add_coefficient);
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
                        FluxTerm::Component(term) => &mut term.coefficient,
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

fn collect<T>(terms: &mut Vec<T>, add: impl Fn(&mut T, &T) -> bool) {
    let mut collected = Vec::new();
    for term in std::mem::take(terms) {
        if !collected.iter_mut().any(|existing| add(existing, &term)) {
            collected.push(term);
        }
    }
    *terms = collected;
}
