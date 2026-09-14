use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, RawId};
use eqiora_schema::kernel::{ExprId, ExprNode, SymbolRef};

use super::data::{Context, Data};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Terms {
    pub(super) constant: Data,
    pub(super) reaction: BTreeMap<RawId, Data>,
    pub(super) storage: BTreeMap<RawId, Data>,
    pub(super) diffusion: BTreeMap<RawId, Data>,
}

impl Terms {
    fn data(constant: Data) -> Self {
        Self {
            constant,
            reaction: BTreeMap::new(),
            storage: BTreeMap::new(),
            diffusion: BTreeMap::new(),
        }
    }
    fn add(mut self, right: Self) -> Self {
        self.constant = self.constant.add(right.constant);
        for (target, terms) in [
            (&mut self.reaction, right.reaction),
            (&mut self.storage, right.storage),
            (&mut self.diffusion, right.diffusion),
        ] {
            for (field, value) in terms {
                let sum = target
                    .remove(&field)
                    .map_or_else(|| value.clone(), |left| left.add(value.clone()));
                target.insert(field, sum);
            }
        }
        self
    }
    pub(super) fn scale(mut self, data: Data) -> Result<Self, Diagnostic> {
        if !self.diffusion.is_empty() && data.spatial() {
            return Err(super::invalid(
                "spatial factors outside divergence require additional weak derivative terms",
            ));
        }
        self.constant = self.constant.multiply(data.clone());
        for value in self
            .reaction
            .values_mut()
            .chain(self.diffusion.values_mut())
            .chain(self.storage.values_mut())
        {
            *value = value.clone().multiply(data.clone());
        }
        Ok(self)
    }
}

impl Context<'_> {
    pub(super) fn conservation(
        &self,
        law: eqiora_schema::kernel::ConservationTerms,
    ) -> Result<Terms, Diagnostic> {
        let (field, coefficient) = self.flux(law.flux(), 0)?;
        let mut row = Terms::data(
            self.data(law.source(), 0)?
                .multiply(Data::constant(self.dimension, -1.0)),
        );
        row.diffusion.insert(
            field,
            coefficient.multiply(Data::constant(self.dimension, -1.0)),
        );
        if let Some((stored, _accumulation)) = law.storage() {
            // Kernel admission independently proves accumulation is d(stored)/dt.
            // This numerical slice reads exact physical storage instead of expanding
            // the compiler's formal partial-operator application.
            let storage = self.terms(stored, 0)?;
            if !storage.diffusion.is_empty()
                || !storage.storage.is_empty()
                || storage.reaction.len() != 1
                || !storage.reaction.contains_key(&field)
                || storage.constant.spatial()
                || storage.constant.evaluate(&vec![0.0; self.dimension])? != 0.0
            {
                return Err(super::invalid(
                    "scalar Law storage requires one coefficient times its exact Field",
                ));
            }
            row.storage = storage.reaction;
        }
        Ok(row)
    }

    pub(super) fn diffusion_orientation(
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

    pub(super) fn terms(&self, id: ExprId, depth: usize) -> Result<Terms, Diagnostic> {
        if depth > 128 {
            return Err(super::invalid("linear expression nesting exceeds 128"));
        }
        if let Ok(data) = self.data(id, depth) {
            return Ok(Terms::data(data));
        }
        let terms = |id| self.terms(id, depth + 1);
        let one = || Data::constant(self.dimension, 1.0);
        let minus = || Data::constant(self.dimension, -1.0);
        match self.dag.node(id) {
            Some(ExprNode::Symbol(SymbolRef::Field(field))) => {
                let mut terms = Terms::data(Data::constant(self.dimension, 0.0));
                terms.reaction.insert(field.erase(), one());
                Ok(terms)
            }
            Some(ExprNode::Symbol(SymbolRef::Derivative(field))) => {
                let mut terms = Terms::data(Data::constant(self.dimension, 0.0));
                terms.storage.insert(field.erase(), one());
                Ok(terms)
            }
            Some(ExprNode::Add(a, b)) => Ok(terms(*a)?.add(terms(*b)?)),
            Some(ExprNode::Sub(a, b)) => Ok(terms(*a)?.add(terms(*b)?.scale(minus())?)),
            Some(ExprNode::Neg(a)) => terms(*a)?.scale(minus()),
            Some(ExprNode::Mul(a, b)) => {
                if let Ok(data) = self.data(*a, depth + 1) {
                    terms(*b)?.scale(data)
                } else if let Ok(data) = self.data(*b, depth + 1) {
                    terms(*a)?.scale(data)
                } else {
                    Err(super::invalid(
                        "nonlinear product of unknown-dependent expressions",
                    ))
                }
            }
            Some(ExprNode::Div(a, b)) => terms(*a)?.scale(one().divide(self.data(*b, depth + 1)?)),
            Some(ExprNode::Divergence(flux)) => {
                let (field, coefficient) = self.flux(*flux, depth + 1)?;
                let mut terms = Terms::data(Data::constant(self.dimension, 0.0));
                terms.diffusion.insert(field, coefficient.multiply(minus()));
                Ok(terms)
            }
            _ => Err(super::invalid(
                "unsupported or nonlinear scalar equation operator",
            )),
        }
    }

    pub(super) fn flux(&self, id: ExprId, depth: usize) -> Result<(RawId, Data), Diagnostic> {
        if depth > 128 {
            return Err(super::invalid("linear flux nesting exceeds 128"));
        }
        match self.dag.node(id) {
            Some(ExprNode::Gradient(field)) => match self.dag.node(*field) {
                Some(ExprNode::Symbol(SymbolRef::Field(field)))
                    if !self.coefficients.contains_key(&field.erase()) =>
                {
                    Ok((field.erase(), Data::constant(self.dimension, 1.0)))
                }
                _ => Err(super::invalid(
                    "diffusive gradient requires one exact unknown Field",
                )),
            },
            Some(ExprNode::Mul(a, b)) => {
                let (data, flux) = if let Ok(data) = self.data(*a, depth + 1) {
                    (data, *b)
                } else {
                    (
                        self.data(*b, depth + 1).map_err(|_| {
                            super::invalid("unknown-dependent diffusion coefficient")
                        })?,
                        *a,
                    )
                };
                let (field, coefficient) = self.flux(flux, depth + 1)?;
                Ok((field, coefficient.multiply(data)))
            }
            Some(ExprNode::Div(a, b)) => {
                let (field, coefficient) = self.flux(*a, depth + 1)?;
                Ok((field, coefficient.divide(self.data(*b, depth + 1)?)))
            }
            Some(ExprNode::Neg(a)) => {
                let (field, coefficient) = self.flux(*a, depth + 1)?;
                Ok((
                    field,
                    coefficient.multiply(Data::constant(self.dimension, -1.0)),
                ))
            }
            _ => Err(super::invalid(
                "unsupported or unknown-dependent diffusion coefficient",
            )),
        }
    }
}
