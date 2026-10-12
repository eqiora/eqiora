use std::collections::BTreeMap;

use eqiora_core::{Diagnostic, RawId};
use eqiora_schema::kernel::{ExprId, ExprNode, SymbolRef};

use super::data::{Context, Data};
use crate::spatial_expression::Coefficient;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Terms<S: Coefficient> {
    pub(super) constant: Data<S>,
    pub(super) reaction: BTreeMap<RawId, Data<S>>,
    pub(super) storage: BTreeMap<RawId, Data<S>>,
    pub(super) diffusion: BTreeMap<RawId, Data<S>>,
    pub(super) transport: BTreeMap<(RawId, usize), Data<S>>,
}

impl<S: Coefficient> Terms<S> {
    /// x=lambda*xi+b: dxi=dx/J and grad_xi=lambda*grad_x.
    /// Preserve the authored reference flux while integrating on the current
    /// physical cells. Geometry history supplies the previous physical measure.
    pub(super) fn on_uniform_chart(
        mut self,
        chart: &super::motion::UniformChart,
    ) -> Result<Self, Diagnostic> {
        let jacobian = chart.scale * chart.scale;
        if !jacobian.is_finite() || jacobian <= 0.0 {
            return Err(super::invalid(
                "moving chart has an unresolved volume scale",
            ));
        }
        let transform = |data: &Data<S>, factor: f64| {
            data.on_uniform_chart(chart)
                .multiply(Data::constant(2, <S as From<f64>>::from(factor)))
        };
        self.constant = transform(&self.constant, 1.0 / jacobian);
        for data in self.reaction.values_mut().chain(self.storage.values_mut()) {
            *data = transform(data, 1.0 / jacobian);
        }
        for data in self.diffusion.values_mut() {
            *data = transform(data, chart.scale * chart.scale / jacobian);
        }
        for data in self.transport.values_mut() {
            *data = transform(data, chart.scale / jacobian);
        }
        Ok(self)
    }

    fn data(constant: Data<S>) -> Self {
        Self {
            constant,
            reaction: BTreeMap::new(),
            storage: BTreeMap::new(),
            diffusion: BTreeMap::new(),
            transport: BTreeMap::new(),
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
        for (key, value) in right.transport {
            let sum = self
                .transport
                .remove(&key)
                .map_or_else(|| value.clone(), |left| left.add(value.clone()));
            self.transport.insert(key, sum);
        }
        self
    }
    pub(super) fn scale(self, data: Data<S>) -> Result<Self, Diagnostic> {
        if (!self.diffusion.is_empty() || !self.transport.is_empty()) && data.spatial() {
            return Err(super::invalid(
                "spatial factors outside divergence require additional weak derivative terms",
            ));
        }
        Ok(self.scale_flux(data))
    }
    fn scale_flux(mut self, data: Data<S>) -> Self {
        self.constant = self.constant.multiply(data.clone());
        for value in self
            .reaction
            .values_mut()
            .chain(self.diffusion.values_mut())
            .chain(self.storage.values_mut())
            .chain(self.transport.values_mut())
        {
            *value = value.clone().multiply(data.clone());
        }
        self
    }
}

impl<S: Coefficient> Context<'_, S> {
    pub(super) fn conservation(
        &self,
        law: eqiora_schema::kernel::ConservationTerms,
    ) -> Result<Terms<S>, Diagnostic> {
        let mut row = self.flux_terms(law.flux(), 0)?;
        if row.diffusion.len() != 1 {
            return Err(super::invalid(
                "scalar Law requires one principal diffusive Field",
            ));
        }
        let field = *row.diffusion.keys().next().expect("one diffusive Field");
        let source = self.terms(law.source(), 0)?;
        if !source.storage.is_empty()
            || !source.diffusion.is_empty()
            || !source.transport.is_empty()
        {
            return Err(super::invalid(
                "scalar Law source requires prescribed data or linear reaction terms",
            ));
        }
        row = row.add(source.scale(Data::constant(self.dimension, <S as From<f64>>::from(-1.0)))?);
        if let Some((stored, _accumulation)) = law.storage() {
            // Kernel admission independently proves accumulation is d(stored)/dt.
            // This numerical slice reads exact physical storage instead of expanding
            // the compiler's formal partial-operator application.
            let storage = self.terms(stored, 0)?;
            if !storage.diffusion.is_empty()
                || !storage.transport.is_empty()
                || !storage.storage.is_empty()
                || storage.reaction.len() != 1
                || !storage.reaction.contains_key(&field)
                || storage.constant.spatial()
                || storage.constant.evaluate(&vec![0.0; self.dimension])?
                    != <S as From<f64>>::from(0.0)
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
            Some(ExprNode::PureOperatorApplication(_))
                if self.dimension == 2
                    && super::super::planar_curl::gradient(self.dag, id).is_some() =>
            {
                Ok(Some(-1))
            }
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

    pub(super) fn terms(&self, id: ExprId, depth: usize) -> Result<Terms<S>, Diagnostic> {
        if depth > 128 {
            return Err(super::invalid("linear expression nesting exceeds 128"));
        }
        if let Ok(data) = self.data(id, depth) {
            return Ok(Terms::data(data));
        }
        let terms = |id| self.terms(id, depth + 1);
        let one = || Data::constant(self.dimension, <S as From<f64>>::from(1.0));
        let minus = || Data::constant(self.dimension, <S as From<f64>>::from(-1.0));
        match self.dag.node(id) {
            Some(ExprNode::Symbol(SymbolRef::Field(field))) => {
                let mut terms =
                    Terms::data(Data::constant(self.dimension, <S as From<f64>>::from(0.0)));
                terms.reaction.insert(field.erase(), one());
                Ok(terms)
            }
            Some(ExprNode::Symbol(SymbolRef::Derivative(field, std::num::NonZeroU32::MIN))) => {
                let mut terms =
                    Terms::data(Data::constant(self.dimension, <S as From<f64>>::from(0.0)));
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
            Some(ExprNode::PureOperatorApplication(_))
                if self.dimension == 2
                    && super::super::planar_curl::gradient(self.dag, id).is_some() =>
            {
                let gradient = super::super::planar_curl::gradient(self.dag, id)
                    .expect("checked planar composition");
                let (field, coefficient) = self.flux(gradient, depth + 1)?;
                let mut terms =
                    Terms::data(Data::constant(self.dimension, <S as From<f64>>::from(0.0)));
                terms.diffusion.insert(field, coefficient);
                Ok(terms)
            }
            Some(ExprNode::Divergence(flux)) => self.flux_terms(*flux, depth + 1),
            _ => Err(super::invalid(
                "unsupported or nonlinear scalar equation operator",
            )),
        }
    }

    pub(super) fn flux(&self, id: ExprId, depth: usize) -> Result<(RawId, Data<S>), Diagnostic> {
        if depth > 128 {
            return Err(super::invalid("linear flux nesting exceeds 128"));
        }
        match self.dag.node(id) {
            Some(ExprNode::Gradient(field)) => match self.dag.node(*field) {
                Some(ExprNode::Symbol(SymbolRef::Field(field)))
                    if !self.coefficients.contains_key(&field.erase()) =>
                {
                    Ok((
                        field.erase(),
                        Data::constant(self.dimension, <S as From<f64>>::from(1.0)),
                    ))
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
                    coefficient
                        .multiply(Data::constant(self.dimension, <S as From<f64>>::from(-1.0))),
                ))
            }
            _ => Err(super::invalid(
                "unsupported or unknown-dependent diffusion coefficient",
            )),
        }
    }
}

mod transport;
