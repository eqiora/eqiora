//! Independent first chain rule for unknown Fields on exact mapped coordinates.
use super::{Atom, Budget, Error, ExactRational, ExprDag, ExprId, ExprNode, Maps, Polynomial};
use super::{require_profile, selector};
use crate::kernel::SymbolRef;
use eqiora_core::{Id, entity::kinds};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Projection {
    Value,
    Time,
    Coordinate(Atom),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct MappedField {
    map: usize,
    field: Id<kinds::Field>,
    projection: Projection,
}

impl Maps {
    pub(in super::super) fn partial(
        &mut self,
        dag: &ExprDag,
        value: ExprId,
        wrt: ExprId,
        budget: &mut Budget,
    ) -> Result<Polynomial, Error> {
        let Some(ExprNode::Symbol(symbol @ SymbolRef::Field(_))) = dag.node(value) else {
            return Err(Error::UnsupportedExpression);
        };
        let key = (*symbol, selector(dag, wrt)?);
        budget.charge(self.partials.len() + 1)?;
        let index = match self.partials.iter().position(|prior| *prior == key) {
            Some(index) => index,
            None => {
                let index = self.partials.len();
                self.partials.push(key);
                index
            }
        };
        Ok(Polynomial::atom(Atom::Partial(index)))
    }

    fn mapped(&mut self, field: MappedField, budget: &mut Budget) -> Result<Polynomial, Error> {
        budget.charge(self.fields.len() + 1)?;
        let index = match self.fields.iter().position(|prior| *prior == field) {
            Some(index) => index,
            None => {
                let index = self.fields.len();
                self.fields.push(field);
                index
            }
        };
        Ok(Polynomial::atom(Atom::MappedField(index)))
    }

    pub(in super::super::super) fn field_rate(
        &mut self,
        index: usize,
        budget: &mut Budget,
    ) -> Result<Polynomial, Error> {
        let field = self.fields[index];
        if field.projection != Projection::Value {
            return Err(Error::UnsupportedExpression);
        }
        let mut result = self.mapped(
            MappedField {
                projection: Projection::Time,
                ..field
            },
            budget,
        )?;
        // At fixed source coordinates: q_t composed with chi, plus each
        // physical coordinate partial composed with chi times chi_t.
        for row in 0..self.maps[field.map].targets.len() {
            let coordinate = self.maps[field.map].targets[row];
            budget.polynomial(&self.maps[field.map].rates[row])?;
            let rate = self.maps[field.map].rates[row].clone();
            let partial = self.mapped(
                MappedField {
                    projection: Projection::Coordinate(coordinate),
                    ..field
                },
                budget,
            )?;
            let term = partial.checked_mul(&rate)?;
            budget.polynomial(&term)?;
            result = result.checked_add(&term)?;
            budget.polynomial(&result)?;
        }
        Ok(result)
    }

    pub(in super::super) fn pullback<'a>(
        &mut self,
        dag: &ExprDag,
        value_id: ExprId,
        source: &[ExprId],
        at: &[(ExprId, ExprId)],
        get: &impl Fn(ExprId) -> Result<&'a Polynomial, Error>,
        budget: &mut Budget,
    ) -> Result<Polynomial, Error> {
        require_profile(dag, value_id, true, true, budget)?;
        let map = self.intern(dag, None, source, at, get, budget)?;
        let value = get(value_id)?;
        let mut result = Polynomial::constant(ExactRational::integer(0));
        for (atoms, coefficient) in value.terms() {
            let mut term = Polynomial::constant(coefficient);
            for atom in atoms {
                let field = match *atom {
                    Atom::Field(field) => Some((field, Projection::Value)),
                    Atom::Derivative(field) => Some((field, Projection::Time)),
                    Atom::Partial(index) => {
                        let (SymbolRef::Field(field), coordinate) = self.partials[index] else {
                            return Err(Error::UnsupportedExpression);
                        };
                        budget.charge(self.maps[map].targets.len())?;
                        if !self.maps[map].targets.contains(&coordinate) {
                            return Err(Error::UnsupportedExpression);
                        }
                        Some((field, Projection::Coordinate(coordinate)))
                    }
                    _ => None,
                };
                let replacement = if let Some((field, projection)) = field {
                    self.mapped(
                        MappedField {
                            map,
                            field,
                            projection,
                        },
                        budget,
                    )?
                } else {
                    if !matches!(atom, Atom::Coordinate(..) | Atom::Parameter(_) | Atom::Time) {
                        return Err(Error::UnsupportedExpression);
                    }
                    budget.charge(at.len())?;
                    if let Some(row) = self.maps[map]
                        .targets
                        .iter()
                        .position(|target| target == atom)
                    {
                        budget.polynomial(&self.maps[map].rows[row])?;
                        self.maps[map].rows[row].clone()
                    } else {
                        Polynomial::atom(*atom)
                    }
                };
                term = term.checked_mul(&replacement)?;
                budget.polynomial(&term)?;
            }
            result = result.checked_add(&term)?;
            budget.polynomial(&result)?;
        }
        Ok(result)
    }
}
