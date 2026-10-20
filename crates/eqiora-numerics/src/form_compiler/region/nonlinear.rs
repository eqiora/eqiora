//! Equation-derived nonlinear flux actions, independent of physical field names.

use crate::form_compiler::region::integration::IntegralField;
use eqiora_assembly::LocalContribution;
use eqiora_core::{Diagnostic, RawId};
use eqiora_meshing::{AffineGeometryMap, QuadratureRule};

use super::{BoundRegionForm, Data, invalid};
use crate::form_compiler::bilinear::BasisComponent;
use crate::form_compiler::spatial::data::field_components::FieldComponent;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ComponentTerm<S: crate::spatial_expression::Coefficient> {
    pub test: BasisComponent,
    pub value: std::sync::Arc<FieldComponent<S>>,
    pub coefficient: Data<S>,
}

impl<S: crate::spatial_expression::Coefficient> ComponentTerm<S> {
    pub(super) fn same_component(&self, other: &Self) -> bool {
        self.test == other.test && self.value.same_expression(&other.value)
    }

    pub(super) fn add_coefficient(&mut self, other: &Self) -> bool {
        if !self.same_component(other) {
            return false;
        }
        self.coefficient = self.coefficient.clone().add(other.coefficient.clone());
        true
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct DyadicTerm<S: crate::spatial_expression::Coefficient> {
    pub field: RawId,
    pub coefficient: Data<S>,
    /// A retained div(field)=0 constraint permits the skew split by parts.
    /// This is a mathematical condition, not a velocity or fluid role.
    pub split: bool,
}

pub(crate) struct RegionLinearization {
    pub(crate) jacobian: Vec<f64>,
    pub(crate) residual: Vec<f64>,
}

impl RegionLinearization {
    pub(crate) fn into_contribution(
        self,
        point: &[f64],
    ) -> Result<LocalContribution<f64>, Diagnostic> {
        let rows = self.residual.len();
        let columns = point.len();
        if rows == 0 || columns == 0 || rows.checked_mul(columns) != Some(self.jacobian.len()) {
            return Err(invalid(
                "region linearization point or matrix shape mismatch",
            ));
        }
        let rhs = self
            .jacobian
            .chunks_exact(columns)
            .zip(&self.residual)
            .map(|(row, residual)| {
                row.iter().zip(point).map(|(j, u)| j * u).sum::<f64>() - residual
            })
            .collect();
        LocalContribution::new(rows, columns, self.jacobian, rhs)
    }
}

impl BoundRegionForm<f64> {
    /// Anonymous source/test pairing, shared by caller-provided mathematical loads.
    pub(crate) fn pair_load(
        &self,
        field: RawId,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
        datum: impl Fn(&[f64]) -> Result<Vec<f64>, Diagnostic>,
    ) -> Result<LocalContribution<f64>, Diagnostic> {
        let row = self
            .fields
            .iter()
            .position(|layout| layout.field == field)
            .ok_or_else(|| invalid("source pairing has a foreign tested Field"))?;
        let offset = self.fields[..row]
            .iter()
            .map(|layout| layout.components)
            .sum::<usize>();
        super::integration::integrate(
            self.reference,
            &self
                .fields
                .iter()
                .map(IntegralField::from)
                .collect::<Vec<_>>(),
            &[],
            geometry,
            quadrature,
            |point, _, forcing, _| {
                let values = datum(point)?;
                if values.len() != self.fields[row].components {
                    return Err(invalid("source pairing component mismatch"));
                }
                forcing[offset..offset + values.len()].copy_from_slice(&values);
                Ok(())
            },
        )
    }

    /// Complete a nonlinear divergence on an authored natural-flux boundary.
    /// A divergence-free skew split retains half the physical dyadic flux;
    /// conservative integration by parts retains its complete flux.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn natural_facet_action(
        &self,
        field: RawId,
        cell: &AffineGeometryMap,
        facet: (
            &AffineGeometryMap,
            eqiora_meshing::EntityIncidence,
            &[usize],
        ),
        rule: &QuadratureRule,
        point: &[f64],
        derivative: bool,
        datum: impl Fn(&[f64], &[f64]) -> Result<Vec<f64>, Diagnostic>,
    ) -> Result<RegionLinearization, Diagnostic> {
        self.prepare_natural_facet(field, cell, facet, rule, datum)?
            .action(point, derivative)
    }
}
