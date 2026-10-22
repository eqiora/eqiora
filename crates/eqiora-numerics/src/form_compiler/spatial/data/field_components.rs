//! Exact Field value/rate/component bindings shared by spatial component actions.
use super::*;
use crate::form_compiler::bilinear::BasisComponent;
use eqiora_core::ScalarDomain;
use eqiora_ir::{ComponentScalarRow, ComponentScalarization, ScalarSymbolCoordinate};
use eqiora_schema::kernel::KernelNode;

mod action;
#[cfg(test)]
mod tests;

/// Real/imaginary tangent demands in exact physical component-column order.
#[derive(Debug, Clone, Copy)]
pub(in crate::form_compiler) struct FieldDerivatives<'a>(
    pub(in crate::form_compiler) &'a [[bool; 2]],
);

impl FieldDerivatives<'_> {
    fn includes(self, columns: &[ComponentColumn], index: usize) -> bool {
        columns.iter().zip(self.0).any(|(column, parts)| {
            (column.real == Some(index) && parts[0])
                || (column.imaginary == Some(index) && parts[1])
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ComponentColumn {
    pub(super) field: RawId,
    pub(super) derivative: bool,
    pub(super) component: BasisComponent,
    pub(super) scalar_domain: ScalarDomain,
    pub(super) real: Option<usize>,
    pub(super) imaginary: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub(in crate::form_compiler) struct FieldComponent<S: Coefficient> {
    pub(super) affine: bool,
    pub(super) rows: Vec<ComponentScalarRow>,
    pub(super) selected: Vec<ScalarSymbolCoordinate>,
    pub(super) bindings: Vec<(ScalarSymbolCoordinate, Input<S>)>,
    pub(super) columns: Vec<ComponentColumn>,
}

impl<S: Coefficient> FieldComponent<S> {
    pub(in crate::form_compiler) fn same_expression(&self, other: &Self) -> bool {
        self.rows.len() == other.rows.len()
            && self.rows.iter().zip(&other.rows).all(|(left, right)| {
                left.is_imaginary() == right.is_imaginary()
                    && left.expression_view().same_expression(
                        right.expression_view(),
                        |a, b| match (
                            self.selected.contains(a.source()),
                            other.selected.contains(b.source()),
                        ) {
                            (true, true) => a.source() == b.source(),
                            (false, false) => self
                                .bindings
                                .iter()
                                .find(|(coordinate, _)| coordinate == a.source())
                                .zip(
                                    other
                                        .bindings
                                        .iter()
                                        .find(|(coordinate, _)| coordinate == b.source()),
                                )
                                .is_some_and(|((a, left), (b, right))| {
                                    left.same_binding(a, right, b)
                                }),
                            _ => false,
                        },
                    )
            })
    }

    pub(in crate::form_compiler) fn is_affine(&self) -> bool {
        self.affine
    }

    pub(in crate::form_compiler) fn columns(
        &self,
    ) -> impl Iterator<Item = (RawId, BasisComponent, bool)> + '_ {
        self.columns
            .iter()
            .map(|column| (column.field, column.component, column.derivative))
    }

    pub(super) fn uses_coordinate(&self, index: usize) -> bool {
        self.rows.iter().any(|row| {
            row.symbols().contains(&self.selected[index])
                && row
                    .constant_coordinate_jacobian(std::slice::from_ref(&self.selected[index]))
                    .map_or(true, |jacobian| {
                        jacobian.coefficients().iter().any(|value| *value != 0.)
                    })
        })
    }

    pub(in crate::form_compiler) fn rate_coordinates(
        &self,
    ) -> Result<Vec<eqiora_core::TimeStateCoordinate>, Diagnostic> {
        let selected = self
            .selected
            .iter()
            .filter(|coordinate| matches!(coordinate.symbol(), SymbolRef::Derivative(_, _)))
            .cloned()
            .collect::<Vec<_>>();
        if selected.is_empty() {
            return Ok(Vec::new());
        }
        for row in &self.rows {
            if !row.affine_in(&selected)? {
                return Err(super::super::invalid(
                    "nonlinear rate dependence requires an explicit initial-rate branch contract",
                ));
            }
        }
        let mut coordinates = Vec::new();
        for column in self.columns.iter().filter(|column| column.derivative) {
            let component = match column.component {
                BasisComponent::Value(component) | BasisComponent::Gradient { component, .. } => {
                    component
                }
            };
            for (imaginary, index) in [(false, column.real), (true, column.imaginary)] {
                if index.is_some_and(|index| self.uses_coordinate(index)) {
                    coordinates.push(eqiora_core::TimeStateCoordinate::new(
                        column.field.downcast().expect("Field column"),
                        0,
                        component,
                        imaginary,
                    ));
                }
            }
        }
        Ok(coordinates)
    }

    pub(in crate::form_compiler) fn polynomial_degree(
        &self,
        value_degree: usize,
        gradient_degree: usize,
    ) -> Option<usize> {
        self.rows.iter().try_fold(0, |degree, row| {
            let inputs = row
                .symbols()
                .iter()
                .map(|coordinate| {
                    if let Some(index) = self.selected.iter().position(|value| value == coordinate)
                    {
                        let column = self.columns.iter().find(|column| {
                            column.real == Some(index) || column.imaginary == Some(index)
                        })?;
                        Some(if column.component.requires_gradient() {
                            gradient_degree
                        } else {
                            value_degree
                        })
                    } else {
                        self.bindings
                            .iter()
                            .find(|(value, _)| value == coordinate)?
                            .1
                            .polynomial_degree()
                    }
                })
                .collect::<Vec<_>>();
            Some(degree.max(row.polynomial_degree(&inputs)?))
        })
    }

    pub(in crate::form_compiler) fn bind_inputs<'a>(
        &self,
        point: impl Into<crate::form_compiler::spatial::InputPoint<'a>>,
    ) -> Result<Self, Diagnostic> {
        let point = point.into();
        let mut bound = self.clone();
        bound.bindings = self
            .bindings
            .iter()
            .map(|(coordinate, input)| {
                input
                    .bind(coordinate, point)
                    .map(|input| (coordinate.clone(), input))
            })
            .collect::<Result<_, _>>()?;
        Ok(bound)
    }
}

impl<S: Coefficient> Context<'_, S> {
    pub(in crate::form_compiler) fn field_components(
        &self,
        id: ExprId,
        depth: usize,
    ) -> Result<Option<Vec<Arc<FieldComponent<S>>>>, Diagnostic> {
        let typed = super::super::super::scalar::typed_relation(self.program, self.owner)?;
        if typed.expression() != self.dag {
            return Err(super::super::invalid(
                "Field components require their exact typed expression",
            ));
        }
        let ty = &typed
            .node_type(id)
            .ok_or_else(|| super::super::invalid("missing component type"))?
            .value_type;
        let lowered = ComponentScalarization::lower_spatial_selected(&typed, &[id], None)?;
        let fields = lowered
            .rows()
            .iter()
            .flat_map(ComponentScalarRow::symbols)
            .filter_map(|coordinate| match coordinate.symbol() {
                SymbolRef::Field(field) if !self.coefficients.contains_key(&field.erase()) => {
                    Some(((field.erase(), false), field))
                }
                SymbolRef::Derivative(field, std::num::NonZeroU32::MIN)
                    if !self.coefficients.contains_key(&field.erase()) =>
                {
                    Some(((field.erase(), true), field))
                }
                _ => None,
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        if fields.is_empty() {
            return Ok(None);
        }
        let complex = ty.scalar_domain() == ScalarDomain::Complex;
        if !matches!(
            ty.scalar_domain(),
            ScalarDomain::Real | ScalarDomain::Complex
        ) || (complex && S::DOMAIN != ScalarDomain::Complex)
        {
            return Err(super::super::invalid(
                "component action requires its exact numeric scalar domain",
            ));
        }
        let mut selected = Vec::new();
        let mut columns = Vec::new();
        for ((_, derivative), field) in fields {
            let symbol = if derivative {
                SymbolRef::Derivative(field, std::num::NonZeroU32::MIN)
            } else {
                SymbolRef::Field(field)
            };
            let Some(KernelNode::Field(definition)) = self.program.node(field.erase()) else {
                return Err(super::super::invalid("component trial has no exact Field"));
            };
            let value_type = definition.value_type();
            if !matches!(
                value_type.scalar_domain(),
                ScalarDomain::Real | ScalarDomain::Complex
            ) || (value_type.scalar_domain() == ScalarDomain::Complex
                && S::DOMAIN != ScalarDomain::Complex)
            {
                return Err(super::super::invalid(
                    "component trial requires its complete numeric scalar domain",
                ));
            }
            let mut coordinates = std::collections::BTreeMap::new();
            for coordinate in lowered
                .rows()
                .iter()
                .flat_map(ComponentScalarRow::symbols)
                .filter(|coordinate| coordinate.symbol() == symbol)
            {
                let component = value_type
                    .shape()
                    .flat_index(coordinate.component_index())
                    .ok_or_else(|| {
                        super::super::invalid("trial component differs from its exact shape")
                    })?;
                let component = match coordinate.spatial_derivative_axes() {
                    [] => BasisComponent::Value(component),
                    [axis] if *axis < self.dimension => BasisComponent::Gradient {
                        component,
                        axis: *axis,
                    },
                    _ => {
                        return Err(super::super::invalid(
                            "spatial component action requires an admitted first-gradient basis",
                        ));
                    }
                };
                coordinates.insert((component, coordinate.is_imaginary()), coordinate.clone());
            }
            for ((component, imaginary), coordinate) in coordinates {
                if columns.last().is_none_or(|column: &ComponentColumn| {
                    column.field != field.erase()
                        || column.component != component
                        || column.derivative != derivative
                }) {
                    columns.push(ComponentColumn {
                        field: field.erase(),
                        derivative,
                        component,
                        scalar_domain: value_type.scalar_domain(),
                        real: None,
                        imaginary: None,
                    });
                }
                let column = columns.last_mut().expect("component column");
                if imaginary {
                    column.imaginary = Some(selected.len());
                } else {
                    column.real = Some(selected.len());
                }
                selected.push(coordinate);
            }
        }
        let mut bindings: Vec<(ScalarSymbolCoordinate, Input<S>)> = Vec::new();
        for coordinate in lowered.rows().iter().flat_map(ComponentScalarRow::symbols) {
            if !selected.contains(coordinate)
                && !bindings.iter().any(|(other, _)| other == coordinate)
            {
                bindings.push((
                    coordinate.clone(),
                    Input::from_coordinate(self, coordinate, depth)?,
                ));
            }
        }
        let parts = if complex { 2 } else { 1 };
        if !lowered.rows().len().is_multiple_of(parts) {
            return Err(super::super::invalid(
                "component action has unpaired scalar coordinates",
            ));
        }
        Ok(Some(
            lowered
                .rows()
                .chunks(parts)
                .map(|rows| {
                    let affine = rows.iter().try_fold(true, |affine, row| {
                        row.affine_in(&selected).map(|value| affine && value)
                    })?;
                    Ok(Arc::new(FieldComponent {
                        affine,
                        rows: rows.to_vec(),
                        selected: selected.clone(),
                        bindings: bindings.clone(),
                        columns: columns.clone(),
                    }))
                })
                .collect::<Result<_, Diagnostic>>()?,
        ))
    }
}
