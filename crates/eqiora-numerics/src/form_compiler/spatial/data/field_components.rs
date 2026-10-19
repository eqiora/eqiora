//! Exact Field/component/gradient bindings shared by spatial component actions.
use super::*;
use crate::form_compiler::bilinear::BasisComponent;
use eqiora_core::ScalarDomain;
use eqiora_ir::{ComponentScalarRow, ComponentScalarization, ScalarSymbolCoordinate};
use eqiora_schema::kernel::KernelNode;

mod action;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ComponentColumn {
    pub(super) field: RawId,
    pub(super) component: BasisComponent,
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
    ) -> impl Iterator<Item = (RawId, BasisComponent)> + '_ {
        self.columns
            .iter()
            .map(|column| (column.field, column.component))
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

    pub(in crate::form_compiler) fn bind_parameter_point(
        &self,
        point: &crate::parameter_point::ParameterPoint,
    ) -> Result<Self, Diagnostic> {
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
                    Some((field.erase(), field))
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
        for (_, field) in fields {
            let Some(KernelNode::Field(definition)) = self.program.node(field.erase()) else {
                return Err(super::super::invalid("component trial has no exact Field"));
            };
            let value_type = definition.value_type();
            if value_type.scalar_domain() != ty.scalar_domain() {
                return Err(super::super::invalid(
                    "mixed real/complex component actions require a real-coordinate realization",
                ));
            }
            let mut coordinates = std::collections::BTreeMap::new();
            for coordinate in lowered
                .rows()
                .iter()
                .flat_map(ComponentScalarRow::symbols)
                .filter(|coordinate| coordinate.symbol() == SymbolRef::Field(field))
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
                    column.field != field.erase() || column.component != component
                }) {
                    columns.push(ComponentColumn {
                        field: field.erase(),
                        component,
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
