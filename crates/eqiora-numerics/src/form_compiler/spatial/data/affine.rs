//! Retained component coefficients use the scalar IR's structural affine proof.
use super::field_components::FieldComponent;
use super::*;
use crate::form_compiler::bilinear::BasisComponent;

pub(in crate::form_compiler) struct ComponentRow<S: Coefficient> {
    pub(in crate::form_compiler) offset: Data<S>,
    pub(in crate::form_compiler) terms: Vec<(RawId, BasisComponent, bool, Data<S>)>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct AffineData<S: Coefficient> {
    row: Arc<FieldComponent<S>>,
    column: Option<usize>,
    input_imaginary: Option<bool>,
    tangent: Option<Vec<Option<Input<S>>>>,
}

impl<S: Coefficient> Context<'_, S> {
    pub(in crate::form_compiler) fn affine_components(
        &self,
        id: ExprId,
        depth: usize,
    ) -> Result<Option<Vec<ComponentRow<S>>>, Diagnostic> {
        let Some(components) = self.field_components(id, depth)? else {
            return Ok(None);
        };
        Ok(Some(components.into_iter().map(project).collect()))
    }
}

pub(in crate::form_compiler) fn project<S: Coefficient>(
    row: Arc<FieldComponent<S>>,
) -> ComponentRow<S> {
    let columns = &row.columns;
    let data = |column| {
        Data(Arc::new(Node::Affine(AffineData {
            row: Arc::clone(&row),
            column,
            input_imaginary: None,
            tangent: None,
        })))
    };
    ComponentRow {
        offset: data(None),
        terms: columns
            .iter()
            .enumerate()
            // Each physical row owns only the trial coordinates present
            // in its exact scalar IR. Keep global column ordinals for
            // projection, without inventing off-axis zero couplings.
            .filter(|(_, column)| {
                [column.real, column.imaginary]
                    .into_iter()
                    .flatten()
                    .any(|index| row.uses_coordinate(index))
            })
            .map(|(index, column)| {
                (
                    column.field,
                    column.component,
                    column.derivative,
                    data(Some(index)),
                )
            })
            .collect(),
    }
}

impl<S: Coefficient> AffineData<S> {
    /// Images of the retained trial's real and imaginary coordinate directions.
    /// Offsets are values, not actions on a trial coordinate.
    pub(super) fn coordinate_images(&self) -> Option<[(Self, bool); 2]> {
        let column = &self.row.columns[self.column?];
        Some([false, true].map(|imaginary| {
            (
                Self {
                    input_imaginary: Some(imaginary),
                    ..self.clone()
                },
                if imaginary {
                    column
                        .imaginary
                        .is_some_and(|index| self.row.uses_coordinate(index))
                } else {
                    column
                        .real
                        .is_some_and(|index| self.row.uses_coordinate(index))
                },
            )
        }))
    }

    pub(super) fn constant_in_direction(
        &self,
        direction: Direction<'_>,
    ) -> Result<bool, Diagnostic> {
        for (index, (coordinate, input)) in self.row.bindings.iter().enumerate() {
            if !input.constant_in_direction(coordinate, direction)? {
                return Ok(false);
            }
            if let Some(values) = &self.tangent
                && let Some(input) = &values[index]
                && !input.constant_in_direction(coordinate, direction)?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(super) fn spatial(&self) -> bool {
        self.row.bindings.iter().any(|(_, input)| input.spatial())
    }
    pub(super) fn bind_inputs<'a>(
        &self,
        point: impl Into<crate::form_compiler::spatial::InputPoint<'a>>,
    ) -> Result<Self, Diagnostic> {
        let point = point.into();
        let row = self.row.bind_inputs(point)?;
        Ok(Self {
            row: Arc::new(row),
            column: self.column,
            input_imaginary: self.input_imaginary,
            tangent: self
                .tangent
                .as_ref()
                .map(|inputs| {
                    inputs
                        .iter()
                        .zip(&self.row.bindings)
                        .map(|(input, (coordinate, _))| {
                            input
                                .as_ref()
                                .map(|input| input.bind(coordinate, point))
                                .transpose()
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?,
        })
    }
    pub(super) fn derivative(
        &self,
        direction: super::Direction<'_>,
        dimension: usize,
    ) -> Result<Self, Diagnostic> {
        if self.tangent.is_some() {
            return Err(super::super::invalid(
                "affine coefficient admits one input derivative",
            ));
        }
        Ok(Self {
            row: Arc::clone(&self.row),
            column: self.column,
            input_imaginary: self.input_imaginary,
            tangent: Some(
                self.row
                    .bindings
                    .iter()
                    .map(|(coordinate, input)| {
                        if input.constant_in_direction(coordinate, direction)? {
                            Ok(None)
                        } else {
                            input.derivative(coordinate, direction, dimension).map(Some)
                        }
                    })
                    .collect::<Result<_, _>>()?,
            ),
        })
    }
    pub(super) fn evaluate(&self, point: &[f64]) -> Result<S, Diagnostic> {
        if !self.row.affine {
            return Err(super::super::invalid(
                "nonlinear Field components require a candidate-dependent action",
            ));
        }
        let bindings = self
            .row
            .bindings
            .iter()
            .map(|(coordinate, input)| {
                input
                    .evaluate(point)
                    .map(|value| (coordinate.clone(), value))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (rows, tangent) = if let Some(tangent) = &self.tangent {
            let bindings = bindings
                .iter()
                .zip(tangent)
                .map(|((coordinate, value), input)| {
                    input
                        .as_ref()
                        .map(|input| input.evaluate(point))
                        .transpose()
                        .map(|tangent| (coordinate.clone(), *value, tangent))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let (rows, tangent): (Vec<_>, Vec<_>) = self
                .row
                .rows
                .iter()
                .map(|row| row.bind_affine_direction(&self.row.selected, &bindings))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .unzip();
            (rows, Some(tangent))
        } else {
            (
                self.row
                    .rows
                    .iter()
                    .map(|row| row.bind_affine(&self.row.selected, &bindings))
                    .collect::<Result<Vec<_>, _>>()?,
                None,
            )
        };
        let value = self.project(&rows)?;
        if let Some(tangent) = tangent {
            self.project(&tangent)
        } else {
            Ok(value)
        }
    }

    fn project(
        &self,
        rows: &[eqiora_ir::BoundAffineScalarIr<ScalarSymbolCoordinate>],
    ) -> Result<S, Diagnostic> {
        let real = &rows[0];
        let imaginary = rows.get(1);
        // The native complex solver consumes a C-linear map. Check its complete
        // real-coordinate blocks, never infer coefficients by evaluating unknowns.
        let coefficient = |row: &eqiora_ir::BoundAffineScalarIr<ScalarSymbolCoordinate>,
                           index: Option<usize>| {
            index.map_or(0., |index| row.coefficients()[index])
        };
        if self.column.is_some() && self.input_imaginary.is_none() {
            for column in &self.row.columns {
                if column.scalar_domain == eqiora_core::ScalarDomain::Real && imaginary.is_none() {
                    continue;
                }
                // A real-valued projection of a complex trial has a zero
                // imaginary row, not an absent complex-linearity obligation.
                let imaginary_coefficient =
                    |index| imaginary.map_or(0., |row| coefficient(row, index));
                if coefficient(real, column.imaginary) != -imaginary_coefficient(column.real)
                    || imaginary_coefficient(column.imaginary) != coefficient(real, column.real)
                {
                    return Err(super::super::invalid(
                        "nonholomorphic spatial component action requires a real-coordinate realization",
                    ));
                }
            }
        }
        let project =
            |row: &eqiora_ir::BoundAffineScalarIr<ScalarSymbolCoordinate>| match self.column {
                Some(column) => coefficient(
                    row,
                    if self.input_imaginary == Some(true) {
                        self.row.columns[column].imaginary
                    } else {
                        self.row.columns[column].real
                    },
                ),
                None => row.offsets()[0],
            };
        S::from_parts(project(real), imaginary.map_or(0., project))
            .ok_or_else(|| super::super::invalid("affine value exceeds its numeric scalar domain"))
    }
}
