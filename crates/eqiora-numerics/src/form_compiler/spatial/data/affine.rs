//! Retained component coefficients use the scalar IR's structural affine proof.
use super::*;
use crate::form_compiler::bilinear::BasisComponent;
use eqiora_core::ScalarDomain;
use eqiora_ir::{ComponentScalarRow, ComponentScalarization, ScalarSymbolCoordinate};
use eqiora_schema::kernel::KernelNode;

pub(in crate::form_compiler) struct ComponentRow<S: Coefficient> {
    pub(in crate::form_compiler) offset: Data<S>,
    pub(in crate::form_compiler) terms: Vec<(RawId, BasisComponent, Data<S>)>,
}

#[derive(Debug, Clone, PartialEq)]
struct Column {
    field: RawId,
    component: BasisComponent,
    real: Option<usize>,
    imaginary: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
struct Row<S: Coefficient> {
    rows: Vec<ComponentScalarRow>,
    selected: Vec<ScalarSymbolCoordinate>,
    bindings: Vec<(ScalarSymbolCoordinate, Input<S>)>,
    columns: Vec<Column>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct AffineData<S: Coefficient> {
    row: Arc<Row<S>>,
    column: Option<usize>,
    tangent: Option<Vec<Input<S>>>,
}

impl<S: Coefficient> Context<'_, S> {
    pub(in crate::form_compiler) fn affine_components(
        &self,
        id: ExprId,
        depth: usize,
    ) -> Result<Option<Vec<ComponentRow<S>>>, Diagnostic> {
        let typed = super::super::super::scalar::typed_relation(self.program, self.owner)?;
        if typed.expression() != self.dag {
            return Err(super::super::invalid(
                "affine components require their exact typed expression",
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
                if columns.last().is_none_or(|column: &Column| {
                    column.field != field.erase() || column.component != component
                }) {
                    columns.push(Column {
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
        let mut result = Vec::new();
        for rows in lowered.rows().chunks(parts) {
            let row = Arc::new(Row {
                rows: rows.to_vec(),
                selected: selected.clone(),
                bindings: bindings.clone(),
                columns: columns.clone(),
            });
            let data = |column| {
                Data(Arc::new(Node::Affine(AffineData {
                    row: Arc::clone(&row),
                    column,
                    tangent: None,
                })))
            };
            result.push(ComponentRow {
                offset: data(None),
                terms: columns
                    .iter()
                    .enumerate()
                    .map(|(index, column)| (column.field, column.component, data(Some(index))))
                    .collect(),
            });
        }
        Ok(Some(result))
    }
}

impl<S: Coefficient> AffineData<S> {
    pub(super) fn constant_in_direction(
        &self,
        direction: Direction<'_>,
    ) -> Result<bool, Diagnostic> {
        for (index, (coordinate, input)) in self.row.bindings.iter().enumerate() {
            if !input.constant_in_direction(coordinate, direction)? {
                return Ok(false);
            }
            if let Some(values) = &self.tangent
                && !values[index].constant_in_direction(coordinate, direction)?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(super) fn spatial(&self) -> bool {
        self.row.bindings.iter().any(|(_, input)| input.spatial())
    }
    pub(super) fn bind_parameter_point(
        &self,
        point: &crate::parameter_point::ParameterPoint,
    ) -> Result<Self, Diagnostic> {
        let mut row = (*self.row).clone();
        row.bindings = row
            .bindings
            .iter()
            .map(|(coordinate, input)| {
                input
                    .bind(coordinate, point)
                    .map(|input| (coordinate.clone(), input))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            row: Arc::new(row),
            column: self.column,
            tangent: self
                .tangent
                .as_ref()
                .map(|inputs| {
                    inputs
                        .iter()
                        .zip(&self.row.bindings)
                        .map(|(input, (coordinate, _))| input.bind(coordinate, point))
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
            tangent: Some(
                self.row
                    .bindings
                    .iter()
                    .map(|(coordinate, input)| input.derivative(coordinate, direction, dimension))
                    .collect::<Result<_, _>>()?,
            ),
        })
    }
    pub(super) fn evaluate(&self, point: &[f64]) -> Result<S, Diagnostic> {
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
                        .evaluate(point)
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
        if let Some(imaginary) = imaginary {
            for column in &self.row.columns {
                if coefficient(real, column.imaginary) != -coefficient(imaginary, column.real)
                    || coefficient(imaginary, column.imaginary) != coefficient(real, column.real)
                {
                    return Err(super::super::invalid(
                        "nonholomorphic spatial component action requires a real-coordinate realization",
                    ));
                }
            }
        }
        let project =
            |row: &eqiora_ir::BoundAffineScalarIr<ScalarSymbolCoordinate>| match self.column {
                Some(column) => coefficient(row, self.row.columns[column].real),
                None => row.offsets()[0],
            };
        let mut result = <S as From<f64>>::from(project(real));
        if let Some(imaginary) = imaginary {
            result += S::imaginary_unit().expect("admitted complex scalar domain")
                * <S as From<f64>>::from(project(imaginary));
        }
        Ok(result)
    }
}
