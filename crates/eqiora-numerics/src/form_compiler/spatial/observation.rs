//! Prescribed typed Fields use the same coefficient meaning in solves and observations.
use super::*;
use eqiora_core::ValueLiteral;
use eqiora_graph::EdgeKind;

pub(crate) struct PrescribedFieldData {
    fields: BTreeMap<RawId, PrescribedField>,
}

struct PrescribedField {
    value_type: ValueType,
    data: Vec<Data<f64>>,
    gradient: Vec<Data<f64>>,
}

/// Derive once per selected Parameter, then reuse at every quadrature point.
pub(crate) struct PrescribedFieldPartials<'a> {
    source: &'a PrescribedFieldData,
    parameters: &'a [eqiora_ir::ScalarSymbolCoordinate],
    dimension: usize,
    derivatives: Vec<PrescribedFieldData>,
}

impl PrescribedFieldPartials<'_> {
    /// Activity follows source identities, never a sampled zero derivative.
    pub(crate) fn active(
        &self,
        field: RawId,
        component: usize,
        axis: Option<usize>,
    ) -> Result<bool, Diagnostic> {
        let data = self
            .source
            .coordinate(field, component, axis, self.dimension)?;
        for parameter in self.parameters {
            if !data.constant_in_direction(&|coordinate| f64::from(coordinate == parameter))? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) fn pullback(
        &self,
        point: &[f64],
        field: RawId,
        component: usize,
        axis: Option<usize>,
        weight: f64,
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if output.len() != self.derivatives.len() {
            return Err(invalid(
                "prescribed Field partial has inconsistent Parameters",
            ));
        }
        for (partial, output) in self.derivatives.iter().zip(output) {
            let data = partial.coordinate(field, component, axis, self.dimension)?;
            *output += weight * data.evaluate(point)?;
            if !output.is_finite() {
                return Err(invalid("prescribed Field pullback is non-finite"));
            }
        }
        Ok(())
    }
}

impl PrescribedFieldData {
    fn coordinate(
        &self,
        field: RawId,
        component: usize,
        axis: Option<usize>,
        dimension: usize,
    ) -> Result<&Data<f64>, Diagnostic> {
        let field = self
            .fields
            .get(&field)
            .ok_or_else(|| invalid("prescribed Field partial is outside the observation"))?;
        match axis {
            None => field.data.get(component),
            Some(axis) if axis < dimension => field.gradient.get(component * dimension + axis),
            _ => None,
        }
        .ok_or_else(|| invalid("prescribed Field partial coordinate is unavailable"))
    }

    pub(crate) fn parameter_partials<'a>(
        &'a self,
        parameters: &'a [eqiora_ir::ScalarSymbolCoordinate],
        dimension: usize,
    ) -> Result<PrescribedFieldPartials<'a>, Diagnostic> {
        let derivatives = parameters
            .iter()
            .map(|parameter| {
                self.fields
                    .iter()
                    .map(|(&id, field)| {
                        let differentiate = |data: &Data<f64>| {
                            data.derivative(
                                &|coordinate| f64::from(coordinate == parameter),
                                dimension,
                            )
                        };
                        Ok((
                            id,
                            PrescribedField {
                                value_type: field.value_type.clone(),
                                data: field
                                    .data
                                    .iter()
                                    .map(differentiate)
                                    .collect::<Result<_, _>>()?,
                                gradient: field
                                    .gradient
                                    .iter()
                                    .map(differentiate)
                                    .collect::<Result<_, _>>()?,
                            },
                        ))
                    })
                    .collect::<Result<_, Diagnostic>>()
                    .map(|fields| Self { fields })
            })
            .collect::<Result<_, _>>()?;
        Ok(PrescribedFieldPartials {
            source: self,
            parameters,
            dimension,
            derivatives,
        })
    }

    pub(crate) fn bind_parameter_point(
        &mut self,
        point: &crate::parameter_point::ParameterPoint,
    ) -> Result<(), Diagnostic> {
        for field in self.fields.values_mut() {
            for value in field.data.iter_mut().chain(&mut field.gradient) {
                *value = value.bind_parameter_point(point)?;
            }
        }
        Ok(())
    }

    pub(crate) fn derive(
        program: &KernelProgram,
        requested: &BTreeSet<RawId>,
        dimension: usize,
    ) -> Result<Self, Diagnostic> {
        let domains = program
            .edges()
            .iter()
            .filter(|edge| edge.kind() == EdgeKind::DefinedOn && requested.contains(&edge.from()))
            .map(|edge| edge.to())
            .filter(|id| matches!(program.node(*id), Some(KernelNode::Domain(_))))
            .collect::<BTreeSet<_>>();
        let mut fields = BTreeMap::new();
        for domain in domains {
            let roles = EquationRoles::derive(program, [domain])?;
            let known = coefficients::<f64>(program, dimension, &roles)?;
            for (field, data) in known {
                if !requested.contains(&field) {
                    continue;
                }
                let value_type = &roles.fields[&field].1;
                let data = eqiora_ir::ScalarSymbolCoordinate::for_value(
                    SymbolRef::Field(field.downcast().expect("Field identity")),
                    value_type,
                )?
                .iter()
                .map(|coordinate| {
                    data.coordinate(
                        coordinate.component_index(),
                        coordinate.is_imaginary(),
                        dimension,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
                let gradient = data
                    .iter()
                    .flat_map(|coordinate| {
                        (0..dimension)
                            .map(move |axis| coordinate.coordinate_derivative(axis, dimension))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                fields.insert(
                    field,
                    PrescribedField {
                        value_type: value_type.clone(),
                        data,
                        gradient,
                    },
                );
            }
        }
        Ok(Self { fields })
    }

    /// Prescribed coefficients are held fixed under State variation.
    pub(crate) fn sample(
        &self,
        point: &[f64],
        mut insert: impl FnMut(RawId, ValueLiteral, Vec<f64>),
    ) -> Result<(), Diagnostic> {
        for (
            &field,
            PrescribedField {
                value_type,
                data,
                gradient,
            },
        ) in &self.fields
        {
            let width = if value_type.scalar_domain() == eqiora_core::ScalarDomain::Complex {
                2
            } else {
                1
            };
            let values = data
                .iter()
                .map(|data| data.evaluate(point))
                .collect::<Result<Vec<_>, _>>()?;
            let value = ValueLiteral::new(
                value_type.clone(),
                values
                    .chunks_exact(width)
                    .map(|part| (part[0], part.get(1).copied().unwrap_or(0.))),
            )
            .map_err(|error| invalid(&error.to_string()))?;
            let gradient = gradient
                .iter()
                .map(|data| data.evaluate(point))
                .collect::<Result<Vec<_>, _>>()?;
            insert(field, value, gradient);
        }
        Ok(())
    }
}
