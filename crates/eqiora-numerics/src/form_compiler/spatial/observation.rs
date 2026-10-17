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

impl PrescribedFieldData {
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
