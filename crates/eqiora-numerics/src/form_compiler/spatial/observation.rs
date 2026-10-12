//! Prescribed scalar Fields use the same coefficient meaning in solves and observations.
use super::*;
use eqiora_core::ValueLiteral;
use eqiora_graph::EdgeKind;

pub(crate) struct PrescribedFieldData {
    fields: BTreeMap<RawId, PrescribedField>,
}

struct PrescribedField {
    value_type: ValueType,
    data: Data<f64>,
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
                require_scalar::<f64>(value_type)?;
                let gradient = (0..dimension)
                    .map(|axis| data.coordinate_derivative(axis, dimension))
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
            let value = ValueLiteral::from_real(value_type.clone(), data.evaluate(point)?)
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
