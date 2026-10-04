//! One Operator IR lowering for original equality and Observable partial actions.
use super::*;
use eqiora_core::ValueLiteral;
use eqiora_ir::{DifferentiationRole, LinearizedRelation, RelationTangent, ScalarOperatorIr};
use eqiora_schema::kernel::KernelNode;

pub(crate) struct ExpressionLinearization {
    pub values: Vec<f64>,
    pub unknown_jacobian: Vec<f64>,
    pub parameter_jacobian: Vec<f64>,
}

impl FiniteConstraintProblem {
    pub(crate) fn linearize_expression(
        &self,
        expression: &ExprDag,
        values: &[f64],
        selected: &[Id<kinds::Parameter>],
    ) -> Result<ExpressionLinearization, Diagnostic> {
        let n = self.symbols.len();
        if values.len() != n || values.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "finite linearization requires complete finite Field coordinates",
            ));
        }
        for (index, id) in selected.iter().enumerate() {
            if selected[..index].contains(id)
                || !matches!(self.kernel.node(id.erase()), Some(KernelNode::Parameter(_)))
            {
                return Err(invalid(
                    "finite linearization requires unique exact Model Parameters",
                ));
            }
        }
        let expression = super::observables::expand(&self.kernel, expression)?;
        let operator = ScalarOperatorIr::lower(&expression)?;
        let mut inputs = Vec::new();
        let mut roles = Vec::new();
        let mut unknown_coordinates = Vec::new();
        let mut parameter_coordinates = Vec::new();
        for symbol in operator.symbols() {
            match symbol {
                SymbolRef::Field(id) => {
                    let coordinate = self
                        .symbols
                        .iter()
                        .position(|value| value == symbol)
                        .ok_or_else(|| invalid("finite expression contains a foreign Field"))?;
                    let Some(KernelNode::Field(field)) = self.kernel.node(id.erase()) else {
                        return Err(invalid("finite Field is absent from its Model"));
                    };
                    inputs.push(
                        ValueLiteral::from_real(field.value_type().clone(), values[coordinate])
                            .map_err(|error| invalid(error.to_string()))?,
                    );
                    roles.push(DifferentiationRole::Unknown);
                    unknown_coordinates.push(coordinate);
                }
                SymbolRef::Parameter(id) => {
                    let Some(KernelNode::Parameter(parameter)) = self.kernel.node(id.erase())
                    else {
                        return Err(invalid("finite Parameter is absent from its Model"));
                    };
                    inputs.push(
                        self.parameter_candidates
                            .iter()
                            .find(|(candidate, _)| candidate == id)
                            .map_or_else(|| parameter.value().clone(), |(_, value)| value.clone()),
                    );
                    if let Some(coordinate) = selected.iter().position(|candidate| candidate == id)
                    {
                        roles.push(DifferentiationRole::Parameter);
                        parameter_coordinates.push(coordinate);
                    } else {
                        roles.push(DifferentiationRole::Frozen);
                    }
                }
                _ => {
                    return Err(invalid(
                        "finite expressions require static Field/Parameter coordinates",
                    ));
                }
            }
        }
        let linearized = operator.linearize_typed(&inputs, &roles)?;
        let rows = expression.roots().len();
        let mut primal = vec![0.0; rows];
        linearized.primal(&mut primal)?;
        let mut unknown_jacobian = vec![0.0; rows * n];
        let mut parameter_jacobian = vec![0.0; rows * selected.len()];
        for (coordinates, width, matrix, parameter_role) in [
            (&unknown_coordinates, n, &mut unknown_jacobian, false),
            (
                &parameter_coordinates,
                selected.len(),
                &mut parameter_jacobian,
                true,
            ),
        ] {
            for (local, coordinate) in coordinates.iter().enumerate() {
                let mut direction = vec![0.0; coordinates.len()];
                direction[local] = 1.0;
                let mut column = vec![0.0; rows];
                let tangent = if parameter_role {
                    RelationTangent::Parameter(&direction)
                } else {
                    RelationTangent::Unknown(&direction)
                };
                linearized.jvp(tangent, &mut column)?;
                for (row, value) in column.into_iter().enumerate() {
                    matrix[row * width + coordinate] = value;
                }
            }
        }
        Ok(ExpressionLinearization {
            values: primal,
            unknown_jacobian,
            parameter_jacobian,
        })
    }
}
