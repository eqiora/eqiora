//! One Operator IR lowering for original equality and Observable partial actions.
use super::*;
use eqiora_core::ScalarDomain;
use eqiora_ir::{ComponentScalarization, DifferentiationRole, LinearizedRelation, RelationTangent};
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
        let n = self.coordinate_count();
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
        for id in selected {
            let Some(KernelNode::Parameter(parameter)) = self.kernel.node(id.erase()) else {
                unreachable!("validated Parameter");
            };
            if parameter.value_type().scalar_domain() != ScalarDomain::Real
                || !parameter.value_type().shape().is_scalar()
            {
                return Err(invalid(
                    "finite Parameter-point differentiation requires real scalar selected Parameters",
                ));
            }
        }
        let expression = super::observables::expand(&self.kernel, expression)?;
        let typed = coordinates::typed_expression(&self.kernel, &expression)?;
        let operator = ComponentScalarization::lower(&typed)?;
        let rows = operator.rows().len();
        let linearized = operator.linearize(|coordinate| match coordinate.symbol() {
            SymbolRef::Field(_) => self
                .coordinates
                .iter()
                .position(|candidate| candidate == coordinate)
                .map(|index| (values[index], DifferentiationRole::Unknown)),
            SymbolRef::Parameter(id) => {
                let Some(KernelNode::Parameter(parameter)) = self.kernel.node(id.erase()) else {
                    return None;
                };
                let value = self
                    .parameter_candidates
                    .iter()
                    .find(|(candidate, _)| *candidate == id)
                    .map_or(parameter.value(), |(_, value)| value);
                let role = if selected.contains(&id) {
                    DifferentiationRole::Parameter
                } else {
                    DifferentiationRole::Frozen
                };
                coordinates::component(value, coordinate)
                    .ok()
                    .map(|value| (value, role))
            }
            _ => None,
        })?;
        // Match retained semantic coordinates, never expression traversal order.
        let unknown_coordinates = linearized
            .unknown_coordinates()
            .iter()
            .map(|coordinate| {
                self.coordinates
                    .iter()
                    .position(|candidate| candidate == coordinate)
                    .expect("bound Field coordinate")
            })
            .collect::<Vec<_>>();
        let parameter_coordinates = linearized
            .parameter_coordinates()
            .iter()
            .map(|coordinate| {
                selected
                    .iter()
                    .position(|id| coordinate.symbol() == SymbolRef::Parameter(*id))
                    .expect("bound Parameter coordinate")
            })
            .collect::<Vec<_>>();
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

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_graph::{GraphStore, InMemoryGraphStore};

    #[test]
    fn finite_component_partials_keep_nonholomorphic_coordinates_and_frozen_inputs() {
        // R=(1+2i)z+conj(z)-(4+2i)p gives R=(2x-2y-4p,2x-2p).
        // At p=3,z=3-3i the residual is zero and J=|z|^2+p=21.
        let source = "model M(){parameter p:1=3;parameter a:complex<1>=math.complex(1,2);variable z:complex<1>;relation r{a*z+math.conj(z)=math.complex(4,2)*p;}observable j:1=math.abs2(z)+p;}";
        let compiled = eqiora_compiler::compile("component-partials.eqi", source)
            .unwrap()
            .pop()
            .unwrap();
        let p = compiled.symbols().get("p").unwrap().downcast().unwrap();
        let j = compiled.symbols().get("j").unwrap().downcast().unwrap();
        let (transaction, model, _) = compiled.into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let kernel = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let problem = lower_finite_constraints(&kernel, None).unwrap();
        assert_eq!(problem.symbols().len(), 1);
        assert_eq!(problem.coordinate_count(), 2);
        let (actions, _) = problem.equality_jacobian(&[3., -3.], &[p]).unwrap();
        assert_eq!(actions.values, [0., 0.]);
        assert_eq!(actions.unknown_jacobian, [2., -2., 2., 0.]);
        assert_eq!(actions.parameter_jacobian, [-4., -2.]);
        let objective = kernel.typed_observable(j).unwrap();
        let actions = problem
            .linearize_expression(objective.expression(), &[3., -3.], &[p])
            .unwrap();
        assert_eq!(actions.values, [21.]);
        assert_eq!(actions.unknown_jacobian, [6., -6.]);
        assert_eq!(actions.parameter_jacobian, [1.]);
        assert!(problem.equality_jacobian(&[3.], &[p]).is_err());
        // A new parameter point changes both affine assembly and independent
        // original operands, while frozen complex a remains exactly 1+2i.
        for (point, expected) in [
            (problem.at_parameters(&[p], &[5.]).unwrap(), 5.),
            (
                problem
                    .at_parameters(&[p], &[5.])
                    .unwrap()
                    .at_parameters(&[], &[])
                    .unwrap(),
                3.,
            ),
        ] {
            let assembled = solve::branch_system(&point, 0).unwrap();
            assert_eq!(assembled.right_hand_side(), &[4. * expected, 2. * expected]);
            assert_eq!(
                point.original_residual(&[expected, -expected]).unwrap(),
                [0., 0.]
            );
            let actions = point
                .linearize_expression(objective.expression(), &[expected, -expected], &[p])
                .unwrap();
            assert_eq!(actions.values, [2. * expected * expected + expected]);
        }
        assert_eq!(problem.original_residual(&[3., -3.]).unwrap(), [0., 0.]);
    }
}
