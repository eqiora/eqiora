//! Metadata-only expansion of an explicit global shared/mapped partition.

use eqiora_core::{Id, entity::kinds};

use super::*;

impl EvaluationMapPlan {
    /// Compose typed shared inputs with ordered typed mapped occurrences.
    ///
    /// Shared leaves may appear in any order. Each mapped product must contain
    /// exactly the remaining Program inputs in Program order. An empty mapped
    /// inventory has no occurrences; all-shared maps use one empty product per
    /// occurrence. Grid axes belong to the consuming map/product projection.
    ///
    /// # Errors
    /// Rejects foreign signatures and incomplete or reordered mapped products,
    /// including invalid shared signatures in an empty map, before execution.
    pub fn from_partition(
        program: Arc<DifferentiableProgram>,
        shared_values: &DifferentiableValues,
        mapped_values: &[DifferentiableValues],
        policy: EvaluationMapExecutionPolicy,
    ) -> Result<Self, Diagnostic> {
        let inputs = program.identity().inputs();
        for signature in shared_values.signatures() {
            if !inputs.contains(signature) {
                return Err(invalid(
                    "shared value differs from the exact Program input signature",
                ));
            }
        }
        let shared_inputs = shared_values
            .signatures()
            .iter()
            .map(|value| {
                value
                    .entity()
                    .downcast()
                    .expect("Program Parameter signature")
            })
            .collect::<Vec<_>>();
        let (shared, mapped) = coordinates(program.identity(), &shared_inputs)?;
        let mapped_signatures = inputs
            .iter()
            .filter(|value| !shared_values.signatures().contains(value))
            .cloned()
            .collect::<Vec<_>>();
        for point in mapped_values {
            if point.signatures() != mapped_signatures {
                return Err(invalid(
                    "mapped values differ from the remaining ordered Program input signatures",
                ));
            }
        }
        program.validate_map_execution()?;
        let estimated_storage_bytes =
            resources::estimate(&program, mapped_values.len(), policy, 0)?;
        let mut points = Vec::with_capacity(mapped_values.len());
        for point in mapped_values {
            let mut values = vec![0.0; program.identity().input_dimension()];
            for (&input, &value) in shared.iter().zip(shared_values.coordinates()) {
                values[input] = value;
            }
            for (&input, &value) in mapped.iter().zip(point.coordinates()) {
                values[input] = value;
            }
            points.push(
                program
                    .point_from_coordinates(&values)
                    .expect("validated native point"),
            );
        }
        Ok(Self {
            program,
            points: points.into(),
            samples: None,
            policy,
            estimated_storage_bytes,
        })
    }
}

/// Resolve whole semantic inputs to their shared real-coordinate storage ranges.
pub(super) fn coordinates(
    identity: &crate::DifferentiableProgramIdentity,
    shared_inputs: &[Id<kinds::Parameter>],
) -> Result<(Vec<usize>, Vec<usize>), Diagnostic> {
    let mut shared = Vec::new();
    for (position, id) in shared_inputs.iter().enumerate() {
        if shared_inputs[..position].contains(id) {
            return Err(invalid("shared input selection repeats a Parameter"));
        }
        let (_, range) = crate::differentiation::signature::layout(identity.inputs())
            .find(|(input, _)| input.parameter() == *id)
            .ok_or_else(|| invalid("shared input is foreign to the exact Program"))?;
        shared.extend(range);
    }
    let mapped = (0..identity.input_dimension())
        .filter(|index| !shared.contains(index))
        .collect();
    Ok((shared, mapped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_numerics::{
        CommonAlgebraicPlan, CommonLinearRequest, CommonSolvePolicy, ResolvedCommonPlan,
    };
    use eqiora_solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};
    use std::num::NonZeroUsize;

    #[test]
    fn sharing_a_shaped_complex_parameter_selects_every_component_and_part() {
        let document = crate::ModelDocument::compile("partition.eqi",
            "model M(){parameter p:1=3;parameter c:array<complex<1>,2>=[math.complex(1,2),math.complex(3,-4)];variable z:array<complex<1>,2>;relation r{z=p*c;}observable output:1=math.abs2(z[0])+math.abs2(z[1]);}"
        ).unwrap();
        let model = eqiora_artifact::ModelEnvelope::from_program(document.program()).unwrap();
        let solver = SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1e-13,
            1e-15,
            NonZeroUsize::new(8).unwrap(),
        )
        .unwrap();
        let plan = CommonAlgebraicPlan::resolve(
            &model,
            CommonSolvePolicy::Linear(
                CommonLinearRequest::exact(solver, REFERENCE_LINEAR_SOLVER.provider()).unwrap(),
            ),
            None,
            &[],
            None,
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
        let p = document.parameter_ref("p").unwrap();
        let c = document.parameter_ref("c").unwrap();
        let program = DifferentiableProgram::compile(
            ResolvedCommonPlan::Algebraic(Box::new(plan)),
            &[p.clone(), c.clone()],
            &[document.observable_ref("output").unwrap().erase()],
            None,
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
        let identity = program.identity();
        assert_eq!(
            coordinates(identity, &[c.id()]).unwrap(),
            (vec![1, 2, 3, 4], vec![0])
        );
        assert_eq!(
            coordinates(identity, &[c.id(), p.id()]).unwrap(),
            (vec![1, 2, 3, 4, 0], vec![])
        );
        assert_eq!(
            coordinates(identity, &[]).unwrap(),
            (vec![], vec![0, 1, 2, 3, 4])
        );
        assert!(coordinates(identity, &[c.id(), c.id()]).is_err());
        assert!(coordinates(identity, &[Id::new()]).is_err());
    }
}
