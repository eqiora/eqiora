//! Metadata-only expansion of an explicit global shared/mapped partition.

use eqiora_core::{Id, entity::kinds};

use super::*;

impl EvaluationMapPlan {
    /// Admit dense mapped inputs and explicitly shared Parameter coordinates.
    ///
    /// `shared_values` follows `shared_inputs`; all remaining inputs follow
    /// Program order in `mapped_values`, shaped `point_shape + [mapped_count]`.
    /// Point axes are row-major, rank at most 32. Empty shape means one point;
    /// zero extents mean no points. Sharing applies to the entire point grid,
    /// not selected individual axes. Shared values are validated even when empty.
    ///
    /// # Errors
    /// Rejects foreign/duplicate coordinates, nonfinite values, wrong shapes,
    /// unaddressable products and excess retained bytes before expanding points.
    pub fn from_partition(
        program: Arc<DifferentiableProgram>,
        shared_inputs: &[Id<kinds::Parameter>],
        shared_values: &[f64],
        mapped_values: &[f64],
        point_shape: &[usize],
        policy: EvaluationMapExecutionPolicy,
    ) -> Result<Self, Diagnostic> {
        if point_shape.len() > 32 {
            return Err(invalid("mapped input point rank exceeds 32"));
        }
        let count = axes::product(point_shape)
            .map_err(|error| invalid(format!("invalid mapped input point shape: {error:?}")))?;
        let (shared, mapped) = coordinates(program.identity(), shared_inputs)?;
        let mut shape = point_shape.to_vec();
        shape.push(mapped.len());
        let expected = axes::product(&shape).map_err(|error| {
            invalid(format!("invalid mapped input coordinate shape: {error:?}"))
        })?;
        if shared_values.len() != shared.len() || mapped_values.len() != expected {
            return Err(invalid(format!(
                "partition inputs require {} shared values and {expected} mapped values",
                shared.len()
            )));
        }
        for (index, value) in shared_values.iter().enumerate() {
            if !value.is_finite() {
                return Err(invalid(format!("shared input {index} must be finite")));
            }
        }
        for (index, value) in mapped_values.iter().enumerate() {
            if !value.is_finite() {
                return Err(invalid(format!(
                    "mapped input at occurrence {}, coordinate {} must be finite",
                    index / mapped.len(),
                    index % mapped.len()
                )));
            }
        }
        program.validate_map_provider()?;
        let estimated_storage_bytes = resources::estimate(&program, count, policy, 0)?;
        let mut points = Vec::with_capacity(count);
        for occurrence in 0..count {
            let mut values = vec![0.0; program.identity().input_dimension()];
            for (&input, &value) in shared.iter().zip(shared_values) {
                values[input] = value;
            }
            for (coordinate, &input) in mapped.iter().enumerate() {
                values[input] = mapped_values[occurrence * mapped.len() + coordinate];
            }
            program.validate_map_point(&values)?;
            points.push(program.map_point(&values));
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
        let input = identity
            .inputs()
            .iter()
            .find(|input| input.parameter() == *id)
            .ok_or_else(|| invalid("shared input is foreign to the exact Program"))?;
        shared.extend(input.coordinates());
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
