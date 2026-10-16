//! Metadata-only expansion of an explicit global shared/mapped partition.

#[cfg(test)]
use eqiora_core::Id;

use super::*;

impl EvaluationMapPlan {
    /// Compose typed shared inputs with ordered typed mapped occurrences.
    ///
    /// Shared leaves may appear in any order. Each mapped product must contain
    /// exactly the remaining Program inputs in Program order. An empty mapped
    /// inventory has no occurrences; all-shared maps use one empty product per
    /// occurrence. The Plan retains this sharing declaration for every derivative view.
    /// Grid axes belong to the consuming map/product projection.
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
        let shared_inputs = selected_parameters(program.identity(), shared_values)?;
        let (shared, mapped) = coordinates(program.identity(), &shared_inputs)?;
        let (_, mapped_leaves) = leaf_indices(program.identity(), &shared_inputs)?;
        let mapped_signatures = mapped_leaves
            .iter()
            .map(|&index| inputs[index].clone())
            .collect::<Vec<_>>();
        for point in mapped_values {
            if point.signatures() != mapped_signatures {
                return Err(invalid(
                    "mapped values differ from the remaining ordered Program input signatures",
                ));
            }
        }
        program.validate_map_execution()?;
        let estimated_storage_bytes = resources::estimate(
            &program,
            mapped_values.len(),
            policy,
            0,
            shared_inputs.len(),
        )?;
        let mut points = Vec::with_capacity(mapped_values.len());
        for point in mapped_values {
            let values = assemble_coordinates(
                program.identity().input_dimension(),
                &shared,
                &mapped,
                shared_values.coordinates(),
                point.coordinates(),
            );
            points.push(
                program
                    .point_from_coordinates(&values)
                    .expect("validated native point"),
            );
        }
        Ok(Self {
            inputs: Arc::new(super::Inputs::Evaluations {
                program,
                points: points.into(),
                samples: None,
                shared_inputs: shared_inputs.into(),
            }),
            policy,
            estimated_storage_bytes,
        })
    }
}

/// One exact leaf-selection check for fixed sampling values and shared map inputs.
pub(crate) fn selected_parameters(
    identity: &DifferentiableProgramIdentity,
    values: &DifferentiableValues,
) -> Result<Vec<Parameter>, Diagnostic> {
    if values
        .signatures()
        .iter()
        .any(|signature| !identity.inputs().contains(signature))
    {
        return Err(invalid(
            "selected value differs from the exact Program input signature",
        ));
    }
    Ok(values
        .signatures()
        .iter()
        .map(|signature| signature.parameter())
        .collect())
}

/// Assemble an already admitted whole-leaf partition into its native coordinates.
pub(crate) fn assemble_coordinates(
    width: usize,
    shared: &[usize],
    mapped: &[usize],
    fixed: &[f64],
    varying: &[f64],
) -> Vec<f64> {
    let mut coordinates = vec![0.0; width];
    for (&index, &value) in shared.iter().zip(fixed).chain(mapped.iter().zip(varying)) {
        coordinates[index] = value;
    }
    coordinates
}

/// Validate a declaration against exact complete points before any execution.
pub(super) fn admit_shared<'a>(
    identity: &DifferentiableProgramIdentity,
    mut points: impl Iterator<Item = &'a DifferentiableValues>,
    shared_inputs: &[Parameter],
) -> Result<(), Diagnostic> {
    let (shared, _) = coordinates(identity, shared_inputs)?;
    if let Some(first) = points.next() {
        for point in points {
            if shared.iter().any(|&input| {
                point.coordinates()[input].to_bits() != first.coordinates()[input].to_bits()
            }) {
                return Err(invalid(
                    "explicit shared input differs between planned points",
                ));
            }
        }
    }
    Ok(())
}

/// Select whole semantic leaves independently of their coordinate representation.
pub(super) fn leaf_indices(
    identity: &DifferentiableProgramIdentity,
    shared_inputs: &[Parameter],
) -> Result<(Vec<usize>, Vec<usize>), Diagnostic> {
    let mut shared = Vec::with_capacity(shared_inputs.len());
    for (position, id) in shared_inputs.iter().enumerate() {
        if shared_inputs[..position].contains(id) {
            return Err(invalid("shared input selection repeats a Parameter"));
        }
        let index = identity
            .inputs()
            .iter()
            .position(|input| input.parameter() == *id)
            .ok_or_else(|| invalid("shared input is foreign to the exact Program"))?;
        shared.push(index);
    }
    let mapped = (0..identity.inputs().len())
        .filter(|index| !shared.contains(index))
        .collect();
    Ok((shared, mapped))
}

/// Derive storage ranges from the semantic leaf selection, never the reverse.
pub(crate) fn coordinates(
    identity: &DifferentiableProgramIdentity,
    shared_inputs: &[Parameter],
) -> Result<(Vec<usize>, Vec<usize>), Diagnostic> {
    let (shared, mapped) = leaf_indices(identity, shared_inputs)?;
    let ranges = crate::differentiation::signature::layout(identity.inputs())
        .map(|(_, range)| range)
        .collect::<Vec<_>>();
    let expand = |leaves: Vec<usize>| {
        leaves
            .into_iter()
            .flat_map(|index| ranges[index].clone())
            .collect()
    };
    Ok((expand(shared), expand(mapped)))
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
            leaf_indices(identity, &[c.id()]).unwrap(),
            (vec![1], vec![0])
        );
        assert_eq!(
            leaf_indices(identity, &[c.id(), p.id()]).unwrap(),
            (vec![1, 0], vec![])
        );
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
