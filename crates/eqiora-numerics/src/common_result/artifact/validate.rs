//! Fail-closed semantic validation for decoded common Result content.

use eqiora_core::{Diagnostic, DimExponents};
use eqiora_solver::{
    ExecutionProvider, ProviderLibrary, SERIAL_EXECUTION_PROVIDER, SolverProvider,
};

use super::{
    CommonAssemblyEvidence, CommonExecutionTopology, CommonFieldAssociation,
    CommonProviderEvidence, CommonResultField, CommonSolveEvidence, CommonTrajectory,
    ResolvedCommonPlan, WireResultFamily, invalid,
};

pub(super) fn validate_fields(
    plan: &ResolvedCommonPlan,
    fields: &[CommonResultField],
) -> Result<(), Diagnostic> {
    let valid = match plan {
        ResolvedCommonPlan::Scalar(plan) => {
            let (space, association) = match plan.spatial() {
                crate::CommonSpatialPolicy::Q1 => {
                    ("continuous-lagrange-p1", CommonFieldAssociation::Vertex)
                }
                crate::CommonSpatialPolicy::CellCenteredTpfa => {
                    ("cell-constant", CommonFieldAssociation::Cell)
                }
                _ => {
                    return Err(invalid(
                        "scalar Result Plan has a non-scalar spatial policy",
                    ));
                }
            };
            fields.len() == plan.fields().len()
                && plan.fields().all(|(id, value_type)| {
                    let Ok((shape, _)) = plan.field_support(id.erase()) else {
                        return false;
                    };
                    let id = id.ulid().to_string();
                    fields
                        .iter()
                        .find(|field| field.field_id == id)
                        .is_some_and(|field| {
                            field_matches(
                                field,
                                &id,
                                value_type.dimension(),
                                &[],
                                space,
                                &[(association, shape.clone())],
                            )
                        })
                })
        }
        ResolvedCommonPlan::Elasticity(plan) => {
            let cells = plan.cells();
            let vertices = (cells[0] + 1) * (cells[1] + 1);
            fields.len() == 1
                && field_matches(
                    &fields[0],
                    plan.displacement_field_id(),
                    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension"),
                    &[2],
                    "continuous-lagrange-p1",
                    &[(CommonFieldAssociation::Vertex, vec![vertices, 2])],
                )
        }
        ResolvedCommonPlan::SteadyStokes(plan) => {
            let Some((vertices, cells)) = simplicial_counts(plan) else {
                return Err(invalid(
                    "steady-Stokes Result Plan omitted its simplicial Mesh",
                ));
            };
            fields.len() == 2
                && field_matches(
                    &fields[0],
                    plan.velocity_field_id(),
                    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension"),
                    &[2],
                    "simplex-p1-bubble",
                    &[
                        (CommonFieldAssociation::Vertex, vec![vertices, 2]),
                        (CommonFieldAssociation::CellBubble, vec![cells, 2]),
                    ],
                )
                && field_matches(
                    &fields[1],
                    plan.pressure_field_id(),
                    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0])
                        .expect("bounded dimension"),
                    &[],
                    "continuous-lagrange-p1",
                    &[(CommonFieldAssociation::Vertex, vec![vertices])],
                )
        }
        ResolvedCommonPlan::Algebraic(_)
        | ResolvedCommonPlan::Ode(_)
        | ResolvedCommonPlan::TransientFlow(_)
        | ResolvedCommonPlan::Fsi(_) => false,
    };
    if !valid {
        return Err(invalid(
            "Result Fields differ from the exact Plan, Mesh, space, or shape",
        ));
    }
    Ok(())
}

fn field_matches(
    field: &CommonResultField,
    id: &str,
    dimension: DimExponents,
    value_shape: &[usize],
    space: &str,
    blocks: &[(CommonFieldAssociation, Vec<usize>)],
) -> bool {
    field.field_id == id
        && field.dimension == dimension
        && field.value_shape == value_shape
        && field.space == space
        && field.blocks.len() == blocks.len()
        && field.blocks.iter().zip(blocks).all(|(actual, expected)| {
            actual.association == expected.0 && actual.logical_shape == expected.1
        })
}

fn simplicial_counts(plan: &crate::CommonSteadyStokesPlan) -> Option<(usize, usize)> {
    let resolved = ResolvedCommonPlan::SteadyStokes(Box::new(plan.clone()));
    resolved
        .authenticated_mesh()?
        .simplicial_mesh()
        .map(|mesh| {
            let mesh = mesh.mesh();
            (mesh.vertices().len(), mesh.cells().len())
        })
}

pub(super) fn require_family(
    plan: &ResolvedCommonPlan,
    family: WireResultFamily,
) -> Result<(), Diagnostic> {
    let matches = matches!(
        (plan, family),
        (
            ResolvedCommonPlan::Algebraic(_),
            WireResultFamily::Algebraic
        ) | (ResolvedCommonPlan::Scalar(_), WireResultFamily::Scalar)
            | (
                ResolvedCommonPlan::Elasticity(_),
                WireResultFamily::Elasticity
            )
            | (
                ResolvedCommonPlan::SteadyStokes(_),
                WireResultFamily::SteadyStokes
            )
            | (ResolvedCommonPlan::Ode(_), WireResultFamily::Ode)
            | (
                ResolvedCommonPlan::TransientFlow(_),
                WireResultFamily::TransientFlow
            )
            | (
                ResolvedCommonPlan::Fsi(_),
                WireResultFamily::FixedReferenceFsi
            )
    );
    if !matches {
        return Err(invalid("common Result family differs from the exact Plan"));
    }
    Ok(())
}

pub(super) fn require_trajectory_family(
    family: WireResultFamily,
    trajectory: &CommonTrajectory,
) -> Result<(), Diagnostic> {
    let matches = matches!(
        (family, trajectory),
        (WireResultFamily::Ode, CommonTrajectory::Ode { .. })
            | (
                WireResultFamily::TransientFlow | WireResultFamily::Scalar,
                CommonTrajectory::SpatialTransient { .. }
            )
            | (
                WireResultFamily::FixedReferenceFsi,
                CommonTrajectory::Fsi { .. }
            )
    );
    if !matches {
        return Err(invalid("Result family differs from its Trajectory"));
    }
    Ok(())
}

pub(in crate::common_result) fn require_plan_solver(
    plan: &ResolvedCommonPlan,
    solve: &CommonSolveEvidence,
) -> Result<(), Diagnostic> {
    let expected = plan
        .effective_solver()
        .ok_or_else(|| invalid("Result solve evidence requires a linear Plan"))?;
    let solver_provider = plan
        .linear_solver_provider()
        .ok_or_else(|| invalid("Result solve evidence requires a solver provider"))?;
    if !solver_provider_matches(&solve.solver, solver_provider)
        || solve.algorithm != expected.algorithm()
        || solve.preconditioner != expected.preconditioner()
        || solve.reduction != expected.reduction()
        || solve.relative_tolerance.to_bits() != expected.relative_tolerance().to_bits()
        || solve.absolute_tolerance.to_bits() != expected.absolute_tolerance().to_bits()
        || solve.maximum_iterations != expected.maximum_iterations().get()
    {
        return Err(invalid("Result solve evidence differs from the exact Plan"));
    }
    let (execution_provider, workers) = plan
        .linear_execution_provider()
        .ok_or_else(|| invalid("Result solve evidence requires an execution provider"))?;
    require_execution(&solve.execution, execution_provider, workers)?;
    require_execution(&solve.verification, SERIAL_EXECUTION_PROVIDER, 1)
}

fn require_execution(
    execution: &super::CommonExecutionEvidence,
    provider: ExecutionProvider,
    workers: usize,
) -> Result<(), Diagnostic> {
    let valid = execution_provider_matches(&execution.provider, provider)
        && execution.adapter == provider.id().as_str()
        && execution.topology == CommonExecutionTopology::Host { workers };
    if !valid {
        return Err(invalid(
            "Result execution evidence differs from the exact Plan",
        ));
    }
    Ok(())
}

pub(super) fn require_reference_assembly(
    assembly: &CommonAssemblyEvidence,
) -> Result<(), Diagnostic> {
    if assembly.adapter != SERIAL_EXECUTION_PROVIDER.id().as_str()
        || assembly.topology != (CommonExecutionTopology::Host { workers: 1 })
    {
        return Err(invalid(
            "Result assembly evidence differs from the accepted reference assembly provider",
        ));
    }
    Ok(())
}

fn solver_provider_matches(actual: &CommonProviderEvidence, expected: SolverProvider) -> bool {
    actual.id == expected.id().as_str()
        && actual.implementation_version == expected.implementation_version()
        && libraries_match(&actual.libraries, expected.libraries())
}

fn execution_provider_matches(
    actual: &CommonProviderEvidence,
    expected: ExecutionProvider,
) -> bool {
    actual.id == expected.id().as_str()
        && actual.implementation_version == expected.implementation_version()
        && libraries_match(&actual.libraries, expected.libraries())
}

fn libraries_match(actual: &[(String, String)], expected: &[ProviderLibrary]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.0 == expected.name() && actual.1 == expected.version())
}

pub(super) fn require_text(value: &str, label: &str) -> Result<(), Diagnostic> {
    if value.is_empty() || value.trim() != value {
        return Err(invalid(format!(
            "Result {label} must be nonempty canonical text"
        )));
    }
    Ok(())
}

pub(super) fn require_finite(values: &[f64], label: &str) -> Result<(), Diagnostic> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!("{label} contains a non-finite value")));
    }
    Ok(())
}

pub(super) fn require_finite_nonnegative(values: &[f64], label: &str) -> Result<(), Diagnostic> {
    if values.iter().any(|value| {
        !value.is_finite() || *value < 0.0 || (*value == 0.0 && value.is_sign_negative())
    }) {
        return Err(invalid(format!(
            "{label} must contain finite non-negative values"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use eqiora_solver::{BackendId, ProviderLibrary, SolverProvider};

    use super::*;

    const LIBRARIES: &[ProviderLibrary] = &[ProviderLibrary::new("solver-core", "1.2.3")];
    const PROVIDER: SolverProvider =
        SolverProvider::new(BackendId::new("eqiora.test.solver"), "4.5.6", LIBRARIES);

    #[test]
    fn exact_provider_inventory_is_part_of_result_acceptance() {
        let exact = CommonProviderEvidence {
            id: "eqiora.test.solver".to_owned(),
            implementation_version: "4.5.6".to_owned(),
            libraries: vec![("solver-core".to_owned(), "1.2.3".to_owned())],
        };
        assert!(solver_provider_matches(&exact, PROVIDER));

        let mut forged = exact;
        forged.libraries[0].1 = "9.9.9".to_owned();
        assert!(!solver_provider_matches(&forged, PROVIDER));
    }

    #[test]
    fn assembly_and_nonnegative_scalars_reject_forged_canonical_content() {
        let exact = CommonAssemblyEvidence {
            adapter: SERIAL_EXECUTION_PROVIDER.id().as_str().to_owned(),
            topology: CommonExecutionTopology::Host { workers: 1 },
            packet_count: 1,
            target_count: 1,
        };
        assert!(require_reference_assembly(&exact).is_ok());

        let mut forged = exact;
        forged.adapter = "eqiora.test.forged".to_owned();
        assert!(require_reference_assembly(&forged).is_err());
        assert!(require_finite_nonnegative(&[-0.0], "test").is_err());
    }
}
