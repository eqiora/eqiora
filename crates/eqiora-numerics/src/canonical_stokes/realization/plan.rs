//! Construct the admitted MINI Field and gauge projection.
use super::*;

pub(crate) fn steady_stokes_algebraic_structure(
    model: &SteadyIncompressibleStokesModel2d,
) -> Result<eqiora_solver::AlgebraicStructure, Diagnostic> {
    let constraints =
        requires_zero_integral_constraint(model)?.then_some(AlgebraicConstraint::ZeroIntegral {
            field: pressure_id(model),
        });
    eqiora_solver::AlgebraicStructure::new([velocity_id(model), pressure_id(model)], constraints)
}

/// Build the sole admitted field-wise MINI plan from exact canonical roles.
///
/// `q` remains immutable canonical coefficient data and receives no discrete
/// unknown block. The plan derives gauge and functional scales from `L/U/P`,
/// fixes positive degree-four Duffy assembly, and admits only the exact
/// reference-MINRES and host-sparse-LU solver tuples verified for this path,
/// without a fluid-named generic space tag.
///
/// # Errors
/// Returns `EQ0807` for an unsupported solver tuple or invalid derived plan.
pub fn steady_stokes_mini_plan_2d(
    model: &SteadyIncompressibleStokesModel2d,
    mesh: MeshArtifactReference,
    scales: IncompressibleFlowScaleProfile2d,
    solver: SolverPlan,
) -> Result<FieldwiseRealizationPlan, Diagnostic> {
    let structure = steady_stokes_algebraic_structure(model)?;
    let with_zero_integral_constraint = !structure.constraints().is_empty();
    require_mini_solver(solver)?;
    let velocity = velocity_id(model);
    let pressure = pressure_id(model);
    let constraints = structure.constraints().to_vec();
    let spatial = FieldwiseSpatialDiscretization::new(
        domain_id(model),
        scales.length,
        [
            FieldSpaceBinding::new(velocity, Space::simplex_p1_bubble()),
            FieldSpaceBinding::new(pressure, Space::continuous_lagrange(NonZeroU16::MIN)),
        ],
        constraints,
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::ImportedSimplicial { artifact: mesh },
            QuadraturePolicy::SimplexDuffyGaussLegendre {
                spatial_dimension: std::num::NonZeroUsize::new(2).expect("two is nonzero"),
                points_per_axis: NonZeroUsize::new(DUFFY_POINTS_PER_AXIS)
                    .expect("three is non-zero"),
            },
        ),
    )
    .map_err(realization_error)?;
    let mut block_scales = vec![
        AlgebraicBlockScale::new(AlgebraicBlock::Field(velocity), scales.velocity),
        AlgebraicBlockScale::new(AlgebraicBlock::Field(pressure), scales.pressure),
    ];
    if with_zero_integral_constraint {
        block_scales.push(AlgebraicBlockScale::new(
            AlgebraicBlock::ConstraintMultiplier { field: pressure },
            scales.gauge,
        ));
    }
    let scaling = SymmetricCongruenceScaling::new(block_scales, scales.weak_functional)
        .map_err(realization_error)?;
    FieldwiseRealizationPlan::new(
        spatial,
        scaling,
        LinearOperatorProperties::SymmetricIndefinite,
        solver,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .map_err(realization_error)
}
