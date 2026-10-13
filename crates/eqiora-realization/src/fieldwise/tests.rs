use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, DynQuantity, Id};
use eqiora_solver::{
    LinearOperatorProperties, LinearSolver, PreconditionerPolicy, ReductionPolicy, SolverPlan,
};

use super::*;
use crate::{
    DiscretizationMethod, ExecutionSchedule, MeshArtifactReference, MeshPolicy, QuadraturePolicy,
    RealizationPlan, Target,
};

#[test]
fn field_and_block_order_do_not_change_the_plan() {
    let domain = Id::<kinds::Domain>::new();
    let first = Id::<kinds::Field>::new();
    let second = Id::<kinds::Field>::new();
    let forward = plan(domain, first, second, false);
    let reversed = plan(domain, first, second, true);

    assert_eq!(forward, reversed);
    assert!(
        forward
            .spatial()
            .field_spaces()
            .windows(2)
            .all(|pair| pair[0].field().ulid() < pair[1].field().ulid())
    );
}

#[test]
fn physical_scales_fail_closed() {
    for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            PositivePhysicalScale::new(DynQuantity::new(value, DimExponents::DIMENSIONLESS))
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
    }

    let field = Id::<kinds::Field>::new();
    let error = FieldwiseSpatialDiscretization::new(
        Id::new(),
        scale(1.0, DimExponents::DIMENSIONLESS),
        [FieldSpaceBinding::new(
            field,
            Space::continuous_lagrange(NonZeroU16::MIN),
        )],
        [],
        discretization(),
    )
    .unwrap_err();
    assert_eq!(error.code(), codes::INVALID_REALIZATION);
}

#[test]
fn bindings_constraints_and_scaling_have_exact_coverage() {
    let domain = Id::<kinds::Domain>::new();
    let velocity = Id::<kinds::Field>::new();
    let pressure = Id::<kinds::Field>::new();
    let duplicate_binding = FieldwiseSpatialDiscretization::new(
        domain,
        length_scale(),
        [
            FieldSpaceBinding::new(velocity, Space::simplex_p1_bubble()),
            FieldSpaceBinding::new(velocity, Space::simplex_p1_bubble()),
        ],
        [],
        discretization(),
    )
    .unwrap_err();
    assert_eq!(duplicate_binding.code(), codes::INVALID_REALIZATION);

    let unbound_constraint = FieldwiseSpatialDiscretization::new(
        domain,
        length_scale(),
        [FieldSpaceBinding::new(velocity, Space::simplex_p1_bubble())],
        [AlgebraicConstraint::ZeroIntegral { field: pressure }],
        discretization(),
    )
    .unwrap_err();
    assert_eq!(unbound_constraint.code(), codes::INVALID_REALIZATION);

    let spatial = spatial(domain, velocity, pressure, false);
    let missing_multiplier = scaling([
        block_scale(AlgebraicBlock::Field(velocity), velocity_dimension()),
        block_scale(AlgebraicBlock::Field(pressure), pressure_dimension()),
    ]);
    assert_eq!(
        FieldwiseRealizationPlan::new(
            spatial.clone(),
            missing_multiplier,
            LinearOperatorProperties::SymmetricIndefinite,
            solver(),
            host(),
            ExecutionSchedule::Offline,
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );

    let duplicate_scale = SymmetricCongruenceScaling::new(
        [
            block_scale(AlgebraicBlock::Field(velocity), velocity_dimension()),
            block_scale(AlgebraicBlock::Field(velocity), velocity_dimension()),
        ],
        scale(1.0, functional_dimension()),
    )
    .unwrap_err();
    assert_eq!(duplicate_scale.code(), codes::INVALID_REALIZATION);
}

#[test]
fn fieldwise_spatial_families_are_exact_and_v1_remains_separate() {
    let domain = Id::<kinds::Domain>::new();
    let field = Id::<kinds::Field>::new();
    let spatial = FieldwiseSpatialDiscretization::new(
        domain,
        length_scale(),
        [FieldSpaceBinding::new(
            field,
            Space::continuous_lagrange(NonZeroU16::MIN),
        )],
        [],
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::ImportedSimplicial {
                artifact: MeshArtifactReference::from_sha256([9; 32]),
            },
            QuadraturePolicy::SimplexCentroid,
        ),
    )
    .unwrap();
    let continuous_scaling = scaling([block_scale(
        AlgebraicBlock::Field(field),
        pressure_dimension(),
    )]);
    assert_eq!(
        FieldwiseRealizationPlan::new(
            spatial,
            continuous_scaling,
            LinearOperatorProperties::SymmetricPositiveDefinite,
            SolverPlan::new(
                LinearSolver::ConjugateGradient,
                1.0e-8,
                1.0e-12,
                NonZeroUsize::new(100).unwrap(),
            )
            .unwrap(),
            host(),
            ExecutionSchedule::Offline,
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );

    assert_eq!(
        RealizationPlan::new(
            Space::simplex_p1_bubble(),
            discretization(),
            solver(),
            host(),
            ExecutionSchedule::Offline,
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );

    let finite_volume = FieldwiseSpatialDiscretization::new(
        domain,
        length_scale(),
        [FieldSpaceBinding::new(field, Space::cell_constant())],
        [],
        Discretization::new(
            DiscretizationMethod::CellCenteredFiniteVolume,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(8).unwrap(),
            },
            QuadraturePolicy::CellCentroid,
        ),
    )
    .unwrap();
    let finite_volume_plan = FieldwiseRealizationPlan::new(
        finite_volume.clone(),
        scaling([block_scale(
            AlgebraicBlock::Field(field),
            pressure_dimension(),
        )]),
        LinearOperatorProperties::General,
        SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1.0e-8,
            1.0e-12,
            NonZeroUsize::new(100).unwrap(),
        )
        .unwrap(),
        host(),
        ExecutionSchedule::Offline,
    )
    .unwrap();
    assert_eq!(finite_volume_plan.spatial(), &finite_volume);

    let mixed_space = FieldwiseSpatialDiscretization::new(
        domain,
        length_scale(),
        [FieldSpaceBinding::new(
            field,
            Space::continuous_lagrange(NonZeroU16::MIN),
        )],
        [],
        finite_volume.discretization(),
    )
    .unwrap();
    assert_eq!(
        FieldwiseRealizationPlan::new(
            mixed_space,
            scaling([block_scale(
                AlgebraicBlock::Field(field),
                pressure_dimension(),
            )]),
            LinearOperatorProperties::General,
            SolverPlan::new(
                LinearSolver::BiConjugateGradientStabilized,
                1.0e-8,
                1.0e-12,
                NonZeroUsize::new(100).unwrap(),
            )
            .unwrap(),
            host(),
            ExecutionSchedule::Offline,
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );
}

fn plan(
    domain: Id<kinds::Domain>,
    velocity: Id<kinds::Field>,
    pressure: Id<kinds::Field>,
    reverse: bool,
) -> FieldwiseRealizationPlan {
    let spatial = spatial(domain, velocity, pressure, reverse);
    let mut scales = vec![
        block_scale(AlgebraicBlock::Field(velocity), velocity_dimension()),
        block_scale(AlgebraicBlock::Field(pressure), pressure_dimension()),
        block_scale(
            AlgebraicBlock::ConstraintMultiplier { field: pressure },
            gauge_dimension(),
        ),
    ];
    if reverse {
        scales.reverse();
    }
    FieldwiseRealizationPlan::new(
        spatial,
        scaling(scales),
        LinearOperatorProperties::SymmetricIndefinite,
        solver(),
        host(),
        ExecutionSchedule::Offline,
    )
    .unwrap()
}

fn spatial(
    domain: Id<kinds::Domain>,
    velocity: Id<kinds::Field>,
    pressure: Id<kinds::Field>,
    reverse: bool,
) -> FieldwiseSpatialDiscretization {
    let mut bindings = vec![
        FieldSpaceBinding::new(velocity, Space::simplex_p1_bubble()),
        FieldSpaceBinding::new(pressure, Space::continuous_lagrange(NonZeroU16::MIN)),
    ];
    if reverse {
        bindings.reverse();
    }
    FieldwiseSpatialDiscretization::new(
        domain,
        length_scale(),
        bindings,
        [AlgebraicConstraint::ZeroIntegral { field: pressure }],
        discretization(),
    )
    .unwrap()
}

fn discretization() -> Discretization {
    Discretization::new(
        DiscretizationMethod::ContinuousGalerkin,
        MeshPolicy::ImportedSimplicial {
            artifact: MeshArtifactReference::from_sha256([7; 32]),
        },
        QuadraturePolicy::SimplexDuffyGaussLegendre {
            spatial_dimension: std::num::NonZeroUsize::new(2).expect("two is nonzero"),
            points_per_axis: NonZeroUsize::new(3).unwrap(),
        },
    )
}

fn scaling(scales: impl IntoIterator<Item = AlgebraicBlockScale>) -> SymmetricCongruenceScaling {
    SymmetricCongruenceScaling::new(scales, scale(1.0, functional_dimension())).unwrap()
}

fn block_scale(block: AlgebraicBlock, dimension: DimExponents) -> AlgebraicBlockScale {
    AlgebraicBlockScale::new(block, scale(1.0, dimension))
}

fn solver() -> SolverPlan {
    SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(10_000).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Reproducible)
}

fn host() -> Target {
    Target::HostCpu {
        threads: NonZeroUsize::MIN,
    }
}

fn scale(value: f64, dimension: DimExponents) -> PositivePhysicalScale {
    PositivePhysicalScale::new(DynQuantity::new(value, dimension)).unwrap()
}

fn length_scale() -> PositivePhysicalScale {
    scale(
        1.0,
        DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension"),
    )
}

fn velocity_dimension() -> DimExponents {
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension")
}

fn pressure_dimension() -> DimExponents {
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension")
}

fn gauge_dimension() -> DimExponents {
    DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).expect("bounded dimension")
}

fn functional_dimension() -> DimExponents {
    DimExponents::from_integers([1, 1, -3, 0, 0, 0, 0]).expect("bounded dimension")
}
