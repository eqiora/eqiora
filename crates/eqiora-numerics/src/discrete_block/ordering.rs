use super::*;

pub(super) fn has_duplicate<T: PartialEq>(values: &[T]) -> bool {
    values.windows(2).any(|pair| pair[0] == pair[1])
}

pub(super) fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

pub(super) fn block_field(block: AlgebraicBlock) -> Id<kinds::Field> {
    match block {
        AlgebraicBlock::Field(field) | AlgebraicBlock::ConstraintMultiplier { field } => field,
    }
}

pub(super) fn block_tag(block: AlgebraicBlock) -> u8 {
    match block {
        AlgebraicBlock::Field(_) => 0,
        AlgebraicBlock::ConstraintMultiplier { .. } => 1,
    }
}

pub(super) fn block_order(left: &AlgebraicBlock, right: &AlgebraicBlock) -> std::cmp::Ordering {
    block_tag(*left)
        .cmp(&block_tag(*right))
        .then_with(|| block_field(*left).ulid().cmp(&block_field(*right).ulid()))
}

pub(super) fn residual_origin_order(
    left: &ResidualOrigin,
    right: &ResidualOrigin,
) -> std::cmp::Ordering {
    match (*left, *right) {
        (ResidualOrigin::Relation(left), ResidualOrigin::Relation(right)) => {
            left.ulid().cmp(&right.ulid())
        }
        (ResidualOrigin::AlgebraicConstraint(left), ResidualOrigin::AlgebraicConstraint(right)) => {
            left.field().ulid().cmp(&right.field().ulid())
        }
        (ResidualOrigin::Relation(_), ResidualOrigin::AlgebraicConstraint(_)) => {
            std::cmp::Ordering::Less
        }
        (ResidualOrigin::AlgebraicConstraint(_), ResidualOrigin::Relation(_)) => {
            std::cmp::Ordering::Greater
        }
    }
}

pub(super) fn residual_block_order(
    left: &ResidualBlock,
    right: &ResidualBlock,
) -> std::cmp::Ordering {
    block_order(&left.tested, &right.tested)
        .then_with(|| support_order(left.support, right.support))
        .then_with(|| left.origins.len().cmp(&right.origins.len()))
}

pub(super) fn transformation_order(
    left: &BlockTransformation,
    right: &BlockTransformation,
) -> std::cmp::Ordering {
    match (left, right) {
        (
            BlockTransformation::EssentialElimination { field: left, .. },
            BlockTransformation::EssentialElimination { field: right, .. },
        ) => left.ulid().cmp(&right.ulid()),
        (
            BlockTransformation::BackwardEulerElimination { state: left, .. },
            BlockTransformation::BackwardEulerElimination { state: right, .. },
        ) => left.ulid().cmp(&right.ulid()),
        (
            BlockTransformation::BackwardEulerDerivative { state: left, .. },
            BlockTransformation::BackwardEulerDerivative { state: right, .. },
        ) => left.ulid().cmp(&right.ulid()),
        (
            BlockTransformation::ConformingTraceQuotient { quotient: left, .. },
            BlockTransformation::ConformingTraceQuotient {
                quotient: right, ..
            },
        ) => left.source().cmp(&right.source()),
        (
            BlockTransformation::EnergySkewConvection { relation: left, .. },
            BlockTransformation::EnergySkewConvection {
                relation: right, ..
            },
        ) => left.ulid().cmp(&right.ulid()),
        (BlockTransformation::EssentialElimination { .. }, _) => std::cmp::Ordering::Less,
        (_, BlockTransformation::EssentialElimination { .. }) => std::cmp::Ordering::Greater,
        (BlockTransformation::BackwardEulerElimination { .. }, _) => std::cmp::Ordering::Less,
        (_, BlockTransformation::BackwardEulerElimination { .. }) => std::cmp::Ordering::Greater,
        (BlockTransformation::BackwardEulerDerivative { .. }, _) => std::cmp::Ordering::Less,
        (_, BlockTransformation::BackwardEulerDerivative { .. }) => std::cmp::Ordering::Greater,
        (BlockTransformation::EnergySkewConvection { .. }, _) => std::cmp::Ordering::Less,
        (_, BlockTransformation::EnergySkewConvection { .. }) => std::cmp::Ordering::Greater,
    }
}

pub(super) fn contribution_order(
    left: &ContributionBatch,
    right: &ContributionBatch,
) -> std::cmp::Ordering {
    left.packet_indices
        .first()
        .cmp(&right.packet_indices.first())
        .then_with(|| left.supports.len().cmp(&right.supports.len()))
        .then_with(|| {
            left.supports
                .iter()
                .zip(&right.supports)
                .find_map(|(left, right)| {
                    let order = support_order(*left, *right);
                    (order != std::cmp::Ordering::Equal).then_some(order)
                })
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

pub(super) fn closure_order(
    left: &AlgebraicClosure,
    right: &AlgebraicClosure,
) -> std::cmp::Ordering {
    closure_key(left).cmp(&closure_key(right))
}

pub(super) fn closure_key(value: &AlgebraicClosure) -> (u8, String) {
    match value {
        AlgebraicClosure::EssentialBoundary { field, .. } => (0, field.ulid().to_string()),
        AlgebraicClosure::ZeroIntegral { field } => (1, field.ulid().to_string()),
        AlgebraicClosure::BoundaryTraction { field, .. } => (2, field.ulid().to_string()),
        AlgebraicClosure::CompleteOperator { field, .. } => (3, field.ulid().to_string()),
    }
}

pub(super) fn support_order(left: BlockSupport, right: BlockSupport) -> std::cmp::Ordering {
    match (left, right) {
        (BlockSupport::Volume(left), BlockSupport::Volume(right))
        | (BlockSupport::Boundary(left), BlockSupport::Boundary(right)) => {
            left.ulid().cmp(&right.ulid())
        }
        (BlockSupport::Volume(_), BlockSupport::Boundary(_)) => std::cmp::Ordering::Less,
        (BlockSupport::Boundary(_), BlockSupport::Volume(_)) => std::cmp::Ordering::Greater,
    }
}
