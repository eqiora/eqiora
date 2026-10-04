//! Current coordinate equation families share one Plan and retained Field inventory.
use super::*;
use eqiora_core::ValueType;

#[derive(Debug, Clone, PartialEq)]
pub(in crate::numerical_admission) enum CellEquations {
    Projection(CellProjection),
    Diffusion(super::diffusion::Diffusion),
}
impl CellEquations {
    pub(in crate::numerical_admission) fn lower(
        program: &KernelProgram,
        grid: &CoordinateGrid,
    ) -> Result<Self, Diagnostic> {
        if program
            .nodes()
            .filter(|node| matches!(node, KernelNode::Field(_)))
            .count()
            == 1
        {
            CellProjection::lower(program, grid).map(Self::Projection)
        } else {
            super::diffusion::Diffusion::lower(program, grid).map(Self::Diffusion)
        }
    }
    pub(in crate::numerical_admission) fn primary_field(&self) -> Id<kinds::Field> {
        match self {
            Self::Projection(value) => value.field,
            Self::Diffusion(value) => value.concentration,
        }
    }
    pub(in crate::numerical_admission) fn fields(&self) -> Vec<(Id<kinds::Field>, ValueType)> {
        match self {
            Self::Projection(value) => vec![(value.field, value.value_type.clone())],
            Self::Diffusion(value) => vec![
                (value.concentration, value.concentration_type.clone()),
                (value.flux, value.flux_type.clone()),
            ],
        }
    }
}
