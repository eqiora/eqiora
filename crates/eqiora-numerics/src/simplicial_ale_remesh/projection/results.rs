use eqiora_meshing::CellId;
use eqiora_solver::SolveReport;

use super::COMPONENTS;

pub(super) struct VectorP1Projection {
    pub(super) coefficients: Vec<[f64; COMPONENTS]>,
    pub(super) reports: Vec<SolveReport>,
    pub(super) right_hand_side_norms: Vec<f64>,
    pub(super) residual_norm: f64,
}

pub(super) struct VelocityProjection {
    pub(super) vertex: Vec<[f64; COMPONENTS]>,
    pub(super) bubble: std::collections::BTreeMap<CellId, [f64; COMPONENTS]>,
    pub(super) report: SolveReport,
    pub(super) right_hand_side_norm: f64,
    pub(super) residual_norm: f64,
    pub(super) independent_constraint_count: usize,
    pub(super) maximum_shared_trace_defect: f64,
    pub(super) maximum_exterior_trace_defect: f64,
    pub(super) weak_divergence_norm: f64,
    pub(super) source_momentum: [f64; COMPONENTS],
    pub(super) target_momentum: [f64; COMPONENTS],
    pub(super) fluid_l2_error: f64,
    pub(super) solid_l2_error: f64,
}

pub(super) struct PressureProjection {
    pub(super) coefficients: Vec<f64>,
    pub(super) report: SolveReport,
    pub(super) right_hand_side_norm: f64,
    pub(super) residual_norm: f64,
    pub(super) source_moment: f64,
    pub(super) target_moment: f64,
    pub(super) l2_error: f64,
}
