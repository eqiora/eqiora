//! Immutable spatial trajectory manifests.

/// Semantic work budgets shared by immutable trajectory and derived-view artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrajectoryDecoderLimits {
    /// Common JSON syntax admission.
    pub json: crate::JsonLimits,
    /// Maximum Field references summarized by one trajectory state.
    pub max_spatial_state_fields: usize,
    /// Maximum state references in one immutable trajectory segment.
    pub max_trajectory_segment_states: usize,
    /// Maximum immutable segments referenced by one trajectory root.
    pub max_trajectory_segments: usize,
    /// Maximum accepted states summarized by one complete trajectory root.
    pub max_trajectory_states: usize,
}

impl Default for TrajectoryDecoderLimits {
    fn default() -> Self {
        Self {
            json: crate::JsonLimits::default(),
            max_spatial_state_fields: 100_000,
            max_trajectory_segment_states: 100_000,
            max_trajectory_segments: 100_000,
            max_trajectory_states: 1_000_000,
        }
    }
}
