//! Durable, storage-independent spatial observations and trajectories.
//!
//! The logical graph is deliberately narrow: existing mesh-bound discrete
//! fields remain the numeric leaves; snapshots add exact Semantic Field and
//! Realization meaning; states, immutable segments, and roots add accepted
//! time identity. Storage realization is a separate typed projection.

mod field;
mod ml_dataset;
mod storage;
mod trajectory;

pub use field::FieldSnapshotEnvelopeV2;
pub use ml_dataset::{
    MlDatasetChannelStatisticsV1, MlDatasetDecoderLimits, MlDatasetDescriptorRoleV1,
    MlDatasetEnvelope, MlDatasetFieldDescriptorV1, MlDatasetObservationReferenceV1,
    MlDatasetSampleSplitV1, MlDatasetSampleV1, MlDatasetStateReferenceV1,
};
pub use storage::{DiscreteFieldStorageEnvelopeV1, StorageChunkSha256V1, StorageChunkV1};
pub use trajectory::TrajectoryDecoderLimits;
