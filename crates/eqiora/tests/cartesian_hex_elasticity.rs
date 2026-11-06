//! Ordinary 3D Hex preparation. Numerical qualification is separately reviewed.
use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use eqiora::api::ModelDocument;
use eqiora::artifact::{
    CartesianMeshCellsV2, CartesianMeshDecoderLimits, CartesianMeshEnvelopeV1,
    GeometryMeshCorrespondenceEnvelopeV1, MeshProductionLineageEnvelopeV1,
};
use eqiora::compiler::StaticBindingValue;
use eqiora::geometry::{GeometryGraph, GeometryTopologyHandle};
use eqiora::meshing::{CartesianMesh, MeshEntity, MeshTopology};

const SOURCE: &str = include_str!("../../../verify/numerics/cartesian-hex-elasticity/model.eqi");
const SIDES: [&str; 6] = [
    "x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper",
];

// One case owns these preparation, observation and explicitly held execution helpers.
include!("cartesian_hex_elasticity/preparation.rs");
include!("cartesian_hex_elasticity/observations.rs");
include!("cartesian_hex_elasticity/execution.rs");
include!("cartesian_hex_elasticity/recovery.rs");

fn harness_source_sha256() -> String {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    for (name, bytes) in [
        (
            "entry",
            include_bytes!("cartesian_hex_elasticity.rs").as_slice(),
        ),
        (
            "preparation",
            include_bytes!("cartesian_hex_elasticity/preparation.rs").as_slice(),
        ),
        (
            "observations",
            include_bytes!("cartesian_hex_elasticity/observations.rs").as_slice(),
        ),
        (
            "execution",
            include_bytes!("cartesian_hex_elasticity/execution.rs").as_slice(),
        ),
        (
            "recovery",
            include_bytes!("cartesian_hex_elasticity/recovery.rs").as_slice(),
        ),
    ] {
        digest.update(name.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    digest
        .finalize()
        .iter()
        .map(|value| format!("{value:02x}"))
        .collect()
}
