//! Runtime-observed MPI provenance used as input by host integration tests.

use std::num::NonZeroUsize;

use eqiora::artifact::{
    DistributedTransportV1, ExecutionProvenanceV1, ExecutionTopologyV1, MpiThreadSupportV1,
};
use eqiora::backends::mpi::{MpiExecutionGroup, MpiThreadSupport};
use eqiora_execution::ExecutionReceipt;

pub(crate) fn observed_execution(
    group: &MpiExecutionGroup,
    receipt: &ExecutionReceipt,
) -> ExecutionProvenanceV1 {
    let raw_library = mpi::environment::library_version()
        .expect("MPI implementation reports a UTF-8 library version");
    let implementation = mpi_implementation(&raw_library);
    let version = normalize_mpi_library_version(&raw_library);
    assert!(!version.is_empty());
    let (standard_major, standard_minor) = mpi::environment::version();
    ExecutionProvenanceV1::from_provider_releases(
        receipt.solver_provider(),
        receipt.execution_provider(),
        ExecutionTopologyV1::Distributed {
            partitions: group.partitions(),
            workers_per_partition: NonZeroUsize::MIN,
            transport: DistributedTransportV1::Mpi {
                implementation: implementation.to_owned(),
                version,
                thread_support: artifact_thread_support(group.thread_support()),
            },
        },
        receipt.report().reduction(),
        [("mpi-standard", format!("{standard_major}.{standard_minor}"))],
    )
    .unwrap()
}

fn normalize_mpi_library_version(value: &str) -> String {
    value
        .split(|character: char| character.is_whitespace() || character.is_control())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn mpi_implementation(library_version: &str) -> &'static str {
    let lower = library_version.to_ascii_lowercase();
    if lower.contains("open mpi") || lower.contains("open-mpi") {
        "openmpi"
    } else if lower.contains("mpich") {
        "mpich"
    } else {
        "system-mpi"
    }
}

const fn artifact_thread_support(value: MpiThreadSupport) -> MpiThreadSupportV1 {
    match value {
        MpiThreadSupport::Single => MpiThreadSupportV1::Single,
        MpiThreadSupport::Funneled => MpiThreadSupportV1::Funneled,
        MpiThreadSupport::Serialized => MpiThreadSupportV1::Serialized,
        MpiThreadSupport::Multiple => MpiThreadSupportV1::Multiple,
    }
}

#[test]
fn mpi_library_version_normalization_removes_transport_controls() {
    let raw = "Open MPI v4.1.4, package: runner\nOpen MPI: 4.1.4\t\0\0";
    assert_eq!(
        normalize_mpi_library_version(raw),
        "Open MPI v4.1.4, package: runner Open MPI: 4.1.4"
    );
}
