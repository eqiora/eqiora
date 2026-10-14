use std::num::NonZeroUsize;

use eqiora::assembly::LinearSystem;
use eqiora_spatial_distribution::DistributedAssemblyEvidence;

pub(crate) fn assert_assembly_evidence(
    evidence: &DistributedAssemblyEvidence,
    partitions: NonZeroUsize,
    packet_count: usize,
    target_count: usize,
) {
    let receipt = evidence.receipt();
    assert_eq!(receipt.packet_count(), packet_count);
    assert_eq!(receipt.target_count(), target_count);
    assert_eq!(receipt.partition_count(), partitions);
    assert_eq!(evidence.target_partitions().len(), target_count);
    assert_eq!(evidence.shards().len(), target_count);
    assert_eq!(evidence.system_identities().len(), target_count);
    for (target, (partition, shards)) in evidence
        .target_partitions()
        .iter()
        .zip(evidence.shards())
        .enumerate()
    {
        assert_eq!(partition.partition_count(), partitions);
        assert_eq!(shards.len(), partitions.get());
        let dimension = partition.global_size().get();
        let mut rows = shards
            .iter()
            .enumerate()
            .flat_map(|(partition_index, shard)| {
                assert_eq!(shard.target().index(), target);
                assert_eq!(
                    shard.partition(),
                    eqiora::distributed::PartitionId::new(partition_index)
                );
                shard.rows().iter().map(|row| row.index())
            })
            .collect::<Vec<_>>();
        rows.sort_unstable();
        assert_eq!(rows, (0..dimension).collect::<Vec<_>>());
        for (row, owner) in partition.owners().iter().copied().enumerate() {
            assert!(
                evidence.shards()[target][owner.index()]
                    .rows()
                    .iter()
                    .any(|candidate| candidate.index() == row)
            );
        }
    }
}

pub(crate) fn assert_system_bits(candidate: &LinearSystem<f64>, reference: &LinearSystem<f64>) {
    assert_eq!(candidate.matrix().rows(), reference.matrix().rows());
    assert_eq!(candidate.matrix().columns(), reference.matrix().columns());
    assert_eq!(
        candidate.matrix().row_offsets(),
        reference.matrix().row_offsets()
    );
    assert_eq!(
        candidate.matrix().column_indices(),
        reference.matrix().column_indices()
    );
    assert_eq!(
        candidate
            .matrix()
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        reference
            .matrix()
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        candidate
            .rhs()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        reference
            .rhs()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}
