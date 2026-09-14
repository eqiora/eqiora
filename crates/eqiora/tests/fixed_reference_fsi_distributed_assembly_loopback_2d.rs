use std::num::NonZeroUsize;
use std::sync::Mutex;

use eqiora::Diagnostic;
use eqiora::assembly::{AssemblyBackend, AssemblyPlan, AssemblyResult, AssemblyWork, LinearSystem};
use eqiora::meshing::MeshEntity;
use eqiora::solver::REFERENCE_LINEAR_SOLVER;
use eqiora_numerics::fsi::finalize_resolved_fixed_reference_fsi_step_2d_with_assembly;
use eqiora_numerics::fsi::lower_fixed_reference_fsi_cartesian_2d;
use eqiora_spatial_distribution::{
    CellOwnershipClaim, DistributedAssemblyEvidence, DistributedMeshLayout,
    LoopbackSpatialAssemblyBackend, MeshRevisionIdentityV1,
};
use support::fixed_reference_fsi::{
    direct_document, execution_context, prestrained_state, spatial_context,
};

mod support;

#[test]
fn fixed_reference_fsi_distributed_assembly_loopback_2d() {
    let document = direct_document();
    let canonical = lower_fixed_reference_fsi_cartesian_2d(document.program())
        .expect("fixed-reference FSI semantics lower");
    let spatial = spatial_context(document.program(), &canonical);
    let execution = execution_context(document.program(), &canonical, &spatial);
    let previous = prestrained_state(document.program(), &spatial, &execution);
    let mesh_sha256 = spatial
        .mesh_artifact
        .digest()
        .expect("authenticated mesh digest")
        .sha256_bytes();
    assert_eq!(mesh_sha256, execution.mesh_reference.sha256());

    let reference_capture =
        CapturingAssemblyBackend::new(&eqiora::assembly::REFERENCE_ASSEMBLY_BACKEND);
    let reference = finalize_resolved_fixed_reference_fsi_step_2d_with_assembly(
        &canonical,
        &execution.resolved,
        execution.mesh_reference,
        &spatial.mesh,
        &spatial.partition,
        &previous,
        &reference_capture,
    )
    .expect("reference checked block assembly finalizes");
    let reference_systems = reference_capture
        .take()
        .expect("reference backend exposes its accepted two-target result")
        .systems()
        .to_vec();
    assert_eq!(reference_systems.len(), 2);
    let reference_fingerprint = reference.linear_system().agreement_fingerprint();

    let mut foreign_mesh_sha256 = mesh_sha256;
    foreign_mesh_sha256[0] ^= 1;
    let foreign_layout = DistributedMeshLayout::derive(
        MeshRevisionIdentityV1::from_sha256(foreign_mesh_sha256),
        &spatial.mesh,
        NonZeroUsize::new(2).unwrap(),
        cell_claims(NonZeroUsize::new(2).unwrap()),
    )
    .expect("same-shape foreign mesh revision still derives structurally");
    let foreign = LoopbackSpatialAssemblyBackend::new(foreign_layout);
    let diagnostic = finalize_resolved_fixed_reference_fsi_step_2d_with_assembly(
        &canonical,
        &execution.resolved,
        execution.mesh_reference,
        &spatial.mesh,
        &spatial.partition,
        &previous,
        &foreign,
    )
    .expect_err("assembly work must remain bound to the authenticated mesh revision");
    assert_eq!(
        diagnostic.code(),
        eqiora::diagnostic::codes::ASSEMBLY_FAILED
    );
    assert!(foreign.accepted_evidence().unwrap().is_none());

    for partition_count in [1, 2, 4] {
        let partition_count = NonZeroUsize::new(partition_count).unwrap();
        let layout = DistributedMeshLayout::derive(
            MeshRevisionIdentityV1::from_sha256(mesh_sha256),
            &spatial.mesh,
            partition_count,
            cell_claims(partition_count),
        )
        .expect("exact cell ownership derives one complete mesh layout");
        assert_eq!(layout.cell_count(), 8);
        if partition_count.get() > 1 {
            let process_facets = layout
                .partition_boundary_entities(1)
                .expect("triangle layout owns a facet stratum");
            for facet in &spatial.interface_facets {
                let entity = MeshEntity::new(1, facet.index());
                assert!(process_facets.contains(&entity));
                assert!(layout.entity_residents(entity).unwrap().len() > 1);
            }
        }

        let distributed = LoopbackSpatialAssemblyBackend::new(layout);
        let capture = CapturingAssemblyBackend::new(&distributed);
        let finalized = finalize_resolved_fixed_reference_fsi_step_2d_with_assembly(
            &canonical,
            &execution.resolved,
            execution.mesh_reference,
            &spatial.mesh,
            &spatial.partition,
            &previous,
            &capture,
        )
        .expect("owner-routed assembly passes the exact canonical block boundary");
        let candidate = capture
            .take()
            .expect("distributed backend exposes its accepted reconstructed targets");
        assert_eq!(candidate.systems().len(), reference_systems.len());
        for (candidate, reference) in candidate.systems().iter().zip(&reference_systems) {
            assert_system_bits(candidate, reference);
        }
        assert_eq!(
            finalized.linear_system().agreement_fingerprint(),
            reference_fingerprint
        );

        let evidence = distributed
            .accepted_evidence()
            .expect("evidence lock remains healthy")
            .expect("successful assembly publishes accepted evidence");
        assert_evidence(&evidence, partition_count);

        let solution = finalized
            .solve(&REFERENCE_LINEAR_SOLVER)
            .expect("reconstructed operator passes unchanged FSI acceptance");
        let numerical = solution.numerical_evidence();
        assert!(numerical.residual_norm() < 1.0e-9);
        assert!(numerical.continuity_residual_norm() < 1.0e-9);
        assert!(numerical.kinematic_residual_norm() < 1.0e-14);
        assert_eq!(numerical.interface_velocity_jump_norm(), 0.0);
        assert!(numerical.interface_action_imbalance_norm() < 1.0e-9);
        assert!(numerical.energy_balance().defect().abs() < 1.0e-9);
    }
}

fn cell_claims(partition_count: NonZeroUsize) -> Vec<CellOwnershipClaim> {
    (0..8)
        .map(|cell| {
            let owner = match partition_count.get() {
                1 => 0,
                2 => usize::from(cell < 4),
                4 => cell % 4,
                _ => unreachable!("the registered fixture admits only one/two/four partitions"),
            };
            CellOwnershipClaim::new(
                MeshEntity::new(2, cell),
                eqiora::distributed::PartitionId::new(owner),
            )
        })
        .collect()
}

fn assert_evidence(evidence: &DistributedAssemblyEvidence, partitions: NonZeroUsize) {
    let receipt = evidence.receipt();
    assert_eq!(receipt.packet_count(), 8);
    assert_eq!(receipt.target_count(), 2);
    assert_eq!(receipt.partition_count(), partitions);
    assert_eq!(evidence.target_partitions().len(), 2);
    assert_eq!(evidence.shards().len(), 2);
    assert_eq!(evidence.system_identities().len(), 2);
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

fn assert_system_bits(candidate: &LinearSystem, reference: &LinearSystem) {
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

#[derive(Debug)]
struct CapturingAssemblyBackend<'a> {
    inner: &'a dyn AssemblyBackend,
    accepted: Mutex<Option<AssemblyResult>>,
}

impl<'a> CapturingAssemblyBackend<'a> {
    fn new(inner: &'a dyn AssemblyBackend) -> Self {
        Self {
            inner,
            accepted: Mutex::new(None),
        }
    }

    fn take(&self) -> Option<AssemblyResult> {
        self.accepted.lock().expect("capture lock").take()
    }
}

impl AssemblyBackend for CapturingAssemblyBackend<'_> {
    fn assemble(
        &self,
        plan: &AssemblyPlan,
        work: &dyn AssemblyWork,
    ) -> Result<AssemblyResult, Diagnostic> {
        let result = self.inner.assemble(plan, work)?;
        *self.accepted.lock().expect("capture lock") = Some(result.clone());
        Ok(result)
    }
}
