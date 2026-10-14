use std::num::NonZeroUsize;
use std::sync::Mutex;

use eqiora::Diagnostic;
use eqiora::assembly::{AssemblyBackend, AssemblyPlan, AssemblyResult, AssemblyWork};
use eqiora::meshing::MeshEntity;
use eqiora::solver::REFERENCE_LINEAR_SOLVER;
use eqiora_numerics::fsi::finalize_resolved_fixed_reference_fsi_step_2d_with_assembly;
use eqiora_numerics::fsi::lower_fixed_reference_fsi_cartesian_2d;
use eqiora_spatial_distribution::{
    DistributedMeshLayout, LoopbackSpatialAssemblyBackend, MeshRevisionIdentityV1,
};
use support::fixed_reference_fsi::{
    cell_claims, direct_document, execution_context, prestrained_state, spatial_context,
};

use support::distributed_assembly::{assert_assembly_evidence, assert_system_bits};

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
        assert_assembly_evidence(&evidence, partition_count, 8, 2);

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

#[derive(Debug)]
struct CapturingAssemblyBackend<'a> {
    inner: &'a dyn AssemblyBackend<f64>,
    accepted: Mutex<Option<AssemblyResult<f64>>>,
}

impl<'a> CapturingAssemblyBackend<'a> {
    fn new(inner: &'a dyn AssemblyBackend<f64>) -> Self {
        Self {
            inner,
            accepted: Mutex::new(None),
        }
    }

    fn take(&self) -> Option<AssemblyResult<f64>> {
        self.accepted.lock().expect("capture lock").take()
    }
}

impl AssemblyBackend<f64> for CapturingAssemblyBackend<'_> {
    fn assemble(
        &self,
        plan: &AssemblyPlan,
        work: &dyn AssemblyWork<f64>,
    ) -> Result<AssemblyResult<f64>, Diagnostic> {
        let result = self.inner.assemble(plan, work)?;
        *self.accepted.lock().expect("capture lock") = Some(result.clone());
        Ok(result)
    }
}
