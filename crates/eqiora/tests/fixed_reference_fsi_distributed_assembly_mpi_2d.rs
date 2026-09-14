#![cfg(feature = "mpi")]

use std::cell::RefCell;
use std::env;
use std::io::Read;
use std::num::NonZeroUsize;
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use eqiora::Diagnostic;
use eqiora::assembly::{AssemblyBackend, AssemblyPlan, AssemblyResult, AssemblyWork, LinearSystem};
use eqiora::backends::mpi::{MpiExecutionGroup, MpiSpatialAssemblyBackend, MpiThreadSupport};
use eqiora::meshing::{FacetId, MeshEntity, MeshTopology};
use eqiora::solver::REFERENCE_LINEAR_SOLVER;
use eqiora_numerics::fsi::finalize_resolved_fixed_reference_fsi_step_2d_with_assembly;
use eqiora_numerics::fsi::lower_fixed_reference_fsi_cartesian_2d;
use eqiora_spatial_distribution::{
    CellOwnershipClaim, DistributedAssemblyEvidence, DistributedMeshLayout, MeshRevisionIdentityV1,
};
use mpi::Threading;
use mpi::traits::CommunicatorCollectives;
use support::fixed_reference_fsi::{
    direct_document, execution_context, prestrained_state, spatial_context,
};

mod support;

const CHILD_ENV: &str = "EQIORA_FIXED_REFERENCE_FSI_ASSEMBLY_MPI_CHILD";
const CHILD_TEST: &str = "fixed_reference_fsi_distributed_assembly_mpi_2d_child";
const CHILD_TIMEOUT: Duration = Duration::from_secs(180);
const CHILD_OUTPUT_LIMIT: usize = 64 * 1024;

#[test]
fn fixed_reference_fsi_distributed_assembly_mpi_2d_runs_on_one_two_and_four_ranks() {
    if env::var_os(CHILD_ENV).is_some() {
        return;
    }

    for ranks in [1, 2, 4] {
        assert_success(ranks, run_mpi_child(ranks));
    }
}

#[test]
fn fixed_reference_fsi_distributed_assembly_mpi_2d_child() {
    if env::var_os(CHILD_ENV).is_none() {
        return;
    }

    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("the child application initializes MPI exactly once");
    let world = universe.world();
    let mut group =
        MpiExecutionGroup::duplicate(&world, provided, MpiThreadSupport::Funneled).unwrap();
    let partitions = group.partitions();
    assert!(matches!(partitions.get(), 1 | 2 | 4));

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
    .expect("independent complete CPU reference assembly finalizes");
    let reference_systems = reference_capture
        .take()
        .expect("reference backend exposes both accepted targets")
        .systems()
        .to_vec();
    assert_eq!(reference_systems.len(), 2);
    let reference_fingerprint = reference.linear_system().agreement_fingerprint();

    if partitions.get() > 1 {
        assert_layout_disagreement_returns_on_every_rank(
            &world,
            &mut group,
            mesh_sha256,
            &spatial.mesh,
        );
    }

    let layout = layout(&spatial.mesh, partitions, mesh_sha256);
    if partitions.get() > 1 {
        let process_facets = layout
            .partition_boundary_entities(1)
            .expect("triangle layout owns a facet stratum");
        for quotient in spatial.partition.quotients() {
            let facet_count = spatial
                .mesh
                .entity_count(1)
                .expect("triangle mesh owns a facet stratum");
            for facet in (0..facet_count).map(FacetId::new).filter(|facet| {
                spatial
                    .partition
                    .facet_sides(quotient.connection(), *facet)
                    .is_some()
            }) {
                let entity = MeshEntity::new(1, facet.index());
                assert!(process_facets.contains(&entity));
                assert!(layout.entity_residents(entity).unwrap().len() > 1);
            }
        }
    }
    {
        let backend = MpiSpatialAssemblyBackend::new(&mut group, layout)
            .expect("the authenticated layout matches the physical execution group");
        let finalized = {
            let capture = CapturingAssemblyBackend::new(&backend);
            let finalized = finalize_resolved_fixed_reference_fsi_step_2d_with_assembly(
                &canonical,
                &execution.resolved,
                execution.mesh_reference,
                &spatial.mesh,
                &spatial.partition,
                &previous,
                &capture,
            )
            .expect("physical MPI owner-routed assembly passes the canonical FSI boundary");
            let candidate = capture
                .take()
                .expect("MPI assembly exposes its accepted reconstructed targets");
            assert_eq!(candidate.systems().len(), reference_systems.len());
            for (candidate, reference) in candidate.systems().iter().zip(&reference_systems) {
                assert_system_bits(candidate, reference);
            }
            assert_eq!(
                finalized.linear_system().agreement_fingerprint(),
                reference_fingerprint
            );
            finalized
        };

        let evidence = backend
            .accepted_evidence()
            .expect("MPI assembly evidence remains readable")
            .expect("successful physical assembly publishes accepted evidence");
        assert_evidence(&evidence, partitions);

        let solution = finalized
            .solve(&REFERENCE_LINEAR_SOLVER)
            .expect("the reconstructed host operator passes unchanged FSI acceptance");
        let numerical = solution.numerical_evidence();
        assert!(numerical.residual_norm() < 1.0e-9);
        assert!(numerical.continuity_residual_norm() < 1.0e-9);
        assert!(numerical.kinematic_residual_norm() < 1.0e-14);
        assert_eq!(numerical.interface_velocity_jump_norm(), 0.0);
        assert!(numerical.interface_action_imbalance_norm() < 1.0e-9);
        assert!(numerical.energy_balance().defect().abs() < 1.0e-9);
    }

    world.barrier();
    drop(group);
}

fn assert_layout_disagreement_returns_on_every_rank(
    world: &impl CommunicatorCollectives,
    group: &mut MpiExecutionGroup,
    mesh_sha256: [u8; 32],
    mesh: &eqiora::meshing::SimplicialMesh,
) {
    let mut claimed_revision = mesh_sha256;
    if group.partition().index() == group.partitions().get() - 1 {
        claimed_revision[0] ^= 1;
    }
    let layout = layout(mesh, group.partitions(), claimed_revision);
    let error = match MpiSpatialAssemblyBackend::new(group, layout) {
        Ok(_) => panic!("a rank-local foreign mesh revision must fail collective admission"),
        Err(error) => error,
    };
    assert_eq!(error.code(), eqiora::diagnostic::codes::ASSEMBLY_FAILED);
    assert_common_diagnostic(world, &error);
}

fn layout(
    mesh: &eqiora::meshing::SimplicialMesh,
    partitions: NonZeroUsize,
    mesh_sha256: [u8; 32],
) -> DistributedMeshLayout {
    DistributedMeshLayout::derive(
        MeshRevisionIdentityV1::from_sha256(mesh_sha256),
        mesh,
        partitions,
        cell_claims(partitions),
    )
    .expect("exact cell ownership derives one complete distributed mesh layout")
}

fn cell_claims(partitions: NonZeroUsize) -> Vec<CellOwnershipClaim> {
    (0..8)
        .map(|cell| {
            let owner = match partitions.get() {
                1 => 0,
                2 => usize::from(cell < 4),
                4 => cell % 4,
                _ => unreachable!("the registered fixture admits only one/two/four ranks"),
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

fn assert_common_diagnostic(world: &impl CommunicatorCollectives, error: &Diagnostic) {
    const DIAGNOSTIC_BYTES: usize = 256;
    let rendered = format!("{}:{}", error.code(), error.message());
    assert!(rendered.len() <= DIAGNOSTIC_BYTES);
    let ranks = usize::try_from(world.size()).unwrap();
    let mut diagnostics = vec![0_u8; ranks * DIAGNOSTIC_BYTES];
    let mut local = [0_u8; DIAGNOSTIC_BYTES];
    local[..rendered.len()].copy_from_slice(rendered.as_bytes());
    world.all_gather_into(&local[..], &mut diagnostics[..]);
    assert!(
        diagnostics
            .as_chunks::<DIAGNOSTIC_BYTES>()
            .0
            .iter()
            .map(|diagnostic| diagnostic.as_slice())
            .all(|diagnostic| diagnostic == local)
    );
}

#[derive(Debug)]
struct CapturingAssemblyBackend<'a> {
    inner: &'a dyn AssemblyBackend,
    accepted: RefCell<Option<AssemblyResult>>,
}

impl<'a> CapturingAssemblyBackend<'a> {
    fn new(inner: &'a dyn AssemblyBackend) -> Self {
        Self {
            inner,
            accepted: RefCell::new(None),
        }
    }

    fn take(&self) -> Option<AssemblyResult> {
        self.accepted.borrow_mut().take()
    }
}

impl AssemblyBackend for CapturingAssemblyBackend<'_> {
    fn assemble(
        &self,
        plan: &AssemblyPlan,
        work: &dyn AssemblyWork,
    ) -> Result<AssemblyResult, Diagnostic> {
        let result = self.inner.assemble(plan, work)?;
        *self.accepted.borrow_mut() = Some(result.clone());
        Ok(result)
    }
}

struct ChildOutput {
    status: ExitStatus,
    stdout: BoundedOutput,
    stderr: BoundedOutput,
}

struct BoundedOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

fn run_mpi_child(ranks: usize) -> ChildOutput {
    let executable = env::current_exe().expect("integration-test executable is available");
    let launcher = env::var_os("EQIORA_MPI_LAUNCHER").unwrap_or_else(|| "mpirun".into());
    let mut command = Command::new(&launcher);
    if launcher_accepts_oversubscribe(&launcher) {
        command.arg("--oversubscribe");
    }
    let mut child = command
        .args(["-n", &ranks.to_string()])
        .arg(executable)
        .args(["--exact", CHILD_TEST, "--nocapture"])
        .env(CHILD_ENV, "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("registered MPI evidence requires mpirun on PATH");
    let stdout = child.stdout.take().expect("MPI child stdout is captured");
    let stderr = child.stderr.take().expect("MPI child stderr is captured");
    let stdout_reader = thread::spawn(move || drain_bounded(stdout, CHILD_OUTPUT_LIMIT));
    let stderr_reader = thread::spawn(move || drain_bounded(stderr, CHILD_OUTPUT_LIMIT));
    let started = Instant::now();
    let (status, timed_out) = loop {
        if let Some(status) = child.try_wait().expect("MPI child status is readable") {
            break (status, false);
        }
        if started.elapsed() >= CHILD_TIMEOUT {
            child.kill().expect("timed-out MPI launcher can be killed");
            break (
                child.wait().expect("the killed MPI launcher is reaped"),
                true,
            );
        }
        thread::sleep(Duration::from_millis(20));
    };
    let stdout = stdout_reader
        .join()
        .expect("MPI stdout reader does not panic")
        .expect("MPI child stdout remains readable");
    let stderr = stderr_reader
        .join()
        .expect("MPI stderr reader does not panic")
        .expect("MPI child stderr remains readable");
    if timed_out {
        panic!(
            "{ranks}-rank fixed-reference FSI assembly MPI child exceeded {CHILD_TIMEOUT:?}\nstdout{}:\n{}\nstderr{}:\n{}",
            truncation_marker(&stdout),
            String::from_utf8_lossy(&stdout.bytes),
            truncation_marker(&stderr),
            String::from_utf8_lossy(&stderr.bytes),
        );
    }
    ChildOutput {
        status,
        stdout,
        stderr,
    }
}

fn launcher_accepts_oversubscribe(launcher: &std::ffi::OsStr) -> bool {
    Command::new(launcher)
        .args(["--oversubscribe", "--version"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn assert_success(ranks: usize, output: ChildOutput) {
    assert!(
        output.status.success(),
        "{ranks}-rank fixed-reference FSI assembly MPI child failed\nstdout{}:\n{}\nstderr{}:\n{}",
        truncation_marker(&output.stdout),
        String::from_utf8_lossy(&output.stdout.bytes),
        truncation_marker(&output.stderr),
        String::from_utf8_lossy(&output.stderr.bytes),
    );
}

fn drain_bounded(mut reader: impl Read, maximum: usize) -> std::io::Result<BoundedOutput> {
    let mut bytes = Vec::with_capacity(maximum);
    let mut truncated = false;
    let mut chunk = [0_u8; 8 * 1024];
    loop {
        let count = reader.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        let retained = count.min(maximum.saturating_sub(bytes.len()));
        bytes.extend_from_slice(&chunk[..retained]);
        truncated |= retained != count;
    }
    Ok(BoundedOutput { bytes, truncated })
}

fn truncation_marker(output: &BoundedOutput) -> &'static str {
    if output.truncated { " (truncated)" } else { "" }
}
