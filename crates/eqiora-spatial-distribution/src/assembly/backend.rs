use std::sync::Mutex;

use eqiora_assembly::{AssemblyBackend, AssemblyPlan, AssemblyResult, AssemblyWork};
use eqiora_core::Diagnostic;
use eqiora_distributed::PartitionId;
use eqiora_solver::{ExecutionId, ExecutionReport};

use crate::DistributedMeshLayout;

use super::codec::invalid;
use super::ownership::{AdmittedRowOwnership, LocalAssemblyProjection};
use super::route::{AssemblyRowRouteV1, DistributedAssemblyRoutePlanV1};
use super::shard::{DistributedAssemblyEvidence, reconstruct_distributed_assembly};

/// Execution identity of the one-process owner-routing oracle.
pub const LOOPBACK_SPATIAL_ASSEMBLY_EXECUTION: ExecutionId =
    ExecutionId::new("eqiora.spatial-assembly.loopback");

/// One-process oracle that executes the exact transport-neutral route protocol.
#[derive(Debug)]
pub struct LoopbackSpatialAssemblyBackend {
    layout: DistributedMeshLayout,
    accepted: Mutex<Option<DistributedAssemblyEvidence>>,
}

impl LoopbackSpatialAssemblyBackend {
    /// Bind the exact mesh layout used to assign packet producers.
    #[must_use]
    pub fn new(layout: DistributedMeshLayout) -> Self {
        Self {
            layout,
            accepted: Mutex::new(None),
        }
    }

    /// Exact distributed mesh layout used by this backend.
    #[must_use]
    pub const fn layout(&self) -> &DistributedMeshLayout {
        &self.layout
    }

    /// Evidence from the latest successful call, if any.
    ///
    /// # Errors
    /// Returns `EQ0806` only if another thread panicked while holding the
    /// operation/evidence lock. A failed assembly clears earlier evidence.
    pub fn accepted_evidence(&self) -> Result<Option<DistributedAssemblyEvidence>, Diagnostic> {
        Ok(self
            .accepted
            .lock()
            .map_err(|_| invalid("spatial assembly operation lock is poisoned"))?
            .clone())
    }
}

impl AssemblyBackend<f64> for LoopbackSpatialAssemblyBackend {
    fn assemble(
        &self,
        plan: &AssemblyPlan,
        work: &dyn AssemblyWork<f64>,
    ) -> Result<AssemblyResult<f64>, Diagnostic> {
        let mut accepted = self
            .accepted
            .lock()
            .map_err(|_| invalid("spatial assembly operation lock is poisoned"))?;
        *accepted = None;
        let projections = (0..self.layout.partition_count().get())
            .map(|producer| {
                LocalAssemblyProjection::evaluate_owned(
                    &self.layout,
                    plan,
                    work,
                    PartitionId::new(producer),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ownership = AdmittedRowOwnership::admit(&self.layout, plan, &projections)?;
        let local_routes = projections
            .iter()
            .map(|projection| projection.routes(&ownership))
            .collect::<Result<Vec<_>, _>>()?;
        let descriptors = local_routes
            .iter()
            .flatten()
            .map(AssemblyRowRouteV1::descriptor)
            .collect::<Vec<_>>();
        let route_plan =
            DistributedAssemblyRoutePlanV1::seal(&self.layout, plan, &ownership, descriptors)?;
        let admissions = projections
            .iter()
            .zip(&local_routes)
            .map(|(projection, routes)| {
                route_plan.admit_local_routes(projection, &ownership, routes)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Deliberately destroy transport arrival order. Inbox admission and
        // target/row/global-packet folding are responsible for restoring the
        // canonical numerical order.
        let mut transported = local_routes.into_iter().flatten().collect::<Vec<_>>();
        transported.reverse();
        let mut shards = Vec::new();
        for destination in 0..self.layout.partition_count().get() {
            let destination = PartitionId::new(destination);
            let inbox = transported
                .iter()
                .filter(|route| route.descriptor.destination == destination)
                .cloned()
                .collect::<Vec<_>>();
            shards.extend(route_plan.fold_inbox(&ownership, destination, inbox)?);
        }
        let execution = ExecutionReport::distributed(
            LOOPBACK_SPATIAL_ASSEMBLY_EXECUTION,
            self.layout.partition_count(),
        );
        let (systems, evidence) = reconstruct_distributed_assembly(
            &self.layout,
            plan,
            &ownership,
            &route_plan,
            admissions,
            shards,
            execution,
        )?;
        let result =
            AssemblyResult::from_complete_systems(plan, systems, work.packet_count(), execution)?;
        *accepted = Some(evidence);
        Ok(result)
    }
}
