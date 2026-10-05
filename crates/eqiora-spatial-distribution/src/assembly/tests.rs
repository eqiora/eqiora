use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

use eqiora_assembly::{
    AssemblyBackend, AssemblyMap, AssemblyPacket, AssemblyPacketSetIdentityV1, AssemblyPlan,
    AssemblyTarget, DofId, IndexedAssemblyWork, LinearSystem, LocalContribution, LocalUnknown,
    REFERENCE_ASSEMBLY_BACKEND, TargetAssemblyMap,
};
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_distributed::PartitionId;
use eqiora_meshing::{MeshEntity, MeshQualityGate, SimplicialMesh};
use eqiora_solver::{CanonicalCsrSystemView, ExecutionReport, LinearOperatorProperties};

use super::*;
use crate::{CellOwnershipClaim, DistributedMeshLayout, MeshRevisionIdentityV1};

fn mesh() -> SimplicialMesh {
    SimplicialMesh::new(
        2,
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
            vec![0.5, 0.5],
        ],
        vec![vec![0, 1, 4], vec![1, 2, 4], vec![2, 3, 4], vec![3, 0, 4]],
        MeshQualityGate::new(0.4).unwrap(),
    )
    .unwrap()
}

fn layout() -> DistributedMeshLayout {
    let mesh = mesh();
    DistributedMeshLayout::derive(
        MeshRevisionIdentityV1::from_sha256([9; 32]),
        &mesh,
        NonZeroUsize::new(2).unwrap(),
        vec![
            CellOwnershipClaim::new(MeshEntity::new(2, 0), PartitionId::new(0)),
            CellOwnershipClaim::new(MeshEntity::new(2, 1), PartitionId::new(1)),
            CellOwnershipClaim::new(MeshEntity::new(2, 2), PartitionId::new(0)),
            CellOwnershipClaim::new(MeshEntity::new(2, 3), PartitionId::new(1)),
        ],
    )
    .unwrap()
}

fn cancellation_work<'a>(
    plan: &'a AssemblyPlan,
    evaluations: &'a AtomicUsize,
) -> IndexedAssemblyWork<impl Fn(usize) -> Result<AssemblyPacket<f64>, Diagnostic> + Sync + 'a> {
    let targets = (0..plan.target_count())
        .map(|index| plan.target_id(index).unwrap())
        .collect::<Vec<_>>();
    IndexedAssemblyWork::for_packet_set(
        AssemblyPacketSetIdentityV1::from_sha256([9; 32]),
        4,
        move |packet| {
            evaluations.fetch_add(1, Ordering::Relaxed);
            let row = usize::from(packet == 3);
            let rhs = [1.0e16, 1.0, -1.0e16, 0.0][packet];
            let dof = DofId::new(row);
            let map = AssemblyMap::new(vec![Some(dof)], vec![LocalUnknown::Free(dof)])?;
            AssemblyPacket::new(
                LocalContribution::new(1, 1, vec![1.0], vec![rhs])?,
                targets
                    .iter()
                    .copied()
                    .map(|target| TargetAssemblyMap::new(target, map.clone()))
                    .collect(),
            )
        },
    )
}

type ProtocolFixture = (
    DistributedMeshLayout,
    AssemblyPlan,
    Vec<LocalAssemblyProjection>,
    AdmittedRowOwnership,
    DistributedAssemblyRoutePlanV1,
    Vec<AssemblyRowRouteV1>,
);

fn protocol_fixture(target_count: usize) -> Result<ProtocolFixture, Diagnostic> {
    let plan = AssemblyPlan::new(
        (0..target_count)
            .map(|_| AssemblyTarget::new(2))
            .collect::<Result<Vec<_>, _>>()?,
    )?;
    let count = AtomicUsize::new(0);
    let work = cancellation_work(&plan, &count);
    let layout = layout();
    let projections = (0..layout.partition_count().get())
        .map(|producer| {
            LocalAssemblyProjection::evaluate_owned(
                &layout,
                &plan,
                &work,
                PartitionId::new(producer),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    drop(work);
    let ownership = AdmittedRowOwnership::admit(&layout, &plan, &projections)?;
    let routes = projections
        .iter()
        .map(|projection| projection.routes(&ownership))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let route_plan = DistributedAssemblyRoutePlanV1::seal(
        &layout,
        &plan,
        &ownership,
        routes.iter().map(AssemblyRowRouteV1::descriptor).collect(),
    )?;
    Ok((layout, plan, projections, ownership, route_plan, routes))
}

fn complete_protocol(
    layout: &DistributedMeshLayout,
    plan: &AssemblyPlan,
    projections: &[LocalAssemblyProjection],
    ownership: &AdmittedRowOwnership,
    route_plan: &DistributedAssemblyRoutePlanV1,
    routes: &[AssemblyRowRouteV1],
    execution: ExecutionReport,
) -> Result<(Vec<LinearSystem<f64>>, DistributedAssemblyEvidence), Diagnostic> {
    let admissions = projections
        .iter()
        .map(|projection| {
            let local_routes = routes
                .iter()
                .filter(|route| route.descriptor().producer() == projection.producer())
                .cloned()
                .collect::<Vec<_>>();
            route_plan.admit_local_routes(projection, ownership, &local_routes)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let shards = collect_shards(layout, ownership, route_plan, routes)?;
    reconstruct_distributed_assembly(
        layout, plan, ownership, route_plan, admissions, shards, execution,
    )
}

fn collect_shards(
    layout: &DistributedMeshLayout,
    ownership: &AdmittedRowOwnership,
    route_plan: &DistributedAssemblyRoutePlanV1,
    routes: &[AssemblyRowRouteV1],
) -> Result<Vec<OwnedRowAssemblyResult>, Diagnostic> {
    let mut shards = Vec::new();
    for destination in 0..layout.partition_count().get() {
        let destination = PartitionId::new(destination);
        let inbox = routes
            .iter()
            .filter(|route| route.descriptor.destination == destination)
            .cloned()
            .collect();
        shards.extend(route_plan.fold_inbox(ownership, destination, inbox)?);
    }
    Ok(shards)
}

#[test]
fn loopback_evaluates_each_cell_once_and_matches_reference_bits() {
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(2).unwrap()]).unwrap();
    let reference_count = AtomicUsize::new(0);
    let reference = REFERENCE_ASSEMBLY_BACKEND
        .assemble(&plan, &cancellation_work(&plan, &reference_count))
        .unwrap();
    let distributed_count = AtomicUsize::new(0);
    let backend = LoopbackSpatialAssemblyBackend::new(layout());
    let distributed = backend
        .assemble(&plan, &cancellation_work(&plan, &distributed_count))
        .unwrap();
    assert_eq!(reference_count.load(Ordering::Relaxed), 4);
    assert_eq!(distributed_count.load(Ordering::Relaxed), 4);
    assert_eq!(distributed.systems(), reference.systems());
    let evidence = backend.accepted_evidence().unwrap().unwrap();
    assert_eq!(evidence.receipt().packet_count(), 4);
    assert_eq!(evidence.receipt().partition_count().get(), 2);
    assert_eq!(
        evidence.receipt().mesh_revision(),
        MeshRevisionIdentityV1::from_sha256([9; 32])
    );
}

#[test]
fn reversed_actual_routes_preserve_reference_fold_and_receipt() -> Result<(), Diagnostic> {
    let (layout, plan, projections, ownership, route_plan, routes) = protocol_fixture(1)?;
    let execution = ExecutionReport::distributed(
        LOOPBACK_SPATIAL_ASSEMBLY_EXECUTION,
        layout.partition_count(),
    );
    let ordered = complete_protocol(
        &layout,
        &plan,
        &projections,
        &ownership,
        &route_plan,
        &routes,
        execution,
    )?;
    let mut reversed = routes;
    reversed.reverse();
    let reversed = complete_protocol(
        &layout,
        &plan,
        &projections,
        &ownership,
        &route_plan,
        &reversed,
        execution,
    )?;
    assert_eq!(ordered, reversed);
    Ok(())
}

#[test]
fn accepted_target_promotes_only_its_owner_rows_to_distributed_algebra() -> Result<(), Diagnostic> {
    let (layout, plan, projections, ownership, route_plan, routes) = protocol_fixture(1)?;
    let execution = ExecutionReport::distributed(
        LOOPBACK_SPATIAL_ASSEMBLY_EXECUTION,
        layout.partition_count(),
    );
    let (systems, evidence) = complete_protocol(
        &layout,
        &plan,
        &projections,
        &ownership,
        &route_plan,
        &routes,
        execution,
    )?;
    let target = plan.target_id(0).unwrap();
    let complete = CanonicalCsrSystemView::new(
        &systems[0],
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )?;
    let bound = evidence.bind_linear_target(target, &complete)?;

    assert_eq!(bound.target(), target);
    assert_eq!(bound.assembly_receipt(), evidence.receipt());
    assert_eq!(
        bound.assembly_system_identity(),
        evidence.target_system_identity(target).unwrap()
    );
    assert!(bound.system().matches_complete(&complete));
    assert_eq!(
        bound.system().partition().owners(),
        evidence.target_partition(target).unwrap().owners()
    );

    let mut changed_rhs = systems[0].rhs().to_vec();
    changed_rhs[0] += 1.0;
    let changed = LinearSystem::new(systems[0].matrix().clone(), changed_rhs)?;
    let changed = CanonicalCsrSystemView::new(
        &changed,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )?;
    assert!(evidence.bind_linear_target(target, &changed).is_err());
    Ok(())
}

#[test]
fn route_wire_round_trip_and_corruption_fail_closed() -> Result<(), Diagnostic> {
    let (_, plan, _, _, _, routes) = protocol_fixture(1)?;
    let route = &routes[0];
    let descriptor_bytes = route.descriptor().to_bytes()?;
    assert_eq!(
        AssemblyRowRouteDescriptorV1::from_bytes(&plan, &descriptor_bytes)?,
        route.descriptor()
    );
    let bytes = route.to_bytes()?;
    let decoded = AssemblyRowRouteV1::from_bytes(&plan, &bytes)?;
    assert_eq!(decoded, *route);
    let mut corrupt = bytes.clone();
    let rhs_start = corrupt.len() - 8;
    corrupt[rhs_start..].copy_from_slice(&f64::NAN.to_bits().to_be_bytes());
    assert!(AssemblyRowRouteV1::from_bytes(&plan, &corrupt).is_err());
    assert!(AssemblyRowRouteV1::from_bytes(&plan, &bytes[..bytes.len() - 1]).is_err());
    Ok(())
}

#[test]
fn route_inventory_falsifiers_fail_closed() -> Result<(), Diagnostic> {
    let (layout, plan, projections, ownership, route_plan, routes) = protocol_fixture(2)?;
    let destination = routes[0].descriptor.destination;
    let expected = || {
        routes
            .iter()
            .filter(|route| route.descriptor.destination == destination)
            .cloned()
            .collect::<Vec<_>>()
    };
    let mut missing = expected();
    missing.pop();
    let mut duplicate = expected();
    duplicate.push(duplicate[0].clone());
    let mut wrong_producer = expected();
    wrong_producer[0].descriptor.producer =
        PartitionId::new(1 - wrong_producer[0].descriptor.producer.index());
    let mut wrong_target = expected();
    wrong_target[0].descriptor.target = plan.target_id(1).unwrap();
    let mut wrong_destination = expected();
    wrong_destination[0].descriptor.destination = PartitionId::new(1 - destination.index());
    let mut wrong_payload = expected();
    wrong_payload[0].rhs *= 2.0;
    let mut wrong_route = expected();
    wrong_route[0].descriptor.packet = (wrong_route[0].descriptor.packet + 1) % 4;
    for inbox in [
        missing,
        duplicate,
        wrong_producer,
        wrong_target,
        wrong_destination,
        wrong_payload,
        wrong_route,
    ] {
        assert_eq!(
            route_plan
                .fold_inbox(&ownership, destination, inbox)
                .unwrap_err()
                .code(),
            codes::ASSEMBLY_FAILED
        );
    }

    let mut descriptors = routes
        .iter()
        .map(AssemblyRowRouteV1::descriptor)
        .collect::<Vec<_>>();
    descriptors[0].destination = PartitionId::new(1 - descriptors[0].destination.index());
    assert!(DistributedAssemblyRoutePlanV1::seal(&layout, &plan, &ownership, descriptors).is_err());
    let mut duplicate_descriptors = routes
        .iter()
        .map(AssemblyRowRouteV1::descriptor)
        .collect::<Vec<_>>();
    duplicate_descriptors.push(duplicate_descriptors[0]);
    assert!(
        DistributedAssemblyRoutePlanV1::seal(&layout, &plan, &ownership, duplicate_descriptors,)
            .is_err()
    );
    let mut missing_descriptors = routes
        .iter()
        .map(AssemblyRowRouteV1::descriptor)
        .collect::<Vec<_>>();
    let missing_producer = missing_descriptors.pop().unwrap().producer();
    let incomplete =
        DistributedAssemblyRoutePlanV1::seal(&layout, &plan, &ownership, missing_descriptors)?;
    let producer_routes = routes
        .iter()
        .filter(|route| route.descriptor().producer() == missing_producer)
        .cloned()
        .collect::<Vec<_>>();
    let projection = projections
        .iter()
        .find(|projection| projection.producer() == missing_producer)
        .expect("fixture has one projection per producer");
    assert!(
        incomplete
            .admit_local_routes(projection, &ownership, &producer_routes)
            .is_err()
    );
    Ok(())
}

#[test]
fn mesh_binding_and_collective_min_proofs_fail_closed() -> Result<(), Diagnostic> {
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(2)?])?;
    let target = plan.target_id(0).unwrap();
    let evaluations = AtomicUsize::new(0);
    let foreign = IndexedAssemblyWork::for_packet_set(
        AssemblyPacketSetIdentityV1::from_sha256([8; 32]),
        4,
        |packet| {
            evaluations.fetch_add(1, Ordering::Relaxed);
            let dof = DofId::new(usize::from(packet == 3));
            AssemblyPacket::new(
                LocalContribution::new(1, 1, vec![1.0], vec![0.0])?,
                vec![TargetAssemblyMap::new(
                    target,
                    AssemblyMap::new(vec![Some(dof)], vec![LocalUnknown::Free(dof)])?,
                )],
            )
        },
    );
    let backend = LoopbackSpatialAssemblyBackend::new(layout());
    assert!(backend.assemble(&plan, &foreign).is_err());
    assert_eq!(evaluations.load(Ordering::Relaxed), 0);
    assert!(backend.accepted_evidence()?.is_none());

    let (layout, plan, projections, _, _, _) = protocol_fixture(1)?;
    let local = projections
        .iter()
        .map(LocalAssemblyProjection::collective_candidates)
        .collect::<Result<Vec<_>, _>>()?;
    let mut minimum = vec![local[0].sentinel(); local[0].values().len()];
    for candidates in &local {
        for (owner, candidate) in minimum.iter_mut().zip(candidates.values()) {
            *owner = (*owner).min(*candidate);
        }
    }
    for (row, wrong_owner) in [(0, 1_u64), (1, 0_u64)] {
        let mut wrong = minimum.clone();
        wrong[row] = wrong_owner;
        let ownership =
            AdmittedRowOwnership::from_collective_min(&layout, &plan, &local[0], &wrong)?;
        let local_routes = projections
            .iter()
            .map(|projection| projection.routes(&ownership))
            .collect::<Result<Vec<_>, _>>()?;
        let route_plan = DistributedAssemblyRoutePlanV1::seal(
            &layout,
            &plan,
            &ownership,
            local_routes
                .iter()
                .flatten()
                .map(AssemblyRowRouteV1::descriptor)
                .collect(),
        )?;
        assert!(
            projections
                .iter()
                .zip(&local_routes)
                .any(|(projection, routes)| route_plan
                    .admit_local_routes(projection, &ownership, routes)
                    .is_err())
        );
    }
    Ok(())
}

#[test]
fn a_target_may_have_an_empty_cell_partition() -> Result<(), Diagnostic> {
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1)?])?;
    let target = plan.target_id(0).unwrap();
    let work = IndexedAssemblyWork::for_packet_set(
        AssemblyPacketSetIdentityV1::from_sha256([9; 32]),
        4,
        |_| {
            let dof = DofId::new(0);
            AssemblyPacket::new(
                LocalContribution::new(1, 1, vec![1.0], vec![1.0])?,
                vec![TargetAssemblyMap::new(
                    target,
                    AssemblyMap::new(vec![Some(dof)], vec![LocalUnknown::Free(dof)])?,
                )],
            )
        },
    );
    let reference = REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work)?;
    let backend = LoopbackSpatialAssemblyBackend::new(layout());
    let distributed = backend.assemble(&plan, &work)?;
    assert_eq!(distributed.systems(), reference.systems());
    let evidence = backend.accepted_evidence()?.unwrap();
    assert_eq!(
        evidence.target_partition(target).unwrap().owners(),
        &[PartitionId::new(0)]
    );
    assert!(evidence.target_shards(target).unwrap()[1].rows().is_empty());
    let complete = CanonicalCsrSystemView::new(
        &distributed.systems()[0],
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )?;
    assert!(evidence.bind_linear_target(target, &complete).is_err());
    Ok(())
}

#[test]
fn malformed_topology_and_shards_fail_closed() -> Result<(), Diagnostic> {
    let (layout, plan, projections, ownership, route_plan, routes) = protocol_fixture(1)?;
    assert!(
        complete_protocol(
            &layout,
            &plan,
            &projections,
            &ownership,
            &route_plan,
            &routes,
            ExecutionReport::host_serial(),
        )
        .is_err()
    );
    assert!(
        complete_protocol(
            &layout,
            &plan,
            &projections,
            &ownership,
            &route_plan,
            &routes,
            ExecutionReport::distributed(LOOPBACK_SPATIAL_ASSEMBLY_EXECUTION, NonZeroUsize::MIN,),
        )
        .is_err()
    );

    let execution = ExecutionReport::distributed(
        LOOPBACK_SPATIAL_ASSEMBLY_EXECUTION,
        layout.partition_count(),
    );
    let admissions = projections
        .iter()
        .map(|projection| {
            let local_routes = routes
                .iter()
                .filter(|route| route.descriptor().producer() == projection.producer())
                .cloned()
                .collect::<Vec<_>>();
            route_plan.admit_local_routes(projection, &ownership, &local_routes)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let admission_bytes = admissions[0].to_bytes()?;
    assert_eq!(
        LocalRouteAdmissionV1::from_bytes(&route_plan, &ownership, &admission_bytes)?,
        admissions[0]
    );
    assert!(
        LocalRouteAdmissionV1::from_bytes(
            &route_plan,
            &ownership,
            &admission_bytes[..admission_bytes.len() - 1],
        )
        .is_err()
    );
    let shards = collect_shards(&layout, &ownership, &route_plan, &routes)?;
    let encoded = shards[0].to_bytes()?;
    assert_eq!(
        OwnedRowAssemblyResult::from_bytes(&plan, route_plan.identity(), &encoded)?,
        shards[0]
    );
    assert!(
        OwnedRowAssemblyResult::from_bytes(
            &plan,
            route_plan.identity(),
            &encoded[..encoded.len() - 1],
        )
        .is_err()
    );

    let mut missing = shards.clone();
    missing.pop();
    let mut duplicate = shards.clone();
    duplicate.push(shards[0].clone());
    let mut wrong_partition = shards.clone();
    wrong_partition[0].partition = PartitionId::new(1);
    let mut wrong_rows = shards.clone();
    wrong_rows[0].rows[0] = DofId::new(wrong_rows[0].global_size);
    let mut malformed_offsets = shards.clone();
    malformed_offsets[0].row_offsets[0] = 1;
    let mut wrong_plan = shards.clone();
    wrong_plan[0].plan = DistributedAssemblyPlanIdentityV1([0; 32]);
    for malformed in [
        missing,
        duplicate,
        wrong_partition,
        wrong_rows,
        malformed_offsets,
        wrong_plan,
    ] {
        assert!(
            reconstruct_distributed_assembly(
                &layout,
                &plan,
                &ownership,
                &route_plan,
                admissions.clone(),
                malformed,
                execution,
            )
            .is_err()
        );
    }
    let mut missing_admission = admissions.clone();
    missing_admission.pop();
    let mut duplicate_admission = admissions.clone();
    duplicate_admission.push(admissions[0]);
    for malformed in [missing_admission, duplicate_admission] {
        assert!(
            reconstruct_distributed_assembly(
                &layout,
                &plan,
                &ownership,
                &route_plan,
                malformed,
                shards.clone(),
                execution,
            )
            .is_err()
        );
    }
    Ok(())
}
