use super::*;
use crate::{DofId, LocalUnknown};

fn target_map(target: AssemblyTargetId, dof: usize) -> TargetAssemblyMap {
    let dof = DofId::new(dof);
    TargetAssemblyMap::new(
        target,
        AssemblyMap::new(vec![Some(dof)], vec![LocalUnknown::Free(dof)]).unwrap(),
    )
}

#[test]
fn packet_canonicalizes_targets_and_rejects_duplicates() {
    let plan = AssemblyPlan::new(vec![
        AssemblyTarget::new(1).unwrap(),
        AssemblyTarget::new(1).unwrap(),
    ])
    .unwrap();
    let first = plan.target_id(0).unwrap();
    let second = plan.target_id(1).unwrap();
    let local = LocalContribution::new(1, 1, vec![1.0], vec![2.0]).unwrap();
    let packet = AssemblyPacket::new(
        local.clone(),
        vec![target_map(second, 0), target_map(first, 0)],
    )
    .unwrap();
    assert_eq!(packet.mappings()[0].target(), first);
    assert_eq!(packet.mappings()[1].target(), second);
    let projected = packet.project(&plan).unwrap();
    assert_eq!(projected[0].target(), first);
    assert_eq!(projected[1].target(), second);
    assert_eq!(projected[0].delta().target_size(), 1);
    assert_eq!(
        AssemblyPacket::new(local, vec![target_map(first, 0), target_map(first, 0)])
            .unwrap_err()
            .code(),
        codes::ASSEMBLY_FAILED
    );
}

#[test]
fn reference_assembly_preserves_packet_accumulation_order() {
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
    let target = plan.target_id(0).unwrap();
    let values = [1.0e16, 1.0, -1.0e16];
    let work = IndexedAssemblyWork::new(values.len(), |index| {
        AssemblyPacket::new(
            LocalContribution::new(1, 1, vec![1.0], vec![values[index]])?,
            vec![target_map(target, 0)],
        )
    });
    let result = REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work).unwrap();
    assert_eq!(result.system(target).unwrap().matrix().values(), &[3.0]);
    assert_eq!(result.system(target).unwrap().rhs(), &[0.0]);
    assert_eq!(result.report().packet_count(), values.len());
    assert_eq!(result.report().target_count(), 1);
}

#[test]
fn prepared_plan_reuses_exact_structure_and_rejects_map_drift() {
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(2).unwrap()]).unwrap();
    let target = plan.target_id(0).unwrap();
    let map = AssemblyMap::new(
        vec![Some(DofId::new(0)), Some(DofId::new(1))],
        vec![
            LocalUnknown::Free(DofId::new(0)),
            LocalUnknown::Free(DofId::new(1)),
        ],
    )
    .unwrap();
    let plan = plan
        .prepare(vec![vec![TargetAssemblyMap::new(target, map.clone())]])
        .unwrap();
    let identity = plan.structure_identity().unwrap().clone();
    for diagonal in [2.0, 3.0] {
        let work = IndexedAssemblyWork::new(1, |_| {
            AssemblyPacket::new(
                LocalContribution::new(2, 2, vec![diagonal, 0.0, 0.0, diagonal], vec![1.0, 1.0])
                    .unwrap(),
                vec![TargetAssemblyMap::new(target, map.clone())],
            )
        });
        let result = REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work).unwrap();
        assert_eq!(result.report().structure_identity(), Some(&identity));
        assert_eq!(result.systems()[0].matrix().values(), &[diagonal, diagonal]);
    }

    let drifted = AssemblyMap::new(
        vec![Some(DofId::new(1)), Some(DofId::new(0))],
        vec![
            LocalUnknown::Free(DofId::new(0)),
            LocalUnknown::Free(DofId::new(1)),
        ],
    )
    .unwrap();
    let work = IndexedAssemblyWork::new(1, |_| {
        AssemblyPacket::new(
            LocalContribution::new(2, 2, vec![1.0, 0.0, 0.0, 1.0], vec![0.0; 2]).unwrap(),
            vec![TargetAssemblyMap::new(target, drifted.clone())],
        )
    });
    assert!(REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work).is_err());
}

#[test]
fn packet_and_projected_scatter_share_ordered_accumulation() {
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
    let target = plan.target_id(0).unwrap();
    let packet = AssemblyPacket::new(
        LocalContribution::new(1, 1, vec![2.0], vec![3.0]).unwrap(),
        vec![target_map(target, 0)],
    )
    .unwrap();
    let projected = packet.project(&plan).unwrap();

    let packet_result = AssemblyAccumulator::new(&plan)
        .unwrap()
        .scatter_packet(0, &packet)
        .unwrap()
        .finish(ExecutionReport::host_serial())
        .unwrap();
    let projected_result = AssemblyAccumulator::new(&plan)
        .unwrap()
        .scatter_projected(0, &projected)
        .unwrap()
        .finish(ExecutionReport::host_serial())
        .unwrap();

    assert_eq!(packet_result.systems(), projected_result.systems());
}

#[test]
fn projected_scatter_still_rejects_out_of_order_packets() {
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
    let foreign_plan = AssemblyPlan::new(vec![
        AssemblyTarget::new(1).unwrap(),
        AssemblyTarget::new(1).unwrap(),
    ])
    .unwrap();
    let foreign_target = foreign_plan.target_id(1).unwrap();
    let packet = AssemblyPacket::new(
        LocalContribution::new(1, 1, vec![1.0], vec![0.0]).unwrap(),
        vec![target_map(foreign_target, 0)],
    )
    .unwrap();

    let diagnostic = AssemblyAccumulator::new(&plan)
        .unwrap()
        .scatter_packet(1, &packet)
        .unwrap_err();
    assert_eq!(diagnostic.code(), codes::ASSEMBLY_FAILED);
    assert_eq!(
        diagnostic.message(),
        "ordered assembly expected packet 0, received 1"
    );

    let valid_target = plan.target_id(0).unwrap();
    let valid_packet = AssemblyPacket::new(
        LocalContribution::new(1, 1, vec![1.0], vec![0.0]).unwrap(),
        vec![target_map(valid_target, 0)],
    )
    .unwrap();
    let valid_projected = valid_packet.project(&plan).unwrap();
    let projected_diagnostic = AssemblyAccumulator::new(&plan)
        .unwrap()
        .scatter_projected(1, &valid_projected)
        .unwrap_err();
    assert_eq!(projected_diagnostic.code(), codes::ASSEMBLY_FAILED);
    assert_eq!(
        projected_diagnostic.message(),
        "ordered assembly expected packet 0, received 1"
    );
}

#[test]
fn projected_scatter_rejects_empty_and_foreign_plan_deltas() {
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
    let empty_diagnostic = AssemblyAccumulator::new(&plan)
        .unwrap()
        .scatter_projected(0, &[])
        .unwrap_err();
    assert_eq!(empty_diagnostic.code(), codes::ASSEMBLY_FAILED);
    assert_eq!(
        empty_diagnostic.message(),
        "projected assembly packet requires at least one target delta"
    );

    let foreign_plan = AssemblyPlan::new(vec![
        AssemblyTarget::new(1).unwrap(),
        AssemblyTarget::new(1).unwrap(),
    ])
    .unwrap();
    let foreign_target = foreign_plan.target_id(1).unwrap();
    let foreign_packet = AssemblyPacket::new(
        LocalContribution::new(1, 1, vec![1.0], vec![0.0]).unwrap(),
        vec![target_map(foreign_target, 0)],
    )
    .unwrap();
    let foreign_projected = foreign_packet.project(&foreign_plan).unwrap();
    let foreign_diagnostic = AssemblyAccumulator::new(&plan)
        .unwrap()
        .scatter_projected(0, &foreign_projected)
        .unwrap_err();
    assert_eq!(foreign_diagnostic.code(), codes::ASSEMBLY_FAILED);
    assert_eq!(
        foreign_diagnostic.message(),
        "projected assembly packet references target 1 outside plan count 1"
    );
}

#[test]
fn empty_work_and_target_mismatch_fail_without_a_result() {
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
    let empty = IndexedAssemblyWork::new(0, |_| unreachable!());
    assert_eq!(
        REFERENCE_ASSEMBLY_BACKEND
            .assemble(&plan, &empty)
            .unwrap_err()
            .code(),
        codes::ASSEMBLY_FAILED
    );

    let foreign_plan = AssemblyPlan::new(vec![
        AssemblyTarget::new(1).unwrap(),
        AssemblyTarget::new(1).unwrap(),
    ])
    .unwrap();
    let foreign = foreign_plan.target_id(1).unwrap();
    let work = IndexedAssemblyWork::new(1, |_| {
        AssemblyPacket::new(
            LocalContribution::new(1, 1, vec![1.0], vec![0.0])?,
            vec![target_map(foreign, 0)],
        )
    });
    assert_eq!(
        REFERENCE_ASSEMBLY_BACKEND
            .assemble(&plan, &work)
            .unwrap_err()
            .code(),
        codes::ASSEMBLY_FAILED
    );
}

fn diagonal_system(size: usize) -> LinearSystem {
    LinearSystem::new(
        crate::CsrMatrix::from_sorted_csr(
            size,
            size,
            (0..=size).collect(),
            (0..size).collect(),
            vec![1.0; size],
        )
        .unwrap(),
        vec![0.0; size],
    )
    .unwrap()
}

#[test]
fn complete_result_constructor_checks_packet_and_target_shape() {
    let plan = AssemblyPlan::new(vec![
        AssemblyTarget::new(1).unwrap(),
        AssemblyTarget::new(2).unwrap(),
    ])
    .unwrap();
    let execution = ExecutionReport::host_serial();

    for result in [
        AssemblyResult::from_complete_systems(
            &plan,
            vec![diagonal_system(1), diagonal_system(2)],
            0,
            execution,
        ),
        AssemblyResult::from_complete_systems(&plan, vec![diagonal_system(1)], 1, execution),
        AssemblyResult::from_complete_systems(
            &plan,
            vec![diagonal_system(2), diagonal_system(1)],
            1,
            execution,
        ),
    ] {
        assert_eq!(result.unwrap_err().code(), codes::ASSEMBLY_FAILED);
    }

    let accepted = AssemblyResult::from_complete_systems(
        &plan,
        vec![diagonal_system(1), diagonal_system(2)],
        3,
        execution,
    )
    .unwrap();
    assert_eq!(accepted.report().packet_count(), 3);
    assert_eq!(accepted.report().target_count(), 2);
}

#[derive(Debug)]
struct FailingWork;

impl AssemblyWork for FailingWork {
    fn packet_set_identity(&self) -> AssemblyPacketSetIdentityV1 {
        AssemblyPacketSetIdentityV1::Unbound
    }

    fn packet_count(&self) -> usize {
        4
    }

    fn evaluate(&self, packet_index: usize) -> Result<AssemblyPacket, Diagnostic> {
        Err(assembly_failed(format!("packet {packet_index} failed")))
    }
}

#[test]
fn reference_reports_the_lowest_failing_packet() {
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
    let diagnostic = REFERENCE_ASSEMBLY_BACKEND
        .assemble(&plan, &FailingWork)
        .unwrap_err();
    assert_eq!(diagnostic.code(), codes::ASSEMBLY_FAILED);
    assert!(diagnostic.message().contains("packet 0 failed"));
}
