use std::sync::atomic::{AtomicUsize, Ordering};

use eqiora_assembly::{
    AssemblyResult, AssemblyTarget, AssemblyWork, REFERENCE_ASSEMBLY_BACKEND, TargetAssemblyMap,
};

use super::*;
use crate::constrained_dofs::ConstrainedDofLayout;
use crate::form_compiler::region::RegionLinearization;

#[derive(Debug)]
struct RepeatedRequests;

impl AssemblyBackend<f64> for RepeatedRequests {
    fn assemble(
        &self,
        plan: &AssemblyPlan,
        work: &dyn AssemblyWork<f64>,
    ) -> Result<AssemblyResult<f64>, Diagnostic> {
        assert_eq!(work.packet_set_identity(), PACKETS);
        for packet in (0..work.packet_count()).rev() {
            work.evaluate(packet)?;
            work.evaluate(packet)?;
        }
        REFERENCE_ASSEMBLY_BACKEND.assemble(plan, work)
    }
}

const PACKETS: AssemblyPacketSetIdentityV1 = AssemblyPacketSetIdentityV1::from_sha256([17; 32]);

fn dense(system: &LinearSystem<f64>) -> Vec<f64> {
    let matrix = system.matrix();
    let mut values = vec![0.0; matrix.rows() * matrix.columns()];
    for row in 0..matrix.rows() {
        for slot in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
            values[row * matrix.columns() + matrix.column_indices()[slot]] = matrix.values()[slot];
        }
    }
    values
}

#[test]
fn coupled_actions_share_exact_full_and_reduced_residuals_and_evaluate_once() {
    // R(u,v) = (u^2 + 3v, v^2 + 2u), on overlapping pairs (a,b) and (b,c).
    // At (1,2,3), the full residual is (7, 6+13, 13); a is prescribed.
    let constraints = ConstrainedDofLayout::new(vec![Some(1.0), None, None]).unwrap();
    let plan = AssemblyPlan::new(vec![
        AssemblyTarget::new(2).unwrap(),
        AssemblyTarget::new(3).unwrap(),
    ])
    .unwrap();
    let counts = [AtomicUsize::new(0), AtomicUsize::new(0)];
    let result = assemble_linearization(&plan, PACKETS, 2, &RepeatedRequests, |packet| {
        counts[packet].fetch_add(1, Ordering::Relaxed);
        let keys = [packet, packet + 1];
        let reduced = constraints.reduced_map(&keys)?;
        let point = mapped_local_point(&reduced, &[2.0, 3.0])?;
        let [u, v] = point.as_slice() else {
            unreachable!()
        };
        let action = RegionLinearization {
            jacobian: vec![2.0 * u, 3.0, 2.0, 2.0 * v],
            residual: vec![u * u + 3.0 * v, v * v + 2.0 * u],
        };
        let residual = action.residual.clone();
        Ok(EvaluatedRegionPacket {
            assembly: AssemblyPacket::new(
                action.into_contribution(&point)?,
                vec![
                    TargetAssemblyMap::new(
                        plan.target_id(1).unwrap(),
                        constraints.full_map(&keys)?,
                    ),
                    TargetAssemblyMap::new(plan.target_id(0).unwrap(), reduced),
                ],
            )?,
            residual,
        })
    })
    .unwrap();
    assert_eq!(counts.map(|value| value.into_inner()), [1, 1]);
    assert_eq!(result.residuals, [vec![19.0, 13.0], vec![7.0, 19.0, 13.0]]);
    assert_eq!(dense(&result.systems[0]), [8.0, 3.0, 2.0, 6.0]);
    assert_eq!(
        dense(&result.systems[1]),
        [2.0, 3.0, 0.0, 2.0, 8.0, 3.0, 0.0, 2.0, 6.0]
    );
    assert_eq!(result.systems[0].rhs(), [6.0, 9.0]);
    assert_eq!(result.systems[1].rhs(), [1.0, 8.0, 9.0]);
    assert_eq!(result.report.packet_count(), 2);
    assert_eq!(result.packets.len(), 2);
}

fn scalar_packet(plan: &AssemblyPlan, residual: Vec<f64>) -> EvaluatedRegionPacket {
    let constraints = ConstrainedDofLayout::new(vec![None]).unwrap();
    EvaluatedRegionPacket {
        assembly: AssemblyPacket::new(
            LocalContribution::new(1, 1, vec![1.0], vec![1e16]).unwrap(),
            vec![TargetAssemblyMap::new(
                plan.target_id(0).unwrap(),
                constraints.full_map(&[0]).unwrap(),
            )],
        )
        .unwrap(),
        residual,
    }
}

#[test]
fn primal_residual_survives_loss_in_linearized_rhs() {
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
    let result = assemble_linearization(&plan, PACKETS, 1, &RepeatedRequests, |_| {
        Ok(scalar_packet(&plan, vec![1.0]))
    })
    .unwrap();
    assert_eq!(result.residuals, [vec![1.0]]);
    // Binary64 rounds 1e16-1 to 1e16, so subtraction cannot recover this residual.
    assert_eq!(
        result.systems[0].matrix().values()[0] * 1e16 - result.systems[0].rhs()[0],
        0.0
    );
    assert_eq!(result.packets[0].residual, [1.0]);
}

#[test]
fn invalid_or_failed_packets_never_return_partial_assembly() {
    let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
    for residual in [vec![], vec![1.0, 2.0], vec![f64::NAN], vec![f64::INFINITY]] {
        assert!(
            assemble_linearization(&plan, PACKETS, 1, &RepeatedRequests, |_| {
                Ok(scalar_packet(&plan, residual.clone()))
            })
            .err()
            .unwrap()
            .message()
            .contains("exact finite primal residual")
        );
    }
    let error = assemble_linearization(&plan, PACKETS, 2, &REFERENCE_ASSEMBLY_BACKEND, |packet| {
        if packet == 1 {
            return Err(invalid("local constitutive failure"));
        }
        Ok(scalar_packet(&plan, vec![1.0]))
    })
    .err()
    .unwrap();
    assert!(error.message().contains("local constitutive failure"));
    // Each packet is finite, while their logical residual sum overflows.
    assert!(
        assemble_linearization(&plan, PACKETS, 2, &RepeatedRequests, |_| {
            Ok(scalar_packet(&plan, vec![f64::MAX]))
        })
        .err()
        .unwrap()
        .message()
        .contains("non-finite")
    );
}
