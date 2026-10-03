use std::collections::BTreeMap;
use std::ops::Range;

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_solver::{
    DiagonalAvailability, LinearOperator, OrientedLinearOperator, RowLinearAction,
};

use crate::{AssemblyPacketSetIdentityV1, AssemblyPlan, AssemblyTargetId, AssemblyWork, DofId};

/// One projected packet row in canonical global-column order.
#[derive(Debug)]
struct PacketActionRow {
    row: DofId,
    entries: Vec<(DofId, f64)>,
}

/// Matrix-free complete-vector action over projected local packets.
///
/// The operator stores packet-local mapped entries, not global sparse storage.
/// Packet order, row order, and column order are fixed at construction. Normal
/// action therefore performs an ordered additive scatter; transpose action
/// reverses the mathematical arrows without changing their logical order.
///
/// This is a host-local complete-vector contract. It contains no mesh,
/// physics, runtime, partition, device, or durable artifact identity.
#[derive(Debug)]
pub struct PacketLinearOperator {
    size: usize,
    packets: Vec<Vec<PacketActionRow>>,
    diagonal: Vec<f64>,
}

impl PacketLinearOperator {
    fn apply_row_range(
        &self,
        rows: Range<usize>,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        let row_count = rows
            .end
            .checked_sub(rows.start)
            .ok_or_else(|| solve_failed("packet action requires a nondecreasing row range"))?;
        if rows.end > self.size || input.len() != self.size || output.len() != row_count {
            return Err(solve_failed(format!(
                "packet action for {:?} of a {}x{} operator has input/output sizes {}/{}",
                rows,
                self.size,
                self.size,
                input.len(),
                output.len()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(solve_failed("packet linear-action input must be finite"));
        }

        output.fill(0.0);
        for packet in &self.packets {
            for row in packet {
                let global_row = row.row.index();
                if global_row < rows.start || global_row >= rows.end {
                    continue;
                }
                let mut value = 0.0;
                for &(column, coefficient) in &row.entries {
                    value += coefficient * input[column.index()];
                }
                let output_value = &mut output[global_row - rows.start];
                *output_value += value;
                if !output_value.is_finite() {
                    return Err(solve_failed(format!(
                        "packet linear action became non-finite at global row {global_row}"
                    )));
                }
            }
        }
        Ok(())
    }
}

impl LinearOperator for PacketLinearOperator {
    type Scalar = f64;

    fn rows(&self) -> usize {
        self.size
    }

    fn columns(&self) -> usize {
        self.size
    }

    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        self.apply_row_range(0..self.size, input, output)
    }

    fn row_action(&self) -> Option<&dyn RowLinearAction<Scalar = f64>> {
        Some(self)
    }

    fn diagonal(&self, output: &mut [f64]) -> Result<DiagonalAvailability, Diagnostic> {
        if output.len() != self.size {
            return Err(solve_failed(format!(
                "packet operator diagonal requires {} values, received {}",
                self.size,
                output.len()
            )));
        }
        output.copy_from_slice(&self.diagonal);
        Ok(DiagonalAvailability::Available)
    }
}

impl RowLinearAction for PacketLinearOperator {
    type Scalar = f64;

    fn apply_rows(
        &self,
        rows: Range<usize>,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        self.apply_row_range(rows, input, output)
    }
}

impl OrientedLinearOperator for PacketLinearOperator {
    fn supports_orientation(&self, _orientation: eqiora_solver::LinearOperatorOrientation) -> bool {
        true
    }

    fn apply_oriented(
        &self,
        orientation: eqiora_solver::LinearOperatorOrientation,
        input: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        if orientation == eqiora_solver::LinearOperatorOrientation::Normal {
            return self.apply(input, output);
        }
        if input.len() != self.size || output.len() != self.size {
            return Err(solve_failed(format!(
                "transposed packet action is {}x{} but input/output have {}/{} values",
                self.size,
                self.size,
                input.len(),
                output.len()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(solve_failed(
                "transposed packet linear-action input must be finite",
            ));
        }

        output.fill(0.0);
        for packet in &self.packets {
            for row in packet {
                let input_value = input[row.row.index()];
                for &(column, coefficient) in &row.entries {
                    let output_value = &mut output[column.index()];
                    *output_value += coefficient * input_value;
                    if !output_value.is_finite() {
                        return Err(solve_failed(format!(
                            "transposed packet linear action became non-finite at global row {}",
                            column.index()
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

/// One complete RHS and its matrix-free packet operator.
///
/// Construction evaluates each [`AssemblyWork`] packet once in increasing
/// logical order and uses [`crate::AssemblyPacket::project`] as the sole
/// constraint and local-to-global projection. Free-column coefficients become
/// the homogeneous operator; fixed-column contributions are moved only to the
/// right-hand side. No global CSR matrix is built or retained.
#[derive(Debug)]
pub struct PacketLinearSystem {
    operator: PacketLinearOperator,
    right_hand_side: Vec<f64>,
    packet_set: AssemblyPacketSetIdentityV1,
}

impl PacketLinearSystem {
    /// Project one target of pure indexed assembly work into a matrix-free
    /// host-local system.
    ///
    /// # Errors
    /// Returns `EQ0806` for an invalid target, empty or drifting work, packet
    /// evaluation/projection failure, structurally empty global row, failure
    /// to reserve retained storage, or non-finite accumulation. No partial
    /// system escapes.
    pub fn from_work(
        plan: &AssemblyPlan,
        target: AssemblyTargetId,
        work: &dyn AssemblyWork,
    ) -> Result<Self, Diagnostic> {
        let target_shape = plan.target(target).ok_or_else(|| {
            assembly_failed(format!(
                "packet linear system target {} is outside plan count {}",
                target.index(),
                plan.target_count()
            ))
        })?;
        let size = target_shape.size();
        let packet_count = work.packet_count();
        if packet_count == 0 {
            return Err(assembly_failed(
                "packet linear system requires at least one logical packet",
            ));
        }
        let packet_set = work.packet_set_identity();

        let mut right_hand_side = Vec::new();
        right_hand_side
            .try_reserve_exact(size)
            .map_err(|_| assembly_failed("packet system RHS exceeds allocation capacity"))?;
        right_hand_side.resize(size, 0.0);

        let mut diagonal = Vec::new();
        diagonal
            .try_reserve_exact(size)
            .map_err(|_| assembly_failed("packet diagonal exceeds allocation capacity"))?;
        diagonal.resize(size, 0.0);

        let mut packets = Vec::new();
        packets
            .try_reserve_exact(packet_count)
            .map_err(|_| assembly_failed("packet action exceeds allocation capacity"))?;
        let mut structural_entries = BTreeMap::<(DofId, DofId), f64>::new();

        for packet_index in 0..packet_count {
            let packet = work.evaluate(packet_index)?;
            let projected = packet.project(plan)?;
            let selected = projected.into_iter().find(|delta| delta.target() == target);
            let Some(selected) = selected else {
                packets.push(Vec::new());
                continue;
            };

            let mut action_rows = Vec::new();
            action_rows
                .try_reserve_exact(selected.delta().rows().len())
                .map_err(|_| assembly_failed("packet rows exceed allocation capacity"))?;
            for row in selected.delta().rows() {
                let global_row = row.row().index();
                let rhs = right_hand_side[global_row] + row.rhs();
                if !rhs.is_finite() {
                    return Err(assembly_failed(format!(
                        "packet RHS became non-finite at global row {global_row}"
                    )));
                }
                right_hand_side[global_row] = rhs;

                let mut entries = Vec::new();
                entries
                    .try_reserve_exact(row.entries().len())
                    .map_err(|_| {
                        assembly_failed("packet row entries exceed allocation capacity")
                    })?;
                for &(column, coefficient) in row.entries() {
                    let accumulated = structural_entries.entry((row.row(), column)).or_insert(0.0);
                    *accumulated += coefficient;
                    if !accumulated.is_finite() {
                        return Err(assembly_failed(format!(
                            "packet structural accumulation became non-finite at global entry ({global_row}, {})",
                            column.index()
                        )));
                    }
                    if column == row.row() {
                        diagonal[global_row] += coefficient;
                        if !diagonal[global_row].is_finite() {
                            return Err(assembly_failed(format!(
                                "packet diagonal became non-finite at global row {global_row}"
                            )));
                        }
                    }
                    entries.push((column, coefficient));
                }
                action_rows.push(PacketActionRow {
                    row: row.row(),
                    entries,
                });
            }
            packets.push(action_rows);
        }

        if work.packet_count() != packet_count || work.packet_set_identity() != packet_set {
            return Err(assembly_failed(
                "assembly work identity or packet count changed during packet projection",
            ));
        }
        let mut structurally_nonempty = Vec::new();
        structurally_nonempty
            .try_reserve_exact(size)
            .map_err(|_| assembly_failed("packet row gate exceeds allocation capacity"))?;
        structurally_nonempty.resize(size, false);
        for ((row, _), value) in structural_entries {
            if value != 0.0 {
                structurally_nonempty[row.index()] = true;
            }
        }
        if let Some(row) = structurally_nonempty.iter().position(|present| !present) {
            return Err(assembly_failed(format!(
                "packet global row {row} has no nonzero entry after canonical accumulation"
            )));
        }

        Ok(Self {
            operator: PacketLinearOperator {
                size,
                packets,
                diagonal,
            },
            right_hand_side,
            packet_set,
        })
    }

    /// Matrix-free complete-vector operator.
    #[must_use]
    pub const fn operator(&self) -> &PacketLinearOperator {
        &self.operator
    }

    /// Constraint-aware projected right-hand side.
    #[must_use]
    pub fn right_hand_side(&self) -> &[f64] {
        &self.right_hand_side
    }

    /// Identity of the ordered packet set supplied by the producer.
    #[must_use]
    pub const fn packet_set_identity(&self) -> AssemblyPacketSetIdentityV1 {
        self.packet_set
    }

    /// Number of evaluated logical packets.
    #[must_use]
    pub fn packet_count(&self) -> usize {
        self.operator.packets.len()
    }
}

fn assembly_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::ASSEMBLY_FAILED, message)
}

fn solve_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AssemblyBackend, AssemblyMap, AssemblyPacket, AssemblyTarget, IndexedAssemblyWork,
        LocalContribution, LocalUnknown, REFERENCE_ASSEMBLY_BACKEND, TargetAssemblyMap,
    };

    fn nonsymmetric_work(plan: &AssemblyPlan) -> (AssemblyTargetId, Vec<AssemblyPacket>) {
        let target = plan.target_id(0).expect("one-target plan");
        let first = AssemblyPacket::new(
            LocalContribution::new(
                3,
                4,
                vec![
                    2.0, 3.0, 5.0, 7.0, 13.0, 17.0, 19.0, 23.0, 31.0, 37.0, 41.0, 43.0,
                ],
                vec![11.0, 29.0, 47.0],
            )
            .unwrap(),
            vec![TargetAssemblyMap::new(
                target,
                AssemblyMap::new(
                    vec![Some(DofId::new(2)), None, Some(DofId::new(2))],
                    vec![
                        LocalUnknown::Free(DofId::new(1)),
                        LocalUnknown::Fixed(2.0),
                        LocalUnknown::Free(DofId::new(1)),
                        LocalUnknown::Free(DofId::new(0)),
                    ],
                )
                .unwrap(),
            )],
        )
        .unwrap();
        let second = AssemblyPacket::new(
            LocalContribution::new(
                2,
                3,
                vec![53.0, 59.0, 61.0, 67.0, 71.0, 73.0],
                vec![79.0, 83.0],
            )
            .unwrap(),
            vec![TargetAssemblyMap::new(
                target,
                AssemblyMap::new(
                    vec![Some(DofId::new(0)), Some(DofId::new(1))],
                    vec![
                        LocalUnknown::Free(DofId::new(2)),
                        LocalUnknown::Free(DofId::new(0)),
                        LocalUnknown::Fixed(-1.5),
                    ],
                )
                .unwrap(),
            )],
        )
        .unwrap();
        let third = AssemblyPacket::new(
            LocalContribution::new(1, 2, vec![2.0, -3.0], vec![5.0]).unwrap(),
            vec![TargetAssemblyMap::new(
                target,
                AssemblyMap::new(
                    vec![Some(DofId::new(2))],
                    vec![
                        LocalUnknown::Free(DofId::new(1)),
                        LocalUnknown::Free(DofId::new(0)),
                    ],
                )
                .unwrap(),
            )],
        )
        .unwrap();
        (target, vec![first, second, third])
    }

    #[test]
    fn packet_system_matches_independent_nonsymmetric_oracle() {
        let plan = AssemblyPlan::new(vec![AssemblyTarget::new(3).unwrap()]).unwrap();
        let (target, packets) = nonsymmetric_work(&plan);
        let work = IndexedAssemblyWork::new(packets.len(), |index: usize| {
            Ok::<_, Diagnostic>(packets[index].clone())
        });

        let system = PacketLinearSystem::from_work(&plan, target, &work).unwrap();
        assert_eq!(system.packet_count(), 3);
        assert_eq!(system.right_hand_side(), &[170.5, 192.5, -17.0]);

        let input = [2.0, -3.0, 5.0];
        let mut output = [f64::INFINITY; 3];
        system.operator().apply(&input, &mut output).unwrap();
        assert_eq!(output, [383.0, 477.0, -149.0]);

        let cotangent = [7.0, 11.0, -13.0];
        let mut transposed = [f64::INFINITY; 3];
        system
            .operator()
            .apply_oriented(
                eqiora_solver::LinearOperatorOrientation::Transposed,
                &cotangent,
                &mut transposed,
            )
            .unwrap();
        assert_eq!(transposed, [583.0, -1053.0, 1108.0]);
        assert_eq!(
            output
                .iter()
                .zip(cotangent)
                .map(|(left, right)| left * right)
                .sum::<f64>(),
            input
                .iter()
                .zip(transposed)
                .map(|(left, right)| left * right)
                .sum::<f64>()
        );

        let mut selected_rows = [f64::INFINITY; 2];
        system
            .operator()
            .apply_rows(1..3, &input, &mut selected_rows)
            .unwrap();
        assert_eq!(selected_rows, [477.0, -149.0]);
        let mut diagonal = [f64::INFINITY; 3];
        assert_eq!(
            system.operator().diagonal(&mut diagonal).unwrap(),
            DiagonalAvailability::Available
        );
        assert_eq!(diagonal, [59.0, 0.0, 0.0]);

        let assembled = REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work).unwrap();
        let assembled = assembled.system(target).unwrap();
        assert_eq!(assembled.rhs(), system.right_hand_side());
        assert_eq!(assembled.matrix().multiply(&input).unwrap(), output);
        let mut assembled_transpose = [0.0; 3];
        assembled
            .matrix()
            .apply_oriented(
                eqiora_solver::LinearOperatorOrientation::Transposed,
                &cotangent,
                &mut assembled_transpose,
            )
            .unwrap();
        assert_eq!(assembled_transpose, transposed);
    }

    #[test]
    fn packet_system_fails_closed_before_or_during_action() {
        let plan = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
        let target = plan.target_id(0).unwrap();
        let empty = IndexedAssemblyWork::new(0, |_| unreachable!());
        assert_eq!(
            PacketLinearSystem::from_work(&plan, target, &empty)
                .unwrap_err()
                .code(),
            codes::ASSEMBLY_FAILED
        );

        let out_of_range = IndexedAssemblyWork::new(1, |_| {
            AssemblyPacket::new(
                LocalContribution::new(1, 1, vec![1.0], vec![0.0])?,
                vec![TargetAssemblyMap::new(
                    target,
                    AssemblyMap::new(
                        vec![Some(DofId::new(1))],
                        vec![LocalUnknown::Free(DofId::new(0))],
                    )?,
                )],
            )
        });
        assert_eq!(
            PacketLinearSystem::from_work(&plan, target, &out_of_range)
                .unwrap_err()
                .code(),
            codes::ASSEMBLY_FAILED
        );

        let incomplete_plan = AssemblyPlan::new(vec![AssemblyTarget::new(2).unwrap()]).unwrap();
        let incomplete_target = incomplete_plan.target_id(0).unwrap();
        let missing_row = IndexedAssemblyWork::new(1, |_| {
            AssemblyPacket::new(
                LocalContribution::new(1, 1, vec![1.0], vec![0.0])?,
                vec![TargetAssemblyMap::new(
                    incomplete_target,
                    AssemblyMap::new(
                        vec![Some(DofId::new(0))],
                        vec![LocalUnknown::Free(DofId::new(0))],
                    )?,
                )],
            )
        });
        assert_eq!(
            PacketLinearSystem::from_work(&incomplete_plan, incomplete_target, &missing_row)
                .unwrap_err()
                .code(),
            codes::ASSEMBLY_FAILED
        );

        let two_target_plan = AssemblyPlan::new(vec![
            AssemblyTarget::new(1).unwrap(),
            AssemblyTarget::new(1).unwrap(),
        ])
        .unwrap();
        let absent_target = two_target_plan.target_id(0).unwrap();
        let other_target = two_target_plan.target_id(1).unwrap();
        let absent = IndexedAssemblyWork::new(1, |_| {
            AssemblyPacket::new(
                LocalContribution::new(1, 1, vec![1.0], vec![0.0])?,
                vec![TargetAssemblyMap::new(
                    other_target,
                    AssemblyMap::new(
                        vec![Some(DofId::new(0))],
                        vec![LocalUnknown::Free(DofId::new(0))],
                    )?,
                )],
            )
        });
        assert_eq!(
            PacketLinearSystem::from_work(&two_target_plan, absent_target, &absent)
                .unwrap_err()
                .code(),
            codes::ASSEMBLY_FAILED
        );

        let cancellation = IndexedAssemblyWork::new(2, |index| {
            AssemblyPacket::new(
                LocalContribution::new(1, 1, vec![if index == 0 { 1.0 } else { -1.0 }], vec![0.0])?,
                vec![TargetAssemblyMap::new(
                    target,
                    AssemblyMap::new(
                        vec![Some(DofId::new(0))],
                        vec![LocalUnknown::Free(DofId::new(0))],
                    )?,
                )],
            )
        });
        assert_eq!(
            PacketLinearSystem::from_work(&plan, target, &cancellation)
                .unwrap_err()
                .code(),
            codes::ASSEMBLY_FAILED
        );

        let valid = IndexedAssemblyWork::new(1, |_| {
            AssemblyPacket::new(
                LocalContribution::new(1, 1, vec![1.0], vec![0.0])?,
                vec![TargetAssemblyMap::new(
                    target,
                    AssemblyMap::new(
                        vec![Some(DofId::new(0))],
                        vec![LocalUnknown::Free(DofId::new(0))],
                    )?,
                )],
            )
        });
        let system = PacketLinearSystem::from_work(&plan, target, &valid).unwrap();
        assert_eq!(
            system
                .operator()
                .apply(&[f64::NAN], &mut [0.0])
                .unwrap_err()
                .code(),
            codes::NUMERICAL_SOLVE_FAILED
        );
        assert_eq!(
            system
                .operator()
                .apply(&[1.0], &mut [0.0, 0.0])
                .unwrap_err()
                .code(),
            codes::NUMERICAL_SOLVE_FAILED
        );
    }
}
