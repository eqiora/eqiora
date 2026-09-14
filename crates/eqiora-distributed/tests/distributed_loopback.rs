use eqiora_core::ScalarType;
use std::num::NonZeroUsize;

use eqiora_distributed::{
    DistributedCsr, DistributedLinearSystem, GlobalVectorSpace, LoopbackExecutor,
    OwnedLinearSystemShard, Partition, PartitionId,
};
use eqiora_solver::{
    CanonicalCsrSystemView, CompleteCsrStorage, LinearOperatorProperties, LinearSolver,
    PreconditionerPolicy, ReductionPolicy, SolverPlan,
};

#[test]
fn partitioned_csr_halo_action_matches_global_reference() {
    let dimension = 12;
    let (offsets, columns, values) = tridiagonal(dimension);
    let input = (0..dimension)
        .map(|index| index as f64 + 0.25)
        .collect::<Vec<_>>();
    let expected = global_apply(&offsets, &columns, &values, &input);
    let space = GlobalVectorSpace::new(NonZeroUsize::new(dimension).unwrap(), ScalarType::F64);

    for partition_count in [1, 2, 4] {
        let partition =
            Partition::balanced_contiguous(space, NonZeroUsize::new(partition_count).unwrap())
                .unwrap();
        let distributed =
            DistributedCsr::from_global_csr(partition, &offsets, &columns, &values).unwrap();
        let actual = LoopbackExecutor.apply(&distributed, &input).unwrap();
        assert_eq!(actual, expected);
        let mut captured_ghost = false;
        for partition_index in 0..partition_count {
            let partition = PartitionId::new(partition_index);
            let shard = distributed.shard(partition).unwrap();
            let capture = shard.capture_execution().unwrap();
            let view = capture.view();
            assert_eq!(view.partition(), partition);
            assert_eq!(view.rows(), view.owned_global_indices().len());
            assert_eq!(
                view.columns(),
                view.owned_global_indices().len() + view.ghost_global_indices().len()
            );
            assert!(
                view.owned_global_indices()
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
            );
            assert!(
                view.ghost_global_indices()
                    .windows(2)
                    .all(|pair| pair[0] < pair[1])
            );
            captured_ghost |= !view.ghost_global_indices().is_empty();

            let owned = view
                .owned_global_indices()
                .iter()
                .map(|global| input[*global])
                .collect::<Vec<_>>();
            let ghosts = view
                .ghost_global_indices()
                .iter()
                .map(|global| input[*global])
                .collect::<Vec<_>>();
            let local_input = owned.iter().chain(&ghosts).copied().collect::<Vec<_>>();
            let mut host_output = vec![0.0; view.rows()];
            shard.apply(&owned, &ghosts, &mut host_output).unwrap();
            let mut captured_output = vec![0.0; view.rows()];
            for (row, output) in captured_output.iter_mut().enumerate() {
                assert!(
                    view.column_indices()[view.row_offsets()[row]..view.row_offsets()[row + 1]]
                        .windows(2)
                        .all(|pair| pair[0] < pair[1])
                );
                for entry in view.row_offsets()[row]..view.row_offsets()[row + 1] {
                    *output += view.values()[entry] * local_input[view.column_indices()[entry]];
                }
            }
            assert_eq!(captured_output, host_output);
            assert_eq!(
                captured_output,
                view.owned_global_indices()
                    .iter()
                    .map(|global| expected[*global])
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(distributed.layouts().len(), partition_count);
        assert_eq!(
            distributed
                .layouts()
                .iter()
                .flat_map(|layout| layout.owned())
                .count(),
            dimension
        );
        if partition_count == 1 {
            assert!(distributed.halo().exchanges().is_empty());
            assert!(!captured_ghost);
        } else {
            assert!(!distributed.halo().exchanges().is_empty());
            assert!(captured_ghost);
            for exchange in distributed.halo().exchanges() {
                assert_ne!(exchange.owner(), exchange.receiver());
                assert!(exchange.indices().windows(2).all(|pair| pair[0] < pair[1]));
            }
        }
    }
}

#[test]
fn local_shard_origin_is_constant_time_and_rejects_equal_clones() {
    let dimension = 4;
    let (offsets, columns, values) = tridiagonal(dimension);
    let space = GlobalVectorSpace::new(NonZeroUsize::new(dimension).unwrap(), ScalarType::F64);
    let partition = Partition::balanced_contiguous(space, NonZeroUsize::new(2).unwrap()).unwrap();
    let first =
        DistributedCsr::from_global_csr(partition.clone(), &offsets, &columns, &values).unwrap();
    let second = DistributedCsr::from_global_csr(partition, &offsets, &columns, &values).unwrap();
    let admitted = first.shard(PartitionId::new(0)).unwrap();
    let repeated = first.shard(PartitionId::new(0)).unwrap();
    let equal_clone = second.shard(PartitionId::new(0)).unwrap();

    assert!(admitted.same_origin(repeated));
    assert_eq!(admitted, equal_clone);
    assert!(!admitted.same_origin(equal_clone));
}

#[test]
fn invalid_partition_and_sparse_structure_fail_closed() {
    let space = GlobalVectorSpace::new(NonZeroUsize::new(3).unwrap(), ScalarType::F64);
    assert!(Partition::balanced_contiguous(space, NonZeroUsize::new(4).unwrap()).is_err());
    assert!(
        Partition::new(
            space,
            NonZeroUsize::new(usize::MAX).unwrap(),
            vec![PartitionId::new(0); 3],
        )
        .is_err()
    );
    let partition = Partition::balanced_contiguous(space, NonZeroUsize::new(2).unwrap()).unwrap();
    assert!(DistributedCsr::from_global_csr(partition, &[0, 1], &[0], &[1.0]).is_err());
}

#[test]
fn balanced_contiguous_partition_puts_longer_ranges_first() {
    let space = GlobalVectorSpace::new(NonZeroUsize::new(5).unwrap(), ScalarType::F64);
    let partition = Partition::balanced_contiguous(space, NonZeroUsize::new(3).unwrap()).unwrap();

    let owners = (0..5)
        .map(|global| partition.owner(global).unwrap().index())
        .collect::<Vec<_>>();
    assert_eq!(owners, [0, 0, 1, 1, 2]);
}

#[test]
fn collective_dot_reduces_unique_owner_contributions_in_partition_order() {
    let space = GlobalVectorSpace::new(NonZeroUsize::new(5).unwrap(), ScalarType::F64);
    let partition = Partition::new(
        space,
        NonZeroUsize::new(3).unwrap(),
        vec![
            PartitionId::new(1),
            PartitionId::new(0),
            PartitionId::new(2),
            PartitionId::new(1),
            PartitionId::new(0),
        ],
    )
    .unwrap();
    let left = [1.0, 2.0, 3.0, 4.0, 5.0];
    let right = [5.0, 4.0, 3.0, 2.0, 1.0];

    let product = LoopbackExecutor
        .dot(&partition, &left, &right, ReductionPolicy::Reproducible)
        .unwrap();

    assert_eq!(product, 35.0);
    assert!(
        LoopbackExecutor
            .dot(&partition, &left, &right, ReductionPolicy::Fast)
            .is_err()
    );
}

#[test]
fn complete_view_derives_arbitrary_owner_shards_and_rhs_in_owned_order() {
    let storage = TridiagonalStorage::new(5, vec![2.0, 3.0, 5.0, 7.0, 11.0]);
    let complete = CanonicalCsrSystemView::new(
        &storage,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    let partition = Partition::new(
        GlobalVectorSpace::new(NonZeroUsize::new(5).unwrap(), ScalarType::F64),
        NonZeroUsize::new(3).unwrap(),
        vec![
            PartitionId::new(2),
            PartitionId::new(0),
            PartitionId::new(1),
            PartitionId::new(2),
            PartitionId::new(0),
        ],
    )
    .unwrap();

    let distributed = DistributedLinearSystem::from_complete(&complete, partition).unwrap();

    assert!(distributed.matches_complete(&complete));
    assert_eq!(
        distributed
            .operator()
            .layouts()
            .iter()
            .map(|layout| layout.owned())
            .collect::<Vec<_>>(),
        vec![&[1, 4][..], &[2][..], &[0, 3][..]]
    );
    assert_eq!(
        distributed
            .local_problem(PartitionId::new(0))
            .unwrap()
            .right_hand_side(),
        &[3.0, 11.0]
    );
    assert_eq!(
        distributed
            .local_problem(PartitionId::new(1))
            .unwrap()
            .right_hand_side(),
        &[5.0]
    );
    assert_eq!(
        distributed
            .local_problem(PartitionId::new(2))
            .unwrap()
            .right_hand_side(),
        &[2.0, 7.0]
    );

    for index in 0..distributed.operator().layouts().len() {
        let shard = distributed
            .operator()
            .shard(PartitionId::new(index))
            .unwrap();
        assert!(std::ptr::eq(
            shard.layout(),
            &distributed.operator().layouts()[index]
        ));
    }

    let rebuilt_partition = Partition::new(
        GlobalVectorSpace::new(NonZeroUsize::new(5).unwrap(), ScalarType::F64),
        NonZeroUsize::new(3).unwrap(),
        vec![
            PartitionId::new(2),
            PartitionId::new(0),
            PartitionId::new(1),
            PartitionId::new(2),
            PartitionId::new(0),
        ],
    )
    .unwrap();
    let rebuilt = DistributedLinearSystem::from_complete(&complete, rebuilt_partition).unwrap();
    assert_eq!(
        rebuilt.partition_identity(),
        distributed.partition_identity()
    );
    assert_eq!(rebuilt.layout_identity(), distributed.layout_identity());
    assert_eq!(
        rebuilt.operator().layouts(),
        distributed.operator().layouts()
    );
    assert_eq!(rebuilt.operator().halo(), distributed.operator().halo());
}

#[test]
fn accepted_owned_rows_are_the_only_source_of_solver_shards() {
    let storage = TridiagonalStorage::new(5, vec![2.0, 3.0, 5.0, 7.0, 11.0]);
    let complete = CanonicalCsrSystemView::new(
        &storage,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    let partition = Partition::new(
        GlobalVectorSpace::new(NonZeroUsize::new(5).unwrap(), ScalarType::F64),
        NonZeroUsize::new(3).unwrap(),
        vec![
            PartitionId::new(2),
            PartitionId::new(0),
            PartitionId::new(1),
            PartitionId::new(2),
            PartitionId::new(0),
        ],
    )
    .unwrap();
    let expected = DistributedLinearSystem::from_complete(&complete, partition.clone()).unwrap();
    let admitted = DistributedLinearSystem::from_owned_shards(
        &complete,
        partition.clone(),
        owned_shards(&storage, &partition),
    )
    .unwrap();

    assert_eq!(admitted.partition_identity(), expected.partition_identity());
    assert_eq!(admitted.layout_identity(), expected.layout_identity());
    assert_eq!(admitted.operator().layouts(), expected.operator().layouts());
    assert_eq!(admitted.operator().halo(), expected.operator().halo());
    assert_eq!(
        LoopbackExecutor
            .apply(admitted.operator(), &[1.0, 2.0, 3.0, 4.0, 5.0])
            .unwrap(),
        LoopbackExecutor
            .apply(expected.operator(), &[1.0, 2.0, 3.0, 4.0, 5.0])
            .unwrap()
    );

    let mut changed = owned_shards(&storage, &partition);
    let row_one = changed
        .iter_mut()
        .find(|shard| shard.rows().contains(&1))
        .unwrap();
    *row_one = OwnedLinearSystemShard::new(
        row_one.partition(),
        NonZeroUsize::new(5).unwrap(),
        vec![1, 4],
        vec![0, 3, 5],
        vec![0, 1, 2, 3, 4],
        vec![-1.0, 2.0, -1.0, -1.0, 2.5],
        vec![3.0, 11.0],
    )
    .unwrap();
    assert!(DistributedLinearSystem::from_owned_shards(&complete, partition, changed).is_err());
}

#[test]
fn agreement_identity_covers_system_partition_layout_and_complete_plan() {
    let storage = TridiagonalStorage::new(4, vec![1.0, 2.0, 3.0, 4.0]);
    let complete = CanonicalCsrSystemView::new(
        &storage,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    let space = GlobalVectorSpace::new(NonZeroUsize::new(4).unwrap(), ScalarType::F64);
    let partition = Partition::balanced_contiguous(space, NonZeroUsize::new(2).unwrap()).unwrap();
    let system = DistributedLinearSystem::from_complete(&complete, partition).unwrap();
    let plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(40).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Jacobi)
    .with_reduction(ReductionPolicy::Reproducible);

    let fingerprint = system.admission_fingerprint(plan).unwrap();
    // Fixed v1 binary-domain goldens: changing one is a protocol change.
    assert_eq!(
        system.partition_identity().as_bytes(),
        [
            138, 234, 172, 64, 196, 215, 142, 18, 191, 172, 54, 141, 137, 250, 94, 170, 74, 189,
            186, 189, 211, 255, 10, 46, 117, 52, 198, 228, 237, 11, 177, 29,
        ]
    );
    assert_eq!(
        system.layout_identity().as_bytes(),
        [
            79, 134, 76, 75, 27, 94, 49, 210, 214, 33, 43, 5, 132, 66, 36, 233, 39, 95, 3, 147,
            104, 106, 177, 181, 160, 58, 248, 46, 150, 82, 119, 170,
        ]
    );
    assert_eq!(
        fingerprint.as_bytes(),
        [
            162, 19, 116, 139, 208, 141, 156, 170, 231, 66, 78, 247, 203, 72, 117, 95, 243, 26,
            198, 228, 218, 178, 16, 190, 163, 138, 176, 188, 97, 195, 91, 177,
        ]
    );
    assert_eq!(system.admission_fingerprint(plan).unwrap(), fingerprint);
    assert_ne!(
        system
            .admission_fingerprint(plan.with_reduction(ReductionPolicy::Fast))
            .unwrap(),
        fingerprint
    );
    let changed_plan = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-11,
        1.0e-14,
        NonZeroUsize::new(40).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Jacobi);
    assert_ne!(
        system.admission_fingerprint(changed_plan).unwrap(),
        fingerprint
    );

    let permuted = Partition::new(
        space,
        NonZeroUsize::new(2).unwrap(),
        vec![
            PartitionId::new(1),
            PartitionId::new(0),
            PartitionId::new(1),
            PartitionId::new(0),
        ],
    )
    .unwrap();
    let permuted = DistributedLinearSystem::from_complete(&complete, permuted).unwrap();
    assert_ne!(permuted.partition_identity(), system.partition_identity());
    assert_ne!(permuted.layout_identity(), system.layout_identity());
    assert_ne!(permuted.admission_fingerprint(plan).unwrap(), fingerprint);

    let changed_storage = TridiagonalStorage::new(4, vec![1.0, 2.0, 3.0, 4.5]);
    let changed_complete = CanonicalCsrSystemView::new(
        &changed_storage,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    assert!(!system.matches_complete(&changed_complete));
}

#[test]
fn distributed_plan_admission_fails_before_execution() {
    let mut storage = TridiagonalStorage::new(3, vec![1.0, 2.0, 3.0]);
    let row = 1;
    let diagonal = storage.offsets[row]
        + storage.columns[storage.offsets[row]..storage.offsets[row + 1]]
            .binary_search(&row)
            .unwrap();
    storage.values[diagonal] = 0.0;
    let complete = CanonicalCsrSystemView::new(
        &storage,
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )
    .unwrap();
    let partition = Partition::balanced_contiguous(
        GlobalVectorSpace::new(NonZeroUsize::new(3).unwrap(), ScalarType::F64),
        NonZeroUsize::new(2).unwrap(),
    )
    .unwrap();
    let system = DistributedLinearSystem::from_complete(&complete, partition).unwrap();
    let jacobi = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(10).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Jacobi);
    assert!(system.admission_fingerprint(jacobi).is_err());

    let general = CanonicalCsrSystemView::new(&storage, LinearOperatorProperties::General).unwrap();
    let partition = Partition::balanced_contiguous(
        GlobalVectorSpace::new(NonZeroUsize::new(3).unwrap(), ScalarType::F64),
        NonZeroUsize::new(2).unwrap(),
    )
    .unwrap();
    let system = DistributedLinearSystem::from_complete(&general, partition).unwrap();
    assert!(system.admission_fingerprint(jacobi).is_err());
}

#[test]
fn symmetric_indefinite_minres_has_one_exact_distributed_admission() {
    let storage = TridiagonalStorage::new(3, vec![1.0, -2.0, 3.0]);
    let complete =
        CanonicalCsrSystemView::new(&storage, LinearOperatorProperties::SymmetricIndefinite)
            .unwrap();
    let partition = Partition::balanced_contiguous(
        GlobalVectorSpace::new(NonZeroUsize::new(3).unwrap(), ScalarType::F64),
        NonZeroUsize::new(2).unwrap(),
    )
    .unwrap();
    let system = DistributedLinearSystem::from_complete(&complete, partition).unwrap();
    let minres = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(20).unwrap(),
    )
    .unwrap();

    let fingerprint = system.admission_fingerprint(minres).unwrap();
    assert_eq!(system.admission_fingerprint(minres).unwrap(), fingerprint);
    assert_ne!(
        system
            .admission_fingerprint(minres.with_reduction(ReductionPolicy::Fast))
            .unwrap(),
        fingerprint
    );
    assert!(
        system
            .admission_fingerprint(minres.with_preconditioner(PreconditionerPolicy::Jacobi))
            .is_err()
    );
}

#[derive(Debug)]
struct TridiagonalStorage {
    dimension: usize,
    offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<f64>,
    rhs: Vec<f64>,
}

impl TridiagonalStorage {
    fn new(dimension: usize, rhs: Vec<f64>) -> Self {
        let (offsets, columns, values) = tridiagonal(dimension);
        Self {
            dimension,
            offsets,
            columns,
            values,
            rhs,
        }
    }
}

impl CompleteCsrStorage for TridiagonalStorage {
    fn rows(&self) -> usize {
        self.dimension
    }

    fn columns(&self) -> usize {
        self.dimension
    }

    fn row_offsets(&self) -> &[usize] {
        &self.offsets
    }

    fn column_indices(&self) -> &[usize] {
        &self.columns
    }

    fn values(&self) -> &[f64] {
        &self.values
    }

    fn right_hand_side(&self) -> &[f64] {
        &self.rhs
    }
}

fn owned_shards(
    storage: &TridiagonalStorage,
    partition: &Partition,
) -> Vec<OwnedLinearSystemShard> {
    (0..partition.count().get())
        .map(|partition_index| {
            let id = PartitionId::new(partition_index);
            let rows = partition.owned_indices(id).collect::<Vec<_>>();
            let mut row_offsets = vec![0];
            let mut columns = Vec::new();
            let mut values = Vec::new();
            let mut right_hand_side = Vec::new();
            for &row in &rows {
                let range = storage.offsets[row]..storage.offsets[row + 1];
                columns.extend_from_slice(&storage.columns[range.clone()]);
                values.extend_from_slice(&storage.values[range]);
                right_hand_side.push(storage.rhs[row]);
                row_offsets.push(columns.len());
            }
            OwnedLinearSystemShard::new(
                id,
                NonZeroUsize::new(storage.dimension).unwrap(),
                rows,
                row_offsets,
                columns,
                values,
                right_hand_side,
            )
            .unwrap()
        })
        .collect()
}

fn tridiagonal(dimension: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut offsets = Vec::with_capacity(dimension + 1);
    let mut columns = Vec::new();
    let mut values = Vec::new();
    offsets.push(0);
    for row in 0..dimension {
        if row > 0 {
            columns.push(row - 1);
            values.push(-1.0);
        }
        columns.push(row);
        values.push(2.0);
        if row + 1 < dimension {
            columns.push(row + 1);
            values.push(-1.0);
        }
        offsets.push(columns.len());
    }
    (offsets, columns, values)
}

fn global_apply(offsets: &[usize], columns: &[usize], values: &[f64], input: &[f64]) -> Vec<f64> {
    (0..input.len())
        .map(|row| {
            (offsets[row]..offsets[row + 1])
                .map(|entry| values[entry] * input[columns[entry]])
                .sum()
        })
        .collect()
}
