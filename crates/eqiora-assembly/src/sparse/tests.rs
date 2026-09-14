use super::*;

fn scatter_scalar(assembler: &mut CooAssembler, row: usize, column: usize, value: f64, rhs: f64) {
    let local = LocalContribution::new(1, 1, vec![value], vec![rhs]).unwrap();
    let map = AssemblyMap::new(
        vec![Some(DofId::new(row))],
        vec![LocalUnknown::Free(DofId::new(column))],
    )
    .unwrap();
    assembler.scatter(&map, &local).unwrap();
}

#[test]
fn assembled_system_matches_recorded_csr_and_float_bits() {
    let mut assembler = CooAssembler::new(3).unwrap();
    let first = LocalContribution::new(2, 2, vec![1.5, -2.0, 3.25, 4.5], vec![8.0, -1.0]).unwrap();
    let first_map = AssemblyMap::new(
        vec![Some(DofId::new(2)), Some(DofId::new(0))],
        vec![
            LocalUnknown::Free(DofId::new(2)),
            LocalUnknown::Free(DofId::new(0)),
        ],
    )
    .unwrap();
    assembler.scatter(&first_map, &first).unwrap();

    let second =
        LocalContribution::new(2, 3, vec![5.0, 0.5, 3.0, -1.25, 2.0, -4.0], vec![7.0, -2.0])
            .unwrap();
    let second_map = AssemblyMap::new(
        vec![Some(DofId::new(0)), Some(DofId::new(1))],
        vec![
            LocalUnknown::Free(DofId::new(1)),
            LocalUnknown::Free(DofId::new(0)),
            LocalUnknown::Fixed(2.0),
        ],
    )
    .unwrap();
    assembler.scatter(&second_map, &second).unwrap();

    let system = assembler.finish().unwrap();
    assert_eq!(system.matrix().row_offsets(), &[0, 3, 5, 7]);
    assert_eq!(system.matrix().column_indices(), &[0, 1, 2, 0, 1, 0, 2]);
    assert_eq!(
        system
            .matrix()
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![
            0x4014_0000_0000_0000,
            0x4014_0000_0000_0000,
            0x400a_0000_0000_0000,
            0x4000_0000_0000_0000,
            0xbff4_0000_0000_0000,
            0xc000_0000_0000_0000,
            0x3ff8_0000_0000_0000,
        ]
    );
    assert_eq!(
        system
            .rhs()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![
            0x0000_0000_0000_0000,
            0x4018_0000_0000_0000,
            0x4020_0000_0000_0000,
        ]
    );
}

#[test]
fn duplicate_entries_preserve_scatter_summation_order() {
    let mut assembler = CooAssembler::new(1).unwrap();
    for value in [2_f64.powi(53), 1.0, -2_f64.powi(53), 4.0] {
        scatter_scalar(&mut assembler, 0, 0, value, 0.0);
    }

    let system = assembler.finish().unwrap();
    assert_eq!(system.matrix().values()[0].to_bits(), 4.0_f64.to_bits());
}

#[test]
fn finish_eliminates_exact_zeros_and_reports_a_cancelled_row() {
    let mut assembler = CooAssembler::new(2).unwrap();
    let first = LocalContribution::new(2, 2, vec![3.0, 2.0, 0.0, 5.0], vec![0.0, 0.0]).unwrap();
    let second = LocalContribution::new(2, 2, vec![0.0, -2.0, 0.0, 0.0], vec![0.0, 0.0]).unwrap();
    let map = AssemblyMap::new(
        vec![Some(DofId::new(0)), Some(DofId::new(1))],
        vec![
            LocalUnknown::Free(DofId::new(0)),
            LocalUnknown::Free(DofId::new(1)),
        ],
    )
    .unwrap();
    assembler.scatter(&map, &first).unwrap();
    assembler.scatter(&map, &second).unwrap();

    let system = assembler.finish().unwrap();
    assert_eq!(system.matrix().row_offsets(), &[0, 1, 2]);
    assert_eq!(system.matrix().column_indices(), &[0, 1]);
    assert_eq!(
        system
            .matrix()
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![3.0_f64.to_bits(), 5.0_f64.to_bits()]
    );

    let mut cancelled_row = CooAssembler::new(2).unwrap();
    scatter_scalar(&mut cancelled_row, 0, 0, 1.0, 0.0);
    scatter_scalar(&mut cancelled_row, 0, 0, -1.0, 0.0);
    scatter_scalar(&mut cancelled_row, 1, 1, 2.0, 0.0);
    let diagnostic = cancelled_row.finish().unwrap_err();
    assert_eq!(diagnostic.code(), codes::ASSEMBLY_FAILED);
    assert_eq!(
        diagnostic.message(),
        "assembled global row 0 has no nonzero entries"
    );
}

#[test]
fn finish_orders_columns_ascending_within_each_row() {
    let mut assembler = CooAssembler::new(4).unwrap();
    for (column, value) in [(3, 4.0), (0, 1.0), (2, 3.0), (1, 2.0)] {
        scatter_scalar(&mut assembler, 0, column, value, 0.0);
    }
    for row in 1..4 {
        scatter_scalar(&mut assembler, row, row, 1.0, 0.0);
    }

    let system = assembler.finish().unwrap();
    for row in 0..4 {
        let start = system.matrix().row_offsets()[row];
        let end = system.matrix().row_offsets()[row + 1];
        assert!(
            system.matrix().column_indices()[start..end]
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
    }
    assert_eq!(&system.matrix().column_indices()[0..4], &[0, 1, 2, 3]);
}

#[test]
fn assembler_rejects_zero_size_and_round_trips_one_equation() {
    let diagnostic = CooAssembler::new(0).unwrap_err();
    assert_eq!(diagnostic.code(), codes::ASSEMBLY_FAILED);
    assert_eq!(
        diagnostic.message(),
        "assembled system requires at least one free equation"
    );

    let mut assembler = CooAssembler::new(1).unwrap();
    scatter_scalar(&mut assembler, 0, 0, -2.5, 7.25);
    let system = assembler.finish().unwrap();
    assert_eq!(system.matrix().row_offsets(), &[0, 1]);
    assert_eq!(system.matrix().column_indices(), &[0]);
    assert_eq!(system.matrix().values()[0].to_bits(), (-2.5_f64).to_bits());
    assert_eq!(system.rhs()[0].to_bits(), 7.25_f64.to_bits());
}

#[test]
fn scatter_eliminates_fixed_columns_and_skips_fixed_rows() {
    let local = LocalContribution::new(2, 2, vec![1.0, -1.0, -1.0, 1.0], vec![0.0, 0.0]).unwrap();
    let map = AssemblyMap::new(
        vec![None, Some(DofId::new(0))],
        vec![LocalUnknown::Fixed(2.0), LocalUnknown::Free(DofId::new(0))],
    )
    .unwrap();
    let mut assembler = CooAssembler::new(1).unwrap();
    assembler.scatter(&map, &local).unwrap();
    let system = assembler.finish().unwrap();
    assert_eq!(system.matrix().entry(0, 0), Some(1.0));
    assert_eq!(system.rhs(), &[2.0]);
}

#[test]
fn scatter_rejects_shape_and_global_index_mismatch() {
    let local = LocalContribution::new(1, 1, vec![1.0], vec![0.0]).unwrap();
    let map = AssemblyMap::new(
        vec![Some(DofId::new(1))],
        vec![LocalUnknown::Free(DofId::new(1))],
    )
    .unwrap();
    let mut assembler = CooAssembler::new(1).unwrap();
    assert_eq!(
        assembler.scatter(&map, &local).unwrap_err().code(),
        codes::ASSEMBLY_FAILED
    );
}

#[test]
fn failed_scatter_is_atomic() {
    let local = LocalContribution::new(1, 1, vec![f64::MAX], vec![0.0]).unwrap();
    let map = AssemblyMap::new(
        vec![Some(DofId::new(0))],
        vec![LocalUnknown::Free(DofId::new(0))],
    )
    .unwrap();
    let mut assembler = CooAssembler::new(1).unwrap();
    assembler.scatter(&map, &local).unwrap();
    let before = assembler.clone().finish().unwrap();
    assert_eq!(
        assembler.scatter(&map, &local).unwrap_err().code(),
        codes::ASSEMBLY_FAILED
    );
    assert_eq!(assembler.finish().unwrap(), before);
}

#[test]
fn mismatched_delta_is_rejected_atomically() {
    let local = LocalContribution::new(1, 1, vec![2.0], vec![3.0]).unwrap();
    let map = AssemblyMap::new(
        vec![Some(DofId::new(0))],
        vec![LocalUnknown::Free(DofId::new(0))],
    )
    .unwrap();
    let mut assembler = CooAssembler::new(1).unwrap();
    assembler.scatter(&map, &local).unwrap();
    let before = assembler.clone().finish().unwrap();
    let foreign = AssemblyDelta::from_local(2, &map, &local).unwrap();

    assert_eq!(
        assembler.scatter_delta(&foreign).unwrap_err().code(),
        codes::ASSEMBLY_FAILED
    );
    assert_eq!(assembler.finish().unwrap(), before);
}

#[test]
fn sorted_csr_constructor_admits_rectangular_and_empty_rows() {
    let matrix =
        CsrMatrix::from_sorted_csr(2, 3, vec![0, 0, 2], vec![0, 2], vec![2.0, -1.0]).unwrap();
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.columns(), 3);
    assert_eq!(matrix.entry(0, 1), Some(0.0));
    assert_eq!(matrix.entry(1, 2), Some(-1.0));
}

#[test]
fn sorted_csr_constructor_rejects_every_broken_invariant() {
    let cases = [
        CsrMatrix::from_sorted_csr(0, 1, vec![0], vec![], vec![]),
        CsrMatrix::from_sorted_csr(1, 1, vec![0], vec![], vec![]),
        CsrMatrix::from_sorted_csr(1, 1, vec![1, 1], vec![0], vec![1.0]),
        CsrMatrix::from_sorted_csr(1, 1, vec![0, 1], vec![], vec![1.0]),
        CsrMatrix::from_sorted_csr(1, 1, vec![0, 1], vec![1], vec![1.0]),
        CsrMatrix::from_sorted_csr(1, 2, vec![0, 2], vec![1, 0], vec![1.0, 1.0]),
        CsrMatrix::from_sorted_csr(1, 1, vec![0, 1], vec![0], vec![f64::NAN]),
    ];
    for result in cases {
        assert_eq!(result.unwrap_err().code(), codes::ASSEMBLY_FAILED);
    }
}

#[test]
fn linear_system_constructor_enforces_complete_canonical_rows() {
    let rectangular = CsrMatrix::from_sorted_csr(1, 2, vec![0, 1], vec![0], vec![1.0]).unwrap();
    let wrong_rhs = CsrMatrix::from_sorted_csr(1, 1, vec![0, 1], vec![0], vec![1.0]).unwrap();
    let non_finite_rhs = wrong_rhs.clone();
    let empty_row = CsrMatrix::from_sorted_csr(2, 2, vec![0, 1, 1], vec![0], vec![1.0]).unwrap();
    let explicit_zero = CsrMatrix::from_sorted_csr(1, 1, vec![0, 1], vec![0], vec![0.0]).unwrap();

    let cases = [
        LinearSystem::new(rectangular, vec![0.0]),
        LinearSystem::new(wrong_rhs, vec![]),
        LinearSystem::new(non_finite_rhs, vec![f64::NAN]),
        LinearSystem::new(empty_row, vec![0.0, 0.0]),
        LinearSystem::new(explicit_zero, vec![0.0]),
    ];
    for result in cases {
        assert_eq!(result.unwrap_err().code(), codes::ASSEMBLY_FAILED);
    }

    let matrix = CsrMatrix::from_sorted_csr(1, 1, vec![0, 1], vec![0], vec![2.0]).unwrap();
    assert_eq!(LinearSystem::new(matrix, vec![3.0]).unwrap().rhs(), &[3.0]);
}
