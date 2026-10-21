use super::*;

#[test]
fn csr_actions_reject_nonfinite_coordinates_even_when_structurally_unused() {
    use eqiora_solver::LinearOperatorOrientation;
    let matrix = CsrMatrix::from_sorted_csr(2, 3, vec![0, 1, 1], vec![0], vec![2.0]).unwrap();
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matrix.multiply(&[1.0, invalid, 2.0]).is_err());
        for orientation in [
            LinearOperatorOrientation::Transposed,
            LinearOperatorOrientation::ConjugateTransposed,
        ] {
            let mut output = [7.0; 3];
            assert!(
                matrix
                    .apply_oriented(orientation, &[3.0, invalid], &mut output)
                    .is_err()
            );
            assert_eq!(output, [7.0; 3], "reject input before writing output");
        }
    }
}

#[test]
fn rectangular_real32_rows_validate_dimensions_ranges_and_overflow() {
    use eqiora_solver::LinearOperatorOrientation as O;
    // [2, 0, -1; 0, 0, 0].
    let matrix =
        CsrMatrix::from_sorted_csr(2, 3, vec![0, 2, 2], vec![0, 2], vec![2_f32, -1.]).unwrap();
    assert_eq!(matrix.multiply(&[3., 9., 4.]).unwrap(), [2., 0.]);
    for orientation in [O::Transposed, O::ConjugateTransposed] {
        let mut output = [9.; 3];
        matrix
            .apply_oriented(orientation, &[3., 7.], &mut output)
            .unwrap();
        assert_eq!(output, [6., 0., -3.]);
        assert!(
            matrix
                .apply_oriented(orientation, &[3.], &mut output)
                .is_err()
        );
    }
    let mut row = [9.];
    matrix.apply_rows(0..1, &[3., 9., 4.], &mut row).unwrap();
    assert_eq!(row, [2.]);
    matrix.apply_rows(2..2, &[3., 9., 4.], &mut []).unwrap();
    for range in [std::ops::Range { start: 2, end: 1 }, 2..3, 0..2] {
        assert!(matrix.apply_rows(range, &[3., 9., 4.], &mut row).is_err());
    }
    assert!(matrix.multiply(&[3., 4.]).is_err());
    assert!(
        matrix
            .apply_rows(0..1, &[3., f32::NAN, 4.], &mut row)
            .is_err()
    );
    let overflow = CsrMatrix::from_sorted_csr(1, 1, vec![0, 1], vec![0], vec![f32::MAX]).unwrap();
    for orientation in [O::Normal, O::Transposed, O::ConjugateTransposed] {
        assert!(
            overflow
                .apply_oriented(orientation, &[2.], &mut row)
                .is_err()
        );
    }
}

fn scatter_scalar(
    assembler: &mut CooAssembler<f64>,
    row: usize,
    column: usize,
    value: f64,
    rhs: f64,
) {
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
    let mut assembler = CooAssembler::<f64>::new(3).unwrap();
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
    let mut assembler = CooAssembler::<f64>::new(1).unwrap();
    for value in [2_f64.powi(53), 1.0, -2_f64.powi(53), 4.0] {
        scatter_scalar(&mut assembler, 0, 0, value, 0.0);
    }

    let system = assembler.finish().unwrap();
    assert_eq!(system.matrix().values()[0].to_bits(), 4.0_f64.to_bits());
}

#[test]
fn finish_eliminates_exact_zeros_and_preserves_a_cancelled_row() {
    let mut assembler = CooAssembler::<f64>::new(2).unwrap();
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

    let mut cancelled_row = CooAssembler::<f64>::new(2).unwrap();
    scatter_scalar(&mut cancelled_row, 0, 0, 1.0, 0.0);
    scatter_scalar(&mut cancelled_row, 0, 0, -1.0, 0.0);
    scatter_scalar(&mut cancelled_row, 1, 1, 2.0, 0.0);
    let cancelled = cancelled_row.finish().unwrap();
    assert_eq!(cancelled.matrix().row_offsets(), &[0, 0, 1]);
    assert_eq!(cancelled.matrix().column_indices(), &[1]);
    assert_eq!(cancelled.matrix().values(), &[2.]);
}

#[test]
fn finish_orders_columns_ascending_within_each_row() {
    let mut assembler = CooAssembler::<f64>::new(4).unwrap();
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
    let diagnostic = CooAssembler::<f64>::new(0).unwrap_err();
    assert_eq!(diagnostic.code(), codes::ASSEMBLY_FAILED);
    assert_eq!(
        diagnostic.message(),
        "assembled system requires at least one free equation"
    );

    let mut assembler = CooAssembler::<f64>::new(1).unwrap();
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
    let mut assembler = CooAssembler::<f64>::new(1).unwrap();
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
    let mut assembler = CooAssembler::<f64>::new(1).unwrap();
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
    let mut assembler = CooAssembler::<f64>::new(1).unwrap();
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
    let mut assembler = CooAssembler::<f64>::new(1).unwrap();
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
fn linear_system_constructor_enforces_shape_and_canonical_storage() {
    let rectangular = CsrMatrix::from_sorted_csr(1, 2, vec![0, 1], vec![0], vec![1.0]).unwrap();
    let wrong_rhs = CsrMatrix::from_sorted_csr(1, 1, vec![0, 1], vec![0], vec![1.0]).unwrap();
    let non_finite_rhs = wrong_rhs.clone();
    let empty_row = CsrMatrix::from_sorted_csr(2, 2, vec![0, 1, 1], vec![0], vec![1.0]).unwrap();
    let explicit_zero = CsrMatrix::from_sorted_csr(1, 1, vec![0, 1], vec![0], vec![0.0]).unwrap();

    let cases = [
        LinearSystem::new(rectangular, vec![0.0]),
        LinearSystem::new(wrong_rhs, vec![]),
        LinearSystem::new(non_finite_rhs, vec![f64::NAN]),
        LinearSystem::new(explicit_zero, vec![0.0]),
    ];
    for result in cases {
        assert_eq!(result.unwrap_err().code(), codes::ASSEMBLY_FAILED);
    }

    let partial = LinearSystem::new(empty_row, vec![0.0, 2.0]).unwrap();
    let mut action = [99.; 2];
    partial.matrix().apply(&[3., 4.], &mut action).unwrap();
    assert_eq!(action, [3., 0.]);
    assert_eq!(partial.rhs(), &[0., 2.]);
    let matrix = CsrMatrix::from_sorted_csr(1, 1, vec![0, 1], vec![0], vec![2.0]).unwrap();
    assert_eq!(LinearSystem::new(matrix, vec![3.0]).unwrap().rhs(), &[3.0]);
}

#[test]
fn compiled_csr_overflow_retains_all_prior_values() {
    let topology = CsrTopology::new(1, vec![0, 1], vec![0]).unwrap();
    let map = AssemblyMap::new(
        vec![Some(DofId::new(0))],
        vec![LocalUnknown::Free(DofId::new(0))],
    )
    .unwrap();
    let projection = crate::projection::CompiledProjection::new(1, &map).unwrap();
    let slots = CsrScatterSlots::new(&topology, &projection).unwrap();
    let local = LocalContribution::new(1, 1, vec![f64::MAX], vec![3.]).unwrap();
    let delta = projection.project(&map, &local).unwrap();
    let mut assembler = CsrValueAssembler::new(topology);
    assembler.scatter_delta(&delta, &slots).unwrap();
    assert!(assembler.scatter_delta(&delta, &slots).is_err());
    let preserved = assembler.finish().unwrap();
    assert_eq!(preserved.matrix().values(), &[f64::MAX]);
    assert_eq!(preserved.rhs(), &[3.]);
}

#[test]
fn prepared_csr_preserves_exact_cancellation_and_rhs() {
    let topology = CsrTopology::new(1, vec![0, 1], vec![0]).unwrap();
    let map = AssemblyMap::new(
        vec![Some(DofId::new(0))],
        vec![LocalUnknown::Free(DofId::new(0))],
    )
    .unwrap();
    let projection = crate::projection::CompiledProjection::new(1, &map).unwrap();
    let slots = CsrScatterSlots::new(&topology, &projection).unwrap();
    let mut assembler = CsrValueAssembler::new(topology);
    for value in [2., -2.] {
        let local = LocalContribution::new(1, 1, vec![value], vec![3.]).unwrap();
        assembler
            .scatter_delta(&projection.project(&map, &local).unwrap(), &slots)
            .unwrap();
    }
    let system = assembler.finish().unwrap();
    assert_eq!(system.matrix().row_offsets(), &[0, 0]);
    assert!(system.matrix().values().is_empty());
    assert_eq!(system.rhs(), &[6.]);
}
