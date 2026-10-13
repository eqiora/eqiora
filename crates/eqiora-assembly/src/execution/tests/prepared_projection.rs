use super::*;

#[test]
fn compiled_multi_target_projection_preserves_independent_folds_and_exact_binding() {
    let base = AssemblyPlan::new(vec![
        AssemblyTarget::new(1).unwrap(),
        AssemblyTarget::new(3).unwrap(),
    ])
    .unwrap();
    let reduced = base.target_id(0).unwrap();
    let full = base.target_id(1).unwrap();
    let reduced_map = |fixed| {
        AssemblyMap::new(
            vec![Some(DofId::new(0)), Some(DofId::new(0)), None],
            vec![
                LocalUnknown::Free(DofId::new(0)),
                LocalUnknown::Free(DofId::new(0)),
                LocalUnknown::Fixed(fixed),
            ],
        )
        .unwrap()
    };
    let maps = vec![
        TargetAssemblyMap::new(reduced, reduced_map(2.)),
        TargetAssemblyMap::new(
            full,
            AssemblyMap::new(
                (0..3).map(|i| Some(DofId::new(i))).collect(),
                (0..3).map(|i| LocalUnknown::Free(DofId::new(i))).collect(),
            )
            .unwrap(),
        ),
    ];
    let plan = base
        .clone()
        .prepare(vec![maps.clone(), maps.clone()])
        .unwrap();
    let packet = |factor| {
        AssemblyPacket::new(
            LocalContribution::new(
                3,
                3,
                [1e16, 1., 3., -1e16, 1., 5., 100., 200., 300.]
                    .map(|v| factor * v)
                    .to_vec(),
                [10., 20., 1000.].map(|v| factor * v).to_vec(),
            )
            .unwrap(),
            maps.clone(),
        )
        .unwrap()
    };
    let work = IndexedAssemblyWork::new(2, |i| Ok(packet((i + 1) as f64)));
    let preparations = (
        crate::projection::compilation_count(),
        crate::sparse::slot_compilation_count(),
    );
    for _ in 0..3 {
        let result = REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work).unwrap();
        // Each reduced packet folds 1e16 + 1 - 1e16 + 1 = 1,
        // before global packet accumulation. Its RHS is 10 - 3*2 + 20 - 5*2 = 14.
        assert_eq!(result.system(reduced).unwrap().matrix().values(), &[3.]);
        assert_eq!(result.system(reduced).unwrap().rhs(), &[42.]);
        assert_eq!(
            result.system(full).unwrap().matrix().values(),
            &[3e16, 3., 9., -3e16, 3., 15., 300., 600., 900.]
        );
        assert_eq!(result.system(full).unwrap().rhs(), &[30., 60., 3000.]);
    }
    assert_eq!(
        preparations,
        (
            crate::projection::compilation_count(),
            crate::sparse::slot_compilation_count()
        )
    );
    let cancellation = IndexedAssemblyWork::new(2, |i| Ok(packet(if i == 0 { 1. } else { -1. })));
    assert!(
        REFERENCE_ASSEMBLY_BACKEND
            .assemble(&plan, &cancellation)
            .is_err()
    );
    let projected = packet(1.).project(&plan, 0).unwrap();
    assert_eq!(
        projected[0].delta().rows()[0].entries(),
        &[(DofId::new(0), 1.)]
    );
    assert_eq!(projected[0].delta().rows()[0].rhs(), 14.);
    assert!(
        AssemblyAccumulator::new(&plan)
            .unwrap()
            .scatter_projected(0, &projected[..1])
            .is_err()
    );
    let wrong_index = packet(1.).project(&plan, 1).unwrap();
    assert!(
        AssemblyAccumulator::new(&plan)
            .unwrap()
            .scatter_projected(0, &wrong_index)
            .is_err()
    );
    assert!(packet(1.).project(&plan, 2).is_err());
    let mut foreign_maps = maps.clone();
    foreign_maps[0] = TargetAssemblyMap::new(reduced, reduced_map(3.));
    let foreign = base
        .prepare(vec![foreign_maps.clone(), foreign_maps.clone()])
        .unwrap();
    let foreign_packet = AssemblyPacket::new(packet(1.).local().clone(), foreign_maps).unwrap();
    assert!(foreign_packet.project(&plan, 0).is_err());
    let foreign_delta = foreign_packet.project(&foreign, 0).unwrap();
    assert!(
        AssemblyAccumulator::new(&plan)
            .unwrap()
            .scatter_projected(0, &foreign_delta)
            .is_err()
    );
    // An invalid attempt cannot poison immutable preparation shared by other attempts.
    std::thread::scope(|scope| {
        let first = scope.spawn(|| REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work).unwrap());
        let second = scope.spawn(|| REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work).unwrap());
        assert_eq!(
            first.join().unwrap().systems(),
            second.join().unwrap().systems()
        );
    });
}

#[test]
fn compiled_complex_duplicate_projection_retains_primal_products() {
    use num_complex::Complex64 as C;
    let base = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
    let target = base.target_id(0).unwrap();
    let mappings = vec![TargetAssemblyMap::new(
        target,
        AssemblyMap::new(
            vec![Some(DofId::new(0)); 2],
            vec![
                LocalUnknown::Free(DofId::new(0)),
                LocalUnknown::Free(DofId::new(0)),
                LocalUnknown::Fixed(C::new(2., -1.)),
            ],
        )
        .unwrap(),
    )];
    let plan = base.prepare(vec![mappings.clone()]).unwrap();
    let work = IndexedAssemblyWork::new(1, |_| {
        AssemblyPacket::new(
            LocalContribution::new(
                2,
                3,
                vec![
                    C::new(1., 2.),
                    C::new(3., -1.),
                    C::new(4., 3.),
                    C::new(-1., 1.),
                    C::new(2., 0.),
                    C::new(1., -2.),
                ],
                vec![C::new(11., 13.), C::new(5., -7.)],
            )
            .unwrap(),
            mappings.clone(),
        )
    });
    let result = REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work).unwrap();
    // Free sum 5+2i; fixed products total (5+i)(2-i)=11-3i.
    assert_eq!(
        result.system(target).unwrap().matrix().values(),
        &[C::new(5., 2.)]
    );
    assert_eq!(result.system(target).unwrap().rhs(), &[C::new(5., 9.)]);
}
