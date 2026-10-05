use super::*;
use crate::{DofId, PacketLinearSystem};
use eqiora_solver::{LinearOperator, LinearOperatorOrientation as O, OrientedLinearOperator};
use num_complex::Complex64 as C;

#[test]
fn complex_prepared_and_matrix_free_paths_share_fixed_value_elimination_and_orientations() {
    let base = AssemblyPlan::new(vec![AssemblyTarget::new(2).unwrap()]).unwrap();
    let target = base.target_id(0).unwrap();
    let maps = (0..2)
        .map(|row| {
            vec![TargetAssemblyMap::new(
                target,
                AssemblyMap::new(
                    vec![Some(DofId::new(row))],
                    vec![
                        LocalUnknown::Free(DofId::new(0)),
                        LocalUnknown::Free(DofId::new(1)),
                        LocalUnknown::Fixed(C::new(1., 1.)),
                    ],
                )
                .unwrap(),
            )]
        })
        .collect::<Vec<_>>();
    let plan = base.prepare(maps.clone()).unwrap();
    let locals = [
        LocalContribution::new(
            1,
            3,
            vec![C::new(1., 1.), C::new(2., -1.), C::new(2., -1.)],
            vec![C::new(7., 9.)],
        )
        .unwrap(),
        LocalContribution::new(
            1,
            3,
            vec![C::new(-3., 2.), C::new(4., 1.), C::new(1., 2.)],
            vec![C::new(-12., 21.)],
        )
        .unwrap(),
    ];
    let packets = locals
        .into_iter()
        .zip(maps)
        .map(|(local, maps)| AssemblyPacket::new(local, maps).unwrap())
        .collect::<Vec<_>>();
    let work = IndexedAssemblyWork::new(2, |index: usize| Ok(packets[index].clone()));
    let assembled = REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work).unwrap();
    let matrix_free = PacketLinearSystem::from_work(&plan, target, &work).unwrap();
    // Fixed-value products are (2-i)(1+i)=3+i and (1+2i)(1+i)=-1+3i.
    let expected_rhs = [C::new(4., 8.), C::new(-11., 18.)];
    assert_eq!(assembled.system(target).unwrap().rhs(), &expected_rhs);
    assert_eq!(matrix_free.right_hand_side(), &expected_rhs);
    let input = [C::new(2., -1.), C::new(-1., 3.)];
    for (orientation, expected) in [
        (O::Normal, [C::new(4., 8.), C::new(-11., 18.)]),
        (O::Transposed, [C::new(0., -10.), C::new(-4., 7.)]),
        (O::ConjugateTransposed, [C::new(10., -10.), C::new(4., 13.)]),
    ] {
        for operator in [
            assembled.system(target).unwrap().matrix() as &dyn OrientedLinearOperator<Scalar = C>,
            matrix_free.operator(),
        ] {
            let mut actual = [C::new(0., 0.); 2];
            operator
                .apply_oriented(orientation, &input, &mut actual)
                .unwrap();
            assert_eq!(actual, expected);
        }
    }
    let mut diagonal = [C::new(0., 0.); 2];
    matrix_free.operator().diagonal(&mut diagonal).unwrap();
    assert_eq!(diagonal, [C::new(1., 1.), C::new(4., 1.)]);
}

#[test]
fn prepared_identity_distinguishes_scalar_domain_precision_and_imaginary_fixed_values() {
    let base = AssemblyPlan::new(vec![AssemblyTarget::new(1).unwrap()]).unwrap();
    let target = base.target_id(0).unwrap();
    let real32 = base
        .clone()
        .prepare(vec![vec![TargetAssemblyMap::new(
            target,
            AssemblyMap::<f32>::new(
                vec![Some(DofId::new(0))],
                vec![LocalUnknown::Free(DofId::new(0))],
            )
            .unwrap(),
        )]])
        .unwrap();
    let real64 = base
        .clone()
        .prepare(vec![vec![TargetAssemblyMap::new(
            target,
            AssemblyMap::<f64>::new(
                vec![Some(DofId::new(0))],
                vec![LocalUnknown::Free(DofId::new(0))],
            )
            .unwrap(),
        )]])
        .unwrap();
    let complex_map = TargetAssemblyMap::new(
        target,
        AssemblyMap::<C>::new(
            vec![Some(DofId::new(0))],
            vec![LocalUnknown::Free(DofId::new(0))],
        )
        .unwrap(),
    );
    let complex = base
        .clone()
        .prepare(vec![vec![complex_map.clone()]])
        .unwrap();
    assert_ne!(real32.structure_identity(), real64.structure_identity());
    assert_ne!(real64.structure_identity(), complex.structure_identity());
    let packet = AssemblyPacket::new(
        LocalContribution::new(1, 1, vec![C::new(1., 0.)], vec![C::new(0., 1.)]).unwrap(),
        vec![complex_map],
    )
    .unwrap();
    let work = IndexedAssemblyWork::new(1, |_| Ok(packet.clone()));
    let accepted = REFERENCE_ASSEMBLY_BACKEND
        .assemble(&complex, &work)
        .unwrap();
    PacketLinearSystem::from_work(&complex, target, &work).unwrap();
    assert!(REFERENCE_ASSEMBLY_BACKEND.assemble(&real64, &work).is_err());
    assert!(PacketLinearSystem::from_work(&real64, target, &work).is_err());
    let twice = base
        .clone()
        .prepare(vec![packet.mappings.clone(), packet.mappings.clone()])
        .unwrap();
    // Both planned packets cover the row, so a nonempty-row check alone cannot
    // detect omission of the second packet.
    assert!(REFERENCE_ASSEMBLY_BACKEND.assemble(&twice, &work).is_err());
    assert!(PacketLinearSystem::from_work(&twice, target, &work).is_err());
    assert!(
        AssemblyResult::from_complete_systems(
            &twice,
            accepted.systems().to_vec(),
            1,
            ExecutionReport::host_serial(),
        )
        .is_err()
    );
    assert!(AssemblyAccumulator::<C>::new(&real64).is_err());
    assert!(AssemblyAccumulator::<f32>::new(&real64).is_err());
    assert!(
        AssemblyResult::from_complete_systems(
            &complex,
            accepted.systems().to_vec(),
            1,
            ExecutionReport::host_serial(),
        )
        .is_ok()
    );
    assert!(
        AssemblyResult::from_complete_systems(
            &real64,
            accepted.systems().to_vec(),
            1,
            ExecutionReport::host_serial(),
        )
        .is_err()
    );
    let fixed_map = |imaginary| {
        TargetAssemblyMap::new(
            target,
            AssemblyMap::new(
                vec![Some(DofId::new(0))],
                vec![
                    LocalUnknown::Free(DofId::new(0)),
                    LocalUnknown::Fixed(C::new(1., imaginary)),
                ],
            )
            .unwrap(),
        )
    };
    let plus = base.clone().prepare(vec![vec![fixed_map(1.)]]).unwrap();
    let minus = base.prepare(vec![vec![fixed_map(-1.)]]).unwrap();
    assert_ne!(plus.structure_identity(), minus.structure_identity());
    let packet = AssemblyPacket::new(
        LocalContribution::new(1, 2, vec![C::new(1., 0.); 2], vec![C::new(0., 0.)]).unwrap(),
        vec![fixed_map(-1.)],
    )
    .unwrap();
    let changed = IndexedAssemblyWork::new(1, |_| Ok(packet.clone()));
    REFERENCE_ASSEMBLY_BACKEND
        .assemble(&minus, &changed)
        .unwrap();
    assert!(
        REFERENCE_ASSEMBLY_BACKEND
            .assemble(&plus, &changed)
            .is_err()
    );
    assert!(PacketLinearSystem::from_work(&plus, target, &changed).is_err());
}
