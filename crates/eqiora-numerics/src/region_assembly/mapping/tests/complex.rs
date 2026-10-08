use super::*;
use eqiora_solver::{LinearOperator, LinearOperatorOrientation as O, OrientedLinearOperator};
use num_complex::Complex64 as C;

#[test]
fn complex_mapping_recovers_scaled_fields_and_rebinds_prescriptions() {
    let source = r#"model Wave() {
        domain body = box(0, 6);
        parameter a: complex<m^2> = math.complex(6[m^2], 6[m^2]);
        variable u: complex<1> on body;
        relation balance on body { -div(a * grad(u)) + u = math.complex(1, 3); }
    }"#;
    let (transaction, model, symbols) = compile("complex-mapping.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let field = symbols.get("u").unwrap();
    let domain = symbols.get("body").unwrap();
    let reference = ReferenceCell::hypercube(1).unwrap();
    let mesh = CartesianMesh::from_axes(vec![vec![0.0, 3.0, 6.0]]).unwrap();
    let spatial = [DomainFieldDiscretization::new(
        domain.downcast().unwrap(),
        [FieldSpaceBinding::new(
            field.downcast().unwrap(),
            Space::continuous_lagrange(NonZeroU16::MIN),
        )],
        [],
    )
    .unwrap()];
    let scales = BTreeMap::from([(field, DynQuantity::new(2.0, DimExponents::DIMENSIONLESS))]);
    let layouts = field_layouts(&program, &spatial, reference, &scales).unwrap();
    let key = |vertex| FieldDof {
        field,
        entity: MeshEntity::new(0, vertex),
        slot: 0,
        component: 0,
    };
    let fixed = BTreeMap::from([(key(0), C::new(2.0, 4.0))]);
    let mapping = RegionDofMap::new(&mesh, &layouts, reference, &[domain; 2], &[], &fixed).unwrap();
    assert_eq!(mapping.full_count(), 3);
    assert_eq!(mapping.free_count(), 2);
    assert!(
        RegionDofMap::<f64>::new(
            &mesh,
            &layouts,
            reference,
            &[domain; 2],
            &[],
            &BTreeMap::new()
        )
        .is_err()
    );
    let free = [C::new(3.0, -1.0), C::new(-2.0, 5.0)];
    let full = mapping.lift(&free, false).unwrap();
    assert_eq!(full, vec![C::new(1.0, 2.0), free[0], free[1]]);
    assert_eq!(mapping.restrict(&full).unwrap(), free);
    assert_eq!(mapping.lift(&free, true).unwrap()[0], C::new(0.0, 0.0));
    let physical = mapping.recover(&free, &[field]).unwrap();
    assert_eq!(
        physical[&field].coefficients,
        BTreeMap::from([
            (key(0), C::new(2.0, 4.0)),
            (key(1), C::new(6.0, -2.0)),
            (key(2), C::new(-4.0, 10.0)),
        ])
    );
    mapping.validate_physical(&physical).unwrap();
    assert!(mapping.recover(&free, &[]).is_err());
    assert!(
        mapping
            .recover(&[C::new(1.0, f64::NAN), free[1]], &[field])
            .is_err()
    );
    let mut corrupt = physical.clone();
    corrupt
        .get_mut(&field)
        .unwrap()
        .coefficients
        .insert(key(1), C::new(1.0, f64::INFINITY));
    assert!(mapping.validate_physical(&corrupt).is_err());
    // Real scaling must not square its factor as complex division would.
    // Both component results are finite at these independently chosen extremes.
    for scale in [1e-300, 1e300] {
        let scales =
            BTreeMap::from([(field, DynQuantity::new(scale, DimExponents::DIMENSIONLESS))]);
        let layouts = field_layouts(&program, &spatial, reference, &scales).unwrap();
        let fixed = BTreeMap::from([(key(0), C::new(2.0 * scale, 4.0 * scale))]);
        let scaled =
            RegionDofMap::new(&mesh, &layouts, reference, &[domain; 2], &[], &fixed).unwrap();
        assert_eq!(scaled.lift(&free, false).unwrap()[0], C::new(2.0, 4.0));
        let recovered = scaled.recover(&free, &[field]).unwrap();
        assert_eq!(
            recovered[&field].coefficients[&key(1)],
            C::new(3.0 * scale, -scale)
        );
        scaled.validate_physical(&recovered).unwrap();
        let rebound = scaled.with_prescribed(&fixed).unwrap();
        assert_eq!(rebound.lift(&free, false).unwrap()[0], C::new(2.0, 4.0));
    }
    // A constant u=1+3i solves -div((6+6i) grad u)+u=1+3i.
    // On each length-three cell, K=a/3[[1,-1],[-1,1]] and
    // M=[[1,1/2],[1/2,1]]. The physical Field scale is two.
    let fixed = BTreeMap::from([(key(0), C::new(1.0, 3.0)), (key(2), C::new(1.0, 3.0))]);
    let assembled_map =
        RegionDofMap::new(&mesh, &layouts, reference, &[domain; 2], &[], &fixed).unwrap();
    let form = CompiledRegionForm::<C>::derive(&program, domain, 1).unwrap();
    let row_dimension = DimExponents::from_integers([0, -1, 0, 0, 0, 0, 0]).unwrap();
    let rows = form
        .rows()
        .map(|(id, _, _)| (id, DynQuantity::new(1.0, row_dimension)))
        .collect();
    let bound = form
        .bind(
            reference,
            &[RegionFieldBinding {
                field,
                space: Space::continuous_lagrange(NonZeroU16::MIN),
                scale: DynQuantity::new(2.0, DimExponents::DIMENSIONLESS),
            }],
            &rows,
            None,
        )
        .unwrap();
    let plan = AssemblyPlan::new(vec![
        AssemblyTarget::new(assembled_map.free_count()).unwrap(),
    ])
    .unwrap();
    let target = plan.target_id(0).unwrap();
    let cells = (0..2)
        .map(|index| RegionAssemblyCell {
            index,
            geometry: mesh.geometry_map(MeshEntity::new(1, index)).unwrap(),
            mappings: vec![TargetAssemblyMap::new(
                target,
                assembled_map.cell_map(index, true).unwrap(),
            )],
            previous: BTreeMap::new(),
        })
        .collect();
    let work = PreparedRegionAssembly::new(
        AssemblyPacketSetIdentityV1::Unbound,
        &plan,
        vec![(bound.clone(), QuadratureRule::gauss_legendre(2).unwrap())],
        &[domain; 2],
        cells,
        vec![],
    )
    .unwrap();
    let assembled = REFERENCE_ASSEMBLY_BACKEND.assemble(&plan, &work).unwrap();
    let system = assembled.system(target).unwrap();
    // After fixing both ends: (12+8i) z = -6+22i, hence z=1/2+3i/2.
    assert!((system.matrix().values()[0] - C::new(12.0, 8.0)).norm() < 1e-12);
    assert!((system.rhs()[0] - C::new(-6.0, 22.0)).norm() < 1e-12);
    let captured =
        eqiora_solver::CanonicalCsrSystemView::new(system, LinearOperatorProperties::General)
            .unwrap();
    let policy = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1e-12,
        1e-14,
        NonZeroUsize::new(20).unwrap(),
    )
    .unwrap();
    let finalized = crate::finalized_spatial::FinalizedLinearCore::new(
        policy,
        eqiora_realization::VectorLayoutKind::Replicated,
        eqiora_realization::Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        std::sync::Arc::new(captured),
    );
    let solution = REFERENCE_LINEAR_SOLVER
        .solve(&finalized.linear_problem().unwrap(), policy)
        .unwrap();
    finalized.validate_solution(&solution).unwrap();
    let recovered = assembled_map.recover(solution.values(), &[field]).unwrap();
    for value in recovered[&field].coefficients.values() {
        assert!((*value - C::new(1.0, 3.0)).norm() < 1e-12);
    }

    // Retain two free coordinates so off-diagonal complex action participates.
    // The same source, quadrature and physical scale feed CSR and packet action.
    let one_fixed = BTreeMap::from([(key(0), C::new(1., 3.))]);
    let action_map =
        RegionDofMap::new(&mesh, &layouts, reference, &[domain; 2], &[], &one_fixed).unwrap();
    let action_plan = AssemblyPlan::new(vec![AssemblyTarget::new(2).unwrap()]).unwrap();
    let target = action_plan.target_id(0).unwrap();
    let cells = (0..2)
        .map(|index| RegionAssemblyCell {
            index,
            geometry: mesh.geometry_map(MeshEntity::new(1, index)).unwrap(),
            mappings: vec![TargetAssemblyMap::new(
                target,
                action_map.cell_map(index, true).unwrap(),
            )],
            previous: BTreeMap::new(),
        })
        .collect();
    let work = PreparedRegionAssembly::new(
        AssemblyPacketSetIdentityV1::Unbound,
        &action_plan,
        vec![(bound, QuadratureRule::gauss_legendre(2).unwrap())],
        &[domain; 2],
        cells,
        vec![],
    )
    .unwrap();
    let csr = REFERENCE_ASSEMBLY_BACKEND
        .assemble(&action_plan, &work)
        .unwrap();
    let csr = csr.system(target).unwrap();
    let packets =
        eqiora_assembly::PacketLinearSystem::from_work(&action_plan, target, &work).unwrap();
    // Each scaled cell has diagonal 6+4i and off-diagonal -3-4i.
    // Eliminating z_left=(1+3i)/2 gives b=(-1.5+15.5i, 1.5+4.5i).
    let expected_rhs = [C::new(-1.5, 15.5), C::new(1.5, 4.5)];
    for rhs in [csr.rhs(), packets.right_hand_side()] {
        for (actual, expected) in rhs.iter().zip(expected_rhs) {
            assert!((*actual - expected).norm() < 1e-12);
        }
    }
    let input = [C::new(2., -1.), C::new(-1., 3.)];
    for (orientation, expected) in [
        (O::Normal, [C::new(47., -1.), C::new(-28., 9.)]),
        (O::Transposed, [C::new(47., -1.), C::new(-28., 9.)]),
        (O::ConjugateTransposed, [C::new(7., -41.), C::new(4., 33.)]),
    ] {
        for operator in [
            csr.matrix() as &dyn OrientedLinearOperator<Scalar = C>,
            packets.operator(),
        ] {
            let mut actual = [C::new(0., 0.); 2];
            operator
                .apply_oriented(orientation, &input, &mut actual)
                .unwrap();
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert!((actual - expected).norm() < 1e-12);
            }
        }
    }
    let mut diagonal = [C::new(0., 0.); 2];
    packets.operator().diagonal(&mut diagonal).unwrap();
    for (actual, expected) in diagonal.into_iter().zip([C::new(12., 8.), C::new(6., 4.)]) {
        assert!((actual - expected).norm() < 1e-12);
    }

    let rebound = mapping
        .with_prescribed(&BTreeMap::from([(key(0), C::new(-2.0, 6.0))]))
        .unwrap();
    assert_eq!(
        rebound.keys().collect::<Vec<_>>(),
        mapping.keys().collect::<Vec<_>>()
    );
    assert_eq!(rebound.free_dof(key(1)), mapping.free_dof(key(1)));
    assert_eq!(rebound.lift(&free, false).unwrap()[0], C::new(-1.0, 3.0));
    assert_eq!(mapping.lift(&free, false).unwrap()[0], C::new(1.0, 2.0));
    assert!(
        mapping
            .with_prescribed(&BTreeMap::from([(key(0), C::new(1.0, f64::NAN))]))
            .is_err()
    );
}
