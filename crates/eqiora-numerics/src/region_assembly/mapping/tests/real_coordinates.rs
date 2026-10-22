//! Independent conjugate-linear global equations, including essential and flux data.
use super::*;
use num_complex::Complex64 as C;

#[test]
fn real_coordinates_assemble_conjugated_trials_and_recover_physical_fields() {
    for natural in [false, true] {
        let forcing = if natural {
            "math.complex(8,2)+math.complex(4,2)*coordinate(0)/3[m]"
        } else {
            "math.complex(1,3)"
        };
        let source = format!(
            "model Wave() {{
            domain body=box(0,6);
            parameter a:complex<m^2>=math.complex(6[m^2],6[m^2]);
            parameter b:complex<1>=math.complex(1,2);
            variable u:complex<1> on body in h1;
            relation balance on body {{ -div(a*grad(u))+u+b*math.conj(u)={forcing}; }}
        }}"
        );
        let (transaction, model, symbols) = compile("real-mapped.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        let field = symbols.get("u").unwrap();
        let domain = symbols.get("body").unwrap();
        let form = CompiledRegionForm::<C>::derive(&program, domain, 1).unwrap();
        let fields = [RegionFieldBinding {
            field,
            space: Space::continuous_lagrange(NonZeroU16::MIN),
            scale: DynQuantity::new(2., DimExponents::DIMENSIONLESS),
        }];
        let rows = form
            .rows()
            .map(|(relation, _, _)| {
                (
                    relation,
                    DynQuantity::new(
                        1.,
                        DimExponents::from_integers([0, -1, 0, 0, 0, 0, 0]).unwrap(),
                    ),
                )
            })
            .collect();
        let reference = ReferenceCell::hypercube(1).unwrap();
        let bound = form.bind(reference, &fields, &rows, None).unwrap();
        let mesh = CartesianMesh::from_axes(vec![vec![0., 3., 6.]]).unwrap();
        let key = |vertex| FieldDof {
            field,
            entity: MeshEntity::new(0, vertex),
            slot: 0,
            component: 0,
        };
        let mut fixed = BTreeMap::from([(key(0), C::new(1., 3.))]);
        if !natural {
            fixed.insert(key(2), C::new(3., -1.));
        }
        let mapping = RegionDofMap::new(
            &mesh,
            &BTreeMap::from([(domain, bound.fields().to_vec())]),
            reference,
            &[domain; 2],
            &[],
            &fixed,
        )
        .unwrap();
        let mut facets = Vec::new();
        if natural {
            let facet = MeshEntity::new(0, 2);
            let parent = mesh.incidence(facet, 1).unwrap()[0];
            facets.push((
                parent.entity.index(),
                prepare_mesh_facet(&mesh, &bound, field, facet, parent, |_, _| {
                    Ok(vec![C::new(0., 4.)])
                })
                .unwrap(),
            ));
        }
        let input = RegionSolveInput {
            conservative: None,
            operator_properties: LinearOperatorProperties::General,
            forms: vec![(bound, QuadratureRule::gauss_legendre(2).unwrap())],
            natural: facets,
            previous: None,
            prescribed_states: BTreeMap::new(),
            geometry_action: None,
        };
        let coordinates = RegionRealCoordinates::new(&mapping).unwrap();
        let (reduced, full, _) = coordinates.assemble_static(&mesh, &input).unwrap();
        assert_eq!(full.rhs().len(), 6);
        assert_eq!(reduced.rhs().len(), if natural { 4 } else { 2 });
        if !natural {
            // h=3: the two-cell interior coefficient is (6+4i)u+(2+4i)conj(u).
            // Essential ends contribute 2-8i, leaving 1+17i. Normalization u=2z
            // gives [16,0;16,8] z=[1,17], hence physical u=1/8+4i.
            let matrix = reduced.matrix();
            let mut dense = [[0.; 2]; 2];
            for (row, entries) in dense.iter_mut().enumerate() {
                for slot in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                    entries[matrix.column_indices()[slot]] = matrix.values()[slot];
                }
            }
            for (actual, expected) in dense.into_iter().flatten().zip([16., 0., 16., 8.]) {
                assert!((actual - expected).abs() < 2e-12);
            }
            for (actual, expected) in reduced.rhs().iter().zip([1., 17.]) {
                assert!((actual - expected).abs() < 2e-12);
            }
        }
        let canonical = eqiora_solver::CanonicalCsrSystemView::new(
            &reduced,
            reduced.rhs(),
            LinearOperatorProperties::General,
        )
        .unwrap();
        let policy = SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1e-12,
            1e-14,
            NonZeroUsize::new(40).unwrap(),
        )
        .unwrap();
        let solution = REFERENCE_LINEAR_SOLVER
            .solve(&canonical.linear_problem().unwrap(), policy)
            .unwrap();
        let recovered = coordinates.recover(solution.values()).unwrap();
        let encoded = coordinates.validated_algebraic_point(&recovered).unwrap();
        for (actual, expected) in encoded.iter().zip(solution.values()) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(
            recovered[&field].value_type,
            mapping.field_layout(field).unwrap().1.value_type
        );
        assert_eq!(recovered[&field].space, fields[0].space);
        let expected = if natural {
            // u=1+3i+x(1+i)/3, a*u'=4i at the right end.
            [C::new(1., 3.), C::new(2., 4.), C::new(3., 5.)]
        } else {
            [C::new(1., 3.), C::new(0.125, 4.), C::new(3., -1.)]
        };
        for (vertex, value) in expected.into_iter().enumerate() {
            assert!((recovered[&field].coefficients[&key(vertex)] - value).norm() < 2e-10);
        }
        assert!(
            coordinates
                .recover(&vec![f64::NAN; reduced.rhs().len()])
                .is_err()
        );
    }
}

#[test]
fn real_fields_keep_one_coordinate_and_prescribed_words() {
    let source = "model Real() { domain body=box(0,1); variable u:1 on body; relation balance on body {u=1;} }";
    let (transaction, model, symbols) = compile("real-coordinate-width.eqi", source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let field = symbols.get("u").unwrap();
    let domain = symbols.get("body").unwrap();
    let mesh = CartesianMesh::from_axes(vec![vec![0., 1.]]).unwrap();
    let reference = ReferenceCell::hypercube(1).unwrap();
    let selected = [DomainFieldDiscretization::new(
        domain.downcast().unwrap(),
        [FieldSpaceBinding::new(
            field.downcast().unwrap(),
            Space::continuous_lagrange(NonZeroU16::MIN),
        )],
        [],
    )
    .unwrap()];
    let layouts = field_layouts(
        &program,
        &selected,
        reference,
        &BTreeMap::from([(field, DynQuantity::new(2., DimExponents::DIMENSIONLESS))]),
    )
    .unwrap();
    let key = |vertex| FieldDof {
        field,
        entity: MeshEntity::new(0, vertex),
        slot: 0,
        component: 0,
    };
    let fixed = C::new(-0., -0.);
    let mapping = RegionDofMap::new(
        &mesh,
        &layouts,
        reference,
        &[domain],
        &[],
        &BTreeMap::from([(key(0), fixed)]),
    )
    .unwrap();
    let coordinates = RegionRealCoordinates::new(&mapping).unwrap();
    let recovered = coordinates.recover(&[3.]).unwrap();
    let actual = recovered[&field].coefficients[&key(0)];
    assert_eq!(actual.re.to_bits(), fixed.re.to_bits());
    assert_eq!(actual.im.to_bits(), fixed.im.to_bits());
    assert_eq!(recovered[&field].coefficients[&key(1)], C::new(6., 0.));
    assert_eq!(
        recovered[&field].value_type.scalar_domain(),
        eqiora_core::ScalarDomain::Real
    );
    assert!(coordinates.recover(&[3., 4.]).is_err());
    assert_eq!(
        coordinates.validated_algebraic_point(&recovered).unwrap(),
        [3.]
    );
    let mut imaginary = recovered.clone();
    imaginary
        .get_mut(&field)
        .unwrap()
        .coefficients
        .get_mut(&key(1))
        .unwrap()
        .im = 1.;
    assert!(coordinates.validated_algebraic_point(&imaginary).is_err());
    assert!(mapping.physical_predictor(&imaginary).is_err());
    // Stale essential values may be replaced by a predictor, but their physical
    // scalar domain cannot be ignored merely because the coordinate is fixed.
    let mut imaginary_fixed = recovered.clone();
    imaginary_fixed
        .get_mut(&field)
        .unwrap()
        .coefficients
        .get_mut(&key(0))
        .unwrap()
        .im = 1.;
    assert!(mapping.physical_predictor(&imaginary_fixed).is_err());
    let invalid = mapping
        .with_prescribed(&BTreeMap::from([(key(0), C::new(1., 2.))]))
        .unwrap();
    assert!(RegionRealCoordinates::new(&invalid).is_err());
}
