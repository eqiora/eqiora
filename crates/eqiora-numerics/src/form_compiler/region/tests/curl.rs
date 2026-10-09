use super::*;
use num_complex::Complex64 as C;

fn fixture<S: Coefficient>(
    complex: bool,
    volume: &str,
    flux: &str,
) -> (
    KernelProgram,
    CompiledRegionForm<S>,
    BTreeMap<String, RawId>,
) {
    let (scalar, parameter) = if complex {
        ("complex<1>", "complex<m^2> = math.complex(2[m^2], -1[m^2])")
    } else {
        ("1", "m^2 = 2[m^2]")
    };
    let source = format!(
        "model CurlBlock() {{
        domain body = box(0, 2, 0, 3, 0, 4);
        domain wall = boundary(body, axis=0, side=lower);
        parameter a: {parameter};
        variable u: vector<{scalar}, 3> on body in smooth;
        relation balance on body {{ {volume} = 0; }}
        relation law on wall {{ {flux} = 0; }}
    }}"
    );
    let (transaction, model, symbols) = compile("curl-region.eqi", &source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let ids = ["body", "wall", "balance", "law", "u"]
        .into_iter()
        .map(|name| (name.to_owned(), symbols.get(name).unwrap()))
        .collect::<BTreeMap<_, _>>();
    let form = CompiledRegionForm::<S>::derive(&program, ids["body"], 3).unwrap();
    (program, form, ids)
}

fn matrix<S: Coefficient>(form: &CompiledRegionForm<S>) -> eqiora_assembly::LocalContribution<S> {
    let field = form.fields().next().unwrap().0;
    let relation = form.rows().next().unwrap().0;
    let reference = ReferenceCell::simplex(3).unwrap();
    let bound = form
        .bind(
            reference,
            &[RegionFieldBinding {
                field,
                space: p1(),
                scale: DynQuantity::new(1.0, DimExponents::DIMENSIONLESS),
            }],
            &BTreeMap::from([(relation, DynQuantity::new(1.0, dim([0, -3, 0, 0, 0, 0, 0])))]),
            None,
        )
        .unwrap();
    let geometry = AffineGeometryMap::new(
        reference,
        3,
        vec![0.0; 3],
        vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0],
    )
    .unwrap();
    bound
        .prepare_cell(&geometry, &simplex_duffy_gauss_legendre(3, 3).unwrap())
        .and_then(|cell| cell.evaluate(&BTreeMap::new()))
        .unwrap()
}

#[test]
fn compiled_vector_curls_use_the_shared_real_and_complex_contraction() {
    let (_, real, _) = fixture::<f64>(false, "a*curl(curl(u))", "tangential_trace(curl(u))");
    let (_, complex, _) = fixture::<C>(true, "a*curl(curl(u))", "tangential_trace(curl(u))");
    assert_eq!(real.rows[0].terms[0].pairing, Pairing::Curl);
    let real = matrix(&real);
    let complex = matrix(&complex);
    // Independent curls of the twelve Cartesian P1 functions on the tetrahedron
    // with intercepts (2,3,4). Volume = 4; gradients are barycentric constants.
    let curls = [
        [0.0, -0.25, 1.0 / 3.0],
        [0.25, 0.0, -0.5],
        [-1.0 / 3.0, 0.5, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5],
        [0.0, -0.5, 0.0],
        [0.0, 0.0, -1.0 / 3.0],
        [0.0, 0.0, 0.0],
        [1.0 / 3.0, 0.0, 0.0],
        [0.0, 0.25, 0.0],
        [-0.25, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ];
    for i in 0..12 {
        for j in 0..12 {
            let geometric = 4.0 * (0..3).map(|a| curls[i][a] * curls[j][a]).sum::<f64>();
            close(real.entry(i, j).unwrap(), 2.0 * geometric);
            let expected = C::new(2.0, -1.0) * geometric;
            assert!((complex.entry(i, j).unwrap() - expected).norm() < 1e-12);
        }
    }
    // u=(-y,x,0), curl(u)=(0,0,2): integral |curl(u)|²=16.
    let rotation = [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let energy = (0..12)
        .flat_map(|i| (0..12).map(move |j| (i, j)))
        .map(|(i, j)| rotation[i] * real.entry(i, j).unwrap() * rotation[j])
        .sum::<f64>();
    close(energy, 32.0);
}

#[test]
fn curl_constitutive_flux_preserves_the_green_identity_sign() {
    // The source tangential trace lowers to normal(T(curl(u))). Its positive
    // sign is the constitutive flux of -curl-curl, not +curl-curl.
    for (volume, flux, valid) in [
        ("-curl(curl(u))", "tangential_trace(curl(u))", true),
        ("curl(curl(u))", "tangential_trace(curl(u))", false),
        ("-curl(curl(u))", "normal(grad(u))", false),
    ] {
        let (program, form, ids) = fixture::<f64>(false, volume, flux);
        let law = typed_relation(&program, ids["law"]).unwrap();
        let result = form.require_boundary_flux(
            &program,
            ids["wall"],
            ids["law"],
            ids["u"],
            law.expression().roots()[0],
            false,
        );
        assert_eq!(result.is_ok(), valid, "{flux}: {result:?}");
    }
}
