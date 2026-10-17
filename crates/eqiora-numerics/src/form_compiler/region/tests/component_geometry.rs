//! Geometry actions differentiate every entry of a prescribed constitutive map.
use super::*;
use eqiora_meshing::AffineGeometryLinearization;
use num_complex::Complex64;

#[test]
fn translated_structured_coefficients_keep_nonsymmetric_mass_and_load_actions() {
    for complex in [false, true] {
        let source = r#"model TranslatedComponents() {
            domain body=box(0,2,0,3);
            coordinate x:m on body from body[0];
            coordinate y:m on body from body[1];
            parameter a:tensor<1,2,2>=tensor_value(frame=body,components=[[1,2],[4,5]]);
            parameter b:tensor<1,2,2>=tensor_value(frame=body,components=[[1,0],[-1,2]]);
            parameter c:tensor<1,2,2>=tensor_value(frame=body,components=[[0,-3],[0,2]]);
            parameter g:vector<1,2>=tensor_value(frame=body,components=[1,-2]);
            variable q:tensor<1,2,2> on body;
            variable u:vector<1,2> on body in h1;
            relation coefficient on body { q=a+(x/1[m])*b+(y/1[m])*c; }
            relation balance on body { contract(q,u+g,axes=((1,0),))=0; }
        }"#;
        let source = if complex {
            source
                .replace("q:tensor<1,2,2>", "q:tensor<complex<1>,2,2>")
                .replace("u:vector<1,2>", "u:vector<complex<1>,2>")
                .replace(
                    "q=a+(x/1[m])*b+(y/1[m])*c",
                    "q=math.complex(2,3)*(a+(x/1[m])*b+(y/1[m])*c)",
                )
        } else {
            source.to_owned()
        };
        let (transaction, model, symbols) = compile("component-geometry.eqi", &source)
            .unwrap()
            .remove(0)
            .into_parts();
        let mut store = InMemoryGraphStore::new();
        store.commit(transaction).unwrap();
        let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
        if complex {
            translated::<Complex64>(
                &program,
                symbols.get("body").unwrap(),
                Complex64::new(2., 3.),
            );
        } else {
            translated::<f64>(&program, symbols.get("body").unwrap(), 1.);
        }
    }
}

fn translated<S: Coefficient>(program: &KernelProgram, domain: RawId, scale: S) {
    let form = CompiledRegionForm::<S>::derive(program, domain, 2).unwrap();
    let (fields, rows) = form
        .si_bindings(|_| Ok(Space::continuous_lagrange(NonZeroU16::MIN)))
        .unwrap();
    let reference = ReferenceCell::hypercube(2).unwrap();
    let bound = form.bind(reference, &fields, &rows, None).unwrap();
    let map = AffineGeometryMap::new(reference, 2, vec![1., 1.5], vec![1., 0., 0., 1.5]).unwrap();
    let direction =
        AffineGeometryLinearization::new(map.clone(), vec![2., -1.], vec![0.; 4]).unwrap();
    let rule = eqiora_meshing::QuadratureRule::tensor_product_gauss_legendre(2, 3).unwrap();
    let (primal, tangent) = bound
        .prepare_affine_actions(&map, Some(&direction), &rule)
        .unwrap();
    let tangent = tangent.unwrap();
    assert_eq!(primal.rows(), 8);
    // Translation preserves basis gradients and measure. dq=2b-c=[[2,3],[-2,2]].
    // Q1 mass is |Omega|/36 times [[4,2,2,1],...]; each basis integrates to 6/4.
    // d(rhs)=-dq*[1,-2] times 6/4=[6,9] at every node.
    let q = [[2., 3.], [-2., 2.]];
    for i in 0..4 {
        for a in 0..2 {
            let expected = scale * <S as From<f64>>::from([6., 9.][a]);
            let actual = tangent.rhs()[2 * i + a];
            assert!(
                (actual.re() - expected.re()).abs() < 1e-12
                    && (actual.im() - expected.im()).abs() < 1e-12
            );
            for j in 0usize..4 {
                let mass = 6. / 36. * 2_f64.powi(2 - (i ^ j).count_ones() as i32);
                for (b, coefficient) in q[a].iter().enumerate() {
                    let expected = scale * <S as From<f64>>::from(coefficient * mass);
                    let actual = tangent.entry(2 * i + a, 2 * j + b).unwrap();
                    assert!(
                        (actual.re() - expected.re()).abs() < 1e-12
                            && (actual.im() - expected.im()).abs() < 1e-12,
                        "component ({a},{b}), nodes ({i},{j})"
                    );
                }
            }
        }
    }
}
