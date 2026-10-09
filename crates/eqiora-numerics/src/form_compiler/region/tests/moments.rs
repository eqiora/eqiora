use super::*;
use num_complex::Complex64 as C;

fn form<S: Coefficient>(complex: bool, operator: &str) -> CompiledRegionForm<S> {
    let (scalar, parameter) = if complex {
        ("complex<1>", "complex<m^2> = math.complex(2[m^2], -1[m^2])")
    } else {
        ("1", "m^2 = 1[m^2]")
    };
    let source = format!(
        "model Moments() {{
        domain body = box(0,2,0,3,0,4);
        parameter a: {parameter};
        variable u: vector<{scalar},3> on body;
        relation balance on body {{ a*({operator}) = 0; }}
    }}"
    );
    let (transaction, model, symbols) = compile("moments.eqi", &source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    CompiledRegionForm::derive(&program, symbols.get("body").unwrap(), 3).unwrap()
}

fn bind<S: Coefficient>(
    form: &CompiledRegionForm<S>,
    space: Space,
    coefficient_power: i32,
    row_power: i32,
) -> Result<BoundRegionForm<S>, Diagnostic> {
    let field = form.fields().next().unwrap().0;
    let relation = form.rows().next().unwrap().0;
    form.bind(
        ReferenceCell::simplex(3).unwrap(),
        &[RegionFieldBinding {
            field,
            space,
            scale: DynQuantity::new(1.0, dim([0, coefficient_power, 0, 0, 0, 0, 0])),
        }],
        &BTreeMap::from([(
            relation,
            DynQuantity::new(1.0, dim([0, row_power, 0, 0, 0, 0, 0])),
        )]),
        None,
    )
}

fn geometry() -> AffineGeometryMap {
    AffineGeometryMap::new(
        ReferenceCell::simplex(3).unwrap(),
        3,
        vec![0.0; 3],
        vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0],
    )
    .unwrap()
}

#[test]
fn source_forms_bind_integral_units_and_execute_real_and_complex_moment_actions() {
    for (space, operator, power, row_power, coefficients, expected_energy) in [
        (
            Space::tetrahedral_edge(),
            "curl(curl(u))",
            1,
            -2,
            vec![0., 0., 0., 6., 0., 0.],
            16.0,
        ),
        (
            Space::tetrahedral_face(),
            "-grad(div(u))",
            2,
            -1,
            vec![0., 0., 0., 12.],
            36.0,
        ),
    ] {
        let real = form::<f64>(false, operator);
        let complex = form::<C>(true, operator);
        let real = bind(&real, space, power, row_power).unwrap();
        let complex = bind(&complex, space, power, row_power).unwrap();
        assert_eq!(real.fields()[0].range.len(), coefficients.len());
        let rule = simplex_duffy_gauss_legendre(3, 3).unwrap();
        let real = real
            .prepare_cell(&geometry(), &rule)
            .and_then(|cell| cell.evaluate(&BTreeMap::new()))
            .unwrap();
        let complex = complex
            .prepare_cell(&geometry(), &rule)
            .and_then(|cell| cell.evaluate(&BTreeMap::new()))
            .unwrap();
        let phase = C::new(2., -1.);
        for (a, b) in real.matrix().iter().zip(complex.matrix()) {
            assert!((*b - phase * *a).norm() <= 4096.0 * f64::EPSILON * a.abs().max(1.));
        }
        let action = real
            .matrix()
            .chunks_exact(coefficients.len())
            .map(|row| {
                row.iter()
                    .zip(&coefficients)
                    .map(|(a, u)| a * u)
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        // Independent polynomial moments: rotation (-y,x,0) has curl (0,0,2),
        // radial (x,y,z) has divergence 3. Physical tetrahedron volume is 4.
        close(
            action.iter().zip(&coefficients).map(|(a, u)| a * u).sum(),
            expected_energy,
        );
        if space == Space::tetrahedral_face() {
            // Each canonical RT basis has constant divergence sign/4, giving
            // integral(div(phi_i) div(phi_j)) = sign_i sign_j / 4.
            let signs = [-1., 1., -1., 1.];
            for i in 0..4 {
                for j in 0..4 {
                    close(real.entry(i, j).unwrap(), signs[i] * signs[j] / 4.);
                }
            }
        } else {
            let expected = [8. / 3., -8. / 3., 0., 8. / 3., 0., 0.];
            for (actual, expected) in action.into_iter().zip(expected) {
                close(actual, expected);
            }
        }
    }
}

#[test]
fn moment_binding_rejects_nodal_units_wrong_test_normalization_and_incompatible_derivatives() {
    for (space, power, row_power, accepted, rejected) in [
        (
            Space::tetrahedral_edge(),
            1,
            -2,
            "curl(curl(u))",
            "-grad(div(u))",
        ),
        (
            Space::tetrahedral_face(),
            2,
            -1,
            "-grad(div(u))",
            "curl(curl(u))",
        ),
    ] {
        let valid = form::<f64>(false, accepted);
        assert!(bind(&valid, space, power, row_power).is_ok());
        assert!(bind(&valid, space, 0, row_power).is_err());
        assert!(bind(&valid, space, power, -3).is_err());
        let invalid = form::<f64>(false, rejected);
        let error = bind(&invalid, space, power, row_power).unwrap_err();
        assert!(error.message().contains("differential pairing"));
        let gradient = form::<f64>(false, "-div(grad(u))");
        assert!(bind(&gradient, space, power, row_power).is_err());
    }
}

#[test]
fn moment_facets_reject_before_evaluating_a_nodal_load() {
    use eqiora_meshing::{EntityIncidence, MeshEntity, OrientationCode};
    for (space, power, row_power, operator) in [
        (Space::tetrahedral_edge(), 1, -2, "curl(curl(u))"),
        (Space::tetrahedral_face(), 2, -1, "-grad(div(u))"),
    ] {
        let form = form::<f64>(false, operator);
        let bound = bind(&form, space, power, row_power).unwrap();
        let facet = AffineGeometryMap::new(
            ReferenceCell::simplex(2).unwrap(),
            3,
            vec![0.0; 3],
            vec![2.0, 0.0, 0.0, 3.0, 0.0, 0.0],
        )
        .unwrap();
        let incidence = EntityIncidence {
            entity: MeshEntity::new(3, 0),
            local_ordinal: 0,
            orientation: OrientationCode::identity(),
        };
        let error = bound
            .evaluate_natural_facet(
                bound.fields()[0].field,
                &geometry(),
                (&facet, incidence, &[0, 1, 2]),
                &simplex_duffy_gauss_legendre(2, 3).unwrap(),
                |_, _| panic!("unadmitted moment trace reached a datum callback"),
            )
            .unwrap_err();
        assert!(error.message().contains("trace admission"));
    }
}
