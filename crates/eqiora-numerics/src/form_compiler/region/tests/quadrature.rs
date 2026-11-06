use super::*;
use eqiora_meshing::QuadratureRule;

#[test]
fn quadratic_diffusion_and_mass_use_the_actual_bound_basis_degree() {
    let quadratic = Space::continuous_lagrange(std::num::NonZeroU16::new(2).unwrap());
    for dimension in [2, 3] {
        let bounds = if dimension == 2 {
            "0,1,0,1"
        } else {
            "0,1,0,1,0,1"
        };
        // A reaction-diffusion residual keeps u an unknown Field; a standalone
        // algebraic equality is correctly classified as prescribed data.
        for (equation, simplex_degree) in [
            ("-div(grad(u))=1[K/m^2]", 2),
            ("-div(grad(u))+u/1[m^2]=1[K/m^2]", 4),
        ] {
            let form = derive_dimension(&format!("model Quadratic() {{ domain body=box({bounds}); variable u:K on body in h1; relation balance on body {{ {equation}; }} }}"), dimension).unwrap();
            let (fields, rows) = form.si_bindings(|_| Ok(quadratic)).unwrap();
            for reference in [
                ReferenceCell::simplex(dimension).unwrap(),
                ReferenceCell::hypercube(dimension).unwrap(),
            ] {
                let bound = form.bind(reference, &fields, &rows, None).unwrap();
                let expected = if reference.family() == eqiora_meshing::ReferenceCellFamily::Simplex
                {
                    simplex_degree
                } else {
                    4
                };
                assert_eq!(bound.required_polynomial_exactness(), expected);
                let n = bound.minimum_gauss_points();
                if reference.family() == eqiora_meshing::ReferenceCellFamily::Hypercube {
                    assert_eq!(n, 3);
                }
                let map = AffineGeometryMap::new(
                    reference,
                    dimension,
                    vec![0.; dimension],
                    (0..dimension * dimension)
                        .map(|i| f64::from(i / dimension == i % dimension))
                        .collect(),
                )
                .unwrap();
                let rule = |n| {
                    if reference.family() == eqiora_meshing::ReferenceCellFamily::Simplex {
                        simplex_duffy_gauss_legendre(dimension, n).unwrap()
                    } else {
                        QuadratureRule::tensor_product_gauss_legendre(dimension, n).unwrap()
                    }
                };
                assert!(bound.prepare_cell(&map, &rule(n)).is_ok());
                assert!(bound.prepare_cell(&map, &rule(n - 1)).is_err());
            }
        }
    }
}

#[test]
fn constant_basis_keeps_the_polynomial_data_degree() {
    use eqiora_meshing::QuadratureRule;
    let reference = ReferenceCell::hypercube(1).unwrap();
    let map = AffineGeometryMap::new(reference, 1, vec![0.5], vec![0.5]).unwrap();
    for (source, points, integral) in [("1", 1, 1.), ("x*x", 2, 1. / 3.)] {
        let form = derive_dimension(
            &format!(
                "model ConstantBasis() {{ domain body=box(0,1);
                variable x:1 on body; state u:1 on body;
                relation position on body {{x=coordinate(0)/1[m];}}
                relation balance on body {{derivative(u)*1[s]-({source})=0;}} }}"
            ),
            1,
        )
        .unwrap();
        let (fields, rows) = form.si_bindings(|_| Ok(Space::cell_constant())).unwrap();
        let bound = form
            .bind(
                reference,
                &fields,
                &rows,
                Some(&RegionTimeBinding {
                    step: DynQuantity::new(1., dim([0, 0, 1, 0, 0, 0, 0])),
                    states: Vec::new(),
                }),
            )
            .unwrap();
        assert_eq!(bound.minimum_gauss_points(), points);
        let rule = QuadratureRule::tensor_product_gauss_legendre(1, points).unwrap();
        let local = bound
            .prepare_cell(&map, &rule)
            .unwrap()
            .evaluate(&BTreeMap::from([(fields[0].field, vec![0.])]))
            .unwrap();
        assert!((local.matrix()[0] - 1.).abs() < 1e-14);
        assert!((local.rhs()[0] - integral).abs() < 1e-14);
        if points > 1 {
            let insufficient = QuadratureRule::tensor_product_gauss_legendre(1, 1).unwrap();
            assert!(bound.prepare_cell(&map, &insufficient).is_err());
        }
    }
}

#[test]
fn polynomial_source_and_diffusion_determine_the_shared_quadrature_degree() {
    let form = derive(
        "model Polynomial() {
        domain body=box(0,1,0,1);
        variable x:1 on body;
        variable u:K on body in h1;
        relation coordinate_value on body { x=coordinate(0)/1[m]; }
        relation balance on body { -div((1+x^6)*grad(u))=1[K/m^2]*x^8; }
    }",
    )
    .unwrap();
    let (fields, rows) = form.si_bindings(|_| Ok(p1())).unwrap();
    let bound = form
        .bind(ReferenceCell::simplex(2).unwrap(), &fields, &rows, None)
        .unwrap();
    let n = bound.minimum_gauss_points();
    assert_eq!(n, 6); // Degree-nine source/test product; Duffy degree is 2n-2.
    let rule = simplex_duffy_gauss_legendre(2, n).unwrap();
    let local = bound
        .prepare_cell(&geometry(), &rule)
        .unwrap()
        .evaluate(&BTreeMap::new())
        .unwrap();
    // int_T x^i*y^j = i!j!/(i+j+2)!. Thus int_T (1+x^6)=29/56.
    // The three hats are 1-x-y, x, y, giving x^8 loads below.
    for (actual, expected) in local.rhs().iter().zip([1. / 990., 1. / 110., 1. / 990.]) {
        assert!((actual - expected).abs() < 1e-14);
    }
    for (i, left) in GRADIENT.iter().enumerate() {
        for (j, right) in GRADIENT.iter().enumerate() {
            let expected = (left[0] * right[0] + left[1] * right[1]) * 29. / 56.;
            assert!((local.matrix()[3 * i + j] - expected).abs() < 1e-13);
        }
    }
    assert!(
        bound
            .prepare_cell(
                &geometry(),
                &simplex_duffy_gauss_legendre(2, n - 1).unwrap()
            )
            .is_err()
    );
}
