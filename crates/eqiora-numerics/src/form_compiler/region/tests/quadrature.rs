use super::*;

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
