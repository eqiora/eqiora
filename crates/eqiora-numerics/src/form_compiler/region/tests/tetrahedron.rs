use super::*;

fn factorial(n: usize) -> f64 {
    (1..=n).product::<usize>() as f64
}

#[test]
fn tetrahedron_mini_vector_and_mixed_blocks_match_barycentric_polynomial_integrals() {
    let source = MIXED
        .replace("box(0, 1, 0, 1)", "box(0, 1, 0, 1, 0, 1)")
        .replace("vector<m / s, 2>", "vector<m / s, 3>");
    let form = derive_dimension(&source, 3).unwrap();
    let (fields, rows, time) = inputs(&form, true);
    let reference = ReferenceCell::simplex(3).unwrap();
    let bound = form.bind(reference, &fields, &rows, Some(&time)).unwrap();
    let geometry = AffineGeometryMap::new(
        reference,
        3,
        vec![0.0; 3],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let velocity = bound
        .fields()
        .iter()
        .find(|layout| layout.components == 3)
        .unwrap();
    let pressure = bound
        .fields()
        .iter()
        .find(|layout| layout.components == 1)
        .unwrap();
    let previous = BTreeMap::from([(velocity.field, vec![0.0; 15])]);
    let local = bound
        .prepare_cell(&geometry, &simplex_duffy_gauss_legendre(3, 7).unwrap())
        .and_then(|cell| cell.evaluate(&previous))
        .unwrap();
    let gradient = [
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    // b=256*lambda0*lambda1*lambda2*lambda3; reference tetrahedron volume=1/6.
    // Every expectation below integrates barycentric monomials algebraically, without quadrature.
    let bubble_integral = 256.0 / factorial(7);
    let derivative_product = |i: usize, a: usize, j: usize, b: usize| match (i == 4, j == 4) {
        (false, false) => gradient[i][a] * gradient[j][b] / 6.0,
        (true, false) | (false, true) => 0.0,
        (true, true) => {
            let mut integral = 0.0;
            for r in 0..4 {
                for s in 0..4 {
                    let numerator = (0..4)
                        .map(|axis| factorial(usize::from(axis != r) + usize::from(axis != s)))
                        .product::<f64>();
                    integral += 256.0_f64.powi(2) * gradient[r][a] * gradient[s][b] * numerator
                        / factorial(9);
                }
            }
            integral
        }
    };
    for i in 0..5 {
        for (a, _) in gradient[0].iter().enumerate() {
            let row = velocity.range.start + 3 * i + a;
            close(local.rhs()[row], 0.0);
            for j in 0..5 {
                for b in 0..3 {
                    let mass = match (i == 4, j == 4) {
                        (false, false) => {
                            if i == j {
                                1.0 / 60.0
                            } else {
                                1.0 / 120.0
                            }
                        }
                        (true, false) | (false, true) => 256.0 * 2.0 / factorial(8),
                        (true, true) => 256.0_f64.powi(2) * 16.0 / factorial(11),
                    };
                    let strain = 2.0
                        * ((if a == b {
                            (0..3).map(|c| derivative_product(i, c, j, c)).sum()
                        } else {
                            0.0
                        }) + derivative_product(i, b, j, a));
                    let expected = 13.0 * 2.0 * ((if a == b { 12.0 * mass } else { 0.0 }) + strain);
                    close(
                        local.matrix()[row * 19 + velocity.range.start + 3 * j + b],
                        expected,
                    );
                }
            }
            for j in 0..4 {
                let divergence_pairing = if i == 4 {
                    -gradient[j][a] * bubble_integral
                } else {
                    gradient[i][a] / 24.0
                };
                close(
                    local.matrix()[row * 19 + pressure.range.start + j],
                    -13.0 * 7.0 * divergence_pairing,
                );
                close(
                    local.matrix()[(pressure.range.start + j) * 19 + row],
                    -11.0 * 2.0 * divergence_pairing,
                );
            }
        }
    }
}
