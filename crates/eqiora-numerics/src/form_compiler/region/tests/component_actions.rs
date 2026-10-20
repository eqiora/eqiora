//! Independent one-cell integrals exercise values, gradients and physical scales.
use super::*;
use eqiora_meshing::QuadratureRule;

#[test]
fn value_only_material_values_do_not_grant_newton_derivatives() {
    for flux in [false, true] {
        let balance = if flux {
            "-div(response(value=u)*grad(u))=0"
        } else {
            "-div(grad(u))+response(value=u)/1[m^2]=0"
        };
        let source = source(balance, true, 1).replace("first_partials", "value_only");
        let form = derive_dimension(&source, 1).unwrap();
        let (fields, rows) = form.si_bindings(|_| Ok(p1())).unwrap();
        let reference = ReferenceCell::hypercube(1).unwrap();
        let bound = form.bind(reference, &fields, &rows, None).unwrap();
        let geometry = AffineGeometryMap::new(reference, 1, vec![1.], vec![1.]).unwrap();
        let cell = bound
            .prepare_cell(
                &geometry,
                &QuadratureRule::tensor_product_gauss_legendre(1, 3).unwrap(),
            )
            .unwrap();
        let residual = cell.residual(&BTreeMap::new(), &[1., 3.]).unwrap();
        let expected = if flux {
            [-16. / 3., 16. / 3.]
        } else {
            [3., 23. / 3.]
        };
        for (actual, expected) in residual.iter().zip(expected) {
            close(*actual, expected);
        }
        assert!(
            cell.linearize(&BTreeMap::new(), &[1., 3.])
                .err()
                .unwrap()
                .message()
                .contains("value-only contract")
        );
    }
}

#[test]
fn nonlinear_reaction_and_constitutive_flux_use_the_shared_component_ir() {
    for material in [false, true] {
        for flux in [false, true] {
            let response = if material {
                "response(value=u)"
            } else {
                "(1+u^2)"
            };
            let balance = if flux {
                format!("-div({response}*grad(u))=0")
            } else {
                format!("-div(grad(u))+{response}/1[m^2]=0")
            };
            let source = source(&balance, material, 1);
            let form = derive_dimension(&source, 1).unwrap();
            assert!(form.is_nonlinear());
            let (mut fields, rows) = form.si_bindings(|_| Ok(p1())).unwrap();
            fields[0].scale = DynQuantity::new(2., fields[0].scale.dim());
            let reference = ReferenceCell::hypercube(1).unwrap();
            let bound = form.bind(reference, &fields, &rows, None).unwrap();
            // The reference interval is [-1,1]: x=1+xi covers [0,2].
            let geometry = AffineGeometryMap::new(reference, 1, vec![1.], vec![1.]).unwrap();
            let rule = QuadratureRule::tensor_product_gauss_legendre(1, 3).unwrap();
            let cell = bound.prepare_cell(&geometry, &rule).unwrap();
            // Physical u=1+x on [0,2], with algebraic coefficients u/2.
            let point = [0.5, 1.5];
            let action = cell.linearize(&BTreeMap::new(), &point).unwrap();
            let (residual, jacobian) = if flux {
                // q=(1+u²)u', integral q dx=32/3; test gradients ±1/2.
                ([-16. / 3., 16. / 3.], [2., -10., -2., 10.])
            } else {
                // Integrate phi_i(1+u²)+phi_i'u'; J uses 2u phi_i phi_j
                // plus stiffness, then the coefficient scale 2 on each column.
                ([3., 23. / 3.], [5., 5. / 3., 5. / 3., 23. / 3.])
            };
            for (actual, expected) in action
                .residual
                .iter()
                .zip(residual)
                .chain(action.jacobian.iter().zip(jacobian))
            {
                assert!((actual - expected).abs() < 2e-12, "{actual} != {expected}");
            }
            assert_eq!(
                cell.residual(&BTreeMap::new(), &point).unwrap(),
                action.residual
            );
            assert!(cell.evaluate(&BTreeMap::new()).is_err());
            assert!(
                bound
                    .prepare_cell(
                        &geometry,
                        &QuadratureRule::tensor_product_gauss_legendre(1, 1).unwrap()
                    )
                    .is_err()
            );
            if material {
                assert!(cell.residual(&BTreeMap::new(), &[-1., -1.]).is_err());
            }
        }
    }
}

#[test]
fn guarded_material_reaction_retains_cubic_simplex_integration() {
    let source = source("-div(grad(u))+response(value=u)/1[m^2]=0", true, 2);
    let form = derive_dimension(&source, 2).unwrap();
    let (mut fields, rows) = form.si_bindings(|_| Ok(p1())).unwrap();
    fields[0].scale = DynQuantity::new(2., fields[0].scale.dim());
    let bound = form
        .bind(ReferenceCell::simplex(2).unwrap(), &fields, &rows, None)
        .unwrap();
    // A quadratic reaction times a linear test requires degree three even
    // though its validity predicate depends on the unknown value.
    assert!(
        bound
            .prepare_cell(&geometry(), &simplex_duffy_gauss_legendre(2, 2).unwrap())
            .is_err()
    );
    let cell = bound
        .prepare_cell(&geometry(), &simplex_duffy_gauss_legendre(2, 3).unwrap())
        .unwrap();
    let u = [1., 2., 3.];
    let point = u.map(|value| value / 2.);
    let action = cell.linearize(&BTreeMap::new(), &point).unwrap();
    let gradient = [[-1., -1.], [1., 0.], [0., 1.]];
    // Exact unit-triangle barycentric moment:
    // integral lambda_0^a lambda_1^b lambda_2^c = a! b! c! / (a+b+c+2)!.
    let moment = |i, j, k| {
        if i == j && j == k {
            1. / 20.
        } else if i == j || i == k || j == k {
            1. / 60.
        } else {
            1. / 120.
        }
    };
    for a in 0..3 {
        let reaction = (0..3)
            .flat_map(|j| (0..3).map(move |k| u[j] * u[k] * moment(a, j, k)))
            .sum::<f64>();
        close(
            action.residual[a],
            0.5 * (gradient[a][0] + 2. * gradient[a][1]) + 1. / 6. + reaction,
        );
        for b in 0..3 {
            let expected = 2.
                * (0.5 * (gradient[a][0] * gradient[b][0] + gradient[a][1] * gradient[b][1])
                    + 2. * (0..3).map(|k| u[k] * moment(a, b, k)).sum::<f64>());
            close(action.jacobian[a * 3 + b], expected);
        }
    }
}

fn source(balance: &str, material: bool, dimension: usize) -> String {
    let bounds = if dimension == 1 { "0,2" } else { "0,1,0,1" };
    if material {
        format!(
            "property contract Response(input value:1):1 {{derivatives first_partials;}}
property release Material:Response {{
 analytic {{value=1+value^2;source_unit:1=1;}}
 validity value>=0 and value<=4;outside reject;branch single;
 citation org.example.manufactured;license spdx.CC0_1_0;
}}
component Equation(support body:volume(ambient_dimension={dimension}),property response:Response) {{
 variable u:1 on body in smooth;
 relation balance on body {{{balance};}}
}}
model Main() {{domain body=box({bounds});instance physics:Equation(body=body,response=Material);}}"
        )
    } else {
        format!(
            "model Main() {{domain body=box({bounds});variable u:1 on body in smooth;relation balance on body {{{balance};}}}}"
        )
    }
}
