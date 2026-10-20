//! Independent one-cell integrals exercise values, gradients and physical scales.
use super::*;
use eqiora_meshing::QuadratureRule;

#[test]
fn natural_facet_retains_material_domain_when_volume_samples_are_valid() {
    use eqiora_core::diagnostic::codes;
    use eqiora_meshing::{EntityIncidence, MeshEntity, OrientationCode};

    for profile in ["first_partials", "value_only"] {
        let source = source("-div(response(value=u)*grad(u))=0", true, 1)
            .replace("value<=4", "value<=1.99")
            .replace("box(0,2)", "box(0,1)")
            .replace("first_partials", profile);
        let form = derive_dimension(&source, 1).unwrap();
        let (fields, rows) = form.si_bindings(|_| Ok(p1())).unwrap();
        let reference = ReferenceCell::hypercube(1).unwrap();
        let bound = form.bind(reference, &fields, &rows, None).unwrap();
        let field = bound.fields()[0].field;
        let geometry = AffineGeometryMap::new(reference, 1, vec![0.5], vec![0.5]).unwrap();
        let point = [1., 2.];
        let cell = bound
            .prepare_cell(
                &geometry,
                &QuadratureRule::tensor_product_gauss_legendre(1, 3).unwrap(),
            )
            .unwrap();
        // u=1+x. Every volume sample has u <= (3+sqrt(3/5))/2 < 1.99;
        // only the outlet u(1)=2 violates the declared material domain.
        assert!(cell.residual(&BTreeMap::new(), &point).is_ok());
        for (side, x, flux) in [(0, 0., -2.), (1, 1., 5.)] {
            let facet = AffineGeometryMap::new(ReferenceCell::point(), 1, vec![x], vec![]).unwrap();
            let incidence = EntityIncidence {
                entity: MeshEntity::new(1, 0),
                local_ordinal: side,
                orientation: OrientationCode::new(0),
            };
            let facet = bound
                .prepare_natural_facet(
                    field,
                    &geometry,
                    (&facet, incidence, &[side]),
                    &QuadratureRule::point(),
                    |_, _| Ok(vec![flux]),
                )
                .unwrap();
            if side == 0 {
                assert_eq!(facet.residual(&point).unwrap(), [2., 0.]);
                // Checking the original primal does not request a derivative
                // of the replaced traction, even for a value-only property.
                assert_eq!(facet.linearize(&point).unwrap().residual, [2., 0.]);
            } else {
                assert_eq!(
                    facet.residual(&point).unwrap_err().code(),
                    codes::EXPRESSION_DOMAIN_VIOLATION
                );
                assert_eq!(
                    facet.linearize(&point).err().unwrap().code(),
                    codes::EXPRESSION_DOMAIN_VIOLATION
                );
            }
        }
    }
}

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
fn constitutive_facet_checks_keep_cell_constant_values() {
    use eqiora_meshing::{EntityIncidence, MeshEntity, OrientationCode};
    let source = source("-div(response(value=c)*grad(u))=0", true, 1)
        .replace("value>=0", "value>=1")
        .replace(
            "variable u:1 on body in smooth;",
            "variable u:1 on body in smooth; variable c:1 on body in smooth;
             relation material_state on body {c^2=4;}",
        );
    let form = derive_dimension(&source, 1).unwrap();
    let flux_field = form
        .rows
        .iter()
        .find(|row| row.has_boundary_flux())
        .unwrap()
        .tested;
    let (fields, rows) = form
        .si_bindings(|field| {
            Ok(if field == flux_field {
                p1()
            } else {
                Space::cell_constant()
            })
        })
        .unwrap();
    let reference = ReferenceCell::hypercube(1).unwrap();
    let bound = form.bind(reference, &fields, &rows, None).unwrap();
    let cell = AffineGeometryMap::new(reference, 1, vec![1.], vec![1.]).unwrap();
    let facet = AffineGeometryMap::new(ReferenceCell::point(), 1, vec![0.], vec![]).unwrap();
    let incidence = EntityIncidence {
        entity: MeshEntity::new(1, 0),
        local_ordinal: 0,
        orientation: OrientationCode::new(0),
    };
    let facet = bound
        .prepare_natural_facet(
            flux_field,
            &cell,
            (&facet, incidence, &[0]),
            &QuadratureRule::point(),
            |_, _| Ok(vec![0.]),
        )
        .unwrap();
    // A cell-interior constant DOF has trace 2, not zero. The H1 field u=0
    // has zero gradient and traction while the material input c stays valid.
    let mut point = vec![0.; 3];
    let material = bound
        .fields()
        .iter()
        .find(|field| field.field != flux_field)
        .unwrap();
    point[material.range.start] = 2.;
    assert_eq!(facet.residual(&point).unwrap(), [0.; 3]);
    assert_eq!(facet.linearize(&point).unwrap().residual, [0.; 3]);
}

#[test]
fn facet_material_inputs_retain_bubble_normal_gradients() {
    use eqiora_core::diagnostic::codes;
    use eqiora_meshing::{EntityIncidence, MeshEntity, OrientationCode, ReferenceTopology};
    let source = source(
        "-div(response(value=contract(grad(u),grad(u),axes=((0,0),))*1[m^2])*grad(u))=0",
        true,
        2,
    );
    let form = derive_dimension(&source, 2).unwrap();
    let (fields, rows) = form
        .si_bindings(|_| Ok(Space::simplex_p1_bubble()))
        .unwrap();
    let reference = ReferenceCell::simplex(2).unwrap();
    let bound = form.bind(reference, &fields, &rows, None).unwrap();
    let cell = geometry();
    let facet = AffineGeometryMap::from_simplex_vertices(vec![vec![0., 0.], vec![1., 0.]]).unwrap();
    let topology = ReferenceTopology::new(reference).unwrap();
    let ordinal = (0..topology.entity_count(1).unwrap())
        .find(|ordinal| topology.entity(1, *ordinal).unwrap().vertex_ordinals() == [0, 1])
        .unwrap();
    let incidence = EntityIncidence {
        entity: MeshEntity::new(2, 0),
        local_ordinal: ordinal,
        orientation: OrientationCode::new(0),
    };
    let facet = bound
        .prepare_natural_facet(
            bound.fields()[0].field,
            &cell,
            (&facet, incidence, &[0, 1]),
            &simplex_duffy_gauss_legendre(1, 1).unwrap(),
            |_, _| Ok(vec![0.]),
        )
        .unwrap();
    // b=27*x*y*(1-x-y) vanishes on y=0, but db/dy=27/4 at x=1/2.
    // The material input |grad(b)|²=729/16 exceeds 4. Scaling b by 1/10
    // makes the input 729/1600, which is inside the original validity domain.
    assert!(facet.residual(&[0., 0., 0., 0.1]).is_ok());
    assert_eq!(
        facet.residual(&[0., 0., 0., 1.]).unwrap_err().code(),
        codes::EXPRESSION_DOMAIN_VIOLATION
    );
}

#[test]
fn affine_facet_checks_prescribed_material_domain() {
    use eqiora_core::diagnostic::codes;
    use eqiora_meshing::{EntityIncidence, MeshEntity, OrientationCode};

    let source = source(
        "-div(response(value=coordinate(0)/1[m])*grad(u))=0",
        true,
        1,
    )
    .replace("value<=4", "value<=0.99")
    .replace("box(0,2)", "box(0,1)");
    let form = derive_dimension(&source, 1).unwrap();
    assert!(!form.is_nonlinear());
    let (fields, rows) = form.si_bindings(|_| Ok(p1())).unwrap();
    let reference = ReferenceCell::hypercube(1).unwrap();
    let bound = form.bind(reference, &fields, &rows, None).unwrap();
    let cell = AffineGeometryMap::new(reference, 1, vec![0.5], vec![0.5]).unwrap();
    // All three volume abscissae lie below 0.99. Only x=1 violates the
    // original prescribed coefficient's domain, independent of the unknown u.
    assert!(
        bound
            .prepare_cell(
                &cell,
                &QuadratureRule::tensor_product_gauss_legendre(1, 3).unwrap()
            )
            .unwrap()
            .evaluate(&BTreeMap::new())
            .is_ok()
    );
    for (side, x) in [(0, 0.), (1, 1.)] {
        let facet = AffineGeometryMap::new(ReferenceCell::point(), 1, vec![x], vec![]).unwrap();
        let incidence = EntityIncidence {
            entity: MeshEntity::new(1, 0),
            local_ordinal: side,
            orientation: OrientationCode::new(0),
        };
        let prepared = bound.prepare_natural_facet(
            bound.fields()[0].field,
            &cell,
            (&facet, incidence, &[side]),
            &QuadratureRule::point(),
            |_, _| Ok(vec![0.]),
        );
        if side == 0 {
            assert!(prepared.is_ok());
        } else {
            assert_eq!(
                prepared.err().unwrap().code(),
                codes::EXPRESSION_DOMAIN_VIOLATION
            );
        }
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
