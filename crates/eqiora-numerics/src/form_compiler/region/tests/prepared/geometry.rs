use super::*;
use eqiora_meshing::AffineGeometryLinearization;

#[test]
fn geometry_action_differentiates_current_and_history_mass_independently() {
    let form = heat(1., 3.);
    let field = form.fields()[0].field;
    let rule = simplex_duffy_gauss_legendre(2, 3).unwrap();
    let original = geometry();
    let history = BTreeMap::from([(field, vec![1., 2., 4.])]);
    let point = [3., 5., 7.];
    for length in [1., 2.] {
        let current = AffineGeometryMap::new(
            ReferenceCell::simplex(2).unwrap(),
            2,
            vec![0., 0.],
            vec![length, 0., 0., length],
        )
        .unwrap();
        let dilation = AffineGeometryLinearization::new(
            current.clone(),
            vec![0., 0.],
            vec![length, 0., 0., length],
        )
        .unwrap();
        for moving_history in [false, true] {
            let previous = if moving_history { &current } else { &original };
            let stationary = AffineGeometryLinearization::stationary(previous.clone()).unwrap();
            let previous_direction = if moving_history {
                &dilation
            } else {
                &stationary
            };
            let (_, tangent) = form
                .prepare_cell_actions(
                    &current,
                    previous,
                    Some((&dilation, previous_direction)),
                    &rule,
                )
                .unwrap();
            let action = tangent.unwrap().linearize(&history, &point).unwrap();
            // In 2D dilation, mass has degree two and stiffness degree zero.
            for i in 0..3 {
                let mut expected = 0.;
                for (j, value) in point.iter().enumerate() {
                    let mass_dot = 2. * length * length * mass(i, j);
                    close(action.jacobian[3 * i + j], mass_dot);
                    expected += mass_dot
                        * (value
                            - if moving_history {
                                history[&field][j]
                            } else {
                                0.
                            });
                }
                close(action.residual[i], expected);
            }
        }
        if length != 1. {
            assert!(
                form.prepare_cell_actions(
                    &original,
                    &original,
                    Some((&dilation, &dilation)),
                    &rule,
                )
                .is_err()
            );
        }
    }
}

#[test]
fn translated_spatial_diffusion_and_source_use_coordinate_chain_rule() {
    let source = HEAT
        .replace("state temperature", "parameter length: m = 1; parameter scale: K = 1; variable c: m^2/s on body; relation coefficient on body { c - diffusivity * (1 + coordinate(0)/length) = 0; } state temperature")
        .replace("div(diffusivity * grad(temperature))", "div(c * grad(temperature)) - c * scale / length^2");
    let compiled = derive(&source).unwrap();
    let (mut fields, mut rows, time) = inputs(&compiled, false);
    // Bind unit scales through the ordinary dimensional checks.
    for field in &mut fields {
        field.scale = DynQuantity::new(1., field.scale.dim());
    }
    for scale in rows.values_mut() {
        *scale = DynQuantity::new(1., scale.dim());
    }
    let form = compiled
        .bind(
            ReferenceCell::simplex(2).unwrap(),
            &fields,
            &rows,
            Some(&time),
        )
        .unwrap();
    let field = form.fields()[0].field;
    let map = geometry();
    let translation =
        AffineGeometryLinearization::new(map.clone(), vec![2., 0.], vec![0.; 4]).unwrap();
    let rule = simplex_duffy_gauss_legendre(2, 3).unwrap();
    let (_, tangent) = form
        .prepare_cell_actions(&map, &map, Some((&translation, &translation)), &rule)
        .unwrap();
    let point = [0., 1., 0.];
    let action = tangent
        .unwrap()
        .linearize(&BTreeMap::from([(field, point.to_vec())]), &point)
        .unwrap();
    // dc/dε = 6. Integrals on the unit triangle are area=1/2 and ∫Ni=1/6.
    for (i, gradient) in GRADIENT.iter().enumerate() {
        close(action.residual[i], 3. * gradient[0] - 1.);
        for (j, trial) in GRADIENT.iter().enumerate() {
            close(
                action.jacobian[3 * i + j],
                3. * (gradient[0] * trial[0] + gradient[1] * trial[1]),
            );
        }
    }
}

#[test]
fn dilated_dyadic_residual_and_state_jacobian_obey_homogeneity() {
    let source = "public operator outer_product(input left: spatial[1], input right: spatial[1]): spatial[2] = component(left, 0) * component(right, 1);
model Transport() {
 domain body = box(0, 1, 0, 1);
 parameter density: kg/m^3 = 4;
 state a: vector<m/s, 2> on body;
 relation balance on body { density*derivative(a) + div(density*outer_product(left = a, right = a)) = 0; }
}";
    let form = bound(&derive(source).unwrap(), false);
    let field = form.fields()[0].field;
    let map = geometry();
    let dilation =
        AffineGeometryLinearization::new(map.clone(), vec![0., 0.], vec![1., 0., 0., 1.]).unwrap();
    let rule = simplex_duffy_gauss_legendre(2, 5).unwrap();
    let (primal, tangent) = form
        .prepare_cell_actions(&map, &map, Some((&dilation, &dilation)), &rule)
        .unwrap();
    let point = [0.3, -0.1, 0.2, 0.4, -0.2, 0.1];
    let history = BTreeMap::from([(field, point.iter().map(|x| 2. * x).collect())]);
    let primal = primal.linearize(&history, &point).unwrap();
    let tangent = tangent.unwrap().linearize(&history, &point).unwrap();
    // Nodal dyadic terms have one gradient: dilation degree d-1=1.
    // Mass has degree d=2. At unchanged physical history only convection
    // remains in the residual, while the Jacobian retains the mass term.
    for i in 0..6 {
        close(tangent.residual[i], primal.residual[i]);
        for j in 0..6 {
            let extra_mass = if i % 2 == j % 2 {
                13. * 4. * 4. * 2. * mass(i / 2, j / 2)
            } else {
                0.
            };
            close(
                tangent.jacobian[6 * i + j],
                primal.jacobian[6 * i + j] + extra_mass,
            );
        }
    }
    assert!(primal.residual.iter().any(|value| value.abs() > 0.01));
}
