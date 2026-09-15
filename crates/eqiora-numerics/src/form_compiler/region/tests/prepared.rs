use super::*;

const HEAT: &str = "model Heat() {
 domain body = box(0, 1, 0, 1);
 parameter diffusivity: m^2/s = 3;
 state temperature: K on body;
 relation balance on body { derivative(temperature) - div(diffusivity * grad(temperature)) = 0; }
}";

fn heat(step: f64, diffusivity: f64) -> BoundRegionForm {
    let compiled = derive(&HEAT.replace("m^2/s = 3", &format!("m^2/s = {diffusivity}"))).unwrap();
    let fields = compiled
        .fields()
        .map(|(field, ty)| RegionFieldBinding {
            field,
            space: p1(),
            scale: DynQuantity::new(1., ty.dimension()),
        })
        .collect::<Vec<_>>();
    let rows = compiled
        .rows()
        .map(|(relation, _, ty)| {
            (
                relation,
                DynQuantity::new(
                    1.,
                    ty.dimension()
                        .mul(dim([0, 2, 0, 0, 0, 0, 0]))
                        .unwrap()
                        .pow(-1, 1)
                        .unwrap(),
                ),
            )
        })
        .collect();
    compiled
        .bind(
            ReferenceCell::simplex(2).unwrap(),
            &fields,
            &rows,
            Some(&RegionTimeBinding {
                step: DynQuantity::new(step, dim([0, 0, 1, 0, 0, 0, 0])),
                states: Vec::new(),
            }),
        )
        .unwrap()
}

#[test]
fn prepared_heat_reuses_exact_mass_and_stiffness_with_fresh_physical_history() {
    let form = heat(1., 3.);
    let field = form.fields()[0].field;
    let rule = simplex_duffy_gauss_legendre(2, 3).unwrap();
    let prepared = form.prepare_cell(&geometry(), &rule).unwrap();
    for history in [vec![2., 2., 2.], vec![1., 2., 4.]] {
        let previous = BTreeMap::from([(field, history.clone())]);
        let action = prepared.linearize(&previous, &history).unwrap();
        let residual = prepared.residual(&previous, &history).unwrap();
        for (i, gradient_i) in GRADIENT.iter().enumerate() {
            let mut expected = 0.;
            for (j, gradient_j) in GRADIENT.iter().enumerate() {
                let stiffness =
                    1.5 * (gradient_i[0] * gradient_j[0] + gradient_i[1] * gradient_j[1]);
                close(action.jacobian[i * 3 + j], mass(i, j) + stiffness);
                expected += stiffness * history[j];
            }
            close(action.residual[i], expected);
            assert_eq!(residual[i], action.residual[i]);
        }
    }
    assert!(prepared.evaluate(&BTreeMap::new()).is_err());
    for values in [vec![1., 2.], vec![1., 2., f64::NAN]] {
        assert!(
            prepared
                .residual(&BTreeMap::from([(field, values)]), &[1., 2., 3.])
                .is_err()
        );
    }
}

#[test]
fn new_heat_geometry_time_and_coefficient_bindings_do_not_reuse_old_actions() {
    let rule = simplex_duffy_gauss_legendre(2, 3).unwrap();
    let base = heat(1., 3.);
    let old = base.prepare_cell(&geometry(), &rule).unwrap();
    for (step, diffusivity, length) in [(2., 3., 1.), (1., 7., 1.), (1., 3., 2.)] {
        let form = heat(step, diffusivity);
        let geom = AffineGeometryMap::new(
            ReferenceCell::simplex(2).unwrap(),
            2,
            vec![0., 0.],
            vec![length, 0., 0., length],
        )
        .unwrap();
        let prepared = form.prepare_cell(&geom, &rule).unwrap();
        let previous = BTreeMap::from([(form.fields()[0].field, vec![2.; 3])]);
        let local = prepared.evaluate(&previous).unwrap();
        for (i, gradient_i) in GRADIENT.iter().enumerate() {
            close(local.rhs()[i], length * length / (3. * step));
            for (j, gradient_j) in GRADIENT.iter().enumerate() {
                let stiffness = diffusivity
                    * 0.5
                    * (gradient_i[0] * gradient_j[0] + gradient_i[1] * gradient_j[1]);
                close(
                    local.matrix()[i * 3 + j],
                    length * length * mass(i, j) / step + stiffness,
                );
            }
        }
    }
    let local = old
        .evaluate(&BTreeMap::from([(base.fields()[0].field, vec![2.; 3])]))
        .unwrap();
    close(local.matrix()[0], mass(0, 0) + 3.);
}
