//! Non-FSI volume feedback: analytic P1 blocks, independent of compiled actions.
use super::*;

fn coupling(row: usize, column: usize, omit: bool) -> f64 {
    if row == column || (omit && row == 0 && column == 1) {
        0.
    } else {
        ((row + 1) * (column + 2)) as f64
    }
}

fn check(count: usize, renamed: bool, omit: bool) -> Vec<f64> {
    let names = if renamed {
        ["cobalt", "amber", "violet"]
    } else {
        ["a", "b", "c"]
    };
    let mut source = String::from(
        "public operator dyad(input left: spatial[1], input right: spatial[1]): spatial[2] = component(left, 0) * component(right, 1);\nmodel Feedback() {\n domain body = box(0, 1, 0, 1);\n parameter rho: kg/m^3 = 4;\n parameter rate: kg/(m^3*s) = 1;\n",
    );
    let order = if renamed {
        (0..count).rev().collect::<Vec<_>>()
    } else {
        (0..count).collect::<Vec<_>>()
    };
    for &i in &order {
        source += &format!("state {}: vector<m/s, 2> on body;\n", names[i]);
    }
    for &i in &order {
        let name = names[i];
        source += &format!(
            "relation balance{i} on body {{ rho*derivative({name}) + div(rho*dyad(left = {name}, right = {name}))",
        );
        for (j, other) in names.iter().enumerate().take(count) {
            let coefficient = coupling(i, j, omit);
            if coefficient != 0. {
                source += &format!(" + {coefficient}*rate*({name} - {other})");
            }
        }
        source += " = 0; }\n";
    }
    source += "}";
    if renamed {
        source = source.replace("Feedback", "RenamedMixture");
    }
    let (transaction, model, symbols) = compile("feedback.eqi", &source)
        .unwrap()
        .remove(0)
        .into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let compiled =
        CompiledRegionForm::<f64>::derive(&program, symbols.get("body").unwrap(), 2).unwrap();
    let ids = names[..count]
        .iter()
        .map(|name| symbols.get(name).unwrap())
        .collect::<Vec<_>>();
    let scales = [2., 3., 5.];
    let row_scales = [1., -2., 3.];
    let bindings = compiled
        .fields()
        .map(|(field, value_type)| RegionFieldBinding {
            field,
            space: p1(),
            scale: DynQuantity::new(
                scales[ids.iter().position(|id| *id == field).unwrap()],
                value_type.dimension(),
            ),
        })
        .collect::<Vec<_>>();
    let rows = compiled
        .rows()
        .map(|(relation, field, value_type)| {
            (
                relation,
                DynQuantity::new(
                    row_scales[ids.iter().position(|id| *id == field).unwrap()],
                    value_type
                        .dimension()
                        .mul(dim([0, 2, 0, 0, 0, 0, 0]))
                        .unwrap()
                        .pow(-1, 1)
                        .unwrap(),
                ),
            )
        })
        .collect();
    let form = compiled
        .bind(
            ReferenceCell::simplex(2).unwrap(),
            &bindings,
            &rows,
            Some(&RegionTimeBinding {
                step: DynQuantity::new(1., dim([0, 0, 1, 0, 0, 0, 0])),
                states: Vec::new(),
            }),
        )
        .unwrap();
    assert_eq!(form.fields().len(), count);
    let offsets = ids
        .iter()
        .map(|id| {
            form.fields()
                .iter()
                .find(|field| field.field == *id)
                .unwrap()
                .range
                .start
        })
        .collect::<Vec<_>>();
    let velocities = [[1., 2.], [2., 1.5], [3., 1.]];
    let mut point = vec![0.; 6 * count];
    let mut previous = BTreeMap::new();
    for i in 0..count {
        previous.insert(ids[i], velocities[i].repeat(3));
        for node in 0..3 {
            for component in 0..2 {
                point[offsets[i] + 2 * node + component] = velocities[i][component] / scales[i];
            }
        }
    }
    let cell = form
        .prepare_cell(&geometry(), &simplex_duffy_gauss_legendre(2, 5).unwrap())
        .unwrap();
    let action = cell.linearize(&previous, &point).unwrap();
    let primal = cell.residual(&previous, &point).unwrap();
    let mut physical_order = Vec::new();
    for i in 0..count {
        for (a, gradient) in GRADIENT.iter().enumerate() {
            let directional = velocities[i][0] * gradient[0] + velocities[i][1] * gradient[1];
            for (component, velocity) in velocities[i].iter().copied().enumerate() {
                let row = offsets[i] + 2 * a + component;
                // Area = 1/2, integral N_a = 1/6. History cancels storage.
                let feedback = (0..count)
                    .map(|j| coupling(i, j, omit) * (velocity - velocities[j][component]))
                    .sum::<f64>();
                let residual = row_scales[i] * (-2. * velocity * directional + feedback / 6.);
                close(action.residual[row], residual);
                close(primal[row], residual);
                physical_order.push(action.residual[row]);
                for j in 0..count {
                    for b in 0..3 {
                        for (k, derivative) in gradient.iter().enumerate() {
                            let delta = f64::from(component == k);
                            let expected = if i == j {
                                let feedback = (0..count)
                                    .map(|other| coupling(i, other, omit))
                                    .sum::<f64>();
                                (4. + feedback) * mass(a, b) * delta
                                    - (4. / 6.) * (delta * directional + velocity * derivative)
                            } else {
                                -coupling(i, j, omit) * mass(a, b) * delta
                            };
                            let column = offsets[j] + 2 * b + k;
                            close(
                                action.jacobian[row * point.len() + column],
                                row_scales[i] * scales[j] * expected,
                            );
                        }
                    }
                }
            }
        }
    }
    physical_order
}

#[test]
fn coupled_transport_preserves_all_scaled_feedback_blocks_and_renaming() {
    for count in [2, 3] {
        let original = check(count, false, false);
        let renamed = check(count, true, false);
        for (actual, expected) in renamed.into_iter().zip(&original) {
            close(actual, *expected);
        }
        let missing = check(count, false, true);
        assert!(
            original[..6]
                .iter()
                .zip(&missing[..6])
                .any(|(a, b)| (a - b).abs() > 0.1)
        );
        for (actual, expected) in missing[6..].iter().zip(&original[6..]) {
            close(*actual, *expected);
        }
    }
}
