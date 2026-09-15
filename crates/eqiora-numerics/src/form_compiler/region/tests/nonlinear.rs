use super::*;
use eqiora_meshing::{MeshEntity, MeshGeometry, MeshQualityGate, MeshTopology, SimplicialMesh};

const TRANSPORT: &str = "public operator outer_product(input left: spatial[1], input right: spatial[1]): spatial[2] = component(left, 0) * component(right, 1);
model Transport() {
 domain body = box(0, 1, 0, 1);
 parameter density: kg/m^3 = 4;
 state a: vector<m/s, 2> on body;
 relation balance on body { density*derivative(a) + div(density*outer_product(left = a, right = a)) = 0; }
}";

fn bind_unit(form: &CompiledRegionForm) -> BoundRegionForm {
    let bindings = form
        .fields()
        .map(|(field, value_type)| RegionFieldBinding {
            field,
            space: p1(),
            scale: DynQuantity::new(1.0, value_type.dimension()),
        })
        .collect::<Vec<_>>();
    let rows = form
        .rows()
        .map(|(relation, _, value_type)| {
            (
                relation,
                DynQuantity::new(
                    1.0,
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
    form.bind(
        ReferenceCell::simplex(2).unwrap(),
        &bindings,
        &rows,
        Some(&RegionTimeBinding {
            step: DynQuantity::new(1.0, dim([0, 0, 1, 0, 0, 0, 0])),
            states: Vec::new(),
        }),
    )
    .unwrap()
}

#[test]
fn vector_transport_without_pressure_uses_conservative_dyadic_form() {
    let compiled = derive(TRANSPORT).unwrap();
    assert!(!compiled.rows[0].dyadics[0].split);
    let form = bind_unit(&compiled);
    let field = form.fields()[0].field;
    let point = vec![2.0, 3.0, 2.0, 3.0, 2.0, 3.0];
    let previous = BTreeMap::from([(field, point.clone())]);
    let rule = simplex_duffy_gauss_legendre(2, 5).unwrap();
    let action = form
        .prepare_cell(&geometry(), &rule)
        .and_then(|cell| cell.linearize(&previous, &point))
        .unwrap();
    // Integral of each P1 gradient is area*gradient. No numerical output is an oracle.
    for (a, gradient) in GRADIENT.iter().enumerate() {
        for (i, component) in point.iter().take(2).enumerate() {
            let directional = 2.0 * gradient[0] + 3.0 * gradient[1];
            close(action.residual[2 * a + i], -2.0 * component * directional);
            for b in 0..3 {
                for (k, derivative) in gradient.iter().enumerate() {
                    let expected = 4.0 * mass(a, b) * if i == k { 1.0 } else { 0.0 }
                        - (4.0 / 6.0)
                            * (if i == k { directional } else { 0.0 } + component * derivative);
                    close(action.jacobian[(2 * a + i) * 6 + 2 * b + k], expected);
                }
            }
        }
    }
    assert!(
        form.prepare_cell(&geometry(), &rule)
            .and_then(|cell| cell.evaluate(&previous))
            .is_err()
    );
}

#[test]
fn spatial_transport_coefficient_does_not_use_unweighted_divergence_split() {
    let declaration = TRANSPORT.split_once("model Transport").unwrap().0;
    let source = format!("{declaration}{}", MIXED
        .replace("state v:", "parameter length: m = 1;\n variable c: kg/m^3 on body;\n relation coefficient on body { c - density * (1 + coordinate(0)/length) = 0; }\n state v:")
        .replace("density * derivative(v)",
                 "density * derivative(v) + div(c * outer_product(left = v, right = v))"));
    let compiled = derive(&source).unwrap();
    let terms = compiled
        .rows
        .iter()
        .flat_map(|row| &row.dyadics)
        .collect::<Vec<_>>();
    assert_eq!(terms.len(), 1);
    assert!(!terms[0].split);
}

#[test]
fn tensor_product_parent_embedding_preserves_the_complete_outward_flux() {
    use eqiora_meshing::CartesianMesh;
    let compiled = derive(TRANSPORT).unwrap();
    let unit = bind_unit(&compiled);
    let fields = unit
        .fields()
        .iter()
        .map(|layout| RegionFieldBinding {
            field: layout.field,
            space: layout.space,
            scale: DynQuantity::new(1., layout.value_type.dimension()),
        })
        .collect::<Vec<_>>();
    let rows = compiled
        .rows()
        .map(|(relation, _, value_type)| {
            (
                relation,
                DynQuantity::new(
                    1.,
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
            ReferenceCell::hypercube(2).unwrap(),
            &fields,
            &rows,
            Some(&RegionTimeBinding {
                step: DynQuantity::new(1., dim([0, 0, 1, 0, 0, 0, 0])),
                states: Vec::new(),
            }),
        )
        .unwrap();
    let field = form.fields()[0].field;
    let mesh = CartesianMesh::from_axes(vec![vec![0., 1.], vec![0., 2.]]).unwrap();
    let entity = MeshEntity::new(2, 0);
    let cell = mesh.geometry_map(entity).unwrap();
    let point = vec![2., 3., 2., 3., 2., 3., 2., 3.];
    let mut residual = form
        .prepare_cell(
            &cell,
            &eqiora_meshing::QuadratureRule::tensor_product_gauss_legendre(2, 3).unwrap(),
        )
        .and_then(|cell| cell.linearize(&BTreeMap::from([(field, point.clone())]), &point))
        .unwrap()
        .residual;
    let cell_vertices = mesh.entity_vertices(entity).unwrap();
    for index in 0..mesh.entity_count(1).unwrap() {
        let facet = MeshEntity::new(1, index);
        let incidence = mesh.incidence(facet, 2).unwrap()[0];
        let vertices = mesh
            .entity_vertices(facet)
            .unwrap()
            .iter()
            .map(|vertex| {
                cell_vertices
                    .iter()
                    .position(|parent| parent == vertex)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let action = form
            .natural_facet_action(
                field,
                &cell,
                (&mesh.geometry_map(facet).unwrap(), incidence, &vertices),
                &eqiora_meshing::QuadratureRule::tensor_product_gauss_legendre(1, 2).unwrap(),
                &point,
                true,
                |_, _| Ok(vec![0.; 2]),
            )
            .unwrap();
        for (r, value) in residual.iter_mut().zip(action.residual) {
            *r += value;
        }
    }
    for value in residual {
        close(value, 0.);
    }
}

#[test]
fn constant_transport_volume_and_all_outward_facets_cancel() {
    let compiled = derive(TRANSPORT).unwrap();
    let form = bind_unit(&compiled);
    let field = form.fields()[0].field;
    let mesh = SimplicialMesh::new(
        2,
        vec![vec![0., 0.], vec![1., 0.], vec![0., 1.]],
        vec![vec![0, 1, 2]],
        MeshQualityGate::new(0.01).unwrap(),
    )
    .unwrap();
    let cell = mesh.geometry_map(MeshEntity::new(2, 0)).unwrap();
    let point = vec![2., 3., 2., 3., 2., 3.];
    let previous = BTreeMap::from([(field, point.clone())]);
    let mut residual = form
        .prepare_cell(&cell, &simplex_duffy_gauss_legendre(2, 5).unwrap())
        .and_then(|cell| cell.linearize(&previous, &point))
        .unwrap()
        .residual;
    let rule = simplex_duffy_gauss_legendre(1, 3).unwrap();
    for facet in 0..mesh.entity_count(1).unwrap() {
        let entity = MeshEntity::new(1, facet);
        let incidence = mesh.incidence(entity, 2).unwrap()[0];
        let vertices = mesh
            .entity_vertices(entity)
            .unwrap()
            .iter()
            .map(|v| v.index())
            .collect::<Vec<_>>();
        let action = form
            .natural_facet_action(
                field,
                &cell,
                (&mesh.geometry_map(entity).unwrap(), incidence, &vertices),
                &rule,
                &point,
                true,
                |_, _| Ok(vec![0.; 2]),
            )
            .unwrap();
        for (r, value) in residual.iter_mut().zip(action.residual) {
            *r += value;
        }
    }
    for value in residual {
        close(value, 0.);
    }
}

#[test]
fn nonlinear_volume_and_facet_derivatives_match_centered_candidates() {
    check_centered_derivative(TRANSPORT);
    let declaration = TRANSPORT.split_once("model Transport").unwrap().0;
    let source = format!(
        "{declaration}{}",
        MIXED.replace(
            "density * derivative(v)",
            "density * derivative(v) + div(density * outer_product(left = v, right = v))",
        )
    );
    check_centered_derivative(&source);
}

fn check_centered_derivative(source: &str) {
    let compiled = derive(source).unwrap();
    let form = bind_unit(&compiled);
    let vector = form
        .fields()
        .iter()
        .find(|layout| !layout.value_type.shape().is_scalar())
        .unwrap();
    let field = vector.field;
    let mesh = SimplicialMesh::new(
        2,
        vec![vec![0., 0.], vec![1., 0.], vec![1., 2.]],
        vec![vec![0, 1, 2]],
        MeshQualityGate::new(0.01).unwrap(),
    )
    .unwrap();
    let cell = mesh.geometry_map(MeshEntity::new(2, 0)).unwrap();
    let mut point = vec![0.; form.fields().last().unwrap().range.end];
    point[vector.range.clone()].copy_from_slice(&[0.4, -0.1, 1.2, 0.3, 0.7, -0.2]);
    let previous = BTreeMap::from([(field, vec![0.; 6])]);
    let rule = simplex_duffy_gauss_legendre(2, 5).unwrap();
    let facet_rule = simplex_duffy_gauss_legendre(1, 3).unwrap();
    let facet = (0..mesh.entity_count(1).unwrap())
        .map(|i| MeshEntity::new(1, i))
        .find(|entity| {
            mesh.entity_vertices(*entity)
                .unwrap()
                .iter()
                .all(|v| mesh.vertices()[v.index()][0] == 1.)
        })
        .unwrap();
    let incidence = mesh.incidence(facet, 2).unwrap()[0];
    let vertices = mesh
        .entity_vertices(facet)
        .unwrap()
        .iter()
        .map(|v| v.index())
        .collect::<Vec<_>>();
    let facet_geometry = mesh.geometry_map(facet).unwrap();
    for boundary in [false, true] {
        let evaluate = |point: &[f64]| {
            if boundary {
                form.natural_facet_action(
                    field,
                    &cell,
                    (&facet_geometry, incidence, &vertices),
                    &facet_rule,
                    point,
                    true,
                    |_, _| Ok(vec![0.3, -0.4]),
                )
                .unwrap()
            } else {
                form.prepare_cell(&cell, &rule)
                    .and_then(|cell| cell.linearize(&previous, point))
                    .unwrap()
            }
        };
        let action = evaluate(&point);
        let residual = if boundary {
            let action = form
                .natural_facet_action(
                    field,
                    &cell,
                    (&facet_geometry, incidence, &vertices),
                    &facet_rule,
                    &point,
                    false,
                    |_, _| Ok(vec![0.3, -0.4]),
                )
                .unwrap();
            assert!(action.jacobian.is_empty());
            action.residual
        } else {
            form.prepare_cell(&cell, &rule)
                .unwrap()
                .residual(&previous, &point)
                .unwrap()
        };
        assert_eq!(residual, action.residual);
        let step = 1e-5;
        for column in 0..point.len() {
            let mut plus = point.clone();
            let mut minus = point.clone();
            plus[column] += step;
            minus[column] -= step;
            let a = evaluate(&plus).residual;
            let b = evaluate(&minus).residual;
            for row in 0..point.len() {
                let numerical = (a[row] - b[row]) / (2. * step);
                assert!((numerical - action.jacobian[row * point.len() + column]).abs() < 2e-10);
            }
        }
    }
}

#[test]
fn operator_definition_not_name_controls_nonlinear_admission() {
    let original = bind_unit(&derive(TRANSPORT).unwrap());
    let renamed = bind_unit(
        &derive(
            &TRANSPORT
                .replace("outer_product", "renamed_product")
                .replace("Transport", "Renamed"),
        )
        .unwrap(),
    );
    let rule = simplex_duffy_gauss_legendre(2, 5).unwrap();
    let point = vec![0.4, -0.1, 1.2, 0.3, 0.7, -0.2];
    let evaluate = |form: &BoundRegionForm| {
        form.prepare_cell(&geometry(), &rule)
            .and_then(|cell| {
                cell.linearize(
                    &BTreeMap::from([(form.fields()[0].field, vec![0.; 6])]),
                    &point,
                )
            })
            .unwrap()
    };
    let a = evaluate(&original);
    let b = evaluate(&renamed);
    assert_eq!(a.residual, b.residual);
    assert_eq!(a.jacobian, b.jacobian);
    let invalid = TRANSPORT.replace(
        "component(left, 0) * component(right, 1);",
        "component(left, 0) * component(right, 1) + component(left, 0) * component(right, 1);",
    );
    assert!(derive(&invalid).is_err());
}

#[test]
fn divergence_constraint_retains_half_flux_on_a_physical_stress_boundary() {
    let declaration = TRANSPORT.split_once("model Transport").unwrap().0;
    let source = format!(
        "{declaration}{}",
        MIXED
            .replace("density: kg / m ^ 3 = 3", "density: kg / m ^ 3 = 4")
            .replace(
                "density * derivative(v)",
                "density * derivative(v) + div(density * outer_product(left = v, right = v))"
            )
    );
    let compiled = derive(&source).unwrap();
    assert!(
        compiled
            .rows
            .iter()
            .flat_map(|row| &row.dyadics)
            .all(|term| term.split)
    );
    let form = bind_unit(&compiled);
    let vector = form
        .fields()
        .iter()
        .find(|layout| !layout.value_type.shape().is_scalar())
        .unwrap();
    let mesh = SimplicialMesh::new(
        2,
        vec![vec![0., 0.], vec![1., 0.], vec![1., 2.]],
        vec![vec![0, 1, 2]],
        MeshQualityGate::new(0.01).unwrap(),
    )
    .unwrap();
    let facet = (0..mesh.entity_count(1).unwrap())
        .map(|index| MeshEntity::new(1, index))
        .find(|entity| {
            mesh.entity_vertices(*entity)
                .unwrap()
                .iter()
                .all(|vertex| mesh.vertices()[vertex.index()][0] == 1.)
        })
        .unwrap();
    let incidence = mesh.incidence(facet, 2).unwrap()[0];
    let vertices = mesh
        .entity_vertices(facet)
        .unwrap()
        .iter()
        .map(|v| v.index())
        .collect::<Vec<_>>();
    let mut point = vec![0.; 9];
    for (i, value) in [2., 3., 2., 3., 2., 3.].iter().enumerate() {
        point[vector.range.start + i] = *value;
    }
    let action = form
        .natural_facet_action(
            vector.field,
            &mesh.geometry_map(MeshEntity::new(2, 0)).unwrap(),
            (&mesh.geometry_map(facet).unwrap(), incidence, &vertices),
            &simplex_duffy_gauss_legendre(1, 3).unwrap(),
            &point,
            true,
            |_, _| Ok(vec![0.; 2]),
        )
        .unwrap();
    for node in 0..3 {
        for i in 0..2 {
            close(
                action.residual[vector.range.start + 2 * node + i],
                if node == 0 {
                    0.
                } else if i == 0 {
                    8.
                } else {
                    12.
                },
            );
        }
    }
    // ∫ edge N_a N_b = L/3 on the diagonal and L/6 off it.
    for a in 1..3 {
        for b in 1..3 {
            for i in 0..2 {
                for k in 0..2 {
                    let mass = if a == b { 2. / 3. } else { 1. / 3. };
                    let expected = 2.
                        * mass
                        * (if k == 0 {
                            point[vector.range.start + i]
                        } else {
                            0.
                        } + if i == k { 2. } else { 0. });
                    close(
                        action.jacobian
                            [(vector.range.start + 2 * a + i) * 9 + vector.range.start + 2 * b + k],
                        expected,
                    );
                }
            }
        }
    }
}
