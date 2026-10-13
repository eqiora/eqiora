use super::*;
use crate::simplicial_mini_transient::{
    MiniGeometryDirection, MiniTransientCell, MiniTransientDirection, MiniTransport,
};
use eqiora_meshing::{
    AffineGeometryLinearization, FixedTopologyGeometryAction, FixedTopologyGeometryState,
    MeshQualityGate, SimplicialMesh,
};

fn compare_reference<const D: usize>() {
    let declaration = "public operator outer_product(input left: spatial[1], input right: spatial[1]): spatial[2] = component(left, 0) * component(right, 1);\n";
    let mut source = format!(
        "{declaration}{}",
        MIXED.replace(
            "density * derivative(v)",
            "density * derivative(v) + div(density * outer_product(left = v, right = v))",
        )
    );
    if D == 3 {
        source = source
            .replace("box(0, 1, 0, 1)", "box(0, 1, 0, 1, 0, 1)")
            .replace("vector<m / s, 2>", "vector<m / s, 3>");
    }
    let compiled = derive_dimension(&source, D).unwrap();
    let (fields, rows, time) = inputs(&compiled, true);
    let form = compiled
        .bind(
            ReferenceCell::simplex(D).unwrap(),
            &fields,
            &rows,
            Some(&time),
        )
        .unwrap();
    let velocity = form
        .fields()
        .iter()
        .find(|field| field.components == D)
        .unwrap();
    let mesh = SimplicialMesh::new(
        D,
        (0..=D)
            .map(|node| {
                (0..D)
                    .map(|axis| if node == axis + 1 { 1. } else { 0. })
                    .collect()
            })
            .collect(),
        vec![(0..=D).collect()],
        MeshQualityGate::new(0.01).unwrap(),
    )
    .unwrap();
    let previous_mesh = FixedTopologyGeometryState::<D>::reference(&mesh).unwrap();
    let current_mesh = FixedTopologyGeometryState::<D>::new(
        &mesh,
        mesh.vertices()
            .iter()
            .map(|point| {
                (0..D)
                    .map(|i| {
                        0.01 * (i + 1) as f64
                            + (0..D)
                                .map(|j| (if i == j { 1.02 } else { 0.01 }) * point[j])
                                .sum::<f64>()
                    })
                    .collect()
            })
            .collect(),
    )
    .unwrap();
    let motion =
        FixedTopologyGeometryAction::new(&mesh, &previous_mesh, &current_mesh, 0.25).unwrap();
    let cell = motion.cell(0).unwrap();
    let geometry_direction = AffineGeometryLinearization::new(
        cell.current_map().clone(),
        (0..D).map(|i| 0.01 * (i + 1) as f64).collect(),
        (0..D * D).map(|i| 0.004 * i as f64 - 0.01).collect(),
    )
    .unwrap();
    let current_velocity = (0..D + 2)
        .map(|node| std::array::from_fn(|i| 0.12 * (node + 1) as f64 - 0.07 * i as f64))
        .collect::<Vec<[f64; D]>>();
    let previous_velocity = current_velocity
        .iter()
        .map(|v| std::array::from_fn(|i| 0.6 * v[i] + 0.01))
        .collect::<Vec<_>>();
    let velocity_direction = (0..D + 2)
        .map(|node| std::array::from_fn(|i| 0.03 * node as f64 - 0.02 * i as f64))
        .collect::<Vec<_>>();
    let pressure = (0..D + 1).map(|i| 0.2 * i as f64 - 0.1).collect::<Vec<_>>();
    let pressure_direction = (0..D + 1)
        .map(|i| 0.01 * i as f64 + 0.02)
        .collect::<Vec<_>>();
    let count = form.fields().iter().map(|field| field.range.len()).sum();
    let mut point = vec![0.; count];
    let mut direction = vec![0.; count];
    for field in form.fields() {
        let (values, tangents) = if field.field == velocity.field {
            (
                current_velocity
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>(),
                velocity_direction
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>(),
            )
        } else {
            (pressure.clone(), pressure_direction.clone())
        };
        for ((point, direction), (value, tangent)) in point[field.range.clone()]
            .iter_mut()
            .zip(&mut direction[field.range.clone()])
            .zip(values.iter().zip(&tangents))
        {
            *point = value / field.scale;
            *direction = tangent / field.scale;
        }
    }
    let history = BTreeMap::from([(
        velocity.field,
        previous_velocity.iter().flatten().copied().collect(),
    )]);
    let rule = simplex_duffy_gauss_legendre(D, D + 4).unwrap();
    let (primal, tangent) = form
        .prepare_gcl_cell(
            &[velocity.field],
            &motion,
            0,
            Some(&geometry_direction),
            &rule,
        )
        .unwrap();
    let primal = primal.linearize(&history, &point).unwrap();
    let shape = tangent.unwrap().residual(&history, &point).unwrap();
    // The retained reference uses its original pointwise momentum, continuity,
    // relative convection and GCL expressions; it does not call Region actions.
    let (reference, reference_jvp) = MiniTransientCell::<D> {
        geometry: cell.current_map(),
        transport: MiniTransport::SkewRelativeGcl(cell),
        density: 3.,
        viscosity: 2.,
        time_step: 0.25,
        previous_velocity: &previous_velocity,
        current_velocity: &current_velocity,
        current_pressure: &pressure,
    }
    .evaluate(
        MiniTransientDirection {
            current_velocity: &velocity_direction,
            current_pressure: &pressure_direction,
            current_geometry: MiniGeometryDirection::Endpoint(&geometry_direction),
        },
        &rule,
    )
    .unwrap()
    .into_parts();
    for field in form.fields() {
        let momentum = field.field == velocity.field;
        // The reference continuity is -div(v); authored continuity is +div(v).
        let scale = form
            .physical_dual_scale(field.field)
            .unwrap()
            .value()
            .recip()
            * if momentum { 1. } else { -1. };
        for (local, row) in field.range.clone().enumerate() {
            let reference_row = if momentum { local } else { (D + 2) * D + local };
            close(primal.residual[row], scale * reference[reference_row]);
            let state = primal.jacobian[row * count..(row + 1) * count]
                .iter()
                .zip(&direction)
                .map(|(a, b)| a * b)
                .sum::<f64>();
            close(state + shape[row], scale * reference_jvp[reference_row]);
        }
    }
    let wrong = derive_dimension(
        &source.replace("density * derivative(v)", "2 * density * derivative(v)"),
        D,
    )
    .unwrap();
    let (fields, rows, time) = inputs(&wrong, true);
    let wrong = wrong
        .bind(
            ReferenceCell::simplex(D).unwrap(),
            &fields,
            &rows,
            Some(&time),
        )
        .unwrap();
    let field = wrong
        .fields()
        .iter()
        .find(|field| field.components == D)
        .unwrap()
        .field;
    let error = wrong
        .prepare_gcl_cell(&[field], &motion, 0, None, &rule)
        .unwrap_err();
    assert!(
        error.message().contains("matching storage"),
        "{}",
        error.message()
    );
}

#[test]
fn triangle_ale_region_matches_the_independent_local_reference() {
    compare_reference::<2>();
}

#[test]
fn tetrahedral_ale_region_matches_the_independent_local_reference() {
    compare_reference::<3>();
}
