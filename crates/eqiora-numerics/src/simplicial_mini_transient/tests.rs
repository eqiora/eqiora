use super::*;
use eqiora_meshing::{
    FixedTopologyGeometryAction, FixedTopologyGeometryState, MeshQualityGate, SimplicialMesh,
    simplex_duffy_gauss_legendre,
};

#[test]
fn transport_exactness_is_dimension_and_identity_specific() {
    assert_eq!(
        MiniTransport::<2>::Disabled.required_quadrature_exactness(),
        6
    );
    assert_eq!(
        MiniTransport::<3>::Disabled.required_quadrature_exactness(),
        8
    );
    let mesh = tetrahedron();
    let state = FixedTopologyGeometryState::<3>::reference(&mesh).unwrap();
    let action = FixedTopologyGeometryAction::<3>::new(&mesh, &state, &state, 0.25).unwrap();
    assert_eq!(
        MiniTransport::SkewRelativeGcl(action.cell(0).unwrap()).required_quadrature_exactness(),
        11
    );
}

#[test]
fn scaled_affine_projection_is_the_disabled_relation_in_two_and_three_dimensions() {
    let triangle = AffineGeometryMap::from_simplex_vertices(vec![
        vec![0.1, -0.2],
        vec![1.3, 0.1],
        vec![-0.2, 0.9],
    ])
    .unwrap();
    assert_scaled_affine_identity(
        &triangle,
        &simplex_duffy_gauss_legendre(2, 4).unwrap(),
        &[[0.17, -0.08], [0.11, 0.04], [-0.06, 0.13], [0.07, -0.03]],
        &[[0.12, -0.03], [0.08, 0.09], [-0.02, 0.11], [0.05, -0.04]],
        &[0.2, -0.07, 0.03],
    );

    let tetrahedron = AffineGeometryMap::from_simplex_vertices(vec![
        vec![0.1, -0.1, 0.05],
        vec![1.3, 0.1, -0.05],
        vec![0.2, 1.1, 0.15],
        vec![-0.1, 0.2, 1.2],
    ])
    .unwrap();
    assert_scaled_affine_identity(
        &tetrahedron,
        &simplex_duffy_gauss_legendre(3, 6).unwrap(),
        &[
            [0.17, -0.08, 0.03],
            [0.11, 0.04, -0.02],
            [-0.06, 0.13, 0.05],
            [0.09, -0.02, 0.07],
            [0.07, -0.03, 0.02],
        ],
        &[
            [0.12, -0.03, 0.04],
            [0.08, 0.09, -0.02],
            [-0.02, 0.11, 0.06],
            [0.03, -0.05, 0.08],
            [0.05, -0.04, 0.01],
        ],
        &[0.2, -0.07, 0.03, -0.04],
    );
}

#[test]
fn three_dimensional_ale_rejects_the_current_degree_nine_rule() {
    let mesh = tetrahedron();
    let previous = FixedTopologyGeometryState::<3>::reference(&mesh).unwrap();
    let current = FixedTopologyGeometryState::<3>::new(
        &mesh,
        vec![
            vec![0.01, -0.02, 0.00],
            vec![1.04, 0.01, 0.02],
            vec![0.02, 0.97, 0.01],
            vec![-0.01, 0.02, 1.03],
        ],
    )
    .unwrap();
    let action = FixedTopologyGeometryAction::<3>::new(&mesh, &previous, &current, 0.25).unwrap();
    let cell = action.cell(0).unwrap();
    let geometry_direction = AffineGeometryLinearization::new(
        cell.current_map().clone(),
        vec![0.01, -0.02, 0.03],
        vec![0.02, -0.01, 0.00, 0.01, 0.03, -0.02, -0.01, 0.02, 0.01],
    )
    .unwrap();
    let previous_velocity = vec![[0.1, -0.1, 0.05]; 5];
    let current_velocity = vec![
        [0.12, -0.08, 0.04],
        [0.09, -0.04, 0.03],
        [0.08, -0.05, 0.06],
        [0.11, -0.06, 0.02],
        [0.02, 0.01, -0.01],
    ];
    let velocity_direction = vec![[0.01, -0.02, 0.03]; 5];
    let pressure = [0.1, -0.03, 0.02, -0.04];
    let pressure_direction = [-0.01, 0.02, 0.03, -0.02];
    let error = MiniTransientCell::<3> {
        geometry: cell.current_map(),
        transport: MiniTransport::SkewRelativeGcl(cell),
        density: 1.2,
        viscosity: 0.04,
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
        &simplex_duffy_gauss_legendre(3, 6).unwrap(),
    )
    .unwrap_err();
    assert!(error.message().contains("at least 11"));
}

#[test]
fn three_dimensional_moving_ale_jvp_matches_centered_reassembly() {
    const STEP: f64 = 0.25;
    let mesh = tetrahedron();
    let previous = FixedTopologyGeometryState::<3>::reference(&mesh).unwrap();
    let current_coordinates = vec![
        vec![0.01, -0.02, 0.00],
        vec![1.04, 0.01, 0.02],
        vec![0.02, 0.97, 0.01],
        vec![-0.01, 0.02, 1.03],
    ];
    let coordinate_direction = [
        [0.01, -0.02, 0.03],
        [0.03, -0.01, 0.01],
        [0.00, 0.02, -0.01],
        [-0.02, 0.01, 0.04],
    ];
    let current = FixedTopologyGeometryState::<3>::new(&mesh, current_coordinates.clone()).unwrap();
    let action = FixedTopologyGeometryAction::<3>::new(&mesh, &previous, &current, STEP).unwrap();
    let geometry_direction = tetrahedron_geometry_direction(
        action.cell(0).unwrap().current_map(),
        &coordinate_direction,
    );
    let previous_velocity = vec![[0.1, -0.1, 0.05]; 5];
    let current_velocity = vec![
        [0.12, -0.08, 0.04],
        [0.09, -0.04, 0.03],
        [0.08, -0.05, 0.06],
        [0.11, -0.06, 0.02],
        [0.02, 0.01, -0.01],
    ];
    let velocity_direction = vec![
        [0.01, -0.02, 0.03],
        [-0.02, 0.01, 0.02],
        [0.03, 0.00, -0.01],
        [0.01, 0.02, -0.02],
        [-0.01, 0.03, 0.01],
    ];
    let pressure = [0.1, -0.03, 0.02, -0.04];
    let pressure_direction = [-0.01, 0.02, 0.03, -0.02];
    let quadrature = simplex_duffy_gauss_legendre(3, 7).unwrap();
    let evaluated = MiniTransientCell::<3> {
        geometry: action.cell(0).unwrap().current_map(),
        transport: MiniTransport::SkewRelativeGcl(action.cell(0).unwrap()),
        density: 1.2,
        viscosity: 0.04,
        time_step: STEP,
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
        &quadrature,
    )
    .unwrap();
    let (_, analytic) = evaluated.into_parts();

    let epsilon = f64::EPSILON.cbrt();
    let perturbed = |sign: f64| {
        let coordinates = current_coordinates
            .iter()
            .zip(coordinate_direction)
            .map(|(coordinate, direction)| {
                coordinate
                    .iter()
                    .zip(direction)
                    .map(|(value, direction)| value + sign * epsilon * direction)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let geometry = FixedTopologyGeometryState::<3>::new(&mesh, coordinates).unwrap();
        let action =
            FixedTopologyGeometryAction::<3>::new(&mesh, &previous, &geometry, STEP).unwrap();
        let velocity = current_velocity
            .iter()
            .zip(&velocity_direction)
            .map(|(velocity, direction)| {
                std::array::from_fn(|axis| velocity[axis] + sign * epsilon * direction[axis])
            })
            .collect::<Vec<_>>();
        let pressure = std::array::from_fn::<_, 4, _>(|basis| {
            pressure[basis] + sign * epsilon * pressure_direction[basis]
        });
        let zero_velocity = vec![[0.0; 3]; 5];
        let zero_pressure = [0.0; 4];
        MiniTransientCell::<3> {
            geometry: action.cell(0).unwrap().current_map(),
            transport: MiniTransport::SkewRelativeGcl(action.cell(0).unwrap()),
            density: 1.2,
            viscosity: 0.04,
            time_step: STEP,
            previous_velocity: &previous_velocity,
            current_velocity: &velocity,
            current_pressure: &pressure,
        }
        .evaluate(
            MiniTransientDirection {
                current_velocity: &zero_velocity,
                current_pressure: &zero_pressure,
                current_geometry: MiniGeometryDirection::Zero,
            },
            &quadrature,
        )
        .unwrap()
        .into_parts()
        .0
    };
    let plus = perturbed(1.0);
    let minus = perturbed(-1.0);
    let centered = plus
        .iter()
        .zip(minus)
        .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
        .collect::<Vec<_>>();
    let error = centered
        .iter()
        .zip(&analytic)
        .map(|(centered, analytic)| (centered - analytic).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale = analytic
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    assert!(error < 5.0e-7 * (1.0 + scale), "{error:e} versus {scale:e}");
    assert!(analytic.iter().all(|value| value.is_finite()));
}

#[test]
fn coefficient_shapes_fail_closed_before_evaluation() {
    let mesh = tetrahedron();
    let state = FixedTopologyGeometryState::<3>::reference(&mesh).unwrap();
    let action = FixedTopologyGeometryAction::<3>::new(&mesh, &state, &state, 0.25).unwrap();
    let cell = action.cell(0).unwrap();
    let short_velocity = vec![[0.0; 3]; 4];
    let pressure = [0.0; 4];
    let direction_velocity = vec![[0.0; 3]; 5];
    let error = MiniTransientCell::<3> {
        geometry: cell.current_map(),
        transport: MiniTransport::Disabled,
        density: 1.0,
        viscosity: 1.0,
        time_step: 0.25,
        previous_velocity: &short_velocity,
        current_velocity: &short_velocity,
        current_pressure: &pressure,
    }
    .evaluate(
        MiniTransientDirection {
            current_velocity: &direction_velocity,
            current_pressure: &pressure,
            current_geometry: MiniGeometryDirection::Zero,
        },
        &simplex_duffy_gauss_legendre(3, 6).unwrap(),
    )
    .unwrap_err();
    assert!(error.message().contains("5 velocity"));
}

#[test]
fn disabled_transport_accepts_linear_exactness_while_skew_transport_rejects_it() {
    let mesh = tetrahedron();
    let state = FixedTopologyGeometryState::<3>::reference(&mesh).unwrap();
    let action = FixedTopologyGeometryAction::<3>::new(&mesh, &state, &state, 0.25).unwrap();
    let cell = action.cell(0).unwrap();
    let velocity = vec![[0.1, -0.05, 0.02]; 5];
    let direction = vec![[0.0; 3]; 5];
    let pressure = [0.0; 4];
    let source_rule = simplex_duffy_gauss_legendre(3, 6).unwrap();
    let linear_rule = QuadratureRule::new(
        source_rule.reference_cell(),
        Some(8),
        source_rule.points().to_vec(),
    )
    .unwrap();
    let evaluate = |transport| {
        MiniTransientCell::<3> {
            geometry: cell.current_map(),
            transport,
            density: 1.0,
            viscosity: 0.1,
            time_step: 0.25,
            previous_velocity: &velocity,
            current_velocity: &velocity,
            current_pressure: &pressure,
        }
        .evaluate(
            MiniTransientDirection {
                current_velocity: &direction,
                current_pressure: &pressure,
                current_geometry: MiniGeometryDirection::Zero,
            },
            &linear_rule,
        )
    };

    evaluate(MiniTransport::Disabled).unwrap();
    let error = evaluate(MiniTransport::SkewRelativeGcl(cell)).unwrap_err();
    assert!(error.message().contains("at least 11"));
}

fn tetrahedron() -> SimplicialMesh {
    SimplicialMesh::new(
        3,
        vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
        vec![vec![0, 1, 2, 3]],
        MeshQualityGate::new(0.01).unwrap(),
    )
    .unwrap()
}

fn assert_scaled_affine_identity<const D: usize>(
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
    previous_velocity: &[[f64; D]],
    current_velocity: &[[f64; D]],
    current_pressure: &[f64],
) {
    let density = 1.7;
    let viscosity = 0.23;
    let time_step = 0.17;
    let scales = MiniAffineScales::new(2.3, 4.1, 7.9).unwrap();
    let zero_velocity = vec![[0.0; D]; D + 2];
    let zero_pressure = vec![0.0; D + 1];
    let residual = MiniTransientCell::<D> {
        geometry,
        transport: MiniTransport::Disabled,
        density,
        viscosity,
        time_step,
        previous_velocity,
        current_velocity,
        current_pressure,
    }
    .evaluate(
        MiniTransientDirection {
            current_velocity: &zero_velocity,
            current_pressure: &zero_pressure,
            current_geometry: MiniGeometryDirection::Zero,
        },
        quadrature,
    )
    .unwrap()
    .into_parts()
    .0;
    let (local_size, matrix, rhs) = MiniScaledAffineCell::<D> {
        geometry,
        density,
        viscosity,
        time_step,
        previous_velocity,
        scales,
    }
    .project(quadrature)
    .unwrap()
    .into_parts();
    let mut point = current_velocity
        .iter()
        .flat_map(|value| value.iter().map(|value| value / scales.velocity))
        .collect::<Vec<_>>();
    point.extend(current_pressure.iter().map(|value| value / scales.pressure));
    assert_eq!(point.len(), local_size);
    for row in 0..local_size {
        let affine = matrix[row * local_size..(row + 1) * local_size]
            .iter()
            .zip(&point)
            .map(|(entry, point)| entry * point)
            .sum::<f64>()
            - rhs[row];
        let row_scale = if row < (D + 2) * D {
            scales.velocity
        } else {
            scales.pressure
        };
        let expected = residual[row] * row_scale / scales.power;
        let tolerance = 8_192.0 * f64::EPSILON * affine.abs().max(expected.abs()).max(1.0);
        assert!(
            (affine - expected).abs() <= tolerance,
            "row {row}: {affine:e} versus {expected:e}",
        );
    }
}

fn tetrahedron_geometry_direction(
    map: &AffineGeometryMap,
    vertices: &[[f64; 3]; 4],
) -> AffineGeometryLinearization {
    let mut jacobian = vec![0.0; 9];
    for row in 0..3 {
        for column in 0..3 {
            jacobian[row * 3 + column] = vertices[column + 1][row] - vertices[0][row];
        }
    }
    AffineGeometryLinearization::new(map.clone(), vertices[0].to_vec(), jacobian).unwrap()
}
