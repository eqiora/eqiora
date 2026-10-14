//! Dimension-independent coordinate admission and fixed-topology action replay.

use eqiora_core::Diagnostic;
use eqiora_meshing::{FixedTopologyGeometryAction, FixedTopologyGeometryState, SimplicialMesh};

use crate::{MeshDecoderLimits, invalid_artifact};

pub(crate) struct GeometryActionReplay {
    pub(crate) velocity: Option<Vec<Vec<f64>>>,
    pub(crate) minimum_path_signed_measure_scale: f64,
}

pub(crate) fn replay_geometry_action<const D: usize>(
    mesh: &SimplicialMesh,
    current: &FixedTopologyGeometryState<D>,
    predecessor: Option<(&[Vec<f64>], f64)>,
) -> Result<GeometryActionReplay, Diagnostic> {
    let previous = match predecessor {
        Some((coordinates, _)) => FixedTopologyGeometryState::<D>::new(mesh, coordinates.to_vec()),
        None => FixedTopologyGeometryState::<D>::reference(mesh),
    }
    .map_err(|error| invalid_artifact(error.message()))?;
    let interval_s = predecessor.map_or(1.0, |(_, interval)| interval);
    let action = FixedTopologyGeometryAction::<D>::new(mesh, &previous, current, interval_s)
        .map_err(|error| invalid_artifact(error.message()))?;
    let velocity = if predecessor.is_some() {
        let mut velocity = action.vertex_velocities().to_vec();
        normalize_geometry_coordinates(&mut velocity)?;
        Some(velocity)
    } else {
        None
    };
    Ok(GeometryActionReplay {
        velocity,
        minimum_path_signed_measure_scale: action.minimum_path_signed_measure_scale(),
    })
}

pub(crate) fn validate_successor(
    step: u64,
    time_s: f64,
    previous_step: u64,
    previous_time_s: f64,
) -> Result<(), Diagnostic> {
    if Some(step) != previous_step.checked_add(1) || time_s <= previous_time_s {
        return Err(invalid_artifact(
            "geometry-state must immediately follow its predecessor in step and time",
        ));
    }
    Ok(())
}

pub(crate) fn validate_coordinate_payload(
    coordinates: &[Vec<f64>],
    velocity: Option<&[Vec<f64>]>,
    dimension: usize,
    limits: MeshDecoderLimits,
) -> Result<(), Diagnostic> {
    let coordinate_values =
        validate_geometry_coordinate_array("geometry-state coordinates", coordinates, limits)?;
    if coordinates[0].len() != dimension {
        return Err(invalid_artifact(
            "geometry-state coordinate dimension differs from its reference",
        ));
    }
    if let Some(velocity) = velocity {
        let velocity_values =
            validate_geometry_coordinate_array("geometry-state mesh velocity", velocity, limits)?;
        if coordinate_values
            .checked_add(velocity_values)
            .is_none_or(|total| total > limits.max_mesh_coordinate_values)
        {
            return Err(invalid_artifact(
                "geometry-state coordinate and velocity aggregate exceeds the decoder limit",
            ));
        }
        if velocity.len() != coordinates.len()
            || velocity
                .iter()
                .zip(coordinates)
                .any(|(left, right)| left.len() != right.len())
        {
            return Err(invalid_artifact(
                "geometry-state mesh velocity shape differs from current coordinates",
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_endpoint_quality(
    mean_ratio: f64,
    signed_measure_scale: f64,
) -> Result<(), Diagnostic> {
    if !mean_ratio.is_finite()
        || mean_ratio <= 0.0
        || mean_ratio > 1.0
        || !signed_measure_scale.is_finite()
        || signed_measure_scale <= 0.0
    {
        return Err(invalid_artifact(
            "geometry-state endpoint quality must be finite and strictly positive, with mean ratio at most one",
        ));
    }
    Ok(())
}

pub(crate) fn validate_path_quality(signed_measure_scale: f64) -> Result<(), Diagnostic> {
    if !signed_measure_scale.is_finite() || signed_measure_scale <= 0.0 {
        return Err(invalid_artifact(
            "geometry-state path quality must be finite and strictly positive",
        ));
    }
    Ok(())
}

pub(crate) fn normalize_geometry_coordinates(values: &mut [Vec<f64>]) -> Result<(), Diagnostic> {
    for value in values.iter_mut().flatten() {
        if !value.is_finite() {
            return Err(invalid_artifact(
                "geometry-state coordinates must be finite",
            ));
        }
        *value = normalize_geometry_zero(*value);
    }
    Ok(())
}

fn validate_geometry_coordinate_array(
    label: &str,
    values: &[Vec<f64>],
    limits: MeshDecoderLimits,
) -> Result<usize, Diagnostic> {
    if values.is_empty() || values.len() > limits.max_mesh_vertices {
        return Err(invalid_artifact(format!(
            "{label} vertex count is empty or exceeds the decoder limit",
        )));
    }
    let dimension = values[0].len();
    let scalar_count = values.iter().try_fold(0_usize, |count, vertex| {
        count
            .checked_add(vertex.len())
            .ok_or_else(|| invalid_artifact(format!("{label} scalar count overflows usize")))
    })?;
    if dimension == 0
        || scalar_count > limits.max_mesh_coordinate_values
        || values.iter().any(|vertex| {
            vertex.len() != dimension
                || vertex
                    .iter()
                    .any(|value| !value.is_finite() || is_negative_geometry_zero(*value))
        })
    {
        return Err(invalid_artifact(format!(
            "{label} must be a bounded rectangular array of finite canonical values",
        )));
    }
    Ok(scalar_count)
}

pub(crate) const fn normalize_geometry_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

pub(crate) fn is_negative_geometry_zero(value: f64) -> bool {
    value == 0.0 && value.is_sign_negative()
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_meshing::MeshQualityGate;

    #[test]
    fn coordinate_budget_counts_both_arrays_in_each_dimension() {
        for dimension in [2, 3] {
            let coordinates = vec![vec![0.0; dimension]; 4];
            let velocity = vec![vec![1.0; dimension]; 4];
            let limits = MeshDecoderLimits {
                max_mesh_coordinate_values: 8 * dimension,
                ..Default::default()
            };
            validate_coordinate_payload(&coordinates, Some(&velocity), dimension, limits).unwrap();
            assert!(
                validate_coordinate_payload(
                    &coordinates,
                    Some(&velocity),
                    dimension,
                    MeshDecoderLimits {
                        max_mesh_coordinate_values: 8 * dimension - 1,
                        ..limits
                    }
                )
                .is_err()
            );
            validate_coordinate_payload(
                &coordinates,
                None,
                dimension,
                MeshDecoderLimits {
                    max_mesh_coordinate_values: 4 * dimension,
                    ..limits
                },
            )
            .unwrap();
            assert!(
                validate_coordinate_payload(&coordinates, Some(&velocity[..3]), dimension, limits)
                    .is_err()
            );
            assert!(
                validate_coordinate_payload(&coordinates, None, dimension + 1, limits).is_err()
            );
            for invalid in [-0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut malformed = coordinates.clone();
                malformed[0][0] = invalid;
                assert!(validate_coordinate_payload(&malformed, None, dimension, limits).is_err());
                assert!(
                    validate_coordinate_payload(&coordinates, Some(&malformed), dimension, limits)
                        .is_err()
                );
            }
        }
        assert!(validate_coordinate_payload(&[], None, 2, Default::default()).is_err());
        assert!(
            validate_coordinate_payload(&[vec![0.0; 2], vec![0.0; 3]], None, 2, Default::default())
                .is_err()
        );
    }

    #[test]
    fn quality_and_successor_boundaries_remain_closed() {
        validate_endpoint_quality(1.0, 1.0).unwrap();
        validate_path_quality(1.0).unwrap();
        for invalid in [-1.0, -0.0, 0.0, f64::NAN, f64::INFINITY] {
            assert!(validate_endpoint_quality(invalid, 1.0).is_err());
            assert!(validate_endpoint_quality(1.0, invalid).is_err());
            assert!(validate_path_quality(invalid).is_err());
        }
        assert!(validate_endpoint_quality(1.01, 1.0).is_err());
        validate_successor(1, 0.25, 0, 0.0).unwrap();
        for (step, time, previous_step, previous_time) in [
            (0, 0.25, u64::MAX, 0.0),
            (2, 0.25, 0, 0.0),
            (1, 0.0, 0, 0.0),
            (1, 0.125, 0, 0.25),
        ] {
            assert!(validate_successor(step, time, previous_step, previous_time).is_err());
        }
    }

    #[test]
    fn translation_replay_derives_velocity_and_preserves_initial_origin() {
        translation::<2>();
        translation::<3>();
    }

    fn translation<const D: usize>() {
        let mut vertices = vec![vec![0.0; D]; D + 1];
        for axis in 0..D {
            vertices[axis + 1][axis] = 1.0;
        }
        let mesh = SimplicialMesh::new(
            D,
            vertices.clone(),
            vec![(0..=D).collect()],
            MeshQualityGate::new(0.05).unwrap(),
        )
        .unwrap();
        let translated = vertices
            .iter()
            .map(|vertex| vertex.iter().map(|x| x + 0.5).collect())
            .collect();
        let current = FixedTopologyGeometryState::<D>::new(&mesh, translated).unwrap();
        let initial = replay_geometry_action(&mesh, &current, None).unwrap();
        assert!(initial.velocity.is_none());
        assert_eq!(initial.minimum_path_signed_measure_scale, 1.0);
        let continuous = replay_geometry_action(&mesh, &current, Some((&vertices, 0.25))).unwrap();
        // Uniform 0.5 m translation over 0.25 s leaves every Jacobian unchanged.
        assert_eq!(continuous.velocity.unwrap(), vec![vec![2.0; D]; D + 1]);
        assert_eq!(continuous.minimum_path_signed_measure_scale, 1.0);
    }
}
