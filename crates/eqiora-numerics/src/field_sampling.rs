//! Physical point reconstruction from exact mesh-associated Field coefficients.

use eqiora_core::{Diagnostic, diagnostic::codes};
use eqiora_meshing::{MeshTopology, SimplicialMesh};

/// Reconstruct vertex-major components with an admitted nodal basis.
/// The exact support owner supplies the sorted vertex inventory.
pub(crate) fn interpolate_nodal_components(
    owned: &[usize],
    coefficients: &[f64],
    components: usize,
    vertices: &[usize],
    weights: &[f64],
) -> Result<Vec<f64>, Diagnostic> {
    if components == 0
        || owned.len().checked_mul(components) != Some(coefficients.len())
        || vertices.len() != weights.len()
    {
        return Err(invalid(
            "nodal reconstruction differs from its exact coefficient association",
        ));
    }
    let mut value = vec![0.0; components];
    for (&vertex, &weight) in vertices.iter().zip(weights) {
        let owned_index = owned
            .binary_search(&vertex)
            .map_err(|_| invalid("point lies outside the exact Field cell closure"))?;
        for (component, value) in value.iter_mut().enumerate() {
            *value += weight * coefficients[owned_index * components + component];
        }
    }
    if value.iter().any(|value| !value.is_finite()) {
        return Err(invalid("nodal reconstruction is non-finite"));
    }
    Ok(value)
}

/// Sample real continuous-P1 components on the explicitly owned cell closure.
/// `owned` is the sorted vertex inventory; values are vertex-major components.
pub(crate) fn sample_continuous_p1_2d(
    mesh: &SimplicialMesh,
    cells: &[usize],
    owned: &[usize],
    coefficients: &[f64],
    components: usize,
    point: [f64; 2],
) -> Result<Vec<f64>, Diagnostic> {
    if point.iter().any(|value| !value.is_finite()) {
        return Err(invalid("sample point coordinates must be finite"));
    }
    if mesh.topological_dimension() != 2
        || components == 0
        || owned.len().checked_mul(components) != Some(coefficients.len())
        || owned.windows(2).any(|pair| pair[0] >= pair[1])
        || owned.iter().any(|index| *index >= mesh.vertices().len())
        || cells.windows(2).any(|pair| pair[0] >= pair[1])
        || cells.iter().any(|index| *index >= mesh.cells().len())
        || coefficients.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(
            "continuous-P1 sample source differs from its exact planar Mesh association",
        ));
    }
    let space = crate::discrete_space::DiscreteSpace::new(
        eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
        eqiora_meshing::ReferenceCell::simplex(2)?,
    )?;
    let mut candidates = Vec::new();
    for &cell_index in cells {
        let cell = &mesh.cells()[cell_index];
        let [a, b, c] = <&[usize; 3]>::try_from(cell.as_slice())
            .map_err(|_| invalid("point-sample Mesh contains a non-triangle cell"))?;
        let [x0, y0] = coordinates_2d(&mesh.vertices()[*a])?;
        let [x1, y1] = coordinates_2d(&mesh.vertices()[*b])?;
        let [x2, y2] = coordinates_2d(&mesh.vertices()[*c])?;
        let determinant = (x1 - x0).mul_add(y2 - y0, -(x2 - x0) * (y1 - y0));
        if !determinant.is_finite() || determinant <= 0.0 {
            return Err(invalid(
                "point-sample Mesh lost finite positive triangle orientation",
            ));
        }
        // Roundoff depends on the coordinates used in this cell's affine map,
        // not on unrelated vertices elsewhere in the authenticated Mesh.
        let coordinate_scale = [x0, y0, x1, y1, x2, y2, point[0], point[1]]
            .into_iter()
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        let inverse_scale =
            ((x1 - x0).abs() + (x2 - x0).abs() + (y1 - y0).abs() + (y2 - y0).abs()) / determinant;
        let tolerance = 131_072.0 * f64::EPSILON * (1.0 + inverse_scale * (1.0 + coordinate_scale));
        let dx = point[0] - x0;
        let dy = point[1] - y0;
        let lambda1 = ((y2 - y0) * dx - (x2 - x0) * dy) / determinant;
        let lambda2 = (-(y1 - y0) * dx + (x1 - x0) * dy) / determinant;
        let mut lambda = [1.0 - lambda1 - lambda2, lambda1, lambda2];
        if lambda
            .iter()
            .any(|value| !value.is_finite() || *value < -tolerance || *value > 1.0 + tolerance)
        {
            continue;
        }
        for value in &mut lambda {
            if value.abs() <= tolerance {
                *value = 0.0;
            } else if (1.0 - *value).abs() <= tolerance {
                *value = 1.0;
            }
        }
        let sum = lambda.iter().sum::<f64>();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(invalid(
                "point sampling could not canonicalize barycentric coordinates",
            ));
        }
        for value in &mut lambda {
            *value /= sum;
        }
        let basis = space.tabulate(&lambda[1..])?;
        let sample = interpolate_nodal_components(
            owned,
            coefficients,
            components,
            &[*a, *b, *c],
            basis.values(),
        )?;
        candidates.push((sample, tolerance));
    }
    let Some((canonical, _)) = candidates.first() else {
        return Err(invalid(
            "sample point lies outside the authenticated Field cell closure",
        ));
    };
    let value_scale = coefficients
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    for (candidate, tolerance) in candidates.iter().skip(1) {
        let agreement = 16.0 * tolerance * (1.0 + value_scale);
        if candidate
            .iter()
            .zip(canonical)
            .any(|(value, reference)| (value - reference).abs() > agreement)
        {
            return Err(invalid(
                "continuous-P1 sample is ambiguous across containing cells",
            ));
        }
    }
    Ok(candidates.remove(0).0)
}

pub(crate) fn coordinates_2d(vertex: &[f64]) -> Result<[f64; 2], Diagnostic> {
    <[f64; 2]>::try_from(vertex)
        .map_err(|_| invalid("Field source Mesh contains a non-planar vertex"))
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_meshing::MeshQualityGate;

    #[test]
    fn continuous_p1_sampling_is_canonical_on_shared_support() {
        let mesh = SimplicialMesh::new(
            2,
            vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
            ],
            vec![vec![0, 1, 2], vec![1, 3, 2]],
            MeshQualityGate::new(1.0e-6).unwrap(),
        )
        .unwrap();
        let values = mesh
            .vertices()
            .iter()
            .map(|point| 2.0 * point[0] - 3.0 * point[1] + 0.5)
            .collect::<Vec<_>>();

        assert_eq!(
            sample_continuous_p1_2d(&mesh, &[0, 1], &[0, 1, 2, 3], &values, 1, [0.2, 0.3]).unwrap(),
            vec![0.0]
        );
        assert_eq!(
            sample_continuous_p1_2d(&mesh, &[0, 1], &[0, 1, 2, 3], &values, 1, [0.5, 0.5]).unwrap(),
            vec![0.0]
        );
        assert!(
            sample_continuous_p1_2d(&mesh, &[0, 1], &[0, 1, 2, 3], &values, 1, [1.1, 0.5]).is_err()
        );
    }
    #[test]
    fn point_sampling_respects_field_cells_and_component_order() {
        let mesh = SimplicialMesh::new(
            2,
            vec![vec![0., 0.], vec![1., 0.], vec![0., 1.], vec![1., 1.]],
            vec![vec![0, 1, 2], vec![1, 3, 2]],
            MeshQualityGate::new(1e-6).unwrap(),
        )
        .unwrap();
        // Two affine components: 2*x-3*y+1 and 4*x+y-2.
        let coefficients = [1., -2., 3., 2., -2., -1., 0., 3.];
        let value = sample_continuous_p1_2d(
            &mesh,
            &[0, 1],
            &[0, 1, 2, 3],
            &coefficients,
            2,
            [0.25, 0.125],
        )
        .unwrap();
        assert_eq!(value, vec![1.125, -0.875]);
        // Every vertex exists in the inventory, but the second cell is not owned.
        assert!(
            sample_continuous_p1_2d(&mesh, &[0], &[0, 1, 2, 3], &coefficients, 2, [0.75, 0.75])
                .is_err()
        );
        // A compact regional inventory does not borrow missing vertex coefficients.
        assert!(
            sample_continuous_p1_2d(&mesh, &[1], &[0, 1, 2], &coefficients[..6], 2, [0.75, 0.75])
                .is_err()
        );
        assert!(
            sample_continuous_p1_2d(&mesh, &[2], &[0, 1, 2, 3], &coefficients, 2, [0.5, 0.5])
                .is_err()
        );
        assert!(
            sample_continuous_p1_2d(&mesh, &[0], &[1, 0, 2, 3], &coefficients, 2, [0.5, 0.5])
                .is_err()
        );
    }
    #[test]
    fn unrelated_distant_cells_do_not_widen_point_membership() {
        let mesh = SimplicialMesh::new(
            2,
            vec![
                vec![0., 0.],
                vec![1., 0.],
                vec![0., 1.],
                vec![1e8, 0.],
                vec![1e8 + 1., 0.],
                vec![1e8, 1.],
            ],
            vec![vec![0, 1, 2], vec![3, 4, 5]],
            MeshQualityGate::new(1e-6).unwrap(),
        )
        .unwrap();
        let values = [1., 3., -2., 1., 3., -2.];
        assert_eq!(
            sample_continuous_p1_2d(
                &mesh,
                &[0, 1],
                &[0, 1, 2, 3, 4, 5],
                &values,
                1,
                [0.25, 0.125]
            )
            .unwrap(),
            vec![1.125]
        );
        // x>1 excludes the first triangle, x<1e8 excludes the second.
        assert!(
            sample_continuous_p1_2d(
                &mesh,
                &[0, 1],
                &[0, 1, 2, 3, 4, 5],
                &values,
                1,
                [1.001, 0.0]
            )
            .is_err()
        );
    }
}
