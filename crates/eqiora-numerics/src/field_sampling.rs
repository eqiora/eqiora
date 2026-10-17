//! Physical point reconstruction from exact mesh-associated Field coefficients.

use eqiora_core::{Diagnostic, diagnostic::codes};
use eqiora_meshing::{MeshEntity, MeshGeometry, MeshTopology, SimplicialMesh};

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

/// Prepared continuous-P1 real coordinates shared by observations and nodal transfer.
/// Preparation validates the exact source association once, independent of sample count.
pub(crate) struct ContinuousP1Sampler<'a> {
    mesh: &'a SimplicialMesh,
    cells: &'a [usize],
    owned: &'a [usize],
    coefficients: &'a [f64],
    components: usize,
    space: crate::discrete_space::DiscreteSpace,
    value_scale: f64,
}

impl<'a> ContinuousP1Sampler<'a> {
    pub(crate) fn new(
        mesh: &'a SimplicialMesh,
        cells: &'a [usize],
        owned: &'a [usize],
        coefficients: &'a [f64],
        components: usize,
    ) -> Result<Self, Diagnostic> {
        if components == 0
            || owned.len().checked_mul(components) != Some(coefficients.len())
            || owned.windows(2).any(|pair| pair[0] >= pair[1])
            || owned.iter().any(|index| *index >= mesh.vertices().len())
            || cells.windows(2).any(|pair| pair[0] >= pair[1])
            || cells.iter().any(|index| *index >= mesh.cells().len())
            || coefficients.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                "continuous-P1 sample source differs from its exact Mesh association",
            ));
        }
        let space = crate::discrete_space::DiscreteSpace::new(
            eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
            eqiora_meshing::ReferenceCell::simplex(mesh.topological_dimension())?,
        )?;
        let value_scale = coefficients.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        Ok(Self {
            mesh,
            cells,
            owned,
            coefficients,
            components,
            space,
            value_scale,
        })
    }

    pub(crate) fn at(&self, point: &[f64]) -> Result<Vec<f64>, Diagnostic> {
        self.at_in_cells(point, self.cells)
    }

    /// Restrict location to authenticated parent cells without widening Field support.
    pub(crate) fn at_in_cells(
        &self,
        point: &[f64],
        cells: &[usize],
    ) -> Result<Vec<f64>, Diagnostic> {
        let dimension = self.mesh.topological_dimension();
        if point.len() != dimension || point.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "sample point requires finite exact-dimensional coordinates",
            ));
        }
        if cells
            .iter()
            .any(|cell| self.cells.binary_search(cell).is_err())
        {
            return Err(invalid(
                "sample candidate lies outside the exact Field cell closure",
            ));
        }
        let mesh = self.mesh;
        let owned = self.owned;
        let coefficients = self.coefficients;
        let components = self.components;
        let mut candidates = Vec::new();
        for &cell_index in cells {
            let cell = &mesh.cells()[cell_index];
            let geometry = mesh
                .geometry_map(MeshEntity::new(dimension, cell_index))
                .ok_or_else(|| invalid("point-sample cell has no authenticated affine map"))?;
            let inverse = geometry.inverse_jacobian()?;
            // Roundoff depends on the coordinates used in this cell's affine map,
            // not on unrelated vertices elsewhere in the authenticated Mesh.
            let coordinate_scale = cell
                .iter()
                .flat_map(|vertex| mesh.vertices()[*vertex].iter())
                .chain(point)
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max);
            let inverse_scale = inverse.iter().map(|value| value.abs()).sum::<f64>();
            let tolerance =
                131_072.0 * f64::EPSILON * (1.0 + inverse_scale * (1.0 + coordinate_scale));
            let mut lambda = vec![0.; dimension + 1];
            for (axis, coordinate) in lambda[1..].iter_mut().enumerate() {
                *coordinate = point
                    .iter()
                    .zip(geometry.origin())
                    .enumerate()
                    .map(|(physical, (point, origin))| {
                        inverse[axis * dimension + physical] * (point - origin)
                    })
                    .sum();
            }
            lambda[0] = 1. - lambda[1..].iter().sum::<f64>();
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
            let basis = self.space.tabulate(&lambda[1..])?;
            let sample = interpolate_nodal_components(
                owned,
                coefficients,
                components,
                cell,
                basis.values(),
            )?;
            candidates.push((sample, tolerance));
        }
        let Some((canonical, _)) = candidates.first() else {
            return Err(invalid(
                "sample point lies outside the authenticated Field cell closure",
            ));
        };
        for (candidate, tolerance) in candidates.iter().skip(1) {
            let agreement = 16.0 * tolerance * (1.0 + self.value_scale);
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

    fn sample(
        mesh: &SimplicialMesh,
        cells: &[usize],
        owned: &[usize],
        coefficients: &[f64],
        components: usize,
        point: [f64; 2],
    ) -> Result<Vec<f64>, Diagnostic> {
        ContinuousP1Sampler::new(mesh, cells, owned, coefficients, components)?.at(&point)
    }

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
            sample(&mesh, &[0, 1], &[0, 1, 2, 3], &values, 1, [0.2, 0.3]).unwrap(),
            vec![0.0]
        );
        assert_eq!(
            sample(&mesh, &[0, 1], &[0, 1, 2, 3], &values, 1, [0.5, 0.5]).unwrap(),
            vec![0.0]
        );
        assert!(sample(&mesh, &[0, 1], &[0, 1, 2, 3], &values, 1, [1.1, 0.5]).is_err());
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
        let value = sample(
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
        assert!(sample(&mesh, &[0], &[0, 1, 2, 3], &coefficients, 2, [0.75, 0.75]).is_err());
        assert!(
            ContinuousP1Sampler::new(&mesh, &[0], &[0, 1, 2, 3], &coefficients, 2)
                .unwrap()
                .at_in_cells(&[0.75, 0.75], &[1])
                .is_err()
        );
        // A compact regional inventory does not borrow missing vertex coefficients.
        assert!(sample(&mesh, &[1], &[0, 1, 2], &coefficients[..6], 2, [0.75, 0.75]).is_err());
        assert!(sample(&mesh, &[2], &[0, 1, 2, 3], &coefficients, 2, [0.5, 0.5]).is_err());
        assert!(sample(&mesh, &[0], &[1, 0, 2, 3], &coefficients, 2, [0.5, 0.5]).is_err());
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
            sample(
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
            sample(
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

    #[test]
    fn simplex_sampling_preserves_affine_coordinates_in_each_dimension() {
        // An upper-triangular shear and unequal axis lengths give a non-unit,
        // non-diagonal map. Sample values come from physical affine functions,
        // independently of the barycentric reconstruction.
        for dimension in 1..=3 {
            let origin = vec![0.5; dimension];
            let mut vertices = vec![origin.clone()];
            for axis in 0..dimension {
                let mut vertex = origin.clone();
                vertex[axis] += (axis + 2) as f64;
                if axis > 0 {
                    vertex[axis - 1] += 0.5;
                }
                vertices.push(vertex);
            }
            for permuted in [false, true] {
                let mut cell = (0..=dimension).collect::<Vec<_>>();
                if permuted && dimension >= 2 {
                    cell[..3].rotate_left(1); // positive simplex permutation
                }
                let mesh = SimplicialMesh::new(
                    dimension,
                    vertices.clone(),
                    vec![cell],
                    MeshQualityGate::new(1e-6).unwrap(),
                )
                .unwrap();
                let values = |point: &[f64]| {
                    let linear = point
                        .iter()
                        .enumerate()
                        .map(|(axis, value)| (axis + 1) as f64 * value)
                        .sum::<f64>();
                    [
                        1. + linear,
                        3. - 2. * linear,
                        -2. + 3. * linear,
                        5. + linear,
                    ]
                };
                let coefficients = vertices.iter().flat_map(|p| values(p)).collect::<Vec<_>>();
                let owned = (0..=dimension).collect::<Vec<_>>();
                let sampler =
                    ContinuousP1Sampler::new(&mesh, &[0], &owned, &coefficients, 4).unwrap();
                let centroid = (0..dimension)
                    .map(|axis| {
                        vertices.iter().map(|p| p[axis]).sum::<f64>() / vertices.len() as f64
                    })
                    .collect::<Vec<_>>();
                for point in [&centroid, &vertices[0], &vertices[dimension]] {
                    let expected = values(point);
                    let actual = sampler.at(point).unwrap();
                    for (a, b) in actual.iter().zip(expected) {
                        assert!((a - b).abs() <= 128. * f64::EPSILON * b.abs().max(1.));
                    }
                }
                assert!(sampler.at(&[]).is_err());
                assert!(sampler.at(&vec![f64::NAN; dimension]).is_err());
                assert!(sampler.at(&vec![-1.; dimension]).is_err());
            }
        }
    }
}
