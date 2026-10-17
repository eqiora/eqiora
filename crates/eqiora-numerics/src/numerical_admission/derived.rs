use super::*;
use crate::field_sampling::{ContinuousP1Sampler, coordinates_2d};

impl CommonTransientFlowPlan {
    /// Derive the exact cell-average scalar curl of this Plan's accepted
    /// two-dimensional MINI velocity field.
    ///
    /// The MINI bubble vanishes on every cell boundary, so its mean curl is
    /// exactly zero by Stokes' theorem. The result is therefore the constant
    /// curl of the affine vertex field on each positively oriented triangle.
    pub fn cell_average_velocity_curl_2d(
        &self,
        state: &CommonState,
    ) -> Result<Box<[f64]>, Diagnostic> {
        if state.state_space_identity() != self.state_space_identity() {
            return Err(invalid(
                "curl source State belongs to a different exact common state space",
            ));
        }
        let CommonStateKind::MiniP1(native) = &state.kind else {
            return Err(invalid(
                "cell-average curl currently requires a two-dimensional MINI velocity State",
            ));
        };
        let mesh = match state.resources.as_ref() {
            NativeMeshResources::AffineTriangleSimplicial { mesh, .. }
            | NativeMeshResources::GmshSimplicial { mesh, .. } => mesh.mesh(),
            _ => {
                return Err(invalid(
                    "cell-average curl currently requires an authenticated simplicial Mesh",
                ));
            }
        };
        if mesh.topological_dimension() != 2 {
            return Err(invalid(
                "cell-average curl currently requires a two-dimensional simplicial Mesh",
            ));
        }
        cell_average_curl_2d(mesh, native.velocity().vertex_values()).map(Vec::into_boxed_slice)
    }

    /// Sample this Plan's continuous P1 pressure at one physical point.
    ///
    /// Cell search is canonical and a point shared by multiple triangles is
    /// admitted only when every continuous-P1 reconstruction agrees within
    /// floating-point evaluation error.
    pub fn sample_pressure_2d(
        &self,
        state: &CommonState,
        point_m: [f64; 2],
    ) -> Result<f64, Diagnostic> {
        if state.state_space_identity() != self.state_space_identity() {
            return Err(invalid(
                "sample source State belongs to a different exact common state space",
            ));
        }
        let CommonStateKind::MiniP1(native) = &state.kind else {
            return Err(invalid(
                "point sampling currently requires a two-dimensional MINI pressure State",
            ));
        };
        let mesh = match state.resources.as_ref() {
            NativeMeshResources::AffineTriangleSimplicial { mesh, .. }
            | NativeMeshResources::GmshSimplicial { mesh, .. } => mesh.mesh(),
            _ => {
                return Err(invalid(
                    "point sampling currently requires an authenticated simplicial Mesh",
                ));
            }
        };
        ContinuousP1Sampler::new(
            mesh,
            &(0..mesh.cells().len()).collect::<Vec<_>>(),
            &(0..mesh.vertices().len()).collect::<Vec<_>>(),
            native.pressure().vertex_values(),
            1,
        )?
        .at(&point_m)
        .map(|values| values[0])
    }
}

fn cell_average_curl_2d(
    mesh: &SimplicialMesh,
    velocity: &[[f64; 2]],
) -> Result<Vec<f64>, Diagnostic> {
    if velocity.len() != mesh.vertices().len() {
        return Err(invalid(
            "curl source velocity differs from its exact Mesh vertex association",
        ));
    }
    let mut values = Vec::with_capacity(mesh.cells().len());
    for cell in mesh.cells() {
        let [a, b, c] = <&[usize; 3]>::try_from(cell.as_slice())
            .map_err(|_| invalid("curl source Mesh contains a non-triangle cell"))?;
        let [x0, y0] = coordinates_2d(&mesh.vertices()[*a])?;
        let [x1, y1] = coordinates_2d(&mesh.vertices()[*b])?;
        let [x2, y2] = coordinates_2d(&mesh.vertices()[*c])?;
        let twice_area = (x1 - x0).mul_add(y2 - y0, -(x2 - x0) * (y1 - y0));
        if !twice_area.is_finite() || twice_area <= 0.0 {
            return Err(invalid(
                "curl source Mesh lost finite positive triangle orientation",
            ));
        }
        let gradients = [
            [(y1 - y2) / twice_area, (x2 - x1) / twice_area],
            [(y2 - y0) / twice_area, (x0 - x2) / twice_area],
            [(y0 - y1) / twice_area, (x1 - x0) / twice_area],
        ];
        let vectors = [velocity[*a], velocity[*b], velocity[*c]];
        let curl = vectors
            .iter()
            .zip(gradients)
            .map(|(value, gradient)| value[1] * gradient[0] - value[0] * gradient[1])
            .sum::<f64>();
        if !curl.is_finite() {
            return Err(invalid("cell-average curl produced a non-finite value"));
        }
        values.push(curl);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eqiora_meshing::MeshQualityGate;

    #[test]
    fn cell_average_curl_is_exact_for_an_affine_velocity_on_arbitrary_triangles() {
        let mesh = SimplicialMesh::new(
            2,
            vec![
                vec![0.0, 0.0],
                vec![2.0, 0.0],
                vec![0.4, 1.3],
                vec![2.0, 1.0],
            ],
            vec![vec![0, 1, 2], vec![1, 3, 2]],
            MeshQualityGate::new(1.0e-6).unwrap(),
        )
        .unwrap();
        let velocity = mesh
            .vertices()
            .iter()
            .map(|point| [-3.0 * point[1] + 0.5, 2.0 * point[0] - 0.25])
            .collect::<Vec<_>>();

        let curl = cell_average_curl_2d(&mesh, &velocity).unwrap();

        assert_eq!(curl, vec![5.0, 5.0]);
    }

    #[test]
    fn cell_average_curl_rejects_a_stale_vertex_association() {
        let mesh = SimplicialMesh::new(
            2,
            vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![0, 1, 2]],
            MeshQualityGate::new(1.0e-6).unwrap(),
        )
        .unwrap();
        assert!(cell_average_curl_2d(&mesh, &[[0.0; 2]; 2]).is_err());
    }
}
