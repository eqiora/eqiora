//! Thin explicit-unit external-source projection of common Mesh import.
use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn import(
    py: Python<'_>,
    geometry: &crate::geometry::PyGeometry,
    data: &Bound<'_, PyBytes>,
    coordinate_unit: &str,
    minimum_mean_ratio: f64,
    maximum_boundary_facets: usize,
    maximum_source_bytes: usize,
) -> PyResult<PyMesh> {
    if coordinate_unit != "m" {
        return Err(request_error(
            py,
            "external MSH import supports only explicit coordinate_unit='m'; no unit conversion is performed",
        ));
    }
    let policy = eqiora::artifact::MshImportPolicyV1::metres(
        minimum_mean_ratio,
        maximum_boundary_facets,
        maximum_source_bytes,
    )
    .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
    if data.as_bytes().len() > maximum_source_bytes {
        return Err(request_error(
            py,
            "external MSH source exceeds the declared byte limit",
        ));
    }
    let owner = AuthenticatedCommonMesh::import_msh41(
        geometry.geometry().clone(),
        policy,
        data.as_bytes().to_vec(),
    )
    .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
    PyMesh::from_authenticated(py, owner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::PyGeometry;
    use eqiora::geometry::NamedEntitySet;

    #[test]
    fn python_import_projection_keeps_tetrahedron_and_replays_native_resources() {
        Python::initialize();
        Python::attach(|py| {
            let geometry = CanonicalGeometryV1::from_convex_polyhedra(
                vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                vec![vec![
                    vec![0, 2, 1],
                    vec![0, 1, 3],
                    vec![0, 3, 2],
                    vec![1, 2, 3],
                ]],
                vec![
                    NamedEntitySet::new("body", 3, vec![0]),
                    NamedEntitySet::new("outer", 2, vec![0, 1, 2, 3]),
                ],
                1e-12,
            )
            .unwrap();
            let source = b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 4 1 4\n3 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n$Elements\n1 1 1 1\n3 1 4 1\n1 1 2 3 4\n$EndElements\n";
            let expected = AuthenticatedCommonMesh::import_msh41(
                geometry.clone(),
                eqiora::artifact::MshImportPolicyV1::metres(0.01, 4, 4096).unwrap(),
                source.to_vec(),
            )
            .unwrap();
            let geometry = PyGeometry::from_geometry(geometry);
            let published =
                import(py, &geometry, &PyBytes::new(py, source), "m", 0.01, 4, 4096).unwrap();
            let owner = published.authenticated_common_mesh().unwrap().unwrap();
            assert_eq!(owner.to_bytes().unwrap(), expected.to_bytes().unwrap());
            assert_eq!(
                owner.simplicial_mesh().unwrap().mesh().cells(),
                [vec![0, 1, 2, 3]]
            );
            let replayed = PyMesh::from_authenticated(
                py,
                AuthenticatedCommonMesh::from_bytes(&owner.to_bytes().unwrap()).unwrap(),
            )
            .unwrap();
            assert_eq!(
                replayed.authenticated_common_mesh().unwrap().unwrap(),
                expected
            );
        });
    }
}
