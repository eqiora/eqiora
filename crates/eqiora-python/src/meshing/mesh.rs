//! Immutable accepted Mesh publication and NumPy projections.

use eqiora::Diagnostic;
use eqiora::artifact::{
    CartesianMeshEnvelopeV1, GeometryMeshCorrespondenceEnvelopeV1, MeshProductionLineageEnvelopeV1,
    SimplicialMeshEnvelopeV1,
};
use eqiora::diagnostic::codes;
use eqiora::geometry::{CanonicalGeometryV1, NamedEntitySet};
use eqiora::meshing::{MeshEntity, MeshTopology};
use eqiora_numerics::AuthenticatedCommonMesh;
use numpy::PyArray2;
use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyTuple};

mod coordinates;

use super::plan::{MeshProviderPolicy, PlannedMesh, PyMeshPlan};
use super::request_error;
use crate::error::{diagnostic_error, validation_error};
use crate::geometry::{PyGeometrySelection, digest_to_hex};
use crate::matrix::ReadOnlyMatrix;
use crate::panic_boundary;

/// Immutable source-bound accepted Mesh.
#[pyclass(
    name = "Mesh",
    module = "eqiora._eqiora",
    frozen,
    skip_from_py_object,
    weakref
)]
pub(crate) struct PyMesh {
    source: AcceptedMeshSource,
    lineage: MeshLineage,
    canonical_bytes: Vec<u8>,
    coordinates: ReadOnlyMatrix<f64>,
    cells: ReadOnlyMatrix<u32>,
}

enum AcceptedMeshSource {
    CoordinateFactors {
        owner: Box<AuthenticatedCommonMesh>,
    },
    SourceOwned {
        geometry: Box<CanonicalGeometryV1>,
        mesh: Box<SimplicialMeshEnvelopeV1>,
        correspondence: Box<GeometryMeshCorrespondenceEnvelopeV1>,
        production: Box<MeshProductionLineageEnvelopeV1>,
        provider_observation: SourceOwnedProviderObservation,
    },
    SourceOwnedCartesian {
        geometry: Box<CanonicalGeometryV1>,
        mesh: Box<CartesianMeshEnvelopeV1>,
        correspondence: Box<GeometryMeshCorrespondenceEnvelopeV1>,
        production: Box<MeshProductionLineageEnvelopeV1>,
    },
}

enum SourceOwnedProviderObservation {
    Gmsh4152 { output: Box<[u8]> },
    AffineTriangle,
}

/// Exact authenticated Cartesian resources admitted for downstream native consumers.
pub(crate) struct AuthenticatedCartesianResources<'a> {
    pub(crate) geometry: &'a CanonicalGeometryV1,
    pub(crate) mesh: &'a CartesianMeshEnvelopeV1,
    pub(crate) correspondence: &'a GeometryMeshCorrespondenceEnvelopeV1,
    pub(crate) production: &'a MeshProductionLineageEnvelopeV1,
}

/// Exact replay-authenticated affine-triangle resources for native integration.
pub(crate) struct AuthenticatedAffineTriangleResources<'a> {
    pub(crate) geometry: &'a CanonicalGeometryV1,
    pub(crate) mesh: &'a SimplicialMeshEnvelopeV1,
    pub(crate) correspondence: &'a GeometryMeshCorrespondenceEnvelopeV1,
    pub(crate) production: &'a MeshProductionLineageEnvelopeV1,
}

struct MeshLineage {
    source_digest: String,
    realized_geometry_digest: Option<String>,
    mesh_digest: String,
    correspondence_digest: Option<String>,
    dimension: usize,
    vertex_count: usize,
    cell_count: usize,
}

impl PyMesh {
    pub(crate) fn exact_mesh_digest(&self) -> &str {
        &self.lineage.mesh_digest
    }

    pub(crate) fn coordinate_array(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        self.coordinates.numpy(py)
    }

    pub(crate) fn cell_array(&self, py: Python<'_>) -> PyResult<Py<PyArray2<u32>>> {
        self.cells.numpy(py)
    }

    pub(crate) fn source_digest_value(&self) -> &str {
        &self.lineage.source_digest
    }

    pub(crate) fn correspondence_digest_value(&self) -> Option<&str> {
        self.lineage.correspondence_digest.as_deref()
    }
    pub(crate) fn from_authenticated(
        py: Python<'_>,
        owner: AuthenticatedCommonMesh,
    ) -> PyResult<Self> {
        if owner.geometry().is_none() {
            return Self::from_coordinate_factors(py, owner);
        }
        let geometry = owner.geometry().ok_or_else(|| {
            PyRuntimeError::new_err("coordinate-factor grids require coordinate Field admission")
        })?;
        let correspondence = owner.correspondence().ok_or_else(|| {
            PyRuntimeError::new_err("physical Mesh omitted Geometry correspondence")
        })?;
        let production = owner
            .production()
            .ok_or_else(|| PyRuntimeError::new_err("physical Mesh omitted production lineage"))?;
        if let Some(mesh) = owner.cartesian_mesh() {
            return Self::from_source_owned_cartesian(
                py,
                geometry,
                mesh,
                correspondence,
                production,
            );
        }
        let mesh = owner.simplicial_mesh().ok_or_else(|| {
            PyRuntimeError::new_err("authenticated Mesh omitted both native Mesh roots")
        })?;
        if let Some(provider_output) = owner.gmsh_provider_output() {
            Self::from_source_parts(
                py,
                geometry,
                mesh,
                correspondence,
                production,
                SourceOwnedProviderObservation::Gmsh4152 {
                    output: provider_output.to_vec().into_boxed_slice(),
                },
            )
        } else {
            Self::from_source_owned_affine_triangle(py, geometry, mesh, correspondence, production)
        }
    }
}

#[pymethods]
impl PyMesh {
    /// Build a grid on exact dimensioned coordinate factors retained by this Model.
    #[staticmethod]
    fn coordinate_factors(
        py: Python<'_>,
        model: &crate::model::PyModel,
        domain: &crate::model::PyModelDomainRef,
        cells_per_factor: Vec<usize>,
    ) -> PyResult<Self> {
        Self::from_coordinate_domain(py, model, domain, cells_per_factor)
    }

    /// Exact Geometry identity retained by the source binding.
    #[getter]
    fn source_digest(&self) -> &str {
        &self.lineage.source_digest
    }

    /// Identity of the realized straight-edged geometry artifact.
    #[getter]
    fn realized_geometry_digest(&self) -> Option<&str> {
        self.lineage.realized_geometry_digest.as_deref()
    }

    /// Identity of the accepted common mesh artifact.
    #[getter]
    fn digest(&self) -> &str {
        &self.lineage.mesh_digest
    }

    /// Identity of the exact Geometry-to-Mesh correspondence artifact.
    #[getter]
    fn correspondence_digest(&self) -> Option<&str> {
        self.lineage.correspondence_digest.as_deref()
    }

    /// Canonical provider occurrence that produced this common Mesh.
    #[getter]
    fn production_lineage_bytes(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let production = self.production_lineage().ok_or_else(|| {
            capability_error(py, "this Mesh has no common production-lineage artifact")
        })?;
        production
            .canonical_json()
            .map(|bytes| PyBytes::new(py, &bytes).unbind())
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }

    /// Identity of the canonical provider occurrence.
    #[getter]
    fn production_lineage_digest(&self, py: Python<'_>) -> PyResult<String> {
        let production = self.production_lineage().ok_or_else(|| {
            capability_error(py, "this Mesh has no common production-lineage artifact")
        })?;
        production
            .digest()
            .map(|digest| digest.to_string())
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }

    /// Canonical bytes of the accepted common Mesh artifact.
    #[getter]
    fn canonical_bytes(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, &self.canonical_bytes).unbind()
    }

    /// Canonical self-contained bytes of this authenticated Mesh occurrence.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.authenticated_common_mesh()
            .and_then(|owner| {
                owner
                    .ok_or_else(|| {
                        Diagnostic::error(
                            codes::INVALID_ARTIFACT,
                            "Mesh is not an authenticated common Mesh occurrence",
                        )
                    })?
                    .to_bytes()
            })
            .map(|bytes| PyBytes::new(py, &bytes))
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))
    }

    /// Decode and reauthenticate one self-contained common Mesh occurrence.
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        AuthenticatedCommonMesh::from_bytes(data)
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))
            .and_then(|owner| Self::from_authenticated(py, owner))
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.lineage.dimension
    }

    #[getter]
    fn vertex_count(&self) -> usize {
        self.lineage.vertex_count
    }

    #[getter]
    fn cell_count(&self) -> usize {
        self.lineage.cell_count
    }

    /// Canonically ordered, read-only coordinates in coherent SI units.
    #[getter]
    fn coordinates(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        self.coordinates.numpy(py)
    }

    /// Canonically ordered, read-only top-cell connectivity.
    #[getter]
    fn cells(&self, py: Python<'_>) -> PyResult<Py<PyArray2<u32>>> {
        self.cells.numpy(py)
    }

    /// Minimum mean ratio measured over every accepted cell.
    #[getter]
    fn minimum_mean_ratio(&self, py: Python<'_>) -> PyResult<f64> {
        match &self.source {
            AcceptedMeshSource::SourceOwned { mesh, .. } => {
                Ok(mesh.mesh().quality_report().minimum_mean_ratio())
            }
            AcceptedMeshSource::CoordinateFactors { .. }
            | AcceptedMeshSource::SourceOwnedCartesian { .. } => Err(capability_error(
                py,
                "minimum_mean_ratio is not defined for this Cartesian Mesh",
            )),
        }
    }

    #[getter]
    fn selection_names(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let names = match &self.source {
            AcceptedMeshSource::CoordinateFactors { .. } => Vec::new(),
            AcceptedMeshSource::SourceOwned { geometry, .. } => geometry
                .entity_sets()
                .iter()
                .map(NamedEntitySet::name)
                .collect::<Vec<_>>(),
            AcceptedMeshSource::SourceOwnedCartesian { geometry, .. } => geometry
                .entity_sets()
                .iter()
                .map(NamedEntitySet::name)
                .collect::<Vec<_>>(),
        };
        Ok(PyTuple::new(py, names)?.unbind())
    }

    /// Count mesh entities proven to realize one exact-source selection.
    fn selection_entity_count(&self, py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<usize> {
        let selection;
        let (name, expected_dimension) = if let Ok(name) = name.extract::<&str>() {
            (name, None)
        } else if let Ok(value) = name.extract::<PyRef<'_, PyGeometrySelection>>() {
            selection = value;
            if selection.bound_source_digest() != self.lineage.source_digest {
                let diagnostic = Diagnostic::error(
                    codes::INVALID_ARTIFACT,
                    "GeometrySelection belongs to a foreign or stale Geometry revision",
                );
                return Err(validation_error(py, std::slice::from_ref(&diagnostic)));
            }
            (
                selection.canonical_name(),
                Some(selection.canonical_dimension()),
            )
        } else {
            return Err(PyTypeError::new_err(
                "name must be a str or GeometrySelection",
            ));
        };
        match &self.source {
            AcceptedMeshSource::CoordinateFactors { .. } => Err(capability_error(
                py,
                "coordinate-factor grids have no Geometry selections",
            )),
            AcceptedMeshSource::SourceOwned {
                geometry,
                correspondence,
                provider_observation: SourceOwnedProviderObservation::AffineTriangle,
                ..
            } => (if geometry.planar_rectangle_bounds().is_some() {
                correspondence.planar_rectangle_v2_entity_set_entities(geometry, name)
            } else {
                correspondence.adjacent_rectangle_partition_entity_set_entities(geometry, name)
            })
            .and_then(|entities| validated_entity_count(entities, expected_dimension))
            .map_err(|diagnostic| validation_error(py, &[diagnostic])),
            AcceptedMeshSource::SourceOwned {
                geometry,
                correspondence,
                ..
            } => correspondence
                .planar_circular_hole_v2_entity_set_entities(geometry, name)
                .and_then(|entities| validated_entity_count(entities, expected_dimension))
                .map_err(|diagnostic| validation_error(py, std::slice::from_ref(&diagnostic))),
            AcceptedMeshSource::SourceOwnedCartesian {
                geometry,
                correspondence,
                ..
            } => (if geometry.cartesian_box_bounds().is_some() {
                correspondence.cartesian_box_v1_entity_set_entities(geometry, name)
            } else {
                correspondence.planar_rectangle_v2_entity_set_entities(geometry, name)
            })
            .and_then(|entities| validated_entity_count(entities, expected_dimension))
            .map_err(|diagnostic| validation_error(py, &[diagnostic])),
        }
    }

    fn __repr__(&self, _py: Python<'_>) -> PyResult<String> {
        Ok(self.representation())
    }
}

impl PyMesh {
    fn from_source_owned_affine_triangle(
        py: Python<'_>,
        source: &CanonicalGeometryV1,
        accepted_mesh: &SimplicialMeshEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
        production: &MeshProductionLineageEnvelopeV1,
    ) -> PyResult<Self> {
        let policy = production.affine_triangle_cells().ok_or_else(|| {
            request_error(
                py,
                "affine-triangle MeshPlan has a non-affine-triangle production policy",
            )
        })?;
        (if source.planar_rectangle_bounds().is_some() {
            correspondence.validate_against_planar_rectangle_v2_affine_triangles(
                source,
                accepted_mesh,
                policy.cells(),
            )
        } else {
            correspondence.validate_against_adjacent_rectangle_partition_affine_triangles(
                source,
                accepted_mesh,
                policy.cells(),
            )
        })
        .and_then(|()| {
            production.validate_against_affine_triangle_rectangle_v1_resources(
                policy,
                source,
                accepted_mesh,
                correspondence,
            )
        })
        .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        let published = Self::from_source_parts(
            py,
            source,
            accepted_mesh,
            correspondence,
            production,
            SourceOwnedProviderObservation::AffineTriangle,
        )?;
        let authenticated = published
            .authenticated_affine_triangle_resources()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?
            .ok_or_else(|| {
                PyRuntimeError::new_err("affine-triangle publication lost its exact resources")
            })?;
        let _ = (
            authenticated.geometry,
            authenticated.mesh,
            authenticated.correspondence,
            authenticated.production,
        );
        Ok(published)
    }

    fn from_source_parts(
        py: Python<'_>,
        source: &CanonicalGeometryV1,
        accepted_mesh: &SimplicialMeshEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
        production: &MeshProductionLineageEnvelopeV1,
        provider_observation: SourceOwnedProviderObservation,
    ) -> PyResult<Self> {
        let dimension = accepted_mesh.dimension();
        let mesh = accepted_mesh.mesh();
        let vertex_count = mesh.vertices().len();
        let cell_count = mesh.cells().len();
        let (coordinates, cells) = project_simplicial_mesh(py, mesh, dimension)?;
        let source_digest = digest_to_hex(&source.digest_bytes());
        let mesh_digest = accepted_mesh
            .digest()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?
            .to_string();
        let correspondence_digest = correspondence
            .digest()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?
            .to_string();
        let canonical_bytes = accepted_mesh
            .canonical_json()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        Ok(Self {
            source: AcceptedMeshSource::SourceOwned {
                geometry: Box::new(source.clone()),
                mesh: Box::new(accepted_mesh.clone()),
                correspondence: Box::new(correspondence.clone()),
                production: Box::new(production.clone()),
                provider_observation,
            },
            lineage: MeshLineage {
                source_digest: source_digest.clone(),
                realized_geometry_digest: Some(source_digest),
                mesh_digest,
                correspondence_digest: Some(correspondence_digest),
                dimension,
                vertex_count,
                cell_count,
            },
            canonical_bytes,
            coordinates,
            cells,
        })
    }

    fn from_source_owned_cartesian(
        py: Python<'_>,
        source: &CanonicalGeometryV1,
        accepted_mesh: &CartesianMeshEnvelopeV1,
        correspondence: &GeometryMeshCorrespondenceEnvelopeV1,
        production: &MeshProductionLineageEnvelopeV1,
    ) -> PyResult<Self> {
        let policy = production.cartesian_cells().ok_or_else(|| {
            request_error(
                py,
                "Cartesian MeshPlan has a non-Cartesian production policy",
            )
        })?;
        (if source.cartesian_box_bounds().is_some() {
            correspondence.validate_against_cartesian_box_v1(source, accepted_mesh, policy.cells())
        } else {
            let cells: [usize; 2] = policy.cells().try_into().map_err(|_| {
                request_error(py, "planar rectangle Cartesian policy must have two axes")
            })?;
            correspondence.validate_against_planar_rectangle_v2_cartesian(
                source,
                accepted_mesh,
                cells,
            )
        })
        .and_then(|()| {
            production.validate_against_structured_cartesian_v2_resources(
                &policy,
                source,
                accepted_mesh,
                correspondence,
            )
        })
        .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        let dimension = accepted_mesh.dimension();
        let native = accepted_mesh.mesh();
        let vertex_count = native
            .entity_count(0)
            .ok_or_else(|| PyRuntimeError::new_err("Cartesian Mesh omitted its vertices"))?;
        let cell_count = native.entity_count(dimension).ok_or_else(|| {
            PyRuntimeError::new_err("Cartesian Mesh omitted its top-dimensional cells")
        })?;
        let (coordinates, cells) =
            project_cartesian_mesh(py, native, dimension, vertex_count, cell_count)?;
        let source_digest = digest_to_hex(&source.digest_bytes());
        let mesh_digest = accepted_mesh
            .digest()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?
            .to_string();
        let correspondence_digest = correspondence
            .digest()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?
            .to_string();
        let canonical_bytes = accepted_mesh
            .canonical_json()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
        let published = Self {
            source: AcceptedMeshSource::SourceOwnedCartesian {
                geometry: Box::new(source.clone()),
                mesh: Box::new(accepted_mesh.clone()),
                correspondence: Box::new(correspondence.clone()),
                production: Box::new(production.clone()),
            },
            lineage: MeshLineage {
                source_digest: source_digest.clone(),
                realized_geometry_digest: Some(source_digest),
                mesh_digest,
                correspondence_digest: Some(correspondence_digest),
                dimension,
                vertex_count,
                cell_count,
            },
            canonical_bytes,
            coordinates,
            cells,
        };
        let authenticated = published
            .authenticated_cartesian_resources()
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?
            .ok_or_else(|| PyRuntimeError::new_err("Cartesian publication lost its resources"))?;
        let _ = (
            authenticated.geometry,
            authenticated.mesh,
            authenticated.correspondence,
            authenticated.production,
        );
        Ok(published)
    }

    fn representation(&self) -> String {
        format!(
            "Mesh(dimension={}, vertices={}, cells={}, digest={:?})",
            self.dimension(),
            self.vertex_count(),
            self.cell_count(),
            self.digest(),
        )
    }

    fn production_lineage(&self) -> Option<&MeshProductionLineageEnvelopeV1> {
        match &self.source {
            AcceptedMeshSource::CoordinateFactors { .. } => None,
            AcceptedMeshSource::SourceOwned { production, .. } => Some(production),
            AcceptedMeshSource::SourceOwnedCartesian { production, .. } => Some(production),
        }
    }

    fn gmsh_provider_output(&self) -> Option<&[u8]> {
        match &self.source {
            AcceptedMeshSource::SourceOwned {
                provider_observation: SourceOwnedProviderObservation::Gmsh4152 { output },
                ..
            } => Some(output),
            _ => None,
        }
    }

    /// Return only exact replay-authenticated Cartesian resources.
    pub(crate) fn authenticated_cartesian_resources(
        &self,
    ) -> Result<Option<AuthenticatedCartesianResources<'_>>, Diagnostic> {
        let AcceptedMeshSource::SourceOwnedCartesian {
            geometry,
            mesh,
            correspondence,
            production,
        } = &self.source
        else {
            return Ok(None);
        };
        let policy = production.cartesian_cells().ok_or_else(|| {
            Diagnostic::error(
                codes::INVALID_ARTIFACT,
                "Cartesian Mesh has a non-Cartesian production policy",
            )
        })?;
        if geometry.cartesian_box_bounds().is_some() {
            correspondence.validate_against_cartesian_box_v1(geometry, mesh, policy.cells())?;
        } else {
            let cells: [usize; 2] = policy.cells().try_into().map_err(|_| {
                Diagnostic::error(
                    codes::INVALID_ARTIFACT,
                    "planar rectangle Cartesian policy must have two axes",
                )
            })?;
            correspondence.validate_against_planar_rectangle_v2_cartesian(geometry, mesh, cells)?;
        }
        production.validate_against_structured_cartesian_v2_resources(
            &policy,
            geometry,
            mesh,
            correspondence,
        )?;
        Ok(Some(AuthenticatedCartesianResources {
            geometry,
            mesh,
            correspondence,
            production,
        }))
    }

    /// Return only exact replay-authenticated affine-triangle resources.
    pub(crate) fn authenticated_affine_triangle_resources(
        &self,
    ) -> Result<Option<AuthenticatedAffineTriangleResources<'_>>, Diagnostic> {
        let AcceptedMeshSource::SourceOwned {
            geometry,
            mesh,
            correspondence,
            production,
            provider_observation: SourceOwnedProviderObservation::AffineTriangle,
        } = &self.source
        else {
            return Ok(None);
        };
        let policy = production.affine_triangle_cells().ok_or_else(|| {
            Diagnostic::error(
                codes::INVALID_ARTIFACT,
                "affine-triangle Mesh has a non-affine-triangle production policy",
            )
        })?;
        if geometry.planar_rectangle_bounds().is_some() {
            correspondence.validate_against_planar_rectangle_v2_affine_triangles(
                geometry,
                mesh,
                policy.cells(),
            )?;
        } else {
            correspondence.validate_against_adjacent_rectangle_partition_affine_triangles(
                geometry,
                mesh,
                policy.cells(),
            )?;
        }
        production.validate_against_affine_triangle_rectangle_v1_resources(
            policy,
            geometry,
            mesh,
            correspondence,
        )?;
        Ok(Some(AuthenticatedAffineTriangleResources {
            geometry,
            mesh,
            correspondence,
            production,
        }))
    }

    /// Reauthenticate this exact published occurrence for the common resolver.
    pub(crate) fn authenticated_common_mesh(
        &self,
    ) -> Result<Option<AuthenticatedCommonMesh>, Diagnostic> {
        match &self.source {
            AcceptedMeshSource::CoordinateFactors { owner } => Ok(Some((**owner).clone())),
            AcceptedMeshSource::SourceOwnedCartesian {
                geometry,
                mesh,
                correspondence,
                production,
            } => AuthenticatedCommonMesh::structured_cartesian(
                (**geometry).clone(),
                (**mesh).clone(),
                (**correspondence).clone(),
                (**production).clone(),
            )
            .map(Some),
            AcceptedMeshSource::SourceOwned {
                geometry,
                production,
                provider_observation: SourceOwnedProviderObservation::Gmsh4152 { output },
                ..
            } => {
                let policy = production.gmsh_mesh_policy().ok_or_else(|| {
                    Diagnostic::error(
                        codes::INVALID_ARTIFACT,
                        "Gmsh common Mesh has a non-Gmsh production policy",
                    )
                })?;
                AuthenticatedCommonMesh::gmsh_4152((**geometry).clone(), policy, output.to_vec())
                    .map(Some)
            }
            AcceptedMeshSource::SourceOwned {
                geometry,
                mesh,
                correspondence,
                production,
                provider_observation: SourceOwnedProviderObservation::AffineTriangle,
            } => {
                if geometry.planar_rectangle_bounds().is_some() {
                    AuthenticatedCommonMesh::affine_triangle_rectangle(
                        (**geometry).clone(),
                        (**mesh).clone(),
                        (**correspondence).clone(),
                        (**production).clone(),
                    )
                    .map(Some)
                } else if geometry.planar_adjacent_rectangle_partition().is_some() {
                    AuthenticatedCommonMesh::adjacent_partition(
                        (**geometry).clone(),
                        (**mesh).clone(),
                        (**correspondence).clone(),
                        (**production).clone(),
                    )
                    .map(Some)
                } else {
                    // The dependent execution slice adds the corresponding
                    // physics-independent common-admission variant.
                    Ok(None)
                }
            }
        }
    }
}

fn project_simplicial_mesh(
    py: Python<'_>,
    mesh: &eqiora::meshing::SimplicialMesh,
    dimension: usize,
) -> PyResult<(ReadOnlyMatrix<f64>, ReadOnlyMatrix<u32>)> {
    let vertex_count = mesh.vertices().len();
    let mut coordinates = Vec::with_capacity(vertex_count * dimension);
    for coordinate in mesh.vertices() {
        if coordinate.len() != dimension {
            return Err(request_error(
                py,
                "accepted Mesh coordinate dimension is inconsistent",
            ));
        }
        coordinates.extend(coordinate);
    }
    let cell_count = mesh.cells().len();
    let cell_width = mesh.cells().first().map_or(0, Vec::len);
    let mut cells = Vec::with_capacity(cell_count * cell_width);
    for cell in mesh.cells() {
        if cell.len() != cell_width {
            return Err(request_error(
                py,
                "accepted Mesh cell arity is inconsistent",
            ));
        }
        cells.extend(
            cell.iter()
                .map(|&vertex| mesh_index(vertex))
                .collect::<PyResult<Vec<_>>>()?,
        );
    }
    Ok((
        ReadOnlyMatrix::new(vertex_count, dimension, coordinates),
        ReadOnlyMatrix::new(cell_count, cell_width, cells),
    ))
}

fn project_cartesian_mesh(
    py: Python<'_>,
    mesh: &eqiora::meshing::CartesianMesh,
    dimension: usize,
    vertex_count: usize,
    cell_count: usize,
) -> PyResult<(ReadOnlyMatrix<f64>, ReadOnlyMatrix<u32>)> {
    let mut coordinates = Vec::with_capacity(vertex_count * dimension);
    for index in 0..vertex_count {
        let coordinate = mesh
            .vertex_coordinates(MeshEntity::new(0, index))
            .ok_or_else(|| request_error(py, "Cartesian Mesh omitted a vertex coordinate"))?;
        if coordinate.len() != dimension {
            return Err(request_error(
                py,
                "Cartesian Mesh coordinate dimension is inconsistent",
            ));
        }
        coordinates.extend(coordinate);
    }
    let cell_width = 1_usize
        .checked_shl(
            u32::try_from(dimension)
                .map_err(|_| PyOverflowError::new_err("Mesh dimension exceeds uint32"))?,
        )
        .ok_or_else(|| PyOverflowError::new_err("Mesh cell arity exceeds local usize"))?;
    let mut cells = Vec::with_capacity(cell_count * cell_width);
    for index in 0..cell_count {
        let vertices = mesh
            .entity_vertices(MeshEntity::new(dimension, index))
            .ok_or_else(|| request_error(py, "Cartesian Mesh omitted cell connectivity"))?;
        if vertices.len() != cell_width {
            return Err(request_error(
                py,
                "Cartesian Mesh cell arity is inconsistent",
            ));
        }
        cells.extend(
            vertices
                .iter()
                .map(|vertex| mesh_index(vertex.index()))
                .collect::<PyResult<Vec<_>>>()?,
        );
    }
    Ok((
        ReadOnlyMatrix::new(vertex_count, dimension, coordinates),
        ReadOnlyMatrix::new(cell_count, cell_width, cells),
    ))
}

fn mesh_index(index: usize) -> PyResult<u32> {
    u32::try_from(index)
        .map_err(|_| PyOverflowError::new_err("Mesh vertex index exceeds Python uint32"))
}

fn capability_error(py: Python<'_>, message: &str) -> PyErr {
    diagnostic_error(py, &[Diagnostic::error(codes::NOT_IMPLEMENTED, message)])
}

/// Execute one resolved provider plan over its owned Geometry and publish only its accepted Mesh.
#[pyfunction]
#[pyo3(signature = (plan, /))]
pub(super) fn generate(py: Python<'_>, plan: &PyMeshPlan) -> PyResult<PyMesh> {
    panic_boundary(py, || match (&plan.planned, &plan.provider) {
        (PlannedMesh::Gmsh(sizing), MeshProviderPolicy::Gmsh(provider)) => {
            let quality_gate = eqiora::meshing::MeshQualityGate::new(provider.minimum_mean_ratio)
                .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
            let generated = super::gmsh::generate(
                &plan.source,
                provider.maximum_boundary_error,
                provider.minimum_mean_ratio,
                provider.maximum_boundary_facets,
                provider.maximum_target_size,
                *sizing,
                quality_gate,
            )
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
            let production = MeshProductionLineageEnvelopeV1::from_gmsh_4152_resources(
                sizing.policy(),
                &plan.source,
                &generated.mesh,
                &generated.correspondence,
            )
            .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
            let published = PyMesh::from_source_parts(
                py,
                &plan.source,
                &generated.mesh,
                &generated.correspondence,
                &production,
                SourceOwnedProviderObservation::Gmsh4152 {
                    output: generated.provider_output.clone().into_boxed_slice(),
                },
            )?;
            if published.gmsh_provider_output() != Some(generated.provider_output.as_slice()) {
                return Err(PyRuntimeError::new_err(
                    "published Gmsh Mesh lost its exact provider observation",
                ));
            }
            Ok(published)
        }
        (PlannedMesh::Cartesian, MeshProviderPolicy::Cartesian(provider)) => {
            let generated = if plan.source.cartesian_box_bounds().is_some() {
                GeometryMeshCorrespondenceEnvelopeV1::from_cartesian_box_v1(
                    &plan.source,
                    provider.policy.cells(),
                )
            } else {
                let cells: [usize; 2] = provider.policy.cells().try_into().map_err(|_| {
                    request_error(py, "planar rectangle Cartesian policy must have two axes")
                })?;
                GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_cartesian(
                    &plan.source,
                    cells,
                )
            };
            let (mesh, correspondence) =
                generated.map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
            let production =
                MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
                    &provider.policy,
                    &plan.source,
                    &mesh,
                    &correspondence,
                )
                .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
            PyMesh::from_source_owned_cartesian(
                py,
                &plan.source,
                &mesh,
                &correspondence,
                &production,
            )
        }
        (PlannedMesh::AffineTriangle, MeshProviderPolicy::AffineTriangle(provider)) => {
            let generated = if plan.source.planar_rectangle_bounds().is_some() {
                GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_affine_triangles(
                    &plan.source,
                    provider.policy.cells(),
                )
            } else {
                GeometryMeshCorrespondenceEnvelopeV1::from_adjacent_rectangle_partition_affine_triangles(
                    &plan.source,
                    provider.policy.cells(),
                )
            };
            let (mesh, correspondence) =
                generated.map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
            let production =
                MeshProductionLineageEnvelopeV1::from_affine_triangle_rectangle_v1_resources(
                    provider.policy,
                    &plan.source,
                    &mesh,
                    &correspondence,
                )
                .map_err(|diagnostic| validation_error(py, &[diagnostic]))?;
            PyMesh::from_source_owned_affine_triangle(
                py,
                &plan.source,
                &mesh,
                &correspondence,
                &production,
            )
        }
        _ => unreachable!("planned mesh and provider remain paired"),
    })
}

fn validated_entity_count(
    entities: Vec<MeshEntity>,
    expected_dimension: Option<usize>,
) -> Result<usize, Diagnostic> {
    if expected_dimension.is_some_and(|dimension| {
        entities
            .iter()
            .any(|entity| entity.dimension() != dimension)
    }) {
        return Err(Diagnostic::error(
            codes::INVALID_ARTIFACT,
            "GeometrySelection dimension differs from correspondence-owned membership",
        ));
    }
    Ok(entities.len())
}

#[cfg(test)]
#[path = "mesh/tests.rs"]
mod tests;

#[path = "mesh/viewer.rs"]
mod viewer;
