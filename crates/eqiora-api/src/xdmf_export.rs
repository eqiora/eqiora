//! Shared typed mesh/Field projection into XDMF and primitive HDF5 arrays.
use eqiora_artifact::ArtifactDigest;
use eqiora_core::{Diagnostic, DimExponents, ValueFrame, diagnostic::codes};
use eqiora_io_hdf5::{
    Hdf5DatasetWrite, Hdf5WriteLimits, WrittenHdf5FileImage, write_hdf5_file_image,
};
use eqiora_io_xml::{XdmfExportLimits, XdmfExportPlan, XdmfField, XdmfFrame};
use eqiora_meshing::{DiscreteFieldAssociation, DiscreteFieldShape};
use std::collections::BTreeMap;

pub(crate) struct ExportField<'a> {
    pub(crate) name: String,
    pub(crate) identity: ArtifactDigest,
    pub(crate) association: DiscreteFieldAssociation,
    pub(crate) shape: DiscreteFieldShape,
    pub(crate) values: &'a [f64],
    pub(crate) visible: bool,
    pub(crate) information: BTreeMap<String, String>,
}

/// Meaning shared by physical-value producers; source-specific lineage is added by each owner.
pub(crate) fn physical_field_information(
    dimension: DimExponents,
    frame: ValueFrame,
    association: DiscreteFieldAssociation,
) -> BTreeMap<String, String> {
    BTreeMap::from([
        ("eqiora.units".into(), "coherent-SI".into()),
        ("eqiora.dimension".into(), dimension.to_string()),
        (
            "eqiora.value-frame".into(),
            match frame {
                ValueFrame::Invariant => "invariant",
                ValueFrame::SpatialCartesian => "spatial-cartesian",
            }
            .into(),
        ),
        (
            "eqiora.representation".into(),
            match association {
                DiscreteFieldAssociation::Vertex => "nodal-values",
                DiscreteFieldAssociation::Cell => "cell-values",
            }
            .into(),
        ),
    ])
}

#[derive(Debug, Default)]
pub(crate) struct Projection {
    datasets: BTreeMap<String, OwnedDataset>,
    frames: Vec<XdmfFrame>,
}

impl Projection {
    pub(crate) fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub(crate) fn finish(
        self,
        locator: &str,
        xdmf_limits: XdmfExportLimits,
        hdf5_limits: Hdf5WriteLimits,
    ) -> Result<(XdmfExportPlan, WrittenHdf5FileImage), Diagnostic> {
        let writes = dataset_writes(&self.datasets)?;
        let hdf5 = write_hdf5_file_image(&writes, hdf5_limits)?;
        let xdmf = XdmfExportPlan::new(locator, self.frames, xdmf_limits)?;
        Ok((xdmf, hdf5))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push_frame(
        &mut self,
        ordinal: u64,
        time_s: Option<f64>,
        geometry_state: ArtifactDigest,
        mesh: &eqiora_artifact::SimplicialMeshEnvelopeV1,
        current_coordinates: &[Vec<f64>],
        fields: &[ExportField<'_>],
        mut information: BTreeMap<String, String>,
    ) -> Result<(), Diagnostic> {
        if !matches!(mesh.dimension(), 2 | 3)
            || mesh
                .mesh()
                .cells()
                .iter()
                .any(|cell| cell.len() != mesh.dimension() + 1)
            || current_coordinates.len() != mesh.mesh().vertices().len()
            || current_coordinates
                .iter()
                .any(|point| point.len() != mesh.dimension())
        {
            return Err(export_error(
                "export requires complete current affine Tri3 or Tet4 geometry",
            ));
        }
        let mesh_digest = mesh.digest()?;
        let topology_path = format!("/meshes/{mesh_digest}/topology");
        let topology = mesh
            .mesh()
            .cells()
            .iter()
            .flatten()
            .map(|vertex| {
                u64::try_from(*vertex)
                    .map_err(|_| export_error("mesh vertex index exceeds portable HDF5 u64"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        insert_dataset(
            &mut self.datasets,
            topology_path.clone(),
            OwnedDataset::U64 {
                shape: vec![
                    portable_extent(mesh.mesh().cells().len())?,
                    portable_extent(mesh.dimension() + 1)?,
                ],
                values: topology,
            },
        )?;

        let geometry_path = format!("/geometry/{geometry_state}/coordinates");
        insert_dataset(
            &mut self.datasets,
            geometry_path.clone(),
            OwnedDataset::F64 {
                shape: vec![
                    portable_extent(current_coordinates.len())?,
                    portable_extent(mesh.dimension())?,
                ],
                values: current_coordinates.iter().flatten().copied().collect(),
            },
        )?;

        let mut xdmf_fields = Vec::new();
        for field in fields {
            let path = format!("/fields/{}/values", field.identity);
            let components = field.shape.component_count()?;
            let count = field
                .values
                .len()
                .checked_div(components)
                .filter(|_| field.values.len().is_multiple_of(components))
                .ok_or_else(|| export_error("export Field component cardinality differs"))?;
            let expected = match field.association {
                DiscreteFieldAssociation::Vertex => mesh.mesh().vertices().len(),
                DiscreteFieldAssociation::Cell => mesh.mesh().cells().len(),
            };
            if count != expected {
                return Err(export_error(
                    "export Field does not cover its exact mesh association",
                ));
            }
            let shape = if components == 1 {
                vec![portable_extent(count)?]
            } else {
                vec![portable_extent(count)?, portable_extent(components)?]
            };
            insert_dataset(
                &mut self.datasets,
                path.clone(),
                OwnedDataset::F64 {
                    shape,
                    values: field.values.to_vec(),
                },
            )?;
            if field.visible {
                xdmf_fields.push(
                    XdmfField::new(&field.name, field.association, field.shape, path)?
                        .with_information(field.information.clone())?,
                );
            }
        }
        information.insert("eqiora.mesh".into(), mesh_digest.to_string());
        information.insert("eqiora.geometry-state".into(), geometry_state.to_string());
        information.insert("eqiora.coordinates-unit".into(), "m".into());
        if time_s.is_some() {
            information.insert("eqiora.time-unit".into(), "s".into());
        }
        self.frames.push(
            XdmfFrame::new(
                ordinal,
                time_s,
                mesh.dimension(),
                mesh.mesh().vertices().len(),
                mesh.mesh().cells().len(),
                geometry_path,
                topology_path,
                xdmf_fields,
            )?
            .with_information(information)?,
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
enum OwnedDataset {
    U64 { shape: Vec<u64>, values: Vec<u64> },
    F64 { shape: Vec<u64>, values: Vec<f64> },
}

fn insert_dataset(
    datasets: &mut BTreeMap<String, OwnedDataset>,
    path: String,
    dataset: OwnedDataset,
) -> Result<(), Diagnostic> {
    if let Some(existing) = datasets.get(&path) {
        if existing != &dataset {
            return Err(export_error(
                "one content-addressed HDF5 path resolves to different arrays",
            ));
        }
    } else {
        datasets.insert(path, dataset);
    }
    Ok(())
}

fn dataset_writes(
    datasets: &BTreeMap<String, OwnedDataset>,
) -> Result<Vec<Hdf5DatasetWrite<'_>>, Diagnostic> {
    datasets
        .iter()
        .map(|(path, dataset)| match dataset {
            OwnedDataset::U64 { shape, values } => {
                Hdf5DatasetWrite::u64(path, shape.clone(), values)
            }
            OwnedDataset::F64 { shape, values } => {
                Hdf5DatasetWrite::f64(path, shape.clone(), values)
            }
        })
        .collect()
}

fn portable_extent(value: usize) -> Result<u64, Diagnostic> {
    u64::try_from(value).map_err(|_| export_error("HDF5 extent exceeds portable u64"))
}

pub(crate) fn export_error(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_EXTERNAL_DATA_EXPORT, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tetrahedral_projection_keeps_three_coordinate_and_vector_components() {
        use eqiora_artifact::SimplicialMeshEnvelopeV1;
        use eqiora_meshing::{MeshQualityGate, SimplicialMesh};
        let coordinates = vec![
            vec![0., 0., 0.],
            vec![1., 0., 0.],
            vec![0., 1., 0.],
            vec![0., 0., 1.],
        ];
        let mesh = SimplicialMeshEnvelopeV1::from_mesh(
            &SimplicialMesh::new(
                3,
                coordinates.clone(),
                vec![vec![0, 1, 2, 3]],
                MeshQualityGate::new(0.01).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let mut projection = Projection::default();
        let identity = ArtifactDigest::from_sha256([1; 32]);
        projection
            .push_frame(
                0,
                None,
                identity.clone(),
                &mesh,
                &coordinates,
                &[ExportField {
                    name: "velocity".into(),
                    identity: identity.clone(),
                    association: DiscreteFieldAssociation::Vertex,
                    shape: DiscreteFieldShape::Vector {
                        components: std::num::NonZeroU32::new(3).unwrap(),
                    },
                    values: &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
                    visible: true,
                    information: BTreeMap::new(),
                }],
                BTreeMap::new(),
            )
            .unwrap();
        let (xml, image) = projection
            .finish("tetra.h5", Default::default(), Default::default())
            .unwrap();
        let xml = std::str::from_utf8(xml.metadata_bytes()).unwrap();
        assert!(xml.contains("TopologyType=\"Tetrahedron\""));
        assert!(xml.contains("GeometryType=\"XYZ\""));
        assert!(xml.contains("Dimensions=\"4 3\""));
        use eqiora_io_hdf5::{
            Hdf5DatasetRequest, Hdf5FileImage, Hdf5ResolvedValues, Hdf5ScalarType,
        };
        let requests = [
            Hdf5DatasetRequest::new(
                format!("/geometry/{identity}/coordinates"),
                Hdf5ScalarType::F64,
                vec![4, 3],
            )
            .unwrap(),
            Hdf5DatasetRequest::new(
                format!("/fields/{identity}/values"),
                Hdf5ScalarType::F64,
                vec![4, 3],
            )
            .unwrap(),
        ];
        let arrays = eqiora_io_hdf5::resolve_hdf5_file_image(
            Hdf5FileImage::new(image.bytes()),
            &requests,
            Default::default(),
        )
        .unwrap();
        assert_eq!(
            arrays.values()[0],
            Hdf5ResolvedValues::F64(coordinates.into_iter().flatten().collect())
        );
        assert_eq!(
            arrays.values()[1],
            Hdf5ResolvedValues::F64((1..=12).map(f64::from).collect())
        );
    }
}
