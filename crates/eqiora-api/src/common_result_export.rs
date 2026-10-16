//! Project accepted common occurrences through the shared XDMF/HDF5 owner.
use std::collections::BTreeMap;
use std::num::NonZeroU16;

use eqiora_artifact::{ArtifactDigest, DiscreteFieldEnvelopeV1, SimplicialMeshEnvelopeV1};
use eqiora_core::{Diagnostic, Id, ScalarDomain, ValueFrame, ValueType, entity::kinds};
use eqiora_io_hdf5::WrittenHdf5FileImage;
use eqiora_io_xml::XdmfExportPlan;
use eqiora_meshing::{
    DiscreteFieldAssociation, DiscreteFieldPayload, DiscreteFieldShape, MeshEntity,
};
use eqiora_numerics::{CommonResult, CommonSpatialPlan, CommonTrajectory, ResolvedCommonPlan};
use eqiora_realization::Space;
use sha2::{Digest, Sha256};

use crate::XdmfHdf5ExportLimits;
use crate::xdmf_export::{ExportField, Projection, export_error, physical_field_information};

/// Export an accepted static Result or its complete retained spatial Trajectory.
///
/// The explicit profile is real scalar or Cartesian-vector P1 / cell-constant
/// Fields on complete affine Tri3/Tet4 supports. Prescribed moving coordinates
/// use the Plan's assembly map at each accepted State. Regional subsets, complex
/// coefficients, MINI bubbles, moments and higher-order spaces reject as a whole.
/// Values use coherent SI; Information retains dimensions, frames and lineage.
/// No locator is opened and no filesystem write occurs here.
pub fn export_common_result_xdmf_hdf5(
    result: &CommonResult,
    locator: &str,
    limits: XdmfHdf5ExportLimits,
) -> Result<(XdmfExportPlan, WrittenHdf5FileImage), Diagnostic> {
    let mut projection = Projection::default();
    if let Some(trajectory) = result.trajectory() {
        project_trajectory(&mut projection, trajectory, Some(result.identity()), limits)?;
    } else {
        let plan = result.plan();
        let spatial = spatial_plan(plan)?;
        let mesh = plan
            .authenticated_mesh()
            .ok_or_else(|| export_error("Result lacks its Mesh"))?;
        let mesh = mesh
            .simplicial_mesh()
            .ok_or_else(|| export_error("export requires an affine simplicial Mesh"))?;
        if result.field_count() != spatial.fields().len() {
            return Err(export_error(
                "Result Field inventory differs from the exact Plan",
            ));
        }
        let mut fields = Vec::new();
        for (field, ty) in spatial.fields() {
            let name = field.ulid().to_string();
            let index = (0..result.field_count())
                .find(|&index| result.field(index).is_some_and(|value| value.0 == name))
                .ok_or_else(|| export_error("Result is missing an exact Plan Field"))?;
            if result.field_block_count(index) != 1 {
                return Err(export_error(
                    "export cannot relabel multiple coefficient blocks as physical values",
                ));
            }
            let (_, values, _) = result.field_block(index, 0).expect("one accepted block");
            fields.push(project_field(
                spatial,
                mesh,
                field,
                ty,
                spatial.field_coefficient_entities(field)?,
                values,
            )?);
        }
        push_frame(
            &mut projection,
            plan,
            mesh,
            None,
            spatial.geometry_coordinates(None)?,
            fields,
            BTreeMap::from([("eqiora.result".into(), result.identity().into())]),
        )?;
    }
    projection.finish(locator, limits.xdmf, limits.hdf5)
}

/// Export a standalone complete spatial Trajectory after exact request replay.
/// Uses the same profile and projection as [`export_common_result_xdmf_hdf5`].
pub fn export_common_trajectory_xdmf_hdf5(
    trajectory: &CommonTrajectory,
    locator: &str,
    limits: XdmfHdf5ExportLimits,
) -> Result<(XdmfExportPlan, WrittenHdf5FileImage), Diagnostic> {
    let mut projection = Projection::default();
    project_trajectory(&mut projection, trajectory, None, limits)?;
    projection.finish(locator, limits.xdmf, limits.hdf5)
}

fn project_trajectory(
    projection: &mut Projection,
    trajectory: &CommonTrajectory,
    result_identity: Option<&str>,
    limits: XdmfHdf5ExportLimits,
) -> Result<(), Diagnostic> {
    let CommonTrajectory::SpatialTransient { request, .. } = trajectory else {
        return Err(export_error("XDMF requires a spatial Trajectory"));
    };
    // The native enum can also be assembled directly, including inside a Result.
    // Both export entry points must reaccept its complete request and exact States.
    let accepted = CommonTrajectory::from_bytes(&trajectory.to_bytes()?, request.plan())?;
    let trajectory = &accepted;
    let CommonTrajectory::SpatialTransient {
        request, states, ..
    } = trajectory
    else {
        return Err(export_error("XDMF requires a spatial Trajectory"));
    };
    if states.len() > limits.xdmf.max_frames {
        return Err(export_error(
            "Trajectory exceeds the configured export frame budget",
        ));
    }
    let plan = request.plan();
    let spatial = spatial_plan(plan)?;
    let mesh = plan
        .authenticated_mesh()
        .ok_or_else(|| export_error("Trajectory lacks its Mesh"))?;
    let mesh = mesh
        .simplicial_mesh()
        .ok_or_else(|| export_error("export requires an affine simplicial Mesh"))?;
    for (_, step) in states {
        let state = step.state();
        let fields = spatial
            .state_fields(state)?
            .map(|(field, ty, entities, values)| {
                project_field(spatial, mesh, field, ty, entities, values)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut information = BTreeMap::from([
            ("eqiora.trajectory".into(), trajectory.identity().into()),
            ("eqiora.request".into(), request.identity().into()),
            ("eqiora.state".into(), state.identity().into()),
        ]);
        if let Some(identity) = result_identity {
            information.insert("eqiora.result".into(), identity.into());
        }
        push_frame(
            projection,
            plan,
            mesh,
            state.time_s(),
            spatial.geometry_coordinates(Some(state))?,
            fields,
            information,
        )?;
    }
    Ok(())
}

fn spatial_plan(plan: &ResolvedCommonPlan) -> Result<&CommonSpatialPlan, Diagnostic> {
    plan.as_spatial()
        .ok_or_else(|| export_error("export requires the common spatial Field realization"))
}

struct FieldProjection {
    name: String,
    block: DiscreteFieldEnvelopeV1,
    information: BTreeMap<String, String>,
}

fn project_field(
    plan: &CommonSpatialPlan,
    mesh: &SimplicialMeshEnvelopeV1,
    field: Id<kinds::Field>,
    ty: &ValueType,
    entities: Vec<MeshEntity>,
    values: &[f64],
) -> Result<FieldProjection, Diagnostic> {
    if ty.scalar_domain() != ScalarDomain::Real {
        return Err(export_error(
            "XDMF physical values require real coefficients; complex parts cannot be discarded",
        ));
    }
    let shape = match (ty.shape().extents(), ty.frame()) {
        ([], ValueFrame::Invariant) => DiscreteFieldShape::Scalar,
        ([components], ValueFrame::SpatialCartesian)
            if usize::try_from(components.get()) == Ok(mesh.dimension()) =>
        {
            DiscreteFieldShape::Vector {
                components: *components,
            }
        }
        _ => {
            return Err(export_error(
                "XDMF profile requires a scalar or a spatial Cartesian vector",
            ));
        }
    };
    let space = plan.field_space(field)?;
    let (association, dimension, count) = if space == Space::continuous_lagrange(NonZeroU16::MIN) {
        (
            DiscreteFieldAssociation::Vertex,
            0,
            mesh.mesh().vertices().len(),
        )
    } else if space == Space::cell_constant() {
        (
            DiscreteFieldAssociation::Cell,
            mesh.dimension(),
            mesh.mesh().cells().len(),
        )
    } else {
        return Err(export_error(
            "coefficient Space is not a physical P1 node or constant cell value; export requires an explicit projection",
        ));
    };
    let values = reorder_values(
        &entities,
        values,
        dimension,
        count,
        shape.component_count()?,
    )?;
    let block = DiscreteFieldEnvelopeV1::from_payload(
        mesh,
        &DiscreteFieldPayload::new(mesh.mesh(), association, shape, values)?,
    )?;
    Ok(FieldProjection {
        name: field.ulid().to_string(),
        block,
        information: physical_field_information(ty.dimension(), ty.frame(), association),
    })
}

fn reorder_values(
    entities: &[MeshEntity],
    values: &[f64],
    dimension: usize,
    count: usize,
    components: usize,
) -> Result<Vec<f64>, Diagnostic> {
    let size = count
        .checked_mul(components)
        .ok_or_else(|| export_error("export Field shape overflows"))?;
    if entities.len() != count || values.len() != size {
        return Err(export_error(
            "export requires a complete physical Field support on its Mesh",
        ));
    }
    let mut output = vec![0.; size];
    let mut seen = vec![false; count];
    for (entity, source) in entities.iter().zip(values.chunks_exact(components)) {
        if entity.dimension() != dimension || entity.index() >= count || seen[entity.index()] {
            return Err(export_error(
                "Field entity order is not a bijection of its exact Mesh association",
            ));
        }
        seen[entity.index()] = true;
        let offset = entity.index() * components;
        output[offset..offset + components].copy_from_slice(source);
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn push_frame(
    projection: &mut Projection,
    plan: &ResolvedCommonPlan,
    mesh: &SimplicialMeshEnvelopeV1,
    time: Option<f64>,
    coordinates: Vec<Vec<f64>>,
    fields: Vec<FieldProjection>,
    mut information: BTreeMap<String, String>,
) -> Result<(), Diagnostic> {
    information.insert("eqiora.plan".into(), plan.identity().into());
    information.insert("eqiora.model".into(), plan.model_digest().into());
    let coordinate_bytes =
        serde_json::to_vec(&coordinates).map_err(|error| export_error(error.to_string()))?;
    let geometry = ArtifactDigest::from_sha256(Sha256::digest(&coordinate_bytes).into());
    let fields = fields
        .iter()
        .map(|field| {
            Ok(ExportField {
                name: field.name.clone(),
                identity: field.block.digest()?,
                association: field.block.association(),
                shape: field.block.component_shape(),
                values: field.block.values(),
                visible: true,
                information: field.information.clone(),
            })
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    projection.push_frame(
        u64::try_from(projection.frame_count())
            .map_err(|_| export_error("frame ordinal overflows"))?,
        time,
        geometry,
        mesh,
        &coordinates,
        &fields,
        information,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn physical_values_follow_exact_entity_indices_without_dropping_components() {
        let entities = [
            MeshEntity::new(0, 2),
            MeshEntity::new(0, 0),
            MeshEntity::new(0, 1),
        ];
        assert_eq!(
            reorder_values(&entities, &[20., 21., 0., 1., 10., 11.], 0, 3, 2).unwrap(),
            [0., 1., 10., 11., 20., 21.]
        );
        for wrong in [
            [
                MeshEntity::new(0, 0),
                MeshEntity::new(0, 0),
                MeshEntity::new(0, 2),
            ],
            [
                MeshEntity::new(0, 0),
                MeshEntity::new(0, 1),
                MeshEntity::new(0, 3),
            ],
            [
                MeshEntity::new(0, 0),
                MeshEntity::new(0, 1),
                MeshEntity::new(1, 2),
            ],
        ] {
            assert!(reorder_values(&wrong, &[0.; 6], 0, 3, 2).is_err());
        }
        assert!(reorder_values(&entities[..2], &[0.; 4], 0, 3, 2).is_err());
    }
}
