use super::*;

pub(crate) fn validate_resources(
    spatial: NativeSpatialPolicy,
    resources: &NativeMeshResources,
) -> Result<(), Diagnostic> {
    match (spatial, resources) {
        (
            NativeSpatialPolicy::ScalarQ1 | NativeSpatialPolicy::ScalarTpfa,
            resources @ NativeMeshResources::Cartesian { .. },
        ) => validate_cartesian_resources(resources),
        (NativeSpatialPolicy::ElasticityQ1, resources @ NativeMeshResources::Cartesian { .. }) => {
            validate_cartesian_resources(resources)
        }
        (
            NativeSpatialPolicy::StokesMiniP1(_),
            resources @ NativeMeshResources::GmshSimplicial { .. },
        ) => validate_simplicial_resources(resources),
        (
            NativeSpatialPolicy::TransientMiniP1(_),
            resources @ (NativeMeshResources::AffineTriangleSimplicial { .. }
            | NativeMeshResources::GmshSimplicial { .. }),
        ) => validate_simplicial_resources(resources),
        (
            NativeSpatialPolicy::TransientCellCentered(_),
            resources @ NativeMeshResources::Cartesian { .. },
        ) => validate_cartesian_resources(resources),
        _ => Err(invalid(
            "requested spatial realization and authenticated common Mesh kind are incompatible",
        )),
    }
}

pub(crate) fn validate_cartesian_resources(
    resources: &NativeMeshResources,
) -> Result<(), Diagnostic> {
    let NativeMeshResources::Cartesian {
        geometry,
        mesh,
        correspondence,
        production,
    } = resources
    else {
        return Err(invalid("authenticated owner is not Cartesian"));
    };
    let policy = production
        .cartesian_cells()
        .ok_or_else(|| invalid("Cartesian resource has a non-Cartesian policy"))?;
    if geometry.cartesian_box_bounds().is_some() {
        correspondence.validate_against_cartesian_box_v1(geometry, mesh, policy.cells())?;
    } else {
        let cells: [usize; 2] = policy
            .cells()
            .try_into()
            .map_err(|_| invalid("planar rectangle Cartesian policy must have two axes"))?;
        correspondence.validate_against_planar_rectangle_v2_cartesian(geometry, mesh, cells)?;
    }
    production.validate_against_structured_cartesian_v2_resources(
        &policy,
        geometry,
        mesh,
        correspondence,
    )?;
    if mesh.dimension() != policy.cells().len()
        || policy
            .cells()
            .iter()
            .enumerate()
            .any(|(axis, &count)| mesh.mesh().axis_cell_count(axis) != Some(count))
    {
        return Err(invalid(
            "Cartesian Mesh topology differs from its exact production policy",
        ));
    }
    Ok(())
}

pub(crate) fn validate_simplicial_resources(
    resources: &NativeMeshResources,
) -> Result<(), Diagnostic> {
    match resources {
        NativeMeshResources::AffineTriangleSimplicial {
            geometry,
            mesh,
            correspondence,
            production,
        } => {
            let policy = production.affine_triangle_cells().ok_or_else(|| {
                invalid("affine-triangle resource has a non-affine-triangle production policy")
            })?;
            correspondence.validate_against_planar_rectangle_v2_affine_triangles(
                geometry,
                mesh,
                policy.cells(),
            )?;
            production.validate_against_affine_triangle_rectangle_v1_resources(
                policy,
                geometry,
                mesh,
                correspondence,
            )?;
        }
        NativeMeshResources::AdjacentPartitionSimplicial {
            geometry,
            mesh,
            correspondence,
            production,
        } => {
            let policy = production.affine_triangle_cells().ok_or_else(|| {
                invalid("adjacent-partition resource has a non-affine-triangle production policy")
            })?;
            correspondence.validate_against_adjacent_rectangle_partition_affine_triangles(
                geometry,
                mesh,
                policy.cells(),
            )?;
            production.validate_against_affine_triangle_rectangle_v1_resources(
                policy,
                geometry,
                mesh,
                correspondence,
            )?;
        }
        NativeMeshResources::GmshSimplicial {
            geometry,
            policy,
            provider_output,
            ..
        } => {
            let replayed =
                derive_gmsh_resources(geometry.clone(), *policy, provider_output.to_vec())?;
            if &replayed != resources {
                return Err(invalid(
                    "Gmsh common Mesh resources differ from exact provider-output replay",
                ));
            }
        }
        NativeMeshResources::Cartesian { .. } => {
            return Err(invalid("authenticated owner is not simplicial"));
        }
    }
    let mesh = match resources {
        NativeMeshResources::AffineTriangleSimplicial { mesh, .. }
        | NativeMeshResources::AdjacentPartitionSimplicial { mesh, .. }
        | NativeMeshResources::GmshSimplicial { mesh, .. } => mesh,
        NativeMeshResources::Cartesian { .. } => unreachable!("rejected above"),
    };
    if mesh.dimension() != 2 {
        return Err(invalid("simplicial common Mesh must be two-dimensional"));
    }
    Ok(())
}

pub(crate) fn derive_gmsh_resources(
    geometry: CanonicalGeometryV1,
    policy: eqiora_artifact::GmshMeshPolicyV1,
    provider_output: Vec<u8>,
) -> Result<NativeMeshResources, Diagnostic> {
    let quality = eqiora_meshing::MeshQualityGate::new(policy.minimum_mean_ratio())?;
    let (mesh, correspondence) = if geometry.region().is_some() {
        let imported = import_msh41(
            &provider_output,
            Msh41Policy::mesh(2, quality)?,
            |_, _, _| {},
        )?;
        let mesh = SimplicialMeshEnvelopeV1::from_mesh(&imported)?;
        let definition = eqiora_artifact::GeometryDefinitionV1::from_canonical(&geometry)?;
        let correspondence = GeometryMeshCorrespondenceEnvelopeV1::from_region(&definition, &mesh)?;
        (mesh, correspondence)
    } else {
        derive_circular_hole_mesh(&geometry, quality, &provider_output)?
    };
    let production = MeshProductionLineageEnvelopeV1::from_gmsh_4152_resources(
        policy,
        &geometry,
        &mesh,
        &correspondence,
    )?;
    Ok(NativeMeshResources::GmshSimplicial {
        geometry,
        policy,
        provider_output: provider_output.into_boxed_slice(),
        mesh,
        correspondence,
        production,
    })
}

fn derive_circular_hole_mesh(
    geometry: &CanonicalGeometryV1,
    quality: eqiora_meshing::MeshQualityGate,
    provider_output: &[u8],
) -> Result<
    (
        SimplicialMeshEnvelopeV1,
        GeometryMeshCorrespondenceEnvelopeV1,
    ),
    Diagnostic,
> {
    CanonicalGeometryV1::decode_planar_circular_hole_v2_canonical(
        geometry.canonical_bytes(),
        eqiora_geometry::CanonicalGeometryLimits::default(),
    )
    .map_err(|_| invalid("Gmsh provider observation requires exact planar circular-hole v2"))?;
    let import_policy = Msh41Policy::ascii_with_entity_assignments(2, quality)?;
    let mut assignments = BTreeMap::new();
    let mesh = import_msh41(provider_output, import_policy, |dimension, tag, indices| {
        assignments.insert((dimension, tag), indices.to_vec());
    })?;
    if assignments.keys().copied().collect::<Vec<_>>()
        != [(1, 1), (1, 5), (1, 6), (1, 7), (1, 8), (2, 1)]
    {
        return Err(invalid(
            "Gmsh provider observation has a foreign entity-tag inventory",
        ));
    }
    let mut source_edge_facets: [Vec<usize>; 5] = std::array::from_fn(|_| Vec::new());
    for (tag, source_edge) in [(1_u32, 4_usize), (5, 2), (6, 1), (7, 3), (8, 0)] {
        source_edge_facets[source_edge] = assignments
            .get(&(1, tag))
            .expect("exact tag inventory checked")
            .clone();
    }
    let mesh = SimplicialMeshEnvelopeV1::from_mesh(&mesh)?;
    let correspondence =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_circular_hole_v2_mesh_assignments(
            geometry,
            &mesh,
            source_edge_facets,
        )?;
    Ok((mesh, correspondence))
}
