use super::*;

pub(crate) fn validate_resources(
    spatial: &NativeSpatialPolicy,
    resources: &NativeMeshResources,
) -> Result<(), Diagnostic> {
    if matches!(
        spatial,
        NativeSpatialPolicy::StokesMiniP1(_) | NativeSpatialPolicy::TransientMiniP1(_)
    ) && resources.geometry()?.ambient_dimension() != 2
    {
        return Err(invalid(
            "this flow realization requires two-dimensional Geometry",
        ));
    }
    match (spatial, resources) {
        (NativeSpatialPolicy::CoordinateCellConstant, NativeMeshResources::Coordinates(_)) => {
            Ok(())
        }
        (NativeSpatialPolicy::ScalarTpfa(_), resources @ NativeMeshResources::Cartesian { .. }) => {
            validate_cartesian_resources(resources)
        }
        (
            NativeSpatialPolicy::FiniteElement(request),
            resources @ NativeMeshResources::Cartesian { .. },
        ) if request
            .policies()
            .all(|policy| policy == CommonSpatialPolicy::Q1) =>
        {
            validate_cartesian_resources(resources)
        }
        (
            NativeSpatialPolicy::FiniteElement(request),
            resources @ NativeMeshResources::MshSimplicial { geometry, mesh, .. },
        ) if ((geometry.region().is_some() && mesh.dimension() == 2)
            || (geometry.polyhedral_vertices().is_some() && mesh.dimension() == 3))
            && (request.policies().all(|policy| {
                matches!(
                    policy,
                    CommonSpatialPolicy::P1 | CommonSpatialPolicy::MiniP1
                )
            }) || (mesh.dimension() == 3
                && request.policies().all(|policy| {
                    matches!(
                        policy,
                        CommonSpatialPolicy::TetrahedralEdge | CommonSpatialPolicy::TetrahedralFace
                    )
                }))) =>
        {
            validate_simplicial_resources(resources)
        }
        (
            NativeSpatialPolicy::FiniteElement(request),
            resources @ NativeMeshResources::AffineTriangleSimplicial { mesh, .. },
        ) if request.policies().all(|policy| {
            matches!(
                policy,
                CommonSpatialPolicy::P1 | CommonSpatialPolicy::MiniP1
            )
        }) && mesh.dimension() == 2 =>
        {
            validate_simplicial_resources(resources)
        }
        (
            NativeSpatialPolicy::StokesMiniP1(_),
            resources @ (NativeMeshResources::MshSimplicial { .. }
            | NativeMeshResources::AffineTriangleSimplicial { .. }),
        ) => validate_simplicial_resources(resources),
        (
            NativeSpatialPolicy::TransientMiniP1(_),
            resources @ (NativeMeshResources::AffineTriangleSimplicial { .. }
            | NativeMeshResources::MshSimplicial { .. }),
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
            let policy = production.affine_triangle_policy().ok_or_else(|| {
                invalid("affine-triangle resource has a non-affine-triangle production policy")
            })?;
            production.validate_against_affine_triangle_resources(
                &policy,
                geometry,
                mesh,
                correspondence,
            )?;
        }
        NativeMeshResources::MshSimplicial {
            geometry,
            policy,
            provider_output,
            ..
        } => {
            let replayed =
                derive_msh_resources(geometry.clone(), *policy, provider_output.to_vec())?;
            if &replayed != resources {
                return Err(invalid(
                    "MSH common Mesh resources differ from exact source replay",
                ));
            }
        }
        NativeMeshResources::Coordinates(_) | NativeMeshResources::Cartesian { .. } => {
            return Err(invalid("authenticated owner is not simplicial"));
        }
    }
    let mesh = match resources {
        NativeMeshResources::AffineTriangleSimplicial { mesh, .. }
        | NativeMeshResources::MshSimplicial { mesh, .. } => mesh,
        NativeMeshResources::Coordinates(_) | NativeMeshResources::Cartesian { .. } => {
            unreachable!("rejected above")
        }
    };
    if mesh.dimension() != resources.geometry()?.ambient_dimension() {
        return Err(invalid(
            "simplicial common Mesh dimension differs from Geometry",
        ));
    }
    Ok(())
}

pub(crate) fn derive_gmsh_resources(
    geometry: CanonicalGeometryV1,
    policy: eqiora_artifact::GmshMeshPolicyV1,
    provider_output: Vec<u8>,
) -> Result<NativeMeshResources, Diagnostic> {
    derive_msh_resources(geometry, MshSourcePolicy::Gmsh4152(policy), provider_output)
}

pub(crate) fn derive_msh_resources(
    geometry: CanonicalGeometryV1,
    source_policy: MshSourcePolicy,
    provider_output: Vec<u8>,
) -> Result<NativeMeshResources, Diagnostic> {
    let (minimum_mean_ratio, maximum_boundary_facets) = match source_policy {
        MshSourcePolicy::Gmsh4152(policy) => (
            policy.minimum_mean_ratio(),
            policy.maximum_boundary_facets(),
        ),
        MshSourcePolicy::External(policy) => {
            if provider_output.len() > policy.maximum_source_bytes() {
                return Err(invalid(
                    "external MSH source exceeds the declared byte limit",
                ));
            }
            if geometry.region().is_none() && geometry.polyhedral_vertices().is_none() {
                return Err(invalid(
                    "external MSH import requires planar-region or convex-polyhedral Geometry",
                ));
            }
            (
                policy.minimum_mean_ratio(),
                policy.maximum_boundary_facets(),
            )
        }
    };
    let quality = eqiora_meshing::MeshQualityGate::new(minimum_mean_ratio)?;
    let (mesh, correspondence) = if geometry.polyhedral_vertices().is_some() {
        let imported = import_msh41(
            &provider_output,
            Msh41Policy::mesh(3, quality)?,
            |_, _, _| {},
        )?;
        let boundary_facets = (0..imported.entity_count(2).unwrap_or(0))
            .filter(|&index| imported.is_boundary_entity(MeshEntity::new(2, index)) == Some(true))
            .count();
        if boundary_facets > maximum_boundary_facets {
            return Err(invalid(
                "polyhedral mesh exceeds the declared boundary facet budget",
            ));
        }
        let mesh = SimplicialMeshEnvelopeV1::from_mesh(&imported)?;
        let definition = eqiora_artifact::GeometryDefinitionV1::from_canonical(&geometry)?;
        let correspondence =
            GeometryMeshCorrespondenceEnvelopeV1::from_polyhedra(&definition, &mesh)?;
        (mesh, correspondence)
    } else if geometry.region().is_some() {
        let imported = import_msh41(
            &provider_output,
            Msh41Policy::mesh(2, quality)?,
            |_, _, _| {},
        )?;
        if matches!(source_policy, MshSourcePolicy::External(_)) {
            let boundary_facets = (0..imported.entity_count(1).unwrap_or(0))
                .filter(|&index| {
                    imported.is_boundary_entity(MeshEntity::new(1, index)) == Some(true)
                })
                .count();
            if boundary_facets > maximum_boundary_facets {
                return Err(invalid(
                    "planar mesh exceeds the declared boundary facet budget",
                ));
            }
        }
        let mesh = SimplicialMeshEnvelopeV1::from_mesh(&imported)?;
        let definition = eqiora_artifact::GeometryDefinitionV1::from_canonical(&geometry)?;
        let correspondence = GeometryMeshCorrespondenceEnvelopeV1::from_region(&definition, &mesh)?;
        (mesh, correspondence)
    } else {
        derive_circular_hole_mesh(&geometry, quality, &provider_output)?
    };
    let production = match source_policy {
        MshSourcePolicy::Gmsh4152(policy) => {
            MeshProductionLineageEnvelopeV1::from_gmsh_4152_resources(
                policy,
                &geometry,
                &mesh,
                &correspondence,
            )?
        }
        MshSourcePolicy::External(policy) => {
            MeshProductionLineageEnvelopeV1::from_external_msh41_resources(
                policy,
                &geometry,
                &mesh,
                &correspondence,
            )?
        }
    };
    Ok(NativeMeshResources::MshSimplicial {
        geometry,
        policy: source_policy,
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
    CanonicalGeometryV1::decode_circular_hole_canonical(
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
