use super::*;

pub(super) fn validate_cartesian_wire(
    wire: &WireGeometryMeshCorrespondenceV1,
    limits: GeometryDecoderLimits,
) -> Result<(), Diagnostic> {
    if wire.schema != CORRESPONDENCE_SCHEMA || wire.encoding != CANONICAL_ENCODING {
        return Err(invalid_artifact(
            "unsupported geometry-mesh correspondence schema or encoding",
        ));
    }
    ArtifactDigest::from_hex(wire.geometry_sha256.clone())?;
    ArtifactDigest::from_hex(wire.mesh_sha256.clone())?;
    let dimension = usize::try_from(wire.dimension)
        .map_err(|_| invalid_artifact("correspondence dimension exceeds local usize"))?;
    if dimension == 0 || wire.bodies.is_empty() || wire.boundaries.is_empty() {
        return Err(invalid_artifact(
            "geometry-mesh correspondence requires positive dimension, bodies, and boundaries",
        ));
    }
    if wire.bodies.len() + wire.boundaries.len() > limits.max_geometry_entities {
        return Err(invalid_artifact(
            "geometry-mesh assignment count exceeds decoder limits",
        ));
    }
    let mut membership_count = 0_usize;
    let mut domains = BTreeSet::new();
    let mut body_entities = BTreeSet::new();
    for body in &wire.bodies {
        parse_ulid(&body.domain_ulid, "correspondence body")?;
        let entity = body.geometry_entity.decode()?;
        let cells = decode_indices(&body.cell_indices)?;
        if entity.dimension() != dimension
            || cells.is_empty()
            || !domains.insert(body.domain_ulid.as_str())
            || !body_entities.insert(entity)
            || !strictly_sorted_unique(&cells)
        {
            return Err(invalid_artifact(
                "body assignments must be unique, full-dimensional, nonempty, and canonical",
            ));
        }
        membership_count = membership_count
            .checked_add(cells.len())
            .ok_or_else(|| invalid_artifact("geometry membership count overflows usize"))?;
    }
    let mut boundary_entities = BTreeMap::<GeometryEntity, (&str, Vec<usize>)>::new();
    for boundary in &wire.boundaries {
        parse_ulid(&boundary.domain_ulid, "correspondence boundary")?;
        parse_ulid(&boundary.parent_ulid, "correspondence parent")?;
        let entity = boundary.geometry_entity.decode()?;
        let facets = decode_indices(&boundary.facet_indices)?;
        if entity.dimension() != dimension - 1
            || facets.is_empty()
            || !domains.insert(boundary.domain_ulid.as_str())
            || !strictly_sorted_unique(&facets)
            || boundary.orientation != WireBoundaryOrientation::ParentOutward
            || usize::try_from(boundary.axis).map_or(true, |axis| axis >= dimension)
        {
            return Err(invalid_artifact(
                "boundary assignments must be unique, codimension-one, nonempty, and parent-outward",
            ));
        }
        if let Some((first_parent, first_facets)) = boundary_entities.get(&entity) {
            if *first_parent == boundary.parent_ulid || *first_facets != facets {
                return Err(invalid_artifact(
                    "a shared geometry boundary requires distinct parents and identical facet membership",
                ));
            }
        } else {
            boundary_entities.insert(entity, (&boundary.parent_ulid, facets.clone()));
        }
        membership_count = membership_count
            .checked_add(facets.len())
            .ok_or_else(|| invalid_artifact("geometry membership count overflows usize"))?;
    }
    if membership_count > limits.max_geometry_mesh_memberships
        || !strictly_sorted_assignments(&wire.bodies, &wire.boundaries)
    {
        return Err(invalid_artifact(
            "geometry-mesh memberships exceed limits or are not canonical",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub(super) enum WireCorrespondenceV1 {
    Cartesian(WireGeometryMeshCorrespondenceV1),
    CartesianBoxV1(cartesian_box_v1_correspondence::WireCartesianBoxV1CorrespondenceV1),
    AuthoredRegion(correspondence_sources::WireAuthoredRegionCorrespondenceV1),
    Polyhedra(polyhedral_correspondence::WirePolyhedra),
    PlanarCircularHoleV2(
        planar_circular_hole_v2_correspondence::WirePlanarCircularHoleV2CorrespondenceV1,
    ),
    PlanarRectangleV2(planar_rectangle_v2_correspondence::WirePlanarRectangleV2CorrespondenceV1),
}

impl WireCorrespondenceV1 {
    pub(super) fn cartesian(&self) -> Option<&WireGeometryMeshCorrespondenceV1> {
        match self {
            Self::Cartesian(wire) => Some(wire),
            Self::CartesianBoxV1(_)
            | Self::Polyhedra(_)
            | Self::AuthoredRegion(_)
            | Self::PlanarCircularHoleV2(_)
            | Self::PlanarRectangleV2(_) => None,
        }
    }

    pub(super) fn geometry_sha256(&self) -> &str {
        match self {
            Self::Cartesian(wire) => &wire.geometry_sha256,
            Self::CartesianBoxV1(wire) => &wire.geometry_sha256,
            Self::AuthoredRegion(wire) => &wire.geometry_sha256,
            Self::Polyhedra(wire) => &wire.geometry_sha256,
            Self::PlanarCircularHoleV2(wire) => &wire.geometry_sha256,
            Self::PlanarRectangleV2(wire) => &wire.geometry_sha256,
        }
    }

    pub(super) fn mesh_sha256(&self) -> &str {
        match self {
            Self::Cartesian(wire) => &wire.mesh_sha256,
            Self::CartesianBoxV1(wire) => &wire.mesh_sha256,
            Self::AuthoredRegion(wire) => &wire.mesh_sha256,
            Self::Polyhedra(wire) => &wire.mesh_sha256,
            Self::PlanarCircularHoleV2(wire) => &wire.mesh_sha256,
            Self::PlanarRectangleV2(wire) => &wire.mesh_sha256,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireGeometryMeshCorrespondenceV1 {
    pub(super) schema: String,
    pub(super) encoding: String,
    pub(super) geometry_sha256: String,
    pub(super) mesh_sha256: String,
    pub(super) dimension: u64,
    pub(super) bodies: Vec<WireBodyAssignment>,
    pub(super) boundaries: Vec<WireBoundaryAssignment>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireBodyAssignment {
    pub(super) domain_ulid: String,
    pub(super) geometry_entity: WireGeometryEntity,
    pub(super) cell_indices: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireBoundaryAssignment {
    pub(super) domain_ulid: String,
    pub(super) parent_ulid: String,
    pub(super) geometry_entity: WireGeometryEntity,
    pub(super) axis: u64,
    pub(super) side: WireBoundarySide,
    pub(super) orientation: WireBoundaryOrientation,
    pub(super) facet_indices: Vec<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum WireBoundarySide {
    Lower,
    Upper,
}

impl WireBoundarySide {
    pub(super) const fn encode(side: BoundarySide) -> Self {
        match side {
            BoundarySide::Lower => Self::Lower,
            BoundarySide::Upper => Self::Upper,
        }
    }

    pub(super) const fn decode(self) -> BoundarySide {
        match self {
            Self::Lower => BoundarySide::Lower,
            Self::Upper => BoundarySide::Upper,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum WireBoundaryOrientation {
    ParentOutward,
}

pub(super) fn validate_coincident_sides(
    first_bounds: &[(f64, f64)],
    axis: usize,
    first_side: BoundarySide,
    second_bounds: &[(f64, f64)],
    second_side: BoundarySide,
    tolerance: f64,
) -> Result<(), Diagnostic> {
    if first_bounds.len() != second_bounds.len() || axis >= first_bounds.len() {
        return Err(invalid_artifact("interface parent dimensions differ"));
    }
    let first_coordinate = match first_side {
        BoundarySide::Lower => first_bounds[axis].0,
        BoundarySide::Upper => first_bounds[axis].1,
    };
    let second_coordinate = match second_side {
        BoundarySide::Lower => second_bounds[axis].0,
        BoundarySide::Upper => second_bounds[axis].1,
    };
    if (first_coordinate - second_coordinate).abs() > tolerance
        || first_bounds.iter().zip(second_bounds).enumerate().any(
            |(candidate_axis, (&first, &second))| {
                candidate_axis != axis
                    && ((first.0 - second.0).abs() > tolerance
                        || (first.1 - second.1).abs() > tolerance)
            },
        )
    {
        return Err(invalid_artifact(
            "interface boundary embeddings are not coincident complete sides",
        ));
    }
    Ok(())
}

pub(super) fn encode_indices(indices: &[usize], label: &str) -> Result<Vec<u64>, Diagnostic> {
    indices
        .iter()
        .map(|&index| {
            u64::try_from(index)
                .map_err(|_| invalid_artifact(format!("{label} index exceeds portable u64")))
        })
        .collect()
}

pub(super) fn decode_indices(indices: &[u64]) -> Result<Vec<usize>, Diagnostic> {
    indices
        .iter()
        .map(|&index| {
            usize::try_from(index)
                .map_err(|_| invalid_artifact("mesh entity index exceeds local usize"))
        })
        .collect()
}

pub(super) fn strictly_sorted_unique(values: &[usize]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

pub(super) fn strictly_sorted_assignments(
    bodies: &[WireBodyAssignment],
    boundaries: &[WireBoundaryAssignment],
) -> bool {
    bodies
        .windows(2)
        .all(|pair| pair[0].geometry_entity < pair[1].geometry_entity)
        && boundaries.windows(2).all(|pair| {
            (
                pair[0].geometry_entity,
                &pair[0].parent_ulid,
                &pair[0].domain_ulid,
            ) < (
                pair[1].geometry_entity,
                &pair[1].parent_ulid,
                &pair[1].domain_ulid,
            )
        })
}

pub(super) fn parse_ulid(value: &str, label: &str) -> Result<Ulid, Diagnostic> {
    Ulid::from_str(value).map_err(|_| invalid_artifact(format!("{label} ULID is invalid")))
}
