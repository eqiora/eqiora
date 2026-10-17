//! Exact Field support is independent of coefficient storage domain.
use super::*;

impl CommonSpatialPlan {
    pub(crate) fn require_fixed_observation_geometry(&self) -> Result<(), Diagnostic> {
        if let RecognizedNativeModel::Spatial(equations) = self.admission.recognized_model()
            && equations
                .regions
                .iter()
                .any(|region| region.form.motion().is_some())
        {
            return Err(invalid(
                "spatial observation requires fixed geometry; moving geometry needs its accepted map",
            ));
        }
        Ok(())
    }
}

pub(super) fn field_support<S: crate::spatial_expression::Coefficient>(
    equations: &ExecutableSpatialEquations<S>,
    mesh: &eqiora_meshing::CartesianMesh,
    field: eqiora_core::RawId,
    space: Space,
) -> Result<(Vec<usize>, Vec<usize>), Diagnostic> {
    let region = equations
        .regions
        .iter()
        .find(|region| {
            region
                .form
                .represented_fields()
                .iter()
                .any(|(id, _)| *id == field)
        })
        .ok_or_else(|| invalid("Field absent from exact Region inventory"))?;
    let mut shape = Vec::new();
    for (axis, bounds) in region.cartesian()?.bounds.iter().enumerate() {
        let coordinates = mesh.axis_coordinates(axis).expect("axis");
        let start = coordinates
            .iter()
            .position(|x| *x == bounds[0])
            .ok_or_else(|| invalid("Field support lower bound absent"))?;
        let end = coordinates
            .iter()
            .position(|x| *x == bounds[1])
            .ok_or_else(|| invalid("Field support upper bound absent"))?;
        shape.push(
            end - start
                + usize::from(space == Space::continuous_lagrange(std::num::NonZeroU16::MIN)),
        );
    }
    let represented = region.form.represented_fields();
    let value_type = &represented
        .iter()
        .find(|(id, _)| *id == field)
        .expect("selected Field")
        .1;
    shape.extend(
        value_type
            .shape()
            .extents()
            .iter()
            .map(|extent| usize::try_from(extent.get()).expect("portable Field extent")),
    );
    let domains = equations.cell_domains(mesh)?;
    let mut entities = BTreeSet::new();
    for (index, domain) in domains.iter().enumerate() {
        if *domain == region.form.domain() {
            if space == Space::cell_constant() {
                entities.insert(index);
                continue;
            }
            entities.extend(
                mesh.incidence(
                    eqiora_meshing::MeshEntity::new(mesh.topological_dimension(), index),
                    0,
                )
                .expect("cell closure")
                .iter()
                .map(|vertex| vertex.entity.index()),
            );
        }
    }
    Ok((shape, entities.into_iter().collect()))
}

pub(super) fn simplicial_support<S: crate::spatial_expression::Coefficient>(
    equations: &ExecutableSpatialEquations<S>,
    mesh: &SimplicialMeshEnvelopeV1,
    field: eqiora_core::RawId,
    selected: &[eqiora_realization::DomainFieldDiscretization],
) -> Result<(Vec<usize>, Vec<usize>), Diagnostic> {
    let (entities, _) = simplicial_topology(equations, mesh, field, selected)?;
    let mut shape = vec![entities.len()];
    let space = selected
        .iter()
        .flat_map(|domain| domain.field_spaces())
        .find(|binding| binding.field().erase() == field)
        .ok_or_else(|| invalid("Field has no selected coefficient Space"))?
        .space();
    if space == Space::continuous_lagrange(std::num::NonZeroU16::MIN) {
        let fields = equations.represented_fields();
        let (_, value_type) = fields
            .iter()
            .find(|(id, _)| *id == field)
            .ok_or_else(|| invalid("Field absent from exact simplicial inventory"))?;
        shape.extend(
            value_type
                .shape()
                .extents()
                .iter()
                .map(|extent| usize::try_from(extent.get()).expect("portable Field extent")),
        );
    }
    Ok((
        shape,
        entities.into_iter().map(|entity| entity.index()).collect(),
    ))
}

pub(super) fn simplicial_topology<S: crate::spatial_expression::Coefficient>(
    equations: &ExecutableSpatialEquations<S>,
    mesh: &SimplicialMeshEnvelopeV1,
    field: eqiora_core::RawId,
    selected: &[eqiora_realization::DomainFieldDiscretization],
) -> Result<
    (
        Vec<eqiora_meshing::MeshEntity>,
        Vec<eqiora_meshing::MeshEntity>,
    ),
    Diagnostic,
> {
    let (domain, binding) = selected
        .iter()
        .find_map(|domain| {
            domain
                .field_spaces()
                .iter()
                .find(|binding| binding.field().erase() == field)
                .map(|binding| (domain.domain().erase(), binding))
        })
        .ok_or_else(|| invalid("Field has no selected coefficient Space"))?;
    let region = equations
        .regions
        .iter()
        .find(|region| {
            region.form.domain() == domain
                && region
                    .form
                    .represented_fields()
                    .iter()
                    .any(|(id, _)| *id == field)
        })
        .ok_or_else(|| invalid("Field is absent from the exact simplicial Region"))?;
    let support = region.simplicial()?;
    if support.mesh != mesh.digest()? {
        return Err(invalid("Field support differs from the authenticated Mesh"));
    }
    let space = crate::discrete_space::DiscreteSpace::new(
        binding.space(),
        eqiora_meshing::ReferenceCell::simplex(mesh.dimension())?,
    )?;
    // Use the same Space-to-cell association as RegionDofMap, without assembling
    // equations or reactions merely to recover coefficient topology.
    let cells = support
        .cells
        .iter()
        .map(|cell| eqiora_meshing::MeshEntity::new(mesh.dimension(), cell.index()))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut entities = BTreeSet::new();
    for &cell in &cells {
        entities.extend(
            space
                .bind_cell(mesh.mesh(), cell)?
                .into_iter()
                .map(|(entity, _)| entity),
        );
    }
    let entities = entities.into_iter().collect();
    Ok((entities, cells))
}

impl CommonSpatialPlan {
    /// Exact volume closure for a real nodal Field; no operator or reaction assembly.
    pub(crate) fn simplicial_point_support(
        &self,
        field: eqiora_core::Id<eqiora_core::entity::kinds::Field>,
        domain: eqiora_core::Id<eqiora_core::entity::kinds::Domain>,
    ) -> Result<(Vec<usize>, Vec<usize>), Diagnostic> {
        if self.field_space(field)? != Space::continuous_lagrange(std::num::NonZeroU16::MIN) {
            return Err(invalid("point sampling requires a continuous-P1 Field"));
        }
        let RecognizedNativeModel::Spatial(equations) = self.admission.recognized_model() else {
            return Err(invalid("point sampling requires real spatial equations"));
        };
        equations
            .regions
            .iter()
            .find(|region| {
                region.form.domain() == domain.erase()
                    && region
                        .form
                        .represented_fields()
                        .iter()
                        .any(|(id, _)| *id == field.erase())
            })
            .ok_or_else(|| invalid("point Domain is not the exact Field volume support"))?;
        let (NativeMeshResources::GmshSimplicial { mesh, .. }
        | NativeMeshResources::AffineTriangleSimplicial { mesh, .. }) = self.admission.resources()
        else {
            return Err(invalid(
                "point sampling requires an authenticated simplicial Mesh",
            ));
        };
        let (vertices, cells) = simplicial_topology(
            equations,
            mesh,
            field.erase(),
            &self.admission.discretizations,
        )?;
        Ok((
            vertices.into_iter().map(|vertex| vertex.index()).collect(),
            cells.into_iter().map(|cell| cell.index()).collect(),
        ))
    }
}

impl CommonSpatialPlan {
    /// Exact volume or parent-oriented boundary measure retained by the admitted Region.
    pub(crate) fn simplicial_observation_support(
        &self,
        domain: eqiora_core::RawId,
    ) -> Result<(Vec<usize>, Option<Vec<eqiora_meshing::MeshEntity>>), Diagnostic> {
        let RecognizedNativeModel::Spatial(equations) = self.admission.recognized_model() else {
            return Err(invalid(
                "simplicial observation requires real spatial equations",
            ));
        };
        for region in &equations.regions {
            let support = region.simplicial()?;
            if region.form.domain() == domain || support.boundaries.contains(&domain) {
                let facets = if region.form.domain() == domain {
                    None
                } else {
                    Some(
                        support
                            .facets
                            .get(&domain)
                            .ok_or_else(|| invalid("boundary observation lacks its exact facets"))?
                            .clone(),
                    )
                };
                return Ok((
                    support.cells.iter().map(|cell| cell.index()).collect(),
                    facets,
                ));
            }
        }
        Err(invalid(
            "Observable Domain is outside the exact simplicial Region inventory",
        ))
    }
}

pub(super) fn observation_support<S: crate::spatial_expression::Coefficient>(
    equations: &ExecutableSpatialEquations<S>,
    domain: eqiora_core::RawId,
) -> Result<ObservableSupport, Diagnostic> {
    if let Some(interface) = equations
        .interfaces
        .iter()
        .find(|interface| interface.physical_support() == Some(domain))
    {
        let side = &interface.sides()[0];
        let region = equations
            .regions
            .iter()
            .find(|region| region.form.domain() == side.domain())
            .ok_or_else(|| invalid("Interface side is outside exact Region inventory"))?;
        let support = region.cartesian()?;
        let orientation = support
            .boundaries
            .iter()
            .find_map(|(orientation, boundary)| {
                (*boundary == side.boundary()).then_some(*orientation)
            })
            .ok_or_else(|| invalid("Interface Boundary is outside its exact Cartesian Region"))?;
        return Ok((support.bounds.clone(), Some(orientation)));
    }
    let region = equations
        .regions
        .iter()
        .find(|region| {
            region.form.domain() == domain
                || region.cartesian().is_ok_and(|support| {
                    support
                        .boundaries
                        .values()
                        .any(|boundary| *boundary == domain)
                })
        })
        .ok_or_else(|| invalid("Observable Domain is outside exact Region inventory"))?;
    let boundary = if domain == region.form.domain() {
        None
    } else {
        Some(region.cartesian()?.boundaries.iter().find_map(|(side, id)| (*id == domain).then_some(*side)).ok_or_else(|| invalid("Observable Domain is not the exact volume or boundary realized by this Plan"))?)
    };
    Ok((region.cartesian()?.bounds.clone(), boundary))
}
