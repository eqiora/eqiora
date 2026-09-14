use super::*;

pub(super) fn natural_fem_facets<B>(
    mesh: &CartesianMesh,
    boundary: &B,
) -> Result<Vec<(LocalContribution, Vec<MeshEntity>)>, Diagnostic>
where
    B: Fn(usize, BoundarySide, &[f64]) -> CartesianBoundaryValue + ?Sized,
{
    let dimension = mesh.topological_dimension();
    let facet_quadrature = scalar_facet_quadrature(dimension)?;
    let facet_dimension = dimension - 1;
    let natural_facets = (0..mesh
        .entity_count(facet_dimension)
        .expect("mesh owns facets"))
        .filter_map(|facet_index| {
            let facet = MeshEntity::new(facet_dimension, facet_index);
            cartesian_boundary_facet_side(mesh, facet)
                .transpose()
                .map(|side| side.map(|side| (facet, side)))
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?
        .into_iter()
        .filter_map(|(facet, (axis, side))| {
            let geometry = mesh.geometry_map(facet).expect("mesh facet has geometry");
            let coordinates = geometry.origin();
            matches!(
                boundary(axis, side, coordinates),
                CartesianBoundaryValue::Natural(_)
            )
            .then_some((facet, axis, side))
        })
        .map(|(facet, axis, side)| {
            let geometry = mesh.geometry_map(facet).expect("mesh facet has geometry");
            let vertices = mesh
                .entity_vertices(facet)
                .expect("mesh facet has a vertex closure");
            let local =
                natural_fem_facet_contribution(&geometry, &facet_quadrature, &|coordinates| {
                    match boundary(axis, side, coordinates) {
                        CartesianBoundaryValue::Natural(value) => value,
                        CartesianBoundaryValue::Essential(_) => f64::NAN,
                    }
                })?;
            Ok((local, vertices))
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    Ok(natural_facets)
}

pub(super) fn essential_fem_values<B>(
    mesh: &CartesianMesh,
    boundary: &B,
) -> Result<Vec<Option<f64>>, Diagnostic>
where
    B: Fn(usize, BoundarySide, &[f64]) -> CartesianBoundaryValue + ?Sized,
{
    let vertex_count = mesh.entity_count(0).expect("mesh owns vertices");
    let mut fixed_values = Vec::with_capacity(vertex_count);
    for vertex_index in 0..vertex_count {
        let vertex = MeshEntity::new(0, vertex_index);
        if mesh
            .is_boundary_entity(vertex)
            .expect("mesh vertex has boundary classification")
        {
            let coordinates = mesh
                .vertex_coordinates(vertex)
                .expect("mesh vertex has geometry");
            let essential = boundary_sides(mesh, &coordinates)?
                .into_iter()
                .filter_map(|(axis, side)| match boundary(axis, side, &coordinates) {
                    CartesianBoundaryValue::Essential(value) => Some(value),
                    CartesianBoundaryValue::Natural(_) => None,
                })
                .try_fold(None, |accepted: Option<f64>, candidate| {
                    require_compatible_boundary_value(accepted, candidate)
                })?;
            fixed_values.push(essential);
        } else {
            fixed_values.push(None);
        }
    }
    Ok(fixed_values)
}

pub(super) fn validate_linearization_inputs(
    model: &ScalarEllipticCartesianModel,
    mesh: &CartesianMesh,
    solution_mesh: &CartesianMesh,
    quadrature: &QuadratureRule,
) -> Result<(), Diagnostic> {
    let bounds_match = (0..model.dimension()).all(|axis| {
        mesh.axis_bounds(axis).is_some_and(|bounds| {
            bounds[0].to_bits() == model.bounds()[axis][0].to_bits()
                && bounds[1].to_bits() == model.bounds()[axis][1].to_bits()
        })
    });
    if mesh != solution_mesh || mesh.topological_dimension() != model.dimension() || !bounds_match {
        return Err(invalid(
            "Cartesian linearization requires the exact primal mesh, model dimension, and Domain bounds",
        ));
    }
    validate_problem(mesh, quadrature)
}

pub(super) fn facet_measure_jvp(
    geometry: &AffineGeometryLinearization,
    quadrature: &QuadratureRule,
) -> Result<(f64, f64), Diagnostic> {
    require_geometry_rule(geometry.map(), quadrature)?;
    let area = quadrature
        .points()
        .iter()
        .map(|point| point.weight * geometry.map().measure_scale())
        .sum::<f64>();
    let tangent = quadrature
        .points()
        .iter()
        .map(|point| point.weight * geometry.measure_scale_tangent())
        .sum::<f64>();
    if !area.is_finite() || area <= 0.0 || !tangent.is_finite() {
        return Err(invalid(
            "Cartesian facet measure/JVP must be finite with positive primal measure",
        ));
    }
    Ok((area, tangent))
}

pub(super) fn transmissibility_jvp(
    diffusion: f64,
    diffusion_tangent: f64,
    area: f64,
    area_tangent: f64,
    distance: f64,
    distance_tangent: f64,
) -> Result<f64, Diagnostic> {
    let tangent = (diffusion_tangent * area + diffusion * area_tangent) / distance
        - diffusion * area * distance_tangent / distance.powi(2);
    if !tangent.is_finite() {
        return Err(invalid(
            "Cartesian transmissibility JVP produced a non-finite value",
        ));
    }
    Ok(tangent)
}

pub(super) fn require_positive_distance(distance: f64) -> Result<(), Diagnostic> {
    (distance.is_finite() && distance > 0.0)
        .then_some(())
        .ok_or_else(|| invalid("Cartesian two-point flux distance must be positive and finite"))
}

pub(super) fn ensure_finite_design_assembly(values: &[f64]) -> Result<(), Diagnostic> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or_else(|| invalid("Cartesian design derivative assembly produced a non-finite value"))
}

pub(super) struct CartesianEllipticCell<'a, K: ?Sized, S: ?Sized> {
    pub(super) coefficient: &'a K,
    pub(super) source: &'a S,
    pub(super) compiled: &'a AdmittedScalarGalerkinForm<'a>,
}

impl<K, S> LocalOperator<AffineGeometryMap> for CartesianEllipticCell<'_, K, S>
where
    K: Fn(&[f64]) -> f64 + ?Sized,
    S: Fn(&[f64]) -> f64 + ?Sized,
{
    fn evaluate(
        &self,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
    ) -> Result<LocalContribution, Diagnostic> {
        self.compiled
            .evaluate(geometry, quadrature, self.coefficient, self.source)
    }
}

pub(super) struct CartesianSourceCell<'a, S: ?Sized> {
    pub(super) source: &'a S,
}

impl<S> LocalOperator<AffineGeometryMap> for CartesianSourceCell<'_, S>
where
    S: Fn(&[f64]) -> f64 + ?Sized,
{
    fn evaluate(
        &self,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
    ) -> Result<LocalContribution, Diagnostic> {
        require_geometry_rule(geometry, quadrature)?;
        let mut physical = vec![0.0; geometry.physical_dimension()];
        let mut integral = 0.0;
        for point in quadrature.points() {
            geometry.map_point(&point.coordinates, &mut physical)?;
            let source = (self.source)(&physical);
            if !source.is_finite() {
                return Err(invalid("Cartesian source returned a non-finite value"));
            }
            integral += point.weight * geometry.measure_scale() * source;
        }
        LocalContribution::new(1, 1, vec![0.0], vec![integral])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct CartesianFacetPacket {
    pub(super) transmissibility: f64,
    pub(super) kind: CartesianFacetKind,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum CartesianFacetKind {
    Interior {
        left: usize,
        right: usize,
    },
    Essential {
        axis: usize,
        side: BoundarySide,
        cell: usize,
        value: f64,
    },
    Natural {
        axis: usize,
        side: BoundarySide,
        cell: usize,
        flux_integral: f64,
    },
}

pub(super) struct CartesianInteriorFlux;

impl LocalOperator<f64> for CartesianInteriorFlux {
    fn evaluate(
        &self,
        transmissibility: &f64,
        _quadrature: &QuadratureRule,
    ) -> Result<LocalContribution, Diagnostic> {
        LocalContribution::new(
            2,
            2,
            vec![
                *transmissibility,
                -*transmissibility,
                -*transmissibility,
                *transmissibility,
            ],
            vec![0.0, 0.0],
        )
    }
}

pub(super) struct CartesianBoundaryFlux;

impl LocalOperator<f64> for CartesianBoundaryFlux {
    fn evaluate(
        &self,
        transmissibility: &f64,
        _quadrature: &QuadratureRule,
    ) -> Result<LocalContribution, Diagnostic> {
        LocalContribution::new(1, 2, vec![*transmissibility, -*transmissibility], vec![0.0])
    }
}

pub(super) fn facet_transmissibility(
    facet_geometry: &AffineGeometryMap,
    distance: f64,
    coefficient: f64,
    quadrature: &QuadratureRule,
) -> Result<f64, Diagnostic> {
    require_geometry_rule(facet_geometry, quadrature)?;
    if !distance.is_finite() || distance <= 0.0 {
        return Err(invalid(
            "Cartesian two-point flux requires a positive finite normal distance",
        ));
    }
    let area = quadrature
        .points()
        .iter()
        .map(|point| point.weight * facet_geometry.measure_scale())
        .sum::<f64>();
    let transmissibility = coefficient * area / distance;
    if !transmissibility.is_finite() || transmissibility <= 0.0 {
        return Err(invalid(
            "Cartesian two-point transmissibility must be finite and positive",
        ));
    }
    Ok(transmissibility)
}

pub(super) fn natural_fem_facet_contribution<G>(
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
    flux: &G,
) -> Result<LocalContribution, Diagnostic>
where
    G: Fn(&[f64]) -> f64 + ?Sized,
{
    require_geometry_rule(geometry, quadrature)?;
    let dimension = geometry.reference_cell().dimension();
    if dimension == 0 {
        return LocalContribution::new(
            1,
            1,
            vec![0.0],
            vec![integrate_boundary_flux(geometry, quadrature, flux)?],
        );
    }
    let space = HypercubeQ1Space::new(dimension)?;
    let dof_count = space.local_dofs().len();
    let mut rhs = vec![0.0; dof_count];
    let mut physical = vec![0.0; geometry.physical_dimension()];
    for point in quadrature.points() {
        let basis = space.tabulate(&point.coordinates)?;
        geometry.map_point(&point.coordinates, &mut physical)?;
        let flux = flux(&physical);
        if !flux.is_finite() {
            return Err(invalid(
                "Cartesian natural boundary returned a non-finite value",
            ));
        }
        let scale = point.weight * geometry.measure_scale();
        for (entry, basis) in rhs.iter_mut().zip(basis.values()) {
            *entry += scale * flux * basis;
        }
    }
    LocalContribution::new(dof_count, dof_count, vec![0.0; dof_count * dof_count], rhs)
}

pub(super) fn integrate_boundary_flux<G>(
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
    flux: &G,
) -> Result<f64, Diagnostic>
where
    G: Fn(&[f64]) -> f64 + ?Sized,
{
    require_geometry_rule(geometry, quadrature)?;
    let mut physical = vec![0.0; geometry.physical_dimension()];
    let mut integral = 0.0;
    for point in quadrature.points() {
        geometry.map_point(&point.coordinates, &mut physical)?;
        let flux = flux(&physical);
        if !flux.is_finite() {
            return Err(invalid(
                "Cartesian natural boundary returned a non-finite value",
            ));
        }
        integral += point.weight * geometry.measure_scale() * flux;
    }
    integral
        .is_finite()
        .then_some(integral)
        .ok_or_else(|| invalid("Cartesian natural boundary integral is non-finite"))
}

pub(super) fn scalar_facet_quadrature(dimension: usize) -> Result<QuadratureRule, Diagnostic> {
    if dimension == 1 {
        Ok(QuadratureRule::point())
    } else {
        QuadratureRule::tensor_product_gauss_legendre(dimension - 1, 2)
    }
}

pub(super) fn cartesian_boundary_facet_side(
    mesh: &CartesianMesh,
    facet: MeshEntity,
) -> Result<Option<(usize, BoundarySide)>, Diagnostic> {
    if !mesh
        .is_boundary_entity(facet)
        .ok_or_else(|| invalid("Cartesian facet has no boundary classification"))?
    {
        return Ok(None);
    }
    let dimension = mesh.topological_dimension();
    let free_axes = mesh
        .entity_free_axes(facet)
        .ok_or_else(|| invalid("Cartesian facet has no tangent-axis classification"))?;
    let axis = (0..dimension)
        .find(|axis| free_axes.binary_search(axis).is_err())
        .ok_or_else(|| invalid("Cartesian boundary facet has no normal axis"))?;
    let coordinate = mesh
        .geometry_map(facet)
        .ok_or_else(|| invalid("Cartesian boundary facet has no geometry"))?
        .origin()[axis];
    let bounds = mesh
        .axis_bounds(axis)
        .ok_or_else(|| invalid("Cartesian boundary facet axis has no bounds"))?;
    let side = if coordinate.to_bits() == bounds[0].to_bits() {
        BoundarySide::Lower
    } else if coordinate.to_bits() == bounds[1].to_bits() {
        BoundarySide::Upper
    } else {
        return Err(invalid(
            "Cartesian boundary facet origin does not lie on its axis bounds",
        ));
    };
    Ok(Some((axis, side)))
}

pub(super) fn boundary_sides(
    mesh: &CartesianMesh,
    coordinates: &[f64],
) -> Result<Vec<(usize, BoundarySide)>, Diagnostic> {
    if coordinates.len() != mesh.topological_dimension() {
        return Err(invalid(
            "Cartesian boundary coordinate dimension is incompatible",
        ));
    }
    let mut sides = Vec::new();
    for (axis, coordinate) in coordinates.iter().copied().enumerate() {
        let bounds = mesh
            .axis_bounds(axis)
            .ok_or_else(|| invalid("Cartesian boundary axis has no bounds"))?;
        if coordinate.to_bits() == bounds[0].to_bits() {
            sides.push((axis, BoundarySide::Lower));
        }
        if coordinate.to_bits() == bounds[1].to_bits() {
            sides.push((axis, BoundarySide::Upper));
        }
    }
    Ok(sides)
}

pub(crate) fn require_compatible_boundary_value(
    accepted: Option<f64>,
    candidate: f64,
) -> Result<Option<f64>, Diagnostic> {
    if !candidate.is_finite() {
        return Err(invalid("Cartesian boundary returned a non-finite value"));
    }
    if let Some(accepted) = accepted {
        let scale = accepted.abs().max(candidate.abs()).max(1.0);
        if (accepted - candidate).abs() > 256.0 * f64::EPSILON * scale {
            return Err(invalid(
                "Cartesian essential boundary values disagree at an edge or corner",
            ));
        }
    }
    Ok(Some(candidate))
}

pub(super) fn prepare_cell_field_reconstruction<B>(
    mesh: &CartesianMesh,
    boundary: &B,
) -> Result<(CartesianMesh, Vec<Option<f64>>), Diagnostic>
where
    B: Fn(usize, BoundarySide, &[f64]) -> CartesianBoundaryValue + ?Sized,
{
    let dimension = mesh.topological_dimension();
    let axes = (0..dimension)
        .map(|axis| {
            let coordinates = mesh
                .axis_coordinates(axis)
                .expect("mesh owns every physical axis");
            let mut dual = Vec::with_capacity(coordinates.len() + 1);
            dual.push(coordinates[0]);
            dual.extend(
                coordinates
                    .windows(2)
                    .map(|pair| pair[0] + 0.5 * (pair[1] - pair[0])),
            );
            dual.push(coordinates[coordinates.len() - 1]);
            dual
        })
        .collect::<Vec<_>>();
    let reconstruction_mesh = CartesianMesh::from_axes(axes)?;
    let vertex_count = reconstruction_mesh
        .entity_count(0)
        .expect("reconstruction mesh owns vertices");
    let mut boundary_values = Vec::with_capacity(vertex_count);
    for vertex_index in 0..vertex_count {
        let vertex = MeshEntity::new(0, vertex_index);
        if reconstruction_mesh
            .is_boundary_entity(vertex)
            .expect("reconstruction vertex has boundary classification")
        {
            let coordinates = reconstruction_mesh
                .vertex_coordinates(vertex)
                .expect("reconstruction vertex has geometry");
            let value = boundary_sides(&reconstruction_mesh, &coordinates)?
                .into_iter()
                .filter_map(|(axis, side)| match boundary(axis, side, &coordinates) {
                    CartesianBoundaryValue::Essential(value) => Some(value),
                    CartesianBoundaryValue::Natural(_) => None,
                })
                .try_fold(None, require_compatible_boundary_value)?;
            boundary_values.push(value);
        } else {
            boundary_values.push(None);
        }
    }
    Ok((reconstruction_mesh, boundary_values))
}

pub(super) fn reconstruct_cell_field_from_boundary_values(
    reconstruction_mesh: CartesianMesh,
    source_mesh: &CartesianMesh,
    cell_values: &[f64],
    boundary_values: Vec<Option<f64>>,
    facets: &[CartesianFacetPacket],
) -> Result<CartesianQ1Field, Diagnostic> {
    let dimension = source_mesh.topological_dimension();
    let reconstruction_vertex_count = reconstruction_mesh
        .entity_count(0)
        .ok_or_else(|| invalid("Cartesian reconstruction mesh has no vertex stratum"))?;
    if Some(cell_values.len()) != source_mesh.entity_count(dimension)
        || boundary_values.len() != reconstruction_vertex_count
    {
        return Err(invalid(
            "Cartesian reconstruction state differs from its finalized system",
        ));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(boundary_values.len())
        .map_err(|_| {
            finish_allocation("Cartesian FVM reconstruction allocation exceeds platform capacity")
        })?;
    for (vertex_index, boundary_value) in boundary_values.into_iter().enumerate() {
        if let Some(value) = boundary_value {
            values.push(value);
            continue;
        }
        let vertex = MeshEntity::new(0, vertex_index);
        if reconstruction_mesh
            .is_boundary_entity(vertex)
            .ok_or_else(|| invalid("Cartesian reconstruction vertex has no boundary class"))?
        {
            let indices = reconstruction_mesh
                .vertex_multi_index(vertex)
                .ok_or_else(|| invalid("Cartesian reconstruction vertex has no multi-index"))?;
            let mut traces = Vec::new();
            for (axis, side) in boundary_sides(
                &reconstruction_mesh,
                &reconstruction_mesh
                    .vertex_coordinates(vertex)
                    .expect("reconstruction vertex has coordinates"),
            )? {
                let target = (0..dimension)
                    .map(|candidate_axis| {
                        let cells = source_mesh
                            .axis_cell_count(candidate_axis)
                            .expect("source mesh owns every axis");
                        if candidate_axis == axis {
                            match side {
                                BoundarySide::Lower => 0,
                                BoundarySide::Upper => cells - 1,
                            }
                        } else {
                            indices[candidate_axis].saturating_sub(1).min(cells - 1)
                        }
                    })
                    .collect::<Vec<_>>();
                if let Some((facet, cell, flux_integral)) = facets.iter().find_map(|facet| {
                    let CartesianFacetKind::Natural {
                        axis: facet_axis,
                        side: facet_side,
                        cell,
                        flux_integral,
                    } = facet.kind
                    else {
                        return None;
                    };
                    (facet_axis == axis
                        && facet_side == side
                        && source_mesh
                            .cell_multi_index(MeshEntity::new(dimension, cell))
                            .is_some_and(|index| index == target))
                    .then_some((facet, cell, flux_integral))
                }) {
                    let cell_value = cell_values.get(cell).copied().ok_or_else(|| {
                        invalid("Cartesian natural reconstruction cell is unavailable")
                    })?;
                    traces.push(cell_value + flux_integral / facet.transmissibility);
                }
            }
            if traces.is_empty() {
                return Err(invalid(
                    "Cartesian natural reconstruction has no matching boundary facet",
                ));
            }
            values.push(traces.iter().sum::<f64>() / traces.len() as f64);
            continue;
        }
        let source_indices = reconstruction_mesh
            .vertex_multi_index(vertex)
            .ok_or_else(|| invalid("Cartesian reconstruction vertex has no multi-index"))?;
        let source_cell_index = source_indices.iter().enumerate().try_fold(
            0_usize,
            |linear, (axis, &dual_index)| {
                let cell_count = source_mesh.axis_cell_count(axis).ok_or_else(|| {
                    invalid("Cartesian reconstruction source axis is unavailable")
                })?;
                let source_index = dual_index.checked_sub(1).ok_or_else(|| {
                    invalid("Cartesian reconstruction interior index is on the boundary")
                })?;
                if source_index >= cell_count {
                    return Err(invalid(
                        "Cartesian reconstruction interior index exceeds its source axis",
                    ));
                }
                linear
                    .checked_mul(cell_count)
                    .and_then(|linear| linear.checked_add(source_index))
                    .ok_or_else(|| invalid("Cartesian reconstruction cell index overflows usize"))
            },
        )?;
        let value = cell_values.get(source_cell_index).copied().ok_or_else(|| {
            invalid("Cartesian reconstruction cell index exceeds its finalized field")
        })?;
        values.push(value);
    }
    CartesianQ1Field::new(reconstruction_mesh, values)
}

pub(super) fn finish_allocation(message: &'static str) -> Diagnostic {
    Diagnostic::error(codes::NUMERICAL_SOLVE_FAILED, message)
}

pub(super) fn validate_problem(
    mesh: &CartesianMesh,
    quadrature: &QuadratureRule,
) -> Result<(), Diagnostic> {
    require_cell_rule(mesh, quadrature)
}

pub(super) fn require_cell_rule(
    mesh: &CartesianMesh,
    quadrature: &QuadratureRule,
) -> Result<(), Diagnostic> {
    let expected = ReferenceCell::hypercube(mesh.topological_dimension())?;
    require_reference(quadrature, expected)
}

pub(super) fn require_facet_rule(
    dimension: usize,
    quadrature: &QuadratureRule,
) -> Result<(), Diagnostic> {
    let expected = if dimension == 1 {
        ReferenceCell::point()
    } else {
        ReferenceCell::hypercube(dimension - 1)?
    };
    require_reference(quadrature, expected)
}

pub(super) fn require_geometry_rule(
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
) -> Result<(), Diagnostic> {
    require_reference(quadrature, geometry.reference_cell())
}

pub(super) fn require_reference(
    quadrature: &QuadratureRule,
    expected: ReferenceCell,
) -> Result<(), Diagnostic> {
    (quadrature.reference_cell() == expected)
        .then_some(())
        .ok_or_else(|| invalid("Cartesian quadrature reference cell does not match its consumer"))
}

pub(super) fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_DISCRETIZATION, message)
}
