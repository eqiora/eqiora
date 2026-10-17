//! Mathematical fields shared by live execution and candidate revalidation.
use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CartesianFemPoint {
    pub(super) field: CartesianQ1Field,
    pub(super) free_vertices: Vec<usize>,
    pub(super) algebraic_values: Vec<f64>,
    pub(super) canonical_system: Arc<CanonicalCsrSystemView>,
}

impl CartesianFemPoint {
    pub(super) fn field(&self) -> &CartesianQ1Field {
        &self.field
    }
    pub(super) fn free_vertices(&self) -> &[usize] {
        &self.free_vertices
    }
    pub(super) fn algebraic_values(&self) -> &[f64] {
        &self.algebraic_values
    }
    pub(super) fn canonical_system(&self) -> &Arc<CanonicalCsrSystemView> {
        &self.canonical_system
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CartesianFvmPoint {
    pub(super) mesh: CartesianMesh,
    pub(super) cell_centers: Vec<Vec<f64>>,
    pub(super) cell_values: Vec<f64>,
    pub(super) canonical_system: Arc<CanonicalCsrSystemView>,
}

impl CartesianFvmPoint {
    pub(super) fn mesh(&self) -> &CartesianMesh {
        &self.mesh
    }
    pub(super) fn cell_centers(&self) -> &[Vec<f64>] {
        &self.cell_centers
    }
    pub(super) fn cell_values(&self) -> &[f64] {
        &self.cell_values
    }
    pub(super) fn canonical_system(&self) -> &Arc<CanonicalCsrSystemView> {
        &self.canonical_system
    }
}

/// Assemble accepted-point design actions for a Cartesian Q1 FEM solve.
///
/// The resulting relation uses only free vertex unknowns. Its `R_p` includes
/// constitutive-coefficient, source, and eliminated essential-boundary
/// derivatives for explicitly selected model-Parameter and Domain-bound
/// coordinates.
///
/// # Errors
/// Returns a numerical/lowering diagnostic for a mismatched solution, invalid
/// quadrature, expression action, geometry, or non-finite assembly.
pub fn linearize_scalar_elliptic_cartesian_fem(
    model: &ScalarEllipticCartesianModel,
    mesh: &CartesianMesh,
    solution: &ScalarEllipticCartesianFemSolution,
    quadrature: &QuadratureRule,
    selected_coordinates: &[SpatialDesignCoordinate],
) -> Result<AssembledLinearizedRelation, Diagnostic> {
    linearize_fem_point(
        model,
        mesh,
        &solution.point,
        quadrature,
        selected_coordinates,
    )
}

fn linearize_fem_point(
    model: &ScalarEllipticCartesianModel,
    mesh: &CartesianMesh,
    solution: &CartesianFemPoint,
    quadrature: &QuadratureRule,
    selected_coordinates: &[SpatialDesignCoordinate],
) -> Result<AssembledLinearizedRelation, Diagnostic> {
    validate_linearization_inputs(model, mesh, solution.field().mesh(), quadrature)?;
    let dimension = mesh.topological_dimension();
    let selected = select_design_coordinates(model, selected_coordinates)?;
    let design_dimension = selected.coordinates.len();
    let unknown_dimension = solution.algebraic_values().len();
    let mut free_indices = vec![None; mesh.entity_count(0).expect("mesh owns vertices")];
    for (dof, vertex) in solution.free_vertices().iter().copied().enumerate() {
        let Some(slot) = free_indices.get_mut(vertex) else {
            return Err(invalid(
                "Cartesian FEM solution contains an invalid free vertex",
            ));
        };
        if slot.replace(dof).is_some() {
            return Err(invalid(
                "Cartesian FEM solution contains a duplicate free vertex",
            ));
        }
    }
    let mut design_jacobian = vec![0.0; unknown_dimension * design_dimension];
    let space = DiscreteSpace::new(
        eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
        eqiora_meshing::ReferenceCell::hypercube(dimension)?,
    )?;
    let mut parameter_tangent = vec![0.0; model.parameter_fields().len()];

    for (coordinate, action) in selected.actions.iter().copied().enumerate() {
        activate_model_parameter(action, &mut parameter_tangent);
        for cell_index in 0..mesh.entity_count(dimension).expect("mesh owns cells") {
            let cell = MeshEntity::new(dimension, cell_index);
            let geometry = design_geometry(mesh, cell, action)?;
            let map = geometry.map();
            require_geometry_rule(map, quadrature)?;
            let inverse = map.inverse_jacobian()?;
            let inverse_tangent = geometry.inverse_jacobian_tangent()?;
            let vertices = mesh
                .entity_vertices(cell)
                .expect("mesh cell has a vertex closure");
            let fixed_tangents = vertices
                .iter()
                .map(|vertex| {
                    if free_indices[vertex.index()].is_some() {
                        Ok(0.0)
                    } else {
                        let vertex_geometry = design_geometry(mesh, *vertex, action)?;
                        let coordinates = vertex_geometry.map().origin();
                        model
                            .essential_boundary_jvp(
                                coordinates,
                                vertex_geometry.origin_tangent(),
                                &parameter_tangent,
                            )
                            .map(|(_, tangent)| tangent)
                    }
                })
                .collect::<Result<Vec<_>, Diagnostic>>()?;
            let mut physical = vec![0.0; dimension];
            let mut physical_tangent = vec![0.0; dimension];
            for point in quadrature.points() {
                let basis = space.tabulate(&point.coordinates)?;
                geometry.map_point_jvp(&point.coordinates, &mut physical, &mut physical_tangent)?;
                let (diffusion, diffusion_tangent) =
                    model.coefficient_jvp(&physical, &physical_tangent, &parameter_tangent)?;
                let (source, source_tangent) =
                    model.source_jvp(&physical, &physical_tangent, &parameter_tangent)?;
                let scale = point.weight * map.measure_scale();
                let scale_tangent = point.weight * geometry.measure_scale_tangent();
                let physical_gradients = (0..vertices.len())
                    .map(|dof| {
                        physical_gradient(
                            basis
                                .gradient(dof)
                                .expect("tabulation owns every basis gradient"),
                            &inverse,
                            dimension,
                        )
                    })
                    .collect::<Vec<_>>();
                let physical_gradient_tangents = (0..vertices.len())
                    .map(|dof| {
                        physical_gradient(
                            basis
                                .gradient(dof)
                                .expect("tabulation owns every basis gradient"),
                            &inverse_tangent,
                            dimension,
                        )
                    })
                    .collect::<Vec<_>>();
                let accepted_values = vertices
                    .iter()
                    .map(|vertex| solution.field().vertex_values()[vertex.index()])
                    .collect::<Vec<_>>();
                let state_gradient = weighted_gradient(&accepted_values, &physical_gradients);
                let state_gradient_tangent = weighted_gradient_tangent(
                    &accepted_values,
                    &fixed_tangents,
                    &physical_gradients,
                    &physical_gradient_tangents,
                );
                for (local_row, vertex) in vertices.iter().enumerate() {
                    let Some(global_row) = free_indices[vertex.index()] else {
                        continue;
                    };
                    let gradient_product = dot(&physical_gradients[local_row], &state_gradient);
                    let energy = diffusion * gradient_product;
                    let energy_tangent = diffusion_tangent * gradient_product
                        + diffusion
                            * (dot(&physical_gradient_tangents[local_row], &state_gradient)
                                + dot(&physical_gradients[local_row], &state_gradient_tangent));
                    let load = source * basis.values()[local_row];
                    let load_tangent = source_tangent * basis.values()[local_row];
                    let residual_tangent =
                        scale_tangent * (energy - load) + scale * (energy_tangent - load_tangent);
                    design_jacobian[global_row * design_dimension + coordinate] += residual_tangent;
                }
            }
        }
        let facet_quadrature = scalar_facet_quadrature(dimension)?;
        let facet_dimension = dimension - 1;
        let facet_space = (facet_dimension > 0)
            .then(|| {
                DiscreteSpace::new(
                    eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
                    eqiora_meshing::ReferenceCell::hypercube(facet_dimension)?,
                )
            })
            .transpose()?;
        for facet_index in 0..mesh
            .entity_count(facet_dimension)
            .expect("mesh owns facets")
        {
            let facet = MeshEntity::new(facet_dimension, facet_index);
            let Some((axis, side)) = cartesian_boundary_facet_side(mesh, facet)? else {
                continue;
            };
            let geometry = design_geometry(mesh, facet, action)?;
            let map = geometry.map();
            let (condition, _, _) = model.boundary_jvp(
                axis,
                side,
                map.origin(),
                geometry.origin_tangent(),
                &parameter_tangent,
            )?;
            if !matches!(
                condition,
                crate::canonical::ScalarEllipticCartesianBoundary::Natural(_)
            ) {
                continue;
            }
            let vertices = mesh
                .entity_vertices(facet)
                .expect("mesh facet has a vertex closure");
            let mut physical = vec![0.0; dimension];
            let mut physical_tangent = vec![0.0; dimension];
            for point in facet_quadrature.points() {
                let basis_values = match &facet_space {
                    Some(space) => space.tabulate(&point.coordinates)?.values().to_vec(),
                    None => vec![1.0],
                };
                geometry.map_point_jvp(&point.coordinates, &mut physical, &mut physical_tangent)?;
                let (_, flux, flux_tangent) = model.boundary_jvp(
                    axis,
                    side,
                    &physical,
                    &physical_tangent,
                    &parameter_tangent,
                )?;
                let scale = point.weight * map.measure_scale();
                let scale_tangent = point.weight * geometry.measure_scale_tangent();
                for (local_row, vertex) in vertices.iter().enumerate() {
                    let Some(global_row) = free_indices[vertex.index()] else {
                        continue;
                    };
                    design_jacobian[global_row * design_dimension + coordinate] -=
                        (scale_tangent * flux + scale * flux_tangent) * basis_values[local_row];
                }
            }
        }
        parameter_tangent.fill(0.0);
    }
    ensure_finite_design_assembly(&design_jacobian)?;
    AssembledLinearizedRelation::from_shared_canonical(
        Arc::clone(solution.canonical_system()),
        solution.algebraic_values().to_vec(),
        selected.coordinates,
        selected.values,
        design_jacobian,
    )
}

/// Assemble accepted-point design actions for a Cartesian TPFA solve.
///
/// The method-native cell unknowns and balances are retained. Constitutive,
/// source, essential-boundary, and orthogonality-preserving geometry actions
/// use the caller's exact selected coordinate order, matching the FEM relation.
///
/// # Errors
/// Returns a numerical/lowering diagnostic for a mismatched solution, invalid
/// quadrature, topology, expression action, or non-finite assembly.
pub fn linearize_scalar_elliptic_cartesian_fvm(
    model: &ScalarEllipticCartesianModel,
    mesh: &CartesianMesh,
    solution: &ScalarEllipticCartesianFvmSolution,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    selected_coordinates: &[SpatialDesignCoordinate],
) -> Result<AssembledLinearizedRelation, Diagnostic> {
    linearize_fvm_point(
        model,
        mesh,
        &solution.point,
        cell_quadrature,
        facet_quadrature,
        selected_coordinates,
    )
}

fn linearize_fvm_point(
    model: &ScalarEllipticCartesianModel,
    mesh: &CartesianMesh,
    solution: &CartesianFvmPoint,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    selected_coordinates: &[SpatialDesignCoordinate],
) -> Result<AssembledLinearizedRelation, Diagnostic> {
    validate_linearization_inputs(model, mesh, solution.mesh(), cell_quadrature)?;
    let dimension = mesh.topological_dimension();
    require_facet_rule(dimension, facet_quadrature)?;
    let selected = select_design_coordinates(model, selected_coordinates)?;
    let design_dimension = selected.coordinates.len();
    let cell_count = mesh.entity_count(dimension).expect("mesh owns cells");
    let mut design_jacobian = vec![0.0; cell_count * design_dimension];
    let mut parameter_tangent = vec![0.0; model.parameter_fields().len()];

    for (coordinate, action) in selected.actions.iter().copied().enumerate() {
        activate_model_parameter(action, &mut parameter_tangent);
        for cell_index in 0..cell_count {
            let geometry = design_geometry(mesh, MeshEntity::new(dimension, cell_index), action)?;
            let map = geometry.map();
            require_geometry_rule(map, cell_quadrature)?;
            let mut physical = vec![0.0; dimension];
            let mut physical_tangent = vec![0.0; dimension];
            let mut source_tangent_integral = 0.0;
            for point in cell_quadrature.points() {
                geometry.map_point_jvp(&point.coordinates, &mut physical, &mut physical_tangent)?;
                let (source, source_tangent) =
                    model.source_jvp(&physical, &physical_tangent, &parameter_tangent)?;
                source_tangent_integral += point.weight
                    * (geometry.measure_scale_tangent() * source
                        + map.measure_scale() * source_tangent);
            }
            design_jacobian[cell_index * design_dimension + coordinate] -= source_tangent_integral;
        }

        let facet_dimension = dimension - 1;
        for facet_index in 0..mesh
            .entity_count(facet_dimension)
            .expect("mesh owns facets")
        {
            let facet = MeshEntity::new(facet_dimension, facet_index);
            let geometry = design_geometry(mesh, facet, action)?;
            let map = geometry.map();
            let free_axes = mesh
                .entity_free_axes(facet)
                .expect("mesh facet exposes its tangent axes");
            let normal_axis = (0..dimension)
                .find(|axis| free_axes.binary_search(axis).is_err())
                .ok_or_else(|| invalid("Cartesian facet has no normal axis"))?;
            let cells = mesh
                .incidence(facet, dimension)
                .ok_or_else(|| invalid("Cartesian facet adjacency is unavailable"))?;
            let (area, area_tangent) = facet_measure_jvp(&geometry, facet_quadrature)?;
            let (diffusion, diffusion_tangent) = model.coefficient_jvp(
                map.origin(),
                geometry.origin_tangent(),
                &parameter_tangent,
            )?;
            match cells.as_slice() {
                [left, right] => {
                    let left_index = left.entity.index();
                    let right_index = right.entity.index();
                    let left_geometry = design_geometry(mesh, left.entity, action)?;
                    let right_geometry = design_geometry(mesh, right.entity, action)?;
                    let delta = solution.cell_centers()[right_index][normal_axis]
                        - solution.cell_centers()[left_index][normal_axis];
                    let distance = delta.abs();
                    require_positive_distance(distance)?;
                    let distance_tangent = delta.signum()
                        * (right_geometry.origin_tangent()[normal_axis]
                            - left_geometry.origin_tangent()[normal_axis]);
                    let transmissibility_tangent = transmissibility_jvp(
                        diffusion,
                        diffusion_tangent,
                        area,
                        area_tangent,
                        distance,
                        distance_tangent,
                    )?;
                    let jump =
                        solution.cell_values()[left_index] - solution.cell_values()[right_index];
                    let contribution = transmissibility_tangent * jump;
                    design_jacobian[left_index * design_dimension + coordinate] += contribution;
                    design_jacobian[right_index * design_dimension + coordinate] -= contribution;
                }
                [cell] => {
                    let cell_index = cell.entity.index();
                    let cell_geometry = design_geometry(mesh, cell.entity, action)?;
                    let boundary_coordinates = map.origin();
                    let (axis, side) = cartesian_boundary_facet_side(mesh, facet)?
                        .expect("one-cell Cartesian facet is on the boundary");
                    let (condition, boundary, boundary_tangent) = model.boundary_jvp(
                        axis,
                        side,
                        boundary_coordinates,
                        geometry.origin_tangent(),
                        &parameter_tangent,
                    )?;
                    if matches!(
                        condition,
                        crate::canonical::ScalarEllipticCartesianBoundary::Natural(_)
                    ) {
                        let mut physical = vec![0.0; dimension];
                        let mut physical_tangent = vec![0.0; dimension];
                        let mut flux_tangent_integral = 0.0;
                        for point in facet_quadrature.points() {
                            geometry.map_point_jvp(
                                &point.coordinates,
                                &mut physical,
                                &mut physical_tangent,
                            )?;
                            let (_, flux, flux_tangent) = model.boundary_jvp(
                                axis,
                                side,
                                &physical,
                                &physical_tangent,
                                &parameter_tangent,
                            )?;
                            flux_tangent_integral += point.weight
                                * (geometry.measure_scale_tangent() * flux
                                    + map.measure_scale() * flux_tangent);
                        }
                        design_jacobian[cell_index * design_dimension + coordinate] -=
                            flux_tangent_integral;
                        continue;
                    }
                    let delta = boundary_coordinates[normal_axis]
                        - solution.cell_centers()[cell_index][normal_axis];
                    let distance = delta.abs();
                    require_positive_distance(distance)?;
                    let distance_tangent = delta.signum()
                        * (geometry.origin_tangent()[normal_axis]
                            - cell_geometry.origin_tangent()[normal_axis]);
                    let transmissibility = diffusion * area / distance;
                    let transmissibility_tangent = transmissibility_jvp(
                        diffusion,
                        diffusion_tangent,
                        area,
                        area_tangent,
                        distance,
                        distance_tangent,
                    )?;
                    design_jacobian[cell_index * design_dimension + coordinate] +=
                        transmissibility_tangent * (solution.cell_values()[cell_index] - boundary)
                            - transmissibility * boundary_tangent;
                }
                _ => {
                    return Err(invalid(
                        "Cartesian facet requires exactly one or two adjacent cells",
                    ));
                }
            }
        }
        parameter_tangent.fill(0.0);
    }
    ensure_finite_design_assembly(&design_jacobian)?;
    AssembledLinearizedRelation::from_shared_canonical(
        Arc::clone(solution.canonical_system()),
        solution.cell_values().to_vec(),
        selected.coordinates,
        selected.values,
        design_jacobian,
    )
}

/// Linearize the complete Q1 Field published by one accepted FEM solution.
///
/// Free vertices project their algebraic tangent directly. Eliminated
/// essential-boundary vertices retain the direct Parameter/geometry action of
/// their canonical trace expression, so the output remains the full semantic
/// Field rather than the shorter method-native unknown vector.
///
/// # Errors
/// Returns a numerical/lowering diagnostic for a mismatched solution,
/// unsupported design coordinate, boundary action, or non-finite projection.
pub fn linearize_scalar_elliptic_cartesian_fem_output(
    model: &ScalarEllipticCartesianModel,
    mesh: &CartesianMesh,
    solution: &ScalarEllipticCartesianFemSolution,
    selected_coordinates: &[SpatialDesignCoordinate],
) -> Result<FieldCoefficientLinearization, Diagnostic> {
    linearize_fem_output_point(model, mesh, &solution.point, selected_coordinates)
}

fn linearize_fem_output_point(
    model: &ScalarEllipticCartesianModel,
    mesh: &CartesianMesh,
    solution: &CartesianFemPoint,
    selected_coordinates: &[SpatialDesignCoordinate],
) -> Result<FieldCoefficientLinearization, Diagnostic> {
    if solution.field().mesh() != mesh {
        return Err(invalid(
            "Cartesian FEM output linearization requires the accepted solution mesh",
        ));
    }
    let selected = select_design_coordinates(model, selected_coordinates)?;
    let output_dimension = solution.field().vertex_values().len();
    let parameter_dimension = selected.coordinates.len();
    let unknown_dimension = solution.algebraic_values().len();
    let mut output_unknowns = vec![None; output_dimension];
    for (dof, &vertex) in solution.free_vertices().iter().enumerate() {
        let Some(slot) = output_unknowns.get_mut(vertex) else {
            return Err(invalid(
                "Cartesian FEM output contains an invalid free vertex",
            ));
        };
        if slot.replace(dof).is_some() {
            return Err(invalid(
                "Cartesian FEM output contains a duplicate free vertex",
            ));
        }
    }

    let mut direct_parameter_jacobian = vec![0.0; output_dimension * parameter_dimension];
    let mut parameter_tangent = vec![0.0; model.parameter_fields().len()];
    for (column, action) in selected.actions.iter().copied().enumerate() {
        activate_model_parameter(action, &mut parameter_tangent);
        for (vertex, unknown) in output_unknowns.iter().enumerate() {
            if unknown.is_some() {
                continue;
            }
            let geometry = design_geometry(mesh, MeshEntity::new(0, vertex), action)?;
            let (_, tangent) = model.essential_boundary_jvp(
                geometry.map().origin(),
                geometry.origin_tangent(),
                &parameter_tangent,
            )?;
            direct_parameter_jacobian[vertex * parameter_dimension + column] = tangent;
        }
        parameter_tangent.fill(0.0);
    }
    FieldCoefficientLinearization::new(
        solution.field().vertex_values().to_vec(),
        unknown_dimension,
        parameter_dimension,
        output_unknowns,
        direct_parameter_jacobian,
    )
}

/// Linearize the complete cell-centred Field published by one accepted FVM
/// solution.
///
/// The current TPFA primary output is exactly its algebraic cell vector; its
/// geometry and Parameter dependence therefore arrives through the implicit
/// state action, with no direct `O_p` term.
///
/// # Errors
/// Returns a numerical/lowering diagnostic for a mismatched solution or
/// unsupported design coordinate.
pub fn linearize_scalar_elliptic_cartesian_fvm_output(
    model: &ScalarEllipticCartesianModel,
    mesh: &CartesianMesh,
    solution: &ScalarEllipticCartesianFvmSolution,
    selected_coordinates: &[SpatialDesignCoordinate],
) -> Result<FieldCoefficientLinearization, Diagnostic> {
    linearize_fvm_output_point(model, mesh, &solution.point, selected_coordinates)
}

fn linearize_fvm_output_point(
    model: &ScalarEllipticCartesianModel,
    mesh: &CartesianMesh,
    solution: &CartesianFvmPoint,
    selected_coordinates: &[SpatialDesignCoordinate],
) -> Result<FieldCoefficientLinearization, Diagnostic> {
    if solution.mesh() != mesh {
        return Err(invalid(
            "Cartesian FVM output linearization requires the accepted solution mesh",
        ));
    }
    let selected = select_design_coordinates(model, selected_coordinates)?;
    let output_dimension = solution.cell_values().len();
    let parameter_dimension = selected.coordinates.len();
    FieldCoefficientLinearization::new(
        solution.cell_values().to_vec(),
        output_dimension,
        parameter_dimension,
        (0..output_dimension).map(Some).collect(),
        vec![0.0; output_dimension * parameter_dimension],
    )
}
