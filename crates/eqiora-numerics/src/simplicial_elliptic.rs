//! P1 scalar elliptic realization on fixed-connectivity affine simplex meshes.

use eqiora_assembly::{
    AssemblyBackend, AssemblyPacket, AssemblyPlan, AssemblyReport, AssemblyTarget, DofId,
    IndexedAssemblyWork, LinearSystem, LocalContribution, REFERENCE_ASSEMBLY_BACKEND,
    TargetAssemblyMap,
};
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_ir::ScalarObjectiveLinearization;
use eqiora_meshing::{
    AffineGeometryLinearization, AffineGeometryMap, GeometryMap, MeshEntity, MeshGeometry,
    MeshTopology, QuadratureRule, SimplicialMesh,
};
use eqiora_solver::{LinearOperatorProperties, LinearProblem, LinearSolveRequest, SolveReport};

use crate::affine_fem::{dot, physical_gradient, weighted_gradient, weighted_gradient_tangent};
use crate::assembled_linearization::AssembledLinearizedRelation;
use crate::canonical::ScalarEllipticCartesianModel;
use crate::constrained_dofs::ConstrainedDofLayout;
use crate::discrete_space::{DiscreteSpace, SimplexP1Space};
use crate::operator::LocalOperator;
use crate::simplicial_motion::SimplicialMeshVelocity;
use crate::spatial_design::SpatialDesignCoordinate;

/// Continuous scalar P1 field on an affine simplex mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct SimplicialP1Field {
    mesh: SimplicialMesh,
    vertex_values: Vec<f64>,
}

impl SimplicialP1Field {
    /// Construct one finite nodal value per mesh vertex.
    ///
    /// # Errors
    /// Returns `EQ0801` for a shape mismatch or non-finite value.
    pub fn new(mesh: SimplicialMesh, vertex_values: Vec<f64>) -> Result<Self, Diagnostic> {
        if vertex_values.len() != mesh.vertices().len()
            || vertex_values.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                "simplicial P1 field requires one finite value per mesh vertex",
            ));
        }
        Ok(Self {
            mesh,
            vertex_values,
        })
    }

    /// Mesh carrying this field.
    #[must_use]
    pub const fn mesh(&self) -> &SimplicialMesh {
        &self.mesh
    }

    /// Nodal values in canonical mesh-vertex order.
    #[must_use]
    pub fn vertex_values(&self) -> &[f64] {
        &self.vertex_values
    }
}

/// Accepted P1 FEM solution and conservation/solver evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarEllipticSimplicialFemSolution {
    field: SimplicialP1Field,
    free_vertices: Vec<usize>,
    algebraic_values: Vec<f64>,
    linear_system: LinearSystem,
    boundary_reaction_sum: f64,
    integrated_source: f64,
    assembly_report: AssemblyReport,
    solve_report: SolveReport,
}

impl ScalarEllipticSimplicialFemSolution {
    /// Continuous P1 result field.
    #[must_use]
    pub const fn field(&self) -> &SimplicialP1Field {
        &self.field
    }

    /// Free-vertex values in assembled equation order.
    #[must_use]
    pub fn algebraic_values(&self) -> &[f64] {
        &self.algebraic_values
    }

    /// Sum of residual reactions on all eliminated boundary vertices.
    #[must_use]
    pub const fn boundary_reaction_sum(&self) -> f64 {
        self.boundary_reaction_sum
    }

    /// Source integral represented by the complete cell load vectors.
    #[must_use]
    pub const fn integrated_source(&self) -> f64 {
        self.integrated_source
    }

    /// Complete assembly placement and accepted packet-shape evidence.
    #[must_use]
    pub const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    /// Complete backend/solver evidence.
    #[must_use]
    pub const fn solve_report(&self) -> &SolveReport {
        &self.solve_report
    }
}

/// Solve one lowered canonical box problem on an explicitly supplied simplex mesh.
///
/// Numerical method, mesh, quadrature, and solver remain realization data. The
/// mesh boundary must conform to the canonical box boundary, but cells and
/// connectivity need not carry Cartesian indexing.
///
/// # Errors
/// Returns a structured lowering/numerical diagnostic for incompatible model,
/// mesh boundary, quadrature, local operator, assembly, or solve data.
pub fn solve_scalar_elliptic_simplicial_fem(
    model: &ScalarEllipticCartesianModel,
    mesh: &SimplicialMesh,
    quadrature: &QuadratureRule,
    solver: LinearSolveRequest<'_>,
) -> Result<ScalarEllipticSimplicialFemSolution, Diagnostic> {
    solve_scalar_elliptic_simplicial_fem_with_assembly(
        model,
        mesh,
        quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        solver,
    )
}

/// Solve affine-simplex P1 FEM through an explicit ordered assembly backend.
///
/// Each canonical cell evaluates its local operator exactly once. That packet
/// feeds both the reduced solve system and the full reaction system, just as
/// in the generated-Cartesian FEM path; imported topology and artifact
/// identity remain validated before this L2 execution boundary.
///
/// # Errors
/// Preserves the reference entry point diagnostics and the selected assembly
/// backend's complete-operation diagnostics.
pub fn solve_scalar_elliptic_simplicial_fem_with_assembly(
    model: &ScalarEllipticCartesianModel,
    mesh: &SimplicialMesh,
    quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend,
    solver: LinearSolveRequest<'_>,
) -> Result<ScalarEllipticSimplicialFemSolution, Diagnostic> {
    validate_problem(model, mesh, quadrature)?;
    let dimension = mesh.topological_dimension();
    let vertex_count = mesh.vertices().len();
    let zero_parameter_tangent = vec![0.0; model.parameter_fields().len()];
    let zero_coordinate_tangent = vec![0.0; dimension];
    let mut fixed_values = Vec::with_capacity(vertex_count);
    for vertex in 0..vertex_count {
        let entity = MeshEntity::new(0, vertex);
        if mesh
            .is_boundary_entity(entity)
            .expect("mesh owns every vertex boundary classification")
        {
            let (value, tangent) = model.essential_boundary_jvp(
                &mesh.vertices()[vertex],
                &zero_coordinate_tangent,
                &zero_parameter_tangent,
            )?;
            debug_assert_eq!(tangent, 0.0);
            fixed_values.push(Some(value));
        } else {
            fixed_values.push(None);
        }
    }
    let constrained_dofs = ConstrainedDofLayout::new(fixed_values)?;
    if constrained_dofs.free_count() == 0 {
        return Err(invalid(
            "simplicial P1 system requires at least one unconstrained interior vertex",
        ));
    }

    let operator = SimplicialEllipticCell { source: model };
    let cell_count = mesh
        .entity_count(dimension)
        .expect("mesh owns its top stratum");
    let assembly_plan = AssemblyPlan::new(vec![
        AssemblyTarget::new(constrained_dofs.free_count())?,
        AssemblyTarget::new(vertex_count)?,
    ])?;
    let reduced_target = assembly_plan
        .target_id(0)
        .expect("two-target simplex assembly plan owns its reduced target");
    let full_target = assembly_plan
        .target_id(1)
        .expect("two-target simplex assembly plan owns its full target");
    let work = IndexedAssemblyWork::new(cell_count, |cell_index| {
        let cell = MeshEntity::new(dimension, cell_index);
        let geometry = mesh
            .geometry_map(cell)
            .expect("accepted simplex cell owns affine geometry");
        let local = operator.evaluate(&geometry, quadrature)?;
        let vertices = mesh
            .entity_vertices(cell)
            .expect("accepted simplex cell owns a vertex closure");
        let global_dofs = vertices
            .iter()
            .map(|vertex| vertex.index())
            .collect::<Vec<_>>();
        let reduced = constrained_dofs.reduced_map(&global_dofs)?;
        let full = constrained_dofs.full_map(&global_dofs)?;
        AssemblyPacket::new(
            local,
            vec![
                TargetAssemblyMap::new(reduced_target, reduced),
                TargetAssemblyMap::new(full_target, full),
            ],
        )
    });
    let (systems, assembly_report) = assembly.assemble(&assembly_plan, &work)?.into_parts();
    let mut systems = systems.into_iter();
    let reduced_system = systems
        .next()
        .expect("two-target simplex assembly returns its reduced system");
    let full_system = systems
        .next()
        .expect("two-target simplex assembly returns its full system");
    debug_assert!(systems.next().is_none());

    let integrated_source = full_system.rhs().iter().sum::<f64>();
    let solved = solver.solve(&LinearProblem::new(
        reduced_system.matrix(),
        reduced_system.rhs(),
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )?)?;
    let values = constrained_dofs.lift(solved.values())?;
    let equilibrium = constrained_dofs.full_residual(&full_system, &values)?;
    let boundary_reaction_sum = constrained_dofs.reaction_sum::<1>(&equilibrium)?[0];
    if !integrated_source.is_finite() || !boundary_reaction_sum.is_finite() {
        return Err(invalid(
            "simplicial FEM source or boundary-reaction evidence is non-finite",
        ));
    }
    let free_vertices = constrained_dofs.free_globals();
    Ok(ScalarEllipticSimplicialFemSolution {
        field: SimplicialP1Field::new(mesh.clone(), values)?,
        free_vertices,
        algebraic_values: solved.values().to_vec(),
        linear_system: reduced_system,
        boundary_reaction_sum,
        integrated_source,
        assembly_report,
        solve_report: solved.report().clone(),
    })
}

/// Assemble `R_w` and selected `R_p` actions for an accepted simplex P1 solve.
///
/// Model Parameters and Domain coordinates share one selected order. Domain
/// coordinates require an explicit realization-local vertex velocity; no mesh
/// motion is inferred from vertex IDs.
///
/// # Errors
/// Returns a diagnostic for an unaccepted/mismatched primal point, missing or
/// duplicate design action, invalid geometry action, or non-finite assembly.
pub fn linearize_scalar_elliptic_simplicial_fem(
    model: &ScalarEllipticCartesianModel,
    mesh: &SimplicialMesh,
    solution: &ScalarEllipticSimplicialFemSolution,
    quadrature: &QuadratureRule,
    selected_coordinates: &[SpatialDesignCoordinate],
    mesh_velocities: &[SimplicialMeshVelocity],
) -> Result<AssembledLinearizedRelation, Diagnostic> {
    validate_linearization_inputs(model, mesh, solution, quadrature)?;
    let selected = select_design(model, mesh, selected_coordinates, mesh_velocities)?;
    let dimension = mesh.topological_dimension();
    let design_dimension = selected.coordinates.len();
    let unknown_dimension = solution.algebraic_values.len();
    let free_indices = free_index_map(mesh, &solution.free_vertices)?;
    let space = SimplexP1Space::new(dimension)?;
    let mut design_jacobian = vec![0.0; unknown_dimension * design_dimension];
    let mut parameter_tangent = vec![0.0; model.parameter_fields().len()];

    for (coordinate, action) in selected.actions.iter().enumerate() {
        activate_parameter(action, &mut parameter_tangent);
        for cell_index in 0..mesh.entity_count(dimension).expect("mesh owns cells") {
            let cell = MeshEntity::new(dimension, cell_index);
            let geometry = design_geometry(mesh, cell, action)?;
            let map = geometry.map();
            require_geometry_rule(map, quadrature)?;
            let inverse = map.inverse_jacobian()?;
            let inverse_tangent = geometry.inverse_jacobian_tangent()?;
            let vertices = mesh
                .entity_vertices(cell)
                .expect("accepted simplex cell owns vertices");
            let fixed_tangents = vertices
                .iter()
                .map(|vertex| {
                    if free_indices[vertex.index()].is_some() {
                        Ok(0.0)
                    } else {
                        let vertex_geometry = design_geometry(mesh, *vertex, action)?;
                        model
                            .essential_boundary_jvp(
                                vertex_geometry.map().origin(),
                                vertex_geometry.origin_tangent(),
                                &parameter_tangent,
                            )
                            .map(|(_, tangent)| tangent)
                    }
                })
                .collect::<Result<Vec<_>, Diagnostic>>()?;
            let accepted_values = vertices
                .iter()
                .map(|vertex| solution.field.vertex_values[vertex.index()])
                .collect::<Vec<_>>();
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
                let gradients = basis
                    .reference_gradients()
                    .chunks_exact(dimension)
                    .map(|gradient| physical_gradient(gradient, &inverse, dimension))
                    .collect::<Vec<_>>();
                let gradient_tangents = basis
                    .reference_gradients()
                    .chunks_exact(dimension)
                    .map(|gradient| physical_gradient(gradient, &inverse_tangent, dimension))
                    .collect::<Vec<_>>();
                let state_gradient = weighted_gradient(&accepted_values, &gradients);
                let state_gradient_tangent = weighted_gradient_tangent(
                    &accepted_values,
                    &fixed_tangents,
                    &gradients,
                    &gradient_tangents,
                );
                for (local_row, vertex) in vertices.iter().enumerate() {
                    let Some(global_row) = free_indices[vertex.index()] else {
                        continue;
                    };
                    let gradient_product = dot(&gradients[local_row], &state_gradient);
                    let energy = diffusion * gradient_product;
                    let energy_tangent = diffusion_tangent * gradient_product
                        + diffusion
                            * (dot(&gradient_tangents[local_row], &state_gradient)
                                + dot(&gradients[local_row], &state_gradient_tangent));
                    let load = source * basis.values()[local_row];
                    let load_tangent = source_tangent * basis.values()[local_row];
                    design_jacobian[global_row.index() * design_dimension + coordinate] +=
                        scale_tangent * (energy - load) + scale * (energy_tangent - load_tangent);
                }
            }
        }
        parameter_tangent.fill(0.0);
    }
    ensure_finite(&design_jacobian, "simplicial residual design action")?;
    AssembledLinearizedRelation::new(
        solution.linear_system.matrix().clone(),
        solution.algebraic_values.clone(),
        solution.linear_system.rhs().to_vec(),
        selected.coordinates,
        selected.values,
        design_jacobian,
    )
}

/// Lower the compliance functional `J_h = integral(source * u_h) dx` and its
/// accepted-point `J_w`/direct `J_p` actions.
///
/// The integral is a continuous-field objective represented by the declared
/// quadrature. It shares geometry, basis, source tape, selected-coordinate
/// order, and boundary elimination with the residual lowering.
///
/// # Errors
/// Returns a diagnostic for an incompatible accepted point/action or a
/// non-finite objective accumulation.
pub fn linearize_scalar_elliptic_simplicial_compliance(
    model: &ScalarEllipticCartesianModel,
    mesh: &SimplicialMesh,
    solution: &ScalarEllipticSimplicialFemSolution,
    quadrature: &QuadratureRule,
    selected_coordinates: &[SpatialDesignCoordinate],
    mesh_velocities: &[SimplicialMeshVelocity],
) -> Result<ScalarObjectiveLinearization, Diagnostic> {
    validate_linearization_inputs(model, mesh, solution, quadrature)?;
    let selected = select_design(model, mesh, selected_coordinates, mesh_velocities)?;
    let dimension = mesh.topological_dimension();
    let design_dimension = selected.coordinates.len();
    let free_indices = free_index_map(mesh, &solution.free_vertices)?;
    let space = SimplexP1Space::new(dimension)?;
    let mut value = 0.0;
    let mut unknown_cotangent = vec![0.0; solution.algebraic_values.len()];

    for cell_index in 0..mesh.entity_count(dimension).expect("mesh owns cells") {
        let cell = MeshEntity::new(dimension, cell_index);
        let map = mesh
            .geometry_map(cell)
            .expect("accepted simplex cell owns geometry");
        let vertices = mesh
            .entity_vertices(cell)
            .expect("accepted simplex cell owns vertices");
        let accepted_values = vertices
            .iter()
            .map(|vertex| solution.field.vertex_values[vertex.index()])
            .collect::<Vec<_>>();
        let mut physical = vec![0.0; dimension];
        for point in quadrature.points() {
            let basis = space.tabulate(&point.coordinates)?;
            map.map_point(&point.coordinates, &mut physical)?;
            let source = model.source().evaluate(&physical)?;
            let field_value = dot(basis.values(), &accepted_values);
            let scale = point.weight * map.measure_scale();
            value += scale * source * field_value;
            for (local, vertex) in vertices.iter().enumerate() {
                if let Some(global) = free_indices[vertex.index()] {
                    unknown_cotangent[global.index()] += scale * source * basis.values()[local];
                }
            }
        }
    }

    let mut parameter_cotangent = vec![0.0; design_dimension];
    let mut parameter_tangent = vec![0.0; model.parameter_fields().len()];
    for (coordinate, action) in selected.actions.iter().enumerate() {
        activate_parameter(action, &mut parameter_tangent);
        for cell_index in 0..mesh.entity_count(dimension).expect("mesh owns cells") {
            let cell = MeshEntity::new(dimension, cell_index);
            let geometry = design_geometry(mesh, cell, action)?;
            let vertices = mesh
                .entity_vertices(cell)
                .expect("accepted simplex cell owns vertices");
            let accepted_values = vertices
                .iter()
                .map(|vertex| solution.field.vertex_values[vertex.index()])
                .collect::<Vec<_>>();
            let fixed_tangents = vertices
                .iter()
                .map(|vertex| {
                    if free_indices[vertex.index()].is_some() {
                        Ok(0.0)
                    } else {
                        let vertex_geometry = design_geometry(mesh, *vertex, action)?;
                        model
                            .essential_boundary_jvp(
                                vertex_geometry.map().origin(),
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
                let (source, source_tangent) =
                    model.source_jvp(&physical, &physical_tangent, &parameter_tangent)?;
                let field_value = dot(basis.values(), &accepted_values);
                let field_direct = dot(basis.values(), &fixed_tangents);
                let scale = point.weight * geometry.map().measure_scale();
                let scale_tangent = point.weight * geometry.measure_scale_tangent();
                parameter_cotangent[coordinate] += scale_tangent * source * field_value
                    + scale * (source_tangent * field_value + source * field_direct);
            }
        }
        parameter_tangent.fill(0.0);
    }
    ensure_finite(&[value], "simplicial compliance value")?;
    ensure_finite(&unknown_cotangent, "simplicial compliance state cotangent")?;
    ensure_finite(
        &parameter_cotangent,
        "simplicial compliance design cotangent",
    )?;
    ScalarObjectiveLinearization::new(value, unknown_cotangent, parameter_cotangent)
}

struct SimplicialEllipticCell<'a> {
    source: &'a ScalarEllipticCartesianModel,
}

impl LocalOperator<AffineGeometryMap> for SimplicialEllipticCell<'_> {
    fn evaluate(
        &self,
        geometry: &AffineGeometryMap,
        quadrature: &QuadratureRule,
    ) -> Result<LocalContribution, Diagnostic> {
        require_geometry_rule(geometry, quadrature)?;
        let dimension = geometry.reference_cell().dimension();
        let space = SimplexP1Space::new(dimension)?;
        let dof_count = dimension + 1;
        let inverse = geometry.inverse_jacobian()?;
        let mut matrix = vec![0.0; dof_count * dof_count];
        let mut rhs = vec![0.0; dof_count];
        let mut physical = vec![0.0; dimension];
        for point in quadrature.points() {
            let basis = space.tabulate(&point.coordinates)?;
            geometry.map_point(&point.coordinates, &mut physical)?;
            let diffusion = self.source.coefficient_expression().evaluate(&physical)?;
            if !diffusion.is_finite() || diffusion <= 0.0 {
                return Err(invalid(
                    "simplicial scalar diffusion coefficient must be finite and positive",
                ));
            }
            let source = self.source.source().evaluate(&physical)?;
            let scale = point.weight * geometry.measure_scale();
            let gradients = basis
                .reference_gradients()
                .chunks_exact(dimension)
                .map(|gradient| physical_gradient(gradient, &inverse, dimension))
                .collect::<Vec<_>>();
            for row in 0..dof_count {
                rhs[row] += scale * source * basis.values()[row];
                for column in 0..dof_count {
                    matrix[row * dof_count + column] +=
                        scale * diffusion * dot(&gradients[row], &gradients[column]);
                }
            }
        }
        LocalContribution::new(dof_count, dof_count, matrix, rhs)
    }
}

#[derive(Clone, Copy)]
enum DesignAction<'a> {
    ModelParameter(usize),
    MeshVelocity(&'a SimplicialMeshVelocity),
}

struct SelectedDesign<'a> {
    coordinates: Vec<SpatialDesignCoordinate>,
    values: Vec<f64>,
    actions: Vec<DesignAction<'a>>,
}

fn select_design<'a>(
    model: &ScalarEllipticCartesianModel,
    mesh: &SimplicialMesh,
    selected: &[SpatialDesignCoordinate],
    velocities: &'a [SimplicialMeshVelocity],
) -> Result<SelectedDesign<'a>, Diagnostic> {
    if selected.is_empty()
        || selected
            .iter()
            .enumerate()
            .any(|(index, coordinate)| selected[..index].contains(coordinate))
    {
        return Err(invalid(
            "simplicial differentiation requires unique explicitly selected coordinates",
        ));
    }
    if velocities.iter().enumerate().any(|(index, velocity)| {
        velocities[..index]
            .iter()
            .any(|previous| previous.coordinate() == velocity.coordinate())
    }) {
        return Err(invalid(
            "simplicial differentiation contains duplicate mesh velocities",
        ));
    }
    let mut values = Vec::with_capacity(selected.len());
    let mut actions = Vec::with_capacity(selected.len());
    for coordinate in selected {
        match *coordinate {
            SpatialDesignCoordinate::ModelParameter {
                parameter,
                component,
                imaginary,
            } => {
                if component != 0 || imaginary {
                    return Err(invalid(
                        "simplicial scalar Parameter requires its real scalar coordinate",
                    ));
                }
                let Some(index) = model
                    .parameter_fields()
                    .iter()
                    .position(|candidate| candidate == &parameter)
                else {
                    return Err(invalid(
                        "selected model Parameter does not affect the lowered simplicial relation",
                    ));
                };
                if velocities
                    .iter()
                    .any(|velocity| velocity.coordinate() == *coordinate)
                {
                    return Err(invalid(
                        "model Parameter action cannot also carry a mesh velocity",
                    ));
                }
                values.push(model.parameter_values()[index]);
                actions.push(DesignAction::ModelParameter(index));
            }
            SpatialDesignCoordinate::CartesianBound { domain, axis, side } => {
                if domain != model.domain_id() || axis >= model.dimension() {
                    return Err(invalid(
                        "selected Domain bound does not belong to the lowered simplicial relation",
                    ));
                }
                let Some(velocity) = velocities
                    .iter()
                    .find(|velocity| velocity.coordinate() == *coordinate)
                else {
                    return Err(invalid(
                        "selected Domain coordinate requires one explicit simplex mesh velocity",
                    ));
                };
                if velocity.vertex_velocities().len() != mesh.vertices().len() {
                    return Err(invalid(
                        "selected simplex mesh velocity does not match the accepted mesh",
                    ));
                }
                values.push(match side {
                    eqiora_schema::kernel::BoundarySide::Lower => model.bounds()[axis][0],
                    eqiora_schema::kernel::BoundarySide::Upper => model.bounds()[axis][1],
                });
                actions.push(DesignAction::MeshVelocity(velocity));
            }
        }
    }
    if velocities
        .iter()
        .any(|velocity| !selected.contains(&velocity.coordinate()))
    {
        return Err(invalid(
            "unselected simplex mesh velocity cannot enter a derivative action",
        ));
    }
    Ok(SelectedDesign {
        coordinates: selected.to_vec(),
        values,
        actions,
    })
}

fn design_geometry(
    mesh: &SimplicialMesh,
    entity: MeshEntity,
    action: &DesignAction<'_>,
) -> Result<AffineGeometryLinearization, Diagnostic> {
    match action {
        DesignAction::ModelParameter(_) => AffineGeometryLinearization::stationary(
            mesh.geometry_map(entity)
                .ok_or_else(|| invalid("simplicial entity geometry is unavailable"))?,
        ),
        DesignAction::MeshVelocity(velocity) => {
            mesh.linearized_geometry_map(entity, velocity.vertex_velocities())
        }
    }
}

fn activate_parameter(action: &DesignAction<'_>, tangent: &mut [f64]) {
    if let DesignAction::ModelParameter(index) = action {
        tangent[*index] = 1.0;
    }
}

fn validate_problem(
    model: &ScalarEllipticCartesianModel,
    mesh: &SimplicialMesh,
    quadrature: &QuadratureRule,
) -> Result<(), Diagnostic> {
    if model.dimension() != mesh.topological_dimension()
        || mesh.geometric_dimension() != mesh.topological_dimension()
    {
        return Err(invalid(
            "simplicial FEM requires matching full dimension and a positive coefficient",
        ));
    }
    let expected = eqiora_meshing::ReferenceCell::simplex(mesh.topological_dimension())?;
    if quadrature.reference_cell() != expected {
        return Err(invalid(
            "simplicial FEM quadrature does not match the cell reference simplex",
        ));
    }
    validate_box_conformity(model, mesh)
}

fn validate_box_conformity(
    model: &ScalarEllipticCartesianModel,
    mesh: &SimplicialMesh,
) -> Result<(), Diagnostic> {
    for vertex in mesh.vertices() {
        if vertex
            .iter()
            .zip(model.bounds())
            .any(|(coordinate, bounds)| {
                let tolerance =
                    256.0 * f64::EPSILON * bounds[0].abs().max(bounds[1].abs()).max(1.0);
                *coordinate < bounds[0] - tolerance || *coordinate > bounds[1] + tolerance
            })
        {
            return Err(invalid(
                "simplicial mesh vertex lies outside the canonical Cartesian Domain",
            ));
        }
    }
    let facet_dimension = mesh.topological_dimension() - 1;
    for facet_index in 0..mesh
        .entity_count(facet_dimension)
        .expect("mesh owns facets")
    {
        let facet = MeshEntity::new(facet_dimension, facet_index);
        if !mesh
            .is_boundary_entity(facet)
            .expect("mesh facet has boundary classification")
        {
            continue;
        }
        let vertices = mesh
            .entity_vertices(facet)
            .expect("mesh facet owns vertices");
        let lies_on_box_side = (0..model.dimension()).any(|axis| {
            [model.bounds()[axis][0], model.bounds()[axis][1]]
                .into_iter()
                .any(|bound| {
                    vertices.iter().all(|vertex| {
                        mesh.vertices()[vertex.index()][axis].to_bits() == bound.to_bits()
                    })
                })
        });
        if !lies_on_box_side {
            return Err(invalid(
                "simplicial boundary facet does not lie on one canonical box side",
            ));
        }
    }
    Ok(())
}

fn validate_linearization_inputs(
    model: &ScalarEllipticCartesianModel,
    mesh: &SimplicialMesh,
    solution: &ScalarEllipticSimplicialFemSolution,
    quadrature: &QuadratureRule,
) -> Result<(), Diagnostic> {
    validate_problem(model, mesh, quadrature)?;
    if solution.field.mesh() != mesh {
        return Err(invalid(
            "simplicial linearization requires the exact accepted mesh revision",
        ));
    }
    Ok(())
}

fn free_index_map(
    mesh: &SimplicialMesh,
    free_vertices: &[usize],
) -> Result<Vec<Option<DofId>>, Diagnostic> {
    let mut indices = vec![None; mesh.vertices().len()];
    for (dof, &vertex) in free_vertices.iter().enumerate() {
        if vertex >= mesh.vertices().len()
            || mesh
                .is_boundary_entity(MeshEntity::new(0, vertex))
                .unwrap_or(true)
            || indices[vertex].replace(DofId::new(dof)).is_some()
        {
            return Err(invalid(
                "simplicial solution has an invalid free-vertex layout",
            ));
        }
    }
    if free_vertices.len() != mesh.vertices().len() - mesh_boundary_vertex_count(mesh) {
        return Err(invalid(
            "simplicial solution free-vertex layout is incomplete",
        ));
    }
    Ok(indices)
}

fn mesh_boundary_vertex_count(mesh: &SimplicialMesh) -> usize {
    (0..mesh.vertices().len())
        .filter(|&vertex| {
            mesh.is_boundary_entity(MeshEntity::new(0, vertex))
                .expect("mesh owns vertices")
        })
        .count()
}

fn require_geometry_rule(
    geometry: &AffineGeometryMap,
    quadrature: &QuadratureRule,
) -> Result<(), Diagnostic> {
    if geometry.reference_cell() != quadrature.reference_cell()
        || geometry.physical_dimension() != geometry.reference_cell().dimension()
    {
        Err(invalid(
            "simplicial local geometry, physical dimension, and quadrature differ",
        ))
    } else {
        Ok(())
    }
}

fn ensure_finite(values: &[f64], name: &str) -> Result<(), Diagnostic> {
    if values.iter().any(|value| !value.is_finite()) {
        Err(invalid(format!("{name} produced a non-finite value")))
    } else {
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_LINEARIZATION, message)
}
