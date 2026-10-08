//! Two-dimensional Cartesian Q1 realization of isotropic linear elasticity.
//!
//! This module owns one deliberately narrow numerical realization. Material
//! coefficients and the conservative load potential arrive from canonical
//! lowering; mesh, quadrature, assembly, and solve policy remain explicit
//! realization inputs.

use std::num::NonZeroU32;
use std::sync::Arc;

use eqiora_assembly::{
    AssemblyBackend, AssemblyPacket, AssemblyPlan, AssemblyReport, AssemblyTarget,
    IndexedAssemblyWork, LocalContribution, REFERENCE_ASSEMBLY_BACKEND, TargetAssemblyMap,
};
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_ir::LocalLinearActionIr;
use eqiora_meshing::{DiscreteFieldAssociation, DiscreteFieldPayload, DiscreteFieldShape};
use eqiora_meshing::{
    GeometryMap, MeshEntity, MeshGeometry, MeshTopology, QuadratureRule, ReferenceCell,
};
use eqiora_solver::{
    CanonicalCsrSystemView, LinearOperatorProperties, LinearSolution, LinearSolveRequest,
    SolveReport,
};

use crate::affine_fem::physical_gradient;
use crate::constrained_dofs::ConstrainedDofLayout;
use crate::discrete_space::{DiscreteSpace, HypercubeQ1Space};
use crate::form_compiler::compile_cartesian_q1_elasticity_form_2d;
use crate::interleaved_dofs::InterleavedDofValues;
use crate::linear_elasticity::IsotropicElasticityMaterial;
use crate::spatial_expression::ScalarSpatialExpression;
use eqiora_meshing::CartesianMesh;

const DIMENSION: usize = 2;
const COMPONENTS: usize = 2;

mod pair;

pub use pair::{
    ConformingCartesianInterfaceMap2d, ConformingCartesianLinearElasticityPair2dSolution,
    ConformingElasticityInterfaceAction2d,
};
pub(crate) use pair::{
    FinalizedConformingCartesianElasticityPair2dAssembly,
    FinalizedConformingCartesianElasticityPair2dState,
    finalize_conforming_cartesian_q1_linear_elasticity_pair_2d,
};

/// Realization-private selection of complete homogeneous essential sides.
///
/// The Semantic Model owns boundary meaning. The Cartesian Q1 assembler
/// receives this essential selection and separately resolved constant tractions
/// after canonical normalization and Realization validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CartesianEssentialSides2d {
    sides: [[bool; 2]; DIMENSION],
}

impl CartesianEssentialSides2d {
    pub(crate) const fn new(sides: [[bool; 2]; DIMENSION]) -> Self {
        Self { sides }
    }

    const fn complete_boundary() -> Self {
        Self::new([[true; 2]; DIMENSION])
    }

    pub(crate) fn has_essential_side(self) -> bool {
        self.sides.iter().flatten().copied().any(|value| value)
    }

    fn constrains_vertex(self, mesh: &CartesianMesh, vertex: MeshEntity) -> bool {
        let index = mesh
            .vertex_multi_index(vertex)
            .expect("a Cartesian vertex owns one index per physical axis");
        (0..DIMENSION).any(|axis| {
            (self.sides[axis][0] && index[axis] == 0)
                || (self.sides[axis][1]
                    && index[axis]
                        == mesh
                            .axis_coordinates(axis)
                            .expect("a two-dimensional mesh owns both axes")
                            .len()
                            - 1)
        })
    }
}

/// Continuous two-component Q1 field on one two-dimensional Cartesian mesh.
///
/// Values use the repository-wide discrete-field order: mesh vertex first,
/// Cartesian component second. Thus component `c` of vertex `v` occupies
/// `2 * v + c`.
#[derive(Debug, Clone, PartialEq)]
pub struct CartesianQ1VectorField2d {
    mesh: CartesianMesh,
    payload: DiscreteFieldPayload,
}

impl CartesianQ1VectorField2d {
    /// Bind finite entity-major values to a two-dimensional Cartesian mesh.
    ///
    /// # Errors
    /// Returns `EQ0801` unless the mesh is exactly two-dimensional, or the
    /// discrete-field contract rejects the value shape or data.
    pub fn new(mesh: CartesianMesh, values: Vec<f64>) -> Result<Self, Diagnostic> {
        require_two_dimensional_mesh(&mesh)?;
        let payload = DiscreteFieldPayload::new(
            &mesh,
            DiscreteFieldAssociation::Vertex,
            DiscreteFieldShape::Vector {
                components: NonZeroU32::new(COMPONENTS as u32)
                    .expect("the two Cartesian components are nonzero"),
            },
            values,
        )?;
        Ok(Self { mesh, payload })
    }

    /// Mesh carrying the nodal field.
    #[must_use]
    pub const fn mesh(&self) -> &CartesianMesh {
        &self.mesh
    }

    /// Checked entity-major discrete values.
    #[must_use]
    pub const fn payload(&self) -> &DiscreteFieldPayload {
        &self.payload
    }

    /// Flat vertex-major, component-minor values.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        self.payload.values()
    }

    /// Two Cartesian values at one mesh vertex.
    #[must_use]
    pub fn vertex_values(&self, vertex: usize) -> Option<&[f64]> {
        self.payload.entity_values(vertex)
    }

    /// Integrate continuous L2 and H1-seminorm errors in one pass.
    ///
    /// `exact` returns the physical vector and its row-major gradient
    /// `[component][physical_axis]` at one physical point.
    ///
    /// # Errors
    /// Returns a discretization diagnostic for incompatible quadrature,
    /// geometry, or non-finite exact/error data.
    pub fn error_norms<E>(
        &self,
        exact: &E,
        quadrature: &QuadratureRule,
    ) -> Result<CartesianVectorErrorNorms, Diagnostic>
    where
        E: Fn(&[f64]) -> ([f64; COMPONENTS], [[f64; DIMENSION]; COMPONENTS]) + ?Sized,
    {
        require_cell_rule(&self.mesh, quadrature)?;
        let space = HypercubeQ1Space::new(DIMENSION)?;
        let cell_count = self
            .mesh
            .entity_count(DIMENSION)
            .expect("a two-dimensional mesh owns its cell stratum");
        let mut l2_squared = 0.0;
        let mut h1_seminorm_squared = 0.0;
        let mut physical = [0.0; DIMENSION];

        for cell_index in 0..cell_count {
            let cell = MeshEntity::new(DIMENSION, cell_index);
            let geometry = self
                .mesh
                .geometry_map(cell)
                .expect("a Cartesian cell owns affine geometry");
            let inverse = geometry.inverse_jacobian()?;
            let vertices = self
                .mesh
                .entity_vertices(cell)
                .expect("a Cartesian cell owns its vertex closure");
            for point in quadrature.points() {
                let basis = space.tabulate(&point.coordinates)?;
                geometry.map_point(&point.coordinates, &mut physical)?;
                let gradients = physical_gradients(&basis, &inverse);
                let mut value = [0.0; COMPONENTS];
                let mut gradient = [[0.0; DIMENSION]; COMPONENTS];
                for (local_vertex, vertex) in vertices.iter().enumerate() {
                    let nodal = self
                        .vertex_values(vertex.index())
                        .expect("the accepted field owns every mesh vertex");
                    for component in 0..COMPONENTS {
                        value[component] += basis.values()[local_vertex] * nodal[component];
                        for axis in 0..DIMENSION {
                            gradient[component][axis] +=
                                gradients[local_vertex][axis] * nodal[component];
                        }
                    }
                }
                let (exact_value, exact_gradient) = exact(&physical);
                if exact_value
                    .iter()
                    .chain(exact_gradient.iter().flatten())
                    .any(|value| !value.is_finite())
                {
                    return Err(invalid("exact vector field returned a non-finite value"));
                }
                let scale = point.weight * geometry.measure_scale();
                l2_squared += scale
                    * value
                        .iter()
                        .zip(exact_value)
                        .map(|(actual, exact)| (actual - exact).powi(2))
                        .sum::<f64>();
                h1_seminorm_squared += scale
                    * gradient
                        .iter()
                        .flatten()
                        .zip(exact_gradient.iter().flatten())
                        .map(|(actual, exact)| (actual - exact).powi(2))
                        .sum::<f64>();
            }
        }
        CartesianVectorErrorNorms::new(l2_squared, h1_seminorm_squared)
    }
}

/// Continuous vector-field error norms from one explicit quadrature rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CartesianVectorErrorNorms {
    l2: f64,
    h1_seminorm: f64,
}

impl CartesianVectorErrorNorms {
    fn new(l2_squared: f64, h1_seminorm_squared: f64) -> Result<Self, Diagnostic> {
        if !l2_squared.is_finite()
            || !h1_seminorm_squared.is_finite()
            || l2_squared < 0.0
            || h1_seminorm_squared < 0.0
        {
            return Err(invalid(
                "Cartesian vector error accumulation is non-finite or negative",
            ));
        }
        Ok(Self {
            l2: l2_squared.sqrt(),
            h1_seminorm: h1_seminorm_squared.sqrt(),
        })
    }

    /// Continuous vector L2 error.
    #[must_use]
    pub const fn l2(self) -> f64 {
        self.l2
    }

    /// Frobenius-gradient H1 seminorm error.
    #[must_use]
    pub const fn h1_seminorm(self) -> f64 {
        self.h1_seminorm
    }

    /// Complete H1 error.
    #[must_use]
    pub fn h1(self) -> f64 {
        self.l2.hypot(self.h1_seminorm)
    }
}

/// Accepted two-dimensional Q1 elasticity solution and balance evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct CartesianLinearElasticity2dSolution {
    displacement: CartesianQ1VectorField2d,
    algebraic_values: Vec<f64>,
    boundary_reaction: [f64; COMPONENTS],
    integrated_body_force: [f64; COMPONENTS],
    assembly_report: AssemblyReport,
    solve_report: SolveReport,
}

impl CartesianLinearElasticity2dSolution {
    /// Continuous nodal displacement field.
    #[must_use]
    pub const fn displacement(&self) -> &CartesianQ1VectorField2d {
        &self.displacement
    }

    /// Free displacement components in assembled equation order.
    #[must_use]
    pub fn algebraic_values(&self) -> &[f64] {
        &self.algebraic_values
    }

    /// Sum of full-system residual reactions over constrained vertices.
    #[must_use]
    pub const fn boundary_reaction(&self) -> [f64; COMPONENTS] {
        self.boundary_reaction
    }

    /// Integral of `grad(q)` represented by the complete nodal load vector.
    #[must_use]
    pub const fn integrated_body_force(&self) -> [f64; COMPONENTS] {
        self.integrated_body_force
    }

    /// Complete assembly placement and packet-shape evidence.
    #[must_use]
    pub const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    /// Complete solver/backend/execution evidence.
    #[must_use]
    pub const fn solve_report(&self) -> &SolveReport {
        &self.solve_report
    }
}

/// Lower the body-free local stiffness of every Cartesian cell into one
/// shape-homogeneous local-action batch.
///
/// # Errors
/// Returns a discretization diagnostic for a non-2D mesh, non-coercive Lamé
/// coefficients, incompatible quadrature, geometry, or allocation failure.
pub fn lower_cartesian_q1_linear_elasticity_local_action_2d(
    mesh: &CartesianMesh,
    shear_modulus: f64,
    first_lame_parameter: f64,
    quadrature: &QuadratureRule,
) -> Result<LocalLinearActionIr, Diagnostic> {
    let material = admit_problem(mesh, shear_modulus, first_lame_parameter, quadrature)?;
    let cell_count = mesh
        .entity_count(DIMENSION)
        .expect("a two-dimensional mesh owns its cell stratum");
    let local_width = HypercubeQ1Space::new(DIMENSION)?
        .local_dofs()
        .len()
        .checked_mul(COMPONENTS)
        .ok_or_else(|| invalid("elasticity local width overflows usize"))?;
    let coefficient_count = cell_count
        .checked_mul(local_width)
        .and_then(|count| count.checked_mul(local_width))
        .ok_or_else(|| invalid("elasticity local-action shape overflows usize"))?;
    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(coefficient_count)
        .map_err(|_| {
            invalid("elasticity local-action batch exceeds platform allocation capacity")
        })?;
    let operator = compile_cartesian_q1_elasticity_form_2d(quadrature)?;
    for cell_index in 0..cell_count {
        let geometry = mesh
            .geometry_map(MeshEntity::new(DIMENSION, cell_index))
            .expect("a Cartesian cell owns affine geometry");
        let local = operator.evaluate(
            &geometry,
            quadrature,
            material.shear_modulus(),
            material.first_lame_parameter(),
            None,
        )?;
        debug_assert_eq!(local.rows(), local_width);
        debug_assert_eq!(local.columns(), local_width);
        coefficients.extend_from_slice(local.matrix());
    }
    LocalLinearActionIr::new(local_width, local_width, coefficients)
}

/// Solve homogeneous-Dirichlet isotropic elasticity using the deterministic
/// reference assembly backend.
///
/// The conservative body force is not supplied as a parallel vector closure.
/// It is the exact coordinate derivative of the same lowered scalar `q` tape
/// at every quadrature point.
///
/// # Errors
/// Returns a diagnostic for incompatible mesh/material/quadrature data,
/// potential evaluation, assembly, constraints, or solve failure.
pub fn solve_cartesian_q1_linear_elasticity_2d(
    mesh: &CartesianMesh,
    shear_modulus: f64,
    first_lame_parameter: f64,
    body_force_potential: &ScalarSpatialExpression<f64>,
    quadrature: &QuadratureRule,
    solver: LinearSolveRequest<'_>,
) -> Result<CartesianLinearElasticity2dSolution, Diagnostic> {
    solve_cartesian_q1_linear_elasticity_2d_with_assembly(
        mesh,
        shear_modulus,
        first_lame_parameter,
        body_force_potential,
        quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        solver,
    )
}

/// Solve the same problem through one explicit ordered assembly backend.
///
/// # Errors
/// Preserves the reference entry point diagnostics and the selected assembly
/// backend's complete-operation diagnostics.
#[allow(clippy::too_many_arguments)]
pub fn solve_cartesian_q1_linear_elasticity_2d_with_assembly(
    mesh: &CartesianMesh,
    shear_modulus: f64,
    first_lame_parameter: f64,
    body_force_potential: &ScalarSpatialExpression<f64>,
    quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend<f64>,
    solver: LinearSolveRequest<'_>,
) -> Result<CartesianLinearElasticity2dSolution, Diagnostic> {
    let material = admit_problem(mesh, shear_modulus, first_lame_parameter, quadrature)?;
    let assembled = finalize_cartesian_q1_linear_elasticity_2d(
        mesh,
        material,
        body_force_potential,
        quadrature,
        CartesianEssentialSides2d::complete_boundary(),
        [[None; 2]; DIMENSION],
        assembly,
    )?;
    let (canonical_system, state) = assembled.into_canonical()?;
    let solved = solver.solve(&canonical_system.linear_problem()?)?;
    state.finish(solved, canonical_system)
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FinalizedCartesianElasticity2dAssembly {
    mesh: CartesianMesh,
    constrained_dofs: ConstrainedDofLayout<f64>,
    linear_system: eqiora_assembly::LinearSystem<f64>,
    full_system: eqiora_assembly::LinearSystem<f64>,
    integrated_body_force: [f64; COMPONENTS],
    assembly_report: AssemblyReport,
}

impl FinalizedCartesianElasticity2dAssembly {
    pub(crate) fn into_canonical(
        self,
    ) -> Result<
        (
            Arc<CanonicalCsrSystemView>,
            FinalizedCartesianElasticity2dState,
        ),
        Diagnostic,
    > {
        let Self {
            mesh,
            constrained_dofs,
            linear_system,
            full_system,
            integrated_body_force,
            assembly_report,
        } = self;
        let canonical_system = Arc::new(CanonicalCsrSystemView::new(
            &linear_system,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )?);
        Ok((
            canonical_system,
            FinalizedCartesianElasticity2dState {
                mesh,
                constrained_dofs,
                full_system,
                integrated_body_force,
                assembly_report,
            },
        ))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FinalizedCartesianElasticity2dState {
    mesh: CartesianMesh,
    constrained_dofs: ConstrainedDofLayout<f64>,
    full_system: eqiora_assembly::LinearSystem<f64>,
    integrated_body_force: [f64; COMPONENTS],
    assembly_report: AssemblyReport,
}

impl FinalizedCartesianElasticity2dState {
    pub(crate) const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    pub(crate) fn finish(
        self,
        solved: LinearSolution,
        canonical_system: Arc<CanonicalCsrSystemView>,
    ) -> Result<CartesianLinearElasticity2dSolution, Diagnostic> {
        if solved.values().len() != canonical_system.rows() {
            return Err(invalid(
                "elasticity solution shape differs from its finalized system",
            ));
        }
        let (algebraic_values, solve_report) = solved.into_parts();
        let displacement = self.constrained_dofs.lift(&algebraic_values)?;
        let residual = self
            .constrained_dofs
            .full_residual(&self.full_system, &displacement)?;
        let boundary_reaction = self
            .constrained_dofs
            .reaction_sum::<COMPONENTS>(&residual)?;
        if boundary_reaction
            .iter()
            .chain(&self.integrated_body_force)
            .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "elasticity reaction or integrated body force is non-finite",
            ));
        }

        Ok(CartesianLinearElasticity2dSolution {
            displacement: CartesianQ1VectorField2d::new(self.mesh, displacement)?,
            algebraic_values,
            boundary_reaction,
            integrated_body_force: self.integrated_body_force,
            assembly_report: self.assembly_report,
            solve_report,
        })
    }
}

pub(crate) fn finalize_cartesian_q1_linear_elasticity_2d(
    mesh: &CartesianMesh,
    material: IsotropicElasticityMaterial<DIMENSION>,
    body_force_potential: &ScalarSpatialExpression<f64>,
    quadrature: &QuadratureRule,
    essential_sides: CartesianEssentialSides2d,
    tractions: [[Option<[f64; COMPONENTS]>; 2]; DIMENSION],
    assembly: &dyn AssemblyBackend<f64>,
) -> Result<FinalizedCartesianElasticity2dAssembly, Diagnostic> {
    require_two_dimensional_mesh(mesh)?;
    require_cell_rule(mesh, quadrature)?;
    if !essential_sides.has_essential_side() {
        return Err(invalid(
            "Cartesian Q1 elasticity requires at least one complete homogeneous essential side to remove rigid modes",
        ));
    }
    if body_force_potential.coordinate_dimension() != DIMENSION {
        return Err(invalid(format!(
            "elasticity body-force potential expects {} coordinates, not {DIMENSION}",
            body_force_potential.coordinate_dimension(),
        )));
    }

    let vertex_count = mesh
        .entity_count(0)
        .expect("a Cartesian mesh owns its vertex stratum");
    let global_width = vertex_count
        .checked_mul(COMPONENTS)
        .ok_or_else(|| invalid("elasticity global DOF count overflows usize"))?;
    let mut fixed_values = vec![Some(0.0); global_width];
    for vertex_index in 0..vertex_count {
        let vertex = MeshEntity::new(0, vertex_index);
        if !essential_sides.constrains_vertex(mesh, vertex) {
            for component in 0..COMPONENTS {
                fixed_values[global_dof(vertex_index, component)?] = None;
            }
        }
    }
    let constrained_dofs = ConstrainedDofLayout::new(fixed_values)?;
    if constrained_dofs.free_count() == 0 {
        return Err(invalid(
            "Cartesian Q1 elasticity requires at least one unconstrained interior vertex",
        ));
    }

    let operator = compile_cartesian_q1_elasticity_form_2d(quadrature)?;
    let faces = prescribed_traction_faces(mesh, &tractions)?;
    let plan = AssemblyPlan::new(vec![
        AssemblyTarget::new(constrained_dofs.free_count())?,
        AssemblyTarget::new(global_width)?,
    ])?;
    let reduced_target = plan
        .target_id(0)
        .expect("the elasticity assembly plan owns its reduced target");
    let full_target = plan
        .target_id(1)
        .expect("the elasticity assembly plan owns its full target");
    let cell_count = mesh
        .entity_count(DIMENSION)
        .expect("a two-dimensional mesh owns its cell stratum");
    let work = IndexedAssemblyWork::new(cell_count + faces.len(), |cell_index: usize| {
        if cell_index >= cell_count {
            let PrescribedTractionFace {
                vertices,
                nodal_load,
            } = &faces[cell_index - cell_count];
            let global_dofs = local_global_dofs(vertices)?;
            let width = global_dofs.len();
            return AssemblyPacket::new(
                LocalContribution::new(
                    width,
                    width,
                    vec![0.0; width * width],
                    vertices.iter().flat_map(|_| *nodal_load).collect(),
                )?,
                vec![
                    TargetAssemblyMap::new(
                        reduced_target,
                        constrained_dofs.reduced_map(&global_dofs)?,
                    ),
                    TargetAssemblyMap::new(full_target, constrained_dofs.full_map(&global_dofs)?),
                ],
            );
        }
        let cell = MeshEntity::new(DIMENSION, cell_index);
        let geometry = mesh
            .geometry_map(cell)
            .expect("a Cartesian cell owns affine geometry");
        let local = operator.evaluate(
            &geometry,
            quadrature,
            material.shear_modulus(),
            material.first_lame_parameter(),
            Some(body_force_potential),
        )?;
        let vertices = mesh
            .entity_vertices(cell)
            .expect("a Cartesian cell owns its vertex closure");
        let global_dofs = local_global_dofs(&vertices)?;
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
    let (systems, assembly_report) = assembly.assemble(&plan, &work)?.into_parts();
    let mut systems = systems.into_iter();
    let reduced_system = systems
        .next()
        .expect("two-target elasticity assembly returns a reduced system");
    let full_system = systems
        .next()
        .expect("two-target elasticity assembly returns a full system");
    debug_assert!(systems.next().is_none());
    let integrated_body_force = if faces.is_empty() {
        InterleavedDofValues::<COMPONENTS>::new(full_system.rhs())?.sum()
    } else {
        // Integrate only the conservative volume load with the same cell rule.
        // Subtracting boundary loads from the combined RHS would lose small
        // body forces through cancellation under large prescribed traction.
        let mut force = [0.0; COMPONENTS];
        for index in 0..cell_count {
            let geometry = mesh
                .geometry_map(MeshEntity::new(DIMENSION, index))
                .unwrap();
            for point in quadrature.points() {
                let mut coordinates = [0.0; DIMENSION];
                geometry.map_point(&point.coordinates, &mut coordinates)?;
                let gradient =
                    body_force_potential.evaluate_gradient::<COMPONENTS>(&coordinates)?;
                for (sum, value) in force.iter_mut().zip(gradient) {
                    *sum += value * point.weight * geometry.measure_scale();
                }
            }
        }
        force
    };
    if integrated_body_force.iter().any(|value| !value.is_finite()) {
        return Err(invalid("integrated elasticity body force is non-finite"));
    }

    Ok(FinalizedCartesianElasticity2dAssembly {
        mesh: mesh.clone(),
        constrained_dofs,
        linear_system: reduced_system,
        full_system,
        integrated_body_force,
        assembly_report,
    })
}

struct PrescribedTractionFace {
    vertices: [MeshEntity; 2],
    nodal_load: [f64; COMPONENTS],
}

// Exact Q1 integration of a constant outward traction on an affine edge:
// each endpoint receives length * traction / 2, including constrained endpoints.
fn prescribed_traction_faces(
    mesh: &CartesianMesh,
    tractions: &[[Option<[f64; COMPONENTS]>; 2]; DIMENSION],
) -> Result<Vec<PrescribedTractionFace>, Diagnostic> {
    let mut faces = Vec::new();
    if tractions.iter().flatten().all(Option::is_none) {
        return Ok(faces);
    }
    for index in 0..mesh.entity_count(1).expect("two-dimensional edge stratum") {
        let edge = MeshEntity::new(1, index);
        if mesh.is_boundary_entity(edge) != Some(true) {
            continue;
        }
        let vertices = mesh.entity_vertices(edge).expect("edge endpoints");
        for (axis, sides) in tractions.iter().enumerate() {
            for (side, traction) in sides.iter().enumerate() {
                let Some(traction) = traction else {
                    continue;
                };
                let endpoint = if side == 0 {
                    0
                } else {
                    mesh.axis_coordinates(axis).unwrap().len() - 1
                };
                if !vertices
                    .iter()
                    .all(|vertex| mesh.vertex_multi_index(*vertex).unwrap()[axis] == endpoint)
                {
                    continue;
                }
                let first = mesh.vertex_coordinates(vertices[0]).unwrap();
                let second = mesh.vertex_coordinates(vertices[1]).unwrap();
                let length = (second[1 - axis] - first[1 - axis]).abs();
                let load = traction.map(|component| component * (length / 2.0));
                if load.iter().any(|value| !value.is_finite()) {
                    return Err(invalid("integrated prescribed traction is non-finite"));
                }
                faces.push(PrescribedTractionFace {
                    vertices: vertices
                        .as_slice()
                        .try_into()
                        .expect("Cartesian edge has two endpoints"),
                    nodal_load: load,
                });
            }
        }
    }
    Ok(faces)
}

fn physical_gradients(
    basis: &crate::discrete_space::BasisTabulation,
    inverse_jacobian: &[f64],
) -> Vec<Vec<f64>> {
    basis
        .reference_gradients()
        .as_chunks::<DIMENSION>()
        .0
        .iter()
        .map(|gradient| physical_gradient(gradient, inverse_jacobian, DIMENSION))
        .collect()
}

fn local_global_dofs(vertices: &[MeshEntity]) -> Result<Vec<usize>, Diagnostic> {
    let capacity = vertices
        .len()
        .checked_mul(COMPONENTS)
        .ok_or_else(|| invalid("elasticity local-to-global shape overflows usize"))?;
    let mut dofs = Vec::new();
    dofs.try_reserve_exact(capacity)
        .map_err(|_| invalid("elasticity local-to-global map exceeds allocation capacity"))?;
    for vertex in vertices {
        for component in 0..COMPONENTS {
            dofs.push(global_dof(vertex.index(), component)?);
        }
    }
    Ok(dofs)
}

fn global_dof(vertex: usize, component: usize) -> Result<usize, Diagnostic> {
    vertex
        .checked_mul(COMPONENTS)
        .and_then(|base| base.checked_add(component))
        .ok_or_else(|| invalid("elasticity global DOF index overflows usize"))
}

fn admit_problem(
    mesh: &CartesianMesh,
    shear_modulus: f64,
    first_lame_parameter: f64,
    quadrature: &QuadratureRule,
) -> Result<IsotropicElasticityMaterial<DIMENSION>, Diagnostic> {
    require_two_dimensional_mesh(mesh)?;
    let material = IsotropicElasticityMaterial::<DIMENSION>::new(
        shear_modulus,
        first_lame_parameter,
    )
    .ok_or_else(|| {
        invalid("two-dimensional isotropic elasticity requires finite mu > 0 and lambda + mu > 0")
    })?;
    require_cell_rule(mesh, quadrature)?;
    Ok(material)
}

fn require_two_dimensional_mesh(mesh: &CartesianMesh) -> Result<(), Diagnostic> {
    if mesh.topological_dimension() != DIMENSION || mesh.geometric_dimension() != DIMENSION {
        Err(invalid(
            "this Cartesian Q1 vector realization requires exactly two dimensions",
        ))
    } else {
        Ok(())
    }
}

fn require_cell_rule(mesh: &CartesianMesh, quadrature: &QuadratureRule) -> Result<(), Diagnostic> {
    require_two_dimensional_mesh(mesh)?;
    let expected = ReferenceCell::hypercube(DIMENSION)?;
    if quadrature.reference_cell() != expected {
        return Err(invalid(format!(
            "two-dimensional Cartesian cell requires {expected:?} quadrature, received {:?}",
            quadrature.reference_cell(),
        )));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_DISCRETIZATION, message)
}

#[cfg(test)]
mod tests;
