use std::sync::Arc;

use eqiora_assembly::{
    AssemblyBackend, AssemblyMap, AssemblyPacket, AssemblyPlan, AssemblyReport, AssemblyTarget,
    DofId, IndexedAssemblyWork, LinearSystem, LocalContribution, LocalUnknown, TargetAssemblyMap,
};
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_ir::LocalLinearActionIr;
use eqiora_meshing::{
    AffineGeometryLinearization, AffineGeometryMap, CartesianMesh, GeometryMap, MeshEntity,
    MeshGeometry, MeshTopology, QuadratureRule, ReferenceCell,
};
use eqiora_schema::kernel::BoundarySide;
use eqiora_solver::{
    CanonicalCsrSystemView, LinearOperatorProperties, LinearSolution, LinearSolveRequest,
    SolveReport,
};

use crate::affine_fem::{dot, physical_gradient, weighted_gradient, weighted_gradient_tangent};
use crate::assembled_linearization::AssembledLinearizedRelation;
use crate::canonical::ScalarEllipticCartesianModel;
use crate::constrained_dofs::ConstrainedDofLayout;
use crate::discrete_space::DiscreteSpace;
use crate::form_compiler::{
    AdmittedScalarGalerkinForm, DerivedScalarGalerkinForm, compile_cartesian_q1_form,
};
use crate::linearized_output::CartesianScalarFieldLinearization;
use crate::operator::LocalOperator;
use crate::spatial_design::SpatialDesignCoordinate;

mod design;
mod point;
pub(crate) use point::{CartesianFemPoint, CartesianFvmPoint, CartesianPoint};
pub use point::{
    linearize_scalar_elliptic_cartesian_fem, linearize_scalar_elliptic_cartesian_fem_output,
    linearize_scalar_elliptic_cartesian_fvm, linearize_scalar_elliptic_cartesian_fvm_output,
};
mod fvm;
pub(crate) use fvm::{
    FinalizedCartesianFvmAssembly, FinalizedCartesianFvmState,
    finalize_scalar_elliptic_cartesian_fvm,
};
pub(crate) mod linear;

use design::{activate_model_parameter, design_geometry, select_design_coordinates};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum CartesianBoundaryValue<S = f64> {
    Essential(S),
    Natural(S),
}

/// Continuous scalar Q1 field on a Cartesian mesh.
///
/// Vertex values follow the mesh's canonical vertex stratum. Evaluation uses
/// the same reference basis and affine geometry contract as assembly.
#[derive(Debug, Clone, PartialEq)]
pub struct CartesianQ1Field {
    mesh: CartesianMesh,
    vertex_values: Vec<f64>,
}

impl CartesianQ1Field {
    /// Construct a finite field with exactly one value per mesh vertex.
    ///
    /// # Errors
    /// Returns `EQ0801` for an incompatible value shape or non-finite data.
    pub fn new(mesh: CartesianMesh, vertex_values: Vec<f64>) -> Result<Self, Diagnostic> {
        if Some(vertex_values.len()) != mesh.entity_count(0)
            || vertex_values.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                "Cartesian Q1 field requires one finite value per mesh vertex",
            ));
        }
        Ok(Self {
            mesh,
            vertex_values,
        })
    }

    /// Mesh carrying this field.
    #[must_use]
    pub const fn mesh(&self) -> &CartesianMesh {
        &self.mesh
    }

    /// Values in canonical mesh-vertex order.
    #[must_use]
    pub fn vertex_values(&self) -> &[f64] {
        &self.vertex_values
    }

    /// Consume the field into values in canonical mesh-vertex order.
    #[must_use]
    pub(crate) fn into_vertex_values(self) -> Vec<f64> {
        self.vertex_values
    }

    /// Continuous L2 error against an independently supplied exact field.
    ///
    /// # Errors
    /// Returns a numerical diagnostic for incompatible quadrature, invalid
    /// exact values, geometry failure, or non-finite accumulation.
    pub fn l2_error<E>(&self, exact: &E, quadrature: &QuadratureRule) -> Result<f64, Diagnostic>
    where
        E: Fn(&[f64]) -> f64 + ?Sized,
    {
        require_cell_rule(&self.mesh, quadrature)?;
        let dimension = self.mesh.topological_dimension();
        let space = DiscreteSpace::new(
            eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
            eqiora_meshing::ReferenceCell::hypercube(dimension)?,
        )?;
        let mut squared_error = 0.0;
        for cell_index in 0..self
            .mesh
            .entity_count(dimension)
            .expect("mesh owns its top stratum")
        {
            let cell = MeshEntity::new(dimension, cell_index);
            let geometry = self
                .mesh
                .geometry_map(cell)
                .expect("mesh top stratum owns geometry");
            let vertices = self
                .mesh
                .entity_vertices(cell)
                .expect("mesh top stratum owns a vertex closure");
            for point in quadrature.points() {
                let basis = space.tabulate(&point.coordinates)?;
                let mut physical = vec![0.0; dimension];
                geometry.map_point(&point.coordinates, &mut physical)?;
                let approximation = basis
                    .values()
                    .iter()
                    .zip(&vertices)
                    .map(|(basis, vertex)| basis * self.vertex_values[vertex.index()])
                    .sum::<f64>();
                let exact_value = exact(&physical);
                if !exact_value.is_finite() {
                    return Err(invalid("exact Cartesian field returned a non-finite value"));
                }
                squared_error +=
                    point.weight * geometry.measure_scale() * (approximation - exact_value).powi(2);
            }
        }
        if !squared_error.is_finite() || squared_error < 0.0 {
            return Err(invalid(
                "Cartesian L2 error accumulation is non-finite or negative",
            ));
        }
        Ok(squared_error.sqrt())
    }
}

/// Q1 finite-element solution with recovered boundary equilibrium evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarEllipticCartesianFemSolution {
    point: CartesianFemPoint,
    boundary_reaction_sum: f64,
    integrated_source: f64,
    assembly_report: AssemblyReport,
    solve_report: SolveReport,
}

impl ScalarEllipticCartesianFemSolution {
    pub(crate) fn into_point(self) -> CartesianFemPoint {
        self.point
    }

    /// Continuous Q1 result field.
    #[must_use]
    pub const fn field(&self) -> &CartesianQ1Field {
        &self.point.field
    }

    /// Sum of residual reactions on all eliminated boundary vertices.
    #[must_use]
    pub const fn boundary_reaction_sum(&self) -> f64 {
        self.boundary_reaction_sum
    }

    /// Source integral represented by the assembled cell load vectors.
    #[must_use]
    pub const fn integrated_source(&self) -> f64 {
        self.integrated_source
    }

    /// Complete assembly placement and accepted packet-shape evidence.
    #[must_use]
    pub const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    /// Complete solver/backend/execution evidence.
    #[must_use]
    pub const fn solve_report(&self) -> &SolveReport {
        &self.solve_report
    }

    /// Free-vertex algebraic unknowns in assembled equation order.
    #[must_use]
    pub fn algebraic_values(&self) -> &[f64] {
        &self.point.algebraic_values
    }

    /// Consume the accepted solution into its complete primary Field values.
    #[must_use]
    pub(crate) fn into_field_values(self) -> Vec<f64> {
        self.point.field.into_vertex_values()
    }
}

/// Cell-centered FVM solution and its explicit continuous reconstruction.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarEllipticCartesianFvmSolution {
    point: CartesianFvmPoint,
    reconstruction: CartesianQ1Field,
    boundary_flux_sum: f64,
    integrated_source: f64,
    assembly_report: AssemblyReport,
    solve_report: SolveReport,
    nullspace: Option<crate::nullspace::NullspaceEvidence>,
}

impl ScalarEllipticCartesianFvmSolution {
    pub(crate) fn into_point(self) -> CartesianFvmPoint {
        self.point
    }

    pub(crate) fn nullspace_evidence(&self) -> Option<&crate::nullspace::NullspaceEvidence> {
        self.nullspace.as_ref()
    }

    /// Mesh carrying the cell-centred algebraic field.
    #[must_use]
    pub const fn mesh(&self) -> &CartesianMesh {
        &self.point.mesh
    }

    /// Physical center of every algebraic cell unknown.
    #[must_use]
    pub fn cell_centers(&self) -> &[Vec<f64>] {
        &self.point.cell_centers
    }

    /// Algebraic values in canonical cell order.
    #[must_use]
    pub fn cell_values(&self) -> &[f64] {
        &self.point.cell_values
    }

    /// Q1 dual-grid reconstruction through boundary samples and cell centers.
    #[must_use]
    pub const fn reconstruction(&self) -> &CartesianQ1Field {
        &self.reconstruction
    }

    /// Sum of reconstructed outward diffusive fluxes on boundary facets.
    #[must_use]
    pub const fn boundary_flux_sum(&self) -> f64 {
        self.boundary_flux_sum
    }

    /// Source integral represented by the assembled cell balances.
    #[must_use]
    pub const fn integrated_source(&self) -> f64 {
        self.integrated_source
    }

    /// Complete assembly placement and accepted packet-shape evidence.
    #[must_use]
    pub const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    /// Original equation residual, separately from the bordered solver report.
    #[must_use]
    pub fn original_residual_norm(&self) -> f64 {
        self.nullspace
            .as_ref()
            .map_or(self.solve_report.true_residual_norm(), |value| {
                value.original_residual_norm
            })
    }

    /// Supplied-mode load balance checked before a constrained solve.
    #[must_use]
    pub fn compatibility_residual(&self) -> Option<f64> {
        self.nullspace
            .as_ref()
            .map(|value| value.compatibility_residual)
    }

    /// Explicit spatial reference residual after a constrained solve.
    #[must_use]
    pub fn gauge_residual(&self) -> Option<f64> {
        self.nullspace.as_ref().map(|value| value.gauge_residual)
    }

    /// Constraint multiplier; no multiplier is added to the physical Field.
    #[must_use]
    pub fn gauge_multiplier(&self) -> Option<f64> {
        self.nullspace.as_ref().map(|value| value.multiplier)
    }

    /// Solver/backend/execution evidence for the bordered system when constrained.
    #[must_use]
    pub const fn solve_report(&self) -> &SolveReport {
        &self.solve_report
    }

    /// Consume the accepted solution into canonical cell-centred Field values.
    #[must_use]
    pub(crate) fn into_field_values(self) -> Vec<f64> {
        self.point.cell_values
    }
}

/// Method-private state retained between finalized FEM assembly and field
/// reconstruction.
///
/// This is crate-visible only so canonical realization can expose one opaque,
/// method-neutral algebraic handoff without making constraint bookkeeping part
/// of the public contract.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FinalizedCartesianFemAssembly {
    mesh: CartesianMesh,
    constrained_dofs: ConstrainedDofLayout<f64>,
    linear_system: LinearSystem<f64>,
    full_system: LinearSystem<f64>,
    integrated_source: f64,
    assembly_report: AssemblyReport,
}

impl FinalizedCartesianFemAssembly {
    pub(crate) fn into_canonical(
        self,
    ) -> Result<(Arc<CanonicalCsrSystemView>, FinalizedCartesianFemState), Diagnostic> {
        let Self {
            mesh,
            constrained_dofs,
            linear_system,
            full_system,
            integrated_source,
            assembly_report,
        } = self;
        let canonical_system = Arc::new(CanonicalCsrSystemView::new(
            &linear_system,
            linear_system.rhs(),
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )?);
        Ok((
            canonical_system,
            FinalizedCartesianFemState {
                mesh,
                constrained_dofs,
                full_system,
                integrated_source,
                assembly_report,
            },
        ))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FinalizedCartesianFemState {
    mesh: CartesianMesh,
    constrained_dofs: ConstrainedDofLayout<f64>,
    full_system: LinearSystem<f64>,
    integrated_source: f64,
    assembly_report: AssemblyReport,
}

impl FinalizedCartesianFemState {
    pub(crate) const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    pub(crate) fn point(
        &self,
        algebraic_values: Vec<f64>,
        canonical_system: Arc<CanonicalCsrSystemView>,
    ) -> Result<CartesianFemPoint, Diagnostic> {
        if algebraic_values.len() != canonical_system.rows() {
            return Err(invalid(
                "Cartesian FEM solution shape differs from its finalized system",
            ));
        }
        let values = self.constrained_dofs.lift(&algebraic_values)?;
        Ok(CartesianFemPoint {
            field: CartesianQ1Field::new(self.mesh.clone(), values)?,
            free_vertices: self.constrained_dofs.free_globals(),
            algebraic_values,
            canonical_system,
        })
    }

    pub(crate) fn finish(
        self,
        solved: LinearSolution,
        canonical_system: Arc<CanonicalCsrSystemView>,
    ) -> Result<ScalarEllipticCartesianFemSolution, Diagnostic> {
        let (algebraic_values, solve_report) = solved.into_parts();
        let point = self.point(algebraic_values, canonical_system)?;
        let equilibrium = self
            .constrained_dofs
            .full_residual(&self.full_system, point.field.vertex_values())?;
        let boundary_reaction_sum = self.constrained_dofs.reaction_sum::<1>(&equilibrium)?[0];
        if !boundary_reaction_sum.is_finite() {
            return Err(invalid("Cartesian FEM reaction sum is non-finite"));
        }

        Ok(ScalarEllipticCartesianFemSolution {
            point,
            boundary_reaction_sum,
            integrated_source: self.integrated_source,
            assembly_report: self.assembly_report,
            solve_report,
        })
    }
}

/// Lower the Cartesian Q1 diffusion cell operator to one shape-homogeneous
/// device-neutral local-action batch.
///
/// The returned inputs and outputs are packed in canonical cell order, then
/// local Q1 basis order. This is deliberately only the anonymous entity-local
/// action: global gather, constraint handling, scatter, and reduction remain
/// separate execution contracts.
///
/// # Errors
/// Returns a numerical or Operator-IR diagnostic for an invalid coefficient,
/// quadrature rule, cell geometry, local contribution, or batch shape.
pub fn lower_cartesian_q1_diffusion_local_action<K>(
    mesh: &CartesianMesh,
    coefficient: &K,
    quadrature: &QuadratureRule,
) -> Result<LocalLinearActionIr, Diagnostic>
where
    K: Fn(&[f64]) -> f64 + ?Sized,
{
    validate_problem(mesh, quadrature)?;
    let dimension = mesh.topological_dimension();
    let cell_count = mesh.entity_count(dimension).expect("mesh owns cells");
    let local_width = DiscreteSpace::new(
        eqiora_realization::Space::continuous_lagrange(std::num::NonZeroU16::MIN),
        eqiora_meshing::ReferenceCell::hypercube(dimension)?,
    )?
    .local_dofs()
    .len();
    let coefficients_per_cell = local_width
        .checked_mul(local_width)
        .ok_or_else(|| invalid("Cartesian Q1 local-action shape overflows usize"))?;
    let coefficient_count = cell_count
        .checked_mul(coefficients_per_cell)
        .ok_or_else(|| invalid("Cartesian Q1 local-action batch overflows usize"))?;
    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(coefficient_count)
        .map_err(|_| {
            invalid("Cartesian Q1 local-action batch exceeds platform allocation capacity")
        })?;
    let zero_source = |_: &[f64]| 0.0;
    let compiled = compile_cartesian_q1_form(dimension, quadrature)?;
    let operator = CartesianEllipticCell {
        coefficient,
        source: &zero_source,
        compiled: &compiled,
    };
    for cell_index in 0..cell_count {
        let cell = MeshEntity::new(dimension, cell_index);
        let geometry = mesh
            .geometry_map(cell)
            .expect("mesh cell has affine geometry");
        let local = operator.evaluate(&geometry, quadrature)?;
        debug_assert_eq!(local.rows(), local_width);
        debug_assert_eq!(local.columns(), local_width);
        coefficients.extend_from_slice(local.matrix());
    }
    LocalLinearActionIr::new(local_width, local_width, coefficients)
}

/// Solve Cartesian Q1 FEM through an explicit ordered assembly backend.
///
/// Each cell evaluates its local operator exactly once and maps that single
/// contribution to both the reduced solve system and full reaction system.
/// The backend may schedule cell evaluation freely, but accepted scatter order
/// and floating-point accumulation remain the canonical cell order.
///
/// # Errors
/// Preserves the reference entry point diagnostics and the selected assembly
/// backend's complete-operation diagnostics.
pub fn solve_scalar_elliptic_cartesian_fem_with_assembly<K, S, B>(
    mesh: &CartesianMesh,
    coefficient: &K,
    source: &S,
    boundary: &B,
    quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend<f64>,
    solver: LinearSolveRequest<'_>,
) -> Result<ScalarEllipticCartesianFemSolution, Diagnostic>
where
    K: Fn(&[f64]) -> f64 + Sync + ?Sized,
    S: Fn(&[f64]) -> f64 + Sync + ?Sized,
    B: Fn(&[f64]) -> f64 + ?Sized,
{
    let boundary = |_: usize, _: BoundarySide, coordinates: &[f64]| {
        CartesianBoundaryValue::Essential(boundary(coordinates))
    };
    let finalized = finalize_scalar_elliptic_cartesian_fem(
        mesh,
        coefficient,
        source,
        &boundary,
        quadrature,
        assembly,
        None,
    )?;
    let (canonical_system, state) = finalized.into_canonical()?;
    let solved = solver.solve(&canonical_system.linear_problem()?)?;
    state.finish(solved, canonical_system)
}

pub(crate) fn finalize_scalar_elliptic_cartesian_fem<K, S, B>(
    mesh: &CartesianMesh,
    coefficient: &K,
    source: &S,
    boundary: &B,
    quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend<f64>,
    form: Option<&DerivedScalarGalerkinForm>,
) -> Result<FinalizedCartesianFemAssembly, Diagnostic>
where
    K: Fn(&[f64]) -> f64 + Sync + ?Sized,
    S: Fn(&[f64]) -> f64 + Sync + ?Sized,
    B: Fn(usize, BoundarySide, &[f64]) -> CartesianBoundaryValue + ?Sized,
{
    validate_problem(mesh, quadrature)?;
    let dimension = mesh.topological_dimension();
    let compiled = match form {
        Some(form) => form.admit_quadrature(quadrature)?,
        None => compile_cartesian_q1_form(dimension, quadrature)?,
    };
    let operator = CartesianEllipticCell {
        coefficient,
        source,
        compiled: &compiled,
    };
    let fixed_values = essential_fem_values(mesh, boundary)?;
    if fixed_values.iter().all(Option::is_none) {
        return Err(invalid(
            "Cartesian Q1 system requires at least one essential boundary vertex",
        ));
    }
    let constrained_dofs = ConstrainedDofLayout::new(fixed_values)?;
    if constrained_dofs.free_count() == 0 {
        return Err(invalid(
            "Cartesian Q1 system requires at least one unconstrained interior vertex",
        ));
    }
    let cell_count = mesh.entity_count(dimension).expect("mesh owns cells");
    let natural_facets = natural_fem_facets(mesh, boundary)?;
    let natural_load = natural_facets
        .iter()
        .flat_map(|(local, _)| local.rhs())
        .sum::<f64>();
    let packet_count = cell_count
        .checked_add(natural_facets.len())
        .ok_or_else(|| invalid("Cartesian FEM packet count overflows usize"))?;
    let (reduced_system, full_system, assembly_report) =
        constrained_dofs.assemble(assembly, packet_count, |packet_index| {
            let (local, vertices) = if packet_index < cell_count {
                let cell = MeshEntity::new(dimension, packet_index);
                let geometry = mesh
                    .geometry_map(cell)
                    .expect("mesh cell has affine geometry");
                (
                    operator.evaluate(&geometry, quadrature)?,
                    mesh.entity_vertices(cell)
                        .expect("mesh cell has a vertex closure"),
                )
            } else {
                natural_facets[packet_index - cell_count].clone()
            };
            let global_dofs = vertices
                .iter()
                .map(|vertex| vertex.index())
                .collect::<Vec<_>>();
            Ok((local, global_dofs))
        })?;

    let integrated_source = full_system.rhs().iter().sum::<f64>() - natural_load;
    if !integrated_source.is_finite() {
        return Err(invalid("Cartesian FEM source integral is non-finite"));
    }
    Ok(FinalizedCartesianFemAssembly {
        mesh: mesh.clone(),
        constrained_dofs,
        linear_system: reduced_system,
        full_system,
        integrated_source,
        assembly_report,
    })
}

/// Solve Cartesian cell-centred TPFA through an explicit assembly backend.
///
/// Logical packets are ordered as all source cells followed by all canonical
/// facets. Interior and boundary facets therefore share the same pure local-
/// contribution path without making topology traversal or physics metadata an
/// assembly-backend responsibility.
///
/// # Errors
/// Preserves the reference entry point diagnostics and the selected assembly
/// backend's complete-operation diagnostics.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn solve_scalar_elliptic_cartesian_fvm_with_assembly<K, S, B>(
    mesh: &CartesianMesh,
    coefficient: &K,
    source: &S,
    boundary: &B,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend<f64>,
    solver: LinearSolveRequest<'_>,
) -> Result<ScalarEllipticCartesianFvmSolution, Diagnostic>
where
    K: Fn(&[f64]) -> f64 + Sync + ?Sized,
    S: Fn(&[f64]) -> f64 + Sync + ?Sized,
    B: Fn(&[f64]) -> f64 + ?Sized,
{
    let boundary = |_: usize, _: BoundarySide, coordinates: &[f64]| {
        Ok(CartesianBoundaryValue::Essential(boundary(coordinates)))
    };
    let finalized = finalize_scalar_elliptic_cartesian_fvm(
        mesh,
        &|x| Ok(coefficient(x)),
        &|_| Ok(0.),
        &|x| Ok(source(x)),
        &boundary,
        cell_quadrature,
        facet_quadrature,
        assembly,
        eqiora_solver::LinearOperatorProperties::SymmetricPositiveDefinite,
    )?;
    let (canonical_system, state) = finalized.into_canonical(None)?;
    state.solve(solver, canonical_system)
}

pub(crate) mod support;
use support::*;

#[cfg(test)]
mod tests;
