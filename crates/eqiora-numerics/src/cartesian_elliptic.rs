use std::sync::Arc;

#[cfg(test)]
use eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND;
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
use crate::discrete_space::{DiscreteSpace, HypercubeQ1Space};
use crate::form_compiler::{
    AdmittedScalarGalerkinForm, DerivedScalarGalerkinForm, compile_cartesian_q1_form,
};
use crate::linearized_output::CartesianScalarFieldLinearization;
use crate::operator::LocalOperator;
use crate::spatial_design::SpatialDesignCoordinate;

mod design;
pub(crate) mod linear;

use design::{activate_model_parameter, design_geometry, select_design_coordinates};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum CartesianBoundaryValue {
    Essential(f64),
    Natural(f64),
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
        let space = HypercubeQ1Space::new(dimension)?;
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
    field: CartesianQ1Field,
    free_vertices: Vec<usize>,
    algebraic_values: Vec<f64>,
    canonical_system: Arc<CanonicalCsrSystemView>,
    boundary_reaction_sum: f64,
    integrated_source: f64,
    assembly_report: AssemblyReport,
    solve_report: SolveReport,
}

impl ScalarEllipticCartesianFemSolution {
    /// Continuous Q1 result field.
    #[must_use]
    pub const fn field(&self) -> &CartesianQ1Field {
        &self.field
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

    pub(crate) fn free_vertices(&self) -> &[usize] {
        &self.free_vertices
    }

    /// Free-vertex algebraic unknowns in assembled equation order.
    #[must_use]
    pub fn algebraic_values(&self) -> &[f64] {
        &self.algebraic_values
    }

    pub(crate) const fn canonical_system(&self) -> &Arc<CanonicalCsrSystemView> {
        &self.canonical_system
    }

    /// Consume the accepted solution into its complete primary Field values.
    #[must_use]
    pub(crate) fn into_field_values(self) -> Vec<f64> {
        self.field.into_vertex_values()
    }
}

/// Cell-centered FVM solution and its explicit continuous reconstruction.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarEllipticCartesianFvmSolution {
    mesh: CartesianMesh,
    cell_centers: Vec<Vec<f64>>,
    cell_values: Vec<f64>,
    canonical_system: Arc<CanonicalCsrSystemView>,
    reconstruction: CartesianQ1Field,
    boundary_flux_sum: f64,
    integrated_source: f64,
    assembly_report: AssemblyReport,
    solve_report: SolveReport,
}

impl ScalarEllipticCartesianFvmSolution {
    /// Mesh carrying the cell-centred algebraic field.
    #[must_use]
    pub const fn mesh(&self) -> &CartesianMesh {
        &self.mesh
    }

    /// Physical center of every algebraic cell unknown.
    #[must_use]
    pub fn cell_centers(&self) -> &[Vec<f64>] {
        &self.cell_centers
    }

    /// Algebraic values in canonical cell order.
    #[must_use]
    pub fn cell_values(&self) -> &[f64] {
        &self.cell_values
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

    /// Complete solver/backend/execution evidence.
    #[must_use]
    pub const fn solve_report(&self) -> &SolveReport {
        &self.solve_report
    }

    pub(crate) const fn canonical_system(&self) -> &Arc<CanonicalCsrSystemView> {
        &self.canonical_system
    }

    /// Consume the accepted solution into canonical cell-centred Field values.
    #[must_use]
    pub(crate) fn into_field_values(self) -> Vec<f64> {
        self.cell_values
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
    constrained_dofs: ConstrainedDofLayout,
    linear_system: LinearSystem,
    full_system: LinearSystem,
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
    constrained_dofs: ConstrainedDofLayout,
    full_system: LinearSystem,
    integrated_source: f64,
    assembly_report: AssemblyReport,
}

impl FinalizedCartesianFemState {
    pub(crate) const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    pub(crate) fn finish(
        self,
        solved: LinearSolution,
        canonical_system: Arc<CanonicalCsrSystemView>,
    ) -> Result<ScalarEllipticCartesianFemSolution, Diagnostic> {
        if solved.values().len() != canonical_system.rows() {
            return Err(invalid(
                "Cartesian FEM solution shape differs from its finalized system",
            ));
        }
        let (algebraic_values, solve_report) = solved.into_parts();
        let free_vertices = self.constrained_dofs.free_globals();
        let values = self.constrained_dofs.lift(&algebraic_values)?;
        let equilibrium = self
            .constrained_dofs
            .full_residual(&self.full_system, &values)?;
        let boundary_reaction_sum = self.constrained_dofs.reaction_sum::<1>(&equilibrium)?[0];
        if !boundary_reaction_sum.is_finite() {
            return Err(invalid("Cartesian FEM reaction sum is non-finite"));
        }

        Ok(ScalarEllipticCartesianFemSolution {
            field: CartesianQ1Field::new(self.mesh, values)?,
            free_vertices,
            algebraic_values,
            canonical_system,
            boundary_reaction_sum,
            integrated_source: self.integrated_source,
            assembly_report: self.assembly_report,
            solve_report,
        })
    }
}

/// Method-private state retained between finalized TPFA assembly and field
/// reconstruction.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FinalizedCartesianFvmAssembly {
    mesh: CartesianMesh,
    cell_centers: Vec<Vec<f64>>,
    linear_system: LinearSystem,
    reconstruction_mesh: CartesianMesh,
    reconstruction_boundary_values: Vec<Option<f64>>,
    facets: Vec<CartesianFacetPacket>,
    assembly_report: AssemblyReport,
}

impl FinalizedCartesianFvmAssembly {
    pub(crate) fn into_canonical(
        self,
    ) -> Result<(Arc<CanonicalCsrSystemView>, FinalizedCartesianFvmState), Diagnostic> {
        let Self {
            mesh,
            cell_centers,
            linear_system,
            reconstruction_mesh,
            reconstruction_boundary_values,
            facets,
            assembly_report,
        } = self;
        let canonical_system = Arc::new(CanonicalCsrSystemView::new(
            &linear_system,
            LinearOperatorProperties::SymmetricPositiveDefinite,
        )?);
        Ok((
            canonical_system,
            FinalizedCartesianFvmState {
                mesh,
                cell_centers,
                reconstruction_mesh,
                reconstruction_boundary_values,
                facets,
                assembly_report,
            },
        ))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FinalizedCartesianFvmState {
    mesh: CartesianMesh,
    cell_centers: Vec<Vec<f64>>,
    reconstruction_mesh: CartesianMesh,
    reconstruction_boundary_values: Vec<Option<f64>>,
    facets: Vec<CartesianFacetPacket>,
    assembly_report: AssemblyReport,
}

impl FinalizedCartesianFvmState {
    pub(crate) const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    pub(crate) fn finish(
        self,
        solved: LinearSolution,
        canonical_system: Arc<CanonicalCsrSystemView>,
    ) -> Result<ScalarEllipticCartesianFvmSolution, Diagnostic> {
        if solved.values().len() != canonical_system.rows() {
            return Err(invalid(
                "Cartesian FVM solution shape differs from its finalized system",
            ));
        }
        let (cell_values, solve_report) = solved.into_parts();
        let (boundary_flux_sum, boundary_load_sum) =
            self.facets
                .iter()
                .try_fold((0.0, 0.0), |(flux, load), facet| match facet.kind {
                    CartesianFacetKind::Interior { .. } => Ok((flux, load)),
                    CartesianFacetKind::Essential { cell, value, .. } => {
                        let cell_value = cell_values.get(cell).copied().ok_or_else(|| {
                            invalid("Cartesian FVM boundary facet exceeds its finalized field")
                        })?;
                        Ok::<_, Diagnostic>((
                            flux + facet.transmissibility * (value - cell_value),
                            load + facet.transmissibility * value,
                        ))
                    }
                    CartesianFacetKind::Natural { flux_integral, .. } => {
                        Ok((flux + flux_integral, load + flux_integral))
                    }
                })?;
        if !boundary_flux_sum.is_finite() {
            return Err(invalid("Cartesian FVM boundary flux sum is non-finite"));
        }
        let integrated_source =
            canonical_system.right_hand_side().iter().sum::<f64>() - boundary_load_sum;
        if !integrated_source.is_finite() {
            return Err(invalid("Cartesian FVM source integral is non-finite"));
        }
        let reconstruction = reconstruct_cell_field_from_boundary_values(
            self.reconstruction_mesh,
            &self.mesh,
            &cell_values,
            self.reconstruction_boundary_values,
            &self.facets,
        )?;

        Ok(ScalarEllipticCartesianFvmSolution {
            mesh: self.mesh,
            cell_centers: self.cell_centers,
            cell_values,
            canonical_system,
            reconstruction,
            boundary_flux_sum,
            integrated_source,
            assembly_report: self.assembly_report,
            solve_report,
        })
    }
}

/// Solve `-div(k grad(u)) = source` with continuous Cartesian Q1 FEM and
/// prescribed values on the complete box boundary.
///
/// # Errors
/// Returns a numerical diagnostic for invalid coefficients, quadrature,
/// boundary/source evaluation, mesh constraints, assembly, or solve failure.
#[cfg(test)]
pub fn solve_scalar_elliptic_cartesian_fem<S, B>(
    mesh: &CartesianMesh,
    diffusion: f64,
    source: &S,
    boundary: &B,
    quadrature: &QuadratureRule,
    solver: LinearSolveRequest<'_>,
) -> Result<ScalarEllipticCartesianFemSolution, Diagnostic>
where
    S: Fn(&[f64]) -> f64 + Sync + ?Sized,
    B: Fn(&[f64]) -> f64 + ?Sized,
{
    solve_scalar_elliptic_cartesian_fem_with_assembly(
        mesh,
        &move |_: &[f64]| diffusion,
        source,
        boundary,
        quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        solver,
    )
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
    let local_width = HypercubeQ1Space::new(dimension)?.local_dofs().len();
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
    assembly: &dyn AssemblyBackend,
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
    assembly: &dyn AssemblyBackend,
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

/// Solve `-div(k grad(u)) = source` with orthogonal cell-centered TPFA and
/// prescribed values on the complete Cartesian-box boundary.
///
/// Cell, interior-facet, and boundary-facet contributions all scatter through
/// the same [`AssemblyMap`] contract as FEM.
///
/// # Errors
/// Returns a numerical diagnostic for invalid coefficients, quadrature,
/// boundary/source evaluation, topology, assembly, or solve failure.
#[cfg(test)]
pub fn solve_scalar_elliptic_cartesian_fvm<S, B>(
    mesh: &CartesianMesh,
    diffusion: f64,
    source: &S,
    boundary: &B,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    solver: LinearSolveRequest<'_>,
) -> Result<ScalarEllipticCartesianFvmSolution, Diagnostic>
where
    S: Fn(&[f64]) -> f64 + Sync + ?Sized,
    B: Fn(&[f64]) -> f64 + ?Sized,
{
    solve_scalar_elliptic_cartesian_fvm_with_assembly(
        mesh,
        &move |_: &[f64]| diffusion,
        source,
        boundary,
        cell_quadrature,
        facet_quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        solver,
    )
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
pub fn solve_scalar_elliptic_cartesian_fvm_with_assembly<K, S, B>(
    mesh: &CartesianMesh,
    coefficient: &K,
    source: &S,
    boundary: &B,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend,
    solver: LinearSolveRequest<'_>,
) -> Result<ScalarEllipticCartesianFvmSolution, Diagnostic>
where
    K: Fn(&[f64]) -> f64 + Sync + ?Sized,
    S: Fn(&[f64]) -> f64 + Sync + ?Sized,
    B: Fn(&[f64]) -> f64 + ?Sized,
{
    let boundary = |_: usize, _: BoundarySide, coordinates: &[f64]| {
        CartesianBoundaryValue::Essential(boundary(coordinates))
    };
    let finalized = finalize_scalar_elliptic_cartesian_fvm(
        mesh,
        coefficient,
        source,
        &boundary,
        cell_quadrature,
        facet_quadrature,
        assembly,
    )?;
    let (canonical_system, state) = finalized.into_canonical()?;
    let solved = solver.solve(&canonical_system.linear_problem()?)?;
    state.finish(solved, canonical_system)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize_scalar_elliptic_cartesian_fvm<K, S, B>(
    mesh: &CartesianMesh,
    coefficient: &K,
    source: &S,
    boundary: &B,
    cell_quadrature: &QuadratureRule,
    facet_quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend,
) -> Result<FinalizedCartesianFvmAssembly, Diagnostic>
where
    K: Fn(&[f64]) -> f64 + Sync + ?Sized,
    S: Fn(&[f64]) -> f64 + Sync + ?Sized,
    B: Fn(usize, BoundarySide, &[f64]) -> CartesianBoundaryValue + ?Sized,
{
    validate_problem(mesh, cell_quadrature)?;
    let dimension = mesh.topological_dimension();
    require_facet_rule(dimension, facet_quadrature)?;
    let cell_count = mesh.entity_count(dimension).expect("mesh owns cells");
    let cell_centers = (0..cell_count)
        .map(|cell_index| {
            mesh.geometry_map(MeshEntity::new(dimension, cell_index))
                .expect("mesh cell has affine geometry")
                .origin()
                .to_vec()
        })
        .collect::<Vec<_>>();

    let facet_dimension = dimension - 1;
    let facet_count = mesh
        .entity_count(facet_dimension)
        .expect("mesh owns its facet stratum");
    let facets = (0..facet_count)
        .map(|facet_index| {
            let facet = MeshEntity::new(facet_dimension, facet_index);
            let facet_geometry = mesh
                .geometry_map(facet)
                .expect("mesh facet has affine geometry");
            let free_axes = mesh
                .entity_free_axes(facet)
                .expect("mesh facet exposes its tangent axes");
            let normal_axis = (0..dimension)
                .find(|axis| free_axes.binary_search(axis).is_err())
                .ok_or_else(|| invalid("Cartesian facet has no normal axis"))?;
            let cells = mesh
                .incidence(facet, dimension)
                .ok_or_else(|| invalid("Cartesian facet adjacency is unavailable"))?;
            let (kind, distance) = match cells.as_slice() {
                [left, right] => {
                    let left_center = &cell_centers[left.entity.index()];
                    let right_center = &cell_centers[right.entity.index()];
                    let distance = (right_center[normal_axis] - left_center[normal_axis]).abs();
                    require_positive_distance(distance)?;
                    (
                        CartesianFacetKind::Interior {
                            left: left.entity.index(),
                            right: right.entity.index(),
                        },
                        distance,
                    )
                }
                [cell] => {
                    let center = &cell_centers[cell.entity.index()];
                    let boundary_coordinates = facet_geometry.origin();
                    let distance = (boundary_coordinates[normal_axis] - center[normal_axis]).abs();
                    let (axis, side) = cartesian_boundary_facet_side(mesh, facet)?
                        .expect("one-cell Cartesian facet is on the boundary");
                    require_positive_distance(distance)?;
                    let kind = match boundary(axis, side, boundary_coordinates) {
                        CartesianBoundaryValue::Essential(value) if value.is_finite() => {
                            CartesianFacetKind::Essential {
                                axis,
                                side,
                                cell: cell.entity.index(),
                                value,
                            }
                        }
                        CartesianBoundaryValue::Natural(_) => {
                            let flux_integral = integrate_boundary_flux(
                                &facet_geometry,
                                facet_quadrature,
                                &|coordinates| match boundary(axis, side, coordinates) {
                                    CartesianBoundaryValue::Natural(value) => value,
                                    CartesianBoundaryValue::Essential(_) => f64::NAN,
                                },
                            )?;
                            CartesianFacetKind::Natural {
                                axis,
                                side,
                                cell: cell.entity.index(),
                                flux_integral,
                            }
                        }
                        CartesianBoundaryValue::Essential(_) => {
                            return Err(invalid("Cartesian boundary returned a non-finite value"));
                        }
                    };
                    (kind, distance)
                }
                _ => {
                    return Err(invalid(
                        "Cartesian facet requires exactly one or two adjacent cells",
                    ));
                }
            };
            let mut face_centroid = vec![0.0; dimension];
            facet_geometry.map_point(&vec![0.0; facet_dimension], &mut face_centroid)?;
            let coefficient_value = coefficient(&face_centroid);
            let transmissibility = facet_transmissibility(
                &facet_geometry,
                distance,
                coefficient_value,
                facet_quadrature,
            )?;
            Ok(CartesianFacetPacket {
                transmissibility,
                kind,
            })
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    if facets
        .iter()
        .all(|facet| !matches!(facet.kind, CartesianFacetKind::Essential { .. }))
    {
        return Err(invalid(
            "Cartesian TPFA system requires at least one essential boundary facet",
        ));
    }
    let (reconstruction_mesh, reconstruction_boundary_values) =
        prepare_cell_field_reconstruction(mesh, boundary)?;

    let assembly_plan = AssemblyPlan::new(vec![AssemblyTarget::new(cell_count)?])?;
    let target = assembly_plan
        .target_id(0)
        .expect("one-target FVM assembly plan owns its system");
    let packet_count = cell_count
        .checked_add(facet_count)
        .ok_or_else(|| invalid("Cartesian FVM packet count overflows usize"))?;
    let source_operator = CartesianSourceCell { source };
    let work = IndexedAssemblyWork::new(packet_count, |packet_index| {
        let (local, map) = if packet_index < cell_count {
            let cell = MeshEntity::new(dimension, packet_index);
            let geometry = mesh
                .geometry_map(cell)
                .expect("mesh cell has affine geometry");
            let local = source_operator.evaluate(&geometry, cell_quadrature)?;
            let dof = DofId::new(packet_index);
            let map = AssemblyMap::new(vec![Some(dof)], vec![LocalUnknown::Free(dof)])?;
            (local, map)
        } else {
            let facet = &facets[packet_index - cell_count];
            match facet.kind {
                CartesianFacetKind::Interior { left, right } => {
                    let left = DofId::new(left);
                    let right = DofId::new(right);
                    let local = CartesianInteriorFlux
                        .evaluate(&facet.transmissibility, facet_quadrature)?;
                    let map = AssemblyMap::new(
                        vec![Some(left), Some(right)],
                        vec![LocalUnknown::Free(left), LocalUnknown::Free(right)],
                    )?;
                    (local, map)
                }
                CartesianFacetKind::Essential { cell, value, .. } => {
                    let cell = DofId::new(cell);
                    let local = CartesianBoundaryFlux
                        .evaluate(&facet.transmissibility, facet_quadrature)?;
                    let map = AssemblyMap::new(
                        vec![Some(cell)],
                        vec![LocalUnknown::Free(cell), LocalUnknown::Fixed(value)],
                    )?;
                    (local, map)
                }
                CartesianFacetKind::Natural {
                    cell,
                    flux_integral,
                    ..
                } => {
                    let cell = DofId::new(cell);
                    let local = LocalContribution::new(1, 1, vec![0.0], vec![flux_integral])?;
                    let map = AssemblyMap::new(vec![Some(cell)], vec![LocalUnknown::Free(cell)])?;
                    (local, map)
                }
            }
        };
        AssemblyPacket::new(local, vec![TargetAssemblyMap::new(target, map)])
    });
    let (systems, assembly_report) = assembly.assemble(&assembly_plan, &work)?.into_parts();
    let mut systems = systems.into_iter();
    let system = systems
        .next()
        .expect("one-target FVM assembly returns its system");
    debug_assert!(systems.next().is_none());

    Ok(FinalizedCartesianFvmAssembly {
        mesh: mesh.clone(),
        cell_centers,
        linear_system: system,
        reconstruction_mesh,
        reconstruction_boundary_values,
        facets,
        assembly_report,
    })
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
    let space = HypercubeQ1Space::new(dimension)?;
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
            .then(|| HypercubeQ1Space::new(facet_dimension))
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
) -> Result<CartesianScalarFieldLinearization, Diagnostic> {
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
    CartesianScalarFieldLinearization::new(
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
) -> Result<CartesianScalarFieldLinearization, Diagnostic> {
    if solution.mesh() != mesh {
        return Err(invalid(
            "Cartesian FVM output linearization requires the accepted solution mesh",
        ));
    }
    let selected = select_design_coordinates(model, selected_coordinates)?;
    let output_dimension = solution.cell_values().len();
    let parameter_dimension = selected.coordinates.len();
    CartesianScalarFieldLinearization::new(
        solution.cell_values().to_vec(),
        output_dimension,
        parameter_dimension,
        (0..output_dimension).map(Some).collect(),
        vec![0.0; output_dimension * parameter_dimension],
    )
}

pub(crate) mod support;
use support::*;

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use eqiora_solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};

    use super::*;

    #[test]
    fn q1_diffusion_lowers_to_anonymous_uniform_local_action() {
        let mesh = CartesianMesh::from_axes(vec![vec![0.0, 0.2, 1.0], vec![-1.0, 0.5]]).unwrap();
        let rule = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
        let action =
            lower_cartesian_q1_diffusion_local_action(&mesh, &|_: &[f64]| 1.7, &rule).unwrap();
        let input = vec![1.0; action.input_len()];
        let mut output = vec![f64::NAN; action.output_len()];

        action.apply_reference(&input, &mut output).unwrap();

        assert_eq!(action.entity_count(), 2);
        assert_eq!((action.rows(), action.columns()), (4, 4));
        assert!(output.iter().all(|value| value.abs() < 8.0e-15));
    }

    #[test]
    fn both_methods_reproduce_a_linear_harmonic_field_on_a_nonuniform_grid() {
        let mesh =
            CartesianMesh::from_axes(vec![vec![0.0, 0.2, 0.65, 1.0], vec![-1.0, -0.1, 0.4, 2.0]])
                .unwrap();
        let cell_rule = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
        let facet_rule = QuadratureRule::gauss_legendre(2).unwrap();
        let plan = SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(256).unwrap(),
        )
        .unwrap();
        let solver = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan);
        let exact = |coordinate: &[f64]| 2.0 + coordinate[0] - 0.5 * coordinate[1];
        let source = |_: &[f64]| 0.0;

        let fem =
            solve_scalar_elliptic_cartesian_fem(&mesh, 1.0, &source, &exact, &cell_rule, solver)
                .unwrap();
        let fvm = solve_scalar_elliptic_cartesian_fvm(
            &mesh,
            1.0,
            &source,
            &exact,
            &cell_rule,
            &facet_rule,
            solver,
        )
        .unwrap();

        assert!(fem.field().l2_error(&exact, &cell_rule).unwrap() < 2.0e-13);
        let dual_rule = QuadratureRule::tensor_product_gauss_legendre(2, 2).unwrap();
        assert!(fvm.reconstruction().l2_error(&exact, &dual_rule).unwrap() < 2.0e-13);
        assert!(fem.boundary_reaction_sum().abs() < 2.0e-12);
        assert!(fvm.boundary_flux_sum().abs() < 2.0e-12);
        assert!((fem.boundary_reaction_sum() + fem.integrated_source()).abs() < 2.0e-12);
        assert!((fvm.boundary_flux_sum() + fvm.integrated_source()).abs() < 2.0e-12);
    }
}
