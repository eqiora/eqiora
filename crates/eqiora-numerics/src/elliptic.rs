use eqiora_assembly::{
    AssemblyBackend, AssemblyMap, AssemblyPacket, AssemblyPacketSetIdentityV1, AssemblyPlan,
    AssemblyReport, AssemblyTarget, AssemblyTargetId, AssemblyWork, DofId, LocalContribution,
    LocalUnknown, REFERENCE_ASSEMBLY_BACKEND, TargetAssemblyMap,
};
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_meshing::{CellId, LineMesh, QuadratureRule, ReferenceCell, SegmentGeometry1d};
use eqiora_solver::{
    CanonicalCsrSystemView, LinearOperatorProperties, LinearProblem, LinearSolveRequest,
    SolveReport,
};

use crate::nullspace::{NullspaceConstraint, NullspaceEvidence, solve_canonical_with_nullspace};
use crate::operator::LocalOperator;
use crate::poisson::PiecewiseLinearField1d;

/// One endpoint condition for `-d/dx(k du/dx) = f`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarBoundaryCondition1d {
    /// Prescribed field value, eliminated through [`AssemblyMap`].
    Essential(f64),
    /// Prescribed outward flux `n k du/dx`, assembled on the right-hand side.
    Natural(f64),
}

impl ScalarBoundaryCondition1d {
    fn value(self) -> f64 {
        match self {
            Self::Essential(value) | Self::Natural(value) => value,
        }
    }

    fn essential_value(self) -> Option<f64> {
        match self {
            Self::Essential(value) => Some(value),
            Self::Natural(_) => None,
        }
    }
}

/// Boundary conditions at the lower and upper endpoints of a line Domain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarBoundaryPair1d {
    lower: ScalarBoundaryCondition1d,
    upper: ScalarBoundaryCondition1d,
    mean_reference: Option<f64>,
}

impl ScalarBoundaryPair1d {
    /// Construct a finite, anchored boundary pair.
    ///
    /// # Errors
    /// Returns `EQ0801` for non-finite data or a pure-Neumann pair, whose
    /// constant nullspace requires a separate gauge contract.
    pub fn new(
        lower: ScalarBoundaryCondition1d,
        upper: ScalarBoundaryCondition1d,
    ) -> Result<Self, Diagnostic> {
        if !lower.value().is_finite() || !upper.value().is_finite() {
            return Err(invalid_discretization(
                "scalar elliptic boundary data must be finite",
            ));
        }
        if lower.essential_value().is_none() && upper.essential_value().is_none() {
            return Err(invalid_discretization(
                "pure-Neumann scalar elliptic problems require an explicit nullspace gauge",
            ));
        }
        Ok(Self {
            lower,
            upper,
            mean_reference: None,
        })
    }

    /// Prescribe both outward fluxes and an explicit spatial mean of the field.
    ///
    /// For `-div(k grad u) = f`, compatibility is `integral(f) + lower + upper = 0`.
    /// The assembled load is checked before solving; incompatible data are never repaired.
    ///
    /// # Errors
    /// Returns `EQ0801` for non-finite fluxes or mean.
    pub fn pure_neumann(lower: f64, upper: f64, mean: f64) -> Result<Self, Diagnostic> {
        if !lower.is_finite() || !upper.is_finite() || !mean.is_finite() {
            return Err(invalid_discretization(
                "Neumann fluxes and mean reference must be finite",
            ));
        }
        Ok(Self {
            lower: ScalarBoundaryCondition1d::Natural(lower),
            upper: ScalarBoundaryCondition1d::Natural(upper),
            mean_reference: Some(mean),
        })
    }

    /// Explicit mean reference for a pure-Neumann pair, absent for anchored data.
    #[must_use]
    pub const fn mean_reference(self) -> Option<f64> {
        self.mean_reference
    }

    /// Lower-coordinate boundary condition.
    #[must_use]
    pub const fn lower(self) -> ScalarBoundaryCondition1d {
        self.lower
    }

    /// Upper-coordinate boundary condition.
    #[must_use]
    pub const fn upper(self) -> ScalarBoundaryCondition1d {
        self.upper
    }
}

/// P1 finite-element solution and method-neutral equilibrium evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarEllipticSolution1d {
    field: PiecewiseLinearField1d,
    cell_gradients: Vec<f64>,
    endpoint_reactions: [Option<f64>; 2],
    assembly_report: AssemblyReport,
    solve_report: SolveReport,
    nullspace_evidence: Option<NullspaceEvidence>,
}

impl ScalarEllipticSolution1d {
    /// Continuous piecewise-linear field.
    #[must_use]
    pub const fn field(&self) -> &PiecewiseLinearField1d {
        &self.field
    }

    /// Constant physical gradient on every mesh cell.
    #[must_use]
    pub fn cell_gradients(&self) -> &[f64] {
        &self.cell_gradients
    }

    /// Reactions at essential endpoints; natural endpoints return `None`.
    /// Ordering is lower then upper coordinate.
    #[must_use]
    pub const fn endpoint_reactions(&self) -> [Option<f64>; 2] {
        self.endpoint_reactions
    }

    /// Complete local-work placement and accepted assembly shape.
    #[must_use]
    pub const fn assembly_report(&self) -> &AssemblyReport {
        &self.assembly_report
    }

    /// Krylov iterations used by the reference solve.
    #[must_use]
    pub const fn linear_iterations(&self) -> usize {
        self.solve_report.completed_iterations()
    }

    /// Original equation residual norm (before bordering when a gauge is present).
    #[must_use]
    pub const fn residual_norm(&self) -> f64 {
        match &self.nullspace_evidence {
            Some(evidence) => evidence.original_residual_norm,
            None => self.solve_report.true_residual_norm(),
        }
    }

    /// Signed source/flux compatibility residual; absent for anchored problems.
    #[must_use]
    pub fn compatibility_residual(&self) -> Option<f64> {
        self.nullspace_evidence
            .as_ref()
            .map(|evidence| evidence.compatibility_residual)
    }

    /// Multiplier of the explicit mean constraint; absent for anchored problems.
    #[must_use]
    pub fn gauge_multiplier(&self) -> Option<f64> {
        self.nullspace_evidence
            .as_ref()
            .map(|evidence| evidence.multiplier)
    }

    /// Signed physical integral residual of the requested mean constraint.
    #[must_use]
    pub fn gauge_residual(&self) -> Option<f64> {
        self.nullspace_evidence
            .as_ref()
            .map(|evidence| evidence.gauge_residual)
    }

    /// Complete solver/backend/execution evidence. For a mean constraint this
    /// describes the bordered system; `residual_norm()` checks the original equations.
    #[must_use]
    pub const fn solve_report(&self) -> &SolveReport {
        &self.solve_report
    }
}

/// Solve a variable-coefficient scalar elliptic problem with continuous P1
/// Galerkin finite elements.
///
/// The local operator evaluates `k`, `f`, basis gradients, and quadrature.
/// Essential values enter only through [`AssemblyMap`]; natural fluxes enter
/// as boundary-local right-hand-side contributions. Reactions are recovered
/// from the residual of the uneliminated full system.
///
/// # Errors
/// Returns a structured numerical diagnostic for invalid coefficients,
/// quadrature, constraints, assembly, or linear solve failure.
pub fn solve_scalar_elliptic_linear_fem<K, S>(
    mesh: &LineMesh,
    diffusion: &K,
    source: &S,
    boundary: ScalarBoundaryPair1d,
    quadrature: &QuadratureRule,
    solver: LinearSolveRequest<'_>,
) -> Result<ScalarEllipticSolution1d, Diagnostic>
where
    K: Fn(f64) -> f64 + Sync + ?Sized,
    S: Fn(f64) -> f64 + Sync + ?Sized,
{
    solve_scalar_elliptic_linear_fem_with_assembly(
        mesh,
        diffusion,
        source,
        boundary,
        quadrature,
        &REFERENCE_ASSEMBLY_BACKEND,
        solver,
    )
}

/// Solve the same P1 relation through an explicit assembly backend.
///
/// Local cell and natural-boundary packets have stable logical indices. The
/// backend may evaluate them concurrently, but the accepted systems retain
/// Eqiora's target/packet/local-entry accumulation order.
///
/// # Errors
/// Returns a structured numerical diagnostic for invalid coefficients,
/// quadrature, constraints, assembly, or linear solve failure.
pub fn solve_scalar_elliptic_linear_fem_with_assembly<K, S>(
    mesh: &LineMesh,
    diffusion: &K,
    source: &S,
    boundary: ScalarBoundaryPair1d,
    quadrature: &QuadratureRule,
    assembly: &dyn AssemblyBackend,
    solver: LinearSolveRequest<'_>,
) -> Result<ScalarEllipticSolution1d, Diagnostic>
where
    K: Fn(f64) -> f64 + Sync + ?Sized,
    S: Fn(f64) -> f64 + Sync + ?Sized,
{
    require_segment_rule(quadrature)?;
    let vertex_count = mesh.vertex_count();
    let essential = [
        boundary.lower().essential_value(),
        boundary.upper().essential_value(),
    ];
    let mut free_indices = vec![None; vertex_count];
    let mut free_count = 0;
    for (vertex, slot) in free_indices.iter_mut().enumerate() {
        let endpoint_value = if vertex == 0 {
            essential[0]
        } else if vertex + 1 == vertex_count {
            essential[1]
        } else {
            None
        };
        if endpoint_value.is_none() {
            *slot = Some(DofId::new(free_count));
            free_count += 1;
        }
    }

    let plan = AssemblyPlan::new(vec![
        AssemblyTarget::new(free_count)?,
        AssemblyTarget::new(vertex_count)?,
    ])?;
    let reduced_target = plan
        .target_id(0)
        .expect("two-target assembly plan has a reduced target");
    let full_target = plan
        .target_id(1)
        .expect("two-target assembly plan has a full target");
    let natural_boundaries = [
        (0, free_indices[0], boundary.lower()),
        (
            vertex_count - 1,
            free_indices[vertex_count - 1],
            boundary.upper(),
        ),
    ]
    .into_iter()
    .filter_map(|(vertex, free, condition)| match condition {
        ScalarBoundaryCondition1d::Essential(_) => None,
        ScalarBoundaryCondition1d::Natural(flux) => {
            Some(NaturalBoundaryPacket { vertex, free, flux })
        }
    })
    .collect::<Vec<_>>();
    let work = EllipticAssemblyWork {
        mesh,
        operator: EllipticCell { diffusion, source },
        quadrature,
        essential,
        free_indices: &free_indices,
        reduced_target,
        full_target,
        natural_boundaries,
    };
    let assembled = assembly.assemble(&plan, &work)?;
    let (mut systems, assembly_report) = assembled.into_parts();
    debug_assert_eq!(systems.len(), 2);
    let full_system = systems.pop().expect("assembly plan has a full target");
    let reduced_system = systems.pop().expect("assembly plan has a reduced target");
    let (solved_values, solve_report, nullspace_evidence) =
        if let Some(mean) = boundary.mean_reference() {
            // Integrate each P1 shape exactly. These are physical measure weights,
            // not an arithmetic nodal average or a silently pinned endpoint.
            let mut weights = vec![0.0; vertex_count];
            for cell in mesh.cells() {
                let geometry = mesh.cell_geometry(cell).expect("valid mesh cell");
                for vertex in mesh.cell_vertices(cell).expect("valid cell topology") {
                    weights[vertex.index()] += 0.5 * geometry.measure();
                }
            }
            let measure: f64 = weights.iter().sum();
            let constraint =
                NullspaceConstraint::new(vec![1.0; vertex_count], weights, mean * measure)?;
            let system =
                CanonicalCsrSystemView::new(&reduced_system, LinearOperatorProperties::Symmetric)?;
            let solved = solve_canonical_with_nullspace(solver, &system, &constraint)?;
            (solved.values, solved.report, Some(solved.evidence))
        } else {
            let problem = LinearProblem::new(
                reduced_system.matrix(),
                reduced_system.rhs(),
                LinearOperatorProperties::SymmetricPositiveDefinite,
            )?;
            let (values, report) = solver.solve(&problem)?.into_parts();
            (values, report, None)
        };
    let mut values = vec![0.0; vertex_count];
    for (vertex, value) in values.iter_mut().enumerate() {
        *value = if vertex == 0 {
            essential[0].unwrap_or_else(|| {
                solved_values[free_indices[vertex]
                    .expect("lower endpoint is free")
                    .index()]
            })
        } else if vertex + 1 == vertex_count {
            essential[1].unwrap_or_else(|| {
                solved_values[free_indices[vertex]
                    .expect("upper endpoint is free")
                    .index()]
            })
        } else {
            solved_values[free_indices[vertex]
                .expect("interior vertex is free")
                .index()]
        };
    }

    let mut coordinates = Vec::with_capacity(vertex_count);
    for vertex in mesh.vertices() {
        coordinates.push(
            mesh.vertex_coordinate(vertex)
                .expect("mesh iterator yields valid vertex geometry"),
        );
    }
    let cell_gradients = coordinates
        .windows(2)
        .zip(values.windows(2))
        .map(|(x, u)| (u[1] - u[0]) / (x[1] - x[0]))
        .collect::<Vec<_>>();

    let mut equilibrium = full_system.matrix().multiply(&values)?;
    for (residual, rhs) in equilibrium.iter_mut().zip(full_system.rhs()) {
        *residual -= rhs;
    }
    let endpoint_reactions = [
        essential[0].map(|_| equilibrium[0]),
        essential[1].map(|_| equilibrium[vertex_count - 1]),
    ];

    Ok(ScalarEllipticSolution1d {
        field: PiecewiseLinearField1d::new(coordinates, values)?,
        cell_gradients,
        endpoint_reactions,
        assembly_report,
        solve_report,
        nullspace_evidence,
    })
}

struct EllipticCell<'a, K: ?Sized, S: ?Sized> {
    diffusion: &'a K,
    source: &'a S,
}

impl<K, S> LocalOperator<SegmentGeometry1d> for EllipticCell<'_, K, S>
where
    K: Fn(f64) -> f64 + ?Sized,
    S: Fn(f64) -> f64 + ?Sized,
{
    fn evaluate(
        &self,
        geometry: &SegmentGeometry1d,
        quadrature: &QuadratureRule,
    ) -> Result<LocalContribution, Diagnostic> {
        require_segment_rule(quadrature)?;
        let derivatives = [-1.0 / geometry.measure(), 1.0 / geometry.measure()];
        let mut matrix = vec![0.0; 4];
        let mut rhs = vec![0.0; 2];
        for point in quadrature.points() {
            let reference = point.coordinates[0];
            let coordinate = geometry.map(reference);
            let coefficient = (self.diffusion)(coordinate);
            let source = (self.source)(coordinate);
            if !coefficient.is_finite() || coefficient <= 0.0 || !source.is_finite() {
                return Err(invalid_discretization(
                    "scalar elliptic coefficient must be finite and positive and source finite",
                ));
            }
            let shapes = [0.5 * (1.0 - reference), 0.5 * (1.0 + reference)];
            let weight = point.weight * geometry.jacobian();
            for row in 0..2 {
                rhs[row] += weight * shapes[row] * source;
                for column in 0..2 {
                    matrix[2 * row + column] +=
                        weight * coefficient * derivatives[row] * derivatives[column];
                }
            }
        }
        LocalContribution::new(2, 2, matrix, rhs)
    }
}

#[derive(Debug, Clone, Copy)]
struct NaturalBoundaryPacket {
    vertex: usize,
    free: Option<DofId>,
    flux: f64,
}

struct EllipticAssemblyWork<'a, K: ?Sized, S: ?Sized> {
    mesh: &'a LineMesh,
    operator: EllipticCell<'a, K, S>,
    quadrature: &'a QuadratureRule,
    essential: [Option<f64>; 2],
    free_indices: &'a [Option<DofId>],
    reduced_target: AssemblyTargetId,
    full_target: AssemblyTargetId,
    natural_boundaries: Vec<NaturalBoundaryPacket>,
}

impl<K: ?Sized, S: ?Sized> std::fmt::Debug for EllipticAssemblyWork<'_, K, S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EllipticAssemblyWork")
            .field("cells", &self.mesh.cell_count())
            .field("natural_boundaries", &self.natural_boundaries.len())
            .finish_non_exhaustive()
    }
}

impl<K, S> AssemblyWork for EllipticAssemblyWork<'_, K, S>
where
    K: Fn(f64) -> f64 + Sync + ?Sized,
    S: Fn(f64) -> f64 + Sync + ?Sized,
{
    fn packet_set_identity(&self) -> AssemblyPacketSetIdentityV1 {
        AssemblyPacketSetIdentityV1::Unbound
    }

    fn packet_count(&self) -> usize {
        self.mesh.cell_count() + self.natural_boundaries.len()
    }

    fn evaluate(&self, packet_index: usize) -> Result<AssemblyPacket, Diagnostic> {
        if packet_index < self.mesh.cell_count() {
            self.cell_packet(CellId::new(packet_index))
        } else {
            let boundary_index = packet_index - self.mesh.cell_count();
            self.natural_boundaries
                .get(boundary_index)
                .copied()
                .ok_or_else(|| {
                    assembly_failed(format!(
                        "elliptic assembly packet {packet_index} is outside work count {}",
                        self.packet_count()
                    ))
                })
                .and_then(|boundary| self.natural_packet(boundary))
        }
    }
}

impl<K, S> EllipticAssemblyWork<'_, K, S>
where
    K: Fn(f64) -> f64 + ?Sized,
    S: Fn(f64) -> f64 + ?Sized,
{
    fn cell_packet(&self, cell: CellId) -> Result<AssemblyPacket, Diagnostic> {
        let geometry = self
            .mesh
            .cell_geometry(cell)
            .ok_or_else(|| assembly_failed("elliptic assembly received an invalid cell"))?;
        let local = self.operator.evaluate(&geometry, self.quadrature)?;
        let vertices = self
            .mesh
            .cell_vertices(cell)
            .ok_or_else(|| assembly_failed("elliptic assembly received invalid cell topology"))?
            .map(|vertex| vertex.index());
        let equations = vertices.map(|vertex| self.free_indices[vertex]);
        let unknowns = vertices.map(|vertex| self.local_unknown(vertex));
        AssemblyPacket::new(
            local,
            vec![
                TargetAssemblyMap::new(
                    self.reduced_target,
                    AssemblyMap::new(equations.to_vec(), unknowns.to_vec())?,
                ),
                TargetAssemblyMap::new(
                    self.full_target,
                    AssemblyMap::new(
                        vertices.map(|vertex| Some(DofId::new(vertex))).to_vec(),
                        vertices
                            .map(|vertex| LocalUnknown::Free(DofId::new(vertex)))
                            .to_vec(),
                    )?,
                ),
            ],
        )
    }

    fn natural_packet(
        &self,
        boundary: NaturalBoundaryPacket,
    ) -> Result<AssemblyPacket, Diagnostic> {
        let free = boundary.free.ok_or_else(|| {
            assembly_failed("a natural endpoint must map to a free algebraic unknown")
        })?;
        let full = DofId::new(boundary.vertex);
        AssemblyPacket::new(
            LocalContribution::new(1, 1, vec![0.0], vec![boundary.flux])?,
            vec![
                TargetAssemblyMap::new(
                    self.reduced_target,
                    AssemblyMap::new(vec![Some(free)], vec![LocalUnknown::Free(free)])?,
                ),
                TargetAssemblyMap::new(
                    self.full_target,
                    AssemblyMap::new(vec![Some(full)], vec![LocalUnknown::Free(full)])?,
                ),
            ],
        )
    }

    fn local_unknown(&self, vertex: usize) -> LocalUnknown {
        let essential = if vertex == 0 {
            self.essential[0]
        } else if vertex + 1 == self.mesh.vertex_count() {
            self.essential[1]
        } else {
            None
        };
        essential.map_or_else(
            || {
                LocalUnknown::Free(
                    self.free_indices[vertex]
                        .expect("every nonessential vertex has a free algebraic unknown"),
                )
            },
            LocalUnknown::Fixed,
        )
    }
}

fn require_segment_rule(quadrature: &QuadratureRule) -> Result<(), Diagnostic> {
    if quadrature.reference_cell() == ReferenceCell::segment() {
        Ok(())
    } else {
        Err(Diagnostic::error(
            codes::INVALID_QUADRATURE,
            "scalar elliptic P1 operator requires segment quadrature",
        ))
    }
}

fn invalid_discretization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_DISCRETIZATION, message)
}

fn assembly_failed(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::ASSEMBLY_FAILED, message)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use eqiora_solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, SolverPlan};

    use super::*;

    #[test]
    fn mixed_boundary_linear_solution_and_reaction_are_exact() {
        let mesh = LineMesh::uniform(0.0, 2.0, 8).unwrap();
        let boundary = ScalarBoundaryPair1d::new(
            ScalarBoundaryCondition1d::Essential(0.0),
            ScalarBoundaryCondition1d::Natural(10.0),
        )
        .unwrap();
        let solution = solve_scalar_elliptic_linear_fem(
            &mesh,
            &|_| 20.0,
            &|_| 0.0,
            boundary,
            &QuadratureRule::gauss_legendre(2).unwrap(),
            LinearSolveRequest::new(
                &REFERENCE_LINEAR_SOLVER,
                SolverPlan::new(
                    LinearSolver::ConjugateGradient,
                    1.0e-12,
                    1.0e-14,
                    NonZeroUsize::new(10_000).unwrap(),
                )
                .unwrap(),
            ),
        )
        .unwrap();

        assert!((solution.field().values().last().unwrap() - 1.0).abs() < 1.0e-12);
        assert!(
            solution
                .cell_gradients()
                .iter()
                .all(|gradient| (*gradient - 0.5).abs() < 1.0e-12)
        );
        assert!((solution.endpoint_reactions()[0].unwrap() + 10.0).abs() < 1.0e-12);
        assert_eq!(solution.endpoint_reactions()[1], None);
    }
}

#[cfg(test)]
mod neumann_tests;
