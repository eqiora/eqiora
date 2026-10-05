use std::f64::consts::PI;
use std::num::NonZeroUsize;

use eqiora_assembly::{AssemblyMap, CooAssembler, DofId, LocalContribution, LocalUnknown};
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_meshing::{
    CellId, LineMesh, QuadratureRule, ReferenceCell, SegmentGeometry1d, VertexId,
};
use eqiora_solver::{
    LinearOperatorProperties, LinearProblem, LinearSolveRequest, LinearSolver,
    REFERENCE_LINEAR_SOLVER, SolveReport, SolverPlan,
};

use crate::elliptic::{
    ScalarBoundaryCondition1d, ScalarBoundaryPair1d, solve_scalar_elliptic_linear_fem,
};
use crate::operator::LocalOperator;

/// Finite Dirichlet values at the two endpoints of a line mesh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirichletBoundary1d {
    left: f64,
    right: f64,
}

impl DirichletBoundary1d {
    /// Construct endpoint values.
    ///
    /// # Errors
    /// Returns `EQ0801` if either value is non-finite.
    pub fn new(left: f64, right: f64) -> Result<Self, Diagnostic> {
        if !left.is_finite() || !right.is_finite() {
            return Err(invalid_discretization(
                "Dirichlet boundary values must be finite",
            ));
        }
        Ok(Self { left, right })
    }

    /// Left endpoint value.
    #[must_use]
    pub const fn left(self) -> f64 {
        self.left
    }

    /// Right endpoint value.
    #[must_use]
    pub const fn right(self) -> f64 {
        self.right
    }
}

/// A one-dimensional piecewise-linear field used for method-neutral evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct PiecewiseLinearField1d {
    coordinates: Vec<f64>,
    values: Vec<f64>,
}

impl PiecewiseLinearField1d {
    /// Construct a field on strictly increasing finite support coordinates.
    ///
    /// # Errors
    /// Returns `EQ0801` for shape mismatch, insufficient support, or
    /// non-finite/unordered data.
    pub fn new(coordinates: Vec<f64>, values: Vec<f64>) -> Result<Self, Diagnostic> {
        if coordinates.len() != values.len() || coordinates.len() < 2 {
            return Err(invalid_discretization(
                "piecewise-linear coordinates and values require equal length of at least two",
            ));
        }
        if coordinates.iter().any(|value| !value.is_finite())
            || values.iter().any(|value| !value.is_finite())
            || coordinates.windows(2).any(|pair| pair[1] <= pair[0])
        {
            return Err(invalid_discretization(
                "piecewise-linear field data must be finite with increasing coordinates",
            ));
        }
        Ok(Self {
            coordinates,
            values,
        })
    }

    /// Support coordinates.
    #[must_use]
    pub fn coordinates(&self) -> &[f64] {
        &self.coordinates
    }

    /// Support values.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Evaluate the field inside its closed support interval.
    #[must_use]
    pub fn evaluate(&self, coordinate: f64) -> Option<f64> {
        if !coordinate.is_finite()
            || coordinate < self.coordinates[0]
            || coordinate > self.coordinates[self.coordinates.len() - 1]
        {
            return None;
        }
        if coordinate == self.coordinates[self.coordinates.len() - 1] {
            return self.values.last().copied();
        }
        let right = self
            .coordinates
            .partition_point(|support| *support <= coordinate);
        let left = right - 1;
        let fraction = (coordinate - self.coordinates[left])
            / (self.coordinates[right] - self.coordinates[left]);
        Some(self.values[left] + fraction * (self.values[right] - self.values[left]))
    }

    /// Continuous `L2` error against an exact field.
    ///
    /// # Errors
    /// Returns a numerical diagnostic for an incompatible quadrature rule or
    /// a non-finite exact value/error accumulation.
    pub fn l2_error<E>(&self, exact: E, quadrature: &QuadratureRule) -> Result<f64, Diagnostic>
    where
        E: Fn(f64) -> f64,
    {
        require_reference_cell(quadrature, ReferenceCell::segment())?;
        let mut squared_error = 0.0;
        for segment in 0..self.coordinates.len() - 1 {
            let left = self.coordinates[segment];
            let right = self.coordinates[segment + 1];
            let geometry = SegmentGeometry1d::new(left, right)?;
            for point in quadrature.points() {
                let reference = point.coordinates[0];
                let coordinate = geometry.map(reference);
                let left_shape = 0.5 * (1.0 - reference);
                let right_shape = 0.5 * (1.0 + reference);
                let approximation =
                    left_shape * self.values[segment] + right_shape * self.values[segment + 1];
                let exact_value = exact(coordinate);
                if !exact_value.is_finite() {
                    return Err(invalid_discretization(
                        "exact comparison field returned a non-finite value",
                    ));
                }
                squared_error +=
                    point.weight * geometry.jacobian() * (approximation - exact_value).powi(2);
            }
        }
        if !squared_error.is_finite() || squared_error < 0.0 {
            return Err(invalid_discretization(
                "L2 error accumulation is non-finite or negative",
            ));
        }
        Ok(squared_error.sqrt())
    }
}

/// Cell-centered finite-volume solution with an explicit continuous evidence
/// reconstruction.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarEllipticFvmSolution1d {
    unknown_coordinates: Vec<f64>,
    unknown_values: Vec<f64>,
    reconstruction: PiecewiseLinearField1d,
    solve_report: SolveReport,
}

/// Compatibility name for the original Poisson numerical precursor.
pub type PoissonSolution1d = ScalarEllipticFvmSolution1d;

impl ScalarEllipticFvmSolution1d {
    /// Coordinates carrying algebraic unknowns before reconstruction.
    #[must_use]
    pub fn unknown_coordinates(&self) -> &[f64] {
        &self.unknown_coordinates
    }

    /// Algebraic unknown values before reconstruction.
    #[must_use]
    pub fn unknown_values(&self) -> &[f64] {
        &self.unknown_values
    }

    /// Common piecewise-linear evidence view.
    #[must_use]
    pub const fn reconstruction(&self) -> &PiecewiseLinearField1d {
        &self.reconstruction
    }

    /// Krylov iterations used by the reference solve.
    #[must_use]
    pub const fn linear_iterations(&self) -> usize {
        self.solve_report.completed_iterations()
    }

    /// Final algebraic residual norm.
    #[must_use]
    pub const fn residual_norm(&self) -> f64 {
        self.solve_report.true_residual_norm()
    }

    /// Complete solver/backend/execution evidence for the accepted system.
    #[must_use]
    pub const fn solve_report(&self) -> &SolveReport {
        &self.solve_report
    }
}

/// Solve `-d/dx(k du/dx) = source` using cell-centered two-point-flux FVM
/// with a positive constant `k`.
///
/// Source integrals are cell-local; diffusion is an interior/boundary-facet
/// local operator. Both contribution families scatter through the same
/// [`AssemblyMap`] contract as FEM.
///
/// # Errors
/// Returns a numerical diagnostic for an invalid coefficient, incompatible
/// quadrature, invalid source data, assembly failure, or linear-solver
/// failure.
pub fn solve_scalar_elliptic_cell_fvm<S>(
    mesh: &LineMesh,
    diffusion: f64,
    source: &S,
    boundary: DirichletBoundary1d,
    quadrature: &QuadratureRule,
    solver: LinearSolveRequest<'_>,
) -> Result<ScalarEllipticFvmSolution1d, Diagnostic>
where
    S: Fn(f64) -> f64 + ?Sized,
{
    require_reference_cell(quadrature, ReferenceCell::segment())?;
    if !diffusion.is_finite() || diffusion <= 0.0 {
        return Err(invalid_discretization(
            "finite-volume diffusion coefficient must be finite and positive",
        ));
    }
    let mut assembler = CooAssembler::new(mesh.cell_count())?;
    let source_operator = FvmSourceCell { source };
    let point_rule = QuadratureRule::point();
    let interior_flux = InteriorTwoPointFlux;
    let boundary_flux = BoundaryTwoPointFlux;

    for cell in mesh.cells() {
        let geometry = mesh
            .cell_geometry(cell)
            .expect("mesh iterator yields valid cell geometry");
        let local = source_operator.evaluate(&geometry, quadrature)?;
        let dof = DofId::new(cell.index());
        let map = AssemblyMap::new(vec![Some(dof)], vec![LocalUnknown::Free(dof)])?;
        assembler.scatter(&map, &local)?;
    }

    for facet in mesh.facets() {
        let facet_geometry = mesh
            .facet_geometry(facet)
            .expect("mesh iterator yields valid facet geometry");
        let cells = mesh
            .facet_cells(facet)
            .expect("mesh iterator yields valid facet topology");
        match (cells.minus, cells.plus) {
            (Some(minus), Some(plus)) => {
                let context = InteriorFluxContext {
                    minus_center: cell_center(mesh, minus),
                    plus_center: cell_center(mesh, plus),
                    diffusion,
                };
                let local = interior_flux.evaluate(&context, &point_rule)?;
                let minus_dof = DofId::new(minus.index());
                let plus_dof = DofId::new(plus.index());
                let map = AssemblyMap::new(
                    vec![Some(minus_dof), Some(plus_dof)],
                    vec![LocalUnknown::Free(minus_dof), LocalUnknown::Free(plus_dof)],
                )?;
                assembler.scatter(&map, &local)?;
            }
            (None, Some(cell)) => {
                assemble_boundary_flux(
                    &mut assembler,
                    &boundary_flux,
                    &point_rule,
                    cell,
                    cell_center(mesh, cell) - facet_geometry.coordinate,
                    diffusion,
                    boundary.left(),
                )?;
            }
            (Some(cell), None) => {
                assemble_boundary_flux(
                    &mut assembler,
                    &boundary_flux,
                    &point_rule,
                    cell,
                    facet_geometry.coordinate - cell_center(mesh, cell),
                    diffusion,
                    boundary.right(),
                )?;
            }
            (None, None) => {
                return Err(invalid_discretization(
                    "line facet has no adjacent finite-volume cell",
                ));
            }
        }
    }

    let system = assembler.finish()?;
    let problem = LinearProblem::new(
        system.matrix(),
        system.rhs(),
        LinearOperatorProperties::SymmetricPositiveDefinite,
    )?;
    let solved = solver.solve(&problem)?;
    let mut unknown_coordinates = Vec::with_capacity(mesh.cell_count());
    for cell in mesh.cells() {
        unknown_coordinates.push(cell_center(mesh, cell));
    }
    let unknown_values = solved.values().to_vec();
    let mut reconstruction_coordinates = Vec::with_capacity(mesh.cell_count() + 2);
    let mut reconstruction_values = Vec::with_capacity(mesh.cell_count() + 2);
    reconstruction_coordinates.push(
        mesh.vertex_coordinate(VertexId::new(0))
            .expect("line mesh has a left endpoint"),
    );
    reconstruction_values.push(boundary.left());
    reconstruction_coordinates.extend(&unknown_coordinates);
    reconstruction_values.extend(&unknown_values);
    reconstruction_coordinates.push(
        mesh.vertex_coordinate(VertexId::new(mesh.vertex_count() - 1))
            .expect("line mesh has a right endpoint"),
    );
    reconstruction_values.push(boundary.right());

    Ok(ScalarEllipticFvmSolution1d {
        unknown_coordinates,
        unknown_values,
        reconstruction: PiecewiseLinearField1d::new(
            reconstruction_coordinates,
            reconstruction_values,
        )?,
        solve_report: solved.report().clone(),
    })
}

/// One refinement row comparing the two realizations of the same problem.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarEllipticComparisonRow1d {
    /// Number of primal mesh cells.
    pub cells: usize,
    /// Largest cell measure.
    pub max_cell_measure: f64,
    /// Continuous L2 error of the FEM reconstruction.
    pub fem_l2_error: f64,
    /// Continuous L2 error of the FVM reconstruction.
    pub fvm_l2_error: f64,
    /// FEM order relative to the preceding row.
    pub fem_order: Option<f64>,
    /// FVM order relative to the preceding row.
    pub fvm_order: Option<f64>,
    /// Dimensionless global FEM balance defect, including recovered endpoint
    /// reactions and the integrated source.
    pub fem_relative_balance_error: f64,
    /// Dimensionless global FVM balance defect, including outward endpoint
    /// fluxes and the integrated source.
    pub fvm_relative_balance_error: f64,
}

/// Compatibility name for the original Poisson numerical precursor.
pub type PoissonComparisonRow = ScalarEllipticComparisonRow1d;

/// Reproducible comparison for `-u'' = pi^2 sin(pi x)`, `u(0)=u(1)=0`.
///
/// Both methods use the same mesh revision, source callback, cell quadrature,
/// sparse assembler, CG oracle, error quadrature, and exact solution. Their
/// local operators, DOF placement, and reconstruction remain method-specific.
///
/// # Errors
/// Returns a numerical diagnostic when levels are not strictly increasing or
/// when either solve/evidence calculation fails.
pub fn compare_sine_poisson_1d(
    cell_counts: &[usize],
) -> Result<Vec<PoissonComparisonRow>, Diagnostic> {
    let source = |coordinate: f64| PI.powi(2) * (PI * coordinate).sin();
    let exact = |coordinate: f64| (PI * coordinate).sin();
    compare_scalar_elliptic_dirichlet_1d(
        [0.0, 1.0],
        1.0,
        &source,
        DirichletBoundary1d::new(0.0, 0.0)?,
        &exact,
        cell_counts,
    )
}

pub(crate) fn compare_scalar_elliptic_dirichlet_1d<S, E>(
    interval: [f64; 2],
    diffusion: f64,
    source: &S,
    boundary: DirichletBoundary1d,
    exact: &E,
    cell_counts: &[usize],
) -> Result<Vec<PoissonComparisonRow>, Diagnostic>
where
    S: Fn(f64) -> f64 + Sync + ?Sized,
    E: Fn(f64) -> f64 + ?Sized,
{
    if cell_counts.len() < 2
        || cell_counts
            .windows(2)
            .any(|pair| pair[0] < 2 || pair[1] <= pair[0])
    {
        return Err(invalid_discretization(
            "Poisson comparison requires at least two strictly increasing cell counts >= 2",
        ));
    }
    if !diffusion.is_finite() || diffusion <= 0.0 {
        return Err(invalid_discretization(
            "comparison diffusion coefficient must be finite and positive",
        ));
    }
    let integration = QuadratureRule::gauss_legendre(4)?;
    let mut rows: Vec<PoissonComparisonRow> = Vec::with_capacity(cell_counts.len());

    for &cells in cell_counts {
        let mesh = LineMesh::uniform(interval[0], interval[1], cells)?;
        let plan = SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(cells.saturating_mul(8).max(32))
                .expect("comparison iteration bound is nonzero"),
        )?;
        let solver = LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, plan);
        let fem_boundary = ScalarBoundaryPair1d::new(
            ScalarBoundaryCondition1d::Essential(boundary.left()),
            ScalarBoundaryCondition1d::Essential(boundary.right()),
        )?;
        let fem = solve_scalar_elliptic_linear_fem(
            &mesh,
            &|_| diffusion,
            source,
            fem_boundary,
            &integration,
            solver,
        )?;
        let fvm = solve_scalar_elliptic_cell_fvm(
            &mesh,
            diffusion,
            source,
            boundary,
            &integration,
            solver,
        )?;
        let fem_l2_error = fem.field().l2_error(exact, &integration)?;
        let fvm_l2_error = fvm.reconstruction().l2_error(exact, &integration)?;
        let integrated_source = integrate_source(&mesh, source, &integration)?;
        let reactions = fem.endpoint_reactions();
        let lower_reaction = reactions[0].expect("Dirichlet comparison has a lower reaction");
        let upper_reaction = reactions[1].expect("Dirichlet comparison has an upper reaction");
        let fem_relative_balance_error = relative_balance_error(
            lower_reaction + upper_reaction + integrated_source,
            [lower_reaction, upper_reaction, integrated_source],
        );
        let [lower_flux, upper_flux] =
            fvm_boundary_fluxes(&mesh, diffusion, boundary, fvm.unknown_values())?;
        let fvm_relative_balance_error = relative_balance_error(
            lower_flux + upper_flux + integrated_source,
            [lower_flux, upper_flux, integrated_source],
        );
        let (fem_order, fvm_order) = rows.last().map_or((None, None), |previous| {
            let mesh_ratio = previous.max_cell_measure / mesh.max_cell_measure();
            (
                Some((previous.fem_l2_error / fem_l2_error).ln() / mesh_ratio.ln()),
                Some((previous.fvm_l2_error / fvm_l2_error).ln() / mesh_ratio.ln()),
            )
        });
        rows.push(PoissonComparisonRow {
            cells,
            max_cell_measure: mesh.max_cell_measure(),
            fem_l2_error,
            fvm_l2_error,
            fem_order,
            fvm_order,
            fem_relative_balance_error,
            fvm_relative_balance_error,
        });
    }
    Ok(rows)
}

struct FvmSourceCell<'a, S: ?Sized> {
    source: &'a S,
}

impl<S> LocalOperator<SegmentGeometry1d> for FvmSourceCell<'_, S>
where
    S: Fn(f64) -> f64 + ?Sized,
{
    fn evaluate(
        &self,
        geometry: &SegmentGeometry1d,
        quadrature: &QuadratureRule,
    ) -> Result<LocalContribution, Diagnostic> {
        require_reference_cell(quadrature, ReferenceCell::segment())?;
        let mut integrated_source = 0.0;
        for point in quadrature.points() {
            let source = (self.source)(geometry.map(point.coordinates[0]));
            if !source.is_finite() {
                return Err(invalid_discretization(
                    "Poisson source returned a non-finite value",
                ));
            }
            integrated_source += point.weight * geometry.jacobian() * source;
        }
        LocalContribution::new(1, 1, vec![0.0], vec![integrated_source])
    }
}

struct InteriorFluxContext {
    minus_center: f64,
    plus_center: f64,
    diffusion: f64,
}

struct InteriorTwoPointFlux;

impl LocalOperator<InteriorFluxContext> for InteriorTwoPointFlux {
    fn evaluate(
        &self,
        context: &InteriorFluxContext,
        quadrature: &QuadratureRule,
    ) -> Result<LocalContribution, Diagnostic> {
        require_reference_cell(quadrature, ReferenceCell::point())?;
        let distance = context.plus_center - context.minus_center;
        let transmissibility = checked_transmissibility(distance, context.diffusion)?;
        LocalContribution::new(
            2,
            2,
            vec![
                transmissibility,
                -transmissibility,
                -transmissibility,
                transmissibility,
            ],
            vec![0.0, 0.0],
        )
    }
}

struct BoundaryFluxContext {
    distance: f64,
    diffusion: f64,
}

struct BoundaryTwoPointFlux;

impl LocalOperator<BoundaryFluxContext> for BoundaryTwoPointFlux {
    fn evaluate(
        &self,
        context: &BoundaryFluxContext,
        quadrature: &QuadratureRule,
    ) -> Result<LocalContribution, Diagnostic> {
        require_reference_cell(quadrature, ReferenceCell::point())?;
        let transmissibility = checked_transmissibility(context.distance, context.diffusion)?;
        LocalContribution::new(1, 2, vec![transmissibility, -transmissibility], vec![0.0])
    }
}

fn assemble_boundary_flux(
    assembler: &mut CooAssembler,
    operator: &BoundaryTwoPointFlux,
    quadrature: &QuadratureRule,
    cell: CellId,
    distance: f64,
    diffusion: f64,
    boundary_value: f64,
) -> Result<(), Diagnostic> {
    let local = operator.evaluate(
        &BoundaryFluxContext {
            distance,
            diffusion,
        },
        quadrature,
    )?;
    let dof = DofId::new(cell.index());
    let map = AssemblyMap::new(
        vec![Some(dof)],
        vec![LocalUnknown::Free(dof), LocalUnknown::Fixed(boundary_value)],
    )?;
    assembler.scatter(&map, &local)
}

fn integrate_source<S>(
    mesh: &LineMesh,
    source: &S,
    quadrature: &QuadratureRule,
) -> Result<f64, Diagnostic>
where
    S: Fn(f64) -> f64 + ?Sized,
{
    let operator = FvmSourceCell { source };
    let mut integral = 0.0;
    for cell in mesh.cells() {
        let geometry = mesh
            .cell_geometry(cell)
            .expect("mesh iterator yields valid cell geometry");
        let contribution = operator.evaluate(&geometry, quadrature)?;
        integral += contribution.rhs()[0];
    }
    if integral.is_finite() {
        Ok(integral)
    } else {
        Err(invalid_discretization(
            "integrated source balance is non-finite",
        ))
    }
}

fn fvm_boundary_fluxes(
    mesh: &LineMesh,
    diffusion: f64,
    boundary: DirichletBoundary1d,
    cell_values: &[f64],
) -> Result<[f64; 2], Diagnostic> {
    if cell_values.len() != mesh.cell_count() {
        return Err(invalid_discretization(
            "finite-volume balance requires one value per cell",
        ));
    }
    let lower_cell = CellId::new(0);
    let upper_cell = CellId::new(mesh.cell_count() - 1);
    let lower_boundary = mesh
        .vertex_coordinate(VertexId::new(0))
        .expect("line mesh has a lower endpoint");
    let upper_boundary = mesh
        .vertex_coordinate(VertexId::new(mesh.vertex_count() - 1))
        .expect("line mesh has an upper endpoint");
    let lower_transmissibility =
        checked_transmissibility(cell_center(mesh, lower_cell) - lower_boundary, diffusion)?;
    let upper_transmissibility =
        checked_transmissibility(upper_boundary - cell_center(mesh, upper_cell), diffusion)?;
    Ok([
        lower_transmissibility * (boundary.left() - cell_values[0]),
        upper_transmissibility * (boundary.right() - cell_values[cell_values.len() - 1]),
    ])
}

fn relative_balance_error<const N: usize>(defect: f64, terms: [f64; N]) -> f64 {
    let scale = terms.iter().map(|term| term.abs()).sum::<f64>();
    defect.abs() / scale.max(f64::MIN_POSITIVE)
}

fn cell_center(mesh: &LineMesh, cell: CellId) -> f64 {
    mesh.cell_geometry(cell)
        .expect("valid cell ID has geometry")
        .center()
}

fn checked_transmissibility(distance: f64, diffusion: f64) -> Result<f64, Diagnostic> {
    if !distance.is_finite() || distance <= 0.0 || !diffusion.is_finite() || diffusion <= 0.0 {
        return Err(invalid_discretization(
            "finite-volume center distance and diffusion must be finite and positive",
        ));
    }
    let transmissibility = diffusion / distance;
    if !transmissibility.is_finite() {
        return Err(invalid_discretization(
            "finite-volume transmissibility must be finite",
        ));
    }
    Ok(transmissibility)
}

fn require_reference_cell(
    quadrature: &QuadratureRule,
    expected: ReferenceCell,
) -> Result<(), Diagnostic> {
    if quadrature.reference_cell() == expected {
        Ok(())
    } else {
        Err(Diagnostic::error(
            codes::INVALID_QUADRATURE,
            format!(
                "local operator requires {expected:?} quadrature, received {:?}",
                quadrature.reference_cell()
            ),
        ))
    }
}

fn invalid_discretization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_DISCRETIZATION, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_solver() -> LinearSolveRequest<'static> {
        LinearSolveRequest::new(
            &REFERENCE_LINEAR_SOLVER,
            SolverPlan::new(
                LinearSolver::ConjugateGradient,
                1.0e-12,
                1.0e-14,
                NonZeroUsize::new(10_000).unwrap(),
            )
            .unwrap(),
        )
    }

    #[test]
    fn methods_keep_their_native_unknown_locations() {
        let mesh = LineMesh::uniform(0.0, 1.0, 4).unwrap();
        let quadrature = QuadratureRule::gauss_legendre(3).unwrap();
        let boundary = DirichletBoundary1d::new(0.0, 0.0).unwrap();
        let source = |coordinate: f64| PI.powi(2) * (PI * coordinate).sin();
        let fem = solve_scalar_elliptic_linear_fem(
            &mesh,
            &|_| 1.0,
            &source,
            ScalarBoundaryPair1d::new(
                ScalarBoundaryCondition1d::Essential(boundary.left()),
                ScalarBoundaryCondition1d::Essential(boundary.right()),
            )
            .unwrap(),
            &quadrature,
            reference_solver(),
        )
        .unwrap();
        let fvm = solve_scalar_elliptic_cell_fvm(
            &mesh,
            1.0,
            &source,
            boundary,
            &quadrature,
            reference_solver(),
        )
        .unwrap();
        assert_eq!(&fem.field().coordinates()[1..4], &[0.25, 0.5, 0.75]);
        assert_eq!(fvm.unknown_coordinates(), &[0.125, 0.375, 0.625, 0.875]);
    }
}
