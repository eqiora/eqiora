use eqiora_core::{Diagnostic, ScalarType};
use eqiora_meshing::{
    CellId, GeometryMap, MeshEntity, MeshGeometry, QuadratureRule, RevisionCellFragment2d,
    SimplicialMesh,
};
use eqiora_solver::{
    DiagonalAvailability, LinearOperator, LinearOperatorProperties, LinearProblem,
    LinearSolveRequest, LinearSolver, PreconditionerPolicy, SolveReport,
};

use crate::discrete_space::{DiscreteSpace, SimplexP1BubbleSpace, SimplexP1Space};

const DIMENSION: usize = 2;
const MAX_DENSE_COEFFICIENTS: usize = 16_000_000;
const MAX_AUXILIARY_COEFFICIENTS: usize = 4_000_000;
const MAX_CONSTRAINT_ORTHOGONALIZATION_WORK: usize = 64_000_000;

pub(super) fn checked_sum(
    left: usize,
    right: usize,
    name: &'static str,
) -> Result<usize, Diagnostic> {
    left.checked_add(right)
        .ok_or_else(|| super::invalid(format!("ALE FSI remesh {name} overflows usize")))
}

pub(super) fn checked_product(
    left: usize,
    right: usize,
    name: &'static str,
) -> Result<usize, Diagnostic> {
    left.checked_mul(right)
        .ok_or_else(|| super::invalid(format!("ALE FSI remesh {name} overflows usize")))
}

pub(super) fn require_auxiliary_budget(
    coefficient_count: usize,
    name: &'static str,
) -> Result<(), Diagnostic> {
    if coefficient_count > MAX_AUXILIARY_COEFFICIENTS {
        return Err(super::invalid(format!(
            "ALE FSI remesh {name} exceeds the bounded reference auxiliary-storage policy",
        )));
    }
    Ok(())
}

pub(super) fn require_constraint_work_budget(
    row_count: usize,
    row_width: usize,
) -> Result<(), Diagnostic> {
    let pair_count = checked_product(row_count, row_count, "constraint row-pair count")?;
    let work = checked_product(
        pair_count,
        row_width,
        "constraint orthogonalization work estimate",
    )?;
    if work > MAX_CONSTRAINT_ORTHOGONALIZATION_WORK {
        return Err(super::invalid(
            "ALE FSI remesh constraints exceed the bounded reference orthogonalization-work policy",
        ));
    }
    Ok(())
}

pub(super) fn require_dense_dimension(dimension: usize) -> Result<usize, Diagnostic> {
    let coefficient_count = checked_product(dimension, dimension, "dense operator shape")?;
    if dimension == 0 || coefficient_count > MAX_DENSE_COEFFICIENTS {
        return Err(super::invalid(
            "ALE FSI remesh dense operator exceeds the bounded reference resource policy",
        ));
    }
    Ok(coefficient_count)
}

pub(super) fn dense_zeroed(dimension: usize) -> Result<Vec<f64>, Diagnostic> {
    let coefficient_count = require_dense_dimension(dimension)?;
    Ok(vec![0.0; coefficient_count])
}

#[derive(Debug, Clone)]
pub(super) struct CellBasis2d {
    pub(super) values: Vec<f64>,
    pub(super) gradients: Vec<[f64; DIMENSION]>,
}

#[derive(Debug, Clone)]
pub(super) struct DenseSymmetricOperator {
    dimension: usize,
    values: Vec<f64>,
}

impl DenseSymmetricOperator {
    pub(super) fn new(values: Vec<f64>, dimension: usize) -> Result<Self, Diagnostic> {
        let expected = dimension
            .checked_mul(dimension)
            .ok_or_else(|| super::invalid("ALE FSI remesh dense operator shape overflowed"))?;
        if dimension == 0
            || expected > MAX_DENSE_COEFFICIENTS
            || values.len() != expected
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(super::invalid(
                "ALE FSI remesh dense operator must be finite, square, nonempty, and within the reference resource bound",
            ));
        }
        for row in 0..dimension {
            for column in 0..row {
                let left = values[row * dimension + column];
                let right = values[column * dimension + row];
                let tolerance = 16_384.0 * f64::EPSILON * (1.0 + left.abs().max(right.abs()));
                if (left - right).abs() > tolerance {
                    return Err(super::invalid(
                        "ALE FSI remesh operator is not numerically symmetric",
                    ));
                }
            }
        }
        Ok(Self { dimension, values })
    }
}

impl LinearOperator for DenseSymmetricOperator {
    fn rows(&self) -> usize {
        self.dimension
    }

    fn columns(&self) -> usize {
        self.dimension
    }

    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        if input.len() != self.dimension
            || output.len() != self.dimension
            || input.iter().any(|value| !value.is_finite())
        {
            return Err(super::invalid(
                "ALE FSI remesh dense action received an incompatible or non-finite vector",
            ));
        }
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.values[row * self.dimension..(row + 1) * self.dimension]
                .iter()
                .zip(input)
                .map(|(coefficient, input)| coefficient * input)
                .sum();
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(super::invalid(
                "ALE FSI remesh dense action produced a non-finite value",
            ));
        }
        Ok(())
    }

    fn diagonal(&self, output: &mut [f64]) -> Result<DiagonalAvailability, Diagnostic> {
        if output.len() != self.dimension {
            return Err(super::invalid(
                "ALE FSI remesh diagonal output has the wrong shape",
            ));
        }
        for (index, value) in output.iter_mut().enumerate() {
            *value = self.values[index * self.dimension + index];
        }
        Ok(DiagonalAvailability::Available)
    }
}

pub(super) fn require_projection_solver(solver: LinearSolveRequest<'_>) -> Result<(), Diagnostic> {
    if solver.plan().algorithm() != LinearSolver::MinimumResidual
        || solver.plan().preconditioner() != PreconditionerPolicy::Identity
    {
        return Err(super::invalid(
            "ALE FSI remesh projection requires the common identity-preconditioned MINRES plan",
        ));
    }
    solver.backend().capabilities().require_problem(
        solver.plan(),
        ScalarType::F64,
        LinearOperatorProperties::SymmetricIndefinite,
    )?;
    Ok(())
}

pub(super) fn solve_dense(
    values: Vec<f64>,
    right_hand_side: &[f64],
    properties: LinearOperatorProperties,
    solver: LinearSolveRequest<'_>,
) -> Result<(Vec<f64>, SolveReport, f64), Diagnostic> {
    let operator = DenseSymmetricOperator::new(values, right_hand_side.len())?;
    solver
        .backend()
        .capabilities()
        .require_problem(solver.plan(), ScalarType::F64, properties)?;
    let problem = LinearProblem::new(&operator, right_hand_side, properties)?;
    let (solution, report) = solver.solve(&problem)?.into_parts();
    let residual = residual_norm(&operator, &solution, right_hand_side)?;
    let tolerance = 8.0 * report.residual_target()
        + 65_536.0
            * f64::EPSILON
            * (1.0
                + right_hand_side
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0_f64, f64::max));
    if residual > tolerance {
        return Err(super::invalid(
            "ALE FSI remesh independent dense residual replay rejected the common-solver output",
        ));
    }
    Ok((solution, report, residual))
}

pub(super) fn residual_norm(
    operator: &dyn LinearOperator,
    solution: &[f64],
    right_hand_side: &[f64],
) -> Result<f64, Diagnostic> {
    let mut action = vec![0.0; operator.rows()];
    operator.apply(solution, &mut action)?;
    euclidean_norm(
        &action
            .iter()
            .zip(right_hand_side)
            .map(|(left, right)| left - right)
            .collect::<Vec<_>>(),
    )
}

pub(super) fn euclidean_norm(values: &[f64]) -> Result<f64, Diagnostic> {
    let squared = values.iter().map(|value| value * value).sum::<f64>();
    let norm = squared.sqrt();
    if norm.is_finite() {
        Ok(norm)
    } else {
        Err(super::invalid(
            "ALE FSI remesh norm replay produced a non-finite value",
        ))
    }
}

pub(super) fn require_quadrature(quadrature: &QuadratureRule) -> Result<(), Diagnostic> {
    if quadrature.reference_cell() != eqiora_meshing::ReferenceCell::simplex(DIMENSION)?
        || quadrature.polynomial_exactness().unwrap_or(0) < 8
    {
        return Err(super::invalid(
            "ALE FSI remesh projection requires triangle quadrature exact through total degree eight",
        ));
    }
    Ok(())
}

pub(super) fn cell_basis(
    mesh: &SimplicialMesh,
    cell: CellId,
    physical: [f64; DIMENSION],
    bubble: bool,
) -> Result<CellBasis2d, Diagnostic> {
    let map = mesh
        .geometry_map(MeshEntity::new(DIMENSION, cell.index()))
        .ok_or_else(|| super::invalid("ALE FSI remesh overlap names an absent mesh cell"))?;
    let inverse = map.inverse_jacobian()?;
    let delta = [physical[0] - map.origin()[0], physical[1] - map.origin()[1]];
    let xi = [
        inverse[0] * delta[0] + inverse[1] * delta[1],
        inverse[2] * delta[0] + inverse[3] * delta[1],
    ];
    let lambda = [1.0 - xi[0] - xi[1], xi[0], xi[1]];
    let coordinate_scale = mesh
        .vertices()
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let tolerance = 131_072.0 * f64::EPSILON * (1.0 + coordinate_scale);
    if lambda
        .iter()
        .any(|value| !value.is_finite() || *value < -tolerance || *value > 1.0 + tolerance)
    {
        return Err(super::invalid(
            "ALE FSI remesh quadrature point is outside an asserted overlap parent cell",
        ));
    }
    let reference = canonical_reference_point(lambda, tolerance)?;
    let tabulation = if bubble {
        SimplexP1BubbleSpace::new(DIMENSION)?.tabulate(&reference)?
    } else {
        SimplexP1Space::new(DIMENSION)?.tabulate(&reference)?
    };
    let values = tabulation.values().to_vec();
    let gradients = (0..values.len())
        .map(|local| {
            let reference_gradient = tabulation.gradient(local).ok_or_else(|| {
                super::invalid("ALE FSI remesh basis tabulation omitted a local gradient")
            })?;
            Ok([
                inverse[0] * reference_gradient[0] + inverse[2] * reference_gradient[1],
                inverse[1] * reference_gradient[0] + inverse[3] * reference_gradient[1],
            ])
        })
        .collect::<Result<Vec<_>, Diagnostic>>()?;
    Ok(CellBasis2d { values, gradients })
}

fn canonical_reference_point(
    mut barycentric: [f64; DIMENSION + 1],
    tolerance: f64,
) -> Result<[f64; DIMENSION], Diagnostic> {
    for coordinate in &mut barycentric {
        if *coordinate < 0.0 && *coordinate >= -tolerance {
            *coordinate = 0.0;
        }
        if *coordinate > 1.0 && *coordinate <= 1.0 + tolerance {
            *coordinate = 1.0;
        }
    }
    let sum = barycentric.iter().sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 || (sum - 1.0).abs() > 4.0 * tolerance {
        return Err(super::invalid(
            "ALE FSI remesh cannot canonicalize an overlap point in its parent simplex",
        ));
    }
    for coordinate in &mut barycentric {
        *coordinate /= sum;
    }
    Ok([barycentric[1], barycentric[2]])
}

pub(super) fn integrate_physical_triangle<F>(
    fragment: &RevisionCellFragment2d,
    quadrature: &QuadratureRule,
    mut evaluate: F,
) -> Result<(), Diagnostic>
where
    F: FnMut([f64; DIMENSION], f64) -> Result<(), Diagnostic>,
{
    let triangle = fragment.triangle();
    let measure_scale = 2.0 * fragment.area();
    if triangle.iter().flatten().any(|value| !value.is_finite())
        || !measure_scale.is_finite()
        || measure_scale <= 0.0
    {
        return Err(super::invalid(
            "ALE FSI remesh overlap integration requires an admitted positive finite fragment",
        ));
    }
    let edges = [
        [
            triangle[1][0] - triangle[0][0],
            triangle[1][1] - triangle[0][1],
        ],
        [
            triangle[2][0] - triangle[0][0],
            triangle[2][1] - triangle[0][1],
        ],
    ];
    if edges.iter().flatten().any(|value| !value.is_finite()) {
        return Err(super::invalid(
            "ALE FSI remesh overlap forward map has a non-finite edge",
        ));
    }
    for point in quadrature.points() {
        if point.coordinates.len() != DIMENSION
            || point.coordinates.iter().any(|value| !value.is_finite())
        {
            return Err(super::invalid(
                "ALE FSI remesh overlap quadrature point has an incompatible coordinate",
            ));
        }
        let physical = std::array::from_fn(|axis| {
            point.coordinates[1].mul_add(
                edges[1][axis],
                point.coordinates[0].mul_add(edges[0][axis], triangle[0][axis]),
            )
        });
        let weight = point.weight * measure_scale;
        if physical.iter().any(|value| !value.is_finite()) || !weight.is_finite() || weight <= 0.0 {
            return Err(super::invalid(
                "ALE FSI remesh overlap forward quadrature produced a non-finite point or non-positive measure",
            ));
        }
        evaluate(physical, weight)?;
    }
    Ok(())
}

pub(super) fn integrate_cell<F>(
    mesh: &SimplicialMesh,
    cell: CellId,
    quadrature: &QuadratureRule,
    mut evaluate: F,
) -> Result<(), Diagnostic>
where
    F: FnMut([f64; DIMENSION], f64) -> Result<(), Diagnostic>,
{
    let map = mesh
        .geometry_map(MeshEntity::new(DIMENSION, cell.index()))
        .ok_or_else(|| super::invalid("ALE FSI remesh integration names an absent mesh cell"))?;
    for point in quadrature.points() {
        let mut physical = [0.0; DIMENSION];
        map.map_point(&point.coordinates, &mut physical)?;
        let weight = point.weight * map.measure_scale();
        evaluate(physical, weight)?;
    }
    Ok(())
}

/// Deterministically reduce a possibly dependent constraint list to an
/// equivalent orthonormal row set.  Dependent rows are admitted only when the
/// right-hand side carries the same linear dependency.
pub(super) fn independent_constraints(
    rows: Vec<(Vec<f64>, f64)>,
) -> Result<Vec<(Vec<f64>, f64)>, Diagnostic> {
    let mut basis: Vec<(Vec<f64>, f64)> = Vec::new();
    for (mut row, mut right_hand_side) in rows {
        if row.iter().any(|value| !value.is_finite()) || !right_hand_side.is_finite() {
            return Err(super::invalid(
                "ALE FSI remesh constraint contains a non-finite coefficient",
            ));
        }
        let original_scale = row.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
        for (direction, direction_rhs) in &basis {
            let coefficient = row
                .iter()
                .zip(direction)
                .map(|(left, right)| left * right)
                .sum::<f64>();
            for (value, direction) in row.iter_mut().zip(direction) {
                *value -= coefficient * direction;
            }
            right_hand_side -= coefficient * direction_rhs;
        }
        let norm = euclidean_norm(&row)?;
        let threshold = 262_144.0 * f64::EPSILON * (1.0 + original_scale);
        if norm <= threshold {
            if right_hand_side.abs() > 262_144.0 * f64::EPSILON * (1.0 + right_hand_side.abs()) {
                return Err(super::invalid(
                    "ALE FSI remesh dependent constraints have an inconsistent right-hand side",
                ));
            }
            continue;
        }
        for value in &mut row {
            *value /= norm;
        }
        right_hand_side /= norm;
        basis.push((row, right_hand_side));
    }
    if basis.is_empty() {
        return Err(super::invalid(
            "ALE FSI remesh velocity projection has no independent physical constraint",
        ));
    }
    Ok(basis)
}
