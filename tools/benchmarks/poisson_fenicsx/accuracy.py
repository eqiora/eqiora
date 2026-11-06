"""Dimension-checked Q1 analytic gates; no solver output is an expectation."""
from __future__ import annotations

import csv
import hashlib
import itertools
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LEVELS = (4, 8, 16, 32)
REFERENCE = ROOT / "verify/numerics/cartesian-poisson-fem-fvm/expected/convergence.csv"
REFERENCES = {2: REFERENCE, 3: ROOT / "verify/numerics/cartesian-poisson-3d-fem-fvm/expected/convergence.csv"}


def declared_levels(dimension: int) -> tuple[int, ...]:
    if dimension not in REFERENCES:
        raise ComparisonError(f"unsupported comparison dimension {dimension}")
    return LEVELS if dimension == 2 else (4, 8, 16)


class ComparisonError(ValueError):
    """The declared mathematical or measurement contract was not met."""


def error_budget(cells: int, reference: float, dimension: int = 2) -> float:
    """Dimension-general FEM port of the existing registered Cartesian error_budget.

    Owner: crates/eqiora-numerics/tests/support/cartesian_poisson.rs, with
    derivation in verify/numerics/cartesian-poisson-fem-fvm/references/README.md.
    Its scope is uniform Q1, 2-point load quadrature, zero essential boundary,
    unit coefficient, same residual policy, and 8-point error integration.
    Both tools have the same effective free operator; zero boundary identity
    rows in PETSc do not change that subspace. This is the inherited comparison
    allowance, not a new rigorous floating-point certification of either tool.
    The residual-to-error implication is independent of the Krylov algorithm.
    """
    if cells not in declared_levels(dimension):
        raise ComparisonError(f"no declared reference for N={cells}; supported {declared_levels(dimension)}")
    h = 1.0 / cells
    eigenvalue_floor = 4.0 * dimension * h**(dimension-2) * np.sin(np.pi*h/2.0)**2 / 3.0**(dimension-1)
    rhs_entry_bound = dimension * np.pi**2 * h**dimension
    coordinate_count_bound = cells**(dimension/2)
    residual_target = max(1e-14, 1e-12 * rhs_entry_bound * coordinate_count_bound)
    eps = np.finfo(np.float64).eps

    def gamma(operations: float) -> float:
        product = operations * eps
        return product / (1.0 - product)

    # N**(d/2) bounds the free coefficient norm factor. The inherited
    # row-sum bound is 4*d*h**(d-2), and analytic amplitude is below two.
    matrix_row_bound = 4.0 * dimension * h**(dimension-2)
    assembly_roundoff = gamma(4096.0) * coordinate_count_bound * (rhs_entry_bound + 2*matrix_row_bound)
    coefficient_error = h**(dimension/2) / eigenvalue_floor * (residual_target + assembly_roundoff)
    samples = (cells + 1) * 8
    return float(coefficient_error + gamma(samples**dimension) * reference + 128.0 * eps)


def nodal_grid(coordinates, values, cells: int, dimension: int = 2) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    declared_levels(dimension)
    count = (cells + 1) ** dimension
    if coordinates.shape != (count, dimension) or values.shape != (count,):
        raise ComparisonError("Q1 coordinate/coefficient inventory differs from the declared grid")
    if not np.isfinite(coordinates).all() or not np.isfinite(values).all():
        raise ComparisonError("non-finite coordinates or coefficients")
    # All declared levels are powers of two: these normalized grid coordinates
    # are exactly representable binary64, so no coordinate-matching tolerance.
    indices = coordinates * cells
    if not np.array_equal(indices, np.rint(indices)) or np.any(indices < 0) or np.any(indices > cells):
        raise ComparisonError("coordinates are not on the declared unit-box dyadic grid")
    indices = indices.astype(int)
    if len({tuple(index) for index in indices}) != count:
        raise ComparisonError("duplicate grid coordinates")
    grid = np.empty((cells + 1,) * dimension)
    grid[tuple(indices.T)] = values
    boundary = np.any((indices == 0) | (indices == cells), axis=1)
    if np.any(values[boundary] != 0.0):
        raise ComparisonError("homogeneous essential boundary coefficients differ from zero")
    return grid


def continuous_l2(grid: np.ndarray) -> float:
    cells = grid.shape[0] - 1
    points, weights = np.polynomial.legendre.leggauss(8)
    points, weights = (points + 1.0) / 2.0, weights / 2.0
    dimension = grid.ndim
    quadrature = np.array(np.meshgrid(*([points]*dimension), indexing="ij"))
    tensor_weights = np.prod(np.array(np.meshgrid(*([weights]*dimension), indexing="ij")), axis=0)
    total = 0.0
    for cell in np.ndindex(*([cells]*dimension)):
        value = np.zeros_like(tensor_weights)
        for corner in itertools.product((0, 1), repeat=dimension):
            basis = np.prod([quadrature[axis] if side else 1-quadrature[axis]
                             for axis, side in enumerate(corner)], axis=0)
            value += basis*grid[tuple(index+side for index, side in zip(cell, corner, strict=True))]
        origin = np.array(cell).reshape((dimension,)+((1,)*dimension))
        exact = np.prod(np.sin(np.pi*(origin+quadrature)/cells), axis=0)
        total += float(np.sum(tensor_weights*(value-exact)**2))/cells**dimension
    return float(np.sqrt(total))


def check_accuracy(coordinates, values, cells: int, dimension: int = 2) -> dict:
    if cells not in declared_levels(dimension):
        raise ComparisonError(f"unsupported reference level {cells}")
    reference = REFERENCE if dimension == 2 else REFERENCES[dimension]
    with reference.open() as source:
        table = {int(row["cells_per_axis"]): float(row["fem_l2"]) for row in csv.DictReader(source)}
    grid = nodal_grid(coordinates, values, cells, dimension)
    error = continuous_l2(grid)
    expected = table[cells]
    if not np.isfinite(expected) or expected <= 0.0:
        raise ComparisonError("independent reference error must be finite and positive")
    allowance = error_budget(cells, expected, dimension)
    if not np.isfinite(error) or abs(error-expected) > allowance:
        raise ComparisonError(f"N={cells} continuous L2 {error} differs from analytic {expected} by more than {allowance}")
    return {"continuous_l2": error, "analytic_discrete_l2": expected,
            "comparison_allowance": allowance, "reference": str(reference.relative_to(ROOT)),
            "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}


def check_identity_row(row, columns, data):
    """Require the actual symmetric-elimination boundary row, not a vacuous row."""
    if len(columns) != len(data):
        raise ComparisonError("boundary column/value inventories differ")
    diagonal = [value for column, value in zip(columns, data, strict=True) if column == row]
    if len(diagonal) != 1 or diagonal[0] != 1.0:
        raise ComparisonError("boundary row requires exactly one stored unit diagonal")
    if any(value != 0.0 for column, value in zip(columns, data, strict=True) if column != row):
        raise ComparisonError("boundary row has a nonzero off-diagonal entry")


def q1_vertex_dof_coordinates(rows, count, dimension: int = 2):
    """Preserve each authoritative vertex point paired with its exact scalar dof."""
    declared_levels(dimension)
    coordinates = np.empty((count, dimension))
    vertices, dofs = set(), set()
    for vertex, indices, point in rows:
        if not 0 <= vertex < count or vertex in vertices or len(indices) != 1:
            raise ComparisonError("Q1 vertex correspondence requires one distinct scalar dof per vertex")
        dof = indices[0]
        if not isinstance(dof, (int, np.integer)) or not 0 <= dof < count or dof in dofs:
            raise ComparisonError("Q1 dof correspondence is duplicate or out of range")
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (dimension,) or not np.isfinite(point).all():
            raise ComparisonError("Q1 vertex has an invalid authoritative Geometry point")
        vertices.add(vertex); dofs.add(dof)
        coordinates[dof] = point
    if vertices != set(range(count)) or dofs != set(range(count)):
        raise ComparisonError("Q1 vertex/dof correspondence is incomplete")
    return coordinates


def check_tensor_gauss(points, weights, dimension: int):
    """Check the actual external tensor rule against its closed-form definition."""
    declared_levels(dimension)
    points, weights = np.asarray(points), np.asarray(weights)
    expected_1d = (1.0 + np.array([-1.0, 1.0])/np.sqrt(3.0))/2.0
    expected = np.array(sorted(itertools.product(expected_1d, repeat=dimension)))
    if points.shape != (2**dimension, dimension) or weights.shape != (2**dimension,):
        raise ComparisonError("external quadrature is not the declared two-point tensor Gauss rule")
    if not np.isfinite(points).all() or not np.isfinite(weights).all():
        raise ComparisonError("external quadrature contains non-finite values")
    actual = np.array(sorted(map(tuple, points)))
    np.testing.assert_array_max_ulp(actual, expected, maxulp=8)
    np.testing.assert_array_max_ulp(weights, np.full(2**dimension, 1.0/2**dimension), maxulp=8)
