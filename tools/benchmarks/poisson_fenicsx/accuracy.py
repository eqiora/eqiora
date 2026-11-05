"""One bounded 2D Q1 accuracy check; no solver output is an expectation."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LEVELS = (4, 8, 16, 32)
REFERENCE = ROOT / "verify/numerics/cartesian-poisson-fem-fvm/expected/convergence.csv"


class ComparisonError(ValueError):
    """The declared mathematical or measurement contract was not met."""


def error_budget(cells: int, reference: float) -> float:
    """Exact d=2 FEM port of the existing registered Cartesian error_budget.

    Owner: crates/eqiora-numerics/tests/support/cartesian_poisson.rs, with
    derivation in verify/numerics/cartesian-poisson-fem-fvm/references/README.md.
    Its scope is uniform Q1, 2-point load quadrature, zero essential boundary,
    unit coefficient, same residual policy, and 8-point error integration.
    Both tools have the same effective free operator; zero boundary identity
    rows in PETSc do not change that subspace. This is the inherited comparison
    allowance, not a new rigorous floating-point certification of either tool.
    The residual-to-error implication is independent of the Krylov algorithm.
    """
    if cells not in LEVELS:
        raise ComparisonError(f"no declared reference for N={cells}; supported {LEVELS}")
    h = 1.0 / cells
    eigenvalue_floor = 8.0 * np.sin(np.pi * h / 2.0) ** 2 / 3.0
    rhs_entry_bound = 2.0 * np.pi**2 * h**2
    residual_target = max(1e-14, 1e-12 * rhs_entry_bound * cells)
    eps = np.finfo(np.float64).eps

    def gamma(operations: float) -> float:
        product = operations * eps
        return product / (1.0 - product)

    # N bounds sqrt(number of free coordinates); each absolute stiffness
    # row sum <=8, and analytic nodal amplitude <2 on the declared levels.
    assembly_roundoff = gamma(4096.0) * cells * (rhs_entry_bound + 16.0)
    coefficient_error = h / eigenvalue_floor * (residual_target + assembly_roundoff)
    samples = (cells + 1) * 8
    return float(coefficient_error + gamma(samples**2) * reference + 128.0 * eps)


def nodal_grid(coordinates, values, cells: int) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    count = (cells + 1) ** 2
    if coordinates.shape != (count, 2) or values.shape != (count,):
        raise ComparisonError("Q1 coordinate/coefficient inventory differs from the declared grid")
    if not np.isfinite(coordinates).all() or not np.isfinite(values).all():
        raise ComparisonError("non-finite coordinates or coefficients")
    # All declared levels are powers of two: these normalized grid coordinates
    # are exactly representable binary64, so no coordinate-matching tolerance.
    indices = coordinates * cells
    if not np.array_equal(indices, np.rint(indices)) or np.any(indices < 0) or np.any(indices > cells):
        raise ComparisonError("coordinates are not on the declared unit-square dyadic grid")
    indices = indices.astype(int)
    if len({tuple(index) for index in indices}) != count:
        raise ComparisonError("duplicate grid coordinates")
    grid = np.empty((cells + 1, cells + 1))
    grid[indices[:, 0], indices[:, 1]] = values
    if np.any(grid[[0, -1], :] != 0.0) or np.any(grid[:, [0, -1]] != 0.0):
        raise ComparisonError("homogeneous essential boundary coefficients differ from zero")
    return grid


def continuous_l2(grid: np.ndarray) -> float:
    cells = grid.shape[0] - 1
    points, weights = np.polynomial.legendre.leggauss(8)
    points, weights = (points + 1.0) / 2.0, weights / 2.0
    total = 0.0
    for i in range(cells):
        for j in range(cells):
            t, s = points[:, None], points[None, :]
            value = ((1-t)*(1-s)*grid[i, j] + t*(1-s)*grid[i+1, j]
                     + (1-t)*s*grid[i, j+1] + t*s*grid[i+1, j+1])
            exact = np.sin(np.pi*(i+t)/cells) * np.sin(np.pi*(j+s)/cells)
            total += float(np.sum(weights[:, None]*weights[None, :]*(value-exact)**2))/cells**2
    return float(np.sqrt(total))


def check_accuracy(coordinates, values, cells: int) -> dict:
    if cells not in LEVELS:
        raise ComparisonError(f"unsupported reference level {cells}")
    with REFERENCE.open() as source:
        table = {int(row["cells_per_axis"]): float(row["fem_l2"]) for row in csv.DictReader(source)}
    grid = nodal_grid(coordinates, values, cells)
    error = continuous_l2(grid)
    expected = table[cells]
    allowance = error_budget(cells, expected)
    if not np.isfinite(error) or abs(error-expected) > allowance:
        raise ComparisonError(f"N={cells} continuous L2 {error} differs from analytic {expected} by more than {allowance}")
    return {"continuous_l2": error, "analytic_discrete_l2": expected,
            "comparison_allowance": allowance, "reference": str(REFERENCE.relative_to(ROOT)),
            "reference_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest()}


def check_identity_row(row, columns, data):
    """Require the actual symmetric-elimination boundary row, not a vacuous row."""
    if len(columns) != len(data):
        raise ComparisonError("boundary column/value inventories differ")
    diagonal = [value for column, value in zip(columns, data, strict=True) if column == row]
    if len(diagonal) != 1 or diagonal[0] != 1.0:
        raise ComparisonError("boundary row requires exactly one stored unit diagonal")
    if any(value != 0.0 for column, value in zip(columns, data, strict=True) if column != row):
        raise ComparisonError("boundary row has a nonzero off-diagonal entry")
