"""Shared Q1 topology, elimination and tensor-quadrature checks."""
import itertools
import numpy as np
from shared_measurement import ComparisonError


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
    """Preserve each authoritative vertex point paired with its exact scalar or blocked dof."""
    if dimension not in (2, 3):
        raise ComparisonError(f"unsupported Q1 comparison dimension {dimension}")
    coordinates = np.empty((count, dimension))
    vertices, dofs = set(), set()
    for vertex, indices, point in rows:
        if not 0 <= vertex < count or vertex in vertices or len(indices) != 1:
            raise ComparisonError("Q1 vertex correspondence requires one distinct scalar or blocked dof per vertex")
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
    if dimension not in (2, 3):
        raise ComparisonError(f"unsupported Q1 comparison dimension {dimension}")
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
