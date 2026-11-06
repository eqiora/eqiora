"""Independent analytic oracle and continuous vector-Q1 reconstruction."""
from __future__ import annotations

import math
import numpy as np

LEVELS = (4, 8, 16, 32)
MU, LAMBDA, Q0, K = 3.0, 2.0, 1.0, math.pi
A = Q0 / (2 * K * (LAMBDA + 2 * MU))


class ComparisonError(ValueError):
    """A candidate does not meet the predeclared comparison protocol."""


def exact(coordinates):
    x, y = np.moveaxis(np.asarray(coordinates), -1, 0)
    sx, sy = np.sin(K*x), np.sin(K*y)
    tx, ty = np.sin(2*K*x), np.sin(2*K*y)
    values = -A * np.stack((tx*sy**2, sx**2*ty), axis=-1)
    gradient = -K*A * np.stack((2*np.cos(2*K*x)*sy**2, tx*ty,
                                tx*ty, 2*sx**2*np.cos(2*K*y)), axis=-1)
    return values, gradient.reshape((*x.shape, 2, 2))


def stress(coordinates):
    _, gradient = exact(coordinates)
    return 2*MU*gradient + LAMBDA*np.trace(gradient, axis1=-2, axis2=-1)[..., None, None]*np.eye(2)


def force(coordinates):
    x, y = np.moveaxis(np.asarray(coordinates), -1, 0)
    return Q0*K*np.stack((np.sin(2*K*x)*(1-4*np.sin(K*y)**2),
                         np.sin(2*K*y)*(1-4*np.sin(K*x)**2)), axis=-1)


def grid_field(coordinates, values, cells):
    if cells not in LEVELS:
        raise ComparisonError("mesh level is outside the declared sequence")
    coordinates, values = np.asarray(coordinates), np.asarray(values)
    count = (cells+1)**2
    if coordinates.shape != (count, 2) or values.shape != (count, 2):
        raise ComparisonError("expected one two-component vector per grid vertex")
    if not np.isfinite(coordinates).all() or not np.isfinite(values).all():
        raise ComparisonError("nonfinite coordinates or displacement")
    scaled = coordinates*cells
    if not np.equal(scaled, np.round(scaled)).all() or np.any(scaled < 0) or np.any(scaled > cells):
        raise ComparisonError("coordinates are not authoritative dyadic grid vertices")
    indices = scaled.astype(int)
    keys = indices[:, 0]*(cells+1)+indices[:, 1]
    if len(np.unique(keys)) != count:
        raise ComparisonError("vertex map is not bijective")
    grid = np.empty((cells+1, cells+1, 2))
    grid[indices[:, 0], indices[:, 1]] = values
    if max(np.max(np.abs(edge)) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])) > 1e-14:
        raise ComparisonError("displacement violates homogeneous essential boundary")
    return grid


def continuous_errors(grid, cells, points):
    """Integrate analytic error of the vector Q1 interpolant, not nodal RMS."""
    nodes, weights = np.polynomial.legendre.leggauss(points)
    nodes, weights = (nodes+1)/2, weights/2
    local = np.array([(s, t) for s in nodes for t in nodes])
    weight = np.outer(weights, weights).reshape(-1)/cells**2
    s, t = local.T
    basis = np.array(((1-s)*(1-t), s*(1-t), (1-s)*t, s*t)).T
    dx = cells*np.array((-(1-t), 1-t, -t, t)).T
    dy = cells*np.array((-(1-s), -s, 1-s, s)).T
    corners = np.stack((grid[:-1, :-1], grid[1:, :-1], grid[:-1, 1:], grid[1:, 1:]), axis=2).reshape(-1, 4, 2)
    origins = np.array([(i, j) for i in range(cells) for j in range(cells)])
    coordinates = (origins[:, None, :]+local[None, :, :])/cells
    actual = np.einsum('qv,cva->cqa', basis, corners)
    gradient = np.stack((np.einsum('qv,cva->cqa', dx, corners),
                         np.einsum('qv,cva->cqa', dy, corners)), axis=-1)
    expected, expected_gradient = exact(coordinates)
    l2 = np.sqrt(np.sum(np.sum((actual-expected)**2, axis=-1)*weight))
    h1 = np.sqrt(np.sum(np.sum((gradient-expected_gradient)**2, axis=(-1, -2))*weight))
    return np.array((l2, h1))


def check_accuracy(coordinates, values, cells):
    grid = grid_field(coordinates, values, cells)
    errors = continuous_errors(grid, cells, 10)
    audit = continuous_errors(grid, cells, 12)
    if not np.isfinite(errors).all() or np.any(np.abs(errors-audit) > 1e-14+1e-10*np.abs(audit)):
        raise ComparisonError("continuous error quadrature did not converge")
    norms = np.array((np.sqrt(3*A*A/8), np.sqrt(2*K*K*A*A)))
    relative = errors/norms
    if np.any(relative >= 1):
        raise ComparisonError("relative continuous displacement/gradient error is not below one")
    return {"l2_error": float(errors[0]), "h1_seminorm_error": float(errors[1]),
            "relative_l2_error": float(relative[0]), "relative_h1_seminorm_error": float(relative[1]),
            "quadrature_audit_absolute_difference": np.abs(errors-audit).tolist()}


def check_convergence(rows):
    if set(rows) != set(LEVELS):
        raise ComparisonError("convergence requires the entire declared level sequence")
    orders = {}
    for key, floor in (("l2_error", 1.9), ("h1_seminorm_error", 0.9)):
        errors = np.array([rows[n][key] for n in LEVELS])
        if not np.isfinite(errors).all() or np.any(errors <= 0) or np.any(np.diff(errors) >= 0):
            raise ComparisonError("continuous errors must be finite, positive and strictly decreasing")
        observed = np.log2(errors[:-1]/errors[1:])
        if np.any(observed[-2:] < floor):
            raise ComparisonError(f"{key} fails predeclared convergence orders")
        orders[key] = observed.tolist()
    return orders


def check_equilibrium(reaction, body_force):
    reaction, body_force = np.asarray(reaction), np.asarray(body_force)
    if reaction.shape != (2,) or body_force.shape != (2,) or not np.isfinite([reaction, body_force]).all():
        raise ComparisonError("invalid equilibrium vectors")
    if np.any(np.abs(body_force-np.array((1., 0.))) > 5e-11):
        raise ComparisonError("affine probe must have independently known nonzero unit force")
    if np.any(np.abs(reaction+body_force) > 5e-11*np.maximum(1, np.abs(reaction)+np.abs(body_force))):
        raise ComparisonError("componentwise reaction/body-force balance failed")
