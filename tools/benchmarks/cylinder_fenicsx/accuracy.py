"""Physical MINI polynomial observations and the predeclared triangle rules."""
import math
import numpy as np
from topology import mesh_arrays
from shared_measurement import ComparisonError

U, H, MU = .3, .41, .001
P, FORCE_SCALE = MU*U/H, MU*U


def duffy_rule(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes, weights = (nodes+1)/2, weights/2
    points, result_weights = [], []
    for i, s in enumerate(nodes):
        for j, t in enumerate(nodes):
            points.append((s, (1-s)*t))
            result_weights.append(weights[i]*weights[j]*(1-s))
    return np.array(points), np.array(result_weights)


def check_rule(points, weights, degree):
    points, weights = np.asarray(points), np.asarray(weights)
    if (points.ndim != 2 or points.shape[1] != 2 or weights.shape != (len(points),)
            or not np.isfinite(points).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0) or np.any(points < 0) or np.any(points.sum(axis=1) > 1)):
        raise ComparisonError('invalid positive reference-triangle quadrature')
    # At most 16 positive terms and degree-six powers in the used rules.
    # 256 eps is a declared arithmetic-control allowance, not a PDE error budget.
    allowance = 256*np.finfo(float).eps
    for a in range(degree+1):
        for b in range(degree+1-a):
            observed = np.sum(weights*points[:, 0]**a*points[:, 1]**b)
            exact = math.factorial(a)*math.factorial(b)/math.factorial(a+b+2)
            if abs(observed-exact) > allowance:
                raise ComparisonError('triangle quadrature failed an analytic monomial moment')


def physical_norms(data, fields):
    coordinates, cells, _ = mesh_arrays(data)
    points, weights = duffy_rule(4)
    check_rule(points, weights, 6)
    barycentric = np.column_stack((1-points.sum(axis=1), points))
    bubble = 27*np.prod(barycentric, axis=1)
    velocity = np.asarray(fields['velocity_vertex'])
    beta = np.asarray(fields['velocity_cell_bubble'])
    pressure = np.asarray(fields['pressure_vertex'])
    if velocity.shape != coordinates.shape or beta.shape != (len(cells), 2) or pressure.shape != (len(coordinates),):
        raise ComparisonError('complete MINI velocity/pressure association inventory differs from mesh')
    if not all(np.isfinite(a).all() for a in (velocity, beta, pressure)):
        raise ComparisonError('nonfinite MINI field')
    geometry = coordinates[cells]
    jacobian = np.stack((geometry[:, 1]-geometry[:, 0], geometry[:, 2]-geometry[:, 0]), axis=-1)
    determinants = np.linalg.det(jacobian)
    if np.any(determinants <= 0):
        raise ComparisonError('reference mesh requires positively oriented affine triangles')
    u = np.einsum('qv,cva->cqa', barycentric, velocity[cells])+bubble[None, :, None]*beta[:, None, :]
    p = np.einsum('qv,cv->cq', barycentric, pressure[cells])
    measure = determinants[:, None]*weights[None, :]
    area = determinants.sum()/2
    return {'velocity': float(np.sqrt(np.sum(np.sum(u*u, axis=-1)*measure)/area)),
            'pressure': float(np.sqrt(np.sum(p*p*measure)/area))}


def field_distances(data, left, right):
    return physical_norms(data, {key: np.asarray(left[key])-np.asarray(right[key]) for key in
                                ('velocity_vertex', 'velocity_cell_bubble', 'pressure_vertex')})


def discrete_inlet_flux(data):
    coordinates, _, _ = mesh_arrays(data)
    segments = coordinates[np.asarray(data['boundaries']['inlet'], dtype=np.int64)]
    if np.any(segments[:, :, 0] != 0):
        raise ComparisonError('inlet edges are not on x=0')
    lengths = np.abs(segments[:, 1, 1]-segments[:, 0, 1])
    return -(2*U*H/3-(2*U/(3*H*H))*np.sum(lengths**3))


def boundary_flux(data, fields, name):
    coordinates, _, _ = mesh_arrays(data)
    segments = np.asarray(data['boundaries'][name], dtype=np.int64)
    # Only inlet/outlet are used: outward normals are fixed by channel geometry.
    if name not in ('inlet', 'outlet'):
        raise ComparisonError('this analytic flux observation supports inlet/outlet only')
    lengths = np.abs(coordinates[segments[:, 1], 1]-coordinates[segments[:, 0], 1])
    values = np.asarray(fields['velocity_vertex'])[segments, 0].mean(axis=1)
    return float(np.sum(lengths*values)*(-1 if name == 'inlet' else 1))


def check_trace_and_flux(data, fields, scaled_residual_cap):
    coordinates, _, _ = mesh_arrays(data)
    expected = np.zeros_like(coordinates)
    inlet_vertices = np.unique(np.asarray(data['boundaries']['inlet']).reshape(-1))
    y = coordinates[inlet_vertices, 1]
    expected[inlet_vertices, 0] = 4*U*y*(H-y)/(H*H)
    zeros = np.unique(np.concatenate([np.asarray(data['boundaries'][name]).reshape(-1)
                                      for name in ('walls', 'cylinder')]))
    expected[zeros] = 0
    essential = np.union1d(inlet_vertices, zeros)
    trace_error = float(np.max(np.abs(np.asarray(fields['velocity_vertex'])[essential]-expected[essential])))
    trace_roundoff = 256*np.finfo(float).eps*U
    if trace_error > trace_roundoff:
        raise ComparisonError('essential velocity trace differs from prescribed nodal inlet/no-slip data')
    inlet, outlet = boundary_flux(data, fields, 'inlet'), boundary_flux(data, fields, 'outlet')
    exact = discrete_inlet_flux(data)
    flux_roundoff = 256*np.finfo(float).eps*U*H*max(1, len(data['boundaries']['inlet']))
    if abs(inlet-exact) > flux_roundoff:
        raise ComparisonError('inlet flux differs from independently integrated piecewise-linear trace')
    # Every pressure row is free. Summing them represents q=1 and gives net
    # boundary flux. A physical pressure-row residual is U H times its scaled
    # counterpart, hence Nv U H r_cap plus the input-only flux summation margin.
    continuity_bound = len(coordinates)*U*H*scaled_residual_cap+flux_roundoff
    if abs(inlet+outlet) > continuity_bound:
        raise ComparisonError('global flux fails the common weak-continuity residual bound')
    return {'trace_max_error': trace_error, 'trace_arithmetic_allowance': trace_roundoff,
            'inlet_flux': inlet, 'outlet_flux': outlet, 'analytic_discrete_inlet_flux': exact,
            'flux_arithmetic_allowance': flux_roundoff, 'net_flux': inlet+outlet,
            'weak_continuity_flux_bound': continuity_bound}
