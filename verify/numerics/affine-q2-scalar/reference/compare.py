#!/usr/bin/env python3
"""Enclose full-vector Q2 residuals and continuous norms, without Eqiora."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import generate as g

HERE = Path(__file__).resolve().parent
FROZEN = HERE / 'intervals-gauss4-audit.json'


class ComparisonError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def require(condition, code, message):
    if not condition:
        raise ComparisonError(code, message)


def keys(obj, expected, label):
    require(isinstance(obj, dict) and set(obj) == set(expected), 'schema', f'{label}: unexpected or missing keys')


def exact_decimal(value):
    require(isinstance(value, str), 'coordinate', 'coordinate must be an exact decimal string')
    try:
        result = g.D(value)
    except Exception as error:
        raise ComparisonError('coordinate', 'invalid coordinate') from error
    require(result.is_finite(), 'coordinate', 'coordinate must be finite')
    return result


def binary64(value):
    require(isinstance(value, str), 'value', 'value_binary64_hex must be a canonical hex string')
    try:
        result = float.fromhex(value)
    except (ValueError, OverflowError) as error:
        raise ComparisonError('value', 'invalid binary64 hex') from error
    require(math.isfinite(result), 'value', 'coefficient must be finite')
    require(result.hex() == value, 'value', 'binary64 hex must use float.hex canonical form')
    # Decimal.from_float captures the exact binary rational, not its decimal repr.
    return g.I(g.D.from_float(result))


def decode(data):
    keys(data, ('schema', 'lengths_m', 'N', 'gauss_points', 'records'), 'input')
    require(type(data['schema']) is int and data['schema'] == 1, 'schema', 'unsupported schema')
    lengths, n = data['lengths_m'], data['N']
    require(isinstance(lengths, list) and all(type(x) is int for x in lengths)
            and tuple(lengths) in ((2, 3), (2, 3, 4)), 'mesh', 'unsupported box')
    require(type(n) is int and n in (2, 4, 8), 'mesh', 'unsupported mesh')
    require(type(data['gauss_points']) is int and data['gauss_points'] == 4,
            'quadrature', 'only frozen four-point audit rule supported')
    require(isinstance(data['records'], list), 'schema', 'records must be a list')
    d = len(lengths)
    all_values = {}
    grid = [{(g.I(l)*k/(2*n)).lo: k for k in range(2*n+1)} for l in lengths]
    for record in data['records']:
        keys(record, ('coordinates_m', 'value_binary64_hex'), 'record')
        coords = record['coordinates_m']
        require(isinstance(coords, list) and len(coords) == d, 'coordinate', 'coordinate dimension mismatch')
        point = []
        for axis, coord in enumerate(coords):
            value = exact_decimal(coord)
            require(value in grid[axis], 'coordinate', 'coordinate is not an exact mesh node')
            point.append(grid[axis][value])
        point = tuple(point)
        require(point not in all_values, 'duplicate', 'duplicate coordinate')
        value = binary64(record['value_binary64_hex'])
        if any(k in (0, 2*n) for k in point):
            require(value.lo == value.hi == 0, 'boundary', 'nonzero boundary coefficient; no boundary projection allowed')
        all_values[point] = value
    require(len(all_values) == (2*n+1)**d, 'missing', 'complete physical grid is required')
    indices = list(itertools.product(range(1, 2*n), repeat=d))
    return lengths, n, indices, [all_values[index] for index in indices]


def matrices(length, n):
    size = 2*n-1
    mass, stiff = ([{} for _ in range(size)] for _ in range(2))
    m = ((4, 2, -1), (2, 16, 2), (-1, 2, 4))
    k = ((7, -8, 1), (-8, 16, -8), (1, -8, 7))
    h = g.I(length)/n
    for cell in range(n):
        for i in range(3):
            row = 2*cell+i-1
            if not 0 <= row < size:
                continue
            for j in range(3):
                col = 2*cell+j-1
                if 0 <= col < size:
                    mass[row][col] = mass[row].get(col, g.I(0)) + h*m[i][j]/30
                    stiff[row][col] = stiff[row].get(col, g.I(0)) + k[i][j]/(3*h)
    return mass, stiff


def axis_apply(vector, rows, axis, d):
    size = len(rows)
    stride = size**(d-axis-1)
    result = [g.I(0)]*len(vector)
    for flat in range(len(vector)):
        row = (flat//stride) % size
        base = flat-row*stride
        result[flat] = sum((weight*vector[base+col*stride] for col, weight in rows[row].items()), g.I(0))
    return result


def tensor_apply(vector, matrices_):
    result = vector
    for axis, rows in enumerate(matrices_):
        result = axis_apply(result, rows, axis, len(matrices_))
    return result


def operators(lengths, n, vector):
    axes = [matrices(length, n) for length in lengths]
    mass = tensor_apply(vector, [axis[0] for axis in axes])
    stiff = [g.I(0)]*len(vector)
    for derivative in range(len(lengths)):
        term = tensor_apply(vector, [axis[int(j == derivative)] for j, axis in enumerate(axes)])
        stiff = [a+b for a, b in zip(stiff, term)]
    return mass, stiff


def sine_position(position, n, p):
    # sin(pi*position/N)=cos(pi*(position/N-1/2)); argument <=pi/2.
    return g.trig(p*(position/n-g.I(1)/2), True)


def loads(length, n, p):
    h = g.I(length)/n
    theta = p/n
    # Exact integral against each complete physical nodal basis function.
    exact_amplitude = g.one_dimensional(length, n, theta, 4)[2]
    exact = [exact_amplitude[index % 2]*sine_position(g.I(index)/2, n, p)
             for index in range(1, 2*n)]
    gauss = [g.I(0)]*(2*n-1)
    nodes, weights = g.gauss_rule(4)
    for cell in range(n):
        for t, weight in zip(nodes, weights):
            value = h*weight*sine_position(cell+t, n, p)
            basis = (2*t*t-3*t+1, 4*t*(1-t), 2*t*t-t)
            for local in range(3):
                index = 2*cell+local-1
                if 0 <= index < len(gauss):
                    gauss[index] += value*basis[local]
    return exact, gauss


def frozen_reference(lengths, n):
    raw = FROZEN.read_bytes()
    table = json.loads(raw)
    require(table['schema'] == 1 and table['decimal_precision'] == 80
            and table['load_quadrature'] == '4-point Gauss-Legendre per cell axis',
            'reference', 'unsupported frozen reference metadata')
    box = next((box for box in table['boxes'] if box['lengths_m'] == lengths), None)
    require(box is not None, 'reference', 'reference box missing')
    level = next((level for level in box['levels'] if level['N'] == n), None)
    require(level is not None and level['quadrature_budget_certified'], 'reference', 'reference level not certified')
    return [g.I(*value) for value in level['gauss4_load_errors']], hashlib.sha256(raw).hexdigest()


def width(value):
    return g.I(value.hi)-g.I(value.lo)


def compare(data):
    g.precision(80)
    lengths, n, indices, vector = decode(data)
    d = len(lengths)
    p = g.pi()
    lam = p*p*sum((g.I(1)/(l*l) for l in lengths), g.I(0))
    mass, stiff = operators(lengths, n, vector)
    axes = [loads(length, n, p) for length in lengths]
    exact_q = [g.product(values) for values in itertools.product(*(axis[0] for axis in axes))]
    gauss_q = [g.product(values) for values in itertools.product(*(axis[1] for axis in axes))]
    residual = [lam*q-a for q, a in zip(gauss_q, stiff)]
    residual_norm = sum((r.square() for r in residual), g.I(0)).sqrt()
    mu = g.product(g.I(length)/(15*n) for length in lengths)
    b1 = residual_norm/(lam*mu).sqrt()
    b0 = b1/lam.sqrt()
    c = g.product(lengths)/(2**d)
    cross = g.dot(exact_q, vector)
    squared = [c-2*cross+g.dot(vector, mass),
               lam*c-2*lam*cross+g.dot(vector, stiff)]
    require(all(value.lo > 0 for value in squared), 'norm', 'continuous error square not certified positive')
    norms = [value.sqrt() for value in squared]
    reference, sha = frozen_reference(lengths, n)
    budgets = []
    for norm, expected, solve in zip(norms, reference, (b0, b1)):
        reference_width, integration_width = width(expected), width(norm)
        total = reference_width+integration_width+g.I(0, solve.hi)
        allowed = g.I('0.001')*expected
        consistent = norm.lo <= g.UP.add(expected.hi, solve.hi) and norm.hi >= g.DOWN.subtract(expected.lo, solve.hi)
        budgets.append({'reference_uncertainty': reference_width.json(),
                        'norm_integration_uncertainty': integration_width.json(),
                        'solve_error_bound': [str(g.D(0)), str(solve.hi)],
                        'total_uncertainty': total.json(), 'allowed_uncertainty': allowed.json(),
                        'budget_certified': total.hi <= allowed.lo,
                        'reference_consistent': consistent})
    return {'schema': 1, 'lengths_m': lengths, 'N': n, 'gauss_points': 4,
            'reference_frozen_sha256': sha, 'physical_coefficient_count': (2*n+1)**d,
            'interior_coefficient_count': len(indices), 'boundary_coefficient_count': (2*n+1)**d-len(indices),
            'coefficient_encoding': 'exact binary64 from canonical float.hex',
            'norm_units': [f'm^({d}/2)', f'm^({d-2}/2)'],
            'continuous_error_intervals': [value.json() for value in norms],
            'reference_error_intervals': [value.json() for value in reference],
            'residual_l2_interval': residual_norm.json(), 'lambda': lam.json(), 'mu': mu.json(),
            'B0': b0.json(), 'B1': b1.json(), 'budgets': budgets,
            'accepted': all(b['budget_certified'] and b['reference_consistent'] for b in budgets)}


def compare_levels(inputs):
    require(isinstance(inputs, list) and len(inputs) == 3, 'levels', 'exactly N=2,4,8 levels required')
    outputs = [compare(data) for data in inputs]
    outputs.sort(key=lambda output: output['N'])
    require([output['N'] for output in outputs] == [2, 4, 8], 'levels', 'duplicate or missing mesh level')
    require(all(output['lengths_m'] == outputs[0]['lengths_m'] for output in outputs), 'levels', 'mixed boxes')
    rates = []
    for lower, upper in zip(outputs, outputs[1:]):
        pair = []
        for j in range(2):
            a = g.I(*lower['continuous_error_intervals'][j])
            b = g.I(*upper['continuous_error_intervals'][j])
            measured = (a/b).ln()/g.I(2).ln()
            ua = g.D(lower['budgets'][j]['total_uncertainty'][1])
            ub = g.D(upper['budgets'][j]['total_uncertainty'][1])
            enlarged_a = a+g.I(ua.copy_negate(), ua)
            enlarged_b = b+g.I(ub.copy_negate(), ub)
            require(enlarged_a.lo > 0 and enlarged_b.lo > 0, 'rate', 'rate uncertainty includes nonpositive error')
            propagated = (enlarged_a/enlarged_b).ln()/g.I(2).ln()
            pair.append({'measured_interval': measured.json(), 'uncertainty_enlarged_interval': propagated.json()})
        rates.append({'N_pair': [lower['N'], upper['N']], 'norm_rates': pair})
    return {'schema': 1, 'levels': outputs, 'rates': rates,
            'predeclared_two_level_rate_uncertainty_bound': ((g.I('1.001')/g.I('0.999')).ln()/g.I(2).ln()).json(),
            'accepted': all(output['accepted'] for output in outputs)}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'schema', 'duplicate JSON object key')
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--levels', action='store_true')
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text(), object_pairs_hook=unique_object)
        result = compare_levels(data) if args.levels else compare(data)
    except (ComparisonError, ValueError, OSError) as error:
        result = {'accepted': False, 'error': getattr(error, 'code', 'input'), 'message': str(error)}
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        return 2
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    return 0 if result['accepted'] else 1


if __name__ == '__main__':
    sys.exit(main())
