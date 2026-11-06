#!/usr/bin/env python3
"""Exact-rational full-vector Q2 polynomial patch/reaction comparator."""
import argparse
from decimal import Decimal, InvalidOperation
from fractions import Fraction as F
import itertools
import hashlib
import json
import math
from pathlib import Path
import sys

RESIDUAL_BUDGET = F(1, 2**40)
UNIT_ROUNDOFF = F(1, 2**53)
ROUNDING_MULTIPLIER = 8192
BASIS = ((F(1), F(-3), F(2)), (F(0), F(4), F(-4)), (F(0), F(-1), F(2)))


class PolynomialError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def require(condition, code, message):
    if not condition:
        raise PolynomialError(code, message)


def keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected), 'schema', f'{label}: unexpected or missing keys')


def rational_json(value):
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}


def binary64(text):
    require(isinstance(text, str), 'value', 'binary64 value must be a canonical hex string')
    try:
        value = float.fromhex(text)
    except (ValueError, OverflowError) as error:
        raise PolynomialError('value', 'invalid binary64 hex') from error
    require(math.isfinite(value), 'value', 'binary64 value must be finite')
    require(value.hex() == text, 'value', 'binary64 value must use canonical float.hex form')
    return F(*value.as_integer_ratio())


def coordinate(text):
    require(isinstance(text, str), 'coordinate', 'coordinate must be an exact decimal string')
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise PolynomialError('coordinate', 'invalid decimal coordinate') from error
    require(value.is_finite(), 'coordinate', 'coordinate must be finite')
    return F(value)


def product(values):
    result = F(1)
    for value in values:
        result *= value
    return result


def integral_product(a, b):
    return sum((x*y/F(i+j+1) for i, x in enumerate(a) for j, y in enumerate(b)), F(0))


def one_axis(length):
    """Full N=2 axis matrices and exact integrals of 1 and p=1+(x/L)^2."""
    n = 2
    h = F(length, n)
    mass = [[F(0) for _ in range(5)] for _ in range(5)]
    stiff = [[F(0) for _ in range(5)] for _ in range(5)]
    constant = [F(0)]*5
    polynomial = [F(0)]*5
    derivatives = [tuple(i*v for i, v in enumerate(b))[1:] for b in BASIS]
    for cell in range(n):
        p = (1+F(cell*cell, n*n), F(2*cell, n*n), F(1, n*n))
        for i, bi in enumerate(BASIS):
            row = 2*cell+i
            constant[row] += h*integral_product(bi, (F(1),))
            polynomial[row] += h*integral_product(bi, p)
            for j, bj in enumerate(BASIS):
                col = 2*cell+j
                mass[row][col] += h*integral_product(bi, bj)
                stiff[row][col] += integral_product(derivatives[i], derivatives[j])/h
    return mass, stiff, constant, polynomial


def build_reference(lengths):
    require(tuple(lengths) in ((2, 3), (2, 3, 4)), 'mesh', 'unsupported box')
    d = len(lengths)
    indices = tuple(itertools.product(range(5), repeat=d))
    axes = [one_axis(length) for length in lengths]
    matrix = []
    volume, natural, exact = [], [], []
    for row in indices:
        matrix.append(tuple(sum((axes[i][1][row[i]][col[i]] * product(
            axes[j][0][row[j]][col[j]] for j in range(d) if j != i)
            for i in range(d)), F(0)) for col in indices))
        volume.append(-sum((F(2, lengths[i]**2)*axes[i][2][row[i]] * product(
            axes[j][3][row[j]] for j in range(d) if j != i)
            for i in range(d)), F(0)))
        natural.append(sum((F(2, lengths[i])*product(axes[j][3][row[j]]
            for j in range(d) if j != i) for i in range(1, d) if row[i] == 4), F(0)))
        exact.append(product(1+F(k*k, 16) for k in row))
    h = [F(length, 2) for length in lengths]
    kinf = sum((F(32)/(3*h[i])*product(2*h[j]/3 for j in range(d) if j != i)
                for i in range(d)), F(0))
    rhs = tuple(a+b for a, b in zip(volume, natural))
    binf = max(abs(value) for value in rhs)
    rho = RESIDUAL_BUDGET*max(F(1), kinf*2**d+binf)
    bu = (F(3200, 3) if d == 2 else F(16000))*RESIDUAL_BUDGET
    return {'lengths': tuple(lengths), 'indices': indices, 'K': tuple(matrix),
            'F': tuple(volume), 'G': tuple(natural), 'rhs': rhs, 'exact': tuple(exact),
            'upper': tuple(i for i, row in enumerate(indices) if row[0] == 4),
            'free': tuple(i for i, row in enumerate(indices) if row[0] != 4),
            'Kinf': kinf, 'Binf': binf, 'rho': rho, 'Bu': bu,
            'target_sum': F(4) if d == 2 else F(64, 3)}


def residual(reference, vector):
    return tuple(sum((weight*value for weight, value in zip(row, vector)), F(0))-rhs
                 for row, rhs in zip(reference['K'], reference['rhs']))


def decode(data, reference):
    lengths = reference['lengths']
    d = len(lengths)
    grids = [{F(length*k, 4): k for k in range(5)} for length in lengths]
    def point(coords):
        require(isinstance(coords, list) and len(coords) == d, 'coordinate', 'coordinate dimension mismatch')
        index = []
        for axis, text in enumerate(coords):
            value = coordinate(text)
            require(value in grids[axis], 'coordinate', 'coordinate is not an exact mesh node')
            index.append(grids[axis][value])
        return tuple(index)
    require(isinstance(data['records'], list), 'schema', 'coefficient records must be a list')
    values = {}
    for record in data['records']:
        keys(record, ('coordinates_m', 'value_binary64_hex'), 'coefficient')
        index = point(record['coordinates_m'])
        require(index not in values, 'duplicate', 'duplicate coefficient coordinate')
        values[index] = binary64(record['value_binary64_hex'])
    require(len(values) == 5**d, 'missing', 'complete physical coefficient grid required')
    vector = tuple(values[index] for index in reference['indices'])
    for i in reference['upper']:
        require(vector[i] == reference['exact'][i], 'essential', 'upper-x coefficient must equal exact essential value')
    require(isinstance(data['upper_reactions'], list), 'schema', 'reaction records must be a list')
    reactions = {}
    for record in data['upper_reactions']:
        keys(record, ('coordinates_m', 'value_binary64_hex'), 'reaction')
        index = point(record['coordinates_m'])
        require(index[0] == 4, 'face', 'reaction is not on upper-x face')
        require(index not in reactions, 'duplicate', 'duplicate reaction coordinate')
        reactions[index] = binary64(record['value_binary64_hex'])
    require(len(reactions) == 5**(d-1), 'missing', 'complete upper-x reaction grid required')
    return vector, tuple(reactions[reference['indices'][i]] for i in reference['upper'])


def compare(data):
    keys(data, ('schema', 'lengths_m', 'N', 'gauss_points', 'records', 'upper_reactions'), 'input')
    require(type(data['schema']) is int and data['schema'] == 1, 'schema', 'unsupported schema')
    require(type(data['N']) is int and data['N'] == 2, 'mesh', 'only N=2 supported')
    require(type(data['gauss_points']) is int and data['gauss_points'] == 4, 'quadrature', 'only declared four-point assembly supported')
    lengths = data['lengths_m']
    require(isinstance(lengths, list) and all(type(x) is int for x in lengths)
            and tuple(lengths) in ((2, 3), (2, 3, 4)), 'mesh', 'unsupported box')
    reference = build_reference(lengths)
    vector, native = decode(data, reference)
    r = residual(reference, vector)
    free_squared = sum((r[i]**2 for i in reference['free']), F(0))
    coefficient_errors = tuple(abs(a-b) for a, b in zip(vector, reference['exact']))
    actual = tuple(r[i] for i in reference['upper'])
    differences = tuple(abs(a-b) for a,b in zip(native, actual))
    m = len(actual)
    actual_sum, native_sum = sum(actual, F(0)), sum(native, F(0))
    sum_native_actual = abs(native_sum-actual_sum)
    sum_target_error = abs(native_sum-reference['target_sum'])
    sum_actual_budget = m*reference['rho']
    sum_target_budget = m*reference['Kinf']*reference['Bu']+sum_actual_budget
    checks = {
        'free_residual_squared': free_squared <= RESIDUAL_BUDGET**2,
        'every_coefficient': all(error <= reference['Bu'] for error in coefficient_errors),
        'every_native_reaction': all(error <= reference['rho'] for error in differences),
        'native_sum_vs_full_vector_reaction_sum': sum_native_actual <= sum_actual_budget,
        'native_sum_vs_analytic_target': sum_target_error <= sum_target_budget,
    }
    return {'schema': 1, 'lengths_m': lengths, 'N': 2, 'gauss_points': 4,
            'reaction_units': f'm^({len(lengths)-2})',
            'coefficient_count': len(vector), 'free_count': len(reference['free']), 'upper_reaction_count': m,
            'reaction_convention': 'unscaled physical K*v-F-G on upper-x face',
            'rounding_ceiling_status': 'declared 8192-unit-roundoff ceiling; not a universal floating-point theorem',
            'free_residual_squared': rational_json(free_squared),
            'free_residual_squared_budget': rational_json(RESIDUAL_BUDGET**2),
            'maximum_coefficient_error': rational_json(max(coefficient_errors)),
            'coefficient_error_budget': rational_json(reference['Bu']),
            'Kinf_bound': rational_json(reference['Kinf']), 'Binf': rational_json(reference['Binf']),
            'reaction_component_budget_rho': rational_json(reference['rho']),
            'maximum_native_reaction_error': rational_json(max(differences)),
            'native_reaction_sum': rational_json(native_sum),
            'full_vector_reaction_sum': rational_json(actual_sum),
            'analytic_reaction_sum': rational_json(reference['target_sum']),
            'native_sum_vs_full_vector_error': rational_json(sum_native_actual),
            'native_sum_vs_full_vector_budget': rational_json(sum_actual_budget),
            'native_sum_vs_target_error': rational_json(sum_target_error),
            'native_sum_vs_target_budget': rational_json(sum_target_budget),
            'checks': checks, 'accepted': all(checks.values())}


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
    args = parser.parse_args()
    raw = None
    try:
        raw = args.input.read_bytes()
        result = compare(json.loads(raw, object_pairs_hook=unique_object))
    except (PolynomialError, ValueError, OSError) as error:
        result = {'accepted': False, 'error': getattr(error, 'code', 'input'), 'message': str(error)}
        if raw is not None:
            result['input_sha256'] = hashlib.sha256(raw).hexdigest()
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        return 2
    result['input_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    return 0 if result['accepted'] else 1


if __name__ == '__main__':
    sys.exit(main())
