#!/usr/bin/env python3
"""Preparatory exact-rational Q1 Robin reference; no product support claim."""
import argparse
from decimal import Decimal, InvalidOperation
from fractions import Fraction as F
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

CASES = {
    'insulated': (2, 1, 3, 0, 280, 320),
    'robin': (2, 1, 3, 2, 280, 320),
    'changed_h': (2, 1, 3, 5, 280, 320),
    'changed_geometry_material_cooling': (3, 2, 2, 4, 310, 280),
}
ETA, SIGMA = F(1, 2**40), F(1, 2**46)
HEAT_CONVENTION = 'physical_outward_heat=-weak_reaction'
UNITS = {'coordinates': 'm', 'A': 'W/(m*K)', 'b': 'W/m', 'coefficients': 'K',
         'left_reactions': 'W/m', 'integrated_heat': 'W/m', 'pointwise_heat_flux': 'W/m^2'}


class ReferenceError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def require(condition, code, message):
    if not condition:
        raise ReferenceError(code, message)


def exact_keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected), 'schema', f'{label}: unexpected or missing keys')


def binary64(text):
    require(isinstance(text, str), 'value', 'value must be a canonical finite binary64 hex string')
    try:
        value = float.fromhex(text)
    except (ValueError, OverflowError) as error:
        raise ReferenceError('value', 'invalid binary64 value') from error
    require(math.isfinite(value) and value.hex() == text, 'value', 'nonfinite or noncanonical binary64 value')
    return F(*value.as_integer_ratio())


def coordinate(text):
    require(isinstance(text, str), 'coordinate', 'coordinate must be an exact decimal string')
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise ReferenceError('coordinate', 'invalid coordinate') from error
    require(value.is_finite(), 'coordinate', 'nonfinite coordinate')
    return F(value)


def vector_norm(vector):
    return max((abs(value) for value in vector), default=F(0))


def matrix_norm(matrix):
    return max((sum((abs(value) for value in row), F(0)) for row in matrix), default=F(0))


def apply(matrix, vector):
    return [sum((a*b for a, b in zip(row, vector)), F(0)) for row in matrix]


def inverse(matrix):
    n = len(matrix)
    augmented = [list(row)+[F(int(i == j)) for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next((row for row in range(col, n) if augmented[row][col]), None)
        require(pivot is not None, 'reference', 'singular reference free operator')
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        divisor = augmented[col][col]
        augmented[col] = [value/divisor for value in augmented[col]]
        for row in range(n):
            if row != col:
                multiplier = augmented[row][col]
                augmented[row] = [a-multiplier*b for a, b in zip(augmented[row], augmented[col])]
    result = [row[n:] for row in augmented]
    for i, j in itertools.product(range(n), repeat=2):
        require(sum((matrix[i][k]*result[k][j] for k in range(n)), F(0)) == int(i == j),
                'reference', 'exact inverse multiplication failed')
    return result


def axis(length):
    ell = F(length, 2)
    mass = [[F(0) for _ in range(3)] for _ in range(3)]
    stiff = [[F(0) for _ in range(3)] for _ in range(3)]
    load = [F(0)]*3
    for cell in range(2):
        for i, j in itertools.product(range(2), repeat=2):
            mass[cell+i][cell+j] += ell*F(2 if i == j else 1, 6)
            stiff[cell+i][cell+j] += F(1 if i == j else -1)/ell
        for i in range(2):
            load[cell+i] += ell/2
    return mass, stiff, load


def reference(case):
    require(isinstance(case, str) and case in CASES, 'case', 'unsupported fixed case')
    length, height, conductivity, transfer, cold, ambient = map(F, CASES[case])
    mx, kx, _ = axis(length)
    my, ky, qy = axis(height)
    nodes = tuple(itertools.product(range(3), repeat=2))  # x-major, y-fast
    coordinates = tuple((length*i/2, height*j/2) for i, j in nodes)
    diffusion, robin, matrix = [], [], []
    for i, j in nodes:
        diffusion.append([conductivity*(kx[i][r]*my[j][s]+mx[i][r]*ky[j][s]) for r, s in nodes])
        robin.append([transfer*my[j][s] if i == r == 2 else F(0) for r, s in nodes])
        matrix.append([a+b for a, b in zip(diffusion[-1], robin[-1])])
    load = [transfer*ambient*qy[j] if i == 2 else F(0) for i, j in nodes]
    slope = transfer*(ambient-cold)/(conductivity+transfer*length)
    exact = [cold+slope*x for x, _ in coordinates]
    left, free = tuple(range(3)), tuple(range(3, 9))
    inv = inverse([[matrix[i][j] for j in free] for i in free])
    c = matrix_norm(inv)
    sa, u = matrix_norm(matrix), max(F(1), abs(cold), abs(ambient))
    bnorm = vector_norm(load)
    sb = sa*u+bnorm
    delta_a, delta_b, tau = ETA*sa, ETA*sb, SIGMA*sb
    require(c*delta_a < 1, 'reference', 'predeclared inverse perturbation denominator not positive')
    e = c*(tau+delta_a*u+delta_b)/(1-c*delta_a)
    recovery = ETA*((sa+delta_a)*(u+e)+bnorm+delta_b)
    rho = delta_a*(u+e)+delta_b+recovery
    targets = [-conductivity*slope*height/4, -conductivity*slope*height/2, -conductivity*slope*height/4]
    actual = [a-b for a, b in zip(apply(matrix, exact), load)]
    require(all(actual[i] == 0 for i in free) and [actual[i] for i in left] == targets,
            'reference', 'analytic affine solution/reaction identity failed')
    return {'case': case, 'parameters': (length, height, conductivity, transfer, cold, ambient),
            'coordinates': coordinates, 'A': matrix, 'b': load, 'diffusion': diffusion,
            'robin': robin, 'qy': qy, 'exact': exact, 'slope': slope, 'left': left, 'free': free,
            'inverse': inv, 'targets': targets,
            'bounds': {'eta': ETA, 'sigma': SIGMA, 'C': c, 'U': u, 'SA': sa, 'b_norm': bnorm,
                       'Sb': sb, 'deltaA': delta_a, 'deltab': delta_b, 'tau': tau, 'E': e, 'R': recovery,
                       'rho': rho, 'reference_residual': tau+delta_a*(u+e)+delta_b,
                       'individual_reaction': sa*e+rho, 'total_reaction': 3*(sa*e+rho)},
            'physical_heat': {'left': conductivity*slope*height, 'right': -conductivity*slope*height}}


def decode(data, ref):
    exact_keys(data, ('schema', 'case', 'order', 'units', 'A_binary64_hex', 'b_binary64_hex',
                      'coefficients', 'left_reactions', 'heat_convention'), 'input')
    require(type(data['schema']) is int and data['schema'] == 1, 'schema', 'unsupported schema')
    require(data['units'] == UNITS, 'units', 'missing or unsupported physical units')
    require(data['heat_convention'] == HEAT_CONVENTION, 'heat_sign', 'incorrect physical heat/weak reaction sign')
    def ordered_points(value, expected, label):
        require(isinstance(value, list) and len(value) == len(expected), 'count', f'{label}: incorrect count')
        result = []
        for coords in value:
            require(isinstance(coords, list) and len(coords) == 2, 'coordinate', f'{label}: invalid coordinate dimension')
            result.append(tuple(coordinate(text) for text in coords))
        require(tuple(result) == tuple(expected), 'order', f'{label}: coordinates/order differ from complete physical grid')
    ordered_points(data['order'], ref['coordinates'], 'operator order')
    a = data['A_binary64_hex']
    require(isinstance(a, list) and len(a) == 9 and all(isinstance(row, list) and len(row) == 9 for row in a),
            'count', 'full physical uneliminated 9x9 operator required')
    matrix = [[binary64(value) for value in row] for row in a]
    require(isinstance(data['b_binary64_hex'], list) and len(data['b_binary64_hex']) == 9,
            'count', 'full nine-entry physical load required')
    load = [binary64(value) for value in data['b_binary64_hex']]
    def records(name, expected):
        value = data[name]
        require(isinstance(value, list), 'schema', f'{name}: list required')
        for record in value:
            exact_keys(record, ('coordinates_m', 'value_binary64_hex'), name)
        ordered_points([record['coordinates_m'] for record in value], expected, name)
        return [binary64(record['value_binary64_hex']) for record in value]
    coefficients = records('coefficients', ref['coordinates'])
    reactions = records('left_reactions', [ref['coordinates'][i] for i in ref['left']])
    require(all(coefficients[i] == ref['exact'][i] for i in ref['left']),
            'essential', 'essential coefficients must be exactly Tc')
    return matrix, load, coefficients, reactions


def encode(value):
    if isinstance(value, F):
        return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def compare(data):
    require(isinstance(data, dict), 'schema', 'input must be an object')
    ref = reference(data.get('case'))
    a, b, v, native = decode(data, ref)
    limits = ref['bounds']
    operator_error = matrix_norm([[x-y for x, y in zip(row, expected)] for row, expected in zip(a, ref['A'])])
    load_error = vector_norm([x-y for x, y in zip(b, ref['b'])])
    physical_residual = [x-y for x, y in zip(apply(a, v), b)]
    reference_residual = [x-y for x, y in zip(apply(ref['A'], v), ref['b'])]
    free_residual = vector_norm([physical_residual[i] for i in ref['free']])
    reference_free = vector_norm([reference_residual[i] for i in ref['free']])
    coefficient_error = vector_norm([x-y for x, y in zip(v, ref['exact'])])
    actual_recovery = vector_norm([native[j]-physical_residual[i] for j, i in enumerate(ref['left'])])
    component_recovery = vector_norm([native[j]-reference_residual[i] for j, i in enumerate(ref['left'])])
    component_target = vector_norm([a-b for a, b in zip(native, ref['targets'])])
    total_target = abs(sum(native)-sum(ref['targets']))
    length, height, _, transfer, _, ambient = ref['parameters']
    right_heat = transfer*sum((ref['qy'][j]*(v[6+j]-ambient) for j in range(3)), F(0))
    left_heat = -sum(native)
    checks = {'full_operator': operator_error <= limits['deltaA'], 'full_load': load_error <= limits['deltab'],
              'physical_free_residual': free_residual <= limits['tau'],
              'every_coefficient': coefficient_error <= limits['E'],
              'reference_free_residual': reference_free <= limits['reference_residual'],
              'every_reaction_vs_exported_physical_operator': actual_recovery <= limits['R'],
              'every_reaction_vs_reference_full_vector': component_recovery <= limits['rho'],
              'every_reaction_vs_exact_target': component_target <= limits['individual_reaction'],
              'reaction_total': total_target <= limits['total_reaction']}
    return encode({'schema': 1, 'preparation_only': True, 'case': ref['case'], 'units': UNITS,
                   'counts': {'operator_rows': 9, 'load': 9, 'coefficients': 9, 'left_reactions': 3, 'free': 6},
                   'ceiling_status': 'eta=8192 binary64 unit roundoffs is declared, not an operation-count theorem',
                   'bounds': limits, 'observations': {'operator_inf_error': operator_error, 'load_inf_error': load_error,
                       'physical_free_residual_inf': free_residual, 'reference_free_residual_inf': reference_free,
                       'coefficient_inf_error': coefficient_error, 'native_vs_exported_recovery_inf_error': actual_recovery,
                       'native_vs_reference_recovery_inf_error': component_recovery,
                       'reaction_target_inf_error': component_target, 'reaction_total_target_error': total_target},
                   'heat_convention': HEAT_CONVENTION, 'physical_outward_heat': {'left': left_heat, 'right': right_heat},
                   'exact_physical_outward_heat': ref['physical_heat'], 'checks': checks, 'accepted': all(checks.values())})


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'schema', 'duplicate JSON object key')
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    make = commands.add_parser('reference')
    make.add_argument('case', choices=CASES)
    check = commands.add_parser('compare')
    check.add_argument('input', type=Path)
    for command in (make, check):
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = None
    try:
        if args.command == 'reference':
            result = encode(reference(args.case))
            result['preparation_only'] = True
        else:
            raw = args.input.read_bytes()
            result = compare(json.loads(raw, object_pairs_hook=unique_object))
    except (ReferenceError, ValueError, OSError) as error:
        result = {'accepted': False, 'error': getattr(error, 'code', 'input'), 'message': str(error)}
    if raw is not None:
        result['input_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    if 'error' in result:
        return 2
    if args.command == 'reference' or result.get('accepted'):
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
