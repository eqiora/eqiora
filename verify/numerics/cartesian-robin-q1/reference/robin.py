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
OBSERVATION_UNITS = {'coordinates': 'm', 'coefficients': 'K'}
OBSERVATION_QUADRATURE = {'family': 'gauss-legendre', 'points_per_segment': 2}


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


def _ordered_points(value, expected, label):
    require(isinstance(value, list) and len(value) == len(expected), 'count', f'{label}: incorrect count')
    result = []
    for coords in value:
        require(isinstance(coords, list) and len(coords) == 2, 'coordinate', f'{label}: invalid coordinate dimension')
        result.append(tuple(coordinate(text) for text in coords))
    require(tuple(result) == tuple(expected), 'order', f'{label}: coordinates/order differ from complete physical grid')


def _coordinate_records(value, expected, label):
    require(isinstance(value, list), 'schema', f'{label}: list required')
    for record in value:
        exact_keys(record, ('coordinates_m', 'value_binary64_hex'), label)
    _ordered_points([record['coordinates_m'] for record in value], expected, label)
    return [binary64(record['value_binary64_hex']) for record in value]


def _exact_essential(ref, coefficients):
    require(all(coefficients[i] == ref['exact'][i] for i in ref['left']),
            'essential', 'essential coefficients must be exactly Tc')


def decode(data, ref):
    exact_keys(data, ('schema', 'case', 'order', 'units', 'A_binary64_hex', 'b_binary64_hex',
                      'coefficients', 'left_reactions', 'heat_convention'), 'input')
    require(type(data['schema']) is int and data['schema'] == 1, 'schema', 'unsupported schema')
    require(data['units'] == UNITS, 'units', 'missing or unsupported physical units')
    require(data['heat_convention'] == HEAT_CONVENTION, 'heat_sign', 'incorrect physical heat/weak reaction sign')
    _ordered_points(data['order'], ref['coordinates'], 'operator order')
    a = data['A_binary64_hex']
    require(isinstance(a, list) and len(a) == 9 and all(isinstance(row, list) and len(row) == 9 for row in a),
            'count', 'full physical uneliminated 9x9 operator required')
    matrix = [[binary64(value) for value in row] for row in a]
    require(isinstance(data['b_binary64_hex'], list) and len(data['b_binary64_hex']) == 9,
            'count', 'full nine-entry physical load required')
    load = [binary64(value) for value in data['b_binary64_hex']]
    coefficients = _coordinate_records(data['coefficients'], ref['coordinates'], 'coefficients')
    reactions = _coordinate_records(data['left_reactions'], [ref['coordinates'][i] for i in ref['left']], 'left_reactions')
    _exact_essential(ref, coefficients)
    return matrix, load, coefficients, reactions


def encode(value):
    if isinstance(value, F):
        return {'numerator': str(value.numerator), 'denominator': str(value.denominator)}
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def _coefficient_measurements(ref, coefficients):
    """Exact full-field mathematics, independent of any reaction observation."""
    reference_residual = [x-y for x, y in zip(apply(ref['A'], coefficients), ref['b'])]
    _, _, _, transfer, _, ambient = ref['parameters']
    return {
        'reference_residual': reference_residual,
        'reference_free_residual_inf': vector_norm([reference_residual[i] for i in ref['free']]),
        'coefficient_inf_error': vector_norm([x-y for x, y in zip(coefficients, ref['exact'])]),
        'right_heat': transfer*sum((ref['qy'][j]*(coefficients[6+j]-ambient) for j in range(3)), F(0)),
    }


def _weak_reaction_measurements(ref, reference_residual, native):
    """Core full-operator measurements; not a claim of a public reaction API."""
    return {
        'native_vs_reference_recovery_inf_error': vector_norm(
            [native[j]-reference_residual[i] for j, i in enumerate(ref['left'])]),
        'reaction_target_inf_error': vector_norm([a-b for a, b in zip(native, ref['targets'])]),
        'reaction_total_target_error': abs(sum(native)-sum(ref['targets'])),
        'left_heat': -sum(native),
    }


def _coefficient_checks(measurements, limits):
    return {
        'every_coefficient': measurements['coefficient_inf_error'] <= limits['E'],
        'reference_free_residual': measurements['reference_free_residual_inf'] <= limits['reference_residual'],
    }


def _result_checks(measurements, limits):
    return {
        **_coefficient_checks(measurements, limits),
        'every_reaction_vs_reference_full_vector': measurements['native_vs_reference_recovery_inf_error'] <= limits['rho'],
        'every_reaction_vs_exact_target': measurements['reaction_target_inf_error'] <= limits['individual_reaction'],
        'reaction_total': measurements['reaction_total_target_error'] <= limits['total_reaction'],
    }


def _affine_functional(ref, coefficients, native, weights, constant=F(0)):
    """A predeclared exact functional; weights never come from candidate data."""
    require(len(coefficients) == len(weights) == 9, 'reference', 'functional requires nine physical coefficients')
    require(all(type(value) in (int, F) for value in weights) and type(constant) in (int, F),
            'reference', 'functional weights and constant must be exact rationals')
    weights, constant = tuple(map(F, weights)), F(constant)
    l1 = sum((abs(value) for value in weights), F(0))
    v_bound = ref['bounds']['U']+ref['bounds']['E']
    arithmetic = ETA*(l1*v_bound+abs(constant))
    target_allowance = l1*ref['bounds']['E']+arithmetic
    actual = sum((weight*value for weight, value in zip(weights, coefficients)), constant)
    target = sum((weight*value for weight, value in zip(weights, ref['exact'])), constant)
    recovery_error, target_error = abs(native-actual), abs(native-target)
    return {'weights_l1': l1, 'constant': constant, 'value_from_coefficients': actual,
            'analytic_target': target, 'arithmetic_ceiling': arithmetic,
            'target_allowance': target_allowance, 'recovery_error': recovery_error,
            'target_error': target_error, 'checks': {'coefficient_functional': recovery_error <= arithmetic,
                                                  'analytic_target': target_error <= target_allowance}}


def compare(data):
    require(isinstance(data, dict), 'schema', 'input must be an object')
    ref = reference(data.get('case'))
    a, b, v, native = decode(data, ref)
    limits = ref['bounds']
    operator_error = matrix_norm([[x-y for x, y in zip(row, expected)] for row, expected in zip(a, ref['A'])])
    load_error = vector_norm([x-y for x, y in zip(b, ref['b'])])
    physical_residual = [x-y for x, y in zip(apply(a, v), b)]
    free_residual = vector_norm([physical_residual[i] for i in ref['free']])
    actual_recovery = vector_norm([native[j]-physical_residual[i] for j, i in enumerate(ref['left'])])
    measured = _coefficient_measurements(ref, v)
    measured.update(_weak_reaction_measurements(ref, measured['reference_residual'], native))
    shared_checks = _result_checks(measured, limits)
    checks = {'full_operator': operator_error <= limits['deltaA'], 'full_load': load_error <= limits['deltab'],
              'physical_free_residual': free_residual <= limits['tau'],
              'every_coefficient': shared_checks['every_coefficient'],
              'reference_free_residual': shared_checks['reference_free_residual'],
              'every_reaction_vs_exported_physical_operator': actual_recovery <= limits['R'],
              'every_reaction_vs_reference_full_vector': shared_checks['every_reaction_vs_reference_full_vector'],
              'every_reaction_vs_exact_target': shared_checks['every_reaction_vs_exact_target'],
              'reaction_total': shared_checks['reaction_total']}
    return encode({'schema': 1, 'preparation_only': True, 'case': ref['case'], 'units': UNITS,
                   'counts': {'operator_rows': 9, 'load': 9, 'coefficients': 9, 'left_reactions': 3, 'free': 6},
                   'ceiling_status': 'eta=8192 binary64 unit roundoffs is declared, not an operation-count theorem',
                   'bounds': limits, 'observations': {'operator_inf_error': operator_error, 'load_inf_error': load_error,
                       'physical_free_residual_inf': free_residual,
                       'reference_free_residual_inf': measured['reference_free_residual_inf'],
                       'coefficient_inf_error': measured['coefficient_inf_error'],
                       'native_vs_exported_recovery_inf_error': actual_recovery,
                       'native_vs_reference_recovery_inf_error': measured['native_vs_reference_recovery_inf_error'],
                       'reaction_target_inf_error': measured['reaction_target_inf_error'],
                       'reaction_total_target_error': measured['reaction_total_target_error']},
                   'heat_convention': HEAT_CONVENTION,
                   'physical_outward_heat': {'left': measured['left_heat'], 'right': measured['right_heat']},
                   'exact_physical_outward_heat': ref['physical_heat'], 'checks': checks, 'accepted': all(checks.values())})


def _observation_specifications(ref):
    length, height, conductivity, transfer, _, ambient = ref['parameters']
    hx, hy = length/2, height/2
    qx, qy = [length/4, length/2, length/4], ref['qy']
    weights = [[F(0)]*9 for _ in range(6)]
    for j in range(3):
        weights[0][j] = -conductivity/hx*qy[j]
        weights[0][3+j] = conductivity/hx*qy[j]
        weights[1][3+j] = conductivity/hx*qy[j]
        weights[1][6+j] = -conductivity/hx*qy[j]
        weights[4][6+j] = qy[j]
        weights[5][6+j] = transfer*qy[j]
    for i in range(3):
        weights[2][3*i] = -conductivity/hy*qx[i]
        weights[2][3*i+1] = conductivity/hy*qx[i]
        weights[3][3*i+1] = conductivity/hy*qx[i]
        weights[3][3*i+2] = -conductivity/hy*qx[i]
    identities = (
        ('left_heat', 'left', 'outward_conductive_heat', 'W/m'),
        ('right_heat', 'right', 'outward_conductive_heat', 'W/m'),
        ('bottom_heat', 'bottom', 'outward_conductive_heat', 'W/m'),
        ('top_heat', 'top', 'outward_conductive_heat', 'W/m'),
        ('right_temperature', 'right', 'temperature_trace', 'K*m'),
        ('right_convective_heat', 'right', 'outward_convective_heat', 'W/m'),
    )
    constants = [F(0)]*5+[-transfer*height*ambient]
    norms = [4*conductivity*height/length]*2+[4*conductivity*length/height]*2+[height, transfer*height]
    left_heat = conductivity*ref['slope']*height
    targets = [left_heat, -left_heat, F(0), F(0), height*ref['exact'][6], -left_heat]
    result = []
    for identity, t, c, l1, target in zip(identities, weights, constants, norms, targets):
        require(sum(abs(value) for value in t) == l1, 'reference', 'functional norm identity failed')
        require(sum((a*b for a,b in zip(t, ref['exact'])), c) == target,
                'reference', 'functional analytic target failed')
        result.append(dict(zip(('name', 'boundary', 'quantity', 'unit'), identity), weights=tuple(t), constant=c))
    return result


def compare_observations(data):
    """Audit public physical observations without claiming an exported A/b audit."""
    exact_keys(data, ('schema', 'kind', 'case', 'order', 'units', 'coefficients', 'observations'), 'input')
    require(type(data['schema']) is int and data['schema'] == 1, 'schema', 'unsupported schema')
    require(data['kind'] == 'public-observations', 'schema', 'explicit public-observations kind required')
    require(data['units'] == OBSERVATION_UNITS, 'units', 'missing or unsupported result units')
    ref = reference(data['case'])
    _ordered_points(data['order'], ref['coordinates'], 'physical result order')
    values = _coordinate_records(data['coefficients'], ref['coordinates'], 'coefficients')
    _exact_essential(ref, values)
    specifications = _observation_specifications(ref)
    observations = data['observations']
    require(isinstance(observations, list) and len(observations) == 6, 'count', 'six ordered observations required')
    reports = []
    for record, spec in zip(observations, specifications):
        exact_keys(record, ('name', 'boundary', 'quantity', 'unit', 'quadrature',
                            'value_binary64_hex', 'replay_value_binary64_hex'), 'observation')
        require(all(record[key] == spec[key] for key in ('name', 'boundary', 'quantity', 'unit')),
                'observation_identity', 'incorrect observation name/order/boundary/quantity/unit')
        exact_keys(record['quadrature'], ('family', 'points_per_segment'), 'observation quadrature')
        require(record['quadrature']['family'] == 'gauss-legendre'
                and type(record['quadrature']['points_per_segment']) is int
                and record['quadrature']['points_per_segment'] == 2,
                'quadrature', 'two-point Gauss-Legendre per selected segment required')
        native = binary64(record['value_binary64_hex'])
        replay = binary64(record['replay_value_binary64_hex'])
        primary_check = _affine_functional(ref, values, native, spec['weights'], spec['constant'])
        replay_check = _affine_functional(ref, values, replay, spec['weights'], spec['constant'])
        checks = {'primary_vs_coefficients': primary_check['checks']['coefficient_functional'],
                  'primary_vs_analytic_target': primary_check['checks']['analytic_target'],
                  'replay_vs_coefficients': replay_check['checks']['coefficient_functional'],
                  'replay_vs_analytic_target': replay_check['checks']['analytic_target']}
        reports.append({**{key: spec[key] for key in ('name', 'boundary', 'quantity', 'unit')},
                        'quadrature': dict(OBSERVATION_QUADRATURE),
                        'weights': spec['weights'], 'constant': spec['constant'],
                        'weights_l1': primary_check['weights_l1'],
                        'value_from_coefficients': primary_check['value_from_coefficients'],
                        'analytic_target': primary_check['analytic_target'],
                        'arithmetic_ceiling': primary_check['arithmetic_ceiling'],
                        'target_allowance': primary_check['target_allowance'],
                        'primary': {'value': native, 'error_vs_coefficients': primary_check['recovery_error'],
                                    'error_vs_target': primary_check['target_error']},
                        'replay': {'value': replay, 'error_vs_coefficients': replay_check['recovery_error'],
                                   'error_vs_target': replay_check['target_error']}, 'checks': checks})
    measured = _coefficient_measurements(ref, values)
    checks = _coefficient_checks(measured, ref['bounds'])
    return encode({'schema': 1, 'kind': 'public-observations', 'preparation_only': True, 'case': ref['case'],
                   'scope': 'public result/observation checks; no measured A/b or private-reaction audit',
                   'units': OBSERVATION_UNITS,
                   'counts': {'coefficients': 9, 'free': 6, 'observations': 6, 'observation_evaluations': 12},
                   'ceiling_status': 'eta=8192 binary64 unit roundoffs is declared, not an operation-count theorem',
                   'coefficient_bounds': {'E': ref['bounds']['E'], 'reference_free_residual': ref['bounds']['reference_residual']},
                   'coefficient_observations': {'coefficient_inf_error': measured['coefficient_inf_error'],
                                                'reference_free_residual_inf': measured['reference_free_residual_inf']},
                   'observations': reports, 'checks': checks,
                   'accepted': all(checks.values()) and all(all(report['checks'].values()) for report in reports)})


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
    observations = commands.add_parser('compare-observations')
    observations.add_argument('input', type=Path)
    for command in (make, check, observations):
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = None
    try:
        if args.command == 'reference':
            result = encode(reference(args.case))
            result['preparation_only'] = True
        else:
            raw = args.input.read_bytes()
            payload = json.loads(raw, object_pairs_hook=unique_object)
            result = compare(payload) if args.command == 'compare' else compare_observations(payload)
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
