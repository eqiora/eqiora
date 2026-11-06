"""External-only empirical engineering budget; Eqiora output cannot calibrate it."""
import itertools
import math
import numpy as np
from accuracy import U, P, FORCE_SCALE, physical_norms, field_distances
from shared_measurement import ComparisonError

METRICS = ('velocity', 'pressure', 'force_x', 'force_y')


def check_force_pair(row):
    on_domain, on_selection = np.asarray(row['force_on_domain']), np.asarray(row['force_on_selection'])
    if (on_domain.shape != (2,) or on_selection.shape != (2,)
            or not np.isfinite(on_domain).all() or not np.isfinite(on_selection).all()
            or not np.array_equal(on_domain, -on_selection)):
        raise ComparisonError('public force labels must retain exact on-domain/on-selection action-reaction')


def differences(data, left, right):
    norms = field_distances(data, left['fields'], right['fields'])
    force = np.abs(np.asarray(left['force_on_selection'])-np.asarray(right['force_on_selection']))
    return {**norms, 'force_x': float(force[0]), 'force_y': float(force[1])}


def residual_ceiling(maximum_row_nonzeros, matrix_inf_norm, reference_x_inf_norm, rhs_inf_norm):
    eps = np.finfo(np.float64).eps
    m = int(maximum_row_nonzeros)+2
    denominator = 1-m*eps
    scale = matrix_inf_norm*reference_x_inf_norm+rhs_inf_norm
    if denominator <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ComparisonError('inconclusive external residual scaling')
    gamma = m*eps/denominator
    return {'m': m, 'epsilon': float(eps), 'gamma': gamma, 'scale_D': scale,
            'absolute_scaled_residual_ceiling': 100*gamma*scale}


def freeze_budget(data, references, residual_control):
    if tuple(references) != ('A', 'B', 'C'):
        raise ComparisonError('requires the designated external A/B/C batch, without reference selection')
    if any(not math.isfinite(r['scaled_residual_inf']) or r['scaled_residual_inf'] < 0 or r['scaled_residual_inf'] > residual_control['absolute_scaled_residual_ceiling']
           for r in references.values()):
        raise ComparisonError('external reference failed the predeclared backward residual ceiling')
    for row in references.values():
        check_force_pair(row)
        force = np.asarray(row['force_on_selection'])
        if force.shape != (2,) or not np.isfinite(force).all():
            raise ComparisonError('invalid external force vector')
    reference = references['A']
    amplitudes = physical_norms(data, reference['fields'])
    fx, fy = np.abs(reference['force_on_selection'])
    amplitudes.update(force_x=float(fx), force_y=float(fy))
    scales = dict(velocity=U, pressure=P, force_x=FORCE_SCALE, force_y=FORCE_SCALE)
    spread = dict.fromkeys(METRICS, 0.)
    for left, right in itertools.combinations(references.values(), 2):
        for key, value in differences(data, left, right).items():
            spread[key] = max(spread[key], value)
    zero = {key: np.zeros_like(values) for key, values in reference['fields'].items()}
    without_bubbles = {key: np.asarray(values).copy() for key, values in reference['fields'].items()}
    without_bubbles['velocity_cell_bubble'][:] = 0
    zero_dist = field_distances(data, reference['fields'], zero)
    bubble_dist = field_distances(data, reference['fields'], without_bubbles)['velocity']
    mutant_distances = dict(velocity=min(zero_dist['velocity'], bubble_dist),
                            pressure=min(zero_dist['pressure'], P),
                            force_x=min(float(fx), FORCE_SCALE), force_y=FORCE_SCALE)
    metrics = {}
    for key in METRICS:
        scale = max(scales[key], amplitudes[key])
        tolerance = max(np.sqrt(np.finfo(float).eps)*scale, 10*spread[key])
        broad_cap = 1e-4*scale
        separation_cap = mutant_distances[key]/20
        if not all(math.isfinite(v) for v in (scale, tolerance, broad_cap, separation_cap)) or tolerance > min(broad_cap, separation_cap):
            raise ComparisonError(f'inconclusive external engineering budget for {key}; do not enlarge it')
        metrics[key] = {'input_scale': scales[key], 'reference_amplitude': amplitudes[key],
                        'scale': scale, 'external_spread': spread[key], 'tolerance': float(tolerance),
                        'broad_cap': broad_cap, 'mutant_distance': mutant_distances[key],
                        'separation_cap': separation_cap}
    return {'kind': 'empirical external-only engineering agreement; not a certified continuum or forward-error bound',
            'designated_reference': 'A', 'reference_order': ['A', 'B', 'C'],
            'constants': {'spread_multiplier': 10, 'broad_cap_fraction': 1e-4, 'mutant_separation': 20,
                          'residual_gamma_multiplier': 100},
            'residual_control': residual_control, 'metrics': metrics}


def compare(data, candidate, reference, frozen):
    check_force_pair(candidate)
    check_force_pair(reference)
    differences_from_a = differences(data, candidate, reference)
    for key, value in differences_from_a.items():
        if not math.isfinite(value) or value > frozen['metrics'][key]['tolerance']:
            raise ComparisonError(f'candidate fails frozen {key} agreement gate')
    residual = candidate['scaled_residual_inf']
    if not math.isfinite(residual) or residual < 0 or residual > frozen['residual_control']['absolute_scaled_residual_ceiling']:
        raise ComparisonError('candidate fails frozen external-operator residual ceiling')
    return differences_from_a


def check_falsifiers(data, reference, frozen):
    """Run the protocol's physical mutants through the same comparison gate."""
    import copy
    mutants = {}
    for name in ('zero_velocity', 'omit_bubbles', 'zero_pressure', 'pressure_shift',
                 'zero_drag', 'reverse_drag', 'shift_drag', 'shift_lift'):
        mutant = copy.deepcopy(reference)
        for key, value in mutant['fields'].items(): mutant['fields'][key] = np.asarray(value).copy()
        mutant['force_on_selection'] = np.asarray(mutant['force_on_selection']).copy()
        if name == 'zero_velocity':
            mutant['fields']['velocity_vertex'][:] = 0; mutant['fields']['velocity_cell_bubble'][:] = 0
        elif name == 'omit_bubbles': mutant['fields']['velocity_cell_bubble'][:] = 0
        elif name == 'zero_pressure': mutant['fields']['pressure_vertex'][:] = 0
        elif name == 'pressure_shift': mutant['fields']['pressure_vertex'] += P
        elif name == 'zero_drag': mutant['force_on_selection'][0] = 0
        elif name == 'reverse_drag': mutant['force_on_selection'][0] *= -1
        elif name == 'shift_drag': mutant['force_on_selection'][0] += FORCE_SCALE
        elif name == 'shift_lift': mutant['force_on_selection'][1] += FORCE_SCALE
        mutant['force_on_domain'] = -mutant['force_on_selection']
        mutants[name] = mutant
    if abs(reference['force_on_selection'][1]) >= FORCE_SCALE:
        for name in ('zero_lift', 'reverse_lift'):
            mutant = copy.deepcopy(reference)
            mutant['force_on_selection'] = np.asarray(mutant['force_on_selection']).copy()
            mutant['force_on_selection'][1] = 0 if name == 'zero_lift' else -mutant['force_on_selection'][1]
            mutant['force_on_domain'] = -mutant['force_on_selection']
            mutants[name] = mutant
    rejected = []
    for name, mutant in mutants.items():
        try:
            compare(data, mutant, reference, frozen)
        except ComparisonError:
            rejected.append(name)
        else:
            raise ComparisonError(f'inconclusive budget admits deliberate {name} mutant')
    return rejected
