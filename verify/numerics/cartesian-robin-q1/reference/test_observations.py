"""Public-observation mathematics and synthetic falsifiers; no imported solves."""
import copy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import robin as r

IDENTITIES = (
    ('left_heat', 'left', 'outward_conductive_heat', 'W/m'),
    ('right_heat', 'right', 'outward_conductive_heat', 'W/m'),
    ('bottom_heat', 'bottom', 'outward_conductive_heat', 'W/m'),
    ('top_heat', 'top', 'outward_conductive_heat', 'W/m'),
    ('right_temperature', 'right', 'temperature_trace', 'K*m'),
    ('right_convective_heat', 'right', 'outward_convective_heat', 'W/m'),
)


def document(case):
    length, height, k, h, tc, ambient = map(F, r.CASES[case])
    slope = h*(ambient-tc)/(k+h*length)
    points = [(length*i/2, height*j/2) for i in range(3) for j in range(3)]
    targets = [k*slope*height, -k*slope*height, F(0), F(0),
               height*(tc+slope*length), h*height*(tc+slope*length-ambient)]
    def coords(point):
        return [str(float(value)) for value in point]
    observations = []
    for identity, target in zip(IDENTITIES, targets):
        observations.append({**dict(zip(('name', 'boundary', 'quantity', 'unit'), identity)),
            'quadrature': {'family': 'gauss-legendre', 'points_per_segment': 2},
            'value_binary64_hex': float(target).hex(), 'replay_value_binary64_hex': float(target).hex()})
    return {'schema': 1, 'kind': 'public-observations', 'case': case,
            'order': [coords(point) for point in points], 'units': {'coordinates': 'm', 'coefficients': 'K'},
            'coefficients': [{'coordinates_m': coords(point), 'value_binary64_hex': float(tc+slope*point[0]).hex()}
                             for point in points], 'observations': observations}


def fraction(value):
    return F(int(value['numerator']), int(value['denominator']))


class ObservationTests(unittest.TestCase):
    def test_four_analytic_cases_and_only_measured_scope(self):
        forbidden = {'full_operator', 'full_load', 'physical_free_residual',
                     'every_reaction_vs_exported_physical_operator', 'every_reaction_vs_reference_full_vector'}
        for case in r.CASES:
            result = r.compare_observations(document(case))
            self.assertTrue(result['accepted'], (case, result))
            self.assertEqual(result['kind'], 'public-observations')
            self.assertEqual(result['counts']['observation_evaluations'], 12)
            self.assertFalse(forbidden.intersection(result['checks']))
            self.assertNotIn('A', result['units'])
            self.assertNotIn('R', result['coefficient_bounds'])
            for report in result['observations']:
                self.assertTrue(all(report['checks'].values()))

    def test_full_q1_face_weights_by_piecewise_exact_integration(self):
        for case in r.CASES:
            ref = r.reference(case)
            length, height, k, h, _, ambient = ref['parameters']
            hx, hy = length/2, height/2
            values = [F(7*i*i+3*j*j*j+i*j+11, 5) for i in range(3) for j in range(3)]
            def v(i, j):
                return values[3*i+j]
            # Integrate each segment's linear endpoint interpolant exactly.
            left = k*sum(hy/2*((v(1,j)-v(0,j))/hx+(v(1,j+1)-v(0,j+1))/hx) for j in range(2))
            right = -k*sum(hy/2*((v(2,j)-v(1,j))/hx+(v(2,j+1)-v(1,j+1))/hx) for j in range(2))
            bottom = k*sum(hx/2*((v(i,1)-v(i,0))/hy+(v(i+1,1)-v(i+1,0))/hy) for i in range(2))
            top = -k*sum(hx/2*((v(i,2)-v(i,1))/hy+(v(i+1,2)-v(i+1,1))/hy) for i in range(2))
            trace = sum(hy/2*(v(2,j)+v(2,j+1)) for j in range(2))
            expected = (left, right, bottom, top, trace, h*(trace-height*ambient))
            for spec, target in zip(r._observation_specifications(ref), expected):
                measured = sum((t*v for t,v in zip(spec['weights'], values)), spec['constant'])
                self.assertEqual(measured, target)

    def test_fixed_functional_norms_and_zero_transfer(self):
        for case in r.CASES:
            result = r.compare_observations(document(case))
            length,height,k,h,_,_ = map(F, r.CASES[case])
            expected = [4*k*height/length]*2+[4*k*length/height]*2+[height,h*height]
            ref = r.reference(case)
            for report, norm in zip(result['observations'], expected):
                self.assertEqual(fraction(report['weights_l1']), norm)
                c = fraction(report['constant'])
                tau = r.ETA*(norm*(ref['bounds']['U']+ref['bounds']['E'])+abs(c))
                self.assertEqual(fraction(report['arithmetic_ceiling']), tau)
                self.assertEqual(fraction(report['target_allowance']), norm*ref['bounds']['E']+tau)
        data = document('insulated')
        zero = r.compare_observations(data)['observations'][-1]
        self.assertEqual(fraction(zero['arithmetic_ceiling']), 0)
        self.assertEqual(fraction(zero['analytic_target']), 0)
        data['observations'][-1]['replay_value_binary64_hex'] = '0x0.0000000000001p-1022'
        self.assertFalse(r.compare_observations(data)['accepted'])

    def test_replay_checked_independently_without_bitwise_promise(self):
        data = document('insulated')
        data['observations'][0]['replay_value_binary64_hex'] = '0x0.0000000000001p-1022'
        self.assertTrue(r.compare_observations(data)['accepted'])
        data = document('robin')
        data['observations'][0]['replay_value_binary64_hex'] = (1000.0).hex()
        result = r.compare_observations(data)
        self.assertFalse(result['accepted'])
        self.assertTrue(result['observations'][0]['checks']['primary_vs_coefficients'])
        self.assertFalse(result['observations'][0]['checks']['replay_vs_coefficients'])

    def test_sign_face_temperature_endpoint_and_coefficient_falsifiers(self):
        for defect in ('sign', 'swap_values', 'omit_endpoint_weights', 'coefficient', 'convective_offset'):
            data = document('robin')
            if defect == 'sign':
                record = data['observations'][0]
                record['value_binary64_hex'] = (-float.fromhex(record['value_binary64_hex'])).hex()
            elif defect == 'swap_values':
                a,b = data['observations'][:2]
                a['value_binary64_hex'],b['value_binary64_hex'] = b['value_binary64_hex'],a['value_binary64_hex']
            elif defect == 'omit_endpoint_weights':
                target = float.fromhex(data['observations'][4]['value_binary64_hex'])
                data['observations'][4]['value_binary64_hex'] = (target/2).hex()
            elif defect == 'coefficient':
                record = data['coefficients'][4]
                record['value_binary64_hex'] = (float.fromhex(record['value_binary64_hex'])+2**-10).hex()
            else:
                _,height,_,h,_,ambient = r.CASES['robin']
                record = data['observations'][-1]
                record['value_binary64_hex'] = (float.fromhex(record['value_binary64_hex'])+h*height*ambient).hex()
            self.assertFalse(r.compare_observations(data)['accepted'], defect)

    def test_strict_complete_identity_controls_and_no_private_inputs(self):
        changes = [lambda x:x['coefficients'].pop(), lambda x:x['coefficients'].reverse(),
                   lambda x:x['coefficients'][1].update(coordinates_m=x['coefficients'][0]['coordinates_m']),
                   lambda x:x['order'].reverse(), lambda x:x['observations'].pop(),
                   lambda x:x['observations'].reverse(), lambda x:x['observations'][0].update(boundary='right'),
                   lambda x:x['observations'][0].update(quantity='weak_reaction'),
                   lambda x:x['observations'][4].update(unit='K'),
                   lambda x:x['observations'][0]['quadrature'].update(points_per_segment=1),
                   lambda x:x['observations'][0]['quadrature'].update(points_per_segment=True),
                   lambda x:x['observations'][0]['quadrature'].update(extra=0),
                   lambda x:x['observations'][0].update(value_binary64_hex='nan'),
                   lambda x:x['observations'][0].update(replay_value_binary64_hex='inf'),
                   lambda x:x['coefficients'][0].update(value_binary64_hex=(280+2**-20).hex()),
                   lambda x:x.update(kind='full-operator'), lambda x:x.update(A_binary64_hex=[]),
                   lambda x:x.update(b_binary64_hex=[]), lambda x:x.update(left_reactions=[]),
                   lambda x:x['units'].update(A='W/(m*K)')]
        for change in changes:
            data = document('robin')
            change(data)
            with self.subTest(change=change):
                with self.assertRaises(r.ReferenceError):
                    r.compare_observations(data)

    def test_cli_input_digest_and_explicit_entry(self):
        with tempfile.TemporaryDirectory(dir=Path.home()) as folder:
            source, output = Path(folder)/'input.json', Path(folder)/'output.json'
            raw = (json.dumps(document('changed_h'), indent=2)+'\n').encode()
            source.write_bytes(raw)
            completed = subprocess.run([sys.executable, str(Path(r.__file__)), 'compare-observations', str(source),
                                        '--output', str(output)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result['input_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(result['kind'], 'public-observations')


if __name__ == '__main__':
    unittest.main()
