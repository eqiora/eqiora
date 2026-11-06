"""Candidate gates on synthetic physical coefficients; no Eqiora execution."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from accuracy import U, H, boundary_flux
from budget import freeze_budget, residual_ceiling
from common_basis import canonical_fields, layout
from reference_reader import Reference, associated, load_reference
from topology import ComparisonError
from test_topology import fixture


def context():
    data = fixture(); coordinates = np.asarray(data['coordinates'])
    fields = {'velocity_vertex': np.column_stack((4*U*coordinates[:, 1]*(H-coordinates[:, 1])/H**2, np.zeros(6))),
              'velocity_cell_bubble': np.array([[.01, -.02], [.02, -.01], [.03, -.04], [.04, -.03]]),
              'pressure_vertex': np.full(6, .01)}
    reference = {'fields': fields, 'force_on_domain': [-.006, -.00001], 'force_on_selection': [.006, .00001],
                 'scaled_residual_inf': 0.}
    basis = layout(data); vector = canonical_fields(fields)[basis['free']]/basis['scales']; n = len(vector)
    operator = {'indptr': np.arange(n+1), 'indices': np.arange(n), 'values': np.ones(n),
                'rhs': vector, 'free': basis['free'], 'scales': basis['scales']}
    control = residual_ceiling(1, 1., float(np.max(np.abs(vector))), float(np.max(np.abs(vector))))
    frozen = freeze_budget(data, {label: reference for label in ('A', 'B', 'C')}, control)
    reader = Reference(Path('.'), frozen, 'unit-reference', data, reference, operator, {})
    extra = associated(fields, reference['force_on_domain'], reference['force_on_selection'],
                       boundary_flux(data, fields, 'inlet'), boundary_flux(data, fields, 'outlet'))
    return reader, coordinates, fields['velocity_vertex'], extra


class CandidateTests(unittest.TestCase):
    def test_complete_associations_are_used_by_the_same_gate(self):
        reader, coordinates, velocity, extra = context()
        result = reader.candidate(coordinates, velocity, extra)
        self.assertEqual(result['agreement']['velocity'], 0.)
        for name in ('zero', 'sign', 'bubble', 'pressure', 'association', 'flux'):
            bad_velocity = velocity.copy(); bad_extra = copy.deepcopy(extra)
            if name == 'zero':
                bad_velocity[:] = 0; bad_extra['velocity_cell_bubble']['values'][:] = 0
            elif name == 'sign':
                bad_extra['cylinder_force_on_domain']['values'] *= -1
                bad_extra['cylinder_force_on_selection']['values'] *= -1
            elif name == 'bubble': bad_extra['velocity_cell_bubble']['values'][:] = 0
            elif name == 'pressure': bad_extra['pressure_vertex']['values'] += .001*.3/.41
            elif name == 'association': bad_extra['pressure_vertex']['association'] = 'cell'
            elif name == 'flux': bad_extra['inlet_flux']['values'][0] *= -1
            with self.subTest(falsifier=name):
                with self.assertRaises(ComparisonError): reader.candidate(coordinates, bad_velocity, bad_extra)
        with self.assertRaises(ComparisonError): reader.candidate(coordinates[::-1], velocity, extra)

    def test_changed_manifest_hash_fails_before_reference_read(self):
        scratch = Path.home()/'.cache/eqiora/cylinder-reader-tests'; scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path = Path(directory)/'manifest.json'; path.write_text('{}')
            with self.assertRaises(ComparisonError): load_reference(path, '0'*64)


if __name__ == '__main__':
    unittest.main()
