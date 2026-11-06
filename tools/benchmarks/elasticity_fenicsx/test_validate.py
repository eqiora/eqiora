"""Saved PASS summaries cannot override the independently reconstructed field."""
import copy
import unittest
from accuracy import ComparisonError, LEVELS
from test_accuracy import fixture
from validate import validate_reports


def reports():
    solve = {'algorithm': 'conjugate-gradient', 'preconditioner': 'identity',
             'relative_tolerance': 1e-12, 'absolute_tolerance': 1e-14,
             'maximum_iterations': 10000, 'initial_residual_norm': 1.,
             'true_residual_norm': 0., 'residual_target': 1e-12}
    result = []
    for tool in ('eqiora', 'fenicsx'):
        for cells in LEVELS:
            coordinates, values = fixture(cells)
            result.append({'tool': tool, 'cells_per_axis': cells,
                           'coordinates': coordinates.tolist(), 'values': values.tolist(),
                           'solve': copy.deepcopy(solve),
                           'equilibrium_probe': {'reaction': [-1., 0.], 'body_force': [1., 0.],
                                                 'solve': copy.deepcopy(solve)}})
    return result


class SavedReportTests(unittest.TestCase):
    def test_entire_analytic_sequence(self):
        self.assertEqual(set(validate_reports(reports())), {'eqiora', 'fenicsx'})

    def test_false_pass_cannot_hide_bad_field(self):
        rows = reports()
        rows[0]['accuracy'] = {'passed': True, 'l2_error': 0.}
        rows[0]['values'] = [[0., 0.] for _ in rows[0]['values']]
        with self.assertRaises(ComparisonError):
            validate_reports(rows)

    def test_policy_residual_and_inventory_falsifiers(self):
        rows = reports()
        for field, value in (('preconditioner', 'jacobi'), ('true_residual_norm', 1e-10),
                             ('true_residual_norm', float('nan')), ('maximum_iterations', 10001)):
            corrupt = copy.deepcopy(rows); corrupt[0]['solve'][field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ComparisonError):
                    validate_reports(corrupt)
        with self.assertRaises(ComparisonError):
            validate_reports(rows+rows[:1])
        with self.assertRaises(ComparisonError):
            validate_reports(rows[:-1])


if __name__ == '__main__':
    unittest.main()
