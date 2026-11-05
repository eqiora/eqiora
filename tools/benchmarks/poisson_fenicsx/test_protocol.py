"""Independent analytic and admission falsifiers; no native library execution."""
import importlib.util
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from accuracy import ComparisonError, LEVELS, check_accuracy, check_identity_row, q1_vertex_dof_coordinates

spec = importlib.util.spec_from_file_location("comparison_driver", Path(__file__).with_name("run.py"))
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def analytic_discrete_solution(cells):
    # Independent existing reference derivation: tensor stiffness eigenvalue
    # 2*k*m, with k=4*sin(theta/2)^2/h and m=h*(2+cos(theta))/3.
    # Two-point Gauss load coefficient b=h*sum(1-t)*cos(theta*t).
    # See verify/numerics/cartesian-poisson-fem-fvm/references/README.md.
    h, theta = 1/cells, np.pi/cells
    k, mass = 4*np.sin(theta/2)**2/h, h*(2+np.cos(theta))/3
    t = (1+np.array([-1., 1.])/np.sqrt(3))/2
    load = h*np.sum((1-t)*np.cos(theta*t))
    amplitude = np.pi**2*load**2/(k*mass)
    coordinates = np.array([(i/cells, j/cells) for i in range(cells+1) for j in range(cells+1)])
    values = amplitude*np.prod(np.sin(np.pi*coordinates), axis=1)
    boundary = np.any((coordinates == 0) | (coordinates == 1), axis=1)
    values[boundary] = 0
    return coordinates, values


class ProtocolTests(unittest.TestCase):
    def test_analytic_discrete_fields_match_existing_continuous_error_table(self):
        for cells in LEVELS:
            coordinates, values = analytic_discrete_solution(cells)
            report = check_accuracy(coordinates, values, cells)
            self.assertLessEqual(abs(report["continuous_l2"]-report["analytic_discrete_l2"]),
                                 report["comparison_allowance"])

    def test_valid_field_falsifiers_do_not_reuse_invalid_baselines(self):
        coordinates, values = analytic_discrete_solution(4)
        check_accuracy(coordinates, values, 4)
        for invalid in (float("nan"), float("inf"), -float("inf")):
            poisoned = values.copy()
            poisoned[12] = invalid
            with self.assertRaises(ComparisonError):
                check_accuracy(coordinates, poisoned, 4)
        with self.assertRaises(ComparisonError):
            check_accuracy(coordinates, values*1.1, 4)
        wrong_boundary = values.copy()
        wrong_boundary[0] = 1
        with self.assertRaises(ComparisonError):
            check_accuracy(coordinates, wrong_boundary, 4)
        duplicate = coordinates.copy()
        duplicate[1] = duplicate[0]
        with self.assertRaises(ComparisonError):
            check_accuracy(duplicate, values, 4)
        with self.assertRaises(ComparisonError):
            check_accuracy(coordinates, values, 64)

    def test_nonfinite_reference_cannot_turn_the_comparator_into_a_vacuous_pass(self):
        coordinates, values = analytic_discrete_solution(4)
        check_accuracy(coordinates, values, 4)
        with tempfile.TemporaryDirectory() as temporary:
            reference = Path(temporary) / "reference.csv"
            for invalid in ("nan", "inf", "-inf"):
                reference.write_text(f"cells_per_axis,fem_l2\n4,{invalid}\n")
                with patch("accuracy.REFERENCE", reference), self.assertRaises(ComparisonError):
                    check_accuracy(coordinates, values, 4)

    def test_boundary_identity_row_requires_one_actual_unit_diagonal(self):
        check_identity_row(3, [1, 3, 4], [0.0, 1.0, 0.0])
        for columns, data in (([], []), ([1, 4], [0.0, 0.0]),
                              ([3], [0.0]), ([3, 3], [1.0, 1.0]),
                              ([3, 4], [1.0, 1.0])):
            with self.assertRaises(ComparisonError):
                check_identity_row(3, columns, data)

    def test_topological_export_preserves_point_coefficient_pairs_under_permutation(self):
        rows = [(0, [2], [0., 0.]), (1, [0], [1., 0.]),
                (2, [3], [1., 1.]), (3, [1], [0., 1.])]
        expected = np.array([[1., 0.], [0., 1.], [0., 0.], [1., 1.]])
        # u=2x+3y gives these independent coefficients in actual dof order.
        values = np.array([2., 3., 0., 5.])
        for reordered in (rows, rows[::-1], [rows[i] for i in (2, 0, 3, 1)]):
            coordinates = q1_vertex_dof_coordinates(reordered, 4)
            np.testing.assert_array_equal(coordinates, expected)
            np.testing.assert_array_equal(2*coordinates[:, 0]+3*coordinates[:, 1], values)
        # Reverse dof numbering AND its coefficients; physical pairs persist.
        reversed_rows = [(vertex, [3-index[0]], point) for vertex, index, point in rows]
        coordinates = q1_vertex_dof_coordinates(reversed_rows, 4)
        np.testing.assert_array_equal(2*coordinates[:, 0]+3*coordinates[:, 1], values[::-1])
        for invalid in (rows[:-1], rows+[rows[0]],
                        [rows[0], (1, [2], [1., 0.]), *rows[2:]],
                        [(0, [2, 3], [0., 0.]), *rows[1:]],
                        [(0, [4], [0., 0.]), *rows[1:]]):
            with self.assertRaises(ComparisonError):
                q1_vertex_dof_coordinates(invalid, 4)

    def test_foreign_loader_environment_is_removed_only_for_fenicsx(self):
        foreign = {"LD_LIBRARY_PATH": "/foreign-openmpi", "LD_PRELOAD": "/foreign.so",
                   "OMPI_MCA_btl": "foreign", "PYTHONPATH": "/repository-source",
                   "MPICC": "/foreign/mpicc", "PETSC_OPTIONS": "-mat_type aijcusparse",
                   "CFLAGS": "-ffast-math"}
        with patch.dict(os.environ, foreign):
            fenicsx = driver.child_environment("fenicsx", Path("/isolated/bin/python"), Path("/home/scratch"))
            eqiora = driver.child_environment("eqiora", Path("/wheel/bin/python"), Path("/home/scratch"))
        for key in foreign:
            self.assertNotIn(key, fenicsx)
        self.assertEqual(eqiora["LD_LIBRARY_PATH"], "/foreign-openmpi")
        self.assertNotIn("LD_PRELOAD", eqiora)
        self.assertTrue(all(fenicsx[key] == "1" for key in driver.THREAD_KEYS))


if __name__ == "__main__":
    unittest.main()
