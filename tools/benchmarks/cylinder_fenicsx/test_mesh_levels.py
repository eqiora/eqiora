"""Frozen mesh inputs are checked against geometry, not observed solutions."""
import math
import unittest
from mesh_levels import LEVELS, mesh_level
from topology import ComparisonError


class MeshLevelTests(unittest.TestCase):
    def test_standard_controls_are_the_original_example(self):
        self.assertEqual(mesh_level('standard').controls(), dict(maximum_target_size=.025,
                         maximum_boundary_error=1e-4, maximum_boundary_facets=50, minimum_mean_ratio=1e-5))

    def test_refinement_inputs_and_circle_sagitta(self):
        self.assertEqual(tuple(LEVELS), ('coarse', 'standard', 'fine'))
        previous = None
        for level in LEVELS.values():
            self.assertLessEqual(.05*(1-math.cos(math.pi/level.maximum_boundary_facets)), level.maximum_boundary_error)
            if previous:
                self.assertEqual(previous.maximum_target_size/level.maximum_target_size, 2)
                self.assertEqual(previous.maximum_boundary_error/level.maximum_boundary_error, 4)
                self.assertEqual(level.maximum_boundary_facets/previous.maximum_boundary_facets, 2)
            previous = level
        with self.assertRaises(ComparisonError): mesh_level('observed-best')


if __name__ == '__main__':
    unittest.main()
