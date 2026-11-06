"""Pure topology/basis falsifiers; toy triangles are not the cylinder benchmark."""
import copy
import unittest
import numpy as np
from topology import (ComparisonError, boundary_facets, check_permutation,
                      hierarchical_bubbles, mesh_arrays, cylinder_geometry_report)
from accuracy import (H, U, boundary_flux, check_rule, discrete_inlet_flux,
                      duffy_rule, physical_norms)


def fixture():
    return {'coordinates': [[0., 0.], [2.2, 0.], [0., H/2], [2.2, H/2], [0., H], [2.2, H]],
            'cells': [[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]],
            'boundaries': {'inlet': [[0, 2], [2, 4]], 'outlet': [[1, 3], [3, 5]],
                           'walls': [[4, 5]], 'cylinder': [[0, 1]]}}


class TopologyTests(unittest.TestCase):
    def test_boundary_mapping_by_identity_after_reordering(self):
        _, _, edges = mesh_arrays(fixture())
        ids = np.array([5, 3, 1, 4, 2, 0])
        inverse = np.argsort(ids)
        original = list(reversed(list(edges)))
        local = [[inverse[a], inverse[b]] for a, b in original]
        indices, labels = boundary_facets(local, ids, edges)
        self.assertEqual(indices.tolist(), list(range(6)))
        self.assertEqual(labels.tolist(), [edges[edge] for edge in original])
        with self.assertRaises(ComparisonError):
            boundary_facets(local[:-1], ids, edges)

    def test_topology_falsifiers(self):
        for mutate in (lambda d: d['boundaries']['inlet'].pop(),
                       lambda d: d['boundaries']['cylinder'].append([0, 2]),
                       lambda d: d['cells'][0].__setitem__(0, 99),
                       lambda d: d['cells'][0].__setitem__(0, 1)):
            data = copy.deepcopy(fixture()); mutate(data)
            with self.assertRaises(ComparisonError):
                mesh_arrays(data)
        with self.assertRaises(ComparisonError):
            check_permutation([0, 0, 2], 3, 'cells')
        np.testing.assert_array_equal(check_permutation([2, 0, 1], 3, 'cells'), [2, 0, 1])

    def test_hierarchical_conversion_preserves_cell_identity(self):
        vertices = np.array([[0., 0.], [3., 0.], [0., 3.], [3., 3.]])
        cells = np.array([[0, 1, 2], [1, 3, 2]])
        expected = np.array([[.5, -.5], [1., -1.]])
        centroids = vertices[cells].mean(axis=1)+expected
        np.testing.assert_array_equal(hierarchical_bubbles(vertices, centroids, cells), expected)
        np.testing.assert_array_equal(hierarchical_bubbles(vertices, centroids[::-1], cells[::-1]), expected[::-1])

    def test_clockwise_hole_and_area_from_cell_topology(self):
        data = {'coordinates': [[0., 0.], [3., 0.], [3., 3.], [0., 3.], [1., 1.], [2., 1.], [2., 2.], [1., 2.]],
                'cells': [[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
                'boundaries': {'inlet': [[3, 0]], 'outlet': [[1, 2]], 'walls': [[0, 1], [2, 3]],
                               'cylinder': [[4, 5], [5, 6], [6, 7], [7, 4]]}}
        report = cylinder_geometry_report(data)
        self.assertEqual(report['triangle_fluid_area'], 8.)
        self.assertEqual(report['hole_signed_area'], -1.)
        data['boundaries']['cylinder'], data['boundaries']['walls'] = data['boundaries']['walls'], data['boundaries']['cylinder']
        with self.assertRaises(ComparisonError):
            cylinder_geometry_report(data)

    def test_exact_triangle_moments_and_weight_falsifier(self):
        for order, degree in ((3, 4), (4, 6)):
            points, weights = duffy_rule(order); check_rule(points, weights, degree)
            with self.assertRaises(ComparisonError):
                check_rule(points, weights*1.01, degree)

    def test_norm_includes_bubble_and_pressure_level(self):
        data = fixture(); count = len(data['coordinates']); cells = len(data['cells'])
        fields = {'velocity_vertex': np.zeros((count, 2)),
                  'velocity_cell_bubble': np.tile([1., 0.], (cells, 1)),
                  'pressure_vertex': np.ones(count)}
        norms = physical_norms(data, fields)
        # integral_T b^2 / area(T) = 2*27^2*(2!)^3/8! = 81/280.
        self.assertAlmostEqual(norms['velocity']**2, 81/280, places=14)
        self.assertAlmostEqual(norms['pressure'], 1., places=14)

    def test_discrete_inlet_flux_is_not_continuum_parabola(self):
        data = fixture(); coordinates = np.array(data['coordinates'])
        vertices = np.column_stack((4*U*coordinates[:, 1]*(H-coordinates[:, 1])/H**2, np.zeros(6)))
        fields = {'velocity_vertex': vertices}
        # Two equal inlet segments give a piecewise-linear triangular profile.
        expected = -U*H/2
        self.assertAlmostEqual(discrete_inlet_flux(data), expected, places=15)
        self.assertAlmostEqual(boundary_flux(data, fields, 'inlet'), expected, places=15)
        self.assertNotAlmostEqual(expected, -2*U*H/3, places=5)


if __name__ == '__main__':
    unittest.main()
