"""Optional locked-environment basis/preparation checks; no PDE solve is run."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from test_topology import fixture

AVAILABLE = importlib.util.find_spec('dolfinx') is not None


@unittest.skipUnless(AVAILABLE, 'requires the existing locked FEniCSx environment')
class NativeBasisTests(unittest.TestCase):
    def test_enriched_basis_and_hierarchical_field_reconstruction(self):
        import basix.ufl
        from dolfinx import fem
        from fenicsx_adapter import prepare, export_fields, cylinder_indicator
        data = fixture(); prepared = prepare(data)
        element = basix.ufl.enriched_element([basix.ufl.element('Lagrange', 'triangle', 1),
                                              basix.ufl.element('Bubble', 'triangle', 3)])
        points = np.array([[.1, .2], [.2, .5], [.3, .3], [0., .5]])
        bary = np.column_stack((1-points.sum(axis=1), points)); bubble = 27*np.prod(bary, axis=1)
        expected_basis = np.column_stack((bary-bubble[:, None]/3, bubble))
        np.testing.assert_allclose(element.tabulate(0, points)[0], expected_basis,
                                   rtol=256*np.finfo(float).eps, atol=256*np.finfo(float).eps)
        coordinates, cells = prepared.coordinates, prepared.cells
        vertex = np.column_stack((1+2*coordinates[:, 0]-3*coordinates[:, 1],
                                  -2+coordinates[:, 0]+.5*coordinates[:, 1]))
        beta = np.column_stack((np.arange(len(cells))+.25, -np.arange(len(cells))-.5))
        pressure = 2+coordinates[:, 0]-coordinates[:, 1]
        vf = fem.Function(prepared.velocity_space)
        nodal = vf.x.array.reshape(-1, 2)
        nodal[prepared.vertex_velocity_dofs] = vertex
        nodal[prepared.cell_centroid_dofs] = vertex[cells].mean(axis=1)+beta
        function = fem.Function(prepared.mixed_space)
        function.x.array[prepared.velocity_to_mixed] = vf.x.array
        function.x.array[prepared.pressure_to_mixed[prepared.vertex_pressure_dofs]] = pressure
        exported = export_fields(prepared, function)
        np.testing.assert_allclose(exported['velocity_vertex'], vertex, rtol=0, atol=256*np.finfo(float).eps)
        np.testing.assert_allclose(exported['velocity_cell_bubble'], beta, rtol=0, atol=256*np.finfo(float).eps)
        np.testing.assert_allclose(exported['pressure_vertex'], pressure, rtol=0, atol=256*np.finfo(float).eps)
        sample_bary = np.array([.2, .3, .5])
        physical = np.einsum('v,cva->ca', sample_bary, coordinates[cells])
        local_cells = np.argsort(prepared.original_cells).astype(np.int32)
        evaluated = vf.eval(np.column_stack((physical, np.zeros(len(cells)))), local_cells)
        expected = np.einsum('v,cva->ca', sample_bary, vertex[cells])+27*np.prod(sample_bary)*beta
        np.testing.assert_allclose(evaluated, expected, rtol=256*np.finfo(float).eps,
                                   atol=256*np.finfo(float).eps)
        for component in range(2):
            indicator = cylinder_indicator(prepared, component)
            values = indicator.x.array.reshape(-1, 2)
            hierarchical = values[prepared.cell_centroid_dofs]-values[prepared.vertex_velocity_dofs][cells].mean(axis=1)
            np.testing.assert_array_equal(hierarchical, np.zeros_like(hierarchical))

    def test_volume_pressure_force_and_facet_flux_without_solving(self):
        from dolfinx import fem
        from fenicsx_adapter import prepare, boundary_flux_forms
        from reference_system import build_system, observations
        data = fixture(); prepared = prepare(data)
        scratch = Path.home()/'.cache/eqiora/cylinder-observer-tests'; scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            jit = {'cache_dir': directory}
            system = build_system(prepared, data, rule='duffy-3', jit_options=jit)
            nv, nc = len(prepared.coordinates), len(prepared.cells)
            # A constant physical field tests observations only, not a PDE solve.
            fields = {'velocity_vertex': np.tile([.1, 0.], (nv, 1)),
                      'velocity_cell_bubble': np.zeros((nc, 2)), 'pressure_vertex': np.full(nv, .2)}
            canonical = np.concatenate((fields['velocity_vertex'].reshape(-1),
                                        fields['velocity_cell_bubble'].reshape(-1), fields['pressure_vertex']))
            source = system.transformation.createVecRight(); source.array[:] = canonical
            nodal = system.transformation.createVecLeft(); system.transformation.mult(source, nodal)
            function = fem.Function(prepared.mixed_space); function.x.array[:] = nodal.array
            row = observations(system, function, fields, canonical, jit_options=jit)
            # For this toy rectangle's bottom-edge indicator, uniform pressure
            # gives upward p*width; x contributions on the two sides cancel.
            np.testing.assert_allclose(row['force_on_domain'], [0., .2*2.2], rtol=0, atol=256*np.finfo(float).eps)
            np.testing.assert_allclose(row['matrix_virtual_work_on_domain'], row['force_on_domain'],
                                       rtol=0, atol=256*np.finfo(float).eps)
            flux = {name: fem.assemble_scalar(fem.form(expression, jit_options=jit))
                    for name, expression in boundary_flux_forms(prepared, function.sub(0)).items()}
            self.assertAlmostEqual(flux['inlet'], -.1*.41, places=14)
            self.assertAlmostEqual(flux['outlet'], .1*.41, places=14)
            for item in (source, nodal, system.matrix, system.rhs, system.transformation, system.raw_nodal_matrix): item.destroy()

    def test_custom_quadrature_assembly_and_rigid_rotation(self):
        from reference_system import build_system
        from fenicsx_adapter import prepare
        data = fixture(); prepared = prepare(data)
        scratch = Path.home()/'.cache/eqiora/cylinder-basis-tests'; scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            system = build_system(prepared, data, rule='duffy-3', jit_options={'cache_dir': directory})
            x, y = prepared.coordinates.T
            canonical = np.concatenate((np.column_stack((-y, x)).reshape(-1),
                                        np.zeros(2*len(prepared.cells)+len(prepared.coordinates))))
            source = system.transformation.createVecRight(); source.array[:] = canonical
            nodal = system.transformation.createVecLeft(); system.transformation.mult(source, nodal)
            action = nodal.duplicate(); system.raw_nodal_matrix.mult(nodal, action)
            # Rigid rotation has epsilon=0. A vector-Laplacian replacement has
            # positive energy and fails this independently derived local form check.
            allowance = 2048*np.finfo(float).eps*.001*np.sum(canonical**2)
            self.assertLessEqual(abs(nodal.dot(action)), allowance)
            for item in (source, nodal, action, system.matrix, system.rhs,
                         system.transformation, system.raw_nodal_matrix): item.destroy()


if __name__ == '__main__':
    unittest.main()
