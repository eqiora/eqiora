"""Independent FEniCSx MINI preparation on the ordinary exported affine mesh."""
from dataclasses import dataclass
import basix
import dolfinx
import basix.ufl
from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl
from topology import (BOUNDARY_TAGS, boundary_facets, check_permutation,
                      hierarchical_bubbles, mesh_arrays)
from shared_measurement import ComparisonError


@dataclass
class Prepared:
    domain: object
    mixed_space: object
    velocity_space: object
    pressure_space: object
    velocity_to_mixed: np.ndarray
    pressure_to_mixed: np.ndarray
    vertex_velocity_dofs: np.ndarray
    cell_centroid_dofs: np.ndarray
    vertex_pressure_dofs: np.ndarray
    original_cells: np.ndarray
    facets: object
    bcs: list
    coordinates: np.ndarray
    cells: np.ndarray
    cylinder_vertices: np.ndarray


def prepare(data):
    if basix.__version__ != '0.11.0' or dolfinx.__version__ != '0.11.0' or MPI.COMM_WORLD.size != 1 or PETSc.ScalarType != np.float64:
        raise ComparisonError('requires locked Basix0.11, one rank and real float64 PETSc')
    coordinates, cells, labelled = mesh_arrays(data)
    coordinate_element = basix.ufl.element('Lagrange', 'triangle', 1, shape=(2,))
    domain = mesh.create_mesh(MPI.COMM_SELF, cells, coordinate_element, coordinates)
    nv, nc = len(coordinates), len(cells)
    geometry_inputs = check_permutation(domain.geometry.input_global_indices, nv, 'Geometry input vertices')
    if not np.array_equal(domain.geometry.x[:, :2], coordinates[geometry_inputs]):
        raise ComparisonError('external Geometry changed authoritative input coordinates')
    original_cells = check_permutation(domain.topology.original_cell_index, nc, 'input cells')
    domain.topology.create_connectivity(0, 2)
    geometry_vertices = mesh.entities_to_geometry(domain, 0, np.arange(nv, dtype=np.int32), permute=False)
    if geometry_vertices.shape != (nv, 1):
        raise ComparisonError('triangle topology vertex does not bind one Geometry vertex')
    input_vertices = check_permutation(geometry_inputs[geometry_vertices[:, 0]], nv, 'topology input vertices')
    domain.topology.create_connectivity(2, 0)
    cell_vertices = domain.topology.connectivity(2, 0)
    for cell in range(nc):
        if set(input_vertices[cell_vertices.links(cell)]) != set(cells[original_cells[cell]]):
            raise ComparisonError('external cell identity/connectivity differs from input')
    domain.topology.create_entities(1)
    domain.topology.create_connectivity(1, 0)
    domain.topology.create_connectivity(1, 2)
    edges = domain.topology.connectivity(1, 0)
    all_edges = [edges.links(i) for i in range(domain.topology.index_map(1).size_local)]
    indices, tags = boundary_facets(all_edges, input_vertices, labelled)
    if set(indices) != set(mesh.exterior_facet_indices(domain.topology)):
        raise ComparisonError('named tags differ from the complete exterior facet inventory')
    facet_tags = mesh.meshtags(domain, 1, indices, tags)
    p1 = basix.ufl.element('Lagrange', 'triangle', 1)
    scalar_mini = basix.ufl.enriched_element([p1, basix.ufl.element('Bubble', 'triangle', 3)])
    velocity_element = basix.ufl.blocked_element(scalar_mini, shape=(2,))
    mixed = fem.functionspace(domain, basix.ufl.mixed_element([velocity_element, p1]))
    velocity, velocity_to_mixed = mixed.sub(0).collapse()
    pressure, pressure_to_mixed = mixed.sub(1).collapse()
    # Installed DOLFINx0.11 exposes each collapse map as a one-entry list of
    # scalar-index arrays. Normalize that public map, preserving its order.
    velocity_to_mixed = np.asarray(velocity_to_mixed, dtype=PETSc.IntType).reshape(-1)
    pressure_to_mixed = np.asarray(pressure_to_mixed, dtype=PETSc.IntType).reshape(-1)
    if velocity_to_mixed.shape != (2*(nv+nc),) or pressure_to_mixed.shape != (nv,):
        raise ComparisonError('collapsed scalar-index maps do not cover MINI/P1')
    if (velocity.dofmap.index_map_bs != 2 or velocity.dofmap.index_map.size_global != nv+nc
            or pressure.dofmap.index_map_bs != 1 or pressure.dofmap.index_map.size_global != nv
            or mixed.dofmap.index_map.size_global*mixed.dofmap.index_map_bs != 3*nv+2*nc):
        raise ComparisonError('MINI/P1 full unknown inventory differs from 3Nv+2Nc')
    vdofs, pdofs, cdofs = np.empty(nv, dtype=np.int32), np.empty(nv, dtype=np.int32), np.empty(nc, dtype=np.int32)
    for vertex in range(nv):
        entity = np.array([vertex], dtype=np.int32)
        v = fem.locate_dofs_topological(velocity, 0, entity)
        p = fem.locate_dofs_topological(pressure, 0, entity)
        if len(v) != 1 or len(p) != 1:
            raise ComparisonError('MINI/P1 vertex does not identify one blocked velocity and one pressure DOF')
        vdofs[input_vertices[vertex]], pdofs[input_vertices[vertex]] = v[0], p[0]
    domain.topology.create_connectivity(2, 2)
    for cell in range(nc):
        interior = fem.locate_dofs_topological(velocity, 2, np.array([cell], dtype=np.int32))
        # locate_dofs_topological selects closure DOFs for full-dimensional entities.
        vertex_dofs = {vdofs[input_vertices[v]] for v in cell_vertices.links(cell)}
        bubble = [i for i in interior if i not in vertex_dofs]
        if len(bubble) != 1:
            raise ComparisonError('MINI cell does not identify one interior centroid DOF')
        cdofs[original_cells[cell]] = bubble[0]
    if set(vdofs) & set(cdofs) or len(set(vdofs) | set(cdofs)) != nv+nc:
        raise ComparisonError('MINI vertex/interior DOF map is not bijective')
    inlet = fem.Function(velocity)
    inlet.interpolate(lambda x: np.vstack((4*.3*x[1]*(.41-x[1])/.41**2, np.zeros(x.shape[1]))))
    zero = fem.Function(velocity)
    bcs = []
    for name in ('inlet', 'walls', 'cylinder'):
        pair = fem.locate_dofs_topological((mixed.sub(0), velocity), 1, facet_tags.find(BOUNDARY_TAGS[name]))
        bcs.append(fem.dirichletbc(inlet if name == 'inlet' else zero, pair, mixed.sub(0)))
    cylinder_vertices = np.unique(np.asarray(data['boundaries']['cylinder']).reshape(-1))
    return Prepared(domain, mixed, velocity, pressure, velocity_to_mixed, pressure_to_mixed,
                    vdofs, cdofs, pdofs, original_cells, facet_tags, bcs,
                    coordinates, cells, cylinder_vertices)


def export_fields(prepared, mixed_function):
    """Export original-order hierarchical coefficients, not raw Basix enrichment DOFs."""
    velocity = mixed_function.sub(0).collapse()
    pressure = mixed_function.sub(1).collapse()
    nodal = velocity.x.array.reshape(-1, 2)
    vertices = nodal[prepared.vertex_velocity_dofs].copy()
    centroids = nodal[prepared.cell_centroid_dofs]
    return {'velocity_vertex': vertices,
            'velocity_cell_bubble': hierarchical_bubbles(vertices, centroids, prepared.cells),
            'pressure_vertex': pressure.x.array[prepared.vertex_pressure_dofs].copy()}


def cylinder_indicator(prepared, component):
    """P1 virtual-velocity lift, with zero hierarchical bubble, in Basix's nodal basis."""
    function = fem.Function(prepared.velocity_space)
    values = function.x.array.reshape(-1, 2)
    vertices = np.zeros((len(prepared.coordinates), 2))
    vertices[prepared.cylinder_vertices, component] = 1.
    values[prepared.vertex_velocity_dofs] = vertices
    values[prepared.cell_centroid_dofs] = vertices[prepared.cells].mean(axis=1)
    function.x.scatter_forward()
    return function


def weak_forms(prepared, *, quadrature_metadata):
    """Reviewed symmetric-gradient/symmetric-pressure-coupling weak form; no gauge."""
    velocity, pressure = ufl.TrialFunctions(prepared.mixed_space)
    test_velocity, test_pressure = ufl.TestFunctions(prepared.mixed_space)
    dx = ufl.Measure('dx', domain=prepared.domain, metadata=quadrature_metadata)
    stiffness = (2*.001*ufl.inner(ufl.sym(ufl.grad(velocity)), ufl.sym(ufl.grad(test_velocity)))
                 -pressure*ufl.div(test_velocity)-test_pressure*ufl.div(velocity))*dx
    load = ufl.inner(ufl.as_vector((0., 0.)), test_velocity)*dx
    return stiffness, load


def cylinder_virtual_work(prepared, velocity, pressure, *, quadrature_metadata):
    """Force on fluid from UFL volume virtual work, never raw boundary stress."""
    dx = ufl.Measure('dx', domain=prepared.domain, metadata=quadrature_metadata)
    result = []
    for component in range(2):
        indicator = cylinder_indicator(prepared, component)
        result.append((2*.001*ufl.inner(ufl.sym(ufl.grad(velocity)), ufl.sym(ufl.grad(indicator)))
                       -pressure*ufl.div(indicator))*dx)
    return result


def boundary_flux_forms(prepared, velocity):
    normal = ufl.FacetNormal(prepared.domain)
    measure = ufl.Measure('ds', domain=prepared.domain, subdomain_data=prepared.facets,
                          metadata={'quadrature_degree': 3, 'quadrature_rule': 'gauss_jacobi'})
    return {name: ufl.dot(velocity, normal)*measure(BOUNDARY_TAGS[name]) for name in ('inlet', 'outlet')}
