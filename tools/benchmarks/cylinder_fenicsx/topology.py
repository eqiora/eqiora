"""Pure topology checks and basis conversion for same-mesh MINI comparison."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_measurement import ComparisonError
import numpy as np

BOUNDARY_TAGS = {'inlet': 1, 'outlet': 2, 'walls': 3, 'cylinder': 4}


def mesh_arrays(data):
    coordinates, raw_cells = np.asarray(data['coordinates']), np.asarray(data['cells'])
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        raise ComparisonError('requires finite two-dimensional mesh coordinates')
    if raw_cells.ndim != 2 or raw_cells.shape[1] != 3 or not np.issubdtype(raw_cells.dtype, np.integer):
        raise ComparisonError('requires integer affine triangle connectivity')
    cells = raw_cells.astype(np.int64)
    if not len(cells) or np.any(cells < 0) or np.any(cells >= len(coordinates)):
        raise ComparisonError('triangle connectivity is empty or out of range')
    if any(len(set(cell)) != 3 for cell in cells):
        raise ComparisonError('triangle repeats a vertex')
    edges = {}
    for cell in cells:
        p = coordinates[cell]
        twice_area = np.linalg.det(np.stack((p[1]-p[0], p[2]-p[0])))
        if twice_area == 0:
            raise ComparisonError('degenerate affine triangle')
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(cell[i]), int(cell[j]))))
            edges[edge] = edges.get(edge, 0)+1
    if any(count > 2 for count in edges.values()):
        raise ComparisonError('nonmanifold triangle edge')
    if set(data['boundaries']) != set(BOUNDARY_TAGS):
        raise ComparisonError('named boundary partition differs from the cylinder example')
    labelled = {}
    for name, rows in data['boundaries'].items():
        if not rows:
            raise ComparisonError('named boundary is empty')
        for row in rows:
            if len(row) != 2 or any(not isinstance(v, (int, np.integer)) for v in row):
                raise ComparisonError('boundary entity must be an integer vertex pair')
            edge = tuple(sorted(map(int, row)))
            if edge in labelled or edges.get(edge) != 1:
                raise ComparisonError('boundary edge overlaps another label or is not exterior')
            labelled[edge] = BOUNDARY_TAGS[name]
    if set(labelled) != {edge for edge, count in edges.items() if count == 1}:
        raise ComparisonError('boundary labels do not cover the complete mesh exterior')
    return coordinates.astype(np.float64), cells, labelled


def check_permutation(actual, count, name):
    actual = np.asarray(actual)
    if actual.shape != (count,) or not np.issubdtype(actual.dtype, np.integer) or set(map(int, actual)) != set(range(count)):
        raise ComparisonError(f'{name} is not a complete input permutation')
    return actual.astype(np.int64)


def boundary_facets(facet_vertices, input_vertex_ids, labelled_edges):
    """Map actual local facets by integer input identity, never point proximity."""
    input_vertex_ids = np.asarray(input_vertex_ids)
    indices, tags = [], []
    seen = set()
    for facet, vertices in enumerate(facet_vertices):
        edge = tuple(sorted(int(input_vertex_ids[v]) for v in vertices))
        if edge in labelled_edges:
            if edge in seen:
                raise ComparisonError('input boundary edge appears as multiple local facets')
            indices.append(facet); tags.append(labelled_edges[edge]); seen.add(edge)
    if seen != set(labelled_edges):
        raise ComparisonError('external mesh did not retain every named boundary edge')
    return np.asarray(indices, dtype=np.int32), np.asarray(tags, dtype=np.int32)


def hierarchical_bubbles(vertex_values, centroid_values, cells):
    """Convert nodal enriched-P1 centroid values to the 27 lambda1 lambda2 lambda3 coefficient."""
    vertex_values, centroid_values, cells = np.asarray(vertex_values), np.asarray(centroid_values), np.asarray(cells)
    if vertex_values.ndim != 2 or vertex_values.shape[1] != 2 or centroid_values.shape != (len(cells), 2):
        raise ComparisonError('MINI field inventories do not match the two-component mesh')
    return centroid_values-vertex_values[cells].mean(axis=1)


def cylinder_geometry_report(data):
    """Recover oriented outer/hole loops from positive cells and audit named topology."""
    coordinates, cells, _ = mesh_arrays(data)
    vertices = coordinates[cells]
    twice = ((vertices[:, 1, 0]-vertices[:, 0, 0])*(vertices[:, 2, 1]-vertices[:, 0, 1])
             -(vertices[:, 1, 1]-vertices[:, 0, 1])*(vertices[:, 2, 0]-vertices[:, 0, 0]))
    if np.any(twice <= 0):
        raise ComparisonError('cylinder comparison requires positively oriented affine cells')
    incidence, oriented = {}, {}
    for cell in cells:
        for a, b in ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0])):
            edge = tuple(sorted((int(a), int(b))))
            incidence[edge] = incidence.get(edge, 0)+1; oriented[edge] = (int(a), int(b))
    successor = {}
    for edge, count in incidence.items():
        if count == 1:
            a, b = oriented[edge]
            if a in successor:
                raise ComparisonError('exterior is not a disjoint union of oriented simple loops')
            successor[a] = b
    remaining, loops = set(successor), []
    while remaining:
        start = min(remaining); current = start; loop = []
        while current in remaining:
            remaining.remove(current); loop.append(current); current = successor[current]
        if current != start:
            raise ComparisonError('exterior boundary loop is not closed')
        points = coordinates[loop]; following = np.roll(points, -1, axis=0)
        area = np.sum(points[:, 0]*following[:, 1]-following[:, 0]*points[:, 1])/2
        loops.append((loop, float(area)))
    cylinder = set(np.asarray(data['boundaries']['cylinder']).reshape(-1))
    holes = [(loop, area) for loop, area in loops if area < 0]
    outer = [(loop, area) for loop, area in loops if area > 0]
    if len(loops) != 2 or len(holes) != 1 or len(outer) != 1 or set(holes[0][0]) != cylinder:
        raise ComparisonError('cylinder tag does not identify the unique clockwise hole loop')
    triangle_area = float(twice.sum()/2)
    boundary_area = sum(area for _, area in loops)
    # Area identity follows from cancellation of interior oriented edges.
    # The allowance covers only summation/products of these finite coordinates.
    allowance = 256*np.finfo(float).eps*len(cells)*max(1., float(np.max(np.abs(coordinates)))**2)
    if abs(triangle_area-boundary_area) > allowance:
        raise ComparisonError('cell and oriented-boundary fluid areas disagree')
    return {'triangle_fluid_area': triangle_area, 'oriented_boundary_fluid_area': boundary_area,
            'outer_signed_area': outer[0][1], 'hole_signed_area': holes[0][1],
            'minimum_cell_area': float(twice.min()/2), 'area_arithmetic_allowance': allowance,
            'hole_orientation': 'clockwise into fluid exterior', 'positive_cells': True}
