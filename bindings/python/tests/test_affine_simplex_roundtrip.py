"""Public skew polygon/polyhedron and original simplex import: no PDE solve."""
from fractions import Fraction
import importlib.util
from itertools import combinations
from pathlib import Path

import eqiora
import pytest

_path=Path(__file__).resolve().parents[3]/'verify'/'numerics'/'affine-p2-scalar'/'fixtures.py'
_spec=importlib.util.spec_from_file_location('_ordinary_affine_p2_fixtures',_path)
assert _spec is not None and _spec.loader is not None
fixtures=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)


@pytest.mark.parametrize('dimension',(2,3))
def test_public_skew_geometry_original_cells_and_full_edge_inventory(dimension):
    geometry=fixtures.geometry(eqiora,dimension)
    source=fixtures.msh_bytes(dimension)
    mesh=eqiora.meshing.Mesh.import_msh(geometry,source,coordinate_unit='m')
    vertices,cells,edges,nodes,_=fixtures.reference.mesh(dimension,2)
    assert mesh.source_digest==geometry.digest
    assert (mesh.dimension,mesh.vertex_count,mesh.cell_count)==(dimension,len(vertices),len(cells))
    assert mesh.coordinates.tolist()==[list(map(float,p)) for p in vertices]
    assert mesh.cells.tolist()==list(map(list,cells))
    assert mesh.selection_entity_count('body')==len(cells)
    for axis in range(dimension):
        for side in ('lower','upper'):
            assert geometry.selection_dimension(f'q{axis}_{side}')==dimension-1
            assert mesh.selection_entity_count(f'q{axis}_{side}')==(2 if dimension==2 else 8)
    actual_cells=mesh.cells.tolist()
    actual_vertices=[tuple(map(Fraction,p)) for p in mesh.coordinates.tolist()]
    actual_edges=sorted({tuple(sorted(e)) for c in actual_cells for e in combinations(c,2)})
    assert actual_edges==edges
    assert len(actual_vertices)+len(actual_edges)==(25 if dimension==2 else 125)
    actual_nodes=actual_vertices+[tuple((actual_vertices[i][k]+actual_vertices[j][k])/2 for k in range(dimension)) for i,j in actual_edges]
    assert actual_nodes==nodes
    _,B=fixtures.reference.domain(dimension)
    for cell in actual_cells:
        J=[[actual_vertices[cell[j+1]][i]-actual_vertices[cell[0]][i] for j in range(dimension)] for i in range(dimension)]
        assert fixtures.reference.determinant(J)==fixtures.reference.determinant(B)/2**dimension
    reopened=eqiora.meshing.Mesh.from_bytes(mesh.to_bytes())
    assert reopened.to_bytes()==mesh.to_bytes()
    assert reopened.source_digest==geometry.digest
    assert reopened.coordinates.tolist()==mesh.coordinates.tolist()
    assert reopened.cells.tolist()==mesh.cells.tolist()
