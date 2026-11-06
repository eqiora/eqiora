"""Approved skew simplex inputs through public geometry and MSH ingress.

Only the frozen independent reference supplies geometry/connectivity; no product
output determines an input or expectation. This module performs no PDE solve.
"""
import importlib.util
from itertools import product
from pathlib import Path

_path = Path(__file__).with_name('reference') / 'exact.py'
_spec = importlib.util.spec_from_file_location('_affine_p2_reference_exact', _path)
assert _spec is not None and _spec.loader is not None
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def geometry(eqiora, dimension):
    b, B = reference.domain(dimension)
    graph = eqiora.geometry.GeometryGraph()
    def point(q):
        return tuple(float(b[i] + sum(B[i][j]*q[j] for j in range(dimension))) for i in range(dimension))
    if dimension == 2:
        op = graph.polygon(vertices=[point(q) for q in ((0,0),(1,0),(1,1),(0,1))], tolerance=1e-12)
        named = {'body':op.region,'q0_lower':op.boundaries[3],'q0_upper':op.boundaries[1],
                 'q1_lower':op.boundaries[0],'q1_upper':op.boundaries[2]}
    elif dimension == 3:
        corners=[point(tuple((i>>axis)&1 for axis in range(3))) for i in range(8)]
        faces=[[0,4,6,2],[1,3,7,5],[0,1,5,4],[2,6,7,3],[0,2,3,1],[4,5,7,6]]
        op=graph.convex_polyhedron(vertices=corners,faces=faces,tolerance=1e-12)
        named={'body':op.region}
        named.update({f'q{axis}_{side}':op.boundaries[2*axis+i] for axis in range(3) for i,side in enumerate(('lower','upper'))})
    else:
        raise ValueError('dimension must be two or three')
    return graph.build(op,named_topology=named)


def msh_bytes(dimension, intervals=2):
    vertices,cells,*_=reference.mesh(dimension,intervals)
    count=len(vertices);total=len(cells)
    lines=['$MeshFormat','4.1 0 8','$EndMeshFormat','$Nodes',f'1 {count} 1 {count}',f'{dimension} 1 0 {count}']
    lines.extend(str(i+1) for i in range(count))
    lines.extend(' '.join(repr(float(x)) for x in tuple(p)+(0,)*(3-dimension)) for p in vertices)
    lines.extend(['$EndNodes','$Elements',f'1 {total} 1 {total}',f'{dimension} 1 {2 if dimension==2 else 4} {total}'])
    lines.extend(' '.join(map(str,[i+1]+[v+1 for v in cell])) for i,cell in enumerate(cells))
    lines.extend(['$EndElements',''])
    return '\n'.join(lines).encode('ascii')
