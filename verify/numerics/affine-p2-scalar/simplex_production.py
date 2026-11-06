"""Authenticate ordinary P2 simplex inputs and full scalar coefficient exports.

No Eqiora import, solve, manufactured-function evaluation, or numerical budget
lives here. The caller binds the frozen reference and actual producer artifact.
"""
from collections import Counter
from fractions import Fraction
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from quadratic_output import (ProductionError, require, sha256,
                              extract_coefficients, artifact_identity)

_spec = importlib.util.spec_from_file_location('_affine_p2_input_owner', HERE/'fixtures.py')
require(_spec is not None and _spec.loader is not None, 'missing input owner')
inputs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inputs)


def authenticate_plan(plan, dimension, n, profile):
    require(dimension in (2,3) and n in (2,4,8), 'undeclared simplex dimension or level')
    require(profile in ('patch','refinement') and (profile!='patch' or n==2), 'undeclared profile or patch level')
    points = 3 if profile=='patch' or dimension==2 else 5
    raw = plan.to_bytes()
    wire = json.loads(raw)
    require(wire.get('schema')=='eqiora.resolved-common-plan/v28', 'unexpected Plan schema')
    require(wire.get('spatial')=={'kind':'uniform','policy':'p2'}, 'actual Plan is not uniform P2')
    require(wire.get('assembly_quadrature')=={'kind':'simplex-duffy-gauss-legendre','spatial_dimension':dimension,'points_per_axis':points}, 'actual effective simplex quadrature differs from frozen protocol')
    mesh = plan.mesh
    require(mesh is not None and mesh.dimension==dimension, 'missing or dimension-wrong actual Mesh')
    require(plan.geometry_digest==mesh.source_digest, 'Plan geometry differs from imported Mesh source')
    lineage = json.loads(mesh.production_lineage_bytes)
    require(lineage.get('provider')=={'identity':'eqiora.external-msh','version':'4.1'}, 'Mesh is not the declared external MSH input')
    require(lineage.get('effective_policy',{}).get('coordinate_unit')=='m', 'actual Mesh lacks explicit metre input policy')
    vertices,cells,edges,nodes,_ = inputs.reference.mesh(dimension,n)
    coordinates = tuple(tuple(float(x) for x in row) for row in mesh.coordinates)
    require(coordinates==tuple(tuple(map(float,row)) for row in vertices), 'actual ordered physical vertices differ from frozen skew input')
    actual_cells = tuple(tuple(int(x) for x in row) for row in mesh.cells)
    require(actual_cells==tuple(cells), 'actual original oriented simplex inventory differs')
    selected_body = tuple(mesh.selection_entities('body'))
    require(len(selected_body)==len(cells) and set(selected_body)=={(dimension,i) for i in range(len(cells))}, 'body selection loses or adds original cells')
    _authenticate_boundaries(mesh,dimension,vertices,cells)
    expected = [('vertex',tuple((0,i) for i in range(len(vertices)))),
                ('edge',tuple((1,i) for i in range(len(edges))))]
    # Authenticate the actual edge ordinal-to-endpoint map, without assuming the
    # public Mesh assigns edge ordinals in the independent reference's order.
    actual_edges = {}
    for entity in expected[1][1]:
        closure = tuple(mesh.entity_vertices(entity))
        require(len(closure)==2 and len(set(closure))==2, 'actual edge has invalid endpoint closure')
        actual_edges[entity] = tuple(sorted(closure))
    require(len(set(actual_edges.values()))==len(edges) and set(actual_edges.values())==set(edges), 'actual global edge inventory loses, duplicates or substitutes an edge')
    return raw,mesh,coordinates,expected,actual_edges,nodes,points


def _authenticate_boundaries(mesh,dimension,vertices,cells):
    incidence = Counter(tuple(sorted(f)) for c in cells for f in combinations(c,dimension))
    boundary = [f for f,count in incidence.items() if count==1]
    b,B = inputs.reference.domain(dimension)
    inv = inputs.reference.inverse(B)
    logical = [tuple(sum(inv[i][j]*(p[j]-b[j]) for j in range(dimension)) for i in range(dimension)) for p in vertices]
    for axis in range(dimension):
        for value,label in ((0,'lower'),(1,'upper')):
            expected = {f for f in boundary if all(logical[i][axis]==value for i in f)}
            entities = tuple(mesh.selection_entities(f'q{axis}_{label}'))
            require(len(entities)==len(set(entities)) and all(e[0]==dimension-1 for e in entities), 'named boundary has invalid entity uniqueness/dimension')
            closures = [tuple(sorted(mesh.entity_vertices(e))) for e in entities]
            require(len(closures)==len(expected) and set(closures)==expected, 'named boundary differs from frozen physical face')


def extract_candidate(plan,result,field,dimension,n,profile):
    raw,mesh,coordinates,expected,edges,nodes,points = authenticate_plan(plan,dimension,n,profile)
    require(tuple(plan.fields)==(field,), 'actual Plan inventory differs from the sole source-selected field')
    def validate_closure(entity,vertices,corners):
        if entity[0]==0:
            require(vertices==(entity[1],), 'vertex entity closure differs from its actual ordinal')
        else:
            require(entity in edges and tuple(sorted(vertices))==edges[entity], 'edge closure changed during field export')
    records,retained = extract_coefficients(plan,result,field,mesh,coordinates,expected,validate_closure)
    require(len(records)==len(nodes) and {tuple(map(Fraction,r['coordinates_m'])) for r in records}==set(nodes), 'complete P2 interpolation point inventory is not bijective')
    candidate = {'schema':1,'dimension':dimension,'N':n,'profile':profile,'gauss_points':points,'records':records}
    identity = artifact_identity(plan,result,mesh,raw,retained)
    identity['shared_output_source_sha256'] = sha256((HERE.parent/'quadratic_output.py').read_bytes())
    return candidate,identity
