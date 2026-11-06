"""Pure public-protocol input/export falsifiers; no Eqiora import or solve."""
from collections import Counter
from fractions import Fraction
from itertools import combinations
import json
from types import SimpleNamespace as Object
import unittest

import simplex_production as p


def fixture(dimension=3,n=2,profile='patch'):
    vertices,cells,edges,nodes,_=p.inputs.reference.mesh(dimension,n)
    coordinates=[list(map(float,x)) for x in vertices]
    closures={(0,i):(i,) for i in range(len(vertices))}
    # Deliberately choose a different edge ordinal ordering than the independent
    # reference. Public topology, not a guessed lattice stratum, owns ordinals.
    closures.update({(1,i):edge for i,edge in enumerate(reversed(edges))})
    incidence=Counter(tuple(sorted(f)) for c in cells for f in combinations(c,dimension))
    if dimension==3:closures.update({(2,i):face for i,face in enumerate(sorted(incidence))})
    closures.update({(dimension,i):tuple(c) for i,c in enumerate(cells)})
    b,B=p.inputs.reference.domain(dimension);inv=p.inputs.reference.inverse(B)
    logical=[tuple(sum(inv[i][j]*(x[j]-b[j]) for j in range(dimension)) for i in range(dimension)) for x in vertices]
    selected={'body':tuple((dimension,i) for i in range(len(cells)))}
    for axis in range(dimension):
        for value,label in ((0,'lower'),(1,'upper')):
            selected[f'q{axis}_{label}']=tuple(e for e,vs in closures.items() if e[0]==dimension-1 and incidence[tuple(sorted(vs))]==1 and all(logical[i][axis]==value for i in vs))
    mesh=Object(dimension=dimension,coordinates=coordinates,cells=list(map(list,cells)),source_digest='source',digest='mesh',
                production_lineage_bytes=json.dumps({'provider':{'identity':'eqiora.external-msh','version':'4.1'},'effective_policy':{'coordinate_unit':'m'}}).encode(),
                entity_vertices=lambda e:closures[e],selection_entities=lambda name:selected[name],to_bytes=lambda:b'mesh')
    entities=tuple((0,i) for i in range(len(vertices)))+tuple((1,i) for i in range(len(edges)))
    points=3 if profile=='patch' or dimension==2 else 5
    wire={'schema':'eqiora.resolved-common-plan/v28','spatial':{'kind':'uniform','policy':'p2'},'assembly_quadrature':{'kind':'simplex-duffy-gauss-legendre','spatial_dimension':dimension,'points_per_axis':points}}
    plan=Object(fields=('u',),mesh=mesh,geometry_digest='source',model_digest='model',identity='plan',field_coefficient_entities=lambda _:entities,to_bytes=lambda:json.dumps(wire,sort_keys=True).encode())
    values={'vertex':[0.25+i/100 for i in range(len(vertices))],'edge':[0.5+i/100 for i in range(len(edges))]}
    output=Object(field='u',mesh=mesh,space='continuous-lagrange-p2',value_type=Object(scalar_domain='real',shape=[]),coefficient_dimension=(0,)*7,associations=('vertex','edge'),
                  coefficient_count=lambda name:len(values[name]),logical_shape=lambda name:(len(values[name]),),values=lambda name:values[name])
    result_wire={'schema':'eqiora.common-result/v28','content':{'plan_identity':'plan'}}
    result=Object(model_digest='model',plan_key='distinct-run-request',output=lambda _:output,to_bytes=lambda:json.dumps(result_wire,sort_keys=True).encode())
    return Object(d=dimension,n=n,profile=profile,plan=plan,result=result,wire=wire,result_wire=result_wire,output=output,values=values,closures=closures,selected=selected,nodes=nodes)


def export(f):return p.extract_candidate(f.plan,f.result,'u',f.d,f.n,f.profile)


class SimplexProductionTests(unittest.TestCase):
    def test_complete_vertex_edge_export_both_dimensions_and_profiles(self):
        for d in (2,3):
            for profile in ('patch','refinement'):
                f=fixture(d,2,profile);candidate,identity=export(f)
                self.assertEqual(len(candidate['records']),25 if d==2 else 125)
                self.assertEqual({tuple(map(Fraction,r['coordinates_m'])) for r in candidate['records']},set(f.nodes))
                self.assertEqual(identity['run_request_key'],'distinct-run-request')
                self.assertEqual(identity['shared_output_source_sha256'],p.sha256((p.HERE.parent/'quadratic_output.py').read_bytes()))
    def test_unexpected_unknown_field_rejects(self):
        f=fixture();f.plan.fields=('u','unexpected')
        with self.assertRaises(p.ProductionError):export(f)
    def test_missing_edge_and_vertex_only_blocks_reject(self):
        f=fixture();f.values['edge'].pop()
        with self.assertRaises(p.ProductionError):export(f)
        f=fixture();f.output.associations=('vertex',)
        with self.assertRaises(p.ProductionError):export(f)
    def test_count_preserving_wrong_entity_rejects(self):
        f=fixture();entities=f.plan.field_coefficient_entities('u')
        f.plan.field_coefficient_entities=lambda _:entities[:-1]+((1,10000),)
        with self.assertRaises(p.ProductionError):export(f)
    def test_wrong_actual_edge_closure_and_duplicate_edge_reject(self):
        for closure in ((0,0),(0,10000)):
            f=fixture();f.closures[(1,0)]=closure
            with self.assertRaises(p.ProductionError):export(f)
        f=fixture();f.closures[(1,0)]=f.closures[(1,1)]
        with self.assertRaises(p.ProductionError):export(f)
    def test_wrong_plan_model_and_result_identity_reject(self):
        f=fixture();f.plan.geometry_digest='foreign'
        with self.assertRaises(p.ProductionError):export(f)
        f=fixture();f.result.model_digest='foreign'
        with self.assertRaises(p.ProductionError):export(f)
        f=fixture();f.result_wire['content']['plan_identity']='run-request-or-other-plan'
        with self.assertRaises(p.ProductionError):export(f)
    def test_units_scalar_shape_and_associations_reject(self):
        f=fixture();f.output.coefficient_dimension=(0,1,0,0,0,0,0)
        with self.assertRaises(p.ProductionError):export(f)
        f=fixture();f.output.value_type.scalar_domain='complex'
        with self.assertRaises(p.ProductionError):export(f)
        f=fixture();f.output.value_type.shape=[2]
        with self.assertRaises(p.ProductionError):export(f)
        f=fixture();f.output.associations=('vertex','cell-bubble')
        with self.assertRaises(p.ProductionError):export(f)
    def test_actual_quadrature_and_policy_reject(self):
        for change in ({'points_per_axis':2},{'kind':'gauss-legendre'},{'spatial_dimension':2}):
            f=fixture();f.wire['assembly_quadrature'].update(change)
            with self.assertRaises(p.ProductionError):export(f)
        f=fixture(profile='refinement');f.wire['assembly_quadrature']['points_per_axis']=3
        with self.assertRaises(p.ProductionError):export(f)
        f=fixture();f.wire['spatial']['policy']='q2'
        with self.assertRaises(p.ProductionError):export(f)
    def test_original_orientation_and_named_face_reject(self):
        f=fixture();f.plan.mesh.cells[0][-1],f.plan.mesh.cells[0][-2]=f.plan.mesh.cells[0][-2],f.plan.mesh.cells[0][-1]
        with self.assertRaises(p.ProductionError):export(f)
        f=fixture();f.selected['q0_upper']=f.selected['q0_lower']
        with self.assertRaises(p.ProductionError):export(f)

if __name__=='__main__':unittest.main()
