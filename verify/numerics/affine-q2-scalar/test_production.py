"""Pure public-protocol topology and export falsifiers; no Eqiora import/solve."""
import copy
from fractions import Fraction
import itertools
import json
import math
from types import SimpleNamespace
import unittest

import production as p


def fixture(dimension=3, n=2):
    lengths = (2, 3, 4)[:dimension]
    coordinates = list(itertools.product(*[[float(Fraction(length*k, n)) for k in range(n+1)] for length in lengths]))
    index = {point: i for i, point in enumerate(coordinates)}
    closures = {}
    for k in range(dimension+1):
        ordinal = 0
        for varying in itertools.combinations(range(dimension), k):
            for anchor in itertools.product(*[range(n) if a in varying else range(n+1) for a in range(dimension)]):
                vertices = []
                for bits in range(2**k):
                    point = tuple(float(Fraction(lengths[a]*(anchor[a]+(((bits >> varying.index(a)) & 1) if a in varying else 0)), n)) for a in range(dimension))
                    vertices.append(index[point])
                closures[k, ordinal] = vertices
                ordinal += 1
    mesh = SimpleNamespace(dimension=dimension, coordinates=coordinates,
                           cells=[vertices for (k, _), vertices in closures.items() if k == dimension],
                           entity_vertices=lambda entity: closures[entity],
                           digest='actual-mesh-digest', to_bytes=lambda: b'actual mesh artifact')
    wire = {'schema': 'eqiora.resolved-common-plan/v28', 'spatial': {'kind': 'uniform', 'policy': 'q2'},
            'assembly_quadrature': {'kind': 'gauss-legendre', 'points_per_axis': 4}}
    entities = sorted(closures)
    plan = SimpleNamespace(identity="actual-plan", model_digest="actual-model", mesh=mesh, to_bytes=lambda: json.dumps(wire).encode(),
                           field_coefficient_entities=lambda field: tuple(entities))
    associations = ['vertex', 'edge']+(['face'] if dimension == 3 else [])+['cell']
    blocks = {association: [float(i+1) for i in range(math.comb(dimension, k)*n**k*(n+1)**(dimension-k))] for k, association in enumerate(associations)}
    output = SimpleNamespace(field="u", mesh=mesh, space='continuous-lagrange-p2',
                             value_type=SimpleNamespace(scalar_domain='real', shape=[]),
                             coefficient_dimension=(0,)*7, associations=associations,
                             coefficient_count=lambda association: len(blocks[association]),
                             logical_shape=lambda association: (len(blocks[association]),),
                             values=lambda association: blocks[association])
    result_wire = {"schema": "eqiora.common-result/v28", "content": {"plan_identity": "actual-plan"}}
    result = SimpleNamespace(plan_key="different-run-request-identity", model_digest="actual-model", output=lambda field: output, to_bytes=lambda: json.dumps(result_wire).encode())
    return plan, result, output, lengths, blocks, entities, wire


class ProductionProtocol(unittest.TestCase):
    def test_complete_quad_and_hex_nodes_preserve_every_coefficient(self):
        for d in (2, 3):
            plan, result, _, lengths, blocks, _, _ = fixture(d)
            candidate, identities = p.extract_candidate(plan, result, 'u', lengths, 2)
            self.assertEqual(len(candidate['records']), 5**d)
            self.assertEqual([r['value_binary64_hex'] for r in candidate['records']], [value.hex() for values in blocks.values() for value in values])
            self.assertEqual({tuple(r['coordinates_m']) for r in candidate['records']}, set(itertools.product(*[[str(p.Decimal.from_float(float(Fraction(length*k, 4)))) for k in range(5)] for length in lengths])))
            self.assertEqual(len(identities['coefficient_entities']), 5**d)
            self.assertEqual(identities['plan_sha256'], p.sha256(plan.to_bytes()))

    def rejected(self, mutate):
        state = fixture()
        mutate(*state)
        with self.assertRaises(p.ProductionError):
            p.extract_candidate(state[0], state[1], 'u', state[3], 2)

    def test_result_from_another_plan_rejected(self):
        self.rejected(lambda plan, result, output, lengths, blocks, entities, wire: setattr(plan, 'identity', 'different-plan'))

    def test_run_request_key_is_retained_separately_from_plan_identity(self):
        plan, result, _, lengths, _, _, _ = fixture()
        _, identity = p.extract_candidate(plan, result, 'u', lengths, 2)
        self.assertNotEqual(result.plan_key, plan.identity)
        self.assertEqual(identity['run_request_key'], result.plan_key)

    def test_result_schema_and_model_link_are_authenticated(self):
        self.rejected(lambda plan, result, *rest: setattr(result, 'to_bytes', lambda: b'{"schema":"old","content":{}}'))
        self.rejected(lambda plan, result, *rest: setattr(result, 'model_digest', 'another-model'))

    def test_wrong_selected_field_rejected(self):
        self.rejected(lambda plan, result, output, lengths, blocks, entities, wire: setattr(output, 'field', 'another-field'))

    def test_missing_cell_coefficient_rejected(self):
        self.rejected(lambda plan, result, output, lengths, blocks, entities, wire: blocks['cell'].pop())

    def test_missing_face_association_rejected(self):
        self.rejected(lambda plan, result, output, lengths, blocks, entities, wire: output.associations.remove('face'))

    def test_shared_edge_alias_rejected(self):
        self.rejected(lambda plan, result, output, lengths, blocks, entities, wire: entities.__setitem__(30, entities[29]))

    def test_wrong_bound_stratum_rejected(self):
        self.rejected(lambda plan, result, output, lengths, blocks, entities, wire: entities.__setitem__(-1, (4, 0)))

    def test_effective_rule_cannot_be_self_declared(self):
        self.rejected(lambda plan, result, output, lengths, blocks, entities, wire: wire['assembly_quadrature'].__setitem__('points_per_axis', 3))

    def test_incorrect_actual_cell_inventory_rejected(self):
        self.rejected(lambda plan, result, output, lengths, blocks, entities, wire: plan.mesh.cells.pop())

    def test_different_output_mesh_rejected(self):
        def mutate(plan, result, output, *rest):
            output.mesh = copy.copy(plan.mesh)
            output.mesh.digest = 'another-mesh'
        self.rejected(mutate)

    def test_nodal_cell_is_not_a_bubble(self):
        self.rejected(lambda plan, result, output, lengths, blocks, entities, wire: output.associations.__setitem__(-1, 'cell-bubble'))

    def test_dimension_or_nonfinite_value_rejected(self):
        self.rejected(lambda plan, result, output, lengths, blocks, entities, wire: setattr(output, 'coefficient_dimension', (1,)+(0,)*6))
        self.rejected(lambda plan, result, output, lengths, blocks, entities, wire: blocks['face'].__setitem__(0, float('nan')))


if __name__ == '__main__':
    unittest.main()
