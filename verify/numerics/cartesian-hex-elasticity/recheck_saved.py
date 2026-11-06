#!/usr/bin/env python3
"""Recheck committed original Hex evidence; no Eqiora import or PDE solve.

Checks byte integrity, saved-artifact associations and independent mathematics.
It does not rerun original processes or replace native artifact admission.
"""
import base64
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = load('hex_saved_reference', HERE/'reference/reference.py')
_previous = sys.modules.get('reference')
sys.modules['reference'] = reference
try:
    protocol = load('hex_saved_protocol', HERE/'reference/protocol.py')
finally:
    if _previous is None:
        sys.modules.pop('reference', None)
    else:
        sys.modules['reference'] = _previous


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(), object_pairs_hook=protocol.unique)


def embedded(obj, key):
    return base64.b64decode(obj[key], validate=True)


def checked_file(root, relative, expected):
    path = (root/relative).resolve()
    require(path.is_relative_to(root.resolve()), 'inventory path escapes archive')
    require(sha256(path.read_bytes()) == expected, f'changed original file: {relative}')


def bind_export(directory, document_path, result_path, preparation_path, limits):
    """Bind the independent projection to all original native field coordinates."""
    model = read(directory/'model.json')
    mesh_raw = (directory/'mesh.json').read_bytes()
    mesh = json.loads(mesh_raw)
    plan = read(directory/'plan.json')
    result = read(result_path)
    preparation = read(preparation_path)
    document = read(document_path)
    protocol.check(document_path.read_text(), limits)
    require(embedded(plan, 'model_base64') == (directory/'model.json').read_bytes(), 'Plan/Model bytes differ')
    require(embedded(plan, 'mesh_base64') == mesh_raw, 'Plan/Mesh bytes differ')
    require(plan['assembly_quadrature'] == {'kind':'gauss-legendre','points_per_axis':2}, 'effective Gauss2 differs')
    require(plan['spatial'] == {'kind':'uniform','policy':'q1'}, 'effective Q1 differs')
    require(plan['identity'] == result['content']['plan_identity'] == preparation['plan_identity'], 'Result/Plan identity differs')
    require(document['lineage'] == {'model':preparation['model_digest'],'mesh':preparation['mesh_digest'],'result':result['identity']}, 'observation lineage differs')
    owner = mesh['resources']['mesh']
    require(owner['kind'] == 'structured-cartesian', 'retained Mesh family differs')
    native_mesh = json.loads(embedded(owner, 'mesh_base64'))
    geometry = json.loads(embedded(owner, 'geometry_base64'))
    require(native_mesh['dimension'] == 3 and native_mesh['cell_family'] == 'hypercube', 'retained Hex dimension/family differs')
    require(native_mesh['vertex_order'] == 'last-axis-fastest' and native_mesh['local_node_order'] == 'tensor-product-z', 'Cartesian ordering metadata differs')
    axes = native_mesh['axes']
    require(axes == [[0.,1.,2.],[0.,1.5,3.],[0.,2.,4.]], 'actual physical axes differ')
    require(geometry['bounds'] == [[0.,2.],[0.,3.],[0.,4.]] and geometry['length_unit'] == 'metre', 'Geometry physical bounds differ')
    correspondence = json.loads(embedded(owner, 'correspondence_base64'))
    require(correspondence['mesh_sha256'] == preparation['mesh_digest'], 'Mesh correspondence identity differs')
    require(correspondence['body']['cell_indices'] == list(range(8)), 'retained eight Hex cells differ')
    fields = result['content']['payload']['fields']
    require(len(fields) == 1 and fields[0]['field_id'] == preparation['field'], 'Result Field identity differs')
    require(fields[0]['space'] == 'continuous-lagrange-p1', 'Result Field space differs')
    field = next(node['definition'] for node in model['nodes'] if node['id'] == {'kind':'field','ulid':preparation['field']})
    ty = field['value_type']
    require(ty['domain'] == 'real' and ty['shape'] == [3] and ty['frame'] == 'spatial-cartesian' and ty['array_rank'] == 0, 'Field physical vector type differs')
    require(ty['dimension'] == [[0,1],[1,1],[0,1],[0,1],[0,1],[0,1],[0,1]], 'Field metre dimension differs')
    blocks = fields[0]['blocks']
    require(len(blocks) == 1 and blocks[0]['association'] == 'vertex', 'Result coefficient association differs')
    require(blocks[0]['logical_shape'] == [len(axis) for axis in axes]+ty['shape'], 'Cartesian tensor-block shape differs')
    values = blocks[0]['values']
    require(len(values) == 81, 'Result omits a displacement coefficient')
    expected = {(tuple(float(v).hex() for v in point), component):float(values[3*n+i]).hex()
                for n,point in enumerate(itertools.product(*axes)) for i,component in enumerate('xyz')}
    actual = {(tuple(row['coordinate_m']),row['component']):row['value_m'] for row in document['coefficients']}
    require(len(actual) == 81 and actual == expected, 'export differs from actual complete physical Result block')
    observable_types = ([1,2,-2,0,0,0,0],)+([1,1,-2,0,0,0,0],)*3
    for index,(identity,dimensions) in enumerate(zip(preparation['observables'],observable_types,strict=True)):
        observable = next(node['definition'] for node in model['nodes'] if node['id'] == {'kind':'observable','ulid':identity})
        require(observable['value_type']['dimension'] == [[v,1] for v in dimensions], 'Observable J/N dimension differs')
        require(observable['reduction']['measure'] == ('volume' if index == 0 else 'boundary'), 'Observable measure differs')
    return document


def main(root=None):
    root = HERE/'evidence' if root is None else Path(root)
    inventory = read(root/'inventory.json')
    for relative,expected in inventory['files_sha256'].items():
        checked_file(root,relative,expected)
    require({str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} == set(inventory['files_sha256'])|{'inventory.json'}, 'unlisted archive file')
    for relative,expected in inventory['reference_sha256'].items():
        checked_file(HERE/'reference',relative,expected)
    campaign = inventory['initial_campaign']
    require(campaign['pde_solves'] == {'python':1,'native':1,'total':2}, 'initial solve count differs')
    require(campaign['native']['process_exit_code'] == 101 and campaign['native']['qualification'] == 'export-failed-after-accepted-result-save', 'initial native failure was relabelled')
    require(campaign['native_readonly_recovery']['new_pde_solves'] == 0, 'recovery claims a new solve')
    for relative,expected in [('python-7612/receipts/python-qualification-7612-attempt1-finish.json',0),
        ('native-682-failed-export/receipts/native-qualification-682-attempt1-finish.json',101),
        ('native-recovery-1d010/receipts/native-readonly-recovery-1d010-attempt1-finish.json',0)]:
        require(read(root/relative)['exit_code'] == expected, 'original terminal process status differs')
    limits = reference.budgets()
    python = root/'python-7612'
    require(read(python/'qualification.json')['pde_solves'] == 1, 'Python initial solve count differs')
    for lane in ('primary','replay'):
        bind_export(python,python/(lane+'.json'),python/'result.json',python/'preparation.json',limits)
    native = root/'native-682-failed-export'
    recovered = root/'native-recovery-1d010'
    require((native/'artifacts/result.json').read_bytes() == (recovered/'original-result.json').read_bytes(), 'recovery did not retain original native Result')
    require(read(recovered/'recovery-qualification.json')['new_pde_solves'] == 0, 'recovery solve count differs')
    recovery_record = read(recovered/'recovery-qualification.json')
    require(recovery_record['original_source_commit'] == campaign['native']['source_commit'] and recovery_record['recovery_harness_commit'] == campaign['native_readonly_recovery']['harness_commit'], 'original/recovery runtime attribution differs')
    require(recovery_record['original_result_sha256'] == sha256((native/'artifacts/result.json').read_bytes()), 'recovery original Result binding differs')
    require(recovery_record['result_identity'] == read(native/'artifacts/result.json')['identity'], 'recovery canonical Result identity differs')
    for name in ('model.eqi','model.json','mesh.json','plan.json'):
        require((native/'artifacts'/name).read_bytes() == (recovered/'artifacts'/name).read_bytes(), 'read-only preparation changed original inputs')
    for lane in ('primary','replay'):
        bind_export(native/'artifacts',recovered/(lane+'.json'),native/'artifacts/result.json',native/'preparation.json',limits)
    print('saved-hex: original hashes, complete physical fields, lineage and independent primary/replay checks PASS; initial PDEs 2, recovery/recheck new PDEs 0')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, TypeError, StopIteration) as error:
        print(f'saved-hex: {type(error).__name__}: {error}',file=sys.stderr)
        sys.exit(2)
