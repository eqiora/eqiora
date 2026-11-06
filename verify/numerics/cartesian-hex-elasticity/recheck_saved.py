#!/usr/bin/env python3
"""Recheck committed original Hex evidence; no Eqiora import or PDE solve.

Checks scientific bytes, explicit metadata projections, saved associations and mathematics.
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


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


def leaves(value,path=''):
    if isinstance(value,dict) and value:
        return {k:v for name,item in value.items() for k,v in leaves(item,path+'/'+name.replace('~','~0').replace('/','~1')).items()}
    if isinstance(value,list) and value:
        return {k:v for index,item in enumerate(value) for k,v in leaves(item,path+'/'+str(index)).items()}
    return {path:value}


def checked_projection(root,relative,record):
    protected = {'model.eqi','model.json','mesh.json','plan.json','result.json','original-result.json','primary.json','replay.json'}
    require(Path(relative).name not in protected and not relative.startswith('archived-sources/'), 'scientific/source bytes cannot be projected')
    checked_file(root,relative,record['public_sha256'])
    if record['format'] == 'json-leaf-projection':
        public = leaves(read(root/relative))
        expected = record['original_leaf_sha256']
        changed = record['changed_string_fields']
        require(set(public) == set(expected) and set(changed) <= set(expected), 'projection changed JSON field inventory')
        for pointer,value in public.items():
            if pointer in changed:
                mapping = changed[pointer]
                require(mapping['original_value_sha256'] == expected[pointer], 'projection original value binding differs')
                require(type(value) is str and '<ARCHIVED-HOST>/' in value and value == mapping['public_value'], 'projection invented a non-path replacement')
            else:
                require(sha256(canonical(value)) == expected[pointer], 'projection changed an unredacted value')
    elif record['format'] == 'text-line-projection':
        public = (root/relative).read_bytes().decode().splitlines(keepends=True)
        expected = record['original_line_sha256']
        changed = record['changed_lines']
        require(len(public) == len(expected) and set(changed) <= {str(i) for i in range(len(expected))}, 'projection changed log line inventory')
        for index,value in enumerate(public):
            if str(index) in changed:
                mapping = changed[str(index)]
                require(mapping['original_value_sha256'] == expected[index], 'log original value binding differs')
                require('<ARCHIVED-HOST>/' in value and value == mapping['public_value'], 'log replacement differs')
            else:
                require(sha256(value.encode()) == expected[index], 'projection changed an unredacted log line')
    else:
        raise ValueError('unknown metadata projection format')


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
    projections = inventory['metadata_projections']
    require(inventory['projected_metadata_file_count'] == len(projections), 'metadata projection count differs')
    require(set(inventory['original_files_sha256']) == set(inventory['files_sha256']), 'original/public file inventory differs')
    require(set(inventory['byte_exact_original_files']) == set(inventory['files_sha256'])-set(projections), 'byte-exact/projection scope differs')
    for relative,expected in inventory['files_sha256'].items():
        checked_file(root,relative,expected)
        if relative in projections:
            record = projections[relative]
            require(record['public_sha256'] == expected and record['original_sha256'] == inventory['original_files_sha256'][relative], 'original/public projection digest binding differs')
            checked_projection(root,relative,record)
        else:
            require(expected == inventory['original_files_sha256'][relative], 'byte-exact original digest differs')
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
    print('saved-hex: scientific originals, metadata projections, physical fields and independent primary/replay checks PASS; initial PDEs 2, recovery/recheck new PDEs 0')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, TypeError, StopIteration) as error:
        print(f'saved-hex: {type(error).__name__}: {error}',file=sys.stderr)
        sys.exit(2)
