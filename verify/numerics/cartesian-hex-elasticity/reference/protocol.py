"""Strict observation projection, independent of Eqiora and its adapter."""
import json
import math
import reference as r


def keys(obj, expected):
    if type(obj) is not dict or set(obj) != set(expected):
        raise ValueError('incomplete or unknown keys')


def binary64(value):
    if type(value) is not str:
        raise ValueError('binary64 must use canonical float.hex string')
    result = float.fromhex(value)
    if not math.isfinite(result) or result.hex() != value or (result == 0 and value.startswith('-')):
        raise ValueError('noncanonical or nonfinite binary64')
    return result


def unique(pairs):
    obj = {}
    for key,value in pairs:
        if key in obj:
            raise ValueError('duplicate JSON key')
        obj[key] = value
    return obj


def decode(text):
    obj = json.loads(text,object_pairs_hook=unique,
                     parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    keys(obj, ('identity','lineage','coefficients','energy_J','left_traction_N'))
    identity = dict(dimension=3, family='Hex', order='Q1', geometry='affine-cartesian',
                    cells=[2,2,2], quadrature=[2,2,2], units=['m','J','N'])
    if obj['identity'] != identity:
        raise ValueError('wrong mathematical identity')
    keys(obj['lineage'],('mesh','model','result'))
    if any(type(v) is not str or not v.strip() for v in obj['lineage'].values()):
        raise ValueError('missing lineage identifier')
    expected = {(tuple(float(v) for v in x),c):3*n+i for n,x in enumerate(r.NODES)
                for i,c in enumerate(('x','y','z'))}
    coefficients = obj['coefficients']
    if type(coefficients) is not list or len(coefficients) != 81:
        raise ValueError('expected 81 displacement coefficients')
    u = [None]*81
    for record in coefficients:
        keys(record,('coordinate_m','component','value_m'))
        xyz = record['coordinate_m']
        if type(xyz) is not list or len(xyz) != 3 or type(record['component']) is not str:
            raise ValueError('invalid coordinate or component')
        pair = (tuple(binary64(v) for v in xyz),record['component'])
        if pair not in expected:
            raise ValueError('duplicate or unknown coordinate/component')
        u[expected.pop(pair)] = r.Q(binary64(record['value_m']))
    if expected:
        raise ValueError('missing coefficient')
    energy = r.Q(binary64(obj['energy_J']))
    traction = obj['left_traction_N']
    if type(traction) is not list or len(traction) != 3:
        raise ValueError('expected three traction integrals')
    return u,energy,[r.Q(binary64(v)) for v in traction]


def check(text):
    u,energy,traction = decode(text)
    K,b = r.assemble()
    if max(abs(a-b) for a,b in zip(u,r.U)) > r.Q(1,10**8)+r.Q(1,10**14):
        raise ValueError('displacement budget')
    if max(abs(u[i]-r.U[i]) for i in r.D) > r.Q(1,10**12)+r.Q(1,10**14):
        raise ValueError('constrained displacement budget')
    if max(map(abs,r.residual(K,b,u))) > r.Q(101,10**12):
        raise ValueError('independent residual budget')
    allowance = r.Q(1,10**6)+r.Q(1,10**10)
    if abs(energy-r.Q(261,32)) > allowance:
        raise ValueError('energy budget')
    if max(abs(traction[i]+12*r.stress()[i][0]) for i in range(3)) > allowance:
        raise ValueError('traction budget')
