"""Recompute exact patch operators/inverses and reject all numerical mutants."""
import gzip
import hashlib
import json
from pathlib import Path
from fractions import Fraction as Q
from generate_patch import generate
from patch_gate import check


def main():
    root=Path(__file__).with_name('data')
    manifest=json.loads((root/'manifest.json').read_text())
    path=root/manifest['patch']['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['files'][path.name]:raise ValueError('compressed patch digest mismatch')
    raw=gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw).hexdigest()!=manifest['patch']['uncompressed_sha256']:raise ValueError('original patch bytes digest mismatch')
    packet=json.loads(raw)
    if [p['dimension'] for p in packet['patches']]!=[2,3]:raise ValueError('incomplete dimension set')
    for p in packet['patches']:
        generated=json.loads(json.dumps(generate(p['dimension']),default=str))
        if generated!=p:raise ValueError('exact patch operator/inverse mismatch')
        values=list(map(Q,p['exact_coefficients']));actions=list(map(Q,p['exact_constrained_actions']))
        check(p,values,actions)
        mutants=[('missing-edge',values[:-1],actions),('zero-field',[Q(0)]*len(values),actions),('negated-field',[-x for x in values],actions),('negated-actions',values,[-x for x in actions])]
        for name,coefficients,reactions in mutants:
            try:check(p,coefficients,reactions)
            except ValueError:pass
            else:raise ArithmeticError(f'mutant accepted: {name}')
        for i in range(3**p['dimension'],len(values)):
            for replacement in (Q(0),-values[i]):
                bad=list(values);bad[i]=replacement
                try:check(p,bad,actions)
                except ValueError:pass
                else:raise ArithmeticError(f'missing or sign-flipped midpoint accepted: {i}')
        for i in p['free']:
            if i>=3**p['dimension'] and Q(p['natural_load'][i]):
                mutant=dict(p,natural_load=list(p['natural_load']))
                mutant['natural_load'][i]='0'
                try:check(mutant,values,actions)
                except ValueError:pass
                else:raise ArithmeticError(f'skipped natural midpoint accepted: {i}')
        print(f"PASS dimension={p['dimension']} exact-operator-inverse-and-all-midpoint-mutants",flush=True)

if __name__=='__main__':main()
