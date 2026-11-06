"""Freeze exact patch operators and independently derived fixed allocations."""
import argparse
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import platform
import time
from exact import assemble_patch, infinity_norm, inverse, mesh, patch_value


def generate(d):
    K,F,G,nodes,essential = assemble_patch(d)
    free = [i for i in range(len(nodes)) if i not in essential]
    A = [[K[i][j] for j in free] for i in free]
    inv = inverse(A)
    # Check the actual rational inverse, not its approximate condition number.
    for i,row in enumerate(A):
        for j in range(len(A)):
            if sum(row[k]*inv[k][j] for k in range(len(A))) != Q(i==j):
                raise ArithmeticError('rational inverse identity failed')
    C = infinity_norm(inv); eps = Q(1,2**40)
    Umax = max(patch_value(x) for x in nodes[:(3**d)])
    kinf = infinity_norm(K); binf = max(abs(f+g) for f,g in zip(F,G))
    rho = eps*max(1,kinf*Umax+binf)
    values = list(map(patch_value,nodes))
    residual = [sum(a*v for a,v in zip(row,values))-f-g for row,f,g in zip(K,F,G)]
    if any(residual[i] for i in free): raise ArithmeticError('patch equilibrium failed')
    return {'dimension':d,'N':2,'scales':{'length_m':'1','solution':'1','diffusion':'1'},
            'nodes_m':nodes,'vertices_m':mesh(d,n=2)[0],'cells':mesh(d,n=2)[1],
            'edges':mesh(d,n=2)[2],'cell_dofs':mesh(d,n=2)[4],'essential':essential,'free':free,'stiffness':K,
            'volume_load':F,'natural_load':G,'exact_coefficients':values,
            'exact_constrained_actions':[residual[i] for i in essential],
            'inverse_free':inv,'epsilon':eps,'Cinf':C,'coefficient_allowance':C*eps,
            'Kinf':kinf,'Binf':binf,'Umax':Umax,'rho':rho,
            'reaction_sum_allowance':len(essential)*(kinf*C*eps+rho)}


def encode(value):
    if isinstance(value,Q): return str(value)
    raise TypeError(type(value).__name__)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('output must not exist')
    started = time.time()
    packet = {'schema':1,'producer':'independent standard-library rational simplex assembly',
              'python':platform.python_version(),'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('exact.py','generate_patch.py')},
              'patches':[generate(d) for d in (2,3)]}
    args.output.write_text(json.dumps(packet,default=encode,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'elapsed_seconds':time.time()-started}))

if __name__ == '__main__': main()
