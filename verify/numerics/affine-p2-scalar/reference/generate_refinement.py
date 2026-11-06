"""Generate all six independent reference levels; stop on first inconclusive level."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from datetime import datetime,timezone
import time
from refinement import certify


def encode(x):
    if isinstance(x,Fraction):return str(x)
    raise TypeError(type(x).__name__)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('output must not exist')
    if len(os.sched_getaffinity(0))!=1:p.error('pin one CPU before generation')
    for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(key)!='1':p.error(f'{key} must equal one')
    args.output.mkdir(parents=True)
    import scipy,numpy
    receipt={'schema':1,'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('exact.py','refinement.py','generate_refinement.py')},
             'python':platform.python_version(),'scipy':scipy.__version__,'numpy':numpy.__version__,
             'affinity':sorted(os.sched_getaffinity(0)),'thread_controls':{k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS')},
             'started_at':datetime.now(timezone.utc).isoformat(),'levels':[]}
    path=args.output/'receipt.json'
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    for d in (2,3):
        for n in (2,4,8):
            started=time.monotonic()
            result=certify(d,n)
            target=args.output/f'd{d}-n{n}.json'
            target.write_text(json.dumps(result,default=encode,indent=2,sort_keys=True)+'\n')
            receipt['levels'].append({'dimension':d,'N':n,'file':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'elapsed_seconds':time.monotonic()-started})
            path.write_text(json.dumps(receipt,indent=2)+'\n')
            print(json.dumps(receipt['levels'][-1]),flush=True)
    receipt['finished_at']=datetime.now(timezone.utc).isoformat();receipt['status']='certified'
    path.write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
