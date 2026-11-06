#!/usr/bin/env python3
"""One installed-wheel regression; initial two-run campaign stays archived."""
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))


def main():
    allowed = os.sched_getaffinity(0)
    if not allowed:
        raise RuntimeError('no allowed serial CPU')
    cpu = min(allowed)
    os.sched_setaffinity(0,{cpu})
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if os.environ.get('LD_LIBRARY_PATH') or os.environ.get('LD_PRELOAD'):
        raise RuntimeError('registered runtime must not depend on loader overrides')
    import recheck_saved
    recheck_saved.main()
    from production import extract, prepare, require, sha256, write
    import eqiora
    import eqiora._eqiora as native
    source_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE).decode().strip()
    extension = Path(native.__file__).resolve()
    runtime = {'native_module':str(extension),'native_module_sha256':sha256(extension.read_bytes()),
               'eqiora_version':importlib.metadata.version('eqiora'),'runtime_source_commit':None}
    receipt_path = os.environ.get('EQIORA_HEX_REGISTERED_RUNTIME_RECEIPT')
    if receipt_path:
        receipt_bytes = Path(receipt_path).read_bytes()
        receipt = json.loads(receipt_bytes)
        require(receipt['extension_sha256'] == runtime['native_module_sha256'], 'regression loaded extension differs from supplied build receipt')
        require(receipt['source_commit'] == source_commit, 'regression runtime belongs to another source revision')
        runtime.update(runtime_source_commit=receipt['source_commit'],build_receipt_sha256=sha256(receipt_bytes),wheel_sha256=receipt['wheel_sha256'])
    parent = Path.home()/'.cache/eqiora/cartesian-hex-elasticity-registered'
    parent.mkdir(parents=True,exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='regression-',dir=parent))
    print(f'hex-registered: retained regression output {output}',flush=True)
    state = {'ordinary_run_attempted':0,'ordinary_run_completed':0}
    write(output,'launch.json',{'kind':'registered-regression','initial_campaign_pde_solves':2,
        'this_regression_declared_pde_solves':1,'pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),
        'source_commit':source_commit,'runtime':runtime,'python':sys.executable,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'environment':{key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','LD_LIBRARY_PATH','LD_PRELOAD')},
        'producer_sha256':sha256((HERE/'production.py').read_bytes()),'registered_driver_sha256':sha256(Path(__file__).read_bytes())})
    try:
        model,plan,field,observables = prepare(eqiora,output)
        state['ordinary_run_attempted'] = 1
        result = eqiora.run(plan)
        state['ordinary_run_completed'] = 1
        result_bytes = result.to_bytes()
        write(output,'result.json',result_bytes)
        replay = eqiora.Result.from_bytes(plan,result_bytes)
        require(replay.to_bytes() == result_bytes,'registered Result canonical replay differs')
        limits = recheck_saved.reference.budgets()
        for name,lane in (('primary',result),('replay',replay)):
            document = extract(plan,model,lane,field,observables)
            write(output,name+'.json',document)
            recheck_saved.protocol.check(json.dumps(document),limits)
        write(output,'qualification.json',{**state,'kind':'registered-regression','status':'passed',
            'this_regression_pde_solves':1,'initial_campaign_pde_solves':2,'profile_calls':0,'warmups':0,
            'plan_identity':plan.identity,'result_sha256':sha256(result_bytes)})
    except Exception as error:
        write(output,'failure.json',{**state,'kind':'registered-regression','type':type(error).__name__,'message':str(error)})
        raise
    print('hex-registered: one new regression Run and independent primary/replay checks PASS; initial campaign unchanged')
    return 0


if __name__ == '__main__':
    sys.exit(main())
