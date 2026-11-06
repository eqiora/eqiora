#!/usr/bin/env python3
"""One reviewed 2D or 3D N2 patch candidate, or its zero-run preflight.

No retries, warmups, samples, private artifact construction or alternate solver
path. Only dimensions 2/3 are selectable; levels and profiles remain fixed.
"""
import argparse
import gzip
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from simplex_plan import prepare
from simplex_production import extract_candidate,require,sha256
from quadratic_output import write_new,utc

THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','NUMEXPR_NUM_THREADS')


def reference(expected,dimension):
    require(dimension in (2,3),'undeclared patch dimension')
    root=HERE/'reference';data=root/'data'
    raw=(data/'manifest.json').read_bytes()
    require(sha256(raw)==expected,'frozen reference manifest digest differs')
    manifest=json.loads(raw)
    for name,digest in manifest['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'invalid frozen reference path')
        require(sha256((data/name).read_bytes())==digest,'frozen reference file digest differs')
    raw=gzip.decompress((data/manifest['patch']['file']).read_bytes())
    require(sha256(raw)==manifest['patch']['uncompressed_sha256'],'original rational operator bytes differ')
    packet=next(p for p in json.loads(raw)['patches'] if p['dimension']==dimension)
    require(packet['N']==2,'frozen patch level differs')
    proof=json.loads((root/'checks'/'supplemental-receipt.json').read_text())
    require(proof['packet_manifest_sha256']==expected,'reference source receipt binds a different packet')
    require(sha256((root/'patch_gate.py').read_bytes())==proof['source_sha256']['patch_gate.py'],'frozen reference gate source differs')
    spec=importlib.util.spec_from_file_location('_affine_p2_frozen_patch_gate',root/'patch_gate.py')
    require(spec is not None and spec.loader is not None,'missing frozen reference gate')
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    return packet,gate.check


def compare_patch(candidate,packet,gate):
    from fractions import Fraction
    require(candidate.get('schema')==1 and candidate.get('dimension')==packet['dimension']
            and candidate.get('N')==packet['N']==2 and candidate.get('profile')=='patch'
            and candidate.get('gauss_points')==3,'candidate metadata differs from frozen patch')
    records=candidate['records']
    values={tuple(map(Fraction,r['coordinates_m'])):float.fromhex(r['value_binary64_hex']) for r in records}
    nodes=[tuple(map(Fraction,x)) for x in packet['nodes_m']]
    require(len(values)==len(records)==len(nodes) and set(values)==set(nodes),'patch coordinate/vector inventory is incomplete')
    require(all(math.isfinite(x) for x in values.values()),'nonfinite patch coefficient')
    # Reorder the complete actual field by physical interpolation coordinates.
    # No expected solution is substituted; all criteria stay in the frozen gate.
    checked=gate(packet,[values[x] for x in nodes])
    require(checked['accepted'] and not checked['constrained_actions_checked'],'unexpected optional-action witness status')
    return {**checked,'native_constrained_action_witness_present':False,
            'scope':'complete coefficients, essential trace and independent free residual; no native reaction export'}


def native_receipt_matches(receipt,source,digest):
    require(receipt['source_revision']==source and receipt['actual_terminal_exit_code']==0,
            'native build receipt does not match reviewed successful producer')
    require(receipt['installed_module']['native_sha256']==digest,'native build receipt digest differs')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('preflight','solve'),required=True)
    p.add_argument('--dimension',type=int,choices=(2,3),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-harness-revision',required=True)
    p.add_argument('--native-receipt',type=Path,required=True)
    p.add_argument('--expected-native-source',required=True)
    p.add_argument('--expected-native-sha256',required=True)
    p.add_argument('--expected-reference-sha256',required=True)
    return p


def main():
    p=parser();args=p.parse_args()
    if args.output.exists():p.error('output must not exist')
    for value in (args.expected_harness_revision,args.expected_native_source):
        if len(value)!=40 or any(c not in '0123456789abcdef' for c in value):p.error('source revisions must be full lowercase Git hashes')
    for value in (args.expected_native_sha256,args.expected_reference_sha256):
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):p.error('digests must be full lowercase SHA256')
    affinity=sorted(os.sched_getaffinity(0))
    if not 1<=len(affinity)<=4:p.error('pin a resource envelope of one to four CPUs')
    if any(os.environ.get(k)!='1' for k in THREADS):p.error('all declared thread controls must equal one')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':p.error('hide GPUs explicitly')
    if any(os.environ.get(k,'1')!='1' for k in ('OMPI_COMM_WORLD_SIZE','PMI_SIZE','WORLD_SIZE')):p.error('MPI/multiple ranks are outside this protocol')
    actual_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
    if actual_revision!=args.expected_harness_revision:p.error('actual harness revision differs')
    if subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=HERE,text=True).strip():p.error('tracked source differs from the frozen harness')
    try:
        packet,gate=reference(args.expected_reference_sha256,args.dimension)
        native_receipt_raw=args.native_receipt.read_bytes();native_receipt=json.loads(native_receipt_raw)
        native_receipt_matches(native_receipt,args.expected_native_source,args.expected_native_sha256)
    except (ValueError,KeyError,OSError) as error:p.error(str(error))
    args.output.mkdir(parents=True)
    closure=[HERE/name for name in ('check.py','simplex_plan.py','source_model.py','fixtures.py','simplex_production.py')]+[HERE.parent/'quadratic_output.py',HERE/'reference'/'exact.py',HERE/'reference'/'patch_gate.py']
    launch={'schema':1,'case':{'profile':'patch','dimension':args.dimension,'N':2},'mode':args.mode,'python_command':list(sys.orig_argv),'process_cwd':os.getcwd(),
            'actual_harness_revision':actual_revision,'source_closure_sha256':{str(path.relative_to(HERE.parent)):sha256(path.read_bytes()) for path in closure},
            'native_receipt_sha256':sha256(native_receipt_raw),'reviewed_native_source_revision':args.expected_native_source,'expected_native_sha256':args.expected_native_sha256,
            'frozen_reference_manifest_sha256':args.expected_reference_sha256,'cpu_affinity_resource_envelope':affinity,
            'thread_controls':{k:os.environ[k] for k in THREADS},'gpu_count':0,'rank_count':1,'actual_plan_workers_required':1,'started_at':utc()}
    write_new(args.output/'launch.json',launch)
    state={'ordinary_run_attempted':0,'ordinary_run_completed':0,'result_persisted':False,'profile_calls':0,'warmups':0,'timed_samples':0,'phase':'native-import'}
    code=0
    try:
        import eqiora
        import eqiora._eqiora as native
        native_path=Path(native.__file__).resolve()
        require(sha256(native_path.read_bytes())==args.expected_native_sha256,'actual imported native bytes differ from reviewed artifact')
        state.update(native_module=str(native_path),native_module_sha256=sha256(native_path.read_bytes()),eqiora_version=importlib.metadata.version('eqiora'))
        state['phase']='plan-preparation'
        plan,field,preflight=prepare(eqiora,args.dimension,2,'patch',args.output)
        state.update(plan_identity=plan.identity,actual_plan_workers=preflight['actual_plan_workers'],actual_plan_placement=preflight['actual_plan_placement'])
        if args.mode=='preflight':state['status']='preflight-passed-no-run'
        else:
            state['phase']='ordinary-run';state['ordinary_run_attempted']=1
            result=eqiora.run(plan)
            state['ordinary_run_completed']=1
            raw=result.to_bytes()
            state['result_sha256']=write_new(args.output/'result.json',raw)
            state['result_persisted']=True
            state['phase']='result-replay-and-export'
            reopened=eqiora.Result.from_bytes(plan,raw)
            require(reopened.to_bytes()==raw,'Result replay changed canonical bytes')
            candidate,identity=extract_candidate(plan,result,field,args.dimension,2,'patch')
            replay_candidate,replay_identity=extract_candidate(plan,reopened,field,args.dimension,2,'patch')
            require(replay_candidate==candidate and replay_identity==identity,'Result replay changed a vertex or edge coefficient')
            state['candidate_sha256']=write_new(args.output/'candidate.json',candidate)
            identity['candidate_sha256']=state['candidate_sha256']
            state['export_identity_sha256']=write_new(args.output/'export-identity.json',identity)
            state['phase']='frozen-reference-gate'
            comparison=compare_patch(candidate,packet,gate)
            state['comparison_sha256']=write_new(args.output/'comparison.json',comparison)
            state['status']='accepted-single-candidate'
    except Exception as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}')
        print(state['error'],file=sys.stderr);code=2
    finally:
        state.update(finished_at=utc(),launch_sha256=sha256((args.output/'launch.json').read_bytes()),actual_harness_revision=actual_revision,native_producer_revision=args.expected_native_source)
        write_new(args.output/'execution.json',state)
    return code

if __name__=='__main__':raise SystemExit(main())
