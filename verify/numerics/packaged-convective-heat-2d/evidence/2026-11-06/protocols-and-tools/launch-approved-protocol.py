from pathlib import Path
import subprocess,os,json,hashlib,datetime,sys
root=Path('/data/nk523/tmp/eqiora-scalar-convection');repo=root/'wt'
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert head=='d3fd0424b7eda8df921a365ae2f4e179c989b7c7'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
assert hashlib.sha256((repo/'verify/numerics/cartesian-robin-q1/reference/robin.py').read_bytes()).hexdigest()=='200cc1ff5de8327d438506a75e4af59e03b188e730d4fb2b2b653ea06caa507d'
kind=sys.argv[1]; assert kind in ('native','python')
evidence=root/f'scientific-{kind}-d3fd-attempt1';evidence.mkdir()
env=dict(os.environ)
settings={key:'1' for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']}
settings.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',TMPDIR=str(root/'scratch'),PYTHONNOUSERSITE='1')
env.update(settings);env.pop('PYTHONPATH',None);env.pop('LD_LIBRARY_PATH',None)
if kind=='native':
 executable=root/'target/debug/deps/standard_thermal_package-ff1beb6dda49cefe'
 assert hashlib.sha256(executable.read_bytes()).hexdigest()=='d5f1f1d3e09428f83b1af942512dea6bd8506aef70c39f6636cf62c1d5879584'
 settings['EQIORA_ROBIN_PACKAGE_EVIDENCE']=str(evidence)
 args=[str(executable),'--exact','robin::observations::imported_convection_public_observations_and_artifact_replay_match_reference','--nocapture','--test-threads=1']
else:
 previous=json.loads((root/'native-launch.json').read_text());assert previous['exit_code']==0
 executable=root/'combined-venv/bin/python'
 settings['EQIORA_ROBIN_PACKAGE_PYTHON_EVIDENCE']=str(evidence)
 args=[str(executable),'-I','-m','pytest','-q',str(repo/'bindings/python/tests/test_convective_heat_observations.py'),'--maxfail=1',f'--basetemp={root}/scientific-python-pytest-scratch']
env.update(settings)
with (root/f'{kind}-protocol.log').open('w') as output:
 process=subprocess.Popen(['taskset','-c','3',*args],cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT)
 receipt={'schema':1,'kind':kind,'source_head':head,'source_tree':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=repo,text=True).strip(),'pid':process.pid,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':['taskset','-c','3',*args],'environment':settings,'unset_environment':['PYTHONPATH','LD_LIBRARY_PATH'],'evidence_root':str(evidence),'executable_path':str(executable),'executable_sha256':hashlib.sha256(executable.read_bytes()).hexdigest(),'build_receipt_sha256':hashlib.sha256((root/'combined-build-receipt.json').read_bytes()).hexdigest()}
 if kind=='python':receipt['installed_extension']=json.loads((root/'combined-installed-identity.json').read_text())
 receipt_path=root/f'{kind}-launch.json';receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
 code=process.wait();receipt['exit_code']=code;receipt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'kind':kind,'exit_code':code,'pid':process.pid,'evidence':str(evidence)}))
sys.exit(code)
