import hashlib,json,os,subprocess,time,shutil
from pathlib import Path
wt=Path.cwd(); root=Path('/data/nk523/.cache/eqiora/robin-campaign/build-619725c18-1')
root.mkdir(parents=True,exist_ok=False)
def output(command):return subprocess.check_output(command,text=True).strip()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,value):
 with (root/name).open('x') as target:target.write(value)
head=output(['git','rev-parse','HEAD']);tree=output(['git','rev-parse','HEAD^{tree}'])
assert head=='619725c1891e5b6bb2d98dd4ed2f3989f6fd17c2'
assert not output(['git','status','--porcelain'])
focused=Path('/data/nk523/.cache/eqiora/tmp/robin/head619-focused-receipt.json')
checks=json.loads(focused.read_text());assert len(checks)==7 and all(c['exit_code']==0 and c['head']==head for c in checks)
command=['taskset','-c','48-51','cargo','test','-p','eqiora-numerics','--lib','--no-run','--locked','--message-format=json']
with (root/'build.jsonl').open('x') as stdout,(root/'build.stderr.log').open('x') as stderr:
 completed=subprocess.run(command,stdout=stdout,stderr=stderr)
save('build-exit.json',json.dumps({'command':command,'exit_code':completed.returncode},indent=2)+'\n')
assert completed.returncode==0
messages=[json.loads(line) for line in (root/'build.jsonl').read_text().splitlines() if line.startswith('{')]
artifacts=[m for m in messages if m.get('reason')=='compiler-artifact' and m.get('executable') and m['target']['name']=='eqiora_numerics']
assert len(artifacts)==1
built_binary=Path(artifacts[0]['executable']);assert built_binary.is_file()
binary=root/'native-producer';shutil.copy2(built_binary,binary);binary.chmod(0o555)
assert sha(binary)==sha(built_binary)
compiler=output(['rustc','-Vv']);cargo=output(['cargo','-Vv'])
save('rustc.txt',compiler+'\n');save('cargo.txt',cargo+'\n')
linked=subprocess.run(['ldd',str(binary)],text=True,capture_output=True)
save('ldd.txt',linked.stdout+linked.stderr)
reference=Path('/data/nk523/tmp/eqiora-robin-q1-reference/wt')
refhead=output(['git','-C',str(reference),'rev-parse','HEAD'])
assert refhead=='023c3786e2a4af4582a3e66170fc1983bb5b46b9'
assert not output(['git','-C',str(reference),'status','--porcelain'])
refdir=reference/'verify/numerics/cartesian-robin-q1/reference'
protocol=Path('/data/nk523/.cache/eqiora/tmp/robin/producer-protocol-attempt2.md')
receipt={'schema':1,'status':'built-not-executed','head':head,'tree':tree,'worktree':str(wt),
 'build_command':command,'build_exit':completed.returncode,'build_jsonl_sha256':sha(root/'build.jsonl'),
 'built_executable':str(built_binary),'built_executable_sha256':sha(built_binary),'binary':str(binary),'binary_sha256':sha(binary),'lock_sha256':sha(wt/'Cargo.lock'),
 'rustc':compiler,'cargo':cargo,'focused_receipt':str(focused),'focused_receipt_sha256':sha(focused),
 'producer_protocol':str(protocol),'producer_protocol_sha256':sha(protocol),
 'reference_head':refhead,'reference_tree':output(['git','-C',str(reference),'rev-parse','HEAD^{tree}']),
 'reference_files':{name:sha(refdir/name) for name in ['robin.py','test_robin.py','README.md']},
 'environment':{key:os.environ.get(key) for key in ['CARGO_TARGET_DIR','TMPDIR','CARGO_BUILD_JOBS','CMAKE_BUILD_PARALLEL_LEVEL','RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CC','CXX','LIBCLANG_PATH','BINDGEN_EXTRA_CLANG_ARGS','SUNDIALS_INCLUDE_DIR','SUNDIALS_LIBRARY_DIR','CUDA_VISIBLE_DEVICES']}}
assert output(['git','rev-parse','HEAD'])==head and not output(['git','status','--porcelain'])
save('build-receipt.json',json.dumps(receipt,indent=2)+'\n')
fixture='numerical_admission::tests::plans::robin::native::ordinary_robin_producer_exports_full_operator_coefficients_and_reactions'
native_root=Path('/data/nk523/.cache/eqiora/robin-campaign/native-619725c18-attempt2');assert not native_root.exists()
launch={'schema':1,'status':'proposed-not-launched','head':head,'tree':tree,
 'build_receipt':str(root/'build-receipt.json'),'build_receipt_sha256':sha(root/'build-receipt.json'),
 'binary':str(binary),'binary_sha256':sha(binary),'cpu_affinity':[63],
 'command':['taskset','-c','63',str(binary),'--ignored','--exact',fixture,'--nocapture','--test-threads=1'],
 'environment':{'EQIORA_ROBIN_OUTPUT':str(native_root),'EQIORA_ROBIN_SOURCE_HEAD':head,
  **{k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS']},
  'CUDA_VISIBLE_DEVICES':'','HIP_VISIBLE_DEVICES':'','ROCR_VISIBLE_DEVICES':''},
 'mpi_launch':False,'conflict_check':'required again immediately before approved launch',
 'maximum_ordinary_public_runs':4,'cumulative_maximum_ordinary_public_runs':5,
 'prior_attempts':[{'source_head':'5e5ab85a33b8453ac98ac17676fde971495fb35c','ordinary_public_runs':1,'comparisons':0,'exit_code':101,
 'metadata_audit':'/data/nk523/.cache/eqiora/robin-campaign/build-5e5ab85a3-1/attempt1-metadata-audit/audit-receipt.json'}],'cases':['insulated','robin','changed_h','changed_geometry_material_cooling'],
 'fixed_inputs':[[2,1,3,0,280,320],[2,1,3,2,280,320],[2,1,3,5,280,320],[3,2,2,4,310,280]],
 'solver':{'algorithm':'BiConjugateGradientStabilized','provider':'reference','reduction':'Reproducible','relative_tolerance':1e-14,'absolute_tolerance':1e-14,'maximum_iterations':10000},
 'space':'Q1','mesh_cells':[2,2],'gauss_points_per_axis':3,'performance_claim':False,
 'reference_script':str(refdir/'robin.py'),'reference_script_sha256':sha(refdir/'robin.py'),
 'reference_budgets':'unchanged approved protocol; exact rational comparison includes direct recovery R',
 'failure_policy':'preserve immutable partial attempts and actual outer-process status; no automatic retry or tolerance changes'}
save('launch-plan.json',json.dumps(launch,indent=2)+'\n')
print('ROBIN_BUILD_RECEIPT',root/'build-receipt.json',flush=True)
print('ROBIN_LAUNCH_PLAN',root/'launch-plan.json',flush=True)
print('ROBIN_BINARY_SHA256',sha(binary),flush=True)
