import hashlib,json,shutil,subprocess
from pathlib import Path
wt=Path('/data/nk523/.cache/eqiora/worktrees/scalar-robin-boundary')
archive=wt/'verify/numerics/cartesian-robin-q1/evidence/2026-11-06';archive.mkdir(parents=True,exist_ok=True)
cache=Path('/data/nk523/.cache/eqiora/robin-campaign');build=cache/'build-619725c18-1';old=cache/'build-5e5ab85a3-1'
inputs=cache/'native-619725c18-attempt2';first=cache/'native-5e5ab85a3-attempt1';bindings=[]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def put(source,relative):
 target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():
  with Path(source).open('rb') as before,target.open('xb') as after:shutil.copyfileobj(before,after)
 assert sha(source)==sha(target)
 bindings.append({'path':str(target.relative_to(archive)),'sha256':sha(target),'bytes':target.stat().st_size,'original_path':str(source)})
for case in ['insulated','robin','changed_h','changed_geometry_material_cooling']:
 native=json.loads((inputs/case/'producer-receipt.json').read_text());report=json.loads((build/'execution/comparisons'/f'{case}.json').read_text())
 assert native['source_head']=='619725c1891e5b6bb2d98dd4ed2f3989f6fd17c2'
 assert native['ordinary_public_runs']==native['ordinary_solves']==1
 assert report['accepted'] and len(report['checks'])==9 and all(report['checks'].values())
 assert sha(inputs/case/'robin-input.json')==native['input_sha256']==report['input_sha256']
 assert sha(inputs/case/'captured-operator.json')==native['captured_operator_sha256']
 assert sha(inputs/case/'result.json')==native['result_sha256']
 candidate=json.loads((inputs/case/'robin-input.json').read_text());captured=json.loads((inputs/case/'captured-operator.json').read_text())
 assert all(candidate[key]==captured[key] for key in ['order','A_binary64_hex','b_binary64_hex'])
 for file in sorted((inputs/case).iterdir()):put(file,Path('attempt2')/case/file.name)
 put(build/'execution/comparisons'/f'{case}.json',Path('attempt2')/case/'comparison.json')
 put(build/'execution/comparisons'/f'{case}-process.json',Path('attempt2')/case/'comparison-process.json')
for file in sorted((first/'insulated').iterdir()):put(file,Path('attempt1/insulated')/file.name)
for label,folder in [('attempt1',old),('attempt2',build)]:
 for name in ['build-receipt.json','build-exit.json','launch-plan.json','build.jsonl','build.stderr.log','rustc.txt','cargo.txt','ldd.txt','primary-build-review.json']:
  put(folder/name,Path('receipts')/label/name)
 for name in ['cpu63-conflict-check.json','producer-start.json','producer-finish.json','producer.stdout.log','producer.stderr.log']:
  put(folder/'execution'/name,Path('receipts')/label/name)
put(build/'execution/comparisons/summary.json','receipts/attempt2/comparison-summary.json')
put(build/'primary-scientific-review.json','receipts/attempt2/primary-scientific-review.json')
for file in sorted((old/'attempt1-metadata-audit').iterdir()):put(file,Path('receipts/attempt1/metadata-audit')/file.name)
falsifiers=build/'execution/payload-falsifiers'
assert json.loads((falsifiers/'summary.json').read_text())['falsifier_count']==34
for file in sorted(falsifiers.rglob('*')):
 if file.is_file():put(file,Path('payload-falsifiers')/file.relative_to(falsifiers))
for name in ['falsifiers.stdout.log','falsifiers.stderr.log']:put(build/'execution'/name,Path('receipts/attempt2')/name)
tmp=Path('/data/nk523/.cache/eqiora/tmp/robin')
for name in ['producer-protocol.md','producer-protocol-attempt2.md','audit_attempt1_metadata.py','payload_falsifiers619.py','compare_verified619.py','launch_verified.py','launch_verified619.py','prepare_build.py','prepare_build619.py','focused_checks.py','focused_checks619.py','archive_science.py']:
 put(tmp/name,Path('protocols-and-tools')/name)
for prefix in ['head5e5','head619']:
 put(tmp/f'{prefix}-focused-receipt.json',Path('receipts')/prefix/'focused-receipt.json')
 for name in ['fmt','clippy','robin-pure','orientation','scalar-baseline','proof-baseline','capture-cleanup']:
  put(tmp/f'{prefix}-{name}.log',Path('receipts')/prefix/f'{name}.log')
lock=wt/'Cargo.lock';assert sha(lock)==json.loads((build/'build-receipt.json').read_text())['lock_sha256'];put(lock,'build/Cargo.lock')
for head,label in [('5e5ab85a33b8453ac98ac17676fde971495fb35c','attempt1'),('619725c1891e5b6bb2d98dd4ed2f3989f6fd17c2','attempt2')]:
 for relative in ['numerical_admission/tests/plans/robin/native.rs','numerical_admission/tests/plans/robin.rs','numerical_admission/scalar/live_tests/capture.rs','numerical_admission/tests/artifact_capture.rs']:
  repo_path='crates/eqiora-numerics/src/'+relative
  raw=subprocess.check_output(['git','show',head+':'+repo_path],cwd=wt)
  target=archive/'producer-sources'/label/relative;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as output:output.write(raw)
  bindings.append({'path':str(target.relative_to(archive)),'sha256':sha(target),'bytes':len(raw),'original_git_object':head+':'+repo_path})
for case in ['insulated','robin','changed_h','changed_geometry_material_cooling']:
 expected=json.loads((inputs/case/'producer-receipt.json').read_text())['producer_source_sha256']
 assert sha(archive/'producer-sources/attempt2/numerical_admission/tests/plans/robin/native.rs')==expected
inventory={'schema':1,'status':'sealed-independent-four-case-witnesses-and-payload-falsifiers',
 'scientific_producer':{'head':'619725c1891e5b6bb2d98dd4ed2f3989f6fd17c2','tree':'00a14fdd63b4b123a47bd36bc544ceecb55850f9',
  'binary_sha256':'87d7c4d1efbb45a0d1a6fd02e9725fa72301bbb8c3afefd7c8cdb30aef8f1c6d','compiled_binary_archived':False},
 'independent_reference':{'head':'023c3786e2a4af4582a3e66170fc1983bb5b46b9','robin_py_sha256':'4855a307d70a52c76086c1d8bc877697c6c297ad739e5db8b32c14d9ea2b3d3c'},
 'ordinary_runs':{'attempt1':1,'attempt2':4,'campaign_total':5,'payload_falsifiers':0},
 'accepted_case_count':4,'accepted_checks_per_case':9,'payload_falsifiers_rejected':34,
 'performance_claim':False,'attempt1_status':'harness-failed-after-first-Run-no-scientific-comparison-no-full-operator-file',
 'reference_preparation_only_marker':'preserved original mathematical-reference metadata; actual native-input comparisons ran and accepted all nine checks',
 'files':sorted(bindings,key=lambda entry:entry['path'])}
with (archive/'inventory.json').open('x') as output:json.dump(inventory,output,indent=2);output.write('\n')
with (archive/'inventory.sha256').open('x') as output:output.write(sha(archive/'inventory.json')+'  inventory.json\n')
print('ROBIN_ARCHIVE',archive,'files',len(bindings),'bytes',sum(item['bytes'] for item in bindings),'inventory_sha256',sha(archive/'inventory.json'))
