from pathlib import Path
import hashlib,json,re,shutil,subprocess
root=Path('/data/nk523/tmp/eqiora-scalar-convection');repo=root/'wt'
archive=repo/'verify/numerics/packaged-convective-heat-2d/evidence/2026-11-06';archive.mkdir(parents=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
entries=[]
def copy(source,relative,metadata=False):
 relative=Path(relative)
 if relative.name=="scalar-README.md": relative=relative.with_name("scalar-README.md.txt")
 target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True)
 raw=source.read_bytes(); data=raw
 if metadata:
  text=raw.decode('utf-8');text=re.sub(r'/home/[^/\s]+/', '[HOME]/',text); data=text.encode('utf-8')
 with target.open('xb') as stream:stream.write(data)
 entries.append({'path':str(target.relative_to(archive)),'sha256':sha(target),'bytes':len(data),'original_sha256':hashlib.sha256(raw).hexdigest(),'original_path':str(source),'metadata_projection':data!=raw})
for kind in ('native','python'):
 for source in sorted((root/f'scientific-{kind}-d3fd-attempt1').rglob('*')):
  if source.is_file() and '__pycache__' not in source.parts and source.suffix!='.pyc':copy(source,Path(kind)/source.relative_to(root/f'scientific-{kind}-d3fd-attempt1'))
for name in ['combined-build-receipt.json','combined-installed-identity.json','primary-combined-build-review.json','primary-public-observation-result-review.json','public-protocol-summary.json','native-launch.json','python-launch.json','native-protocol.log','python-protocol.log','combined-pre-run-getters.json','frozen-native-no-pde-controls.json']:
 copy(root/name,Path('receipts')/name)
build=json.loads((root/'combined-build-receipt.json').read_text())
for name in build['logs']:
 copy(root/name,Path('receipts/build-history')/name,metadata=True)
for name in ['launch-approved-protocol.py','summarize-protocol.py','write-build-receipt.py','inspect-installed.py','inspect-pre-run-getters.py','archive-public-protocol.py']:
 copy(root/name,Path('protocols-and-tools')/name)
for path in ['crates/eqiora/tests/standard_thermal_package.rs','crates/eqiora/tests/standard_thermal_package/robin.rs','crates/eqiora/tests/standard_thermal_package/robin/observations.rs','bindings/python/tests/test_convective_heat_observations.py','examples/heated-body/run.py','Cargo.lock']:
 raw=subprocess.check_output(['git','show','d3fd0424b7eda8df921a365ae2f4e179c989b7c7:'+path],cwd=repo)
 target=archive/'producer-sources'/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
 entries.append({'path':str(target.relative_to(archive)),'sha256':sha(target),'bytes':len(raw),'original_git_object':'d3fd0424b7eda8df921a365ae2f4e179c989b7c7:'+path,'metadata_projection':False})
inventory={'schema':1,'status':'primary-audited-eight-public-workflow-witnesses','source_head':'d3fd0424b7eda8df921a365ae2f4e179c989b7c7','source_tree':'6fd1c08983340ade596474d49de72ad1f58b4482','ordinary_public_runs':{'native_initial':4,'installed_python_initial':4,'initial_total':8},'reference_sha256':'200cc1ff5de8327d438506a75e4af59e03b188e730d4fb2b2b653ea06caa507d','accepted_reports':8,'scope':'public coefficients and six physical observations plus replay; no package full A/b or private weak reactions','metadata_policy':'Only personal home prefixes in build diagnostics are projected; every original digest is retained. Scientific inputs, artifacts, receipts and comparator values are copied unchanged. Original build logs remain immutable at the recorded home-backed paths.','files':sorted(entries,key=lambda row:row['path'])}
(archive/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n');(archive/'inventory.sha256').write_text(sha(archive/'inventory.json')+'  inventory.json\n')
print(json.dumps({'archive':str(archive),'files':len(entries),'metadata_projections':sum(row['metadata_projection'] for row in entries),'inventory_sha256':sha(archive/'inventory.json')}))
