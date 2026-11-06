from pathlib import Path
import hashlib, importlib.metadata, json, subprocess, sys, zipfile
import eqiora
import eqiora._eqiora as native

root=Path('/data/nk523/tmp/eqiora-scalar-convection')
wheel,=list((root/'combined-wheels').glob('*.whl'))
extension=Path(native.__file__).resolve(strict=True)
assert extension.is_relative_to(root/'combined-venv')
assert Path(eqiora.__file__).resolve().is_relative_to(root/'combined-venv')
with zipfile.ZipFile(wheel) as archive:
 names=[name for name in archive.namelist() if name.endswith('.so') and '_eqiora' in name]
 assert len(names)==1,names
 embedded=archive.read(names[0])
 distribution=importlib.metadata.distribution('eqiora')
 dependencies={}
 for name in archive.namelist():
  if not (name.endswith('.so') or '.so.' in Path(name).name):continue
  installed=Path(distribution.locate_file(name)).resolve(strict=True)
  contents=archive.read(name)
  assert installed.read_bytes()==contents
  dependencies[name]={'installed_path':str(installed),'sha256':hashlib.sha256(contents).hexdigest(),'matches_wheel':True}
assert embedded==extension.read_bytes()
assert eqiora.__version__==importlib.metadata.version('eqiora')
sha=lambda data: hashlib.sha256(data).hexdigest()
receipt={'python':sys.version,'python_executable':sys.executable,'eqiora_version':eqiora.__version__,
 'package_path':str(Path(eqiora.__file__).resolve()),'native_extension_path':str(extension),
 'native_extension_sha256':sha(extension.read_bytes()),'wheel_path':str(wheel),
 'wheel_sha256':sha(wheel.read_bytes()),'wheel_native_extension_member':names[0],
 'installed_native_extension_matches_wheel':True,'source_shadowing_rejected':True,
 'native_dependencies':dependencies,'ldd':subprocess.check_output(['ldd',str(extension)],text=True),
 'public_getters':{name:hasattr(owner,attribute) for name,owner,attribute in [
 ('Result.observe',eqiora.Result,'observe'),('Result.to_bytes',eqiora.Result,'to_bytes'),
 ('Result.model_digest',eqiora.Result,'model_digest'),('Plan.identity',eqiora.Plan,'identity'),
 ('Model.digest',eqiora.Model,'digest'),('Model.observable',eqiora.Model,'observable')]}}
assert all(receipt['public_getters'].values())
assert 'not found' not in receipt['ldd']
(root/'combined-installed-identity.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('installed artifact identities and public entry points verified')
