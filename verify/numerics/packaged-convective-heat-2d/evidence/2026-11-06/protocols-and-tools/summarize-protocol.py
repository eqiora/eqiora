from pathlib import Path
import hashlib,json,subprocess
root=Path('/data/nk523/tmp/eqiora-scalar-convection');sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
summary={'schema':1,'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root/'wt',text=True).strip(),'native':[],'python':[],'initial_public_runs':8,'scope':'public coefficients and six physical observations plus artifact replay; core full operator/private reactions remain separate'}
for kind in ('native','python'):
 launch=json.loads((root/f'{kind}-launch.json').read_text());assert launch['exit_code']==0
 summary[f'{kind}_launch']={'path':str(root/f'{kind}-launch.json'),'sha256':sha(root/f'{kind}-launch.json'),'pid':launch['pid']}
 for case in ('insulated','robin','changed_h','changed_geometry_material_cooling'):
  directory=Path(launch['evidence_root'])/case
  report=json.loads((directory/'comparison.json').read_text());receipt=json.loads((directory/'producer-receipt.json').read_text())
  assert report['accepted'] and report['counts']=={'coefficients':9,'free':6,'observations':6,'observation_evaluations':12}
  assert all(report['checks'].values()) and all(all(o['checks'].values()) for o in report['observations'])
  assert report['input_sha256']==sha(directory/'observations.json')
  if kind=='native':
   hashes={file:receipt[key] for file,key in [('model.json','model_sha256'),('plan.eqplan','plan_sha256'),('mesh.eqmesh','mesh_sha256'),('result.eqresult','result_sha256'),('package-compilation.json','package_compilation_sha256'),('observations.json','input_sha256')]}
  else:hashes=receipt['artifact_sha256']
  for file,expected in hashes.items():assert sha(directory/file)==expected,(kind,case,file)
  summary[kind].append({'case':case,'directory':str(directory),'report_sha256':sha(directory/'comparison.json'),'input_sha256':sha(directory/'observations.json'),'producer_receipt_sha256':sha(directory/'producer-receipt.json'),'accepted':True,'counts':report['counts'],'artifact_hashes_verified':hashes,'model_identity':receipt.get('model_digest'),'plan_identity':receipt['plan_identity'],'result_identity':receipt['result_identity']})
(root/'public-protocol-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('8 reports accepted; 8 initial public Runs; 72 coefficients and 96 primary/replay observation evaluations; artifact and input hashes verified')
