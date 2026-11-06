import json, subprocess, time
from pathlib import Path
root=Path('/data/nk523/.cache/eqiora/tmp/robin')
checks=[('fmt',['cargo','fmt','--all','--check']),
 ('clippy',['cargo','clippy','-p','eqiora-numerics','--tests','--','-D','warnings']),
 ('robin-pure',['cargo','test','-p','eqiora-numerics','--lib','robin','--','--nocapture']),
 ('orientation',['cargo','test','-p','eqiora-numerics','--lib','physical_outward_boundary_values_convert_to_diffusion_conormal_once']),
 ('scalar-baseline',['cargo','test','-p','eqiora-numerics','--lib','form_compiler::scalar::']),
 ('proof-baseline',['cargo','test','-p','eqiora-numerics','--lib','form_compiler::vocabulary::replay::tests::']),
 ('capture-cleanup',['cargo','test','-p','eqiora-numerics','--lib','capture_scope_disarms_on_drop_and_unwind_without_a_solve'])]
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
records=[]
for name,command in checks:
 log=root/f'head5e5-{name}.log';start=time.monotonic()
 with log.open('w') as output:
  result=subprocess.run(['taskset','-c','48-51',*command],stdout=output,stderr=subprocess.STDOUT)
 record={'name':name,'command':command,'exit_code':result.returncode,'seconds':time.monotonic()-start,'log':str(log),'head':head}
 records.append(record);(root/'head5e5-focused-receipt.json').write_text(json.dumps(records,indent=2)+'\n')
 print(f'ROBIN_FOCUSED {name} exit={result.returncode} log={log}',flush=True)
 if result.returncode: raise SystemExit(result.returncode)
