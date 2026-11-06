import copy,datetime,hashlib,importlib.util,itertools,json,os,subprocess,time
from fractions import Fraction as F
from pathlib import Path
build=Path('/data/nk523/.cache/eqiora/robin-campaign/build-619725c18-1');plan=json.loads((build/'launch-plan.json').read_text())
root=Path(plan['environment']['EQIORA_ROBIN_OUTPUT']);out=build/'execution'/'payload-falsifiers';out.mkdir(exist_ok=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(plan['reference_script'])==plan['reference_script_sha256']
spec=importlib.util.spec_from_file_location('approved_robin_reference',plan['reference_script']);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
base={case:json.loads((root/case/'robin-input.json').read_text()) for case in plan['cases']}
original={case:sha(root/case/'robin-input.json') for case in base}
def h(x):return float(x).hex()
def matrix(data,a):data['A_binary64_hex']=[[h(x) for x in row] for row in a]
def load(data,b):data['b_binary64_hex']=[h(x) for x in b]
records=[]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();begin=time.monotonic()
for case in plan['cases'][1:]:
 ref=r.reference(case);a,b,v,_=r.decode(base[case],ref);variants=[]
 def add(name,mutate,required_false=None,required_true=None):
  data=copy.deepcopy(base[case]);mutate(data);assert data!=base[case]
  variants.append((name,data,required_false,required_true))
 add('omitted-robin-matrix',lambda d:matrix(d,[[x-y for x,y in zip(row,rob)] for row,rob in zip(a,ref['robin'])]),'full_operator')
 def reversed_boundary(d):
  matrix(d,[[x-2*y for x,y in zip(row,rob)] for row,rob in zip(a,ref['robin'])]);load(d,[-x for x in b])
 add('reversed-robin-boundary-sign',reversed_boundary,'full_operator')
 length,height,k,transfer,tc,ambient=ref['parameters'];my,_,qy=r.axis(height);nodes=list(itertools.product(range(3),repeat=2))
 def wrong_face(d):
  matrix(d,[[a[i][j]-ref['robin'][i][j]+(transfer*my[p[1]][q[1]] if p[0]==q[0]==0 else 0)
   for j,q in enumerate(nodes)] for i,p in enumerate(nodes)])
  load(d,[b[i]-ref['b'][i]+(transfer*ambient*qy[p[1]] if p[0]==0 else 0) for i,p in enumerate(nodes)])
 add('wrong-robin-face',wrong_face,'full_operator')
 def eliminated_trace(d):
  changed=copy.deepcopy(a);rhs=list(b)
  for i in range(6,9):
   for j in range(9):changed[i][j]=F(int(i==j));changed[j][i]=F(int(i==j))
   rhs[i]=ref['exact'][i]
  matrix(d,changed);load(d,rhs)
 add('incorrectly-eliminated-right-test-trace',eliminated_trace,'full_operator')
 def endpoints(d):
  changed=copy.deepcopy(a);rhs=list(b)
  for i in (6,8):
   rhs[i]=F(0)
   for j in range(9):changed[i][j]-=ref['robin'][i][j]
  matrix(d,changed);load(d,rhs)
 add('missing-endpoint-trace-weights',endpoints,'full_operator')
 def reaction_sign(d):
  for row in d['left_reactions']:row['value_binary64_hex']=(-float.fromhex(row['value_binary64_hex'])).hex()
 add('reversed-weak-reaction-sign',reaction_sign,'every_reaction_vs_exact_target')
 add('wrong-physical-heat-convention',lambda d:d.update(heat_convention='physical_outward_heat=weak_reaction'))
 def direct_recovery(d):
  physical=[x-y for x,y in zip(r.apply(a,v),b)]
  offset=(ref['bounds']['R']+ref['bounds']['rho'])/2
  d['left_reactions'][0]['value_binary64_hex']=h(physical[0]+offset)
 add('direct-recovery-defect-inside-looser-rho',direct_recovery,'every_reaction_vs_exported_physical_operator','every_reaction_vs_reference_full_vector')
 def coefficient(d):
  d['coefficients'][4]['value_binary64_hex']=(float.fromhex(d['coefficients'][4]['value_binary64_hex'])+2**-10).hex()
 add('changed-interior-coefficient',coefficient,'every_coefficient')
 def reaction(d):
  d['left_reactions'][1]['value_binary64_hex']=(float.fromhex(d['left_reactions'][1]['value_binary64_hex'])+2**-10).hex()
 add('changed-left-reaction',reaction,'every_reaction_vs_exact_target')
 if case in ['robin','changed_h']:
  other='changed_h' if case=='robin' else 'robin'
  add('changed-h-only-matrix',lambda d:d.update(A_binary64_hex=copy.deepcopy(base[other]['A_binary64_hex'])),'full_operator')
  add('changed-h-only-load',lambda d:d.update(b_binary64_hex=copy.deepcopy(base[other]['b_binary64_hex'])),'full_load')
 for name,data,required_false,required_true in variants:
  directory=out/case/name;directory.mkdir(parents=True,exist_ok=False)
  payload=directory/'mutant.json';report=directory/'rejection.json'
  with payload.open('x') as stream:json.dump(data,stream,indent=2);stream.write('\n')
  command=['taskset','-c','63','python3',plan['reference_script'],'compare',str(payload),'--output',str(report)]
  with (directory/'stdout.log').open('x') as stdout,(directory/'stderr.log').open('x') as stderr:
   child=subprocess.Popen(command,stdout=stdout,stderr=stderr);code=child.wait()
  result=json.loads(report.read_text());assert code in [1,2] and result['accepted'] is False,(case,name,code,result)
  if required_false:assert result['checks'][required_false] is False,(case,name,required_false)
  if required_true:assert result['checks'][required_true] is True,(case,name,required_true)
  record={'case':case,'defect':name,'pid':child.pid,'command':command,'exit_code':code,
   'required_false':required_false,'required_true':required_true,'input_sha256':sha(payload),'report_sha256':sha(report),
   'base_native_input_sha256':original[case],'accepted':False,'status':'expected-rejection'}
  with (directory/'process.json').open('x') as target:json.dump(record,target,indent=2);target.write('\n')
  records.append(record)
assert len(records)==34
assert original=={case:sha(root/case/'robin-input.json') for case in base}
assert sha(plan['reference_script'])==plan['reference_script_sha256']
summary={'schema':1,'status':'all-predeclared-payload-defects-rejected','pid':os.getpid(),'start_utc':start,
 'finish_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-begin,
 'reference_script_sha256':plan['reference_script_sha256'],'script_sha256':sha(__file__),
 'native_public_runs':0,'new_PDE_solves':0,'falsifier_count':len(records),'records':records,
 'method':'mutate actual accepted payloads with the unchanged independently defined affine Robin defects; no new expectations or budgets'}
with (out/'summary.json').open('x') as target:json.dump(summary,target,indent=2);target.write('\n')
print('ROBIN_PAYLOAD_FALSIFIERS count=34 expected-rejections=34 new-PDE-solves=0 summary='+str(out/'summary.json'),flush=True)
