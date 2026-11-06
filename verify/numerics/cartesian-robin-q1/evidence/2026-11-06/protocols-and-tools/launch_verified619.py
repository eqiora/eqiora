import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/data/nk523/.cache/eqiora/robin-campaign/build-619725c18-1')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def output(command):return subprocess.check_output(command,text=True).strip()
launch_path=root/'launch-plan.json'
assert sha(launch_path)=='eb723d565136c4bac996dd1535bcb75ea28c64e849a965085c76f88f5d9f5156'
plan=json.loads(launch_path.read_text());receipt=json.loads((root/'build-receipt.json').read_text())
assert (root/'primary-build-review.json').is_file()
assert output(['git','rev-parse','HEAD'])==plan['head'] and output(['git','rev-parse','HEAD^{tree}'])==plan['tree']
assert not output(['git','status','--porcelain'])
assert sha(plan['binary'])==plan['binary_sha256']==receipt['binary_sha256']
assert sha(plan['reference_script'])==plan['reference_script_sha256']
assert not Path(plan['environment']['EQIORA_ROBIN_OUTPUT']).exists()
execution=root/'execution';execution.mkdir(exist_ok=False)
def save(name,data):
 with (execution/name).open('x') as stream:json.dump(data,stream,indent=2);stream.write('\n')
processes=[];conflicts=[]
for line in output(['ps','-u','nk523','-o','pid=,pcpu=,comm=']).splitlines():
 pid,pcpu,comm=line.strip().split(maxsplit=2);pid=int(pid);pcpu=float(pcpu)
 try: affinity=sorted(os.sched_getaffinity(pid))
 except ProcessLookupError:continue
 if 63 in affinity and (pcpu>=5 or len(affinity)<=8):
  item={'pid':pid,'pcpu':pcpu,'comm':comm,'affinity':affinity};processes.append(item)
  if pid!=os.getpid() and pcpu>=5 and (len(affinity)<=8 or comm in ['rustc','cargo','eqiora_numerics','cc1','cc1plus']):conflicts.append(item)
save('cpu63-conflict-check.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'processes':processes,'conflicts':conflicts})
assert not conflicts,'CPU63 has active owned computation; do not launch'
mpi={k:os.environ[k] for k in ['OMPI_COMM_WORLD_SIZE','PMI_SIZE','PMIX_RANK','SLURM_NTASKS','WORLD_SIZE'] if k in os.environ}
assert not mpi,f'active MPI/job environment: {mpi}'
env=os.environ.copy();env.update(plan['environment'])
os.sched_setaffinity(0,{63});assert os.sched_getaffinity(0)=={63}
start=datetime.datetime.now(datetime.timezone.utc).isoformat();begin=time.monotonic()
with (execution/'producer.stdout.log').open('x') as stdout,(execution/'producer.stderr.log').open('x') as stderr:
 child=subprocess.Popen(plan['command'],cwd=receipt['worktree'],env=env,stdout=stdout,stderr=stderr)
 try:affinity=sorted(os.sched_getaffinity(child.pid))
 except ProcessLookupError:affinity=None
 save('producer-start.json',{'utc':start,'pid':child.pid,'command':plan['command'],'environment':plan['environment'],
  'parent_affinity':sorted(os.sched_getaffinity(0)),'child_affinity_at_start':affinity,'mpi_environment':mpi,
  'head':plan['head'],'tree':plan['tree'],'binary_sha256':sha(plan['binary']),'launch_plan_sha256':sha(launch_path)})
 code=child.wait()
save('producer-finish.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'start_utc':start,
 'pid':child.pid,'exit_code':code,'elapsed_seconds':time.monotonic()-begin,
 'stdout_sha256':sha(execution/'producer.stdout.log'),'stderr_sha256':sha(execution/'producer.stderr.log'),
 'native_root':plan['environment']['EQIORA_ROBIN_OUTPUT'],'status':'exported-not-mathematically-compared' if code==0 else 'failed-no-retry'})
print(f'ROBIN_PRODUCER_EXIT={code} pid={child.pid} execution={execution}',flush=True)
raise SystemExit(code)
