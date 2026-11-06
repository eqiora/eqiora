import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
build=Path('/data/nk523/.cache/eqiora/robin-campaign/build-619725c18-1');plan=json.loads((build/'launch-plan.json').read_text())
execution=build/'execution';finish=json.loads((execution/'producer-finish.json').read_text());assert finish['exit_code']==0
root=Path(plan['environment']['EQIORA_ROBIN_OUTPUT']);reports=execution/'comparisons';reports.mkdir(exist_ok=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(plan['reference_script'])==plan['reference_script_sha256']
assert sorted(p.name for p in root.iterdir())==sorted(plan['cases'])
records=[]
for case in plan['cases']:
 directory=root/case;native=json.loads((directory/'producer-receipt.json').read_text())
 assert native['source_head']==plan['head'] and native['native_executable_sha256']==plan['binary_sha256']
 assert native['ordinary_solves']==native['ordinary_public_runs']==1
 payload=directory/'robin-input.json';operator=directory/'captured-operator.json';result=directory/'result.json'
 before={name:sha(directory/name) for name in ['robin-input.json','captured-operator.json','result.json']}
 assert before['robin-input.json']==native['input_sha256']
 assert before['captured-operator.json']==native['captured_operator_sha256']
 assert before['result.json']==native['result_sha256']
 report=reports/f'{case}.json';assert not report.exists()
 command=['taskset','-c','63','python3',plan['reference_script'],'compare',str(payload),'--output',str(report)]
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();begin=time.monotonic()
 with (reports/f'{case}.stdout.log').open('x') as stdout,(reports/f'{case}.stderr.log').open('x') as stderr:
  child=subprocess.Popen(command,stdout=stdout,stderr=stderr);code=child.wait()
 output=json.loads(report.read_text())
 assert before=={name:sha(directory/name) for name in before}
 record={'case':case,'command':command,'pid':child.pid,'start_utc':start,'finish_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'exit_code':code,'elapsed_seconds':time.monotonic()-begin,'accepted':output.get('accepted'),
  'input_sha256':before['robin-input.json'],'captured_operator_sha256':before['captured-operator.json'],
  'result_sha256':before['result.json'],'report':str(report),'report_sha256':sha(report)}
 records.append(record)
 with (reports/f'{case}-process.json').open('x') as target:json.dump(record,target,indent=2);target.write('\n')
 print(f'ROBIN_COMPARE {case} exit={code} accepted={output.get("accepted")} report={report}',flush=True)
 if code!=0 or output.get('accepted') is not True:raise SystemExit(code or 1)
assert sha(plan['reference_script'])==plan['reference_script_sha256']
summary={'schema':1,'status':'independent-reference-accepted','producer_exit':finish['exit_code'],'producer_pid':finish['pid'],
 'head':plan['head'],'tree':plan['tree'],'binary_sha256':plan['binary_sha256'],
 'reference_script_sha256':plan['reference_script_sha256'],'case_count':len(records),'records':records,
 'ordinary_runs_attempt2':4,'ordinary_runs_attempt1':1,'cumulative_ordinary_runs':5,'performance_claim':False}
with (reports/'summary.json').open('x') as target:json.dump(summary,target,indent=2);target.write('\n')
print('ROBIN_REFERENCE_SUMMARY',reports/'summary.json',flush=True)
