import base64,hashlib,json,math,struct,subprocess
from pathlib import Path
root=Path('/data/nk523/.cache/eqiora/robin-campaign/native-5e5ab85a3-attempt1/insulated')
out=Path('/data/nk523/.cache/eqiora/robin-campaign/build-5e5ab85a3-1/attempt1-metadata-audit');out.mkdir(exist_ok=False)
wt=Path('/data/nk523/.cache/eqiora/worktrees/scalar-robin-boundary')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
files=['case-attempt.json','model.eqi','model.json','mesh.json','plan.json','result.json']
before={name:sha(root/name) for name in files}
plan=json.loads((root/'plan.json').read_text());result=json.loads((root/'result.json').read_text())
model=json.loads((root/'model.json').read_text());owner=json.loads((root/'mesh.json').read_text())
assert base64.b64decode(plan['model_base64'])==(root/'model.json').read_bytes()
assert base64.b64decode(plan['mesh_base64'])==(root/'mesh.json').read_bytes()
assert result['content']['plan_identity']==plan['identity']
payload=result['content']['payload'];assert payload['kind']=='static' and len(payload['fields'])==1
field=payload['fields'][0];assert field['space']=='continuous-lagrange-p1' and len(field['blocks'])==1
block=field['blocks'][0];assert block['association']=='vertex' and block['logical_shape']==[3,3]
values=block['values'];assert len(values)==math.prod(block['logical_shape'])==9
assert all(type(v) in [float,int] and math.isfinite(v) for v in values)
# This audits lossless encoding/finite shape only, not scientific expected temperatures.
assert all(struct.pack('>d',float.fromhex(float(v).hex()))==struct.pack('>d',float(v)) for v in values)
fielddef=next(n['definition'] for n in model['nodes'] if n['id']['ulid']==field['field_id'])
assert fielddef['kind']=='field' and fielddef['value_type']['domain']=='real'
assert fielddef['value_type']['shape']==[] and fielddef['value_type']['dimension']==[[int(i==4),1] for i in range(7)]
mesh=json.loads(base64.b64decode(owner['resources']['mesh']['mesh_base64']))
assert mesh['dimension']==2 and [len(axis) for axis in mesh['axes']]==[3,3]
assert mesh['vertex_order']=='last-axis-fastest'
# Mesh axes independently fix support cardinality/ordering, not recovered values.
assert mesh['axes']==[[0.,1.,2.],[0.,0.5,1.]]
assert plan['assembly_quadrature']=={'kind':'gauss-legendre','points_per_axis':3}
contracts={
 'logical_shape':'common_result/spatial.rs:185-232 Q1 uses field_support; scalar/support.rs:59-110 nodal extent=end-start+1; existing plans/weak_evidence/curl.rs:117 asserts [3,3]',
 'coefficient_order':'scalar.rs:598 field_coefficient_entities uses the same field_support entity order; scalar/support.rs sorts vertex entities; scalar/fields.rs:161 physical_fields consumes exact FieldDof coefficient order',
 'reaction_selection':'region_assembly/reactions/interfaces/projection.rs:24 selects exactly mapping.free_dof(key).is_none(); only left3 are essential in this authored source',
 'reaction_units':'form_compiler/region/binding.rs:193 physical_dual_scale is inverse row multiplier and residual units*area/testbasis; heat W/m3* m2 gives W/m; single real K Field and SI scales1 checked in producer',
 'reaction_values':'region_assembly/reactions/interfaces.rs:31 physical_value scales normalized weak action; scalar/reactions.rs:6 recovers from the same Plan and accepted physical values; NO reaction reconstruction was executed by this audit',
 'reference_export':'independent robin.py decode requires all81A/9b/9coeff/3left actions with strict x-major/y-fast order and physical_outward_heat=-weak_reaction; shape is not a scientific tolerance',
 'binary64_export':'shared tests/artifact_capture.rs encodes finite exact binary64; pure existing helper test executed separately below',
 'remaining_limits':'No captured-operator file or recovered reaction file survived attempt1; this audit does not substitute reassembly, qualify a full reference payload, or establish numerical acceptance.'}
source_paths=['crates/eqiora-numerics/src/common_result/spatial.rs','crates/eqiora-numerics/src/numerical_admission/scalar/support.rs','crates/eqiora-numerics/src/numerical_admission/scalar/fields.rs','crates/eqiora-numerics/src/numerical_admission/scalar/reactions.rs','crates/eqiora-numerics/src/region_assembly/reactions/interfaces.rs','crates/eqiora-numerics/src/region_assembly/reactions/interfaces/projection.rs','crates/eqiora-numerics/src/form_compiler/region/binding.rs','crates/eqiora-numerics/src/numerical_admission/tests/artifact_capture.rs']
binary=Path('/data/nk523/.cache/eqiora/targets/scalar-robin-boundary/debug/deps/eqiora_numerics-83b3b6dd7399df9d')
assert sha(binary)=='e0672eeef171c5ea4e16b49112d809747c0343058f0cfe242cbf8342ca211049'
command=['taskset','-c','48-51',str(binary),'--exact','numerical_admission::tests::artifact_capture::binary64_hex_preserves_exact_finite_bits']
with (out/'binary64-pure.log').open('x') as log:completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
assert completed.returncode==0
assert before=={name:sha(root/name) for name in files}
receipt={'schema':1,'status':'metadata-audited-no-PDE-solve-no-scientific-acceptance',
 'attempt1_ordinary_public_runs':1,'attempt1_reference_comparisons':0,'actual_producer_exit':101,
 'immutable_attempt_files':before,'plan_identity':plan['identity'],'result_identity':result['identity'],
 'observed_metadata':{'field_id':field['field_id'],'space':field['space'],'association':block['association'],
 'logical_shape':block['logical_shape'],'coefficient_count':len(values),'real_scalar_units':'K','mesh_dimension':2,'axis_lengths':[3,3],
 'vertex_order':mesh['vertex_order'],'plan_result_binding':True,'embedded_model_mesh_match':True,'all_coefficients_finite_and_binary64_hex_roundtrip':True},
 'source_contract_review':contracts,'source_contract_hashes':{name:sha(wt/name) for name in source_paths},
 'reaction_metadata_audit':'source-only; actual reactions were not reached or reconstructed',
 'binary64_pure_command':command,'binary64_pure_exit':completed.returncode,'audit_script_sha256':sha(__file__)}
with (out/'audit-receipt.json').open('x') as target:json.dump(receipt,target,indent=2);target.write('\n')
print('ROBIN_METADATA_AUDIT',out/'audit-receipt.json')
