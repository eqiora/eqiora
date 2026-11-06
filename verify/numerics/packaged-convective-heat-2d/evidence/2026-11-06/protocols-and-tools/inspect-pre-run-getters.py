from pathlib import Path
import importlib.util,json,tempfile
import eqiora as q
root=Path('/data/nk523/tmp/eqiora-scalar-convection')
spec=importlib.util.spec_from_file_location('producer',root/'wt/bindings/python/tests/test_convective_heat_observations.py')
producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
reports=[]
for case,inputs in producer.CASES:
 geometry,bindings=producer.example.convection_geometry_and_bindings(*inputs)
 model=q.compile(path=producer.SPECIMEN/'src/main.eqi',entry='ConvectiveHeatedBody',geometry=geometry,bindings=bindings)
 encoded=model.to_bytes();restored=q.Model.from_bytes(encoded)
 assert restored.to_bytes()==encoded and restored.digest==model.digest
 field=model.field('definition.temperature')
 assert field==restored.field(field.id)
 selected=[]
 for name,_,boundary,_,_ in producer.IDENTITIES:
  observable=model.observable(f'definition.{name}');other=restored.observable(observable.id)
  domain=model.domain(boundary)
  assert observable.id==other.id and observable.model_digest==other.model_digest==domain.model_digest==model.digest
  selected.append({'observable_id':observable.id,'boundary_domain_id':domain.id})
 reports.append({'case':case,'model_digest':model.digest,'canonical_model_reopen':True,'exact_field_reopen':True,'observable_boundary_bindings':selected})
(root/'combined-pre-run-getters.json').write_text(json.dumps({'no_pde_runs':True,'cases':reports},indent=2)+'\n')
print('four-case canonical Model/Field/Observable/Domain getter bindings verified without runs')
