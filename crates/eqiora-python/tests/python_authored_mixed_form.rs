use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyModule};

#[test]
fn authored_mixed_inspection_checks_actual_equations_before_exposure() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", public_module(py)?)?;
        py.run(c_str!(r#"
graph=eqiora.geometry.GeometryGraph()
rectangle=graph.rectangle(x_bounds=(0.,1.),y_bounds=(0.,1.))
names=("left","right","bottom","top")
geometry=graph.build(rectangle,named_topology={"body":rectangle.region,**dict(zip(names,rectangle.boundaries))})
source='''
public component Flow(support body:volume(ambient_dimension=2),support surface:complete_exterior(parent=body),parameter mu:kg/m/s,parameter load:kg/m/s^2,parameter length:m) {
 variable u:vector<m/s,2> on body;
 variable p:kg/m/s^2 on body;
 variable F:kg/m/s^2 on body;
 relation force on body { F - load*coordinate(0)/length = 0; }
 relation momentum on body { -div(2*mu*symmetric_part(grad(u))-isotropic_lift(p))-grad(F)=0; }
 relation continuity on body { div(u)=0; }
 relation fixed[face in surface] on face { trace(u)=0; }
 form weak for momentum, continuity {
  test v:1 for u zero_on surface;
  test q:1 for p;
  integrate(body,frobenius(grad(v),2*mu*symmetric_part(grad(u)))-p*div(v)) = integrate(body,dot(v,grad(F)));
  integrate(body,q*div(u)) = 0;
 }
}
'''
bindings={"body":geometry.selection("body"),"surface":(tuple(geometry.selection(n) for n in names),geometry.selection("body")),"mu":2.5,"load":3.,"length":2.}
model=eqiora.compile(source=source,geometry=geometry,entry="Flow",bindings=bindings)
form,=model.authored_formulations
assert form.kind=="mixed-galerkin"
assert len(form.relation_ids)==len(form.trial_field_ids)==len(form.test_restrictions)==2
assert [test[0] for test in form.test_restrictions]==["v","q"]
assert len(form.test_restrictions[0][2])==4 and form.test_restrictions[1][2]==[]
assert form.implication=="strong-implies-weak"
assert "admissible-l2-pressure-test" in form.assumptions
assert len(model.render_formulations())==1
assert eqiora.Model.from_bytes(model.to_bytes()).authored_formulations==()
# Python authoring reaches the same typed source and independent admission owner.
q=eqiora.lang
module=eqiora.Module("main")
component=module.component("Flow")
body=component.volume("body",dimensions=2)
surface=component.complete_exterior("surface",parent=body)
velocity_type=eqiora.ValueType.vector(eqiora.ValueType.real(eqiora.Dimension(length=1,time=-1)),2)
pressure_type=eqiora.ValueType.real(eqiora.Dimension(mass=1,length=-1,time=-2))
u=component.field("u",value_type=velocity_type,role=eqiora.FieldRole.Variable,on=body)
p=component.field("p",value_type=pressure_type,role=eqiora.FieldRole.Variable,on=body)
F=component.field("F",value_type=pressure_type,role=eqiora.FieldRole.Variable,on=body)
mu=component.parameter("mu",value_type=eqiora.ValueType.real(eqiora.Dimension(mass=1,length=-1,time=-1)))
load=component.parameter("load",value_type=pressure_type)
length=component.parameter("length",value_type=eqiora.ValueType.real(eqiora.Dimension(length=1)))
component.relation("force",q.equation(F,load*q.coordinate(0)/length),on=body)
momentum_law=component.relation("momentum",q.equation(-q.div(2*mu*q.symmetric_part(q.grad(u))-q.isotropic_lift(p))-q.grad(F),0),on=body)
continuity_law=component.relation("continuity",q.equation(q.div(u),0),on=body)
component.relation("fixed",q.equation(q.trace(u),0),on=surface.member("face"))
v=component.test("v",for_=u,zero_on=surface)
pressure_test=component.test("q",for_=p)
component.weak_form("weak",[momentum_law,continuity_law],equations=[
 (q.integrate(body,q.frobenius(q.grad(v),2*mu*q.symmetric_part(q.grad(u)))-p*q.div(v)),q.integrate(body,q.dot(v,q.grad(F)))),
 (q.integrate(body,pressure_test*q.div(u)),0),
])
python_model=eqiora.compile(source=module,geometry=geometry,entry="Flow",bindings=bindings)
emitted_model=eqiora.compile(source=module.to_eqi(),geometry=geometry,entry="Flow",bindings=bindings)
assert python_model.authored_formulations[0].source_identity==emitted_model.authored_formulations[0].source_identity
assert python_model.authored_formulations[0].kind=="mixed-galerkin"
# Deleting an entire mixed equation must not reclassify the vector form as scalar.
incomplete=source.replace("momentum, continuity", "momentum").replace("  test q:1 for p;", "").replace("  integrate(body,q*div(u)) = 0;", "")
try:
 eqiora.compile(source=incomplete,geometry=geometry,entry="Flow",bindings=bindings)
except eqiora.ValidationError:
 pass
else:
 raise AssertionError("incomplete vector system bypassed the mixed checker")
# Source order carries explicit equation ownership; joint permutation is admissible.
momentum="  integrate(body,frobenius(grad(v),2*mu*symmetric_part(grad(u)))-p*div(v)) = integrate(body,dot(v,grad(F)));"
continuity="  integrate(body,q*div(u)) = 0;"
permuted=source.replace("momentum, continuity","continuity, momentum").replace(momentum+"\n"+continuity,continuity+"\n"+momentum)
assert len(eqiora.compile(source=permuted,geometry=geometry,entry="Flow",bindings=bindings).authored_formulations[0].relation_ids)==2
for label,old,new in [
 ("missing equation","  integrate(body,q*div(u)) = 0;",""),
 ("test swap","test v:1 for u zero_on surface;","test v:1 for p zero_on surface;"),
 ("foreign pressure","-p*div(v)","-F*div(v)"),
 ("pressure sign","-p*div(v)","+p*div(v)"),
 ("source sign","dot(v,grad(F))","-dot(v,grad(F))"),
 ("boundary omission","test v:1 for u zero_on surface;","test v:1 for u;"),
 ("extra term","-p*div(v)","-p*div(v)+p*div(v)"),
 ("foreign forcing","dot(v,grad(F))","dot(v,grad(p))"),
]:
 mutant=source.replace(old,new)
 assert mutant!=source
 try:
  eqiora.compile(source=mutant,geometry=geometry,entry="Flow",bindings=bindings)
 except eqiora.ValidationError:
  pass
 else:
  raise AssertionError("unchecked authored mixed form exposed: "+label)
"#),Some(&locals),Some(&locals))
    })
}

fn public_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let native = pyo3::wrap_pymodule!(_eqiora::_eqiora)(py);
    let package_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bindings/python/python/eqiora")
        .canonicalize()?;
    let locals = PyDict::new(py);
    locals.set_item("native", native.bind(py))?;
    locals.set_item("package_directory", package_directory.to_string_lossy())?;
    py.run(
        c_str!(
            r#"
import importlib.util
import pathlib
import sys

package_path = pathlib.Path(package_directory)
spec = importlib.util.spec_from_file_location(
    "eqiora",
    package_path / "__init__.py",
    submodule_search_locations=[str(package_path)],
)
package = importlib.util.module_from_spec(spec)
sys.modules["eqiora"] = package
sys.modules["eqiora._eqiora"] = native
spec.loader.exec_module(package)
"#
        ),
        None,
        Some(&locals),
    )?;
    Ok(locals
        .get_item("package")?
        .expect("public package must load")
        .cast_into::<PyModule>()?)
}
