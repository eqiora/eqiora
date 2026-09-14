use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyModule};

#[test]
fn python_interval_form_closes_tpfa_selection_run_and_replay() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", module)?;
        py.run(c_str!(r#"
graph = eqiora.geometry.GeometryGraph()
interval = graph.interval(bounds=(0.0, 8.0))
geometry = graph.build(interval, named_topology={"body": interval.region, "left": interval.boundaries[0], "right": interval.boundaries[1]})
source = '''
public component Balance(support body: volume(ambient_dimension=1), support left:boundary(parent=body), support right:boundary(parent=body), parameter k:kg*m/s^3/K, parameter s:kg/m/s^3) {
 variable T:K on body;
 law balance on body { flux -k*grad(T); source s; }
 relation left_value on left { trace(T)=0; }
 relation right_value on right { trace(T)=0; }
 form conservative for balance {
  interval segment(a,b) on body;
  outward_flux(segment,a,-k*grad(T)) + outward_flux(segment,b,-k*grad(T)) = integrate(segment,s);
 }
}
'''
model = eqiora.compile(source=source, geometry=geometry, entry="Balance", bindings={"body": geometry.selection("body"), "left":(geometry.selection("left"),geometry.selection("body")), "right":(geometry.selection("right"),geometry.selection("body")), "k": 3.0, "s": 12.0})
form, = model.authored_formulations
assert form.kind == "integral-conservative"
assert form.interval == ("segment", "a", "b")
assert form.test_name is None
assert form.zero_on_domain_ids == []
assert form.implication == "strong-implies-interval-conservation"
assert form.assumptions == ["fixed-one-dimensional-domain", "classical-divergence-and-boundary-trace", "every-ordered-subinterval-of-parent"]
assert "integral-conservative" in repr(form)
assert len(model.render_formulations()) == 1

mesh = eqiora.meshing.generate(eqiora.meshing.resolve(geometry, eqiora.meshing.CartesianMesher(cells=(4,))))
linear = eqiora.solve.Linear(
 algorithm=eqiora.solve.LinearSolver.ConjugateGradient,
 preconditioner=eqiora.solve.Preconditioner.Identity,
 reduction=eqiora.solve.Reduction.Reproducible,
 provider=eqiora.solve.SolverProvider.reference(),
 relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=100,
)
plain = eqiora.Model.from_bytes(model.to_bytes())
plans = [
 eqiora.resolve(model,mesh=mesh,spatial=eqiora.fvm.CellCenteredTpfa(),solve=linear),
 eqiora.resolve(plain,mesh=mesh,spatial=eqiora.fvm.CellCenteredTpfa(),solve=linear),
 eqiora.resolve(plain,mesh=mesh,spatial=eqiora.fvm.CellCenteredTpfa(),solve=linear,formulation=eqiora.formulation.IntegralConservative),
]
assert len({plan.identity for plan in plans}) == 3
assert [plan.formulation.requested for plan in plans] == [eqiora.FormulationSelectionMode.Authored,eqiora.FormulationSelectionMode.Automatic,eqiora.FormulationSelectionMode.Exact]
assert plans[0].formulation.requested_source_identity == form.source_identity
assert all(plan.formulation.requested_source_identity is None for plan in plans[1:])
# Independent four-cell TPFA equations, h=2, k=3, source=12:
# (k/h)*(3*T0-T1)=24; interior (k/h)*(2*Ti-Tprev-Tnext)=24.
# Their solution is [16,32,32,16], not continuum center samples [14,30,30,14].
# ||A^-1||inf=4/3; ||b||2=48, so the requested residual implies error<6.4e-11.
for plan in plans:
 assert plan.formulation.effective == eqiora.FormulationKind.IntegralConservative
 replay = eqiora.Plan.from_bytes(plan.to_bytes())
 assert replay.identity == plan.identity
 assert replay.to_bytes() == plan.to_bytes()
 assert replay.formulation.requested == plan.formulation.requested
 result = eqiora.run(replay)
 recovered = eqiora.Result.from_bytes(replay,result.to_bytes())
 assert recovered.to_bytes() == result.to_bytes()
 field = model.field(form.trial_field_id)
 values = recovered.output(field).values("cell").numpy().reshape(-1).tolist()
 assert len(values) == 4
 assert all(abs(actual-expected)<1e-9 for actual,expected in zip(values,[16,32,32,16]))
 fluxes=[-3*values[0]]+[-1.5*(values[i+1]-values[i]) for i in range(3)]+[3*values[-1]]
 assert all(abs(actual-expected)<1e-8 for actual,expected in zip(fluxes,[-48,-24,0,24,48]))
 assert all(abs(fluxes[i+1]-fluxes[i]-24)<1e-8 for i in range(4))
# Canonical payload mutations must reach mathematical admission, not merely fail JSON parsing.
import base64, json
original = plans[0].to_bytes()
wire = json.loads(original)
encoded = wire["authored_formulation_base64"]
projection = base64.b64decode(encoded).decode()
projection_data = json.loads(projection)
for label, mutant in [
 ("normal",projection.replace('"normal":-1','"normal":1',1)),
 ("duplicate endpoint",projection.replace('"endpoint":"a"','"endpoint":"b"',1)),
 ("scope",projection.replace('"interval":"segment"','"interval":"body"',1)),
 ("assumptions",projection.replace('fixed-one-dimensional-domain','moving-domain')),
 ("foreign Law",projection.replace(projection_data["relation_ulid"],projection_data["trial_ulid"],1)),
]:
 assert mutant != projection
 forged = original.replace(encoded.encode(),base64.b64encode(mutant.encode()),1)
 try:
  eqiora.Plan.from_bytes(forged)
 except eqiora.ValidationError as error:
  assert "interval" in str(error).lower() or "hypotheses" in str(error).lower(), (label,str(error))
 else:
  raise AssertionError("mutated interval Plan admitted: " + label)
try:
 eqiora.resolve(plain,mesh=mesh,spatial=eqiora.fvm.CellCenteredTpfa(),solve=linear,formulation=eqiora.formulation.PrimalGalerkin)
except eqiora.ValidationError:
 pass
else:
 raise AssertionError("wrong explicit TPFA form admitted")
try:
 eqiora.Result.from_bytes(plans[1],eqiora.run(plans[0]).to_bytes())
except eqiora.ValidationError:
 pass
else:
 raise AssertionError("Result from foreign requested Plan admitted")
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
