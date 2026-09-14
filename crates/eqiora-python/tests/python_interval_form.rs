use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyModule};

#[test]
fn python_inspects_and_renders_authored_interval() -> PyResult<()> {
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
public component Balance(support body: volume(ambient_dimension=1), parameter k:kg*m/s^3/K, parameter s:kg/m/s^3) {
 variable T:K on body;
 law balance on body { flux -k*grad(T); source s; }
 form conservative for balance {
  interval segment(a,b) on body;
  outward_flux(segment,a,-k*grad(T)) + outward_flux(segment,b,-k*grad(T)) = integrate(segment,s);
 }
}
'''
model = eqiora.compile(source=source, geometry=geometry, entry="Balance", bindings={"body": geometry.selection("body"), "k": 3.0, "s": 12.0})
form, = model.authored_formulations
assert form.kind == "integral-conservative"
assert form.interval == ("segment", "a", "b")
assert form.test_name is None
assert form.zero_on_domain_ids == []
assert form.implication == "strong-implies-interval-conservation"
assert form.assumptions == ["fixed-one-dimensional-domain", "classical-divergence-and-boundary-trace", "every-ordered-subinterval-of-parent"]
assert "integral-conservative" in repr(form)
assert len(model.render_formulations()) == 1
"#),None,Some(&locals))
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
