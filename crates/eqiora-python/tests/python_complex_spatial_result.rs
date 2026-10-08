use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyModule};
use std::path::Path;

#[test]
fn complex_spatial_results_retain_coefficients_and_immutable_transport() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", public_module(py)?)?;
        py.run(c_str!(r#"
import gc
import sys
assert "numpy" not in sys.modules
source = """
public component Wave(
 support body:volume(ambient_dimension=1),
 support left:boundary(parent=body),
 support right:boundary(parent=body)
) {
 parameter a:complex<m^2>=math.complex(6[m^2],6[m^2]);
 parameter q:complex<1>=math.complex(1,1);
 parameter f:complex<1>=math.complex(-2,4);
 variable u:complex<1> on body;
 relation balance on body { -div(a*grad(u))+q*u=f; }
 relation fixed_left on left { trace(u)=math.complex(1,3); }
 relation fixed_right on right { trace(u)=math.complex(1,3); }
 form weak for balance {
  test eta:1 for u zero_on left,right;
  integrate(body,inner(grad(eta),a*grad(u))+inner(eta,q*u))=integrate(body,inner(eta,f));
 }
}
"""
graph = eqiora.geometry.GeometryGraph()
interval = graph.interval(bounds=(0.0,1.0))
geometry = graph.build(interval, named_topology={"body":interval.region,"left":interval.boundaries[0],"right":interval.boundaries[1]})
bindings = {"body":geometry.selection("body"),"left":(geometry.selection("left"),geometry.selection("body")),"right":(geometry.selection("right"),geometry.selection("body"))}
model = eqiora.compile(source=source,geometry=geometry,entry="Wave",bindings=bindings)
mesh = eqiora.meshing.generate(eqiora.meshing.resolve(geometry,eqiora.meshing.CartesianMesher(cells=(4,))))
linear = eqiora.solve.Linear(algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,preconditioner=eqiora.solve.Preconditioner.Identity,reduction=eqiora.solve.Reduction.Reproducible,provider=eqiora.solve.SolverProvider.reference(),relative_tolerance=1e-12,absolute_tolerance=1e-14,maximum_iterations=128)
plan = eqiora.resolve(model,mesh=mesh,spatial=eqiora.fem.Q1(),solve=linear)
plan = eqiora.Plan.from_bytes(plan.to_bytes())
result = eqiora.run(plan)
field = model.field(model.authored_formulations[0].trial_field_ids[0])
output = result.output(field)
array = output.values("vertex")
assert output.coefficient_count("vertex") == 5
assert output.logical_shape("vertex") == (5,)
assert output.value_shape == ()
assert array.shape == (5,) and array.strides == (16,)
assert array.dtype == "complex128" and array.device == "cpu"
assert array.byte_order == sys.byteorder
assert array.readonly and array.aligned and array.c_contiguous
assert isinstance(array[0],complex) and abs(array[-1]-(1+3j)) < 1e-10
assert "numpy" not in sys.modules
for index in (5,-6):
    try:
        array[index]
    except IndexError:
        pass
    else:
        raise AssertionError("complex array admitted an out-of-bounds index")
# Constant u=1+3i has zero gradient and q*u=(1+i)(1+3i)=-2+4i.
# It belongs exactly to Q1; this oracle does not depend on assembled coefficients.
assert all(abs(array[i]-(1+3j)) < 1e-10 for i in range(5))
wire = result.to_bytes()
restored = eqiora.Result.from_bytes(plan,wire)
assert restored.to_bytes() == wire
assert all(abs(restored.output(field).values("vertex")[i]-(1+3j)) < 1e-10 for i in range(5))
view = array.numpy(copy=False)
import numpy as np
assert view.dtype == np.complex128 and view.shape == (5,)
assert view is array.numpy(copy=None)
assert not view.flags.writeable and not view.flags.owndata
try:
    view.setflags(write=True)
except ValueError:
    pass
else:
    raise AssertionError("complex canonical storage became writeable")
copied = array.numpy(copy=True)
assert copied.flags.writeable and not np.shares_memory(copied,view)
copied[0] = 7-9j
assert abs(view[0]-(1+3j)) < 1e-10
snapshot = np.from_dlpack(array)
assert snapshot.dtype == np.complex128
assert not np.shares_memory(snapshot,view)
np.testing.assert_allclose(snapshot,np.full(5,1+3j),rtol=0,atol=1e-10)
if snapshot.flags.writeable:
    snapshot[0] = -8+4j
assert abs(view[0]-(1+3j)) < 1e-10
try:
    np.from_dlpack(array,copy=False)
except BufferError:
    pass
else:
    raise AssertionError("complex DLPack exposed canonical storage")
# A conjugated imaginary load changes the interior solution, not prescribed ends.
changed = eqiora.compile(source=source.replace("math.complex(-2,4)","math.complex(-2,-4)"),geometry=geometry,entry="Wave",bindings=bindings)
changed_plan = eqiora.resolve(changed,mesh=mesh,spatial=eqiora.fem.Q1(),solve=linear)
changed_result = eqiora.run(changed_plan)
changed_field = changed.field(changed.authored_formulations[0].trial_field_ids[0])
changed_array = changed_result.output(changed_field).values("vertex")
assert abs(changed_array[0]-(1+3j)) < 1e-10
assert abs(changed_array[-1]-(1+3j)) < 1e-10
assert abs(changed_array[2]-(1+3j)) > 1e-3
# Wrong test/trial conjugation cannot acquire an authored Plan.
try:
    wrong = eqiora.compile(source=source.replace("inner(eta,f)","inner(f,eta)"),geometry=geometry,entry="Wave",bindings=bindings)
    eqiora.resolve(wrong,mesh=mesh,spatial=eqiora.fem.Q1(),solve=linear)
except eqiora.ValidationError as error:
    assert "form" in str(error).lower() or "test" in str(error).lower()
else:
    raise AssertionError("wrong conjugation acquired a Plan")
del result,restored,output,array
gc.collect()
np.testing.assert_allclose(view,np.full(5,1+3j),rtol=0,atol=1e-10)
"#), Some(&locals), Some(&locals))
    })
}

fn public_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let native = pyo3::wrap_pymodule!(_eqiora::_eqiora)(py);
    let package_directory = Path::new(env!("CARGO_MANIFEST_DIR"))
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
assert spec is not None and spec.loader is not None
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
        .expect("the package loader must bind eqiora")
        .cast_into::<PyModule>()?)
}
