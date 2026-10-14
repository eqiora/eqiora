mod support;
use support::public_module;

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

#[test]
fn python_geometry_graph_exposes_direct_handles_and_atomic_naming() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", module)?;
        py.run(
            c_str!(
                r#"
graph = eqiora.geometry.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.0, 2.2), y_bounds=(0.0, 0.41))
circle = graph.circle(center=(0.2, 0.2), radius=0.05)
fluid = graph.subtract(rectangle, circle)
assert rectangle.region.dimension == 2
assert len(rectangle.boundaries) == 4
assert len(circle.boundaries) == 1
assert len(fluid.boundaries) == 5
assert not hasattr(fluid, "face_handle")

geometry = graph.build(fluid, named_topology={
    "fluid": fluid.region,
    "inlet": rectangle.boundaries[0],
    "outlet": rectangle.boundaries[1],
    "walls": rectangle.boundaries[2:],
    "cylinder": circle.boundaries[0],
})
assert geometry.classification_tolerance is None
assert geometry.bounds == ((0.0, 2.2), (0.0, 0.41))
assert geometry.selection_names == ("cylinder", "inlet", "outlet", "walls", "fluid")

foreign_graph = eqiora.geometry.GeometryGraph()
foreign = foreign_graph.rectangle(x_bounds=(0.0, 2.2), y_bounds=(0.0, 0.41))
try:
    graph.build(fluid, named_topology={
        "fluid": fluid.region,
        "inlet": foreign.boundaries[0],
        "outlet": rectangle.boundaries[1],
        "walls": rectangle.boundaries[2:],
        "cylinder": circle.boundaries[0],
    })
except eqiora.ValidationError:
    pass
else:
    raise AssertionError("foreign direct handle reached Geometry publication")
"#
            ),
            None,
            Some(&locals),
        )
    })
}
