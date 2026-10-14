mod support;
use support::public_module;

use eqiora::geometry::{CanonicalGeometryLimits, CanonicalGeometryV1};
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyBytes, PyDict, PyDictMethods};

const EXPECTED: &[u8] = br#"{"schema":"eqiora.planar-circular-hole-envelope/v2","encoding":"eqiora.canonical-json/v1","kind":"axis-aligned-rectangle-with-circular-hole-v2","length_unit":"metre","bounds":[[0.0,2.2],[0.0,0.41]],"circle":{"center":[0.2,0.2],"radius_m":0.05},"entity_sets":[{"name":"cylinder","dimension":1,"members":[4]},{"name":"inlet","dimension":1,"members":[0]},{"name":"outlet","dimension":1,"members":[1]},{"name":"walls","dimension":1,"members":[2,3]},{"name":"fluid","dimension":2,"members":[0]}]}"#;

#[test]
fn python_exact_circular_hole_geometry_replays_rust_owned_identity() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", module)?;
        locals.set_item("expected_json", PyBytes::new(py, EXPECTED))?;
        py.run(
            c_str!(
                r#"
def make(**overrides):
    arguments = {
        "bounds": ((0.0, 2.2), (0.0, 0.41)),
        "circle_center": (0.2, 0.2),
        "circle_radius": 0.05,
        "region": "fluid",
        "x_lower": "inlet",
        "x_upper": "outlet",
        "y_lower": "walls",
        "y_upper": "walls",
        "hole": "cylinder",
    }
    arguments.update(overrides)
    x_bounds, y_bounds = arguments["bounds"]
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=x_bounds, y_bounds=y_bounds)
    circle = graph.circle(
        center=arguments["circle_center"], radius=arguments["circle_radius"]
    )
    fluid = graph.subtract(rectangle, circle)
    named_topology = {}
    for name, handle in (
        (arguments["region"], fluid.region),
        (arguments["x_lower"], rectangle.boundaries[0]),
        (arguments["x_upper"], rectangle.boundaries[1]),
        (arguments["y_lower"], rectangle.boundaries[2]),
        (arguments["y_upper"], rectangle.boundaries[3]),
        (arguments["hole"], circle.boundaries[0]),
    ):
        named_topology.setdefault(name, []).append(handle)
    return graph.build(
        fluid,
        named_topology=named_topology,
    )

geometry = make()
assert type(geometry).__module__ == "eqiora._eqiora"
assert type(geometry).__name__ == "Geometry"
assert geometry.canonical_bytes == expected_json
assert geometry.selection_names == (
    "cylinder", "inlet", "outlet", "walls", "fluid"
)
assert tuple(geometry.selection_dimension(name) for name in geometry.selection_names) == (
    1, 1, 1, 1, 2
)

same = make()
signed_zero = make(bounds=((-0.0, 2.2), (-0.0, 0.41)))
swapped = make(x_lower="outlet", x_upper="inlet")
assert geometry == same == signed_zero
assert hash(geometry) == hash(same) == hash(signed_zero)
assert geometry != swapped
assert geometry.digest != swapped.digest

oriented = make(y_lower="floor", y_upper="ceiling")
assert oriented.selection_names == (
    "ceiling", "cylinder", "floor", "inlet", "outlet", "fluid"
)

off_axis = make(circle_center=(0.3, 0.2))
assert off_axis != geometry

try:
    geometry.selection_dimension("missing")
except eqiora.ValidationError as error:
    assert error.category == "validation"
    assert error.diagnostics
else:
    raise AssertionError("an unknown exact selection returned a value")

canonical_json = geometry.canonical_bytes
python_digest = geometry.digest
"#
            ),
            Some(&locals),
            Some(&locals),
        )?;

        let canonical_json = locals
            .get_item("canonical_json")?
            .expect("Python geometry must expose canonical bytes")
            .extract::<Vec<u8>>()?;
        let python_digest = locals
            .get_item("python_digest")?
            .expect("Python geometry must expose its digest")
            .extract::<String>()?;
        let replayed = CanonicalGeometryV1::decode_circular_hole_canonical(
            &canonical_json,
            CanonicalGeometryLimits::default(),
        )
        .expect("Python bytes must replay through the public Rust geometry contract");
        let rust_digest = replayed
            .digest_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(python_digest, rust_digest);
        Ok(())
    })
}
