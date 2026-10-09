use eqiora::artifact::{GmshMeshPolicyV1, ModelEnvelope};
use eqiora::compiler::{CompiledModel, StaticBindingValue};
use eqiora::geometry::{CanonicalGeometryV1, NamedEntitySet};
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::sem::KernelProgram;
use eqiora_numerics::AuthenticatedCommonMesh;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyDictMethods, PyModule};
use std::path::Path;

fn fixture(face: bool, complex: bool, permuted: bool) -> (Vec<u8>, Vec<u8>) {
    let geometry = CanonicalGeometryV1::from_convex_polyhedra(
        vec![[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.]],
        vec![vec![
            vec![0, 2, 1],
            vec![0, 1, 3],
            vec![1, 2, 3],
            vec![2, 0, 3],
        ]],
        vec![
            NamedEntitySet::new("body", 3, vec![0]),
            NamedEntitySet::new("wall", 2, vec![0, 1, 2, 3]),
        ],
        1e-12,
    )
    .unwrap();
    let operator = if face {
        "-grad(div(u))"
    } else {
        "curl(curl(u))"
    };
    let boundary = if face {
        "normal(a*isotropic_lift(div(u)))"
    } else {
        "tangential_trace(-a*curl(u))"
    };
    let scalar = if complex { "complex<m>" } else { "m" };
    let vector = if complex {
        "vector<complex<1>,3>"
    } else {
        "vector<1,3>"
    };
    let potential = "2*coordinate(0)+3*coordinate(1)+4*coordinate(2)";
    let potential = if complex {
        format!("math.complex({potential},2*({potential}))")
    } else {
        potential.to_owned()
    };
    let source = format!("public component Flux(support body:volume(ambient_dimension=3), support wall:boundary(parent=body)) {{
        parameter a:m^2=2[m^2]; variable u:{vector} on body;
        variable potential:{scalar} on body;
        relation prescribed on body {{ potential={potential}; }}
        relation balance on body {{ a*({operator})+u=grad(potential); }}
        relation law on wall {{ {boundary}=0; }}
    }}");
    let body = geometry.entity_set("body").unwrap();
    let compiled = CompiledModel::compile_selected(
        "moments.eqi",
        &source,
        "Flux",
        &[
            (
                "body",
                StaticBindingValue::GeometrySupport {
                    geometry: &geometry,
                    selection: body,
                    parent: None,
                },
            ),
            (
                "wall",
                StaticBindingValue::GeometrySupport {
                    geometry: &geometry,
                    selection: geometry.entity_set("wall").unwrap(),
                    parent: Some(body),
                },
            ),
        ],
    )
    .unwrap();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let program =
        KernelProgram::from_snapshot_with_geometry(&store.snapshot(), model, &[&geometry]).unwrap();
    let model = ModelEnvelope::from_program(&program)
        .unwrap()
        .canonical_json()
        .unwrap();
    // Synthetic MSH input tests authenticated import/replay, not Gmsh generation.
    let cell = if permuted { "2 1 4 3" } else { "1 2 3 4" };
    let msh = format!(
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 4 1 4\n3 1 0 4\n1\n2\n3\n4\n0 0 0\n2 0 0\n0 3 0\n0 0 4\n$EndNodes\n$Elements\n1 1 1 1\n3 1 4 1\n1 {cell}\n$EndElements\n"
    );
    let mesh = AuthenticatedCommonMesh::gmsh_4152(
        geometry,
        GmshMeshPolicyV1::explicit(1e-12, 0.01, 8, 5.).unwrap(),
        msh.into_bytes(),
    )
    .unwrap();
    (model, mesh.to_bytes().unwrap())
}

#[test]
fn python_oriented_moments_execute_and_replay_without_nodal_reinterpretation() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", public_module(py)?)?;
        for face in [false, true] {
            for complex in [false, true] {
                for permuted in [false, true] {
                    let (model, mesh) = fixture(face, complex, permuted);
                    locals.set_item("model_bytes", PyBytes::new(py, &model))?;
                    locals.set_item("mesh_bytes", PyBytes::new(py, &mesh))?;
                    locals.set_item("face", face)?;
                    locals.set_item("is_complex", complex)?;
                    py.run(c_str!(r#"
model = eqiora.Model.from_bytes(model_bytes)
mesh = eqiora.meshing.Mesh.from_bytes(mesh_bytes)
policy = eqiora.fem.TetrahedralFace() if face else eqiora.fem.TetrahedralEdge()
association = "face" if face else "edge"
assert policy.space == "tetrahedral-" + association
assert policy == type(policy)() and hash(policy) == hash(type(policy)())
linear = eqiora.solve.Linear(algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
    preconditioner=eqiora.solve.Preconditioner.Identity,
    reduction=eqiora.solve.Reduction.Reproducible,provider=eqiora.solve.SolverProvider.reference(),
    relative_tolerance=1e-13,absolute_tolerance=1e-14,maximum_iterations=2000)
plan = eqiora.resolve(model,mesh=mesh,spatial=policy,solve=linear)
assert plan.spatial == policy
assert isinstance(plan.capability, eqiora.LinearPlanView)
assert plan.capability.kind == "linear"
assert not hasattr(eqiora, "ScalarPlanView")
assert plan.capability.coefficient_sampling == "quadrature-point"
plan_wire = plan.to_bytes()
assert b'"family":"linear"' in plan_wire
plan = eqiora.Plan.from_bytes(plan_wire)
assert plan.to_bytes() == plan_wire and plan.spatial == policy
result = eqiora.run(plan)
field = plan.capability.fields[0]
entities = plan.field_coefficient_entities(field)
assert entities == tuple((2 if face else 1,i) for i in range(4 if face else 6))
derivative = plan.field_exterior_derivative(field)
if face:
    assert derivative == {(3,0): (((2,0),-1),((2,1),1),((2,2),-1),((2,3),1))}
    try:
        plan.field_gradient_modes(field)
    except eqiora.ValidationError:
        pass
    else:
        raise AssertionError("face flux coefficients were interpreted as edge gradients")
else:
    modes = plan.field_gradient_modes(field)
    assert modes == {
        (0,1): (((1,0),1),((1,3),-1),((1,4),-1)),
        (0,2): (((1,1),1),((1,3),1),((1,5),-1)),
        (0,3): (((1,2),1),((1,4),1),((1,5),1)),
    }
    assert len(derivative) == 4
    for column in modes.values():
        coefficients = dict(column)
        for row in derivative.values():
            assert sum(sign*coefficients.get(entity,0) for entity,sign in row) == 0
if "previous_field" in globals() and previous_field.model_digest != field.model_digest:
    for inspect in (plan.field_coefficient_entities, plan.field_exterior_derivative, plan.field_gradient_modes):
        try:
            inspect(previous_field)
        except TypeError:
            pass
        else:
            raise AssertionError("coefficient inspection crossed Model identity")
previous_field = field
# Mutating a returned dictionary cannot alter the retained Plan topology.
derivative.clear()
assert plan.field_exterior_derivative(field)
# Canonically ascending vertex orientations. Integrate (2,3,4) along
# each edge or dot it with half the oriented face cross product.
expected = [12.,-12.,12.,36.] if face else [4.,9.,16.,5.,12.,7.]
if is_complex:
    expected = [v*(1+2j) for v in expected]
for output in (result.output(field), eqiora.Result.from_bytes(plan,result.to_bytes()).output(field)):
    assert output.space == policy.space and output.associations == (association,)
    assert output.value_shape == (3,)
    assert output.dimension == (0,0,0,0,0,0,0)
    assert output.coefficient_dimension == (0,2 if face else 1,0,0,0,0,0)
    assert output.coefficient_count(association) == len(expected)
    assert output.logical_shape(association) == (len(expected),)
    values = output.values(association)
    assert values.shape == (len(expected),)
    assert values.dtype == ("complex128" if is_complex else "float64")
    assert all(abs(values[i]-v) <= 1e-9*max(1,abs(v)) for i,v in enumerate(expected))
assert eqiora.Result.from_bytes(plan,result.to_bytes()).to_bytes() == result.to_bytes()
try:
    eqiora.resolve(model,mesh=mesh,spatial=eqiora.fem.Q1(),solve=linear)
except eqiora.ValidationError:
    pass
else:
    raise AssertionError("moment vector admitted a nodal policy substitution")
"#), Some(&locals), Some(&locals))?;
                }
            }
        }
        Ok(())
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
