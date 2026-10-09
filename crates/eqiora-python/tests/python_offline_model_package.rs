use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, NormalizedRelativePath, PackageManifestV1,
    PackageSourcesV1, QualifiedName, ResolutionRecordV1, SourceFileV1, prepare_package_release_v1,
};
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule};

const SOURCE: &str = r#"
public property contract Diffusivity(): 1 { derivatives value_only; }
public property release ReferenceDiffusivity: Diffusivity {
  analytic { value = 25;
  source_unit: 1 = 1 / 1000; }
  validity unconditional; outside reject; branch single;
  citation org.example.measurement;
  license spdx.CC0_1_0;
}

public component PoissonLaw(
  support region: volume(ambient_dimension = 2),
  support left: boundary(parent = region),
  support right: boundary(parent = region),
  support bottom: boundary(parent = region),
  support top: boundary(parent = region),
  parameter wave_number: 1 / m,
  parameter source_scale: 1 / m ^ 2,
  property diffusivity: Diffusivity
) {

  variable potential: 1 on region in h1;
  relation balance on region {
    -div(diffusivity * grad(potential))
      - source_scale * math.sin(wave_number * coordinate(0))
        * math.sin(wave_number * coordinate(1)) = 0;
  }
  relation left_value on left { trace(potential) = 0; }
  relation right_value on right { trace(potential) = 0; }
  relation bottom_value on bottom { trace(potential) = 0; }
  relation top_value on top { trace(potential) = 0; }
}

public component PoissonRectangle(
  support region: volume(ambient_dimension = 2),
  support left: boundary(parent = region),
  support right: boundary(parent = region),
  support bottom: boundary(parent = region),
  support top: boundary(parent = region),
  parameter wave_number: 1 / m,
  parameter source_scale: 1 / m ^ 2
) {
  instance equation: PoissonLaw(
    region = region,
    left = left,
    right = right,
    bottom = bottom,
    top = top,
    wave_number = wave_number,
    source_scale = source_scale,
    diffusivity = ReferenceDiffusivity
  );
}
"#;

static SCRATCH_SERIAL: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);

impl Scratch {
    fn create() -> Self {
        let serial = SCRATCH_SERIAL.fetch_add(1, Ordering::Relaxed);
        let root = std::env::var_os("HOME")
            .map(PathBuf::from)
            .expect("HOME-backed test scratch")
            .join(".cache/eqiora/python-package-geometry-tests")
            .join(format!("{}-{serial}", std::process::id()));
        fs::create_dir_all(&root).expect("create package test scratch");
        Self(root)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn locked_store(source: &str) -> (Scratch, Vec<u8>, String) {
    let path = NormalizedRelativePath::parse("src/poisson.eqi").expect("source path");
    let manifest = PackageManifestV1::new(
        "poisson",
        QualifiedName::parse("org.example.geometry_poisson").expect("package name"),
        ExactVersion::parse("1.0.0").expect("version"),
        vec![],
        vec![BundleEntryV1::new(path.clone(), BundleRoleV1::ModelSource)],
    )
    .expect("manifest");
    let sources = PackageSourcesV1::new(
        manifest,
        vec![SourceFileV1::new(
            path,
            BundleRoleV1::ModelSource,
            source.as_bytes().to_vec(),
        )],
    )
    .expect("closed package sources");
    let release = prepare_package_release_v1(sources, &[]).expect("prepared package release");
    let identity = release.package_identity().expect("exact package identity");
    let namespace = format!(
        "{}/{}/{}",
        identity.name,
        identity.version,
        identity.semantic_digest.to_hex()
    );
    let resolution = ResolutionRecordV1::from_exact_releases(&release, &[])
        .expect("exact package resolution")
        .canonical_json()
        .expect("canonical resolution");
    let scratch = Scratch::create();
    let digest = release.source_digest().expect("source digest");
    fs::write(
        scratch.0.join(format!("{digest}.json")),
        release.canonical_json().expect("canonical release"),
    )
    .expect("publish exact release into selected store");
    (scratch, resolution, namespace)
}

#[test]
fn local_package_project_locks_and_compiles_a_model_through_python() -> PyResult<()> {
    let scratch = Scratch::create();
    let project_root = scratch.0.join("project");
    let package_root = project_root.join("root");
    let store_root = scratch.0.join("store");
    fs::create_dir_all(package_root.join("src")).expect("create package source directory");
    fs::create_dir(&store_root).expect("create package store");
    let path = NormalizedRelativePath::parse("src/main.eqi").expect("source path");
    fs::write(
        package_root.join(path.as_str()),
        "public model Main() { parameter gain: 1 = 2; relation law { gain - 2 = 0; } }",
    )
    .expect("write package source");
    fs::write(
        project_root.join("eqiora.toml"),
        "[package]\nname = \"org.example.LocalRoot\"\nversion = \"1.0.0\"\nsource = \"root/src\"\nentry = \"main\"\n",
    )
    .expect("write project manifest");
    fs::create_dir_all(scratch.0.join("library/src")).unwrap();
    fs::write(
        scratch.0.join("library/src/main.eqi"),
        "public model Shared() { parameter gain: 1 = 2; relation law { gain - 2 = 0; } }",
    )
    .unwrap();
    fs::write(
        scratch.0.join("library/eqiora.toml"),
        "[package]\nname = \"org.example.Library\"\nversion = \"1.0.0\"\nentry = \"main\"\n",
    )
    .unwrap();

    Python::initialize();
    Python::attach(|py| {
        let public = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", public)?;
        locals.set_item("project", project_root.to_string_lossy())?;
        locals.set_item("store", store_root.to_string_lossy())?;
        py.run(
            c_str!(
                r#"
resolution = eqiora.resolve_local_project(project, store)
assert type(resolution) is bytes
assert eqiora.open_project(project, store) == resolution
model = eqiora.compile_package(store, resolution, entry='Main')
assert model.package_compilation_digest is not None
assert len(model.parameter_ids) == 1
before_manifest = open(project + "/eqiora.toml", "rb").read()
before_lock = open(project + "/eqiora.lock", "rb").read()
try:
    eqiora.add_local_dependency(project, store, "org.example.Library", version="2.0.0", path="../library")
except eqiora.CompatibilityError:
    pass
else:
    raise AssertionError("wrong exact version accepted")
assert open(project + "/eqiora.toml", "rb").read() == before_manifest
assert open(project + "/eqiora.lock", "rb").read() == before_lock
assert eqiora.open_project(project, store) == resolution
added = eqiora.add_local_dependency(project, store, "org.example.Library", version="1.0.0", path="../library")
assert added != resolution
assert eqiora.open_project(project, store) == added
eqiora.compile_package(store, added, entry='Main')
from pathlib import Path
source_path = Path(project) / "root/src/main.eqi"
original_source = source_path.read_text()
source_path.write_text("import org.example.Library.main as library; " + original_source)
imported = eqiora.resolve_local_project(project, store)
eqiora.compile_package(store, imported, entry='library.Shared')
source_path.write_text(original_source)
removed = eqiora.remove_local_dependency(project, store, "org.example.Library")
assert removed == resolution
assert eqiora.open_project(project, store) == removed
"#
            ),
            None,
            Some(&locals),
        )
    })
}

#[test]
fn package_component_uses_caller_geometry_common_plan_and_run() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let public = public_module(py)?;
        let author_locals = PyDict::new(py);
        author_locals.set_item("eqiora", public)?;
        py.run(
            c_str!(
                r#"
q = eqiora.lang
u = eqiora.units
source = eqiora.Module('main')
contract = source.property_contract("Diffusivity", value_type=eqiora.ValueType.real())
release = source.property_release(
    "ReferenceDiffusivity",
    implements=contract,
    value=25,
    source_unit=u.one,
    source_scale=0.001,
    citation="org.example.measurement",
    license="spdx.CC0_1_0",
)
law = source.component("PoissonLaw")
region = law.volume("region", dimensions=2)
left = law.boundary("left", parent=region)
right = law.boundary("right", parent=region)
bottom = law.boundary("bottom", parent=region)
top = law.boundary("top", parent=region)
source_scale = law.parameter("source_scale", value_type=eqiora.ValueType.real(eqiora.Dimension(length=-2)))
diffusivity = law.property("diffusivity", contract=contract)
potential = law.field("potential", role=eqiora.FieldRole.Variable, on=region, spatial_regularity=eqiora.SpatialRegularity.H1, value_type=eqiora.ValueType.real())
law.relation("balance", eqiora.lang.equation(-q.div(diffusivity * q.grad(potential)) - source_scale, 0), on=region)
law.relation("left_value", eqiora.lang.equation(q.trace(potential), 0), on=left)
law.relation("right_value", eqiora.lang.equation(q.trace(potential), 0), on=right)
law.relation("bottom_value", eqiora.lang.equation(q.trace(potential), 0), on=bottom)
law.relation("top_value", eqiora.lang.equation(q.trace(potential), 0), on=top)

root = source.component("PoissonRectangle")
root_region = root.volume("region", dimensions=2)
root_left = root.boundary("left", parent=root_region)
root_right = root.boundary("right", parent=root_region)
root_bottom = root.boundary("bottom", parent=root_region)
root_top = root.boundary("top", parent=root_region)
root_source_scale = root.parameter("source_scale", value_type=eqiora.ValueType.real(eqiora.Dimension(length=-2)))
root.instance('equation', component=law, bindings={'region': root_region, 'left': root_left, 'right': root_right, 'bottom': root_bottom, 'top': root_top, 'source_scale': root_source_scale, 'diffusivity': release})
authored_source = source.to_eqi()
"#
            ),
            None,
            Some(&author_locals),
        )?;
        let authored_source = author_locals
            .get_item("authored_source")?
            .expect("authored source")
            .extract::<String>()?;
        let (store, resolution, namespace) = locked_store(&authored_source);
        let native = pyo3::wrap_pymodule!(_eqiora::_eqiora)(py);
        let locals = PyDict::new(py);
        locals.set_item("package", native.bind(py))?;
        locals.set_item(
            "store",
            store.0.to_str().expect("Unicode HOME-backed scratch"),
        )?;
        locals.set_item("resolution", PyBytes::new(py, &resolution))?;
        locals.set_item("namespace", namespace)?;
        py.run(
            c_str!(
                r#"
graph = package.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
geometry = graph.build(rectangle, named_topology={
    "region": rectangle.region,
    "left": rectangle.boundaries[0],
    "right": rectangle.boundaries[1],
    "bottom": rectangle.boundaries[2],
    "top": rectangle.boundaries[3],
})
model = package.compile_package(store, resolution, geometry=geometry, entry='PoissonRectangle', bindings={'region': geometry.selection('region'), 'left': (geometry.selection('left'), geometry.selection('region')), 'right': (geometry.selection('right'), geometry.selection('region')), 'bottom': (geometry.selection('bottom'), geometry.selection('region')), 'top': (geometry.selection('top'), geometry.selection('region')), **{'source_scale': 1.0}})
assert model.package_compilation_digest is not None
assert model.domain_ids
assert isinstance(model.property_bindings, tuple)
assert len(model.property_bindings) == 1
binding = model.property_bindings[0]
assert binding.composition is None
assert binding.contract == f"{namespace}::poisson::Diffusivity"
assert binding.release == f"{namespace}::poisson::ReferenceDiffusivity"
assert binding.component == f"{namespace}::poisson::PoissonLaw"
assert binding.requirement == "diffusivity"
assert binding.normalized_value == 0.025
assert binding.validity == "unconditional"
assert binding.citation == "org.example.measurement"
assert binding.license == "spdx.CC0_1_0"
try:
    package.PropertyBinding()
except TypeError:
    pass
else:
    raise AssertionError("Python forged a package-owned property binding")
try:
    binding.citation = "org.example.rewritten"
except AttributeError:
    pass
else:
    raise AssertionError("Python mutated a package-owned property binding")

mesher = package.CartesianMesher(cells=(4, 4))
mesh_plan = package.resolve(geometry, mesher)
mesh = package.generate(mesh_plan)
linear = package.Linear(
    algorithm=package.LinearSolver.BiConjugateGradientStabilized,
    preconditioner=package.Preconditioner.Identity,
    reduction=package.Reduction.Reproducible,
    provider=package.SolverProvider.reference(),
    relative_tolerance=1e-10,
    absolute_tolerance=1e-12,
    maximum_iterations=10000,
)
plan = package._resolve_plan(model, mesh=mesh, spatial=package.Q1(), solve=linear)
assert plan.model is model
assert plan.mesh is mesh
assert plan.package_compilation_digest == model.package_compilation_digest
run = package.submit_plan(plan)
assert run.package_compilation_digest == model.package_compilation_digest
result = run.result()
assert result.model_digest == model.digest
assert model.property_bindings[0].citation == "org.example.measurement"

replayed = package.Model.from_bytes(model.to_bytes())
assert len(replayed.property_bindings) == 1
retained = replayed.property_bindings[0]
for attribute in (
    "composition", "contract", "release", "component", "requirement",
    "normalized_value", "validity", "citation", "license",
):
    assert getattr(retained, attribute) == getattr(binding, attribute), attribute
replayed_plan = package._resolve_plan(replayed, mesh=mesh, spatial=package.Q1(), solve=linear)
assert replayed_plan.identity == plan.identity
assert replayed_plan.package_compilation_digest is None

wide_rectangle = graph.rectangle(x_bounds=(0.0, 2.0), y_bounds=(0.0, 1.0))
wide_geometry = graph.build(wide_rectangle, named_topology={
    "region": wide_rectangle.region,
    "left": wide_rectangle.boundaries[0],
    "right": wide_rectangle.boundaries[1],
    "bottom": wide_rectangle.boundaries[2],
    "top": wide_rectangle.boundaries[3],
})
wide_mesh_plan = package.resolve(wide_geometry, mesher)
wide_mesh = package.generate(wide_mesh_plan)
try:
    package._resolve_plan(replayed, mesh=wide_mesh, spatial=package.Q1(), solve=linear)
except package.ValidationError:
    pass
else:
    raise AssertionError("replayed bound Model crossed into a foreign Geometry/Mesh lineage")

foreign_rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
foreign = graph.build(foreign_rectangle, named_topology={
    "other": foreign_rectangle.region,
    "left": foreign_rectangle.boundaries[0],
    "right": foreign_rectangle.boundaries[1],
    "bottom": foreign_rectangle.boundaries[2],
    "top": foreign_rectangle.boundaries[3],
})
try:
    package.compile_package(store, resolution, geometry=foreign, entry='PoissonRectangle', bindings={'region': foreign.selection('region'), 'left': (foreign.selection('left'), foreign.selection('region')), 'right': (foreign.selection('right'), foreign.selection('region')), 'bottom': (foreign.selection('bottom'), foreign.selection('region')), 'top': (foreign.selection('top'), foreign.selection('region')), **{'source_scale': 1.0}})
except package.ValidationError as error:
    assert "region" in error.diagnostics[0].message
else:
    raise AssertionError("equal bounds bypassed exact support-name binding")
"#
            ),
            None,
            Some(&locals),
        )
    })
}

#[test]
fn package_compile_argument_and_resolution_boundaries_fail_closed() -> PyResult<()> {
    let (store, resolution, _) = locked_store(SOURCE);
    Python::initialize();
    Python::attach(|py| {
        let native = pyo3::wrap_pymodule!(_eqiora::_eqiora)(py);
        let locals = PyDict::new(py);
        locals.set_item("package", native.bind(py))?;
        locals.set_item("store", store.0.to_str().expect("Unicode scratch"))?;
        locals.set_item("resolution", PyBytes::new(py, &resolution))?;
        py.run(
            c_str!(
                r#"
graph = package.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
geometry = graph.build(rectangle, named_topology={
    "region": rectangle.region,
    "left": rectangle.boundaries[0],
    "right": rectangle.boundaries[1],
    "bottom": rectangle.boundaries[2],
    "top": rectangle.boundaries[3],
})
for args, kwargs in (
    ((store, bytearray(resolution)), dict(entry="PoissonRectangle")),
    ((store, resolution), dict(geometry=geometry)),
    ((store, resolution), dict(entry="Main", component="PoissonRectangle")),
    ((store, resolution), dict(entry="Main", parameters={})),
):
    try:
        package.compile_package(*args, **kwargs)
    except TypeError:
        pass
    else:
        raise AssertionError("invalid package compile arguments were admitted")

try:
    package.compile_package(store, b' ' + resolution, geometry=geometry, entry='PoissonRectangle', bindings={'region': geometry.selection('region'), 'left': (geometry.selection('left'), geometry.selection('region')), 'right': (geometry.selection('right'), geometry.selection('region')), 'bottom': (geometry.selection('bottom'), geometry.selection('region')), 'top': (geometry.selection('top'), geometry.selection('region')), **{}})
except package.CompatibilityError:
    pass
else:
    raise AssertionError("noncanonical resolution bytes were admitted")
"#
            ),
            None,
            Some(&locals),
        )
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

#[test]
fn typed_property_projection_preserves_scaled_complex_channels() -> PyResult<()> {
    let (store, resolution, _) = locked_store(
        r#"
public property contract Response(): array<complex<1>, 2> { derivatives value_only; }
public property release Reference: Response {
  analytic { value = [math.complex(1, 2), math.complex(3, -4)];
  source_unit: 1 = 2; }
  validity unconditional; outside reject; branch single;
  citation org.example.measurement;
  license spdx.CC0_1_0;
}
public component Consumer(
  property response: Response
) {
  variable x: complex<1>;
  relation law { x = response[1]; }
}
public model Main() {
  instance consumer: Consumer(response = Reference);
}
"#,
    );
    Python::initialize();
    Python::attach(|py| {
        let native = pyo3::wrap_pymodule!(_eqiora::_eqiora)(py);
        let locals = PyDict::new(py);
        locals.set_item("eqiora", native.bind(py))?;
        locals.set_item("store", store.0.to_str().expect("Unicode scratch"))?;
        locals.set_item("resolution", PyBytes::new(py, &resolution))?;
        py.run(
            c_str!(
                r#"
model = eqiora.compile_package(store, resolution, entry='Main')
(binding,) = model.property_bindings
assert binding.normalized_value == (2 + 4j, 6 - 8j)
assert binding.value_type == eqiora.ValueType.array(eqiora.ValueType.complex(), 2)
assert model.package_compilation_digest is not None
"#
            ),
            Some(&locals),
            None,
        )
    })
}
