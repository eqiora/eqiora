use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyModule};
use std::path::Path;

#[test]
fn python_hermitian_plan_run_result_and_replay() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", &module)?;
        py.run(c_str!(r#"
model = eqiora.compile(source="""
space Spin=orthonormal(up,down);
model Quantum() {
 parameter h:map<complex<J>,Spin,Spin>=linear_map(Spin,Spin,[[2[J],math.complex(0[J],-1[J])],[math.complex(0[J],1[J]),2[J]]]);
 variable u:coordinates<complex<1>,Spin>;
 variable lambda:J;
 relation states {apply(h,u)=lambda*u;}
}
""")
controls = dict(count=2, provider=eqiora.solve.SolverProvider.faer(), residual_tolerance=1e-12, normalization_tolerance=1e-12)
plan = eqiora.resolve(model, solve=eqiora.solve.HermitianEigen(**controls))
assert plan.formulation.effective == eqiora.FormulationKind.FiniteHermitianPencil
assert plan.capability.mode_field == model.field("u")
assert plan.capability.eigenvalue_field == model.field("lambda")
assert plan.fields == (model.field("u"), model.field("lambda"))
assert plan.solve.count == plan.requested_solve.count == 2
restored = eqiora.Plan.from_bytes(plan.to_bytes())
assert restored.to_bytes() == plan.to_bytes()
assert restored.solve.provider == plan.solve.provider
result = eqiora.run(restored)
assert result.eigen_convergence == "converged"
assert result.eigenpair_count == 2
assert result.eigen_candidate_counts == (2, 0)
for i, expected in enumerate((1., 3.)):
    pair = result.eigenpair(i)
    assert isinstance(pair, eqiora.Eigenpair)
    assert abs(pair.eigenvalue - expected) < 1e-12
    assert pair.relative_residual < 1e-12 and pair.normalization_defect < 1e-12
    assert pair.mode_field == model.field("u")
    assert pair.eigenvalue_field == model.field("lambda")
    assert any(abs(complex(v).imag) > 0 for v in pair.mode)
projector, projector_type = result.eigenprojector([0, 1])
for row in range(2):
    for col in range(2):
        assert abs(projector[row][col] - (1 if row == col else 0)) < 1e-12
replayed = eqiora.Result.from_bytes(restored, result.to_bytes())
assert replayed.to_bytes() == result.to_bytes()
assert replayed.eigen_convergence == "converged"
dimension = result.eigenpair(0).eigenvalue_type.dimension
partial_plan = eqiora.resolve(model, solve=eqiora.solve.HermitianEigen(**controls, interval=((0., dimension), (2., dimension))))
assert partial_plan.solve.interval == ((0., dimension), (2., dimension))
assert eqiora.run(partial_plan).eigen_convergence == "partial"
nearest = dict(controls, count=1)
target_plan = eqiora.resolve(model, solve=eqiora.solve.HermitianEigen(**nearest, target=(2.75, dimension)))
assert abs(eqiora.run(target_plan).eigenpair(0).eigenvalue - 3.) < 1e-12
for action in (
    lambda: eqiora.resolve(model, solve=eqiora.solve.HermitianEigen(**dict(controls, provider=eqiora.solve.SolverProvider.reference()))),
    lambda: eqiora.run(plan, until_s=1.),
    lambda: eqiora.State.initial(plan),
    lambda: eqiora.Result.from_bytes(partial_plan, result.to_bytes()),
    lambda: result.eigenprojector([0, 0]),
):
    try:
        action()
    except (eqiora.EqioraError, TypeError, ValueError):
        pass
    else:
        raise AssertionError("unsupported or inconsistent spectral request must reject")
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
