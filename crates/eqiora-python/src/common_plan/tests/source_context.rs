use super::*;

#[test]
fn live_package_context_keeps_harmonic_effective_and_source_models_distinct() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let native = pyo3::wrap_pymodule!(crate::_eqiora)(py);
        let package_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bindings/python/python/eqiora")
            .canonicalize()?;
        let source = include_str!("../../../../../docs/language/harmonic-rc.md")
            .split_once("```eqiora\n")
            .unwrap()
            .1
            .split_once("```")
            .unwrap()
            .0;
        let locals = PyDict::new(py);
        locals.set_item("native", native.bind(py))?;
        locals.set_item("package_directory", package_directory.to_string_lossy())?;
        locals.set_item("source", source)?;
        py.run(c_str!(r#"
import importlib.util, pathlib, sys
package_path = pathlib.Path(package_directory)
spec = importlib.util.spec_from_file_location("eqiora", package_path / "__init__.py", submodule_search_locations=[str(package_path)])
eqiora = importlib.util.module_from_spec(spec)
sys.modules["eqiora"] = eqiora
sys.modules["eqiora._eqiora"] = native
spec.loader.exec_module(eqiora)
model = eqiora.compile(source="import Eqiora.Electrical.Basic.basic as electrical;\n" + source, entry="RC")
assert model.package_compilation_digest is not None
linear = eqiora.solve.Linear(
    algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
    preconditioner=eqiora.solve.Preconditioner.Identity,
    reduction=eqiora.solve.Reduction.Reproducible,
    provider=eqiora.solve.SolverProvider.reference(),
    relative_tolerance=1e-14, absolute_tolerance=1e-16, maximum_iterations=32,
)
plan = eqiora.resolve(model, solve=linear)
assert plan.model.digest == plan.model_digest != model.digest
assert plan.harmonic_original_model is model
assert plan.model.package_compilation_digest is None
assert plan.package_compilation_digest == model.package_compilation_digest
assert all(field.model_digest == plan.model_digest for field in plan.fields)
run = eqiora.submit(plan, state=eqiora.State.initial(plan))
assert run.package_compilation_digest == model.package_compilation_digest
assert run.result().plan_key == plan.identity
portable = eqiora.Plan.from_bytes(plan.to_bytes())
assert portable.identity == plan.identity
assert portable.model.digest == plan.model_digest
assert portable.harmonic_original_model.digest == model.digest
assert portable.harmonic_original_model.package_compilation_digest is None
assert portable.package_compilation_digest is None
"#), Some(&locals), Some(&locals))?;
        let plan = locals
            .get_item("plan")?
            .unwrap()
            .extract::<Py<crate::common_plan::PyPlan>>()?;
        let effective = plan.borrow(py).model_handle(py);
        let native = plan.borrow(py).native().clone();
        let bare = crate::common_plan::PyPlan::from_native_artifact(py, native)?;
        let error = bare.with_source_model(py, effective).unwrap_err();
        assert!(error.to_string().contains("source Model differs"));
        Ok(())
    })
}
