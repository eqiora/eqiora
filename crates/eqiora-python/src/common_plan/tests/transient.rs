use super::*;

#[test]
fn root_plan_resolves_and_runs_transient_flow_through_common_state() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let native = pyo3::wrap_pymodule!(crate::_eqiora)(py);
        let package_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bindings/python/python/eqiora")
            .canonicalize()?;
        let locals = PyDict::new(py);
        locals.set_item("native", native.bind(py))?;
        locals.set_item("package_directory", package_directory.to_string_lossy())?;
        let model = PyModel::from_document(
            py,
            ModelDocument::compile("transient-direct.eqi", TRANSIENT_SOURCE).unwrap(),
        )?;
        locals.set_item("model", Py::new(py, model)?)?;
        py.run(
                c_str!(r#"
import importlib.util, pathlib, sys, tempfile
package_path = pathlib.Path(package_directory)
spec = importlib.util.spec_from_file_location("eqiora", package_path / "__init__.py", submodule_search_locations=[str(package_path)])
package = importlib.util.module_from_spec(spec)
sys.modules["eqiora"] = package
sys.modules["eqiora._eqiora"] = native
spec.loader.exec_module(package)

graph = package.geometry.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
source = graph.build(rectangle, named_topology={
    "region": rectangle.region,
    "left": rectangle.boundaries[0],
    "right": rectangle.boundaries[1],
    "bottom": rectangle.boundaries[2],
    "top": rectangle.boundaries[3],
})
affine_plan = package.meshing.resolve(source, package.meshing.AffineTriangleMesher(cells=(2, 3)))
affine = package.meshing.generate(affine_plan)
cartesian_plan = package.meshing.resolve(source, package.meshing.CartesianMesher(cells=(4, 4)))
cartesian = package.meshing.generate(cartesian_plan)
linear = package.solve.Linear(
    algorithm=package.solve.LinearSolver.SparseLu,
    preconditioner=package.solve.Preconditioner.Identity,
    reduction=package.solve.Reduction.Fast,
    provider=package.solve.SolverProvider.faer(),
    relative_tolerance=1e-10,
    absolute_tolerance=1e-12,
    maximum_iterations=2000,
)
fvm_linear = package.solve.Linear(
    relative_tolerance=1e-10,
    absolute_tolerance=1e-12,
    maximum_iterations=2000,
    algorithm=package.solve.LinearSolver.BiConjugateGradientStabilized,
    preconditioner=package.solve.Preconditioner.Identity,
    reduction=package.solve.Reduction.Reproducible,
    provider=package.solve.SolverProvider.reference(),
)
fvm_newton = package.solve.Newton(linear=fvm_linear)
newton = package.solve.Newton(linear=linear)
custom_newton = package.solve.Newton(
    linear=linear,
    relative_tolerance=2e-9,
    absolute_tolerance=3e-11,
    maximum_iterations=19,
    maximum_line_search_steps=7,
)
temporal = package.time.BackwardEuler(0.01)
scaling = package.fluid.IncompressibleScaling(length_m=1.0, velocity_m_per_s=2.0, pressure_pa=3.0)

model_fingerprint = model.structural_fingerprint
model_bytes = model.to_bytes()
mini = package.resolve(model, mesh=affine, spatial=package.fem.MiniP1(), solve=newton, scaling=scaling, temporal=temporal)
fvm = package.resolve(model, mesh=cartesian, spatial=package.fvm.CellCentered(), solve=fvm_newton, scaling=scaling, temporal=temporal)
mini_bytes = mini.to_bytes()
portable_mini = package.Plan.from_bytes(mini_bytes)
fvm_bytes = fvm.to_bytes()
portable_fvm = package.Plan.from_bytes(fvm_bytes)
planned = {}
for name, objective in (
    ("robust", package.solve.Robust),
    ("fast", package.solve.Fast),
    ("low-memory", package.solve.LowMemory),
):
    planned_linear = package.solve.Linear(
        relative_tolerance=1e-10,
        absolute_tolerance=1e-12,
        maximum_iterations=2000,
        objective=objective,
    )
    planned[name] = package.resolve(
        model,
        mesh=cartesian,
        spatial=package.fvm.CellCentered(),
        solve=package.solve.Newton(linear=planned_linear),
        scaling=scaling,
        temporal=temporal,
    )
mini_exact = package.resolve(
    model, mesh=affine, spatial=package.fem.MiniP1(),
    formulation=package.formulation.MixedGalerkin,
    solve=newton, scaling=scaling, temporal=temporal,
)
fvm_exact = package.resolve(
    model, mesh=cartesian, spatial=package.fvm.CellCentered(),
    formulation=package.formulation.IntegralConservative,
    solve=fvm_newton, scaling=scaling, temporal=temporal,
)
replayed = package.resolve(package.Model.from_bytes(model.to_bytes()), mesh=affine, spatial=package.fem.MiniP1(), solve=newton, scaling=scaling, temporal=temporal)
custom = package.resolve(model, mesh=affine, spatial=package.fem.MiniP1(), solve=custom_newton, scaling=scaling, temporal=temporal)
assert mini.identity == replayed.identity
assert portable_mini.identity == mini.identity
assert portable_fvm.identity == fvm.identity
assert portable_mini.to_bytes() == mini_bytes
assert portable_fvm.to_bytes() == fvm_bytes
assert portable_mini.mesh.to_bytes() == affine.to_bytes()
assert portable_fvm.mesh.to_bytes() == cartesian.to_bytes()
assert portable_mini.spatial == package.fem.MiniP1()
assert portable_fvm.spatial == package.fvm.CellCentered()
assert portable_mini.temporal.step_s == mini.temporal.step_s
assert portable_mini.requested_solve.relative_tolerance == newton.relative_tolerance
assert portable_mini.requested_solve.linear.maximum_iterations == linear.maximum_iterations
assert portable_mini.solve.maximum_iterations == mini.solve.maximum_iterations
assert portable_mini.capability.scaling.length_m == mini.capability.scaling.length_m
assert mini.identity != fvm.identity
assert mini.identity != custom.identity
assert mini.model is model and mini.mesh is affine
assert fvm.model is model and fvm.mesh is cartesian
assert mini.fields == (mini.capability.velocity, mini.capability.pressure)
assert fvm.fields == (fvm.capability.velocity, fvm.capability.pressure)
assert mini.solve is not newton and mini.solve.linear is not linear
assert mini.requested_solve is newton
assert mini.solve.relative_tolerance == 1e-9
assert mini.solve.absolute_tolerance == 1e-11
assert mini.solve.maximum_iterations == 16
assert mini.solve.maximum_line_search_steps == 12
assert mini.solve.linear.relative_tolerance == linear.relative_tolerance
assert mini.solve.linear.absolute_tolerance == linear.absolute_tolerance
assert mini.solve.linear.maximum_iterations == linear.maximum_iterations
assert custom.solve.relative_tolerance == 2e-9
assert custom.solve.absolute_tolerance == 3e-11
assert custom.solve.maximum_iterations == 19
assert custom.solve.maximum_line_search_steps == 7
assert mini.temporal is temporal and mini.temporal.step_s == 0.01
assert len(mini.realization_digest) == 64 and len(fvm.realization_digest) == 64
assert mini.realization_digest != fvm.realization_digest
assert mini.capability.velocity_space == "simplex-p1-bubble"
assert mini.capability.pressure_space == "continuous-lagrange-p1"
assert fvm.capability.velocity_space == fvm.capability.pressure_space == "cell-constant"
assert mini.capability.pressure_gauge is package.fluid.PressureGauge2d.ZeroIntegral
assert fvm.capability.pressure_gauge is package.fluid.PressureGauge2d.ZeroIntegral
assert isinstance(mini.formulation, package.FormulationView)
assert mini.formulation.requested is package.FormulationSelectionMode.Automatic
assert mini.formulation.effective is package.formulation.MixedGalerkin
assert mini.formulation.boundary_treatment == "explicit-trace-flux-laws"
assert len(mini.formulation.rule_ids) == 6
assert mini.formulation.selection_reason_codes == [
    "eqiora.formulation.auto.mixed-galerkin-for-mini-p1/v1",
]
assert fvm.formulation.requested is package.FormulationSelectionMode.Automatic
assert fvm.formulation.effective is package.formulation.IntegralConservative
assert fvm.formulation.boundary_treatment == "explicit-trace-flux-laws"
assert len(fvm.formulation.rule_ids) == 7
assert fvm.formulation.selection_reason_codes == [
    "eqiora.formulation.auto.integral-conservative-for-cell-centered-fvm/v1",
]
assert mini_exact.formulation.requested is package.FormulationSelectionMode.Exact
assert fvm_exact.formulation.requested is package.FormulationSelectionMode.Exact
assert mini_exact.formulation.effective is mini.formulation.effective
assert fvm_exact.formulation.effective is fvm.formulation.effective
assert mini_exact.identity != mini.identity
assert fvm_exact.identity != fvm.identity
for wrong_mesh, wrong_spatial, wrong_formulation in (
    (affine, package.fem.MiniP1(), package.formulation.IntegralConservative),
    (cartesian, package.fvm.CellCentered(), package.formulation.MixedGalerkin),
):
    try:
        package.resolve(
            model, mesh=wrong_mesh, spatial=wrong_spatial,
            formulation=wrong_formulation,
            solve=newton, scaling=scaling, temporal=temporal,
        )
    except package.ValidationError:
        pass
    else:
        raise AssertionError("incompatible exact Formulation was admitted")
assert mini.solve.linear.algorithm == "sparse-lu"
assert fvm.solve.linear.algorithm == "bicgstab"
assert mini.solve.linear.reduction == "fast" and fvm.solve.linear.reduction == "reproducible"
assert mini.solve.linear.backend == "eqiora.faer"
assert fvm.solve.linear.backend == "eqiora.reference"
for numerical_plan in (mini, fvm, mini_exact, fvm_exact, *planned.values()):
    assert numerical_plan.model is model
    assert numerical_plan.model.structural_fingerprint == model_fingerprint
    assert numerical_plan.model.to_bytes() == model_bytes
# No diagonal claim is established before assembly. Both Jacobi tuples are
# ineligible, so identity LU is the sole admissible candidate for every objective.
expected_planning = {
    name: (objective, "eqiora.faer.sparse-lu-general-identity-fast-f64", "eqiora.faer", "sparse-lu", "fast")
    for name, objective in (("robust", package.solve.Robust), ("fast", package.solve.Fast), ("low-memory", package.solve.LowMemory))
}
for name, plan in planned.items():
    objective, candidate, backend, algorithm, reduction = expected_planning[name]
    resolved = plan.solve.linear
    assert resolved.objective is objective
    assert resolved.planning_policy_id == "eqiora.host-serial-solver-planning/v2"
    assert resolved.selected_candidate_id == candidate
    assert resolved.selected_evidence_case is not None
    assert len(resolved.planning_reasons) == 4
    assert resolved.backend == backend
    assert resolved.algorithm == algorithm
    assert resolved.reduction == reduction
    assert plan.identity != fvm.identity
assert len({plan.identity for plan in planned.values()}) == 3
planned_mini = package.resolve(model, mesh=affine, spatial=package.fem.MiniP1(),
    solve=package.solve.Newton(linear=package.solve.Linear(relative_tolerance=1e-10, absolute_tolerance=1e-12,
        maximum_iterations=2000, objective=package.solve.Robust)), scaling=scaling, temporal=temporal)
assert planned_mini.solve.linear.algorithm == "sparse-lu"
assert planned_mini.solve.linear.backend == "eqiora.faer"
assert mini.capability.scaling.length_m == 1.0 and mini.capability.scaling.velocity_m_per_s == 2.0 and mini.capability.scaling.pressure_pa == 3.0

for kwargs in (
    dict(temporal=None),
    dict(solve=linear, temporal=temporal),
    dict(scaling=None, temporal=temporal),
    dict(scaling=package.fluid.IncompressibleScaling(), temporal=temporal),
    dict(scaling=package.fluid.IncompressibleScaling(length_m=1.0), temporal=temporal),
):
    request = dict(mesh=affine, spatial=package.fem.MiniP1(), solve=newton, scaling=scaling, temporal=temporal)
    request.update(kwargs)
    try:
        package.resolve(model, **request)
    except package.ValidationError:
        pass
    else:
        raise AssertionError(f"invalid transient request admitted: {kwargs}")
for forbidden in ("state", "horizon", "output"):
    request = dict(mesh=affine, spatial=package.fem.MiniP1(), solve=newton, scaling=scaling, temporal=temporal)
    request[forbidden] = object()
    try:
        package.resolve(model, **request)
    except TypeError:
        pass
    else:
        raise AssertionError(f"future execution argument admitted: {forbidden}")
for invalid_step in (True, 1, 0.0, -0.0, -1.0, float("nan"), float("inf"), -float("inf")):
    try:
        package.time.BackwardEuler(invalid_step)
    except (TypeError, package.ValidationError):
        pass
    else:
        raise AssertionError(f"invalid BackwardEuler step admitted: {invalid_step!r}")
for kwargs in (
    dict(relative_tolerance=True),
    dict(relative_tolerance=1),
    dict(relative_tolerance=-1e-9),
    dict(relative_tolerance=1.0),
    dict(relative_tolerance=float("nan")),
    dict(relative_tolerance=float("inf")),
    dict(absolute_tolerance=True),
    dict(absolute_tolerance=1),
    dict(absolute_tolerance=-1e-11),
    dict(absolute_tolerance=float("nan")),
    dict(maximum_iterations=True),
    dict(maximum_iterations=0),
    dict(maximum_iterations=-1),
    dict(maximum_iterations=1.0),
    dict(maximum_line_search_steps=True),
    dict(maximum_line_search_steps=-1),
    dict(maximum_line_search_steps=65),
    dict(maximum_line_search_steps=1.0),
    dict(relative_tolerance=0.0, absolute_tolerance=0.0),
):
    try:
        package.solve.Newton(linear=linear, **kwargs)
    except (TypeError, package.ValidationError):
        pass
    else:
        raise AssertionError(f"invalid Newton controls admitted: {kwargs}")
mini_zero = package.State.zero(mini)
assert mini_zero.time_s == 0.0
mini_zero_bytes = mini_zero.to_bytes()
mini_zero_replayed = package.State.from_bytes(mini, mini_zero_bytes)
assert mini_zero_replayed == mini_zero
assert mini_zero_replayed.to_bytes() == mini_zero_bytes
assert mini_zero_replayed.source_kind == "artifact"
assert mini_zero.mesh is affine
assert mini_zero.model is model
assert mini_zero.source_plan_identity == mini.identity
assert mini_zero.source_request_identity is None
assert mini_zero.source_trajectory_identity is None
assert mini_zero.source_kind == "zero"
assert len(mini_zero.fields) == 2
assert mini_zero.field(mini.capability.velocity).associations == ("vertex", "cell")
assert mini_zero.field(mini.capability.pressure).associations == ("vertex",)

mini_one_sync = package.run(mini, state=mini_zero, steps=1, output_steps=(1,))
mini_one_async = package.submit(mini, state=mini_zero, steps=1, output_steps=(1,)).result()
assert mini_one_sync.trajectory.digest == mini_one_async.trajectory.digest
mini_one_time = package.run(mini, state=mini_zero, until_s=0.01, output_times_s=(0.01,))
assert mini_one_sync.trajectory.digest == mini_one_time.trajectory.digest
mini_two = package.run(mini, state=mini_zero, steps=2, output_steps=(1, 2))
assert tuple(state.step for state in mini_two.trajectory.states) == (1, 2)
assert tuple(state.time_s for state in mini_two.trajectory.states) == (0.01, 0.02)
mini_trajectory_bytes = mini_two.trajectory.to_bytes()
mini_trajectory_replayed = package.trajectory.Trajectory.from_bytes(mini, mini_trajectory_bytes)
assert mini_trajectory_replayed == mini_two.trajectory
assert mini_trajectory_replayed.to_bytes() == mini_trajectory_bytes
assert tuple(state.digest for state in mini_trajectory_replayed.states) == tuple(
    state.digest for state in mini_two.trajectory.states
)
trajectory_directory_owner = tempfile.TemporaryDirectory()
trajectory_directory = pathlib.Path(trajectory_directory_owner.name)
trajectory_path = trajectory_directory / "run.eqtrajectory"
mini_two.trajectory.write(trajectory_path)
assert trajectory_path.read_bytes() == mini_trajectory_bytes
mini_trajectory_file = package.trajectory.Trajectory.read(mini, trajectory_path)
assert mini_trajectory_file == mini_two.trajectory
assert mini_trajectory_file.to_bytes() == mini_trajectory_bytes
assert tuple(state.digest for state in mini_trajectory_file.states) == tuple(
    state.digest for state in mini_two.trajectory.states
)

mini_result_bytes = mini_two.to_bytes()
mini_result_path = trajectory_directory / "run.eqresult"
mini_two.write(mini_result_path)
mini_result_file = package.Result.read(mini, mini_result_path)
assert mini_result_path.read_bytes() == mini_result_bytes
assert mini_result_file.to_bytes() == mini_result_bytes
assert mini_result_file.trajectory.to_bytes() == mini_trajectory_bytes

for name, rejected in (
    ("truncated.eqtrajectory", mini_trajectory_bytes[:-1]),
    ("trailing.eqtrajectory", mini_trajectory_bytes + b"\n"),
    (
        "unknown-version.eqtrajectory",
        mini_trajectory_bytes.replace(b"common-trajectory/v3", b"common-trajectory/v9"),
    ),
):
    rejected_path = trajectory_directory / name
    rejected_path.write_bytes(rejected)
    try:
        package.trajectory.Trajectory.read(mini, rejected_path)
    except package.CompatibilityError:
        pass
    else:
        raise AssertionError(f"hostile Trajectory file must reject: {name}")

wrong_trajectory_suffix = trajectory_directory / "run.json"
for operation in ("write", "read"):
    try:
        if operation == "write":
            mini_two.trajectory.write(wrong_trajectory_suffix)
        else:
            package.trajectory.Trajectory.read(mini, wrong_trajectory_suffix)
    except package.CompatibilityError:
        pass
    else:
        raise AssertionError("Trajectory file paths require the exact .eqtrajectory suffix")

try:
    package.trajectory.Trajectory.read(fvm, trajectory_path)
except package.CompatibilityError:
    pass
else:
    raise AssertionError("Trajectory file was crossed with a different Plan")
trajectory_directory_owner.cleanup()
mini_restart = package.State.from_result(custom, mini_one_sync, time_s=0.01)
assert mini_restart.state_space_identity == mini_zero.state_space_identity
assert mini_restart.source_plan_identity == mini.identity
assert mini_restart.source_request_identity == mini_one_sync.plan_key
assert mini_restart.source_trajectory_identity == mini_one_sync.trajectory.digest
assert mini_restart.source_kind == "result"
assert mini_restart.mesh is affine
assert mini_restart.field(mini.capability.velocity).values("vertex").flags.writeable is False

fvm_zero = package.State.zero(fvm)
fvm_zero_bytes = fvm_zero.to_bytes()
fvm_zero_replayed = package.State.from_bytes(fvm, fvm_zero_bytes)
assert fvm_zero_replayed == fvm_zero
assert fvm_zero_replayed.to_bytes() == fvm_zero_bytes
assert fvm_zero_replayed.source_kind == "artifact"
assert fvm_zero.mesh is cartesian
assert fvm_zero.field(fvm.capability.velocity).associations == ("cell",)
assert fvm_zero.field(fvm.capability.pressure).associations == ("cell",)
for plan in planned.values():
    exact_linear = package.solve.Linear(relative_tolerance=1e-10, absolute_tolerance=1e-12,
        maximum_iterations=2000, algorithm=package.solve.LinearSolver.SparseLu,
        preconditioner=package.solve.Preconditioner.Identity,
        reduction=package.solve.Reduction.Fast, provider=package.solve.SolverProvider.faer())
    exact_plan = package.resolve(model, mesh=cartesian, spatial=package.fvm.CellCentered(),
        solve=package.solve.Newton(linear=exact_linear), scaling=scaling, temporal=temporal)
    assert exact_plan.solve.linear.objective is None
    assert exact_plan.solve.linear.provider == plan.solve.linear.provider
    plan = package.Plan.from_bytes(plan.to_bytes())
    ranked_result = package.run(plan, state=package.State.zero(plan), steps=1, output_steps=(1,))
    exact_result = package.run(exact_plan, state=package.State.zero(exact_plan), steps=1, output_steps=(1,))
    ranked_state = ranked_result.trajectory.states[0]
    exact_state = exact_result.trajectory.states[0]
    for ranked_field, exact_field in zip(plan.fields, exact_plan.fields):
        ranked_output, exact_output = ranked_state.field(ranked_field), exact_state.field(exact_field)
        for association in ranked_output.associations:
            assert ranked_output.values(association).tolist() == exact_output.values(association).tolist()
fvm_two = package.run(fvm, state=fvm_zero, steps=2, output_steps=(2,))
assert fvm_two.trajectory.plan_identity == fvm.identity
assert fvm_two.trajectory.realization_digest == fvm.realization_digest
assert fvm_two.trajectory.request_identity == fvm_two.plan_key
assert fvm_two.trajectory.run_digest == fvm_two.plan_key
fvm_first = package.run(fvm, state=fvm_zero, steps=1, output_steps=(1,))
fvm_restart = package.State.from_result(fvm, fvm_first, time_s=0.01)
fvm_second = package.run(fvm, state=fvm_restart, steps=1, output_steps=(1,))
assert fvm_two.trajectory.states[0] == fvm_second.trajectory.states[0]
alternate_scaling = package.fluid.IncompressibleScaling(length_m=2.0, velocity_m_per_s=4.0, pressure_pa=6.0)
fvm_alternate = package.resolve(
    model, mesh=cartesian, spatial=package.fvm.CellCentered(), solve=fvm_newton,
    scaling=alternate_scaling, temporal=temporal,
)
assert fvm_alternate.identity != fvm.identity
fvm_second_alternate = package.run(
    fvm_alternate, state=fvm_restart, steps=1, output_steps=(1,),
)
assert fvm_second_alternate.trajectory.states[0] == fvm_second.trajectory.states[0]

for invalid_time in (float("nan"), float("inf"), -1.0, -0.0):
    try:
        package.State.zero(mini, time_s=invalid_time)
    except (ValueError, package.ValidationError):
        pass
    else:
        raise AssertionError(f"invalid zero-State time was admitted: {invalid_time!r}")
for invalid_operation in range(6):
    try:
        if invalid_operation == 0:
            package.submit(fvm, state=mini_zero, steps=1, output_steps=(1,))
        elif invalid_operation == 1:
            package.State.from_result(fvm, mini_one_sync, time_s=0.01)
        elif invalid_operation == 2:
            package.submit(mini, state=mini_zero, steps=2, output_steps=(0,))
        elif invalid_operation == 3:
            package.submit(mini, state=mini_zero, steps=2, output_steps=(1, 1))
        elif invalid_operation == 4:
            package.submit(mini, state=mini_zero, steps=2, output_steps=(3,))
        else:
            package.submit(mini, state=mini_zero, until_s=0.015, output_times_s=(0.01,))
    except (ValueError, package.ValidationError):
        pass
    else:
        raise AssertionError("foreign State or invalid exact schedule was admitted")

for kwargs in (
    {},
    dict(state=mini_zero),
    dict(state=mini_zero, steps=1),
    dict(state=mini_zero, output_steps=(1,)),
    dict(state=mini_zero, steps=1, output_steps=(1,), until_s=0.01, output_times_s=(0.01,)),
):
    try:
        package.submit(mini, **kwargs)
    except TypeError:
        pass
    else:
        raise AssertionError(f"incomplete or mixed transient Run controls admitted: {kwargs}")
try:
    package.submit(model=model, end_time=0.01, max_step=0.01)
except TypeError:
    pass
else:
    raise AssertionError("legacy root submit form remained callable")
"#),
                None,
                Some(&locals),
            )
    })
}
