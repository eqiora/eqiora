use super::*;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    solver_request::register(module)?;
    module.add_class::<PyQ1>()?;
    module.add_class::<PyMiniP1>()?;
    module.add_class::<PyP1>()?;
    module.add_class::<PyScopedSpatialBinding>()?;
    module.add_class::<PyCellCenteredTpfa>()?;
    module.add_class::<PyCellCentered>()?;
    module.add_class::<PySolverPlanningObjective>()?;
    module.add_class::<PyLinear>()?;
    module.add_class::<enforcement::PyConstraintTolerance>()?;
    module.add_class::<enforcement::PyActiveSet>()?;
    module.add_class::<PyNewton>()?;
    module.add_class::<PyResolvedLinear>()?;
    module.add_class::<PyResolvedNewton>()?;
    module.add_class::<PyResolvedExecution>()?;
    module.add_class::<algebraic::PyAlgebraicPlanView>()?;
    module.add_class::<PyOdePlanView>()?;
    module.add_class::<PyScalarPlanView>()?;
    module.add_class::<PyElasticityPlanView>()?;
    module.add_class::<PyIncompressibleFlowPlanView>()?;
    module.add_class::<PyFormulationKind>()?;
    module.add_class::<PyFormulationSelectionMode>()?;
    module.add_class::<PyFormulationView>()?;
    module.add_class::<PyFixedReferenceFsiPlanView>()?;
    module.add_class::<PyPressureGauge2d>()?;
    module.add_class::<PyBackwardEuler>()?;
    module.add_class::<PyTsitouras45>()?;
    module.add_class::<PyPlan>()?;
    scaling::register(module)?;
    event_policy::register(module)?;
    forward_policy::register(module)?;
    module.add_function(wrap_pyfunction!(resolve_plan, module)?)?;
    Ok(())
}
