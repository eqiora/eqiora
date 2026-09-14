from typing import assert_type

import numpy as np
import numpy.typing as npt

import eqiora
from eqiora.colab import prepare
from eqiora.fsi import (
    FsiConnectionEvidence,
    FsiDomainEvidence,
    FsiEvidence,
    FsiInterfaceActionEvidence,
    FsiStateEvidence,
)
from eqiora.meshing import Mesh
from eqiora.solid import LinearElasticityEvidence
from eqiora.trajectory import FieldSnapshot, Trajectory, State


def check_language_source() -> None:
    source = eqiora.Module('main')
    component = source.component("Poisson")
    volume = component.volume("volume", dimensions=2)
    value = component.field("value", role=eqiora.FieldRole.Variable, on=volume, value_type=eqiora.ValueType.real(eqiora.Dimension(length=1)))
    natural = component.relation("balance", eqiora.lang.equation(eqiora.lang.div(value), -eqiora.lang.math.pi * value), on=volume)
    residual = component.relation("reference", eqiora.lang.equation(value, 0), on=volume)
    assert_type(natural, eqiora.lang.Relation)
    assert_type(residual, eqiora.lang.Relation)
    assert_type(eqiora.lang.math.pi, eqiora.lang.Expression)
    assert_type(source.to_eqi(), str)
    assert_type(eqiora.compile(source=source), eqiora.Model)


def check_native_modeling() -> None:
    length = eqiora.Dimension(length=1)
    domain = eqiora.Domain.box("rod", (0.0, 1.0))

    temperature = eqiora.Field(
        "temperature",
        role=eqiora.FieldRole.Variable,
        domain=domain,

        value_type=eqiora.ValueType.real(length),
    )
    conductivity = eqiora.Parameter("conductivity", value=1.0)
    assert_type(conductivity.value_type, eqiora.ValueType)
    assert_type(temperature.value_type, eqiora.ValueType)
    assert_type(temperature.role, eqiora.FieldRole)
    assert_type(eqiora.ValueType.tensor(eqiora.ValueType.complex(length), 2, 2), eqiora.ValueType)
    balance = eqiora.Relation(
        "balance",
        domain=domain,
        equations=[(eqiora.div(conductivity * eqiora.grad(temperature)), 0)],
    )

    assert_type(
        eqiora.compile(source=eqiora.Module(
            "thermal",
            domain,

            temperature,
            conductivity,
            balance,
        )),
        eqiora.Model,
    )


def check_execution(
    plan: eqiora.Plan,
    state: eqiora.State,
    array: eqiora.Array,
) -> None:
    assert_type(eqiora.run(plan), eqiora.Result)
    transient = eqiora.submit(plan, state=state, until_s=1.0, output_times_s=(1.0,))
    assert_type(transient, eqiora.Run[eqiora.Result])
    assert_type(transient.result(), eqiora.Result)
    assert_type(array.numpy(copy=False), npt.NDArray[np.float64])

    eqiora.Run[eqiora.Result](object())  # type: ignore[arg-type]


class _InvalidModelSubclass(eqiora.Model):  # type: ignore[misc]
    pass


def check_viewer(
    geometry: eqiora.geometry.Geometry,
    mesh: Mesh,
    field: eqiora.FieldOutput,
) -> None:
    view = eqiora.View()
    assert_type(view.add(geometry), eqiora.View)
    assert_type(view.add(mesh), eqiora.View)
    assert_type(view.add(field), eqiora.View)
    assert_type(view.show(), eqiora.View)
    assert_type(view.close(), None)
    assert_type(prepare(), None)


def check_structural_result(plan: eqiora.Plan, result: eqiora.Result) -> None:
    assert_type(plan, eqiora.Plan)
    capability = plan.capability
    assert isinstance(capability, eqiora.solid.ElasticityPlanView)
    assert_type(capability.displacement, eqiora.FieldRef)
    assert_type(
        eqiora.solid.linear_elasticity_evidence(result),
        LinearElasticityEvidence,
    )


def check_fsi_result(plan: eqiora.Plan, state: eqiora.State) -> None:
    assert_type(plan, eqiora.Plan)
    capability = plan.capability
    assert isinstance(capability, eqiora.fsi.FixedReferenceFsiPlanView)
    assert_type(
        plan.temporal, eqiora.time.BackwardEuler | eqiora.time.Tsitouras45 | None
    )
    assert_type(
        eqiora.submit(plan, state=state, steps=2, output_steps=(1, 2)),
        eqiora.Run[eqiora.Result],
    )
    result = eqiora.run(plan, state=state, steps=2, output_steps=(1, 2))
    assert_type(result, eqiora.Result)
    assert_type(result.trajectory, Trajectory)
    assert_type(result.trajectory.coordinates, npt.NDArray[np.float64])
    assert_type(result.trajectory.cells, npt.NDArray[np.uint32])
    assert_type(result.trajectory.states, tuple[State, ...])
    assert_type(result.trajectory.state(1).fields, tuple[FieldSnapshot, ...])
    assert_type(
        result.trajectory.state(1).field(plan.model.field("definition.fluid_velocity")),
        FieldSnapshot,
    )
    assert_type(
        result.trajectory.state(1).field(plan.model.field("definition.fluid_velocity")).support_indices("vertex"),
        npt.NDArray[np.uint32],
    )
    evidence = eqiora.fsi.evidence(result)
    assert_type(evidence, FsiEvidence)
    assert_type(
        evidence.state(result.trajectory.state(1)),
        FsiStateEvidence,
    )
    assert_type(evidence.domains, tuple[FsiDomainEvidence, ...])
    assert_type(evidence.domains[0].cells, npt.NDArray[np.uint32])
    assert_type(evidence.connections, tuple[FsiConnectionEvidence, ...])
    assert_type(evidence.connections[0].facets, npt.NDArray[np.uint32])
    assert_type(
        evidence.states,
        tuple[FsiStateEvidence, ...],
    )
    assert_type(
        evidence.state(result.trajectory.state(1)).interface_actions,
        tuple[FsiInterfaceActionEvidence, ...],
    )
    assert_type(
        evidence.state(result.trajectory.state(1)).interface_actions[0].endpoint_actions,
        npt.NDArray[np.float64],
    )
