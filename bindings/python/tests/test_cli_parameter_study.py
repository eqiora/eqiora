"""Cross-client CLI transport uses installed Python's exact Plan/collection owners.

Set EQIORA_STUDY_BIN to the CLI built with cli,faer from the same source revision
as the freshly installed candidate wheel. Matching version labels do not promise
older artifact compatibility. CLI execution is optional in the installed Python
gate, whose default environment has no Rust CLI.
"""
import json
import os
import subprocess

import eqiora as q
import numpy as np
import pytest

from test_parameter_evaluation_map import problem, mapped, values, TOLERANCE, linear


@pytest.fixture
def cli():
    binary = os.environ.get("EQIORA_STUDY_BIN")
    if not binary:
        pytest.skip("matching cli,faer binary not supplied")
    return binary


def command(cli, *arguments, success=True):
    result = subprocess.run([cli, *map(str, arguments)], capture_output=True, text=True, timeout=30)
    assert result.returncode == (0 if success else 1), result.stderr
    return result


def encoded_points(collection):
    # Reuse the public existing collection's complete typed literal encoding.
    return json.loads(collection.to_bytes())["groups"][0]["request"]["points"]


@pytest.mark.parametrize("kind", ["rc", "wave"])
def test_cli_run_inspect_and_exact_cross_client_reopen(cli, kind, tmp_path):
    plan, omega, frequencies, expected = problem(kind)
    plan_path, output = tmp_path / "input.eqplan", tmp_path / "study.eqcollection"
    plan_path.write_bytes(plan.to_bytes())
    arguments = ["study", "run", plan_path, "--parameter", omega.id, "--output", output]
    # Complete handwritten public literal: exact dimension exponents M,L,T,I,Theta,N,J.
    # The scalar is angular frequency, with negative time exponent and no shape axes.
    for frequency in frequencies:
        point = [{"value_type": {"domain": "real", "dimension":
            [[0, 1], [0, 1], [-1, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
            "shape": [], "frame": "invariant", "array_rank": 0,
            "basis": {"kind": "ordinary"}},
            "components": {"kind": "dense", "values": [[frequency, 0.]]}}]
        arguments.extend(("--point", json.dumps(point)))
    command(cli, *arguments)
    reopened = q.EvaluationCollection.read(output)
    assert reopened.complete and len(reopened[0]) == 3
    assert reopened[0][0].plan.identity == reopened[0][2].plan.identity != reopened[0][1].plan.identity
    for member, wanted, frequency in zip(reopened[0], expected, frequencies, strict=True):
        assert member.point == (frequency,)
        np.testing.assert_allclose(values(member, kind), wanted, rtol=0, atol=TOLERANCE)
    inspected = json.loads(command(cli, "study", "inspect", output).stdout)
    assert inspected["identity"] == reopened.identity
    for printed, member in zip(inspected["groups"][0]["members"], reopened[0], strict=True):
        assert printed["status"] == "accepted"
        assert printed["plan"] == member.plan.identity
        assert printed["result"] == member.reference
        assert printed["fields"]
        assert printed["harmonic"]["angular_frequency_per_second"] == member.plan.harmonic_angular_frequency
        assert "harmonic.derive.v1.negative-exponential-peak" in printed["harmonic"]["formulation_rules"]
        for mapping, (name, original, amplitude) in zip(printed["harmonic"]["amplitudes"], member.plan.harmonic_amplitudes, strict=True):
            assert mapping == {"name": name, "original": original.id, "amplitude": amplitude.id}
        if kind == "rc":
            components = [field["value"]["components"]["values"][0] for field in printed["fields"]]
            np.testing.assert_allclose([complex(*pair) for pair in components], values(member, kind), rtol=0, atol=TOLERANCE)
        else:
            block = printed["fields"][0]["blocks"][0]
            assert block["association"] == "vertex"
            components = [value["components"]["values"][0] for value in block["values"]]
            np.testing.assert_allclose([complex(*pair) for pair in components], values(member, kind), rtol=0, atol=TOLERANCE)
    before = output.read_bytes()
    command(cli, *arguments, success=False)
    assert output.read_bytes() == before
    # Replace a frequency point with the existing dimensionless literal type.
    nonfrequency = q.compile(source="model M(){parameter p:1=2;variable u:1;relation r{u=p;}}")
    other = q.EvaluationMapPlan.from_parameters(q.resolve(nonfrequency, solve=linear()),
        [nonfrequency.parameter("p")], [(2.,)]).execute_recorded()
    wrong = encoded_points(other)[0]
    arguments[-1] = json.dumps(wrong)
    arguments[arguments.index("--output")+1] = tmp_path / "wrong.eqcollection"
    command(cli, *arguments, success=False)
    assert not (tmp_path / "wrong.eqcollection").exists()


def test_cli_retry_preserves_success_and_failed_history_and_retries_only_unsuccessful(cli, tmp_path):
    plan, omega, _, _ = problem("rc")
    original = mapped(plan, omega, [500., 0., 1000.]).execute_recorded()
    path, retried = tmp_path / "partial.eqcollection", tmp_path / "retry.eqcollection"
    original.write(path)
    before = path.read_bytes()
    command(cli, "study", "retry", path, "--group", 0, "--output", retried, success=False)
    combined = q.EvaluationCollection.read(retried)
    assert len(combined) == 2 and not combined.complete
    assert combined[0].execution_id == original[0].execution_id != combined[1].execution_id
    assert combined[0][0].result.to_bytes() == original[0][0].result.to_bytes()
    assert combined[0][0].occurrence_id == original[0][0].occurrence_id
    assert [member.point for member in combined[1]] == [(0.,), (1000.,)]
    assert [member.status for member in combined[1]] == ["failed", "not_started"]
    assert path.read_bytes() == before
    inspection = json.loads(command(cli, "study", "inspect", retried).stdout)
    assert [member["status"] for member in inspection["groups"][1]["members"]] == ["failed", "not_started"]


def test_cli_nonfrequency_cancelled_study_retry_and_typed_outputs(cli, tmp_path):
    model = q.compile(source="model M(){parameter p:V=2[V];variable u:V;relation r{u=p;}}")
    plan = q.resolve(model, solve=linear())
    request = q.EvaluationMapPlan.from_parameters(plan, [model.parameter("p")], [(2.,), (3.,), (2.,)])
    cancellation = q.EvaluationMapCancellation()
    cancellation.cancel()
    original = request.execute_recorded(cancellation=cancellation)
    path, output = tmp_path / "cancelled.eqcollection", tmp_path / "retried.eqcollection"
    original.write(path)
    # A successful new execution exits zero; its incomplete history remains visible.
    command(cli, "study", "retry", path, "--group", 0, "--output", output)
    combined = q.EvaluationCollection.read(output)
    assert combined[1].complete
    assert [member.point for member in combined[1]] == [(2.,), (3.,), (2.,)]
    inspected = json.loads(command(cli, "study", "inspect", output).stdout)
    for printed, member, wanted in zip(inspected["groups"][1]["members"], combined[1], [2., 3., 2.], strict=True):
        assert printed["fields"][0]["association"] == "global"
        assert printed["fields"][0]["value"]["components"]["values"] == [[wanted, 0.]]
        assert list(member.result.output(member.plan.fields[0]).values("global")) == pytest.approx([wanted])
    command(cli, "study", "retry", output, "--group", 1, "--output", tmp_path / "empty", success=False)
    assert not (tmp_path / "empty").exists()


def test_cli_rejects_nonregular_input_before_open(cli, tmp_path):
    command(cli, "study", "inspect", tmp_path, success=False)
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)
        command(cli, "study", "inspect", fifo, success=False)
