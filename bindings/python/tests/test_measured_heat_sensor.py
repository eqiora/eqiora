"""The sensor example uses physical reconstruction, native residuals and exact time selection."""
import runpy
from pathlib import Path

import eqiora as q
import pytest

PROGRAM = Path(__file__).resolve().parents[3] / "examples/python/measured_heat_sensor.py"


@pytest.fixture
def example():
    from functools import partial
    namespace = runpy.run_path(str(PROGRAM))
    namespace["prepare"] = partial(namespace["prepare"], license_text="CC0-1.0 synthetic data.")
    return namespace


@pytest.mark.parametrize("times,bias", [((.25,.75),0.), ((.5,.75),.01)])
def test_recorded_sensor_residuals_and_reopening(example, tmp_path, times, bias):
    # For four Q1 cells, M_cc=1/9, K_cc=8/3, F_c=1/4. With dt=1/4,
    # the center excess is 3/32*(1-7^-n). The Q1 hat at (1/4,1/2) is 1/2.
    text = "time_ms,x_mm,y_mm,temperature_K\n0,250,500,300\n"
    for time in times:
        temperature = 300 + 3/64*(1-7**(-int(time/.25))) + bias
        text += f"{time*1000},250,500,{temperature!r}\n"
    model, plan, result, rows = example["prepare"](tmp_path, text)
    ids = {name: model.observable("plate."+name).id
           for name in ("profile", "residual", "normalized", "loss")}
    model_bytes, plan_bytes, result_bytes = model.to_bytes(), plan.to_bytes(), result.to_bytes()
    replay_model = q.Model.from_bytes(model_bytes)
    replay_plan = q.Plan.from_bytes(plan_bytes)
    replay_result = q.Result.from_bytes(replay_plan, result_bytes)
    for current_model, current_result in ((model,result),(replay_model,replay_result)):
        observations = example["compare"](current_model,current_result,rows,ids)
        for index, (time, observed) in enumerate(zip(times, observations, strict=True)):
            expected = 300+3/64*(1-7**(-int(time/.25)))
            for name, value, dimension in (
                ("profile",expected,q.Dimension(temperature=1)),
                ("residual",-bias,q.Dimension(temperature=1)),
                ("normalized",-bias/.01,q.Dimension()),
                ("loss",.5*(bias/.01)**2,q.Dimension()),
            ):
                sample = observed[name]
                assert sample.value == pytest.approx(value, abs=1e-8, rel=0)
                assert sample.value_type == q.ValueType.real(dimension)
                assert sample.output == (index,time,current_result.trajectory.states[index].digest)
                assert sample.point == (model.domain("body").id,
                                        [(.25,q.Dimension(length=1)),(.5,q.Dimension(length=1))])
                assert sample.result_identity == observed["profile"].result_identity
        assert current_result.to_bytes() == result_bytes
    # A recorded coordinate is checked against the selected Observable's exact support.
    profile = model.observable(ids["profile"])
    for point in ([(-.1,q.Dimension(length=1)),(.5,q.Dimension(length=1))],
                  [(.25,q.Dimension(time=1)),(.5,q.Dimension(length=1))]):
        with pytest.raises(q.ValidationError):
            result.observe_at(profile,point,output_index=0)
    with pytest.raises(q.ValidationError):
        result.observe_at(profile,[(.25,q.Dimension(length=1)),(.5,q.Dimension(length=1))],output_index=2)
    altered_rows = (rows[0], (.375,*rows[1][1:]), rows[2])
    with pytest.raises(ValueError,match="Recorded time differs"):
        example["compare"](model,result,altered_rows,ids)
    assert model.to_bytes() == model_bytes


def test_sensor_units_and_unretained_times_reject_before_solving(example,tmp_path):
    columns = tuple((name,q.units.s if name=="x_mm" else unit) for name,unit in example["COLUMNS"])
    with pytest.raises(ValueError,match="dimensions"):
        example["prepare"](tmp_path,example["CSV"],columns=columns)
    with pytest.raises(ValueError,match="step grid"):
        example["prepare"](tmp_path,example["CSV"].replace("250,250,500", "375,250,500"))


def test_sensor_duplicate_time_uses_property_admission(example,tmp_path):
    text=example["CSV"].replace("750,250", "250,250")+"750,250,500,300.05\n"
    with pytest.raises(q.CompatibilityError,match="package source analysis produced"):
        example["prepare"](tmp_path,text)


def test_sensor_coordinate_never_loses_its_scalar_domain(example,tmp_path):
    import numpy as np
    model,plan,result,rows=example['prepare'](tmp_path,example['CSV'])
    profile=model.observable('plate.profile')
    for value in (True,.25+2j,np.complex128(.25+2j)):
        with pytest.raises(TypeError):
            result.observe_at(profile,[(value,q.Dimension(length=1)),(.5,q.Dimension(length=1))],output_index=0)
    sample=result.observe_at(profile,[(np.float64(.25),q.Dimension(length=1)),(.5,q.Dimension(length=1))],output_index=0)
    assert sample.value == pytest.approx(300+9/224,abs=1e-10,rel=0)


def test_changed_sensor_data_changes_the_resolved_identity(example,tmp_path):
    original=example['prepare'](tmp_path/'original',example['CSV'])
    changed=example['prepare'](tmp_path/'changed',example['CSV'].replace('300.04017857142856','300.05017857142856'))
    assert original[0].to_bytes() != changed[0].to_bytes()
    old_id=original[0].observable('plate.residual')
    new_id=changed[0].observable('plate.residual')
    point=[(.25,q.Dimension(length=1)),(.5,q.Dimension(length=1))]
    old=original[2].observe_at(old_id,point,output_index=0)
    new=changed[2].observe_at(new_id,point,output_index=0)
    assert new.value-old.value == pytest.approx(-.01,abs=1e-10,rel=0)
    with pytest.raises(ValueError, match="different exact Model artifact"):
        original[2].observe_at(new_id,point,output_index=0)


def test_downloaded_sensor_script_runs_with_installed_eqiora(tmp_path):
    import shutil
    import subprocess
    import sys
    script=tmp_path/'sensor.py'
    shutil.copyfile(PROGRAM,script)
    output=tmp_path/'accepted'
    completed=subprocess.run([sys.executable,'-I',str(script),'--output',str(output)],
                             cwd=tmp_path,capture_output=True,text=True,check=True)
    assert len(completed.stdout.splitlines()) == 2
    assert all((output/(name+'.json')).is_file() for name in ('model','plan','result'))
