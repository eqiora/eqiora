"""Ordinary Cartesian Hex API preparation; execution is reviewed separately."""
import importlib.util
from pathlib import Path

import eqiora
import pytest

ROOT = Path(__file__).resolve().parents[3]
CASE = ROOT/'verify/numerics/cartesian-hex-elasticity'
SPEC = importlib.util.spec_from_file_location('hex_elasticity_production', CASE/'production.py')
production = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(production)


def test_ordinary_hex_source_compiles_resolves_and_reopens_without_solving(tmp_path):
    model, plan, field, observables = production.prepare(eqiora, tmp_path)
    assert len(observables) == 4
    assert field.model_digest == model.digest == plan.model_digest
    assert len(plan.field_coefficient_entities(field)) == 27
    assert (tmp_path/'model.json').is_file()
    assert (tmp_path/'mesh.json').is_file()
    assert (tmp_path/'plan.json').is_file()
    assert not (tmp_path/'result.json').exists()


@pytest.mark.parametrize('bounds', [(2., 0.), (0., 0.), (0., float('inf')), (0., float('nan'))])
def test_invalid_cartesian_geometry_rejects_before_result(bounds):
    graph = eqiora.geometry.GeometryGraph()
    with pytest.raises(eqiora.ValidationError):
        graph.cuboid(x_bounds=bounds, y_bounds=(0., 3.), z_bounds=(0., 4.))


def test_real_cartesian_inspectable_resource_budget_rejects_before_result():
    graph = eqiora.geometry.GeometryGraph()
    box = graph.cuboid(x_bounds=(0., 2.), y_bounds=(0., 3.), z_bounds=(0., 4.))
    geometry = graph.build(box, named_topology={'body': box.region, **dict(zip(production.SIDES, box.boundaries, strict=True))})
    with pytest.raises(eqiora.ValidationError, match='limit|exceed|budget'):
        eqiora.meshing.generate(eqiora.meshing.resolve(geometry, eqiora.meshing.CartesianMesher(cells=(10000, 10000, 10000))))


def altered_sources():
    source = (CASE/'model.eqi').read_text()
    equations, weak = source.split('  form weak for balance {', 1)
    assert weak.count('dot(trace(w),tx)') == 1
    return {
        'changed-component': equations + '  parameter incorrect_tx: vector<Pa,3> = tensor_value(frame=body,components=[1.125,0.5,0.1875]);\n'
            + '  form weak for balance {' + weak.replace('dot(trace(w),tx)', 'dot(trace(w),incorrect_tx)', 1),
        'reversed-traction': equations + '  form weak for balance {' + weak.replace('dot(trace(w),tx)', 'dot(trace(w),-tx)', 1),
        'partial-contraction': equations + '  form weak for balance {' + weak.replace('frobenius(grad(w),(2*mu*symmetric_part(grad(displacement)) + lambda*isotropic_lift(div(displacement))))', 'contract(grad(w),(2*mu*symmetric_part(grad(displacement)) + lambda*isotropic_lift(div(displacement))),axes=((0,0),))', 1),
        'wrong-unit': source.replace('parameter tx: vector<Pa,3>', 'parameter tx: vector<N,3>', 1),
    }


@pytest.mark.parametrize('control', ('changed-component', 'reversed-traction'))
def test_physical_authored_traction_mismatches_reject_before_any_result(tmp_path, control):
    reject_altered_source(tmp_path, control)


@pytest.mark.parametrize('control', ('partial-contraction', 'wrong-unit'))
def test_unsupported_or_illtyped_authored_forms_reject_before_any_result(tmp_path, control):
    reject_altered_source(tmp_path, control)


def reject_altered_source(tmp_path, control):
    baseline = tmp_path/'baseline'
    baseline.mkdir()
    production.prepare(eqiora, baseline)
    mutated = tmp_path/'mutated'
    mutated.mkdir()
    source = altered_sources()[control]
    assert source != (CASE/'model.eqi').read_text()
    with pytest.raises(eqiora.ValidationError) as caught:
        production.prepare(eqiora, mutated, source=source.encode())
    assert 'coefficient alias exceeds the authored expression inventory' not in str(caught.value)
    assert not (mutated/'result.json').exists()
