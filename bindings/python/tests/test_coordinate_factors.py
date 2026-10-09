"""Installed interval bindings exercise the ordinary source and Model replay owners."""
import json
import pytest
import eqiora

SOURCE = """model Distribution(support position:interval(m), support velocity:interval(m/s)) {
  support phase:product(position,velocity);
  variable f:s/m^2 on phase;
  relation retain on phase { f=0[s/m^2]; }
  observable count:1=integral(f,measure(phase));
}"""


def bindings():
    return {
        "position": eqiora.CoordinateInterval(0, 1, dimension=eqiora.Dimension(length=1)),
        "velocity": eqiora.CoordinateInterval(-2, 2, dimension=eqiora.Dimension(length=1, time=-1)),
    }


def test_coordinate_factor_source_and_replay_without_geometry():
    inputs = bindings()
    assert inputs["velocity"].lower == -2 and inputs["velocity"].upper == 2
    assert inputs["velocity"].dimension == eqiora.Dimension(length=1, time=-1)
    model = eqiora.compile(source=SOURCE, entry="Distribution", bindings=inputs)
    assert eqiora.Model.from_bytes(model.to_bytes()).digest == model.digest
    wire = json.loads(model.to_bytes())
    assert wire["schema"] == "eqiora.model-envelope/v44"
    domains = {node["id"]["ulid"]: node["definition"]["domain"]
               for node in wire["nodes"] if node["definition"]["kind"] == "domain"}
    product = next(value for value in domains.values() if value["kind"] == "coordinate-product")
    factors = [domains[factor["ulid"]] for factor in product["factors"]]
    # Independent SI exponent check: position has T^0, velocity has T^-1.
    assert [factor["lower"]["dimension"][2] for factor in factors] == [[0, 1], [-1, 1]]
    assert len(model.domain_ids) == 3


def test_coordinate_factor_units_and_invalid_measure_reject():
    inputs = bindings()
    inputs["velocity"] = inputs["position"]
    with pytest.raises(eqiora.ValidationError, match="coordinate units"):
        eqiora.compile(source=SOURCE, entry="Distribution", bindings=inputs)
    with pytest.raises(eqiora.ValidationError):
        eqiora.compile(source=SOURCE.replace("count:1", "count:s"), entry="Distribution", bindings=bindings())


@pytest.mark.parametrize("lower,upper", [(0,0),(1,0),(0,float("inf")),(float("nan"),1),(True,1),(0,1+0j)])
def test_coordinate_factor_bounds_reject(lower, upper):
    with pytest.raises((TypeError, ValueError)):
        eqiora.CoordinateInterval(lower, upper, dimension=eqiora.Dimension(length=1))


def test_coordinate_interval_value_protocol_normalizes_signed_zero():
    dimension = eqiora.Dimension(length=1)
    positive = eqiora.CoordinateInterval(0.0, 1.0, dimension=dimension)
    negative = eqiora.CoordinateInterval(-0.0, 1.0, dimension=dimension)
    assert positive == negative
    assert hash(positive) == hash(negative)
    assert repr(positive).startswith("CoordinateInterval(0.0, 1.0, dimension=Dimension(")
