"""The root unit owner projects compiler metadata and exact Module inputs."""

from decimal import Decimal, localcontext
from fractions import Fraction
import importlib

import pytest
import eqiora

q = eqiora.lang
u = eqiora.units


def test_units_catalog_projects_the_compiler_owner_without_old_aliases():
    symbols, prefixes = eqiora._eqiora._unit_catalog()
    assert set(u.__all__) == {"Unit", "PresentedQuantity"} | {"one" if name == "1" else name for name, _ in symbols}
    for name, prefixable in symbols:
        value = getattr(u, "one" if name == "1" else name)
        assert isinstance(value, u.Unit)
        for prefix in prefixes:
            if prefixable:
                assert u.Unit is type(value.prefixed(prefix))
            else:
                with pytest.raises(ValueError, match="prefix"):
                    value.prefixed(prefix)
    assert not hasattr(q, "units")
    assert not hasattr(eqiora, "m")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("eqiora.lang.units")


def test_unit_expression_bounds_precede_repeated_composition():
    value = u.m
    for _ in range(11):
        value = value * value
    with pytest.raises(ValueError, match="4096"):
        value * value
    value = u.m
    for _ in range(63):
        value = value * u.s
    with pytest.raises(ValueError, match="64"):
        value / u.s
    with pytest.raises(AttributeError):
        u.m._text = "s"
    with pytest.raises(TypeError):
        u.Unit()
    with pytest.raises(ValueError):
        u.s.prefixed("c").prefixed("m")
    with pytest.raises(ValueError):
        u.kg.prefixed("m")
    with pytest.raises(TypeError):
        u.m ** 0.5
    assert isinstance(u.m ** Fraction(1, 2), u.Unit)


def test_decimal_quantity_authoring_is_bounded_and_independent_of_decimal_context():
    source = eqiora.Module("main")
    component = source.component("Quantities")
    with localcontext() as context:
        context.prec = 2
        component.let_alias("density", q.quantity(Decimal("998.2"), u.kg / u.m**3))
        component.let_alias("small", q.quantity(Decimal("1e-1000000"), u.s))
        component.let_alias("integer", q.quantity(9007199254740993, u.one))
        component.let_alias("float_value", q.quantity(0.1, u.s))
        component.let_alias("length", q.quantity(10, u.m.prefixed("c")))
    text = source.to_eqi()
    assert "998.2 [kg / m ^ 3]" in text
    assert "1e-1000000 [s]" in text
    assert "9007199254740993 [1]" in text
    assert "0.1 [s]" in text
    assert "10 [cm]" in text
    for value in (Decimal("NaN"), Decimal("Infinity"), float("inf"), True):
        with pytest.raises((TypeError, ValueError)):
            q.quantity(value, u.s)
    for value in (Decimal("1" * 257), Decimal("1." + "2" * 255), 10**256):
        with pytest.raises(ValueError, match="256"):
            q.quantity(value, u.one)
    with pytest.raises(TypeError):
        q.quantity(Fraction(1, 3), u.s)


def test_decimal_and_centiprefix_quantity_matches_coherent_source():
    source = eqiora.Module("main")
    component = source.component("Length")
    region = component.volume("region", dimensions=2)
    field = component.field("length", on=region, role=eqiora.FieldRole.Variable,
                            value_type=eqiora.ValueType.real(eqiora.Dimension(length=1)))
    component.relation("law", eqiora.lang.equation(field, q.quantity(Decimal("10"), u.m.prefixed("c"))), on=region)
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0, 1), y_bounds=(0, 1))
    geometry = graph.build(rectangle, named_topology={
        "region": rectangle.region,
        "left": rectangle.boundaries[0],
        "right": rectangle.boundaries[1],
        "bottom": rectangle.boundaries[2],
        "top": rectangle.boundaries[3],
    })
    authored = eqiora.compile(source=source, geometry=geometry, entry='Length', bindings={'region': geometry.selection('region')})
    reference = eqiora.compile(source="""
public component Length(support region: volume(ambient_dimension = 2)) {
  variable length: m on region;
  relation law on region { length = 0.1 [m]; }
}
""", geometry=geometry, entry='Length', bindings={'region': geometry.selection('region')})
    assert authored.structural_fingerprint == reference.structural_fingerprint


def test_clock_seconds_emit_exact_quantity_ratios_without_float_conversion():
    source = eqiora.Module("main")
    component = source.component("ExactClocks")
    maximum = (1 << 64) - 1
    component.clock("third", period_s=Fraction(1, 3))
    component.clock("maximum", period_s=maximum, phase_s=Fraction(1, maximum))
    text = source.to_eqi()
    assert "periodic(1 [s] / 3 [1], phase = 0 [s] / 1 [1]" in text
    assert f"periodic({maximum} [s] / 1 [1], phase = 1 [s] / {maximum} [1]" in text
    for value in (0.1, Decimal("0.1"), True):
        with pytest.raises(TypeError):
            eqiora.Module("main").component("InvalidClock").clock("tick", period_s=value)


def test_numeric_inputs_share_exact_source_unit_normalization():
    # Decimal scaling precedes binary64 conversion, including inputs outside
    # the standalone binary64 range. These expected decimal quantities are exact.
    cases = (
        (Decimal("1e310"), u.s.prefixed("m"), 1e307, eqiora.Dimension(time=1)),
        (Decimal("1e-326"), u.s.prefixed("k"), 1e-323, eqiora.Dimension(time=1)),
        (Decimal("998.2"), u.kg / u.m**3, 998.2, eqiora.Dimension(mass=1, length=-3)),
        (Decimal("-2.5"), u.N.prefixed("m"), -0.0025, eqiora.Dimension(mass=1, length=1, time=-2)),
        (0.1, u.s.prefixed("m"), 0.0001, eqiora.Dimension(time=1)),
        (10, u.m.prefixed("c"), 0.1, eqiora.Dimension(length=1)),
    )
    with localcontext() as context:
        context.prec = 2
        for value, unit, expected, dimension in cases:
            actual, actual_dimension = unit.normalize(value)
            assert actual.hex() == expected.hex()
            assert actual_dimension == dimension
    value, dimension = u.s.normalize(Decimal("-0"))
    assert value.hex() == 0.0.hex()
    assert dimension == eqiora.Dimension(time=1)


def test_numeric_input_conversion_rejects_invalid_values_before_use():
    for value in (True, Fraction(1, 3), "1.0", None):
        with pytest.raises(TypeError):
            u.s.normalize(value)
    for value in (Decimal("NaN"), Decimal("Infinity"), float("nan"), float("inf"),
                  Decimal("1e1000"), Decimal("1e-1000"), Decimal("1" * 257), 10**256):
        with pytest.raises(ValueError):
            u.s.normalize(value)
    with pytest.raises(ValueError, match="scale root"):
        (u.m.prefixed("m") ** Fraction(1, 2)).normalize(1)
