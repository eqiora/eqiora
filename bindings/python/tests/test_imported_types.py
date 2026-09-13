"""Imported nominal type handles keep the exact explicit Module path."""

import pytest

import eqiora


q = eqiora.lang

LIBRARY = """
public record Sample { value: 1, }
record PrivateSample { value: 1, }
public enum Mode { Off, On }
enum PrivateMode { Off, On }
public space Species = orthonormal(A, B);
space PrivateSpecies = orthonormal(A, B);
"""


def test_imported_record_enum_and_space_compile_without_copying_declarations():
    source = eqiora.Module("main")
    types = source.import_module("types", eqiora.Module.parse("types", LIBRARY))
    sample = types.record("Sample")
    mode = types.enum("Mode")
    species = types.space("Species")
    assert isinstance(sample, q.ImportedRecord)
    assert not isinstance(sample, q.Record)

    consumer = source.component("Consumer")
    config = consumer.parameter("config", value_type=sample)
    selected = consumer.parameter("selected", value_type=mode.value_type)
    population = consumer.parameter(
        "population", value_type=eqiora.ValueType.counts(species)
    )
    consumer.set_default(config, sample(value=2))
    consumer.set_default(selected, mode.member("On"))
    consumer.set_default(population, consumer.counts(species, (3, 5)))
    result = consumer.output("result", value_type=eqiora.ValueType.real())
    consumer.relation("law", q.equation(result, config.member("value")))

    root = source.model("Main")
    occurrence = root.instance("consumer", component=consumer, bindings={})
    observed = root.output("result", value_type=eqiora.ValueType.real())
    root.relation("observe", q.equation(observed, occurrence["result"]))
    text = source.to_eqi()
    assert "parameter config: types.Sample = types.Sample(value = 2)" in text
    assert "parameter selected: types.Mode = types.Mode.On" in text
    assert "parameter population: counts<types.Species>" in text
    assert "counts(types.Species, [3, 5])" in text
    assert "record Sample" not in text
    assert "enum Mode" not in text
    assert "space Species" not in text
    assert eqiora.compile(source=source, entry="Main").digest


@pytest.mark.parametrize(
    ("method", "name"),
    (
        ("record", "PrivateSample"),
        ("record", "Missing"),
        ("enum", "PrivateMode"),
        ("enum", "Missing"),
        ("space", "PrivateSpecies"),
        ("space", "Missing"),
    ),
)
def test_imported_type_descriptors_reject_private_unknown_and_transitive_names(method, name):
    source = eqiora.Module("main")
    types = source.import_module("types", eqiora.Module.parse("types", LIBRARY))
    with pytest.raises(q.ModuleError, match="public"):
        getattr(types, method)(name)
    with pytest.raises(q.ModuleError):
        getattr(types, method)("other.Exported")


def test_equal_imported_nominal_shapes_from_distinct_modules_remain_distinct():
    source = eqiora.Module("main")
    left = source.import_module("left", eqiora.Module.parse("left", LIBRARY))
    right = source.import_module("right", eqiora.Module.parse("right", LIBRARY))
    component = source.component("Consumer")
    value = component.parameter("value", value_type=left.enum("Mode").value_type)
    component.set_default(value, right.enum("Mode").member("On"))
    output = component.output("output", value_type=eqiora.ValueType.real())
    component.relation("law", q.equation(output, 0))
    root = source.model("Main")
    occurrence = root.instance("consumer", component=component, bindings={})
    observed = root.output("output", value_type=eqiora.ValueType.real())
    root.relation("observe", q.equation(observed, occurrence["output"]))
    with pytest.raises(eqiora.ValidationError):
        eqiora.compile(source=source, entry="Main")
