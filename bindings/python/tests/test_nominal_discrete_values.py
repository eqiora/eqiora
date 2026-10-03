"""Finite basis and index values preserve exact declarations, shape, and integers."""

import pytest

import eqiora


q = eqiora.lang


def native_model(name, *declarations):
    observed = eqiora.Field("observed", role=eqiora.FieldRole.Variable, value_type=eqiora.ValueType.real())
    return eqiora.compile(source=eqiora.Module(name, *declarations, observed,
                               eqiora.Relation("observe", equations=[(observed, 0)])))


def add_observation(owner):
    observed = owner.field("observed", role=eqiora.FieldRole.Variable, value_type=eqiora.ValueType.real())
    owner.relation("observe", eqiora.lang.equation(observed, 0))


def test_native_finite_space_values_preserve_identity_and_exact_components():
    space = eqiora.FiniteSpace("Species", labels=("A", "B"))
    foreign = eqiora.FiniteSpace("Species", labels=("A", "B"))
    assert space != foreign
    assert len({space, foreign}) == 2
    assert space.labels == ("A", "B")
    counts = eqiora.ValueType.counts(space)
    coordinates = eqiora.ValueType.coordinates(eqiora.ValueType.integer(), space)
    assert counts != coordinates
    assert counts != eqiora.ValueType.counts(foreign)
    assert counts.shape == [2] and counts.array_rank == 0
    values = (2**53 + 1, 2**53 + 2)
    parameter = eqiora.Parameter("population", value_type=counts, value=values)
    model = native_model("Population", space, parameter)
    reference = model.parameter("population")
    assert reference.value_type == counts
    assert reference.value == values
    replayed = eqiora.Model.from_bytes(model.to_bytes())
    assert replayed.digest == model.digest
    assert replayed.parameter(reference.id).value_type == counts
    assert replayed.parameter(reference.id).value == values
    changed = model.commit(model.preview_value_edit("population", (values[0] + 1, values[1])))
    assert changed.parameter("population").value == (2**53 + 2, 2**53 + 2)
    assert reference.value == values
    signed = eqiora.Parameter("change", value_type=coordinates, value=(-1, 1))
    assert signed.value == (-1, 1)
    for invalid in ((-1, 2), (1,), (True, 2), (1.0, 2), 1):
        with pytest.raises((TypeError, ValueError, OverflowError)):
            eqiora.Parameter("invalid", value_type=counts, value=invalid)
    with pytest.raises(eqiora.ValidationError, match="(?i)nominal|registered|scope|space|index"):
        native_model("Foreign", foreign, parameter)
    with pytest.raises(ValueError):
        counts.to_eqi()
    with pytest.raises(ValueError):
        eqiora.ValueType.array(counts, 2)
    with pytest.raises(AttributeError):
        space.labels = ("C",)


def test_native_index_set_values_are_exact_bounded_and_nominal():
    rows = eqiora.IndexSet("Rows", extent=3)
    other = eqiora.IndexSet("Rows", extent=3)
    assert rows != other and len({rows, other}) == 2
    kind = eqiora.ValueType.index(rows)
    assert kind != eqiora.ValueType.index(other)
    parameter = eqiora.Parameter("selected", value_type=kind, value=2)
    model = native_model("Index", rows, parameter)
    reference = model.parameter("selected")
    assert reference.value == 2
    assert reference.value_type == kind
    replayed = eqiora.Model.from_bytes(model.to_bytes())
    assert replayed.digest == model.digest
    assert replayed.parameter(reference.id).value_type == kind
    assert replayed.parameter(reference.id).value == 2
    for invalid in (-1, 3, True, 1.0):
        with pytest.raises((TypeError, ValueError, OverflowError)):
            model.preview_value_edit("selected", invalid)
    for invalid in (0, -1, True, 1.0, 2**32):
        with pytest.raises((TypeError, ValueError, OverflowError)):
            eqiora.IndexSet("Invalid", extent=invalid)
    with pytest.raises(eqiora.ValidationError, match="(?i)nominal|registered|scope|space|index"):
        native_model("Foreign", other, parameter)


def test_source_nominal_constructors_share_scope_and_file_meaning(tmp_path):
    source = eqiora.Module("main")
    species = source.space("Species", labels=("A", "B"), doc="Ordered species basis.")
    alternate = source.space("Alternate", labels=("A", "B"))
    owner = source.model("Population")
    add_observation(owner)
    rows = owner.index_set("Rows", extent=3, doc="Three fixed rows.")
    populations = owner.parameter("population", value_type=eqiora.ValueType.counts(species))
    other = owner.parameter("other", value_type=eqiora.ValueType.counts(alternate))
    added = owner.parameter("added", value_type=eqiora.ValueType.counts(species))
    selected = owner.parameter("selected", value_type=eqiora.ValueType.index(rows))
    change = owner.parameter("change", value_type=eqiora.ValueType.coordinates(eqiora.ValueType.integer(), species))
    ordinal = owner.parameter("ordinal", value_type=eqiora.ValueType.integer())
    owner.set_default(populations, owner.counts(species, (2**53 + 1, 2)))
    owner.set_default(other, owner.counts(alternate, (2**53 + 1, 2)))
    owner.set_default(added, populations + owner.coordinates(species, (1, 0)))
    owner.set_default(selected, owner.index(rows, 2))
    owner.set_default(change, owner.coordinates(species, (-1, 1)))
    owner.set_default(ordinal, q.ordinal(selected))
    text = source.to_eqi()
    assert "/// Ordered species basis.\npublic space Species = orthonormal(A, B);" in text
    assert "/// Three fixed rows.\n  indexset Rows = range(3);" in text
    assert "counts(Species, [9007199254740993, 2])" in text
    compiled = eqiora.compile(source=source, entry="Population")
    path = tmp_path / "population.eqi"
    source.write_eqi(path)
    replayed_source = eqiora.compile(path=path, entry="Population")
    assert replayed_source.to_bytes() == compiled.to_bytes()
    assert compiled.parameter("population").value == (2**53 + 1, 2)
    assert compiled.parameter("added").value == (2**53 + 2, 2)
    assert compiled.parameter("population").value_type != compiled.parameter("other").value_type
    assert compiled.parameter("selected").value == compiled.parameter("ordinal").value == 2
    assert compiled.parameter("change").value == (-1, 1)


def test_source_nominal_ownership_and_constant_extent_reject_before_mutation():
    source = eqiora.Module("main")
    foreign_source = eqiora.Module("main")
    species = source.space("Species", labels=("A", "B"))
    foreign = foreign_source.space("Species", labels=("A", "B"))
    left = source.component("Left")
    right = source.component("Right")
    rows = left.index_set("Rows", extent=3)
    right_rows = right.index_set("Rows", extent=3)
    n = left.parameter("n", value_type=eqiora.ValueType.integer())
    with pytest.raises(TypeError, match="constant"):
        left.index_set("Deferred", extent=n)
    left.index_set("Deferred", extent=2)
    with pytest.raises(q.ModuleError):
        left.parameter("foreign", value_type=eqiora.ValueType.counts(foreign))
    left.parameter("foreign", value_type=eqiora.ValueType.counts(species))
    with pytest.raises(q.ModuleError):
        right.parameter("foreign_index", value_type=eqiora.ValueType.index(rows))
    right.parameter("foreign_index", value_type=eqiora.ValueType.index(right_rows))
    for create in (left.counts, left.coordinates):
        with pytest.raises(q.ModuleError, match="Module"):
            create(foreign, (1, 2))
        with pytest.raises(q.ModuleError, match="Component"):
            right.let_alias("capture", create(species, (1, 2)))
    with pytest.raises(q.ModuleError, match="Component"):
        right.index(rows, 1)
    with pytest.raises(q.ModuleError, match="Component"):
        right.index(right_rows, n)
    source.to_eqi()
    with pytest.raises(q.ModuleError, match="frozen"):
        source.space("Frozen", labels=("A",))
    with pytest.raises(q.ModuleError, match="frozen"):
        left.index_set("Frozen", extent=1)


def test_source_counts_reject_equal_shaped_foreign_basis_arithmetic():
    source = eqiora.Module("main")
    first = source.space("First", labels=("A", "B"))
    second = source.space("Second", labels=("A", "B"))
    owner = source.model("WrongBasis")
    add_observation(owner)
    value = owner.parameter("value", value_type=eqiora.ValueType.counts(first))
    owner.set_default(value, owner.counts(first, (1, 2)) + owner.coordinates(second, (3, 4)))
    with pytest.raises(eqiora.ValidationError, match="(?i)nominal|space|basis|type") as error:
        eqiora.compile(source=source, entry="WrongBasis")
    assert "EQ0604" not in str(error.value)
