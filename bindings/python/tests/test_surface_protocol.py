"""Protocol obligations every exported type owes a Python caller.

These hold for the whole module rather than for a named list, so a type exposed
later inherits them. A binding that skips `__repr__` or defines `__eq__` without
`__hash__` is not caught by any test of what it computes; it is only felt, once,
by someone printing it at a prompt.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import eqiora


def exported_classes() -> list[tuple[str, type]]:
    return sorted(
        (name, value)
        for name, value in vars(eqiora).items()
        if inspect.isclass(value) and not name.startswith("_")
    )


CLASSES = exported_classes()
IDS = [name for name, _ in CLASSES]


def test_the_module_exports_classes_at_all() -> None:
    # Every obligation below is quantified over this list, so an empty list
    # would satisfy all of them without asserting anything.
    assert len(CLASSES) > 20


@pytest.mark.parametrize(("name", "cls"), CLASSES, ids=IDS)
def test_every_class_defines_a_repr(name: str, cls: type) -> None:
    assert cls.__repr__ is not object.__repr__, (
        f"{name} falls back to object.__repr__, which prints an address and no state"
    )


@pytest.mark.parametrize(("name", "cls"), CLASSES, ids=IDS)
def test_equality_implies_hashability(name: str, cls: type) -> None:
    # Python sets __hash__ to None when __eq__ is defined without it, which
    # removes the type from every dict and set silently.
    if cls.__eq__ is not object.__eq__:
        assert cls.__hash__ is not None, (
            f"{name} defines __eq__ but is unhashable, so it cannot be a dict key"
        )


@pytest.mark.parametrize(("name", "cls"), CLASSES, ids=IDS)
def test_every_class_is_documented(name: str, cls: type) -> None:
    assert (cls.__doc__ or "").strip(), f"{name} has no docstring"


@pytest.mark.parametrize(("name", "cls"), CLASSES, ids=IDS)
def test_every_class_is_declared_in_the_stub(name: str, cls: type) -> None:
    from pathlib import Path

    stub = Path(eqiora.__file__).with_name("__init__.pyi")
    contents = stub.read_text(encoding="utf-8")
    declarations = (f"class {name}", f"import {name} as {name}")
    assert any(declaration in contents for declaration in declarations), (
        f"{name} is exported but absent from __init__.pyi, so type checkers cannot see it"
    )


@pytest.mark.parametrize(("name", "cls"), CLASSES, ids=IDS)
def test_a_sized_class_can_also_be_read(name: str, cls: type) -> None:
    # Reporting a length while offering no way to reach the elements is the one
    # container shape Python has no idiom for. The array protocol counts: a
    # buffer that hands its contents to NumPy in one call does not also owe an
    # element-wise path, and offering one would hide a per-element copy.
    if hasattr(cls, "__len__"):
        assert (
            hasattr(cls, "__iter__")
            or hasattr(cls, "__getitem__")
            or hasattr(cls, "__array__")
        ), f"{name} defines __len__ but offers no way to read its elements"


def test_array_satisfies_the_numpy_protocol() -> None:
    assert hasattr(eqiora.Array, "__array__")
    assert hasattr(eqiora.Array, "__dlpack__")
    assert hasattr(eqiora.Array, "__dlpack_device__")


DECAY = """
model decay() {
  state x: 1;
  initial { x = 1; }
  parameter rate: 1 / s = 1;
  relation flow {
    derivative(x) + rate * x = 0;
  }
}
"""


def decayed_series() -> eqiora.Series:
    model = eqiora.compile(source=DECAY)
    field = model.field(model.field_ids[0])
    plan = eqiora.resolve(
        model,
        temporal=eqiora.time.Tsitouras45(
            initial_step_s=0.01,
            relative_tolerance=1.0e-9,
            absolute_tolerances={(field, 0, 0, False): 1.0e-11},
        ),
    )
    result = eqiora.run(
        plan,
        state=eqiora.State.initial(plan),
        until_s=0.2,
        output_times_s=(0.05, 0.1, 0.2),
    )
    return result.series(field)


def test_asarray_needs_no_method_call_and_stays_zero_copy() -> None:
    import numpy as np

    values = decayed_series().values
    viewed = np.asarray(values)
    assert viewed.dtype == np.float64
    # The zero-copy view is the read-only one, so asarray must not have copied.
    assert not viewed.flags.writeable
    assert viewed[0] == pytest.approx(np.exp(-0.05), rel=2.0e-8)


def test_asarray_honours_a_requested_dtype() -> None:
    import numpy as np

    # Silently returning float64 to a caller asking for float32 is the failure
    # mode of ignoring the dtype argument.
    assert np.asarray(decayed_series().values, dtype=np.float32).dtype == np.float32


def test_a_series_iterates_time_and_value_pairs() -> None:
    series = decayed_series()
    samples = list(series)
    assert len(samples) == len(series)
    assert samples[0] == (pytest.approx(0.05), pytest.approx(np.exp(-0.05), rel=2.0e-8))
    assert [time for time, _ in samples] == pytest.approx([0.05, 0.1, 0.2])
