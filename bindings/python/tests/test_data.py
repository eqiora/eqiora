"""Numeric format adapters reuse the native array codec and unit arithmetic."""
import csv
import hashlib
import json
import math
from decimal import localcontext

import pytest
import numpy as np

import eqiora as q
from eqiora.data import ResolvedArray, csv_array
from eqiora import units as u


def test_resolved_array_canonical_identity_replay_and_owned_values():
    source = [-0., 1.5, -2., 4.]
    array = ResolvedArray.from_f64((2, 2), source)
    source[1] = 99.
    expected = (b'{"schema":"eqiora.resolved-array/v1","encoding":"eqiora.canonical-json/v1",'
                b'"scalar":"f64","shape":[2,2],"values":[0.0,1.5,-2.0,4.0]}')
    assert array.to_bytes() == expected
    assert array.digest == hashlib.sha256(b"eqiora.resolved-array/v1\0" + expected).hexdigest()
    assert array.values == (0., 1.5, -2., 4.)
    assert math.copysign(1., array.values[0]) == 1.
    assert array.shape == (2, 2) and array.scalar == "f64"
    restored = ResolvedArray.from_bytes(expected)
    assert restored.to_bytes() == expected and restored.digest == array.digest
    assert ResolvedArray.from_f64((4,), array.values).digest != array.digest
    assert ResolvedArray.from_f64((2, 2), (0., 1.5, -2., 5.)).digest != array.digest
    with pytest.raises(AttributeError):
        array.shape = (4,)
    integers = ResolvedArray.from_u64((2,), (2**53+1, 2**64-1))
    assert integers.values == (2**53+1, 2**64-1)
    assert integers.scalar == "u64"
    assert ResolvedArray.from_bytes(integers.to_bytes()).values == integers.values


@pytest.mark.parametrize("shape,values", [([], []), ([0], []), ([2], [1.]),
                                         ([1], [math.nan]), ([1], [math.inf])])
def test_resolved_array_uses_native_shape_and_finite_value_admission(shape, values):
    with pytest.raises(q.EqioraError):
        ResolvedArray.from_f64(shape, values)


@pytest.mark.parametrize("value", [True, 1+2j, np.complex128(1+2j)])
def test_real_array_never_discards_a_scalar_domain(value):
    with pytest.raises(TypeError):
        ResolvedArray.from_f64((1,), [value])


@pytest.mark.parametrize("value", [True, 1., 1+0j])
def test_array_integer_components_require_exact_integers(value):
    with pytest.raises(TypeError):
        ResolvedArray.from_u64((1,), [value])
    with pytest.raises(TypeError):
        ResolvedArray.from_f64((value,), [0.])


@pytest.mark.parametrize("value", [-1, 2**64])
def test_unsigned_array_rejects_out_of_range_integers(value):
    with pytest.raises(OverflowError):
        ResolvedArray.from_u64((1,), [value])


def test_resolved_array_decoder_rejects_substituted_schema_and_noncanonical_zero():
    content = json.loads(ResolvedArray.from_f64((1,), (0.,)).to_bytes())
    for changed in (dict(content, schema="eqiora.resolved-array/v0"),
                    dict(content, values=[-0.]), dict(content, extra=1)):
        with pytest.raises(q.EqioraError):
            ResolvedArray.from_bytes(json.dumps(changed).encode())


def test_csv_columns_reorder_exact_units_and_preserve_sampling_order():
    text = 'note,force_mN,time_ms\r\n"quoted, text",2000,500\r\n"two\nlines",-500,125\r\n'
    array, dimensions = csv_array(text, columns=(("time_ms", u.s.prefixed("m")),
                                                ("force_mN", u.N.prefixed("m"))))
    assert array.shape == (2, 2)
    assert array.values == (.5, 2., .125, -.5)
    assert dimensions == (q.Dimension(time=1), q.Dimension(mass=1, length=1, time=-2))
    # No sorting or duplicate-time interpretation belongs to a generic numeric adapter.
    duplicate, _ = csv_array("t,x\n1,2\n1,3\n", columns=(("t", u.s), ("x", u.m)))
    assert duplicate.values == (1., 2., 1., 3.)
    changed, _ = csv_array(text.replace("2000", "2001"), columns=(("time_ms", u.s.prefixed("m")),
                                                                            ("force_mN", u.N.prefixed("m"))))
    assert changed.digest != array.digest
    semicolon, dims = csv_array("t;x\n500;2000\n", columns=(("t", u.s.prefixed("m")),
                                                          ("x", u.N.prefixed("m"))), delimiter=";")
    assert semicolon.values == (.5, 2.) and dims == dimensions


def test_csv_decimal_scaling_precedes_binary64_and_ignores_decimal_context():
    with localcontext() as context:
        context.prec = 2
        array, dimensions = csv_array("large,small\n1e310,1e-326\n", columns=(
            ("large", u.s.prefixed("m")), ("small", u.s.prefixed("k")),
        ))
    assert array.values == (1e307, 1e-323)
    assert dimensions == (q.Dimension(time=1), q.Dimension(time=1))


@pytest.mark.parametrize("text", ["", "x,x\n1,2\n", ",x\n1,2\n", "y\n1\n",
                                    "x\n", "x\n\n", "x,y\n1\n", "x\n1,2\n",
                                    "x\n \n", 'x\n"unterminated', "x\nnope\n",
                                    "x\nNaN\n", "x\nInfinity\n", "x\n1e400\n"])
def test_csv_rejects_ambiguous_missing_malformed_and_nonfinite_input(text):
    with pytest.raises((ValueError, csv.Error, q.EqioraError)):
        csv_array(text, columns=(("x", u.m),))


@pytest.mark.parametrize("columns", [(), (("x", u.m), ("x", u.s)), (("x", "m"),),
                                       (("", u.m),), ("x",)])
def test_csv_requires_explicit_unambiguous_column_mapping(columns):
    with pytest.raises((TypeError, ValueError)):
        csv_array("x\n1\n", columns=columns)
