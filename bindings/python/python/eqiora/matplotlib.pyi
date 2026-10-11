"""Plot Eqiora results with Matplotlib.

Authority: ``bindings/python/python/eqiora/matplotlib.py``.
"""

from typing import overload

from matplotlib.figure import Figure

from . import FieldRef, Result
from .trajectory import DerivedFieldSnapshot, Trajectory

@overload
def plot_deformed_field(
    trajectory: Trajectory,
    /,
    *,
    step: int,
    field: FieldRef,
    scale: float = 1.0,
) -> Figure:
    """Compare exact reference and scaled-deformed support geometry.

    Authority: ``bindings/python/python/eqiora/matplotlib.py::plot_deformed_field``.
    """

    ...

@overload
def plot_deformed_field(
    result: Result,
    /,
    *,
    field: FieldRef,
    scale: float = 1.0,
) -> Figure:
    """Compare exact reference and scaled-deformed support geometry.

    Authority: ``bindings/python/python/eqiora/matplotlib.py::plot_deformed_field``.
    """

    ...

@overload
def plot_scalar_field(
    trajectory: Trajectory,
    /,
    *,
    step: int,
    field: FieldRef | DerivedFieldSnapshot,
) -> Figure:
    """Plot one accepted vertex field or cell-associated derived scalar.

    Authority: ``bindings/python/python/eqiora/matplotlib.py::plot_scalar_field``.
    """

    ...

@overload
def plot_scalar_field(
    result: Result,
    /,
    *,
    field: FieldRef,
) -> Figure:
    """Plot one accepted vertex field or cell-associated derived scalar.

    Authority: ``bindings/python/python/eqiora/matplotlib.py::plot_scalar_field``.
    """

    ...

__all__ = [
    "plot_deformed_field",
    "plot_scalar_field",
]
