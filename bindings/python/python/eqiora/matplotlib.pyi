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
    """Plot the reference geometry and the geometry deformed by a scaled displacement.

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
    """Plot the reference geometry and the geometry deformed by a scaled displacement.

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
    """Plot a vertex field or a derived per-cell scalar from a Result.

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
    """Plot a vertex field or a derived per-cell scalar from a Result.

    Authority: ``bindings/python/python/eqiora/matplotlib.py::plot_scalar_field``.
    """

    ...

__all__ = [
    "plot_deformed_field",
    "plot_scalar_field",
]
