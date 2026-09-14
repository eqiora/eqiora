"""Matplotlib presentation adapters for accepted Eqiora results."""

import math

import numpy as np

try:
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
except ModuleNotFoundError as error:
    if error.name not in {"matplotlib", "matplotlib.figure"}:
        raise
    raise ImportError(
        "eqiora.matplotlib requires the optional 'matplotlib' dependency; "
        "install eqiora[matplotlib]"
    ) from error

from ._eqiora import DerivedFieldSnapshot, FieldRef, Result, Trajectory
from .fluid import steady_stokes_evidence
from .solid import linear_elasticity_evidence

__all__ = [
    "plot_deformed_field",
    "plot_scalar_field",
]

_MISSING = object()


def plot_scalar_field(
    trajectory: Result | Trajectory,
    /,
    *,
    step=_MISSING,
    field: FieldRef | DerivedFieldSnapshot,
) -> Figure:
    """Plot one accepted vertex field or cell-associated derived scalar."""

    if isinstance(trajectory, Result):
        if step is not _MISSING:
            raise TypeError("plot_scalar_field() does not accept step for Result")
        if isinstance(field, DerivedFieldSnapshot):
            raise TypeError(
                "plot_scalar_field() requires a Trajectory for a derived snapshot"
            )
        evidence = steady_stokes_evidence(trajectory)
        bounds = evidence.exact_bounds
        minimum = evidence.pressure_minimum
        maximum = evidence.pressure_maximum
        output = trajectory.output(field)
        spatial = output.mesh
        state = None
        scalar_kind = "vertex"
        scalar_label = _result_scalar_label(output.dimension)
    elif isinstance(trajectory, Trajectory):
        if step is _MISSING:
            raise TypeError("plot_scalar_field() requires step for Trajectory")
        state = trajectory.state(step)
        spatial = trajectory
        bounds = None
        if isinstance(field, DerivedFieldSnapshot):
            snapshot = field
            _validate_derived_snapshot(trajectory, state, snapshot)
            scalar_label = f"Vorticity [{_coherent_si_unit(snapshot.dimension)}]"
            scalar_kind = "cell"
        else:
            snapshot = state.field(field)
            scalar_label = f"Value [{_coherent_si_unit(snapshot.dimension)}]"
            scalar_kind = "vertex"
    else:
        raise TypeError(
            "plot_scalar_field() requires eqiora.Result or eqiora.trajectory.Trajectory"
        )

    if state is None:
        if output.value_shape != ():
            raise ValueError("plot_scalar_field() requires scalar value shape ()")
        coordinates, triangles, values, support = _static_field_arrays(output, cell_arity=3)
        restricted = values[support]
    else:
        if scalar_kind == "cell":
            coordinates, triangles, values, support = _cell_field_arrays(
                spatial, snapshot, cell_arity=3
            )
            restricted = values
        else:
            if snapshot.value_shape != ():
                raise ValueError("plot_scalar_field() requires scalar value shape ()")
            if snapshot.frame != "invariant":
                raise ValueError("plot_scalar_field() requires the invariant frame")
            coordinates, triangles, values, support = _vertex_field_arrays(
                spatial, snapshot, cell_arity=3
            )
            restricted = values[support]
    if state is not None:
        minimum = float(restricted.min())
        maximum = float(restricted.max())

    figure = Figure(figsize=(8.0, 5.2), facecolor="#ffffff")
    axes = figure.add_axes((0.09, 0.13, 0.76, 0.76))
    axes.set_facecolor("#f8fafc")
    if state is not None and scalar_kind == "cell":
        magnitude = max(abs(minimum), abs(maximum))
        scalar = axes.tripcolor(
            coordinates[:, 0],
            coordinates[:, 1],
            triangles=triangles,
            facecolors=values,
            shading="flat",
            cmap="coolwarm",
            vmin=-magnitude if magnitude > 0.0 else None,
            vmax=magnitude if magnitude > 0.0 else None,
        )
    else:
        scalar = axes.tripcolor(
            coordinates[:, 0],
            coordinates[:, 1],
            values,
            triangles=triangles,
            shading="gouraud",
            cmap="viridis",
            vmin=minimum,
            vmax=maximum,
        )
    axes.triplot(
        coordinates[:, 0],
        coordinates[:, 1],
        triangles,
        color="#0f172a",
        linewidth=0.35,
        alpha=0.3,
    )
    if state is not None:
        if scalar_kind == "cell":
            _set_field_axes_bounds(axes, coordinates[np.unique(triangles)])
            axes.set_title(f"Vorticity — step {state.step}, t = {state.time_s:g} s")
        else:
            _set_field_axes_bounds(axes, coordinates[support])
            axes.set_title(f"Scalar field — step {state.step}, t = {state.time_s:g} s")
    else:
        (x_minimum, x_maximum), (y_minimum, y_maximum) = bounds
        axes.set_xlim(x_minimum, x_maximum)
        axes.set_ylim(y_minimum, y_maximum)
        _finish_field_axes(axes)
        axes.set_title("Scalar field")
    colorbar = figure.colorbar(scalar, ax=axes, fraction=0.03, pad=0.04)
    colorbar.set_label(scalar_label)
    return figure


def plot_deformed_field(
    trajectory: Result | Trajectory,
    /,
    *,
    step=_MISSING,
    field: FieldRef,
    scale: float = 1.0,
) -> Figure:
    """Compare exact reference and scaled-deformed support geometry."""

    scale = float(scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("plot_deformed_field() scale must be finite and nonnegative")

    if isinstance(trajectory, Result):
        if step is not _MISSING:
            raise TypeError("plot_deformed_field() does not accept step for Result")
        # The current static vector arm is the accepted structural Result. This
        # typed selection rejects scalar and temporal Results before rendering.
        linear_elasticity_evidence(trajectory)
        output = trajectory.output(field)
        spatial = output.mesh
        state = None
        cell_arity = 4
    elif isinstance(trajectory, Trajectory):
        if step is _MISSING:
            raise TypeError("plot_deformed_field() requires step for Trajectory")
        state, snapshot = _trajectory_snapshot(trajectory, step, field)
        spatial = trajectory
        cell_arity = 3
    else:
        raise TypeError(
            "plot_deformed_field() requires eqiora.Result or eqiora.trajectory.Trajectory"
        )

    if state is None:
        if output.value_shape != (spatial.dimension,):
            raise ValueError("plot_deformed_field() value shape must match the spatial dimension")
        if output.dimension != (0, 1, 0, 0, 0, 0, 0):
            raise ValueError("plot_deformed_field() requires the SI length dimension")
        coordinates, cells, values, support = _static_field_arrays(
            output, cell_arity=cell_arity
        )
    else:
        if snapshot.value_shape != (spatial.dimension,):
            raise ValueError(
                "plot_deformed_field() value shape must match the spatial dimension"
            )
        if snapshot.frame != "spatial-cartesian":
            raise ValueError("plot_deformed_field() requires the spatial-cartesian frame")
        if snapshot.dimension != (0, 1, 0, 0, 0, 0, 0):
            raise ValueError("plot_deformed_field() requires the SI length dimension")
        coordinates, cells, values, support = _vertex_field_arrays(
            spatial, snapshot, cell_arity=cell_arity
        )
    edges = _cell_edges(cells)
    edge_indices = list(edges)
    deformed = coordinates + scale * values

    figure = Figure(figsize=(7.0, 6.0), facecolor="#ffffff")
    axes = figure.add_axes((0.11, 0.12, 0.84, 0.79))
    axes.set_facecolor("#f8fafc")
    axes.add_collection(
        LineCollection(
            coordinates[edge_indices],
            colors="#64748b",
            linewidths=0.7,
            linestyles="dashed",
            alpha=0.72,
            label="Reference mesh",
        )
    )
    axes.add_collection(
        LineCollection(
            deformed[edge_indices],
            colors="#c2410c",
            linewidths=1.15,
            alpha=0.95,
            label=f"Deformed mesh (scale = {scale:g})",
        )
    )

    _set_field_axes_bounds(
        axes,
        np.concatenate((coordinates[support], deformed[support])),
    )
    if state is None:
        axes.set_title(f"Deformed field — scale {scale:g}")
    else:
        axes.set_title(
            f"Deformed field — step {state.step}, t = {state.time_s:g} s, scale {scale:g}"
        )
    axes.legend(loc="upper right")
    return figure


def _trajectory_snapshot(trajectory, step, field):
    if not isinstance(trajectory, Trajectory):
        raise TypeError("field stills require eqiora.trajectory.Trajectory")
    state = trajectory.state(step)
    return state, state.field(field)


def _validate_derived_snapshot(trajectory, state, snapshot):
    if snapshot.source_state_digest != state.digest:
        raise ValueError("derived snapshot belongs to a different accepted State")
    if snapshot.mesh_digest != trajectory.mesh_digest:
        raise ValueError("derived snapshot belongs to a different exact Mesh")
    if snapshot.operator != "curl":
        raise ValueError("plot_scalar_field() does not support this derived operator")
    if snapshot.value_shape != ():
        raise ValueError("plot_scalar_field() requires scalar value shape ()")
    if snapshot.frame != "spatial-axial":
        raise ValueError("derived curl requires the spatial-axial frame")


def _cell_field_arrays(spatial, snapshot, *, cell_arity):
    if snapshot.associations != ("cell",):
        raise ValueError("derived field stills require exactly one cell association")

    coordinates = spatial.coordinates
    cells = spatial.cells
    values = snapshot.values("cell")
    support = snapshot.support_indices("cell")
    if spatial.dimension != 2 or coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("field stills require two-dimensional coordinates")
    if cells.ndim != 2 or cells.shape[1] != cell_arity:
        raise ValueError("field stills require affine triangle topology")
    if support.ndim != 1 or support.size == 0:
        raise ValueError("cell support must be one-dimensional and nonempty")
    if int(support.max()) >= cells.shape[0]:
        raise ValueError("cell support exceeds the spatial topology")
    if not np.array_equal(support, np.unique(support)):
        raise ValueError("cell support must be sorted and unique")
    if cells.size > 0 and int(cells.max()) >= coordinates.shape[0]:
        raise ValueError("cell topology exceeds the spatial coordinates")
    admitted_cells = cells[support]
    if values.shape == (cells.shape[0],):
        values = values[support]
    elif values.shape != (support.shape[0],):
        raise ValueError("field value shape does not match its cell support")
    return coordinates, admitted_cells, values, support


def _static_field_arrays(output, *, cell_arity):
    spatial = output.mesh
    coordinates = spatial.coordinates
    cells = spatial.cells
    vertex_count = output.coefficient_count("vertex")
    values = output.values("vertex").numpy(copy=False)
    if output.value_shape:
        values = values.reshape(vertex_count, *output.value_shape)
    support = np.arange(vertex_count, dtype=np.uint32)
    if spatial.dimension != 2 or coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("field stills require two-dimensional coordinates")
    if cells.ndim != 2 or cells.shape[1] != cell_arity:
        topology = "affine triangle" if cell_arity == 3 else "quadrilateral"
        raise ValueError(f"field stills require {topology} topology")
    if values.shape[0] != vertex_count:
        raise ValueError("vertex values differ from their declared support")
    return coordinates, cells, values, support


def _vertex_field_arrays(spatial, snapshot, *, cell_arity):
    if snapshot.associations != ("vertex",):
        raise ValueError("field stills require exactly one vertex association")

    coordinates = spatial.coordinates
    cells = spatial.cells
    values = snapshot.values("vertex")
    support = snapshot.support_indices("vertex")
    if spatial.dimension != 2 or coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("field stills require two-dimensional coordinates")
    if cells.ndim != 2 or cells.shape[1] != cell_arity:
        if cell_arity == 3:
            raise ValueError("field stills require affine triangle topology")
        if cell_arity == 4:
            raise ValueError("field stills require quadrilateral topology")
        raise ValueError("field stills received an unsupported topology contract")
    if support.ndim != 1 or support.size == 0:
        raise ValueError("vertex support must be one-dimensional and nonempty")
    if int(support.max()) >= coordinates.shape[0]:
        raise ValueError("vertex support exceeds the spatial coordinates")
    if not np.array_equal(support, np.unique(support)):
        raise ValueError("vertex support must be sorted and unique")
    if values.shape[0] == support.shape[0]:
        expanded_shape = (coordinates.shape[0], *values.shape[1:])
        expanded = np.zeros(expanded_shape, dtype=values.dtype)
        expanded[support] = values
        values = expanded
    elif values.shape[0] != coordinates.shape[0]:
        raise ValueError("field value shape does not match its vertex support")
    if cells.size > 0 and int(cells.max()) >= coordinates.shape[0]:
        raise ValueError("cell topology exceeds the spatial coordinates")

    inside = np.isin(cells, support)
    admitted_cells = cells[np.all(inside, axis=1)]
    if admitted_cells.size == 0:
        raise ValueError("vertex support admits no complete cell")
    if not np.array_equal(np.unique(admitted_cells), support):
        raise ValueError("admitted cell closure differs from vertex support")
    return coordinates, admitted_cells, values, support


def _set_field_axes_bounds(axes, points):
    x_minimum = float(points[:, 0].min())
    x_maximum = float(points[:, 0].max())
    y_minimum = float(points[:, 1].min())
    y_maximum = float(points[:, 1].max())
    padding = 0.05 * max(x_maximum - x_minimum, y_maximum - y_minimum, 1.0)
    axes.set_xlim(x_minimum - padding, x_maximum + padding)
    axes.set_ylim(y_minimum - padding, y_maximum + padding)
    _finish_field_axes(axes)


def _finish_field_axes(axes):
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x [m]")
    axes.set_ylabel("y [m]")


def _result_scalar_label(dimension):
    if dimension == (1, -1, -2, 0, 0, 0, 0):
        return "Pressure [Pa]"
    return f"Value [{_coherent_si_unit(dimension)}]"


def _coherent_si_unit(dimension):
    terms = [
        base if exponent == 1 else f"{base}^{exponent}"
        for base, exponent in zip(
            ("kg", "m", "s", "A", "K", "mol", "cd"),
            dimension,
            strict=True,
        )
        if exponent != 0
    ]
    return "·".join(terms) if terms else "1"


def _quadrilateral_edges(cells):
    edges = {
        tuple(sorted((int(cell[first]), int(cell[second]))))
        for cell in cells
        for first, second in ((0, 1), (1, 3), (3, 2), (2, 0))
    }
    return tuple(sorted(edges))


def _triangle_edges(cells):
    edges = {
        tuple(sorted((int(cell[first]), int(cell[second]))))
        for cell in cells
        for first, second in ((0, 1), (1, 2), (2, 0))
    }
    return tuple(sorted(edges))


def _cell_edges(cells):
    if cells.shape[1] == 3:
        return _triangle_edges(cells)
    if cells.shape[1] == 4:
        return _quadrilateral_edges(cells)
    raise ValueError("field stills require triangle or quadrilateral topology")
