"""Typed numeric input adapters and bounded selected-observation exports."""
import csv
import io
from decimal import Decimal, InvalidOperation

from ._eqiora import ResolvedArray, _export_observations_csv
from .units import Unit

__all__ = ["ResolvedArray", "csv_array", "export_observations_csv"]


def csv_array(text, *, columns, delimiter=","):
    """Normalize explicitly mapped CSV columns, returning (array, dimensions).

    ``columns`` is an ordered sequence of ``(header, Unit)`` pairs. Values
    become coherent-SI binary64 only after exact decimal/unit normalization.
    Header names are exact and unique. Missing cells/rows, malformed quoting,
    inconsistent row widths and nonnumeric selected cells reject. Other
    columns remain uninterpreted. Row order is preserved; no sorting, filling,
    deduplication, time alignment or interpolation is performed.

    The array has shape ``(rows, columns)``. Its identity covers normalized
    values and shape; the returned dimensions follow the selected column order.
    Keep raw sources and mapping/attribution with the consuming package when
    source provenance is required. Property declarations own interpolation,
    duplicate-axis admission, derivative rules and extrapolation.
    """
    if not isinstance(text, str):
        raise TypeError("CSV input must be text decoded by the caller")
    selected = tuple(columns)
    if not selected:
        raise ValueError("CSV requires at least one explicitly mapped column")
    names, units = [], []
    for entry in selected:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise TypeError("CSV columns must be (header, Unit) pairs")
        name, unit = entry
        if not isinstance(name, str) or not name:
            raise ValueError("CSV column names must be nonempty strings")
        if not isinstance(unit, Unit):
            raise TypeError("CSV columns require explicit Unit values")
        names.append(name)
        units.append(unit)
    if len(set(names)) != len(names):
        raise ValueError("CSV column mapping repeats a header")
    dimensions = tuple(unit.normalize(0)[1] for unit in units)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
    header = next(reader, None)
    if not header or any(not name for name in header) or len(set(header)) != len(header):
        raise ValueError("CSV requires a nonempty header with unique nonempty names")
    if any(name not in header for name in names):
        raise ValueError("CSV is missing a mapped column")
    positions = tuple(header.index(name) for name in names)
    values = []
    rows = 0
    for row in reader:
        rows += 1
        if len(row) != len(header):
            raise ValueError(f"CSV data row {rows} differs from the header width")
        for column, (position, unit) in enumerate(zip(positions, units), 1):
            cell = row[position]
            if not cell.strip():
                raise ValueError(f"CSV data row {rows}, selected column {column} is missing")
            try:
                decimal = Decimal(cell)
            except InvalidOperation as error:
                raise ValueError(f"CSV data row {rows}, selected column {column} is not decimal") from error
            value, _dimension = unit.normalize(decimal)
            values.append(value)
    return ResolvedArray.from_f64((rows, len(names)), values), dimensions


def export_observations_csv(path, observations, *, unit, projection="real", component=0,
                            decimal_places=6, locale="C"):
    """Write selected typed observation components through the shared native exporter.

    Order and duplicate occurrences are retained. Each row records exact Result,
    Observable, optional State/time and point support, intrinsic type/basis,
    selected original real/imaginary values, effective quadrature and display
    policy. Component mathematics and unit conversion use the existing owners.
    Phase requires units.one and explicitly records principal radians; zero phase
    is undefined. Primal observations only: directional provenance is unavailable.

    This is a lossy table, not a Result/restart artifact. Unselected components,
    equations, complete Mesh/Model and time history are not embedded. Keep the
    original Result for reopening. The native owner bounds complete output to
    65,536 rows and 16 MiB. A failed conversion or write preserves the destination.
    """
    if not isinstance(unit, Unit):
        raise TypeError("CSV export requires an explicit Unit")
    for name, value in (("component", component), ("decimal_places", decimal_places)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")
    return _export_observations_csv(path, observations, unit._ast, projection, component,
                                    decimal_places, locale)
