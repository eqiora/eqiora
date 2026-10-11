"""Composable read-only Eqiora viewer with a lazily loaded Notebook host."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from functools import lru_cache
from importlib import resources
from typing import Any, Final

from ._eqiora import DerivedFieldSnapshot, FieldOutput, Geometry, Mesh, _compose_view

_TEXT_MIME: Final = "text/plain"
_WIDGET_MIME: Final = "application/vnd.jupyter.widget-view+json"
_SUPPORTED = (Geometry, Mesh, FieldOutput, DerivedFieldSnapshot)
_ViewValue = Geometry | Mesh | FieldOutput | DerivedFieldSnapshot


class View:
    """One disposable presentation scene over accepted Eqiora values.

    The scene transport is private and replaceable. Camera, visibility,
    selection, and colour state never enter accepted scientific identity.
    """

    __slots__ = ("_values", "_delegate", "_closed")

    def __init__(self) -> None:
        self._values: list[_ViewValue] = []
        self._delegate: object | None = None
        self._closed = False

    def add(self, value: _ViewValue, /) -> View:
        """Add one accepted typed value; semantic admission occurs in Rust."""

        if self._closed:
            raise RuntimeError("View is closed")
        if type(value) not in _SUPPORTED:
            raise TypeError(
                "View.add accepts only accepted Geometry, Mesh, scalar FieldOutput, or scalar DerivedFieldSnapshot values"
            )
        self._close_delegate()
        self._values.append(value)
        return self

    def show(self) -> View:
        """Display in a supported Notebook host, or print deterministic text."""

        try:
            from IPython.display import display
        except ModuleNotFoundError:
            print(repr(self))
        else:
            display(self)
        return self

    def close(self) -> None:
        """Release the optional widget delegate and accepted-object references."""

        if self._closed:
            return
        self._closed = True
        self._close_delegate()
        self._values.clear()

    def __enter__(self) -> View:
        if self._closed:
            raise RuntimeError("View is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        kinds = ", ".join(type(value).__name__ for value in self._values)
        return f"View(layers=[{kinds}], closed={self._closed})"

    def _repr_mimebundle_(
        self,
        include: object = None,
        exclude: object = None,
    ) -> dict[str, object] | tuple[dict[str, object], dict[str, object]]:
        selected = _selected_mime_types(include, exclude)
        if not selected:
            return {}
        text = repr(self)
        if self._closed:
            return {_TEXT_MIME: f"{text}\nViewer unavailable: this View is closed."}
        if _WIDGET_MIME not in selected:
            return {_TEXT_MIME: text} if _TEXT_MIME in selected else {}
        try:
            scene = _compose_view(tuple(self._values))
            delegate = self._delegate
            if delegate is None:
                widget_type = _widget_type()
                delegate = widget_type(
                    scene_metadata=scene.metadata_json,
                    buffers=list(scene.buffers),
                    _eqiora_view_id=uuid.uuid4().hex,
                )
                self._delegate = delegate
            bundle = delegate._repr_mimebundle_(include=include, exclude=exclude)  # type: ignore[attr-defined]
        except Exception as error:
            diagnostic = f"Viewer unavailable: {error}"
            return {_TEXT_MIME: f"{text}\n{diagnostic}"} if _TEXT_MIME in selected else {}
        return _with_text(bundle, text) if _TEXT_MIME in selected else bundle

    def _close_delegate(self) -> None:
        delegate = self._delegate
        self._delegate = None
        if delegate is None:
            return
        try:
            delegate.close()  # type: ignore[attr-defined]
        except Exception:
            pass


def _selected_mime_types(include: object, exclude: object) -> set[str]:
    selected = {_TEXT_MIME, _WIDGET_MIME}
    if include is not None:
        if isinstance(include, (str, bytes)):
            raise TypeError("include must be None or a collection of MIME strings")
        try:
            included = set(include)  # type: ignore[arg-type]
        except TypeError as error:
            raise TypeError("include must be None or a collection of MIME strings") from error
        if any(type(value) is not str for value in included):
            raise TypeError("include must be None or a collection of MIME strings")
        selected.intersection_update(included)
    if exclude is not None:
        if isinstance(exclude, (str, bytes)):
            raise TypeError("exclude must be None or a collection of MIME strings")
        try:
            excluded = set(exclude)  # type: ignore[arg-type]
        except TypeError as error:
            raise TypeError("exclude must be None or a collection of MIME strings") from error
        if any(type(value) is not str for value in excluded):
            raise TypeError("exclude must be None or a collection of MIME strings")
        selected.difference_update(excluded)
    return selected


def _with_text(
    bundle: object,
    value: str,
) -> dict[str, object] | tuple[dict[str, object], dict[str, object]]:
    if isinstance(bundle, tuple):
        if (
            len(bundle) != 2
            or not isinstance(bundle[0], Mapping)
            or not isinstance(bundle[1], Mapping)
        ):
            raise TypeError("viewer returned an invalid MIME bundle")
        data = dict(bundle[0])
        data[_TEXT_MIME] = value
        return data, dict(bundle[1])
    if not isinstance(bundle, Mapping):
        raise TypeError("viewer returned an invalid MIME bundle")
    data = dict(bundle)
    data[_TEXT_MIME] = value
    return data


@lru_cache(maxsize=1)
def _widget_type() -> type[Any]:
    import anywidget
    import traitlets

    esm, css = _load_assets()

    class _EqioraViewerWidget(anywidget.AnyWidget):
        _esm = esm
        _css = css

        scene_metadata = traitlets.Unicode().tag(sync=True)
        buffers = traitlets.List(traitlets.Bytes()).tag(sync=True)
        _eqiora_view_id = traitlets.Unicode().tag(sync=True)

        def __init__(self, **values: object) -> None:
            self._eqiora_sealed = False
            super().__init__(**values)
            self._eqiora_values = {
                name: getattr(self, name)
                for name in ("scene_metadata", "buffers", "_eqiora_view_id")
            }
            self._eqiora_sealed = True

        @traitlets.validate("scene_metadata", "buffers", "_eqiora_view_id")
        def _immutable_scene(self, proposal: dict[str, Any]) -> object:
            if getattr(self, "_eqiora_sealed", False):
                name = proposal["trait"].name
                if proposal["value"] != self._eqiora_values[name]:
                    raise traitlets.TraitError("Eqiora viewer scene payload is immutable")
            return proposal["value"]

    return _EqioraViewerWidget


def _load_assets() -> tuple[str, str]:
    static = resources.files("eqiora._viewer").joinpath("static")
    esm = static.joinpath("viewer.mjs").read_text(encoding="utf-8")
    css = static.joinpath("viewer.css").read_text(encoding="utf-8")
    if not esm.strip() or not css.strip():
        raise RuntimeError("installed Eqiora viewer assets are empty")
    return esm, css


__all__ = ["View"]
