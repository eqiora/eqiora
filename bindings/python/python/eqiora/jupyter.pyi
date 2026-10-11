"""IPython extension that compiles `%%eqiora` notebook cells.

Authority: ``bindings/python/python/eqiora/jupyter.py``.
"""

from typing import Any

def load_ipython_extension(ipython: Any) -> None:
    """Register the `%%eqiora` cell magic; called by `%load_ext eqiora.jupyter`.

    Authority: ``bindings/python/python/eqiora/jupyter.py::load_ipython_extension``.
    """

def unload_ipython_extension(ipython: Any) -> None:
    """Remove the cell magic. Models that were already compiled stay available.

    Authority: ``bindings/python/python/eqiora/jupyter.py::unload_ipython_extension``.
    """

__all__ = ["load_ipython_extension", "unload_ipython_extension"]
