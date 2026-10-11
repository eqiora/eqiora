"""Compile Eqiora notebook cells through the ordinary Python compiler.

Authority: ``bindings/python/python/eqiora/jupyter.py``.
"""

from typing import Any

def load_ipython_extension(ipython: Any) -> None:
    """Register the cell magic when IPython loads this extension explicitly.

    Authority: ``bindings/python/python/eqiora/jupyter.py::load_ipython_extension``.
    """

def unload_ipython_extension(ipython: Any) -> None:
    """Remove the cell magic while preserving compiled Python variables.

    Authority: ``bindings/python/python/eqiora/jupyter.py::unload_ipython_extension``.
    """

__all__ = ["load_ipython_extension", "unload_ipython_extension"]
