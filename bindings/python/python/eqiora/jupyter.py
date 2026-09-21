"""Opt-in IPython execution of Eqiora source cells.

Load with ``%load_ext eqiora.jupyter``. The body of ``%%eqiora model`` is
ordinary Eqiora source; successful compilation binds ``model`` in the Python
namespace. This module does not install frontend syntax highlighting.
"""

from __future__ import annotations

import keyword
import unicodedata

from IPython.core.error import UsageError
from IPython.core.magic import Magics, cell_magic, magics_class, no_var_expand
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

import eqiora


def _identifier(value: str) -> str:
    name = unicodedata.normalize("NFKC", value)
    if not name.isidentifier() or keyword.iskeyword(name):
        raise UsageError(f"expected a Python variable name, got {value!r}")
    return name


def _diagnostic_notes(error: eqiora.EqioraError, source: str, filename: str) -> None:
    encoded = source.encode("utf-8")
    for diagnostic in error.diagnostics:
        span = diagnostic.source_span
        if span is None or span[0] != filename or not 0 <= span[1] <= len(encoded):
            continue
        before = encoded[:span[1]].decode("utf-8")
        # The magic header is notebook line 1; native spans remain body-relative.
        line = before.count("\n") + 2
        column = len(before.rsplit("\n", 1)[-1]) + 1
        error.add_note(
            f"{filename}:{line}:{column}: {diagnostic.code}: {diagnostic.message}"
        )


@magics_class
class _EqioraMagics(Magics):
    @cell_magic
    @no_var_expand
    @magic_arguments()
    @argument("variable", help="Python variable receiving the compiled Model")
    @argument("--entry", help="Model or Component selected by the ordinary compiler")
    @argument("--bindings", help="Python variable holding compiler bindings")
    @argument("--geometry", help="Python variable holding compiler Geometry")
    def eqiora(self, line: str, cell: str) -> None:
        """Compile an Eqiora cell, publishing its Model only on success."""
        args = parse_argstring(self.eqiora, line)
        name = _identifier(args.variable)
        options = {"entry": args.entry}
        for option in ("bindings", "geometry"):
            value = getattr(args, option)
            if value is not None:
                binding = _identifier(value)
                if binding not in self.shell.user_ns:
                    raise UsageError(f"Python variable {binding!r} is not defined")
                options[option] = self.shell.user_ns[binding]
        filename = f"In[{self.shell.execution_count}].eqi"
        try:
            model = eqiora.compile(source=cell, filename=filename, **options)
        except eqiora.EqioraError as error:
            _diagnostic_notes(error, cell, filename)
            raise
        self.shell.push({name: model})


def load_ipython_extension(ipython) -> None:
    """Register ``%%eqiora`` on the supplied IPython shell."""
    ipython.register_magics(_EqioraMagics)


def unload_ipython_extension(ipython) -> None:
    """Remove this extension's magic without deleting compiled Python variables."""
    magic = ipython.find_cell_magic("eqiora")
    if isinstance(getattr(magic, "__self__", None), _EqioraMagics):
        del ipython.magics_manager.magics["cell"]["eqiora"]
        ipython.magics_manager.registry.pop("_EqioraMagics", None)


__all__ = ["load_ipython_extension", "unload_ipython_extension"]
