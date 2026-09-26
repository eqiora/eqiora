"""Opt-in IPython execution, completion and Jupyter hover for Eqiora source cells.

Load with ``%load_ext eqiora.jupyter``. The body of ``%%eqiora model`` is
ordinary Eqiora source; successful compilation binds ``model`` in the Python
namespace. This module does not install frontend syntax highlighting.
"""

from __future__ import annotations

import keyword
import unicodedata

from IPython.core.completer import SimpleCompletion, context_matcher
from IPython.core.error import UsageError
from IPython.core.magic import Magics, cell_magic, magics_class, no_var_expand
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

import eqiora
from eqiora._eqiora import _complete_source_cell, _hover_source_cell


_HOVER_TARGET = "eqiora.source_hover"
_MAX_HOVER_SOURCE = 256 * 1024


def _hover_request(comm, message):
    """Answer one transient request using the current, unexecuted source body."""
    try:
        data = message.get("content", {}).get("data", {})
        source = data.get("source") if isinstance(data, dict) else None
        cursor = data.get("cursor") if isinstance(data, dict) else None
        result = None
        if (
            isinstance(source, str)
            and len(source) <= _MAX_HOVER_SOURCE
            and len(source.encode("utf-8")) <= _MAX_HOVER_SOURCE
            and type(cursor) is int
            and 0 <= cursor <= len(source)
        ):
            header, separator, body = source.partition("\n")
            prefix = len(header) + 1
            header = header.removesuffix("\r")
            if separator and cursor >= prefix and (
                header == "%%eqiora" or header.startswith(("%%eqiora ", "%%eqiora\t"))
            ):
                hover = _hover_source_cell(body, cursor - prefix)
                if hover is not None:
                    start, end, text = hover
                    result = [start + prefix, end + prefix, text]
        comm.send(data={"result": result})
    except UnicodeError:
        comm.send(data={"result": None})
    finally:
        comm.close()


@context_matcher(priority=100, identifier="eqiora.source_cells")
def _complete_eqiora(context):
    """Complete the current magic body without executing or retaining a cell."""
    header, separator, body = context.full_text.partition("\n")
    header = header.removesuffix("\r")
    if (
        not separator
        or context.cursor_line < 1
        or not (header == "%%eqiora" or header.startswith(("%%eqiora ", "%%eqiora\t")))
    ):
        return {"completions": []}
    lines = body.split("\n")
    line = context.cursor_line - 1
    if not 0 <= line < len(lines) or not 0 <= context.cursor_position <= len(lines[line]):
        return {"completions": []}
    cursor = sum(len(part) + 1 for part in lines[:line]) + context.cursor_position
    result = _complete_source_cell(body, cursor, context.limit if context.limit is not None else 500)
    if result is None:
        return {"completions": []}
    fragment, candidates = result
    return {
        "completions": [SimpleCompletion(candidate) for candidate in candidates],
        "matched_fragment": fragment,
        "ordered": True,
        "suppress": True,
    }


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
    """Register the magic, completion and optional kernel hover target."""
    ipython.register_magics(_EqioraMagics)
    if _complete_eqiora not in ipython.Completer.custom_matchers:
        ipython.Completer.custom_matchers.append(_complete_eqiora)
    # A terminal InteractiveShell has no kernel or Comm transport.
    kernel = getattr(ipython, "kernel", None)
    if kernel is not None:
        kernel.comm_manager.register_target(_HOVER_TARGET, _hover_request)


def unload_ipython_extension(ipython) -> None:
    """Remove owned adapters without deleting compiled Python variables."""
    if _complete_eqiora in ipython.Completer.custom_matchers:
        ipython.Completer.custom_matchers.remove(_complete_eqiora)
    kernel = getattr(ipython, "kernel", None)
    if kernel is not None and kernel.comm_manager.targets.get(_HOVER_TARGET) is _hover_request:
        kernel.comm_manager.unregister_target(_HOVER_TARGET, _hover_request)
    magic = ipython.find_cell_magic("eqiora")
    if isinstance(getattr(magic, "__self__", None), _EqioraMagics):
        del ipython.magics_manager.magics["cell"]["eqiora"]
        ipython.magics_manager.registry.pop("_EqioraMagics", None)


__all__ = ["load_ipython_extension", "unload_ipython_extension"]
