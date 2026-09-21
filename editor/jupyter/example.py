"""Generate the mixed notebook from the maintained Eqiora example, without outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def notebook() -> dict:
    source = (ROOT / "examples/decay.eqi").read_text(encoding="utf-8")
    cells = []
    for name, text in (
        ("setup", "%load_ext eqiora.jupyter\nimport eqiora"),
        ("source", "%%eqiora model\n" + source),
        ("python", "# The source cell publishes an ordinary Model to this Python kernel.\nprint(model.digest)\nassert isinstance(model, eqiora.Model)"),
    ):
        cells.append({
            "id": name, "cell_type": "code", "metadata": {},
            "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True),
        })
    return {
        "nbformat": 4, "nbformat_minor": 5, "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path("decay.ipynb"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(notebook(), indent=2) + "\n", encoding="utf-8")
